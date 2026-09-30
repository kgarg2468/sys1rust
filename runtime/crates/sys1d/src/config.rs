//! Server configuration: CLI flags, each falling back to an environment variable, and the
//! served checkpoint (a known name resolved in the local Hugging Face cache, or a directory).

use anyhow::{bail, Context, Result};
use clap::Parser;
use laya_core::resolve::{hf_cache_dir, resolve_model_dir};
use laya_core::BackendOptions;
use std::path::{Path, PathBuf};

/// Upstream's default admission bound, also the fallback for an invalid `LAYA_MAX_CONCURRENT`.
pub const DEFAULT_MAX_CONCURRENT: usize = 16;
/// Engine settings measured in `results/SPIKE.md`: fp16 GELU, a 512 MiB MLX buffer cache
/// (unbounded, it grows to about RAM size and pushes the machine into swap), 2 GiB wired.
pub const DEFAULT_TUNING: &str = "f16gelu,cache=512,wired=2048";

/// Checkpoint names and their Hugging Face repos, in upstream's naming.
pub const CHECKPOINTS: [(&str, &str); 3] = [
    ("typed-decisions", "convaiinnovations/laya-typed-decisions"),
    ("multilingual", "convaiinnovations/laya-multilingual"),
    ("english", "convaiinnovations/laya"),
];

/// Published ids a client may put in `model`. `convaiinnovations/laya` is deliberately absent,
/// as upstream: it means "let the server choose", which here is the only checkpoint served.
pub const PUBLISHED_MODEL_IDS: [(&str, &str); 2] = [
    ("convaiinnovations/laya-multilingual", "multilingual"),
    ("convaiinnovations/laya-typed-decisions", "typed-decisions"),
];

#[derive(Parser, Debug, Clone)]
#[command(
    name = "sys1d",
    about = "Local HTTP server for Laya System 1 decisions (POST /v1/systemone, GET /health)"
)]
pub struct Config {
    /// Checkpoint to serve: `typed-decisions`, `multilingual`, `english`, a published repo id,
    /// or a local checkpoint directory. Hub ids are only looked up in the local HF cache.
    #[arg(long, env = "SYS1_MODEL", default_value = "typed-decisions")]
    pub model: String,
    /// Use `snapshots/<sha>` of the cached repo instead of the newest snapshot.
    #[arg(long, env = "SYS1_REVISION")]
    pub revision: Option<String>,
    /// Bind address. Upstream defaults to 0.0.0.0; this is a local runtime.
    #[arg(long, env = "LAYA_HOST", default_value = "127.0.0.1")]
    pub host: String,
    /// Bind port; 0 picks a free port (see the ready line on stdout).
    #[arg(long, env = "LAYA_PORT", default_value_t = 8000)]
    pub port: u16,
    /// If set, `/v1/systemone` requires `Authorization: Bearer <key>`.
    #[arg(long, env = "LAYA_API_KEY")]
    pub api_key: Option<String>,
    /// Requests admitted past auth at once; excess gets 503. Invalid values fall back to 16.
    #[arg(long, env = "LAYA_MAX_CONCURRENT")]
    pub max_concurrent: Option<String>,
    /// laya-mlx settings (comma list, see `Knobs` in crates/laya-mlx).
    #[arg(long, env = "SYS1_MLX_TUNING", default_value = DEFAULT_TUNING)]
    pub tuning: String,
    /// Run the transformer in f32 instead of the checkpoint's f16.
    #[arg(long, env = "SYS1_F32")]
    pub f32: bool,
}

impl Config {
    /// The api key, or `None` when unset or empty (upstream: `os.environ.get(...) or None`).
    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref().filter(|k| !k.is_empty())
    }

    pub fn revision(&self) -> Option<&str> {
        self.revision
            .as_deref()
            .map(str::trim)
            .filter(|r| !r.is_empty())
    }

    pub fn max_concurrent(&self) -> usize {
        resolve_max_concurrent(self.max_concurrent.as_deref())
    }

    pub fn backend_options(&self) -> BackendOptions {
        BackendOptions {
            f32: self.f32,
            tuning: Some(self.tuning.clone()),
            ..Default::default()
        }
    }
}

/// Upstream `_resolve_max_concurrent`: unset, unparseable or non-positive means the default.
pub fn resolve_max_concurrent(raw: Option<&str>) -> usize {
    match raw
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::parse::<i64>)
    {
        Some(Ok(n)) if n > 0 => n as usize,
        None => DEFAULT_MAX_CONCURRENT,
        Some(_) => {
            crate::log(format!(
                "invalid LAYA_MAX_CONCURRENT {:?}; falling back to {DEFAULT_MAX_CONCURRENT}",
                raw.unwrap_or_default()
            ));
            DEFAULT_MAX_CONCURRENT
        }
    }
}

/// Map a checkpoint name or published id (case-insensitive, trimmed) to the canonical name.
pub fn checkpoint_name(s: &str) -> Option<&'static str> {
    let key = s.trim().to_ascii_lowercase();
    CHECKPOINTS
        .iter()
        .find(|(name, _)| *name == key)
        .map(|(name, _)| *name)
        .or_else(|| {
            PUBLISHED_MODEL_IDS
                .iter()
                .find(|(id, _)| *id == key)
                .map(|(_, name)| *name)
        })
}

fn repo_of(name: &str) -> Option<&'static str> {
    CHECKPOINTS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, repo)| *repo)
}

/// The one checkpoint this process serves.
#[derive(Debug, Clone)]
pub struct ServedModel {
    /// Upstream's checkpoint name (`typed-decisions`), or the directory name for a local path.
    pub name: String,
    /// Hugging Face repo id, or the local path.
    pub repo: String,
    /// Snapshot sha when loaded from the hub cache.
    pub revision: Option<String>,
    pub dir: PathBuf,
}

/// Resolve `--model` / `--revision` to a checkpoint directory. Never downloads.
pub fn resolve_served(model: &str, revision: Option<&str>) -> Result<ServedModel> {
    let model = model.trim();
    if let Some(name) = checkpoint_name(model) {
        let repo = repo_of(name).expect("every known name has a repo");
        let dir = match revision {
            Some(rev) => {
                let snap = hf_cache_dir()
                    .join(format!("models--{}", repo.replace('/', "--")))
                    .join("snapshots")
                    .join(rev);
                if !snap.is_dir() {
                    bail!(
                        "revision {rev} of {repo} is not in the local HF cache ({})",
                        snap.display()
                    );
                }
                resolve_model_dir(&snap.to_string_lossy(), None)?
            }
            None => resolve_model_dir(repo, None)?,
        };
        let revision = snapshot_sha(&dir);
        return Ok(ServedModel {
            name: name.into(),
            repo: repo.into(),
            revision,
            dir,
        });
    }
    if !Path::new(model).exists() {
        bail!(
            "{model:?} is neither a checkpoint name ({}) nor an existing directory",
            CHECKPOINTS
                .iter()
                .map(|(n, _)| *n)
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if revision.is_some() {
        crate::log("--revision is ignored for a local checkpoint directory");
    }
    let dir = resolve_model_dir(model, None)?;
    let dir = dir
        .canonicalize()
        .with_context(|| format!("canonicalize {}", dir.display()))?;
    // A path into the hub cache still gets its repo and sha, so routing and /health say what
    // is really served; any other directory is named after itself.
    if let Some((repo, sha)) = hub_layout(&dir) {
        let name = CHECKPOINTS
            .iter()
            .find(|(_, r)| *r == repo)
            .map(|(n, _)| n.to_string())
            .unwrap_or_else(|| repo.rsplit('/').next().unwrap_or(&repo).to_string());
        return Ok(ServedModel {
            name,
            repo,
            revision: Some(sha),
            dir,
        });
    }
    let name = dir
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_else(|| model.to_string());
    Ok(ServedModel {
        name,
        repo: dir.to_string_lossy().into_owned(),
        revision: None,
        dir,
    })
}

/// `.../models--org--name/snapshots/<sha>` -> (`org/name`, sha).
fn hub_layout(dir: &Path) -> Option<(String, String)> {
    let sha = dir.file_name()?.to_str()?;
    let snapshots = dir.parent()?;
    if snapshots.file_name()?.to_str()? != "snapshots" {
        return None;
    }
    let repo_dir = snapshots.parent()?.file_name()?.to_str()?;
    let repo = repo_dir.strip_prefix("models--")?.replace("--", "/");
    Some((repo, sha.to_string()))
}

fn snapshot_sha(dir: &Path) -> Option<String> {
    hub_layout(dir).map(|(_, sha)| sha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_concurrent_falls_back_like_upstream() {
        assert_eq!(resolve_max_concurrent(None), 16);
        assert_eq!(resolve_max_concurrent(Some("")), 16);
        assert_eq!(resolve_max_concurrent(Some("abc")), 16);
        assert_eq!(resolve_max_concurrent(Some("0")), 16);
        assert_eq!(resolve_max_concurrent(Some("-3")), 16);
        assert_eq!(resolve_max_concurrent(Some(" 4 ")), 4);
    }

    #[test]
    fn checkpoint_names_and_published_ids() {
        assert_eq!(checkpoint_name("typed-decisions"), Some("typed-decisions"));
        assert_eq!(checkpoint_name(" English "), Some("english"));
        assert_eq!(
            checkpoint_name("CONVAIINNOVATIONS/LAYA-MULTILINGUAL"),
            Some("multilingual")
        );
        assert_eq!(checkpoint_name("convaiinnovations/laya"), None);
        assert_eq!(checkpoint_name("jev-1"), None);
    }

    #[test]
    fn hub_layout_is_detected() {
        let p =
            Path::new("/x/hub/models--convaiinnovations--laya-typed-decisions/snapshots/abc123");
        assert_eq!(
            hub_layout(p),
            Some((
                "convaiinnovations/laya-typed-decisions".into(),
                "abc123".into()
            ))
        );
        assert_eq!(hub_layout(Path::new("/x/my-checkpoint")), None);
    }

    #[test]
    fn defaults_and_env_style_flags_parse() {
        let c = Config::try_parse_from(["sys1d"]).unwrap();
        assert_eq!(c.model, "typed-decisions");
        assert_eq!((c.host.as_str(), c.port), ("127.0.0.1", 8000));
        assert_eq!(c.max_concurrent(), 16);
        assert_eq!(c.tuning, DEFAULT_TUNING);
        assert!(!c.f32 && c.api_key().is_none() && c.revision().is_none());
        let c = Config::try_parse_from([
            "sys1d",
            "--port",
            "0",
            "--max-concurrent",
            "x",
            "--api-key",
            "",
            "--f32",
        ])
        .unwrap();
        assert_eq!(c.port, 0);
        assert_eq!(c.max_concurrent(), 16);
        assert!(c.api_key().is_none());
        assert!(c.backend_options().f32);
        assert!(Config::try_parse_from(["sys1d", "--port", "70000"]).is_err());
    }
}
