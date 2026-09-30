//! Locate a checkpoint directory: a local path, or a Hugging Face hub cache snapshot.

use crate::{Error, Result};
use std::path::{Path, PathBuf};

/// The hub cache root, following huggingface_hub: `$HF_HUB_CACHE`, else `$HF_HOME/hub`, else
/// `$XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub`.
pub fn hf_cache_dir() -> PathBuf {
    hf_cache_dir_from(|k| std::env::var(k).ok())
}

/// [`hf_cache_dir`] over an environment lookup. Empty variables count as unset.
fn hf_cache_dir_from(env: impl Fn(&str) -> Option<String>) -> PathBuf {
    let var = |k: &str| env(k).filter(|v| !v.is_empty());
    if let Some(p) = var("HF_HUB_CACHE") {
        return PathBuf::from(p);
    }
    if let Some(p) = var("HF_HOME") {
        return PathBuf::from(p).join("hub");
    }
    let cache = match var("XDG_CACHE_HOME") {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from(var("HOME").unwrap_or_else(|| ".".into())).join(".cache"),
    };
    cache.join("huggingface").join("hub")
}

fn is_checkpoint(dir: &Path) -> bool {
    dir.join("rl_agent_config.json").exists() && dir.join("model.safetensors").exists()
}

fn with_subfolder(dir: &Path, subfolder: Option<&str>) -> PathBuf {
    match subfolder {
        Some(s) => dir.join(s),
        None => dir.to_path_buf(),
    }
}

/// Resolve `convaiinnovations/laya` (+ optional subfolder) or a local directory.
///
/// Hub ids are only looked up in the local HF cache; download with
/// `hf download convaiinnovations/laya` (or the Python package) first.
pub fn resolve_model_dir(id_or_path: &str, subfolder: Option<&str>) -> Result<PathBuf> {
    let local = PathBuf::from(id_or_path);
    if local.exists() {
        let dir = with_subfolder(&local, subfolder);
        if !is_checkpoint(&dir) {
            return Err(Error::Config(format!(
                "{} is not a Laya checkpoint (needs rl_agent_config.json and model.safetensors)",
                dir.display()
            )));
        }
        return Ok(dir);
    }
    if id_or_path.starts_with('/') || id_or_path.starts_with("./") || id_or_path.starts_with("../")
    {
        return Err(Error::Config(format!(
            "Local model path not found: {id_or_path:?}"
        )));
    }
    resolve_in_cache(&hf_cache_dir(), id_or_path, subfolder)
}

/// Pick the snapshot of a hub repo the way huggingface_hub does: `refs/main` names the revision
/// of the default branch. Without that ref (a download by commit hash writes none), accept a
/// single cached snapshot and refuse to guess between several.
fn resolve_in_cache(cache: &Path, repo_id: &str, subfolder: Option<&str>) -> Result<PathBuf> {
    let repo_dir = cache.join(format!("models--{}", repo_id.replace('/', "--")));
    let snapshots = repo_dir.join("snapshots");
    let not_found = |detail: String| {
        Error::Config(format!(
            "checkpoint {repo_id:?}{} not found locally ({detail}); download it first, e.g. `hf download {repo_id}`",
            subfolder.map(|s| format!(" subfolder {s:?}")).unwrap_or_default(),
        ))
    };

    let main_ref = repo_dir.join("refs").join("main");
    if let Ok(rev) = std::fs::read_to_string(&main_ref) {
        let rev = rev.trim();
        let dir = with_subfolder(&snapshots.join(rev), subfolder);
        if is_checkpoint(&dir) {
            return Ok(dir);
        }
        return Err(not_found(format!(
            "refs/main names revision {rev} but {} has no checkpoint",
            dir.display()
        )));
    }

    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&snapshots)
        .map(|rd| {
            rd.flatten()
                .map(|e| with_subfolder(&e.path(), subfolder))
                .filter(|dir| is_checkpoint(dir))
                .collect()
        })
        .unwrap_or_default();
    candidates.sort();
    match candidates.as_slice() {
        [only] => Ok(only.clone()),
        [] => Err(not_found(format!("looked in {}", snapshots.display()))),
        many => Err(Error::Config(format!(
            "checkpoint {repo_id:?} has {} cached snapshots and no refs/main to pick one; pass the snapshot directory instead: {}",
            many.len(),
            many.iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A fresh fake HF cache under the system temp dir, removed on drop.
    struct FakeCache(PathBuf);

    impl FakeCache {
        fn new() -> Self {
            static N: AtomicUsize = AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!(
                "laya-core-resolve-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
        fn repo(&self) -> PathBuf {
            self.0.join("models--convaiinnovations--laya")
        }
        fn snapshot(&self, rev: &str, subfolder: Option<&str>) -> PathBuf {
            let dir = with_subfolder(&self.repo().join("snapshots").join(rev), subfolder);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("rl_agent_config.json"), "{}").unwrap();
            std::fs::write(dir.join("model.safetensors"), "").unwrap();
            dir
        }
        fn set_main(&self, rev: &str) {
            let refs = self.repo().join("refs");
            std::fs::create_dir_all(&refs).unwrap();
            // huggingface_hub writes the bare hash; tolerate a trailing newline too.
            std::fs::write(refs.join("main"), format!("{rev}\n")).unwrap();
        }
        fn resolve(&self, subfolder: Option<&str>) -> Result<PathBuf> {
            resolve_in_cache(&self.0, "convaiinnovations/laya", subfolder)
        }
    }

    impl Drop for FakeCache {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn env_of<'a>(pairs: &'a [(&str, &str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| {
            pairs
                .iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn cache_dir_follows_huggingface_hub_precedence() {
        let p = |s: &str| PathBuf::from(s);
        assert_eq!(
            hf_cache_dir_from(env_of(&[
                ("HF_HUB_CACHE", "/c"),
                ("HF_HOME", "/h"),
                ("HOME", "/u")
            ])),
            p("/c")
        );
        assert_eq!(
            hf_cache_dir_from(env_of(&[
                ("HF_HOME", "/h"),
                ("XDG_CACHE_HOME", "/x"),
                ("HOME", "/u")
            ])),
            p("/h/hub")
        );
        assert_eq!(
            hf_cache_dir_from(env_of(&[("XDG_CACHE_HOME", "/x"), ("HOME", "/u")])),
            p("/x/huggingface/hub")
        );
        assert_eq!(
            hf_cache_dir_from(env_of(&[("HOME", "/u")])),
            p("/u/.cache/huggingface/hub")
        );
        // Empty values count as unset, as they would make a relative path.
        assert_eq!(
            hf_cache_dir_from(env_of(&[
                ("HF_HOME", ""),
                ("XDG_CACHE_HOME", ""),
                ("HOME", "/u")
            ])),
            p("/u/.cache/huggingface/hub")
        );
    }

    #[test]
    fn one_snapshot_named_by_refs_main() {
        let c = FakeCache::new();
        let snap = c.snapshot("aaa", None);
        c.set_main("aaa");
        assert_eq!(c.resolve(None).unwrap(), snap);
    }

    #[test]
    fn subfolder_inside_the_main_snapshot() {
        let c = FakeCache::new();
        c.snapshot("aaa", None);
        let multi = c.snapshot("aaa", Some("multilingual"));
        c.set_main("aaa");
        assert_eq!(c.resolve(Some("multilingual")).unwrap(), multi);
    }

    #[test]
    fn refs_main_wins_over_a_newer_snapshot() {
        let c = FakeCache::new();
        let main = c.snapshot("aaa", None);
        // Written second, so it is the newest by mtime.
        c.snapshot("bbb", None);
        c.set_main("aaa");
        assert_eq!(c.resolve(None).unwrap(), main);
    }

    #[test]
    fn refs_main_pointing_at_a_partial_snapshot_is_an_error() {
        let c = FakeCache::new();
        c.snapshot("bbb", None);
        std::fs::create_dir_all(c.repo().join("snapshots").join("aaa")).unwrap();
        c.set_main("aaa");
        let e = c.resolve(None).unwrap_err().to_string();
        assert!(e.contains("refs/main names revision aaa"), "{e}");
    }

    #[test]
    fn single_snapshot_without_refs_is_accepted() {
        // A download by commit hash writes no refs/main.
        let c = FakeCache::new();
        let snap = c.snapshot("aaa", None);
        assert_eq!(c.resolve(None).unwrap(), snap);
    }

    #[test]
    fn several_snapshots_without_refs_are_refused() {
        let c = FakeCache::new();
        c.snapshot("aaa", None);
        c.snapshot("bbb", None);
        let e = c.resolve(None).unwrap_err().to_string();
        assert!(e.contains("2 cached snapshots"), "{e}");
        assert!(e.contains("aaa") && e.contains("bbb"), "{e}");
    }

    #[test]
    fn missing_repo_and_missing_subfolder_are_not_found() {
        let c = FakeCache::new();
        let e = c.resolve(None).unwrap_err().to_string();
        assert!(e.contains("not found locally"), "{e}");
        c.snapshot("aaa", None);
        let e = c.resolve(Some("multilingual")).unwrap_err().to_string();
        assert!(e.contains("subfolder \"multilingual\""), "{e}");
    }

    #[test]
    fn local_path_must_be_a_checkpoint() {
        let c = FakeCache::new();
        let snap = c.snapshot("aaa", None);
        assert_eq!(
            resolve_model_dir(snap.to_str().unwrap(), None).unwrap(),
            snap
        );
        let e = resolve_model_dir(c.0.to_str().unwrap(), None)
            .unwrap_err()
            .to_string();
        assert!(e.contains("is not a Laya checkpoint"), "{e}");
        let e = resolve_model_dir("/nonexistent/laya", None)
            .unwrap_err()
            .to_string();
        assert!(e.contains("Local model path not found"), "{e}");
    }
}
