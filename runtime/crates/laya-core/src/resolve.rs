//! Locate a checkpoint directory: a local path, or a Hugging Face hub cache snapshot.

use crate::{Error, Result};
use std::path::{Path, PathBuf};

/// The hub cache root: `$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`.
pub fn hf_cache_dir() -> PathBuf {
    if let Ok(p) = std::env::var("HF_HUB_CACHE") {
        return PathBuf::from(p);
    }
    if let Ok(p) = std::env::var("HF_HOME") {
        return PathBuf::from(p).join("hub");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("hub")
}

fn is_checkpoint(dir: &Path) -> bool {
    dir.join("rl_agent_config.json").exists() && dir.join("model.safetensors").exists()
}

/// Resolve `convaiinnovations/laya` (+ optional subfolder) or a local directory.
///
/// Hub ids are only looked up in the local HF cache; download with
/// `hf download convaiinnovations/laya` (or the Python package) first.
pub fn resolve_model_dir(id_or_path: &str, subfolder: Option<&str>) -> Result<PathBuf> {
    let local = PathBuf::from(id_or_path);
    if local.exists() {
        let dir = match subfolder {
            Some(s) => local.join(s),
            None => local,
        };
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
    let repo_dir = hf_cache_dir().join(format!("models--{}", id_or_path.replace('/', "--")));
    let snapshots = repo_dir.join("snapshots");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&snapshots)
        .map(|rd| rd.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    candidates.sort_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());
    for snap in candidates.iter().rev() {
        let dir = match subfolder {
            Some(s) => snap.join(s),
            None => snap.clone(),
        };
        if is_checkpoint(&dir) {
            return Ok(dir);
        }
    }
    Err(Error::Config(format!(
        "checkpoint {id_or_path:?}{} not found locally (looked in {}); download it first, e.g. `hf download {id_or_path}`",
        subfolder.map(|s| format!(" subfolder {s:?}")).unwrap_or_default(),
        snapshots.display()
    )))
}
