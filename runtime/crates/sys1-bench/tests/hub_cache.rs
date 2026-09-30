//! Both binaries look for the pinned checkpoint in the hub cache huggingface_hub downloads
//! into: `HF_HUB_CACHE`, else `HF_HOME/hub`, else `XDG_CACHE_HOME/huggingface/hub`, else
//! `~/.cache/huggingface/hub`. Each binary runs as a child with one combination of those
//! variables and a lock file that pins a sha nothing has downloaded, so its error names the
//! cache it searched. No model is loaded: the lookup fails (or, in the last test, succeeds and
//! the run stops at the missing workload) before anything else.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A bench root with a lock file, plus three empty cache roots, under the system temp dir.
struct Sandbox {
    root: PathBuf,
}

impl Sandbox {
    fn new() -> Self {
        static N: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "sys1-bench-hub-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        for d in ["bench", "hub", "home", "xdg", "user"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        std::fs::write(
            root.join("bench/models.lock.json"),
            r#"{"typed-decisions": {"repo": "convaiinnovations/laya-typed-decisions", "sha": "0000deadbeef", "license": "apache-2.0"}}"#,
        )
        .unwrap();
        Self { root }
    }

    fn dir(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    /// Run `bin` against this sandbox with exactly the hub variables in `env` set; returns
    /// stderr. Neither binary gets far enough to need a workload.
    fn run(&self, bin: &str, env: &[(&str, &Path)]) -> String {
        let exe = match bin {
            "sys1-bench" => env!("CARGO_BIN_EXE_sys1-bench"),
            "sys1-probe" => env!("CARGO_BIN_EXE_sys1-probe"),
            other => panic!("no binary {other}"),
        };
        let mut cmd = Command::new(exe);
        for k in ["HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME", "HOME", "SYS1_MLX"] {
            cmd.env_remove(k);
        }
        cmd.env("BENCH_ROOT", self.dir("bench"));
        for (k, v) in env {
            cmd.env(k, v);
        }
        let workload = self.dir("missing-workload.jsonl");
        match bin {
            "sys1-bench" => cmd.args(["--variant", "mlx-fp16", "--model", "typed-decisions", "--workload"]).arg(&workload).arg("--out").arg(self.dir("out.jsonl")),
            _ => cmd.args(["--model", "typed-decisions", "--workload"]).arg(&workload),
        };
        let out = cmd.output().unwrap_or_else(|e| panic!("run {exe}: {e}"));
        assert!(!out.status.success(), "{bin} exited 0 with no checkpoint and no workload");
        String::from_utf8_lossy(&out.stderr).into_owned()
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// The error a lookup in `cache` gives when the pinned snapshot is not there.
fn not_in(cache: &Path) -> String {
    format!("is not downloaded: {} does not exist (hub cache {}", cache.join("models--convaiinnovations--laya-typed-decisions/snapshots/0000deadbeef").display(), cache.display())
}

#[test]
fn hub_cache_precedence_is_huggingface_hubs() {
    let s = Sandbox::new();
    let (hub, home, xdg, user) = (s.dir("hub"), s.dir("home"), s.dir("xdg"), s.dir("user"));
    for bin in ["sys1-bench", "sys1-probe"] {
        // HF_HUB_CACHE wins over everything, even with HF_HOME set to another place.
        let err = s.run(bin, &[("HF_HUB_CACHE", &hub), ("HF_HOME", &home), ("XDG_CACHE_HOME", &xdg), ("HOME", &user)]);
        assert!(err.contains(&not_in(&hub)), "{bin}: {err}");
        // Then HF_HOME/hub.
        let err = s.run(bin, &[("HF_HOME", &home), ("XDG_CACHE_HOME", &xdg), ("HOME", &user)]);
        assert!(err.contains(&not_in(&home.join("hub"))), "{bin}: {err}");
        // Then XDG_CACHE_HOME/huggingface/hub.
        let err = s.run(bin, &[("XDG_CACHE_HOME", &xdg), ("HOME", &user)]);
        assert!(err.contains(&not_in(&xdg.join("huggingface/hub"))), "{bin}: {err}");
        // Then ~/.cache/huggingface/hub.
        let err = s.run(bin, &[("HOME", &user)]);
        assert!(err.contains(&not_in(&user.join(".cache/huggingface/hub"))), "{bin}: {err}");
    }
}

/// A snapshot downloaded under `HF_HUB_CACHE` is found although `HF_HOME/hub` is empty (the
/// case of a bench wrapper that preserves `HF_HUB_CACHE`): both binaries get past the lookup
/// and stop at the workload that does not exist.
#[test]
fn snapshot_under_hf_hub_cache_is_found_with_hf_home_elsewhere() {
    let s = Sandbox::new();
    let (hub, home) = (s.dir("hub"), s.dir("home"));
    std::fs::create_dir_all(hub.join("models--convaiinnovations--laya-typed-decisions/snapshots/0000deadbeef")).unwrap();
    for bin in ["sys1-bench", "sys1-probe"] {
        let err = s.run(bin, &[("HF_HUB_CACHE", &hub), ("HF_HOME", &home)]);
        assert!(!err.contains("is not downloaded"), "{bin}: {err}");
        assert!(err.contains("missing-workload.jsonl"), "{bin}: {err}");
    }
}
