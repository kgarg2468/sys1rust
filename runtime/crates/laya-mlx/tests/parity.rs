//! Parity of the MLX backend against the Python fixtures (`runtime/tests/fixtures/*.json`,
//! written by the fork's `scripts/ref_dump.py`; not in this repository).
//!
//! Ignored by default because the fixtures and the checkpoints are external. Run with
//! `--ignored` and a missing fixture or checkpoint fails the test instead of passing it.
//! Every test here runs `BackendOptions::default()` unless it says otherwise: the settings
//! that reproduce laya-r-mlx 914c9a7, GELU promoted to f32 included. The f16 production path
//! is checked against the upstream reference in `tests/reference.rs`.

use laya_core::testing::{fixtures_path, run_parity};
use laya_core::BackendOptions;

fn check(fixture: &str, subfolder: Option<&str>, opts: &BackendOptions) {
    let Some(report) = run_parity(fixture, subfolder, laya_mlx::factory(), opts).expect("parity run") else {
        panic!(
            "{fixture}: fixture {} or checkpoint convaiinnovations/laya{} not available; \
             this test cannot pass without them",
            fixtures_path(fixture).display(),
            subfolder.map(|s| format!(" (subfolder {s})")).unwrap_or_default()
        );
    };
    println!("{}", report.summary());
    assert_eq!(report.decision_mismatches(), 0, "decision mismatches");
    assert!(
        report.logits_max_abs() < 0.15,
        "logits max abs diff {}",
        report.logits_max_abs()
    );
    assert!(
        report.prob_max_abs() < 0.02,
        "prob max abs diff {}",
        report.prob_max_abs()
    );
}

#[test]
#[ignore = "needs runtime/tests/fixtures/laya.json from the fork's scripts/ref_dump.py and the checkpoint in the HF cache"]
fn parity_laya() {
    check("laya", None, &BackendOptions::default());
}

#[test]
#[ignore = "needs runtime/tests/fixtures/multilingual.json from the fork's scripts/ref_dump.py and the checkpoint in the HF cache"]
fn parity_multilingual() {
    check("multilingual", Some("multilingual"), &BackendOptions::default());
}

/// `BackendOptions::f32` (f32 weights and compute on the GPU) must also match.
#[test]
#[ignore = "needs runtime/tests/fixtures/laya.json from the fork's scripts/ref_dump.py and the checkpoint in the HF cache"]
fn parity_laya_f32() {
    check(
        "laya",
        None,
        &BackendOptions {
            f32: true,
            ..Default::default()
        },
    );
}

/// `Device::Cpu` runs the same graph on MLX's CPU stream (slow).
#[test]
#[ignore = "needs runtime/tests/fixtures/multilingual.json from the fork's scripts/ref_dump.py and the checkpoint in the HF cache; slow"]
fn parity_multilingual_cpu() {
    check(
        "multilingual",
        Some("multilingual"),
        &BackendOptions {
            device: laya_core::Device::Cpu,
            ..Default::default()
        },
    );
}
