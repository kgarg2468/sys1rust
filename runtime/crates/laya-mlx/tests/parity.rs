//! Parity of the MLX backend against the Python fixtures (`tests/fixtures/*.json`).
//! Skipped silently when the checkpoint or fixture is not available locally.

use laya_core::testing::run_parity;
use laya_core::BackendOptions;

fn check(fixture: &str, subfolder: Option<&str>) {
    let opts = BackendOptions::default();
    let Some(report) =
        run_parity(fixture, subfolder, laya_mlx::factory(), &opts).expect("parity run")
    else {
        eprintln!("skipping {fixture}: checkpoint or fixture not available");
        return;
    };
    println!("{}", report.summary());
    assert_eq!(report.choice_mismatches(), 0, "choice mismatches");
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
fn parity_laya() {
    check("laya", None);
}

#[test]
fn parity_multilingual() {
    check("multilingual", Some("multilingual"));
}

/// `BackendOptions::f32` (f32 weights and compute on the GPU) must also match.
#[test]
fn parity_laya_f32() {
    let opts = BackendOptions {
        f32: true,
        ..Default::default()
    };
    let Some(report) = run_parity("laya", None, laya_mlx::factory(), &opts).expect("parity run")
    else {
        return;
    };
    println!("{}", report.summary());
    assert_eq!(report.choice_mismatches(), 0);
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

/// `Device::Cpu` runs the same graph on MLX's CPU stream (slow; run with `--ignored`).
#[test]
#[ignore]
fn parity_multilingual_cpu() {
    let opts = BackendOptions {
        device: laya_core::Device::Cpu,
        ..Default::default()
    };
    let Some(report) = run_parity(
        "multilingual",
        Some("multilingual"),
        laya_mlx::factory(),
        &opts,
    )
    .expect("parity run") else {
        return;
    };
    println!("{}", report.summary());
    assert_eq!(report.choice_mismatches(), 0);
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
