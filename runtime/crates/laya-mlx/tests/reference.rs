//! The smoke workload (`bench/workloads/smoke.jsonl`) through the engine with the production
//! settings, against the upstream fp32 CPU reference `bench/reference/<name>/smoke.jsonl`, for
//! every checkpoint in the HF cache. This is the check that the f16 GPU path serves upstream's
//! answers: the same decision on every answer and every reported probability within `TOL`.
//!
//! Ignored by default: it needs the downloaded checkpoints (`source bench/env.sh` first). A
//! checkpoint that is not in the cache is skipped with a note; at least one must run.
//!
//! Run with: `cargo test -p laya-mlx --release --test reference -- --ignored --nocapture`

mod common;

use common::{bench_root, categorical, prob_diff, read_jsonl, CHECKPOINTS};
use laya_core::resolve::resolve_model_dir;
use laya_core::{Agent, BackendOptions};
use serde_json::json;

/// The settings answers are served with: GELU kept in f16 (upstream promotes it to f32), a
/// 512 MiB MLX buffer cache and 2 GiB wired. The bench's `mlx-fp16-fast` variant.
const PRODUCTION: &str = "f16gelu,cache=512,wired=2048";

/// The f16 GPU path against the fp32 CPU reference. typed-decisions measures 0.0008 on the
/// smoke set, multilingual 0.0102 on one near-tie answer. The parity fixtures use the same 0.02.
const TOL: f64 = 0.02;

/// The bench harness flags a run below this agreement (`bench/harness/compare.py`).
const MIN_AGREEMENT: f64 = 0.99;

#[test]
#[ignore = "needs the checkpoints in the HF cache; source bench/env.sh first"]
fn smoke_matches_upstream_reference() {
    let bench = bench_root();
    let smoke = read_jsonl(&bench.join("workloads/smoke.jsonl"));
    let mut ran = 0;
    for (name, repo) in CHECKPOINTS {
        let Ok(dir) = resolve_model_dir(repo, None) else {
            eprintln!("{name:<16} skipped: {repo} is not in the HF cache");
            continue;
        };
        let reference = read_jsonl(&bench.join(format!("reference/{name}/smoke.jsonl")));
        let opts = BackendOptions { tuning: Some(PRODUCTION.into()), ..Default::default() };
        let agent = Agent::load(&dir, &opts, Box::new(laya_mlx::make_backend)).unwrap();
        let (mut worst, mut worst_at, mut agree, mut total) = (0.0f64, String::new(), 0usize, 0usize);
        for row in &smoke {
            let id = row["id"].as_str().unwrap();
            let want = &reference
                .iter()
                .find(|r| r["id"] == *id)
                .unwrap_or_else(|| panic!("{name}: {id} is not in the reference"))["answers"];
            let got = agent.predict(&row["body"]["state"], &row["body"]["questions"]).unwrap();
            let got = got["answers"].as_object().unwrap();
            assert_eq!(
                got.len(),
                want.as_object().unwrap().len(),
                "{name}/{id}: answered questions differ from the reference"
            );
            for (qid, ans) in got {
                let d = prob_diff(ans, &want[qid]).unwrap_or_else(|e| panic!("{name}/{id}/{qid}: {e}"));
                assert!(d <= TOL, "{name}/{id}/{qid}: probability diff {d} vs reference: {ans} vs {}", want[qid]);
                if d > worst {
                    (worst, worst_at) = (d, format!("{id}/{qid}"));
                }
                total += 1;
                agree += usize::from(categorical(ans) == categorical(&want[qid]));
            }
        }
        let agreement = agree as f64 / total as f64;
        eprintln!(
            "{name:<16} {total} answers over {} requests: agreement {agree}/{total} ({:.1}%), max probability diff {worst:.4} at {worst_at}",
            smoke.len(),
            100.0 * agreement
        );
        assert!(agreement >= MIN_AGREEMENT, "{name}: agreement {:.1}% is below {:.0}%", 100.0 * agreement, 100.0 * MIN_AGREEMENT);
        ran += 1;
    }
    assert!(ran > 0, "no checkpoint in the HF cache; source bench/env.sh and download one");
}

#[test]
fn prob_diff_is_the_largest_difference_over_the_same_keys() {
    let a = json!({"type": "choice", "probabilities": {"x": 0.7, "y": 0.3}});
    let b = json!({"type": "choice", "probabilities": {"x": 0.65, "y": 0.35}});
    assert!((prob_diff(&a, &b).unwrap() - 0.05).abs() < 1e-12);
    assert_eq!(prob_diff(&a, &a).unwrap(), 0.0);
    let n = json!({"type": "noul", "noul": 0.61});
    let m = json!({"type": "noul", "noul": 0.6});
    assert!((prob_diff(&n, &m).unwrap() - 0.01).abs() < 1e-12);
}

#[test]
fn prob_diff_rejects_missing_or_non_numeric_values() {
    let a = json!({"type": "choice", "probabilities": {"x": 0.7, "y": 0.3}});
    // A key missing on either side, or an extra one, is an error, not a zero.
    assert!(prob_diff(&a, &json!({"probabilities": {"x": 0.7}})).is_err());
    assert!(prob_diff(&a, &json!({"probabilities": {"x": 0.7, "y": 0.3, "z": 0.0}})).is_err());
    assert!(prob_diff(&a, &json!({"probabilities": {"x": 0.7, "y": null}})).is_err());
    assert!(prob_diff(&a, &json!({"probabilities": {"x": 0.7, "y": "0.3"}})).is_err());
    // Probabilities on one side only, or a noul without its number.
    assert!(prob_diff(&a, &json!({"type": "noul", "noul": 0.5})).is_err());
    assert!(prob_diff(&json!({"noul": 0.5}), &json!({})).is_err());
    assert!(prob_diff(&json!({"noul": f64::NAN}), &json!({"noul": 0.5})).is_err());
}

#[test]
fn categorical_is_the_decision_of_each_answer_type() {
    assert_eq!(categorical(&json!({"type": "choice", "choice": "refund"})), json!("refund"));
    assert_eq!(categorical(&json!({"type": "noul", "noul": 0.51})), json!(true));
    assert_eq!(categorical(&json!({"type": "noul", "noul": 0.49})), json!(false));
    assert_eq!(categorical(&json!({"type": "score", "score": 2.4})), json!(2.0));
    assert_eq!(categorical(&json!({"type": "score", "score": 2.6})), json!(3.0));
}
