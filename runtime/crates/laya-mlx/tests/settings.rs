//! Equivalence of the work-reduction settings (`dense_upto=1024`, `headprune`, `unpad`, the
//! three together, and `fuserope` alone and with the three), and of boolean masks (`mask=bool`),
//! with the plain path, on the real checkpoints (ignored by default; needs them in the HF
//! cache, `source bench/env.sh` first).
//! Every checkpoint found is run, a missing one is skipped with a note. Pass criteria per
//! question: the same chosen answer (argmax choice, rounded score, noul side) and every
//! reported probability within 1e-3.
//!
//! The cases are the `bench/workloads/smoke.jsonl` requests plus built edge cases: one question
//! with 2 options and with 1 option, 20 options, a state past `max_len` (truncated), two states
//! of very different lengths in one request (heavy padding), the three answer types in one
//! request (padded marker slots), and a single question (no padding, so `unpad` is skipped).
//!
//! Run from the repo root with:
//! `cargo test --manifest-path runtime/Cargo.toml -p laya-mlx --release --test settings -- --ignored --nocapture`

mod common;

use common::{bench_root, categorical, prob_diff, read_jsonl, CHECKPOINTS};
use laya_core::resolve::resolve_model_dir;
use laya_core::{parse_questions, Agent, BackendOptions};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Mutex;

/// The settings the work reductions are measured against (`results/SPEED.md`). Their answers
/// against the upstream fp32 reference are checked in `tests/reference.rs`.
const BASE: &str = "f16gelu,cache=512,wired=2048";
const TOL: f64 = 1e-3;
/// Against the fp32 CPU reference of `bench/reference/<name>/smoke.jsonl`: the f16 GPU
/// tolerance of `tests/parity.rs`. typed-decisions measures 0.0008, multilingual 0.0102 on
/// one answer, with or without the work reductions.
const REFERENCE_TOL: f64 = 0.02;

/// Two agents at a time is the budget; the tests take turns so `cargo test` cannot load six.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// One request: the states (each against every question) and the questions object.
struct Case {
    name: String,
    states: Vec<Value>,
    questions: Value,
}

fn words(n: usize) -> String {
    let stock = [
        "The", "customer", "reports", "the", "invoice", "total", "changed", "after", "the",
        "plan", "upgrade", "and", "asks", "for", "a", "refund", "of", "the", "difference",
        "before", "the", "next", "billing", "cycle", "starts", "on", "Monday",
    ];
    (0..n).map(|i| stock[i % stock.len()]).collect::<Vec<_>>().join(" ")
}

fn choice(instructions: &str, options: &[&str]) -> Value {
    json!({"type": "choice", "instructions": instructions, "criteria": options})
}

fn cases(smoke: &[Value]) -> Vec<Case> {
    let short_state = json!({"account": {"tier": "standard", "tenure_months": 3}, "thread": [{"role": "customer", "text": "My payment did not go through."}]});
    let long_state = Value::String(words(900));
    let over_max_state = Value::String(words(2600));
    let many: Vec<String> = (0..20).map(|i| format!("option_{i}: outcome number {i}")).collect();
    let many: Vec<&str> = many.iter().map(String::as_str).collect();
    let mixed = json!({
        "next": choice("What should the assistant do next?", &["answer_directly", "escalate", "refund", "ask_more"]),
        "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["none", "low", "medium", "high", "critical"]},
        "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        "refund_ok": {"type": "noul", "instructions": "Is a refund warranted?", "criteria": {"true": "the customer was overcharged", "false": "no charge error"}, "labels": {"false": "deny", "true": "grant"}},
    });
    let mut v = vec![
        Case {
            name: "two_options".into(),
            states: vec![short_state.clone()],
            questions: json!({"q": choice("Is this about billing or delivery?", &["billing", "delivery"])}),
        },
        Case {
            name: "one_option".into(),
            states: vec![short_state.clone()],
            questions: json!({"q": choice("Pick the only option.", &["only"])}),
        },
        Case {
            name: "many_options".into(),
            states: vec![short_state.clone()],
            questions: json!({"q": choice("Which outcome fits?", &many)}),
        },
        Case {
            name: "over_max_len".into(),
            states: vec![over_max_state],
            questions: mixed.clone(),
        },
        Case {
            name: "heavy_padding".into(),
            states: vec![short_state.clone(), long_state, Value::String("Hi.".into())],
            questions: mixed.clone(),
        },
        Case {
            name: "mixed_types".into(),
            states: vec![short_state.clone()],
            questions: mixed,
        },
        Case {
            name: "single_question".into(),
            states: vec![short_state],
            questions: json!({"q": {"type": "noul", "instructions": "Is the account locked?"}}),
        },
    ];
    for row in smoke {
        v.push(Case {
            name: row["id"].as_str().unwrap().to_string(),
            states: vec![row["body"]["state"].clone()],
            questions: row["body"]["questions"].clone(),
        });
    }
    v
}

/// Everything one agent produces for the cases: the answers per state, and the raw backend
/// output (unmasked logits and pooled) for the collated batch.
struct Run {
    answers: Vec<Vec<Value>>,
    logits: Vec<Vec<f32>>,
    pooled: Vec<Vec<f32>>,
}

fn run(agent: &Agent, cases: &[Case]) -> Run {
    let mut r = Run { answers: Vec::new(), logits: Vec::new(), pooled: Vec::new() };
    for c in cases {
        let out = agent.predict_batch(&c.states, &c.questions, None).unwrap_or_else(|e| panic!("{}: {e}", c.name));
        r.answers.push(out.into_iter().map(|mut o| o["answers"].take()).collect());
        let qs = parse_questions(&c.questions).unwrap();
        let mut items = Vec::new();
        for st in &c.states {
            items.extend(agent.encode(st, &qs).unwrap());
        }
        let batch = agent.collate(&items);
        let bo = agent.backend().forward(&batch).unwrap();
        let mut logits = Vec::new();
        for row in 0..batch.n {
            logits.extend_from_slice(&bo.logits[row * batch.kmax..row * batch.kmax + batch.marker_count[row]]);
        }
        r.logits.push(logits);
        r.pooled.push(bo.pooled);
    }
    r
}

fn load(dir: &Path, tuning: &str) -> Agent {
    let opts = BackendOptions { tuning: Some(tuning.into()), ..Default::default() };
    Agent::load(dir, &opts, Box::new(laya_mlx::make_backend)).unwrap()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// The largest magnitude in `a` and the f16 spacing (ulp) at that magnitude, to read an
/// absolute diff against: the hidden state has outlier dimensions past 1,000, where one f16
/// ulp is 1.0, and a one-ulp change there shows up whole wherever the residual and the FFN
/// output nearly cancel.
fn f16_scale(a: &[f32]) -> (f32, f32) {
    let scale = a.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let ulp = if scale > 0.0 { 2f32.powi(scale.log2().floor() as i32 - 10) } else { 0.0 };
    (scale, ulp)
}

/// The edge cases really are what their names say, for this checkpoint's `max_len`.
fn check_shapes(agent: &Agent, cases: &[Case]) {
    let max_len = agent.cfg.agent.max_len;
    let batch_of = |name: &str| {
        let c = cases.iter().find(|c| c.name == name).unwrap();
        let qs = parse_questions(&c.questions).unwrap();
        let mut items = Vec::new();
        for st in &c.states {
            items.extend(agent.encode(st, &qs).unwrap());
        }
        agent.collate(&items)
    };
    let b = batch_of("over_max_len");
    assert_eq!(b.len, max_len, "over_max_len is padded to max_len");
    assert!(b.seq_lens.iter().all(|&l| l == max_len), "every row truncated to max_len: {:?}", b.seq_lens);
    let b = batch_of("heavy_padding");
    assert_eq!(b.n, 12);
    let (lo, hi) = (b.seq_lens.iter().min().unwrap(), b.seq_lens.iter().max().unwrap());
    assert!(*hi > 4 * *lo, "rows of very different lengths: {:?}", b.seq_lens);
    assert_eq!(b.kmax, 5);
    assert!(b.marker_count.contains(&2) && b.marker_count.contains(&4), "padded marker slots");
    let b = batch_of("single_question");
    assert_eq!((b.n, b.total_tokens()), (1, b.len), "no padding");
    let b = batch_of("many_options");
    assert_eq!(b.kmax, 20);
    assert_eq!(batch_of("one_option").kmax, 1);
}

/// Load the base agent and the agent with `extra` on top, run both over the cases and compare.
/// Returns the max probability diff seen, after asserting every criterion.
fn compare(name: &str, dir: &Path, cases: &[Case], extra: &str) -> f64 {
    let base = run(&load(dir, BASE), cases);
    let tuned = run(&load(dir, &format!("{BASE},{extra}")), cases);
    let (mut worst, mut worst_logit, mut worst_pooled, mut answers, mut exact) = (0.0f64, 0.0f32, 0.0f32, 0usize, 0usize);
    let (mut pooled_scale, mut pooled_ulp) = (0.0f32, 0.0f32);
    for (i, c) in cases.iter().enumerate() {
        worst_logit = worst_logit.max(max_abs(&base.logits[i], &tuned.logits[i]));
        worst_pooled = worst_pooled.max(max_abs(&base.pooled[i], &tuned.pooled[i]));
        let (scale, ulp) = f16_scale(&base.pooled[i]);
        if scale > pooled_scale {
            (pooled_scale, pooled_ulp) = (scale, ulp);
        }
        for (a, b) in base.answers[i].iter().zip(&tuned.answers[i]) {
            if a == b {
                exact += 1;
            }
            let (qa, qb) = (a.as_object().unwrap(), b.as_object().unwrap());
            assert_eq!(qa.len(), qb.len(), "{name}/{}/{extra}", c.name);
            for (qid, ans) in qa {
                let other = &qb[qid];
                assert_eq!(categorical(ans), categorical(other), "{name}/{}/{qid}/{extra}: {ans} vs {other}", c.name);
                // Same probability keys, all finite, or `prob_diff` says which one is not.
                let d = prob_diff(ans, other).unwrap_or_else(|e| panic!("{name}/{}/{qid}/{extra}: {e}: {ans} vs {other}", c.name));
                assert!(d <= TOL, "{name}/{}/{qid}/{extra}: probability diff {d}: {ans} vs {other}", c.name);
                // `answer_confidence` comes from the logits, `act_probability` from pooled.
                for key in [&["answer_confidence"][..], &["action", "act_probability"]] {
                    let (mut x, mut y) = (ans, other);
                    for k in key {
                        (x, y) = (&x[k], &y[k]);
                    }
                    let (Some(x), Some(y)) = (x.as_f64().filter(|v| v.is_finite()), y.as_f64().filter(|v| v.is_finite())) else {
                        panic!("{name}/{}/{qid}/{extra}: {} is not a finite number on both sides: {x} vs {y}", c.name, key.join("."));
                    };
                    let dc = (x - y).abs();
                    assert!(dc <= TOL, "{name}/{}/{qid}/{extra}: {} diff {dc}", c.name, key.join("."));
                    worst = worst.max(dc);
                }
                worst = worst.max(d);
                answers += 1;
            }
        }
    }
    let states: usize = base.answers.iter().map(Vec::len).sum();
    eprintln!(
        "{name:<16} {extra:<32} {answers} answers over {} cases: max prob diff {worst:.2e}, {exact}/{states} states byte-identical, raw logits {worst_logit:.2e}, pooled {worst_pooled:.2e} (one f16 ulp at its max |x| {pooled_scale:.0} is {pooled_ulp})",
        cases.len()
    );
    worst
}

/// Run `extra` against the base settings on every checkpoint in the cache.
fn every_checkpoint(extra: &str) {
    let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let smoke = read_jsonl(&bench_root().join("workloads/smoke.jsonl"));
    let cases = cases(&smoke);
    let mut ran = 0;
    for (name, repo) in CHECKPOINTS {
        let Ok(dir) = resolve_model_dir(repo, None) else {
            eprintln!("{name:<16} skipped: {repo} is not in the HF cache");
            continue;
        };
        check_shapes(&load(&dir, BASE), &cases);
        compare(name, &dir, &cases, extra);
        ran += 1;
    }
    assert!(ran > 0, "no checkpoint in the HF cache; source bench/env.sh and download one");
}

#[test]
#[ignore]
fn dense_upto_matches_plain() {
    every_checkpoint("dense_upto=1024");
}

#[test]
#[ignore]
fn headprune_matches_plain() {
    every_checkpoint("headprune");
}

#[test]
#[ignore]
fn unpad_matches_plain() {
    every_checkpoint("unpad");
}

#[test]
#[ignore]
fn all_three_match_plain() {
    every_checkpoint("dense_upto=1024,headprune,unpad");
}

/// The `fuserope` kernel on the padded layout (no `unpad`): the kernel is checked bit for bit
/// against the MLX ops in `split_rope.rs`, this checks the answers end to end.
#[test]
#[ignore]
fn fuserope_matches_plain() {
    every_checkpoint("fuserope");
}

/// The `fuserope` kernel on the packed layout, the round 2 default of sys1d: with `unpad` the
/// kernel also does the expand through the packing's index.
#[test]
#[ignore]
fn all_four_match_plain() {
    every_checkpoint("dense_upto=1024,headprune,unpad,fuserope");
}

/// Boolean masks on every path: `over_max_len` and `heavy_padding` pad past `4 * window`, so
/// with the base settings they take the chunked local attention, whose boolean mask must give
/// the additive path's answers, padded query rows (all false would be NaN) included; the short
/// cases take the dense boolean mask.
#[test]
#[ignore]
fn bool_masks_match_additive() {
    every_checkpoint("mask=bool");
}

/// The smoke set against the fp32 CPU reference, per checkpoint, with the three work
/// reductions on top of the base settings: probabilities within `REFERENCE_TOL` and the same
/// decision on at least 99% of the answers. `tests/reference.rs` runs the same check with the
/// base settings; its numbers tell a precision regression here from the f16 gap.
#[test]
#[ignore]
fn smoke_matches_reference_with_all_three() {
    let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let bench = bench_root();
    let smoke = read_jsonl(&bench.join("workloads/smoke.jsonl"));
    let extra = "dense_upto=1024,headprune,unpad";
    let mut ran = 0;
    for (name, repo) in CHECKPOINTS {
        let Ok(dir) = resolve_model_dir(repo, None) else {
            eprintln!("{name:<16} skipped: {repo} is not in the HF cache");
            continue;
        };
        let reference = read_jsonl(&bench.join(format!("reference/{name}/smoke.jsonl")));
        let agent = load(&dir, &format!("{BASE},{extra}"));
        let (mut worst, mut worst_at, mut agree, mut total) = (0.0f64, String::new(), 0usize, 0usize);
        for row in &smoke {
            let id = row["id"].as_str().unwrap();
            let want = &reference.iter().find(|r| r["id"] == *id).unwrap_or_else(|| panic!("{id} not in reference"))["answers"];
            let got = agent.predict(&row["body"]["state"], &row["body"]["questions"]).unwrap();
            let got = got["answers"].as_object().unwrap();
            assert_eq!(got.len(), want.as_object().unwrap().len(), "{name}/{id}: answered questions differ from the reference");
            for (qid, ans) in got {
                let d = prob_diff(ans, &want[qid]).unwrap_or_else(|e| panic!("{name}/{id}/{qid}: {e}"));
                assert!(d <= REFERENCE_TOL, "{name}/{id}/{qid}: probability diff {d} vs reference");
                if d > worst {
                    (worst, worst_at) = (d, format!("{id}/{qid}"));
                }
                total += 1;
                agree += usize::from(categorical(ans) == categorical(&want[qid]));
            }
        }
        let pct = 100.0 * agree as f64 / total as f64;
        eprintln!("{name:<16} reference, all three: {total} answers, max prob diff {worst:.4} at {worst_at}, agreement {agree}/{total} ({pct:.1}%)");
        assert!(pct >= 99.0, "{name}/{extra}: agreement {pct:.1}% < 99%");
        ran += 1;
    }
    assert!(ran > 0, "no checkpoint in the HF cache");
}
