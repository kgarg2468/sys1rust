//! Steady-state latency per request shape, for comparing backend settings quickly.
//!
//! sys1-probe --workload FILE [--model M] [--tuning SPEC] [--f32] [--warmup N] [--iters N]
//!            [--order grouped|mixed] [--ab SPEC_A SPEC_B [--check]]
//!
//! Takes the first row of each shape (`shape.state_tokens`, `shape.n_questions`). `grouped` runs
//! each shape `iters` times in a row after `warmup` runs; `mixed` cycles through the shapes
//! `iters` times, so every request follows a different shape. Prints min, p50 and max ms.
//! Backend settings are a laya-mlx `Knobs` spec: `--tuning` (default: none, the upstream
//! reproduction), or `SYS1_MLX` when `--tuning` is absent. Every spec is checked at startup, so
//! a bad one is an error and not a run reported under settings that were never applied. The
//! probe takes specs, not sys1-bench variant names; `sys1-bench --list-variants` shows each
//! variant's spec.
//!
//! `--ab` loads two agents from the same checkpoint, one per settings spec, warms both and runs
//! them alternately (A, B, A, B, ...) on every shape so clock and thermal drift hit both the
//! same. Prints p50 of each and B/A per shape, then the geo-mean of B/A. `--check` also
//! compares A's and B's answers on every picked row: the largest absolute difference over the
//! reported probabilities, over `score` and over `action.act_probability`, and whether every
//! choice, score and noul answer is the same.
//! Separate process runs on this machine differ by about 5%, which hides 3% effects; the
//! in-process alternation is what makes the comparison usable. The two specs must agree on
//! the process-wide MLX limits (`cache`, `wired`): both agents share one allocator, so the
//! probe refuses a pair that differs there instead of timing both under the second one's.

use anyhow::{Context, Result};
use laya_core::{Agent, BackendOptions};
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let mut workload = String::new();
    let mut model = "typed-decisions".to_string();
    let mut tuning: Option<String> = None;
    let mut f32 = false;
    let mut warmup = 3usize;
    let mut iters = 10usize;
    let mut mixed = false;
    let mut ab: Option<(String, String)> = None;
    let mut check = false;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--workload" => workload = val()?,
            "--model" => model = val()?,
            "--tuning" => tuning = Some(val()?),
            "--f32" => f32 = true,
            "--variant" => anyhow::bail!(
                "--variant {}: the probe takes a settings spec (--tuning SPEC, --f32), not a \
                 sys1-bench variant name; `sys1-bench --list-variants` shows each variant's spec",
                val()?
            ),
            "--warmup" => warmup = val()?.parse()?,
            "--iters" => iters = val()?.parse()?,
            "--order" => mixed = parse_order(&val()?)?,
            "--ab" => ab = Some((val()?, val()?)),
            "--check" => check = true,
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    if check && ab.is_none() {
        anyhow::bail!("--check needs --ab");
    }
    if iters == 0 {
        anyhow::bail!("--iters must be at least 1 (there is no p50 of no runs)");
    }
    // The settings the run is reported under; a bad one is an error, not a silent default.
    match (&ab, &tuning) {
        (Some((a, b)), _) => {
            laya_mlx::check_settings(a).with_context(|| format!("--ab A `{a}`"))?;
            laya_mlx::check_settings(b).with_context(|| format!("--ab B `{b}`"))?;
            if let Err(e) = same_process_wide_limits(a, b) {
                anyhow::bail!("--ab: {e}");
            }
        }
        (None, Some(spec)) => laya_mlx::check_settings(spec).with_context(|| format!("--tuning `{spec}`"))?,
        (None, None) => {
            if let Ok(spec) = std::env::var("SYS1_MLX") {
                laya_mlx::check_settings(&spec).context("SYS1_MLX")?;
            }
        }
    }
    let bench = std::env::var("BENCH_ROOT").context("source bench/env.sh")?;
    let lock: Value = serde_json::from_str(&std::fs::read_to_string(format!("{bench}/models.lock.json"))?)?;
    let pin = &lock[&model];
    let dir = PathBuf::from(std::env::var("HF_HOME")?)
        .join("hub")
        .join(format!("models--{}", pin["repo"].as_str().context("repo")?.replace('/', "--")))
        .join("snapshots")
        .join(pin["sha"].as_str().context("sha")?);

    let mut picked: Vec<(String, Value)> = Vec::new();
    for line in BufReader::new(std::fs::File::open(&workload)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)?;
        let shape = format!("s{}_q{}", row["shape"]["state_tokens"], row["shape"]["n_questions"]);
        if !picked.iter().any(|(s, _)| *s == shape) {
            picked.push((shape, row));
        }
    }
    if picked.is_empty() {
        anyhow::bail!("{workload}: no requests, nothing to time");
    }

    if let Some((spec_a, spec_b)) = ab {
        return ab_run(&dir, f32, &picked, &spec_a, &spec_b, warmup, iters, mixed, check);
    }

    let opts = BackendOptions { f32, tuning, ..Default::default() };
    let agent = Agent::load(&dir, &opts, Box::new(laya_mlx::make_backend))?;
    let run = |row: &Value| -> Result<(f64, u64)> {
        let (ms, tok, _) = predict(&agent, row)?;
        Ok((ms, tok))
    };

    let mut times: Vec<Vec<f64>> = vec![Vec::new(); picked.len()];
    let mut tokens = vec![0u64; picked.len()];
    if mixed {
        for (_, row) in &picked {
            for _ in 0..warmup {
                run(row)?;
            }
        }
        for _ in 0..iters {
            for (i, (_, row)) in picked.iter().enumerate() {
                let (ms, tok) = run(row)?;
                times[i].push(ms);
                tokens[i] = tok;
            }
        }
    } else {
        for (i, (_, row)) in picked.iter().enumerate() {
            for _ in 0..warmup {
                run(row)?;
            }
            for _ in 0..iters {
                let (ms, tok) = run(row)?;
                times[i].push(ms);
                tokens[i] = tok;
            }
        }
    }
    println!("shape\ttokens\tmin\tp50\tmax");
    let mut p50s = Vec::with_capacity(picked.len());
    for (i, (shape, _)) in picked.iter().enumerate() {
        let mut t = times[i].clone();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = p50(&times[i]);
        p50s.push(p50);
        println!("{shape}\t{}\t{:.1}\t{:.1}\t{:.1}", tokens[i], t[0], p50, t[t.len() - 1]);
    }
    println!("geo_p50\t{:.1}", geo_mean(&p50s));
    print_mlx_mb();
    Ok(())
}

/// One request through `agent`: latency in ms, input tokens and the answers object.
fn predict(agent: &Agent, row: &Value) -> Result<(f64, u64, Value)> {
    let t0 = Instant::now();
    let (mut out, _) =
        agent.predict_batch_timed(
            std::slice::from_ref(&row["body"]["state"]),
            &row["body"]["questions"],
            None,
        )?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mut out = out.remove(0);
    let tokens = out["usage"]["input_tokens"].as_u64().unwrap_or(0);
    Ok((ms, tokens, out["answers"].take()))
}

/// `--ab`: A and B on every shape, strictly alternating, plus the answer check.
#[allow(clippy::too_many_arguments)]
fn ab_run(
    dir: &std::path::Path,
    f32: bool,
    picked: &[(String, Value)],
    spec_a: &str,
    spec_b: &str,
    warmup: usize,
    iters: usize,
    mixed: bool,
    check: bool,
) -> Result<()> {
    let load = |spec: &str| -> Result<Agent> {
        let opts = BackendOptions { f32, tuning: Some(spec.to_string()), ..Default::default() };
        Ok(Agent::load(dir, &opts, Box::new(laya_mlx::make_backend))?)
    };
    let agents = [load(spec_a)?, load(spec_b)?];
    println!("A\t{spec_a}\nB\t{spec_b}");

    let mut times: [Vec<Vec<f64>>; 2] = [vec![Vec::new(); picked.len()], vec![Vec::new(); picked.len()]];
    let mut tokens = vec![0u64; picked.len()];
    let mut answers: Vec<[Value; 2]> = vec![[Value::Null, Value::Null]; picked.len()];
    let mut pair = |i: usize, row: &Value, record: bool| -> Result<()> {
        for (side, agent) in agents.iter().enumerate() {
            let (ms, tok, ans) = predict(agent, row)?;
            if record {
                times[side][i].push(ms);
                tokens[i] = tok;
                if answers[i][side].is_null() {
                    answers[i][side] = ans;
                }
            }
        }
        Ok(())
    };
    for (i, (_, row)) in picked.iter().enumerate() {
        for _ in 0..warmup {
            pair(i, row, false)?;
        }
        if !mixed {
            for _ in 0..iters {
                pair(i, row, true)?;
            }
        }
    }
    if mixed {
        for _ in 0..iters {
            for (i, (_, row)) in picked.iter().enumerate() {
                pair(i, row, true)?;
            }
        }
    }

    println!("shape\ttokens\tp50_A\tp50_B\tB/A");
    let mut ratios = Vec::with_capacity(picked.len());
    for (i, (shape, _)) in picked.iter().enumerate() {
        let (a, b) = (p50(&times[0][i]), p50(&times[1][i]));
        ratios.push(b / a);
        println!("{shape}\t{}\t{a:.1}\t{b:.1}\t{:.3}", tokens[i], b / a);
    }
    println!("geo_B/A\t{:.3}", geo_mean(&ratios));
    if check {
        let mut total = AnswerCheck::default();
        for [a, b] in &answers {
            total.merge(&compare_answers(a, b));
        }
        println!(
            "check\trows {}\tanswers {}\tmax_prob_diff {:.6}\tmax_score_diff {:.6}\tmax_act_diff {:.6}\t\
             mismatch {}\t(choice {} score {} noul {})\tmissing_values {}",
            picked.len(),
            total.answers,
            total.max_prob_diff,
            total.max_score_diff,
            total.max_act_diff,
            total.mismatches(),
            total.choice_mismatch,
            total.score_mismatch,
            total.noul_mismatch,
            total.missing
        );
    }
    print_mlx_mb();
    Ok(())
}

/// The value of a knob in a spec (the last one wins, as laya-mlx `Knobs` parses it), `None`
/// when the spec does not set it.
fn knob_value(spec: &str, knob: &str) -> Option<String> {
    spec.split(',')
        .map(|kv| kv.split_once('=').unwrap_or((kv, "1")))
        .filter(|(k, _)| *k == knob)
        .next_back()
        .map(|(_, v)| v.to_string())
}

/// `Err` naming the knob when two `--ab` specs would run under different process-wide MLX
/// limits. `cache` and `wired` are applied to the allocator at load, so with two agents in one
/// process both would run under the second load's values.
fn same_process_wide_limits(a: &str, b: &str) -> std::result::Result<(), String> {
    for knob in ["cache", "wired"] {
        if knob_value(a, knob) != knob_value(b, knob) {
            return Err(format!(
                "A and B set different `{knob}` limits; these are process-wide, so B's would \
                 apply to both. Give both specs the same cache and wired values."
            ));
        }
    }
    Ok(())
}

/// `--order`: `mixed` cycles through the shapes, `grouped` runs each shape's iterations in a row.
fn parse_order(order: &str) -> Result<bool> {
    match order {
        "mixed" => Ok(true),
        "grouped" => Ok(false),
        other => anyhow::bail!("unknown order {other}: use mixed or grouped"),
    }
}

fn print_mlx_mb() {
    let mb = |r: mlx_rs::error::Result<usize>| r.map(|b| b >> 20).unwrap_or(0);
    println!(
        "mlx_mb\tactive {}\tcache {}\tpeak {}",
        mb(mlx_rs::memory::active_memory()),
        mb(mlx_rs::memory::cache_memory()),
        mb(mlx_rs::memory::peak_memory())
    );
}

/// Median as the upper middle element (`sorted[len / 2]`), the convention of the bench harness.
fn p50(times: &[f64]) -> f64 {
    let mut t = times.to_vec();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[t.len() / 2]
}

fn geo_mean(xs: &[f64]) -> f64 {
    (xs.iter().map(|x| x.ln()).sum::<f64>() / xs.len() as f64).exp()
}

/// How two answer objects for the same request differ.
#[derive(Debug, Default, Clone, PartialEq)]
struct AnswerCheck {
    /// Questions compared.
    answers: usize,
    /// Largest absolute difference over `probabilities.*` and `noul`.
    max_prob_diff: f64,
    /// Largest absolute difference of the expected `score` (score questions only).
    max_score_diff: f64,
    /// Largest absolute difference of `action.act_probability`, which comes from the pooled
    /// output rather than the marker logits, so it can move when nothing else does.
    max_act_diff: f64,
    /// Answers whose reported `choice`, `score` or `noul` (4 decimals) is not the same.
    choice_mismatch: usize,
    score_mismatch: usize,
    noul_mismatch: usize,
    /// Values (`probabilities.*`, `noul`, `score`, `action.act_probability`) that one side
    /// reports as a number and the other side not at all or not as a number. These never count
    /// as a difference of zero.
    missing: usize,
}

impl AnswerCheck {
    fn mismatches(&self) -> usize {
        self.choice_mismatch + self.score_mismatch + self.noul_mismatch
    }
    fn merge(&mut self, other: &AnswerCheck) {
        self.answers += other.answers;
        self.max_prob_diff = self.max_prob_diff.max(other.max_prob_diff);
        self.max_score_diff = self.max_score_diff.max(other.max_score_diff);
        self.max_act_diff = self.max_act_diff.max(other.max_act_diff);
        self.choice_mismatch += other.choice_mismatch;
        self.score_mismatch += other.score_mismatch;
        self.noul_mismatch += other.noul_mismatch;
        self.missing += other.missing;
    }
}

/// Compare the `answers` objects of two results for the same request, question by question.
/// A question missing or of another type on the B side counts as a mismatch of A's type; a
/// probability key on one side only, or a non-numeric value, counts in `missing`.
fn compare_answers(a: &Value, b: &Value) -> AnswerCheck {
    let mut c = AnswerCheck::default();
    let Some(qa) = a.as_object() else {
        return c;
    };
    for (qid, ans_a) in qa {
        c.answers += 1;
        let ans_b = &b[qid];
        let same = |key: &str| ans_a[key] == ans_b[key] && !ans_a[key].is_null();
        match ans_a["type"].as_str() {
            Some("choice") => c.choice_mismatch += usize::from(!same("choice")),
            Some("score") => c.score_mismatch += usize::from(!same("score")),
            Some("noul") => c.noul_mismatch += usize::from(!same("noul")),
            _ => {}
        }
        // `None` when neither side has the value (a key the type does not report); `Some(None)`
        // when only one side has a number.
        let diff = |x: &Value, y: &Value| -> Option<Option<f64>> {
            match (x.as_f64(), y.as_f64()) {
                (Some(x), Some(y)) => Some(Some((x - y).abs())),
                (None, None) if x.is_null() && y.is_null() => None,
                _ => Some(None),
            }
        };
        let mut missing = 0usize;
        let mut note = |d: Option<Option<f64>>, worst: &mut f64| match d {
            Some(Some(d)) => *worst = worst.max(d),
            Some(None) => missing += 1,
            None => {}
        };
        let (pa, pb) = (ans_a["probabilities"].as_object(), ans_b["probabilities"].as_object());
        if let Some(pa) = pa {
            for (k, pv) in pa {
                note(diff(pv, &ans_b["probabilities"][k]), &mut c.max_prob_diff);
            }
        }
        note(diff(&ans_a["noul"], &ans_b["noul"]), &mut c.max_prob_diff);
        note(diff(&ans_a["score"], &ans_b["score"]), &mut c.max_score_diff);
        note(diff(&ans_a["action"]["act_probability"], &ans_b["action"]["act_probability"]), &mut c.max_act_diff);
        // Keys B reports that A does not.
        let extra = pb.map_or(0, |pb| pb.keys().filter(|k| !pa.is_some_and(|pa| pa.contains_key(*k))).count());
        c.missing += missing + extra;
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn p50_is_the_upper_middle_of_the_sorted_times() {
        assert_eq!(p50(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(p50(&[4.0, 1.0, 3.0, 2.0]), 3.0);
        assert_eq!(p50(&[5.0]), 5.0);
    }

    #[test]
    fn geo_mean_of_ratios() {
        assert!((geo_mean(&[2.0, 0.5]) - 1.0).abs() < 1e-12);
        assert!((geo_mean(&[4.0, 1.0]) - 2.0).abs() < 1e-12);
        assert!((geo_mean(&[0.9, 0.9, 0.9]) - 0.9).abs() < 1e-12);
    }

    #[test]
    fn compare_answers_reports_prob_diff_and_mismatches() {
        let a = json!({
            "pick": {"type": "choice", "choice": "x", "probabilities": {"x": 0.7, "y": 0.3}, "action": {"act_probability": 0.8}},
            "rate": {"type": "score", "score": 1.25, "probabilities": {"0": 0.25, "1": 0.25, "2": 0.5}, "action": {"act_probability": 0.5}},
            "yes": {"type": "noul", "noul": 0.61, "action": {"act_probability": 0.3}},
        });
        let same = compare_answers(&a, &a);
        assert_eq!(
            same,
            AnswerCheck { answers: 3, max_prob_diff: 0.0, ..Default::default() }
        );

        let b = json!({
            "pick": {"type": "choice", "choice": "y", "probabilities": {"x": 0.45, "y": 0.55}, "action": {"act_probability": 0.8}},
            "rate": {"type": "score", "score": 1.2501, "probabilities": {"0": 0.25, "1": 0.25, "2": 0.5}, "action": {"act_probability": 0.52}},
            "yes": {"type": "noul", "noul": 0.6, "action": {"act_probability": 0.3}},
        });
        let c = compare_answers(&a, &b);
        assert_eq!(c.answers, 3);
        assert!((c.max_prob_diff - 0.25).abs() < 1e-12);
        assert!((c.max_score_diff - 0.0001).abs() < 1e-12);
        // The action probability is compared on its own: it comes from the pooled output.
        assert!((c.max_act_diff - 0.02).abs() < 1e-12);
        assert_eq!((c.choice_mismatch, c.score_mismatch, c.noul_mismatch), (1, 1, 1));
        assert_eq!(c.mismatches(), 3);

        // Only the action probability differs: no mismatch, but a nonzero act diff.
        let mut act_only = a.clone();
        act_only["yes"]["action"]["act_probability"] = json!(0.31);
        let c = compare_answers(&a, &act_only);
        assert_eq!((c.mismatches(), c.missing, c.max_prob_diff), (0, 0, 0.0));
        assert!((c.max_act_diff - 0.01).abs() < 1e-12);

        // A question B does not have counts as a mismatch of its type, and every number A
        // reported for it as missing (5 probabilities, 1 noul, 1 score, 3 action probabilities).
        let c = compare_answers(&a, &json!({}));
        assert_eq!((c.choice_mismatch, c.score_mismatch, c.noul_mismatch), (1, 1, 1));
        assert_eq!((c.max_prob_diff, c.max_score_diff, c.max_act_diff), (0.0, 0.0, 0.0));
        assert_eq!(c.missing, 10);
    }

    /// A probability B leaves out, reports as something other than a number, or adds is not a
    /// difference of zero: it shows up in `missing` even when the choice is the same.
    #[test]
    fn compare_answers_counts_missing_and_non_numeric_probabilities() {
        let a = json!({"pick": {"type": "choice", "choice": "x", "probabilities": {"x": 0.7, "y": 0.3}}});
        let dropped = json!({"pick": {"type": "choice", "choice": "x", "probabilities": {"x": 0.7}}});
        let c = compare_answers(&a, &dropped);
        assert_eq!((c.mismatches(), c.missing, c.max_prob_diff), (0, 1, 0.0));
        let text = json!({"pick": {"type": "choice", "choice": "x", "probabilities": {"x": 0.7, "y": "0.3"}}});
        assert_eq!(compare_answers(&a, &text).missing, 1);
        let extra = json!({"pick": {"type": "choice", "choice": "x", "probabilities": {"x": 0.7, "y": 0.3, "z": 0.0}}});
        assert_eq!(compare_answers(&a, &extra).missing, 1);
        // Keys the answer type does not report (no `score` on a choice) are not missing, and
        // neither is an `action` block absent on both sides.
        assert_eq!(compare_answers(&a, &a).missing, 0);
        // An action probability on one side only is missing.
        let mut with_act = a.clone();
        with_act["pick"]["action"] = json!({"act_probability": 0.9});
        assert_eq!(compare_answers(&with_act, &a).missing, 1);
        assert_eq!(compare_answers(&a, &with_act).missing, 1);
        let mut t = AnswerCheck::default();
        t.merge(&c);
        assert_eq!(t.missing, 1);
    }

    #[test]
    fn order_is_mixed_or_grouped() {
        assert!(parse_order("mixed").unwrap());
        assert!(!parse_order("grouped").unwrap());
        for bad in ["Mixed", "mixd", "random", ""] {
            assert!(parse_order(bad).is_err(), "{bad:?} was accepted");
        }
    }

    #[test]
    fn ab_specs_must_share_the_process_wide_limits() {
        assert!(same_process_wide_limits("f16gelu,cache=512,wired=2048", "f16gelu,cache=512,wired=2048,unpad").is_ok());
        assert!(same_process_wide_limits("wired=2048,cache=512", "cache=512,wired=2048").is_ok());
        assert!(same_process_wide_limits("", "f16gelu").is_ok());
        let e = same_process_wide_limits("cache=512", "cache=1024").unwrap_err();
        assert!(e.contains("`cache`"), "{e}");
        let e = same_process_wide_limits("cache=512,wired=2048", "cache=512").unwrap_err();
        assert!(e.contains("`wired`"), "{e}");
        assert!(same_process_wide_limits("", "wired=1024").is_err());
    }

    #[test]
    fn answer_check_merges() {
        let mut t = AnswerCheck { answers: 1, max_prob_diff: 0.1, ..Default::default() };
        t.merge(&AnswerCheck {
            answers: 2,
            max_prob_diff: 0.3,
            max_score_diff: 0.2,
            max_act_diff: 0.05,
            score_mismatch: 1,
            ..Default::default()
        });
        assert_eq!(t.answers, 3);
        assert_eq!((t.max_prob_diff, t.max_score_diff, t.max_act_diff), (0.3, 0.2, 0.05));
        assert_eq!(t.mismatches(), 1);
    }
}
