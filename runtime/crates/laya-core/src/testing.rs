//! Shared parity/benchmark harness for backend crates (`tests/fixtures/*.json`).

use crate::backend::{Backend, BackendOptions};
use crate::decode::{decode_answers, ActHead, Temperatures};
use crate::question::parse_questions;
use crate::weights::Weights;
use crate::{Agent, ModelConfig, Result};
use serde_json::Value;
use std::path::PathBuf;

pub type Factory =
    Box<dyn FnOnce(&Weights, &ModelConfig, &BackendOptions) -> Result<Box<dyn Backend>>>;

#[derive(Debug, Clone, Default)]
pub struct CaseReport {
    pub name: String,
    pub rows: usize,
    pub logits_max_abs: f32,
    pub pooled_max_abs: f32,
    pub encoder_hidden_max_abs: Option<f32>,
    pub answers_equal: bool,
    pub choice_mismatches: usize,
    pub prob_max_abs: f64,
    pub forward_ms: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ParityReport {
    pub backend: String,
    pub cases: Vec<CaseReport>,
}

impl ParityReport {
    pub fn logits_max_abs(&self) -> f32 {
        self.cases
            .iter()
            .map(|c| c.logits_max_abs)
            .fold(0.0, f32::max)
    }
    pub fn prob_max_abs(&self) -> f64 {
        self.cases
            .iter()
            .map(|c| c.prob_max_abs)
            .fold(0.0, f64::max)
    }
    pub fn choice_mismatches(&self) -> usize {
        self.cases.iter().map(|c| c.choice_mismatches).sum()
    }
    pub fn all_answers_equal(&self) -> bool {
        self.cases.iter().all(|c| c.answers_equal)
    }
    pub fn summary(&self) -> String {
        let mut s = format!("backend {}\n", self.backend);
        for c in &self.cases {
            s += &format!(
                "  {:<12} rows {:<3} logits {:.4}  pooled {:.4}  enc {}  probs {:.4}  choice_mismatch {}  exact {}  fwd {:.1} ms\n",
                c.name, c.rows, c.logits_max_abs, c.pooled_max_abs,
                c.encoder_hidden_max_abs.map(|x| format!("{x:.4}")).unwrap_or_else(|| "-".into()),
                c.prob_max_abs, c.choice_mismatches, c.answers_equal, c.forward_ms
            );
        }
        s
    }
}

pub fn fixtures_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(format!("{name}.json"))
}

fn flat_f32(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .flat_map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
        })
        .collect()
}

/// Largest element-wise difference. A backend output of the wrong length is an error rather
/// than a comparison over the shorter prefix, which would report zero for a truncated output.
fn max_abs_diff(case: &str, what: &str, got: &[f32], want: &[f32]) -> Result<f32> {
    if got.len() != want.len() {
        return Err(crate::Error::Backend(format!(
            "{case}: {what} has {} values, the fixture has {}",
            got.len(),
            want.len()
        )));
    }
    Ok(got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max))
}

/// Walk two answer objects collecting the max abs difference over numeric leaves and the
/// number of differing `choice` strings.
fn compare_answers(got: &Value, want: &Value, prob_max: &mut f64, choice_mismatch: &mut usize) {
    match (got, want) {
        (Value::Object(a), Value::Object(b)) => {
            for (k, wv) in b {
                match a.get(k) {
                    Some(gv) => {
                        if k == "choice" && gv != wv {
                            *choice_mismatch += 1;
                        }
                        compare_answers(gv, wv, prob_max, choice_mismatch);
                    }
                    None => {
                        *choice_mismatch += 1;
                    }
                }
            }
        }
        (Value::Number(a), Value::Number(b)) => {
            let d = (a.as_f64().unwrap_or(0.0) - b.as_f64().unwrap_or(0.0)).abs();
            if d > *prob_max {
                *prob_max = d;
            }
        }
        _ => {}
    }
}

/// Run every fixture case through `factory`'s backend and compare with the Python outputs.
/// Returns `Ok(None)` when the fixture file or the checkpoint is not available locally.
pub fn run_parity(
    fixture: &str,
    subfolder: Option<&str>,
    factory: Factory,
    opts: &BackendOptions,
) -> Result<Option<ParityReport>> {
    let Ok(bytes) = std::fs::read(fixtures_path(fixture)) else {
        return Ok(None);
    };
    let fx: Value = serde_json::from_slice(&bytes)?;
    let Ok(dir) = crate::resolve::resolve_model_dir("convaiinnovations/laya", subfolder) else {
        return Ok(None);
    };
    let agent = Agent::load(&dir, opts, factory)?;
    let weights = Weights::open(&dir)?;
    let act_head = ActHead::load(&weights)?;
    let temps = Temperatures::from_config(&agent.cfg.agent);
    let d = agent.cfg.hidden_size();
    let mut report = ParityReport {
        backend: agent.backend_name(),
        cases: Vec::new(),
    };
    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap().to_string();
        let qs = parse_questions(&case["questions"])?;
        let items = agent.encode(&case["state"], &qs)?;
        let batch = agent.collate(&items);
        agent.backend().forward(&batch)?; // warm
        let t0 = std::time::Instant::now();
        let out = agent.backend().forward(&batch)?;
        let forward_ms = t0.elapsed().as_secs_f64() * 1e3;
        let want_logits = flat_f32(&case["masked_logits"]);
        let want_pooled = flat_f32(&case["pooled"]);
        let mut logits_max = 0f32;
        for r in 0..batch.n {
            for k in 0..batch.marker_count[r] {
                let i = r * batch.kmax + k;
                logits_max = logits_max.max((out.logits[i] - want_logits[i]).abs());
            }
        }
        let pooled_max = max_abs_diff(&name, "pooled", &out.pooled, &want_pooled)?;
        let encoder_hidden_max_abs = match case.get("encoder_hidden_item0") {
            Some(h) if !h.is_null() => {
                let want = flat_f32(h);
                let row0 = crate::sequence::collate(&items[..1], agent.tokenizer.pad_id);
                match agent.backend().encoder_hidden(&row0)? {
                    Some(got) => Some(max_abs_diff(&name, "encoder hidden state", &got, &want)?),
                    None => None,
                }
            }
            _ => None,
        };
        let answers = decode_answers(&out, &batch, &act_head, &temps, &qs, 0)?;
        let want = &case["result"]["answers"];
        let (mut prob_max, mut choice_mismatches) = (0f64, 0usize);
        compare_answers(&answers, want, &mut prob_max, &mut choice_mismatches);
        report.cases.push(CaseReport {
            name,
            rows: batch.n,
            logits_max_abs: logits_max,
            pooled_max_abs: pooled_max,
            encoder_hidden_max_abs,
            answers_equal: &answers == want,
            choice_mismatches,
            prob_max_abs: prob_max,
            forward_ms,
        });
    }
    let _ = d;
    Ok(Some(report))
}
