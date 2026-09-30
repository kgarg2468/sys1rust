//! Temperature scaling, calibrated confidence, the action head and typed answer decoding
//! (`Agent._decode_answers`, `DecisionModel.forward`'s act-head tail, `confidence_from_probs`).
//!
//! Changed in sys1rust from laya-r-mlx 914c9a7: every answer carries `answer_confidence`
//! (laya 0.3.21), the probability mass on the reported answer, between `confidence` and
//! `action`.

use crate::backend::{BackendOutput, Batch};
use crate::config::AgentConfig;
use crate::pyjson::round_dp;
use crate::question::{QType, Question};
use crate::weights::Weights;
use crate::{Error, Result};
use serde_json::{json, Map, Value};

pub const TEMP_MIN: f64 = 0.5;
pub const TEMP_MAX: f64 = 5.0;

/// `clamp_temperature`: a usable temperature confined to `[0.5, 5.0]`, `1.0` when not a number.
pub fn clamp_temperature(v: &Value) -> f64 {
    let t = match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    };
    match t {
        Some(t) if t.is_finite() => t.clamp(TEMP_MIN, TEMP_MAX),
        _ => 1.0,
    }
}

/// Calibration temperatures as the runtime applies them (already clamped).
#[derive(Debug, Clone)]
pub struct Temperatures {
    pub by_type: [f64; 3],
    pub by_options: Map<String, Value>,
    /// Entries the checkpoint shipped that were rejected/clamped (`name=raw -> applied`).
    pub rejected: Vec<String>,
}

impl Temperatures {
    pub fn from_config(cfg: &AgentConfig) -> Self {
        let mut by_type = [1.0; 3];
        let mut rejected = Vec::new();
        for (i, raw) in cfg.temperature.iter().enumerate().take(3) {
            by_type[i] = clamp_temperature(raw);
            if raw.as_f64() != Some(by_type[i]) {
                rejected.push(format!("temperature[{i}]={raw} -> {}", by_type[i]));
            }
        }
        let mut by_options = Map::new();
        for (k, raw) in &cfg.temperature_by_options {
            let t = clamp_temperature(raw);
            if raw.as_f64() != Some(t) {
                rejected.push(format!("{k}={raw} -> {t}"));
            }
            by_options.insert(k.clone(), Value::from(t));
        }
        Self {
            by_type,
            by_options,
            rejected,
        }
    }

    pub fn temp_bucket(qtype: QType, k: usize) -> String {
        let size = if k <= 2 {
            "2"
        } else if k <= 5 {
            "3-5"
        } else if k <= 10 {
            "6-10"
        } else {
            "11+"
        };
        format!("{}:{}", qtype.name(), size)
    }

    pub fn scale(&self, qtype: QType, k: usize) -> f64 {
        self.by_options
            .get(&Self::temp_bucket(qtype, k))
            .and_then(Value::as_f64)
            .unwrap_or(self.by_type[qtype.index()])
    }
}

/// `confidence_from_probs`: normalized Shannon entropy confidence `1 - H(p) / log(k)`.
pub fn confidence_from_probs(p: &[f64], k: usize) -> f64 {
    if k < 2 {
        return 1.0;
    }
    let ent: f64 = p[..k].iter().map(|&x| -x * x.clamp(1e-12, 1.0).ln()).sum();
    (1.0 - ent / (k as f64).ln()).clamp(0.0, 1.0)
}

fn softmax_f64(z: &[f64]) -> Vec<f64> {
    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|&x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

fn softmax_f32(z: &[f32]) -> Vec<f32> {
    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|&x| (x - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

/// Exact (erf) GELU, as `torch.nn.GELU()` / `F.gelu` default.
pub fn gelu(x: f32) -> f32 {
    let xf = x as f64;
    (0.5 * xf * (1.0 + libm::erf(xf / std::f64::consts::SQRT_2))) as f32
}

/// `act_head = Linear(d + 4, 256) -> GELU -> Linear(256, n_act)`, evaluated on the CPU in f32.
#[derive(Debug, Clone)]
pub struct ActHead {
    pub d_in: usize,
    pub hidden: usize,
    pub n_act: usize,
    w0: Vec<f32>,
    b0: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl ActHead {
    pub fn load(w: &Weights) -> Result<Self> {
        let (s0, w0) = w.tensor_f32("act_head.0.weight")?;
        let (_, b0) = w.tensor_f32("act_head.0.bias")?;
        let (s2, w2) = w.tensor_f32("act_head.2.weight")?;
        let (_, b2) = w.tensor_f32("act_head.2.bias")?;
        if s0.len() != 2 || s2.len() != 2 || s2[1] != s0[0] {
            return Err(Error::Weights(format!(
                "act_head shapes {s0:?} / {s2:?} are not a 2-layer MLP"
            )));
        }
        Ok(Self {
            d_in: s0[1],
            hidden: s0[0],
            n_act: s2[0],
            w0,
            b0,
            w2,
            b2,
        })
    }

    /// Action-class probabilities for one row.
    pub fn probs(&self, pooled: &[f32], feats: [f32; 4]) -> Vec<f32> {
        debug_assert_eq!(pooled.len() + 4, self.d_in);
        let mut h = vec![0f32; self.hidden];
        for (j, hj) in h.iter_mut().enumerate() {
            let row = &self.w0[j * self.d_in..(j + 1) * self.d_in];
            let mut acc = self.b0[j];
            for (a, b) in row[..pooled.len()].iter().zip(pooled) {
                acc += a * b;
            }
            for (a, b) in row[pooled.len()..].iter().zip(feats.iter()) {
                acc += a * b;
            }
            *hj = gelu(acc);
        }
        let mut out = vec![0f32; self.n_act];
        for (j, oj) in out.iter_mut().enumerate() {
            let row = &self.w2[j * self.hidden..(j + 1) * self.hidden];
            *oj = self.b2[j] + row.iter().zip(&h).map(|(a, b)| a * b).sum::<f32>();
        }
        softmax_f32(&out)
    }
}

/// The four scalar features the action head sees, from one row of masked logits.
pub fn act_features(logits_row: &[f32], marker_count: usize) -> [f32; 4] {
    let p = softmax_f32(logits_row);
    let k = marker_count.max(2) as f32;
    let ent: f32 = -p.iter().map(|&x| x * x.max(1e-9).ln()).sum::<f32>() / k.ln();
    let mut sorted = p.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let top1 = sorted.first().copied().unwrap_or(0.0);
    let top2 = sorted.get(1).copied().unwrap_or(0.0);
    [top1, top1 - top2, ent, k / 255.0]
}

/// Decode one state's rows (`rows[0]..rows[n]` of `out`) into the Python result object.
pub fn decode_answers(
    out: &BackendOutput,
    batch: &Batch,
    act_head: &ActHead,
    temps: &Temperatures,
    questions: &[Question],
    offset: usize,
) -> Result<Value> {
    let d = act_head.d_in - 4;
    let mut answers = Map::with_capacity(questions.len());
    for (j, q) in questions.iter().enumerate() {
        let r = offset + j;
        let k = batch.marker_count[r];
        let row = &out.logits[r * batch.kmax..(r + 1) * batch.kmax];
        let pooled = &out.pooled[r * d..(r + 1) * d];
        let act = act_head.probs(pooled, act_features(row, k));
        let act_probability = round_dp(act[0] as f64, 4);

        let t_scale = temps.scale(q.qtype, k);
        let z: Vec<f64> = row[..k].iter().map(|&x| x as f64 / t_scale).collect();
        let p = softmax_f64(&z);
        let conf = round_dp(confidence_from_probs(&p, k), 4);
        let ext = json!({ "act_probability": act_probability });

        let ans = match q.qtype {
            QType::Choice => {
                let argmax = p
                    .iter()
                    .enumerate()
                    .fold(0, |b, (i, &x)| if x > p[b] { i } else { b });
                let mut probs = Map::new();
                for (key, &v) in q.choice_keys.iter().zip(&p) {
                    probs.insert(key.clone(), Value::from(round_dp(v, 4)));
                }
                json!({
                    "type": "choice",
                    "choice": q.choice_keys[argmax],
                    "probabilities": probs,
                    "confidence": conf,
                    "answer_confidence": round_dp(p[argmax], 4),
                    "action": ext,
                })
            }
            QType::Score => {
                let exp_score: f64 = p.iter().enumerate().map(|(i, &x)| i as f64 * x).sum();
                let mut legend = Map::new();
                let mut probs = Map::new();
                for (i, c) in q.score_legend.iter().enumerate() {
                    legend.insert(i.to_string(), c.clone());
                }
                for (i, &v) in p.iter().enumerate() {
                    probs.insert(i.to_string(), Value::from(round_dp(v, 4)));
                }
                let top = p.iter().cloned().fold(0.0, f64::max);
                json!({
                    "type": "score",
                    "score": round_dp(exp_score, 4),
                    "legend": legend,
                    "probabilities": probs,
                    "confidence": conf,
                    "answer_confidence": round_dp(top, 4),
                    "action": ext,
                })
            }
            QType::Noul => {
                let p1 = p.get(1).copied().unwrap_or(0.0);
                // `max(p_yes, p_no)`: for noul the two confidences are the same number.
                let top = round_dp(p1.max(1.0 - p1), 4);
                json!({
                    "type": "noul",
                    "noul": round_dp(p1, 4),
                    "confidence": top,
                    "answer_confidence": top,
                    "action": ext,
                })
            }
        };
        answers.insert(q.id.clone(), ans);
    }
    Ok(Value::Object(answers))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_clamping() {
        assert_eq!(clamp_temperature(&json!(0.1006)), 0.5);
        assert_eq!(clamp_temperature(&json!(7.0)), 5.0);
        assert_eq!(clamp_temperature(&json!("1.5")), 1.5);
        assert_eq!(clamp_temperature(&json!("abc")), 1.0);
        assert_eq!(clamp_temperature(&Value::Null), 1.0);
        assert_eq!(Temperatures::temp_bucket(QType::Choice, 20), "choice:11+");
        assert_eq!(Temperatures::temp_bucket(QType::Noul, 2), "noul:2");
        assert_eq!(Temperatures::temp_bucket(QType::Score, 5), "score:3-5");
        assert_eq!(Temperatures::temp_bucket(QType::Choice, 6), "choice:6-10");
    }

    #[test]
    fn confidence() {
        assert_eq!(confidence_from_probs(&[1.0], 1), 1.0);
        assert!((confidence_from_probs(&[0.5, 0.5], 2)).abs() < 1e-12);
        assert!((confidence_from_probs(&[1.0, 0.0], 2) - 1.0).abs() < 1e-9);
    }

    /// An act head with zero weights: `act_probability` is a uniform softmax, so the test is
    /// only about the answer fields.
    fn zero_act_head(d: usize) -> ActHead {
        ActHead {
            d_in: d + 4,
            hidden: 1,
            n_act: 2,
            w0: vec![0.0; d + 4],
            b0: vec![0.0],
            w2: vec![0.0; 2],
            b2: vec![0.0; 2],
        }
    }

    fn question(id: &str, qtype: QType, k: usize) -> Question {
        let names: Vec<String> = (0..k).map(|i| format!("o{i}")).collect();
        Question {
            id: id.into(),
            qtype,
            instructions: String::new(),
            options: names.clone(),
            choice_keys: if qtype == QType::Choice {
                names
            } else {
                vec![]
            },
            score_legend: if qtype == QType::Score {
                (0..k).map(|i| json!(format!("level {i}"))).collect()
            } else {
                vec![]
            },
        }
    }

    #[test]
    fn answer_confidence_per_type() {
        // One row per type: choice over 3 options, score over 3 levels, noul (2 options,
        // padded to kmax 3). Temperatures are 1, so the probabilities are plain softmaxes.
        let qs = [
            question("pick", QType::Choice, 3),
            question("rate", QType::Score, 3),
            question("yes", QType::Noul, 2),
        ];
        let batch = Batch {
            n: 3,
            len: 1,
            kmax: 3,
            input_ids: vec![0; 3],
            attention_mask: vec![1; 3],
            seq_lens: vec![1; 3],
            marker_pos: vec![0; 9],
            marker_count: vec![3, 3, 2],
            qtype: vec![0, 1, 2],
        };
        let out = BackendOutput {
            logits: vec![2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, -1e4],
            pooled: vec![0.0; 3 * 2],
        };
        let temps = Temperatures {
            by_type: [1.0; 3],
            by_options: Map::new(),
            rejected: vec![],
        };
        let answers = decode_answers(&out, &batch, &zero_act_head(2), &temps, &qs, 0).unwrap();
        let keys = |a: &Value| -> Vec<String> { a.as_object().unwrap().keys().cloned().collect() };

        let pick = &answers["pick"];
        assert_eq!(
            keys(pick),
            [
                "type",
                "choice",
                "probabilities",
                "confidence",
                "answer_confidence",
                "action"
            ]
        );
        assert_eq!(pick["choice"], "o0");
        // e^2 / (e^2 + 2) = 0.78699, rounded to 4 dp like the probabilities.
        assert_eq!(pick["answer_confidence"], 0.787);
        assert_eq!(pick["answer_confidence"], pick["probabilities"]["o0"]);

        let rate = &answers["rate"];
        assert_eq!(
            keys(rate),
            [
                "type",
                "score",
                "legend",
                "probabilities",
                "confidence",
                "answer_confidence",
                "action"
            ]
        );
        // e / (e + 2) = 0.57612, the largest level probability.
        assert_eq!(rate["answer_confidence"], 0.5761);
        assert_eq!(rate["answer_confidence"], rate["probabilities"]["1"]);

        let yes = &answers["yes"];
        assert_eq!(
            keys(yes),
            ["type", "noul", "confidence", "answer_confidence", "action"]
        );
        // p_yes = 1 / (1 + e) = 0.26894; both confidences are max(p_yes, p_no).
        assert_eq!(yes["noul"], 0.2689);
        assert_eq!(yes["confidence"], 0.7311);
        assert_eq!(yes["answer_confidence"], yes["confidence"]);
    }
}
