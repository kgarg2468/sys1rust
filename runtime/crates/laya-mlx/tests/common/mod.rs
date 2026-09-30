//! Helpers shared by the model tests of this crate: the checkpoints, the bench files and the
//! answer comparison. Included with `mod common;` from each test file that needs it.

#![allow(dead_code)]

use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// The three published checkpoints, as `bench/models.lock.json` names them.
pub const CHECKPOINTS: [(&str, &str); 3] = [
    ("typed-decisions", "convaiinnovations/laya-typed-decisions"),
    ("multilingual", "convaiinnovations/laya-multilingual"),
    ("english", "convaiinnovations/laya"),
];

/// `bench/` of this checkout.
pub fn bench_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../bench")
        .canonicalize()
        .unwrap()
}

pub fn read_jsonl(path: &Path) -> Vec<Value> {
    BufReader::new(std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
        .lines()
        .map(|l| l.unwrap())
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(&l).unwrap())
        .collect()
}

/// What an answer decides: the choice label, the noul side, or the rounded score.
pub fn categorical(answer: &Value) -> Value {
    match answer["type"].as_str() {
        Some("choice") => answer["choice"].clone(),
        Some("noul") => Value::from(answer["noul"].as_f64().unwrap_or(0.0) >= 0.5),
        Some("score") => Value::from(answer["score"].as_f64().unwrap_or(0.0).round()),
        _ => Value::Null,
    }
}

/// Largest absolute difference between two answers' reported probabilities: the values of
/// `probabilities` for choice and score answers, `noul` for noul answers.
///
/// Both answers must report the same keys with finite numbers, or this is an `Err` naming the
/// first offending key. A missing or non-numeric value is never a difference of zero.
pub fn prob_diff(a: &Value, b: &Value) -> Result<f64, String> {
    let finite = |v: &Value, what: &str| -> Result<f64, String> {
        match v.as_f64() {
            Some(x) if x.is_finite() => Ok(x),
            _ => Err(format!("{what} is {v}, not a finite number")),
        }
    };
    match (a["probabilities"].as_object(), b["probabilities"].as_object()) {
        (Some(pa), Some(pb)) => {
            if let Some(k) = pb.keys().find(|k| !pa.contains_key(*k)) {
                return Err(format!("probabilities.{k} only on one side"));
            }
            let mut worst = 0.0f64;
            for (k, va) in pa {
                let vb = pb.get(k).ok_or_else(|| format!("probabilities.{k} only on one side"))?;
                let d = (finite(va, &format!("probabilities.{k}"))? - finite(vb, &format!("probabilities.{k}"))?).abs();
                worst = worst.max(d);
            }
            Ok(worst)
        }
        (None, None) => Ok((finite(&a["noul"], "noul")? - finite(&b["noul"], "noul")?).abs()),
        _ => Err("probabilities only on one side".into()),
    }
}
