//! Shared parity/benchmark harness for backend crates (`tests/fixtures/*.json`) and a
//! synthetic backend for tests that need answers without a model.

use crate::backend::{Backend, BackendOptions, BackendOutput, Batch};
use crate::decode::{decode_answers, ActHead, Temperatures};
use crate::question::parse_questions;
use crate::weights::Weights;
use crate::{Agent, Error, ModelConfig, Result};
use serde_json::Value;
use std::path::PathBuf;

pub type Factory =
    Box<dyn FnOnce(&Weights, &ModelConfig, &BackendOptions) -> Result<Box<dyn Backend>>>;

/// A backend whose outputs are a deterministic function of each row's own tokens and marker
/// positions. The same row gets the same logits and pooled state whatever batch it sits in
/// and however far it is padded, so a test can check the batch path against single-state
/// calls. `pad_to` rounds the batch length up to a multiple, like a bucketing backend.
#[derive(Debug, Clone)]
pub struct SyntheticBackend {
    pub hidden: usize,
    pub pad_to: usize,
}

impl SyntheticBackend {
    /// FNV-1a over the row's real tokens, so the hash is independent of padding.
    fn row_hash(tokens: &[u32]) -> u64 {
        tokens.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &t| {
            (h ^ t as u64).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }
    /// A value in `[0, 1)` from a hash and a salt.
    fn unit(h: u64, salt: u64) -> f32 {
        let mut x = h ^ salt.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
        x ^= x >> 33;
        (x >> 40) as f32 / (1u64 << 24) as f32
    }
}

impl Backend for SyntheticBackend {
    fn name(&self) -> String {
        "synthetic".into()
    }
    fn padded_len(&self, len: usize, _rows: usize) -> usize {
        let m = self.pad_to.max(1);
        len.div_ceil(m) * m
    }
    fn forward(&self, b: &Batch) -> Result<BackendOutput> {
        let mut logits = vec![-1e4f32; b.n * b.kmax];
        let mut pooled = vec![0f32; b.n * self.hidden];
        for r in 0..b.n {
            let row = &b.input_ids[r * b.len..(r + 1) * b.len];
            let mask = &b.attention_mask[r * b.len..(r + 1) * b.len];
            let n = b.seq_lens[r];
            if mask[..n].iter().any(|&m| m != 1) || mask[n..].iter().any(|&m| m != 0) {
                return Err(Error::Backend(format!(
                    "row {r}: attention mask does not match seq_len {n}"
                )));
            }
            let h = Self::row_hash(&row[..n]);
            for k in 0..b.marker_count[r] {
                let pos = b.marker_pos[r * b.kmax + k] as u64;
                logits[r * b.kmax + k] = Self::unit(h, pos | (k as u64) << 32) * 4.0 - 2.0;
            }
            for j in 0..self.hidden {
                pooled[r * self.hidden + j] = Self::unit(h, 1_000_000 + j as u64) - 0.5;
            }
        }
        Ok(BackendOutput { logits, pooled })
    }
}

#[derive(Debug, Clone, Default)]
pub struct CaseReport {
    pub name: String,
    pub rows: usize,
    pub logits_max_abs: f32,
    pub pooled_max_abs: f32,
    pub encoder_hidden_max_abs: Option<f32>,
    pub answers_equal: bool,
    /// Answers whose decision differs from the fixture: the `choice` label, the noul
    /// decision (`noul >= 0.5`) or the score level with the largest probability, the same
    /// rules as `bench/harness/compare.py`. A missing answer counts too.
    pub decision_mismatches: usize,
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
    pub fn decision_mismatches(&self) -> usize {
        self.cases.iter().map(|c| c.decision_mismatches).sum()
    }
    pub fn all_answers_equal(&self) -> bool {
        self.cases.iter().all(|c| c.answers_equal)
    }
    pub fn summary(&self) -> String {
        let mut s = format!("backend {}\n", self.backend);
        for c in &self.cases {
            s += &format!(
                "  {:<12} rows {:<3} logits {:.4}  pooled {:.4}  enc {}  probs {:.4}  decision_mismatch {}  exact {}  fwd {:.1} ms\n",
                c.name, c.rows, c.logits_max_abs, c.pooled_max_abs,
                c.encoder_hidden_max_abs.map(|x| format!("{x:.4}")).unwrap_or_else(|| "-".into()),
                c.prob_max_abs, c.decision_mismatches, c.answers_equal, c.forward_ms
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

/// `bench/models.lock.json` of this checkout: the published checkpoints with the hub revision
/// every contender must load.
pub fn models_lock_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../bench/models.lock.json")
}

/// The lock entry of the English checkpoint, whose repo at its pinned revision also bundles
/// the other two as subfolders (`multilingual/`, `typed-decisions/`) holding the same blobs
/// as the separate repos at their pins.
const ENGLISH: &str = "english";

/// One entry of `bench/models.lock.json`: a hub repo and the commit hash it is pinned at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pin {
    pub repo: String,
    pub sha: String,
}

/// The lock entry for a checkpoint variant (`None` is `english`; `Some("multilingual")` and
/// `Some("typed-decisions")` name the other two).
pub fn checkpoint_pin(lock: &Value, variant: Option<&str>) -> Result<Pin> {
    let name = variant.unwrap_or(ENGLISH);
    let entry = lock
        .get(name)
        .ok_or_else(|| Error::Config(format!("bench/models.lock.json has no entry {name:?}")))?;
    let field = |k: &str| {
        entry.get(k).and_then(Value::as_str).ok_or_else(|| {
            Error::Config(format!(
                "bench/models.lock.json entry {name:?} has no string {k:?}"
            ))
        })
    };
    Ok(Pin {
        repo: field("repo")?.to_string(),
        sha: field("sha")?.to_string(),
    })
}

/// The directory of a published checkpoint in the local HF cache, at the revision
/// `bench/models.lock.json` pins for `variant`. The pinned snapshot of the variant's own repo
/// is taken when it is cached, whatever `refs/main` says; a moved default branch must not
/// swap the weights under the parity fixtures. Otherwise, for a variant other than English,
/// the English bundle's subfolder of that name at the bundle's own pin is the same
/// checkpoint (the same blobs at both pins), so it is the fallback. That match cannot be
/// checked here without the separate repo's files, which is the case that falls back. The
/// error names the pin when neither is cached.
pub fn checkpoint_dir(variant: Option<&str>) -> Result<PathBuf> {
    let path = models_lock_path();
    let lock: Value = serde_json::from_slice(
        &std::fs::read(&path)
            .map_err(|e| Error::Config(format!("cannot read {}: {e}", path.display())))?,
    )?;
    checkpoint_dir_in(&crate::resolve::hf_cache_dir(), &lock, variant)
}

fn checkpoint_dir_in(
    cache: &std::path::Path,
    lock: &Value,
    variant: Option<&str>,
) -> Result<PathBuf> {
    use crate::resolve::is_checkpoint;
    let snapshots = |repo: &str| {
        cache
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots")
    };
    let pin = checkpoint_pin(lock, variant)?;
    let pinned = snapshots(&pin.repo).join(&pin.sha);
    if is_checkpoint(&pinned) {
        return Ok(pinned);
    }
    if let Some(v) = variant.filter(|v| *v != ENGLISH) {
        let english = checkpoint_pin(lock, None)?;
        let bundled = snapshots(&english.repo).join(&english.sha).join(v);
        if is_checkpoint(&bundled) {
            return Ok(bundled);
        }
    }
    let cached: Vec<String> = std::fs::read_dir(snapshots(&pin.repo))
        .map(|rd| {
            rd.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    Err(Error::Config(format!(
        "checkpoint {} at pinned revision {} (bench/models.lock.json {:?}) is not in the HF cache at {}; cached snapshots: [{}]; fetch it with snapshot_download({:?}, revision={:?}) as bench/workloads/README.md does",
        pin.repo,
        pin.sha,
        variant.unwrap_or(ENGLISH),
        snapshots(&pin.repo).display(),
        cached.join(", "),
        pin.repo,
        pin.sha
    )))
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
/// than a comparison over the shorter prefix, which would report zero for a truncated output,
/// and a value that is not finite on either side is an error rather than a NaN difference
/// that `f32::max` would drop.
fn max_abs_diff(case: &str, what: &str, got: &[f32], want: &[f32]) -> Result<f32> {
    if got.len() != want.len() {
        return Err(crate::Error::Backend(format!(
            "{case}: {what} has {} values, the fixture has {}",
            got.len(),
            want.len()
        )));
    }
    let mut worst = 0f32;
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        let d = (a - b).abs();
        if !d.is_finite() {
            return Err(crate::Error::Backend(format!(
                "{case}: {what}[{i}] is {a}, the fixture has {b}; not a finite difference"
            )));
        }
        worst = worst.max(d);
    }
    Ok(worst)
}

/// The key with the largest value, first in object order on a tie (`max(probs, key=...)`).
fn argmax_key(probs: &Value) -> Option<&str> {
    let mut best: Option<(&str, f64)> = None;
    for (k, v) in probs.as_object()? {
        let x = v.as_f64()?;
        if best.is_none_or(|(_, b)| x > b) {
            best = Some((k, x));
        }
    }
    best.map(|(k, _)| k)
}

/// What an answer decides, by the rules of `bench/harness/compare.py`: the `choice` label, the
/// noul decision (`noul >= 0.5`), or the score level with the largest probability. `None`
/// when the answer has no decision to read.
pub fn decision(answer: &Value) -> Option<Value> {
    match answer.get("type").and_then(Value::as_str)? {
        "choice" => answer
            .get("choice")
            .cloned()
            .or_else(|| argmax_key(&answer["probabilities"]).map(Value::from)),
        "noul" => Some(Value::from(answer.get("noul")?.as_f64()? >= 0.5)),
        "score" => match answer.get("probabilities") {
            Some(p) if p.as_object().is_some_and(|m| !m.is_empty()) => {
                Some(Value::from(argmax_key(p)?.parse::<i64>().ok()?))
            }
            _ => {
                let s = answer.get("score")?.as_f64()?;
                (s.fract() == 0.0).then(|| Value::from(s as i64))
            }
        },
        _ => None,
    }
}

/// Whether two answers make the same decision (`compare.py`'s `agree`).
pub fn same_decision(got: &Value, want: &Value) -> bool {
    if got.get("type") != want.get("type") {
        return false;
    }
    match (decision(got), decision(want)) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

/// Largest absolute difference over the numeric leaves of the fixture answer (`want`),
/// `act_probability` included. Every number the fixture has must be a finite number in `got`
/// too: a NaN from the backend turns into JSON `null` through `round_dp`, and skipping it would
/// let a broken forward pass through with its decision unchanged. `path` names the leaf.
fn numeric_max_diff(path: &str, got: &Value, want: &Value, prob_max: &mut f64) -> Result<()> {
    match want {
        Value::Object(b) => {
            // `get` on a non-object is `None`: a missing object fails at its first number.
            for (k, wv) in b {
                let gv = got.get(k).unwrap_or(&Value::Null);
                numeric_max_diff(&format!("{path}.{k}"), gv, wv, prob_max)?;
            }
            Ok(())
        }
        Value::Number(b) => {
            let finite = |v: &Value| v.as_f64().filter(|x| x.is_finite());
            let (Some(x), Some(y)) = (finite(got), finite(want)) else {
                return Err(Error::Backend(format!(
                    "{path} is {got}, the fixture has {b}; not a finite number"
                )));
            };
            *prob_max = prob_max.max((x - y).abs());
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Compare two `answers` objects: the max abs difference over numeric leaves and the number
/// of questions whose decision differs or whose answer is missing. An answer whose numbers
/// cannot be compared is an error, see [`numeric_max_diff`].
fn compare_answers(got: &Value, want: &Value) -> Result<(f64, usize)> {
    let (mut prob_max, mut mismatches) = (0f64, 0usize);
    let Some(want) = want.as_object() else {
        return Ok((prob_max, mismatches));
    };
    for (qid, w) in want {
        match got.get(qid) {
            Some(g) => {
                if !same_decision(g, w) {
                    mismatches += 1;
                }
                numeric_max_diff(qid, g, w, &mut prob_max)?;
            }
            None => mismatches += 1,
        }
    }
    Ok((prob_max, mismatches))
}

/// Fixtures from a laya older than 0.3.21 have no `answer_confidence`; drop it from `got`
/// so the exact comparison is over the keys the fixture has.
fn drop_answer_confidence_if_absent(got: &mut Value, want: &Value) {
    let fixture_has_key = want
        .as_object()
        .is_some_and(|m| m.values().any(|a| a.get("answer_confidence").is_some()));
    if fixture_has_key {
        return;
    }
    if let Some(answers) = got.as_object_mut() {
        for a in answers.values_mut() {
            if let Some(a) = a.as_object_mut() {
                a.remove("answer_confidence");
            }
        }
    }
}

/// Run every fixture case through `factory`'s backend and compare with the Python outputs.
/// `variant` names the checkpoint as [`checkpoint_dir`] does (`None` for English,
/// `Some("multilingual")` for `convaiinnovations/laya-multilingual`), at the revision
/// `bench/models.lock.json` pins. Returns `Ok(None)` when the fixture file or that snapshot
/// of the checkpoint is not available locally.
pub fn run_parity(
    fixture: &str,
    variant: Option<&str>,
    factory: Factory,
    opts: &BackendOptions,
) -> Result<Option<ParityReport>> {
    let Ok(bytes) = std::fs::read(fixtures_path(fixture)) else {
        return Ok(None);
    };
    let fx: Value = serde_json::from_slice(&bytes)?;
    let Ok(dir) = checkpoint_dir(variant) else {
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
        // Only the real option slots of each row; the padded ones hold the mask fill.
        let (mut got_l, mut want_l) = (Vec::new(), Vec::new());
        for r in 0..batch.n {
            for k in 0..batch.marker_count[r] {
                let i = r * batch.kmax + k;
                got_l.push(out.logits[i]);
                want_l.push(*want_logits.get(i).ok_or_else(|| {
                    Error::Backend(format!(
                        "{name}: masked_logits has {} values, row {r} slot {k} needs index {i}",
                        want_logits.len()
                    ))
                })?);
            }
        }
        let logits_max = max_abs_diff(&name, "masked logits", &got_l, &want_l)?;
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
        let mut answers = decode_answers(&out, &batch, &act_head, &temps, &qs, 0)?;
        let want = &case["result"]["answers"];
        drop_answer_confidence_if_absent(&mut answers, want);
        let (prob_max, decision_mismatches) = compare_answers(&answers, want)?;
        report.cases.push(CaseReport {
            name,
            rows: batch.n,
            logits_max_abs: logits_max,
            pooled_max_abs: pooled_max,
            encoder_hidden_max_abs,
            answers_equal: &answers == want,
            decision_mismatches,
            prob_max_abs: prob_max,
            forward_ms,
        });
    }
    let _ = d;
    Ok(Some(report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A fake HF cache under the system temp dir, removed on drop. `snapshot` writes the two
    /// files a checkpoint directory must have and names the snapshot in `refs/main`.
    struct FakeCache(PathBuf);

    impl FakeCache {
        fn new() -> Self {
            static N: AtomicUsize = AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!(
                "laya-core-testing-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
        fn snapshot(&self, repo: &str, rev: &str, subfolder: Option<&str>) -> PathBuf {
            let repo_dir = self.0.join(format!("models--{}", repo.replace('/', "--")));
            let mut dir = repo_dir.join("snapshots").join(rev);
            if let Some(s) = subfolder {
                dir = dir.join(s);
            }
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("rl_agent_config.json"), "{}").unwrap();
            std::fs::write(dir.join("model.safetensors"), "").unwrap();
            std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
            std::fs::write(repo_dir.join("refs").join("main"), rev).unwrap();
            dir
        }
    }

    impl Drop for FakeCache {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The real lock of this checkout: the three checkpoints the fixtures and benchmarks use,
    /// each a 40-hex commit hash, and an unknown name is an error.
    #[test]
    fn checkpoint_pins_read_models_lock() {
        let lock: Value =
            serde_json::from_slice(&std::fs::read(models_lock_path()).unwrap()).unwrap();
        for (variant, repo) in [
            (None, "convaiinnovations/laya"),
            (Some("english"), "convaiinnovations/laya"),
            (Some("multilingual"), "convaiinnovations/laya-multilingual"),
            (
                Some("typed-decisions"),
                "convaiinnovations/laya-typed-decisions",
            ),
        ] {
            let pin = checkpoint_pin(&lock, variant).unwrap();
            assert_eq!(pin.repo, repo);
            assert!(
                pin.sha.len() == 40 && pin.sha.chars().all(|c| c.is_ascii_hexdigit()),
                "{variant:?}: {}",
                pin.sha
            );
        }
        let e = checkpoint_pin(&lock, Some("klingon"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("no entry \"klingon\""), "{e}");
    }

    /// A lock like `bench/models.lock.json` with short fake revisions.
    fn fake_lock() -> Value {
        json!({
            "english": {"repo": "convaiinnovations/laya", "sha": "aaa"},
            "multilingual": {"repo": "convaiinnovations/laya-multilingual", "sha": "bbb"},
        })
    }

    /// The pinned snapshot of the published repo wins even after `refs/main` moved to a newer
    /// snapshot, and even when the English bundle is cached too.
    #[test]
    fn checkpoint_dir_takes_the_pinned_snapshot_not_refs_main() {
        let c = FakeCache::new();
        let lock = fake_lock();
        let pinned = c.snapshot("convaiinnovations/laya-multilingual", "bbb", None);
        assert_eq!(
            checkpoint_dir_in(&c.0, &lock, Some("multilingual")).unwrap(),
            pinned
        );
        c.snapshot("convaiinnovations/laya-multilingual", "ccc", None);
        assert_eq!(
            std::fs::read_to_string(
                c.0.join("models--convaiinnovations--laya-multilingual/refs/main")
            )
            .unwrap(),
            "ccc"
        );
        assert_eq!(
            checkpoint_dir_in(&c.0, &lock, Some("multilingual")).unwrap(),
            pinned
        );
        c.snapshot("convaiinnovations/laya", "aaa", Some("multilingual"));
        assert_eq!(
            checkpoint_dir_in(&c.0, &lock, Some("multilingual")).unwrap(),
            pinned
        );
        // English has no fallback repo: its pin or nothing, whatever `refs/main` names.
        let english = c.snapshot("convaiinnovations/laya", "aaa", None);
        c.snapshot("convaiinnovations/laya", "zzz", None);
        assert_eq!(checkpoint_dir_in(&c.0, &lock, None).unwrap(), english);
        assert_eq!(
            checkpoint_dir_in(&c.0, &lock, Some("english")).unwrap(),
            english
        );
    }

    /// Without the published repo's pinned snapshot, the English bundle's subfolder stands
    /// in, but only at the bundle's own pin: the same subfolder at another revision of the
    /// bundle is not the pinned checkpoint.
    #[test]
    fn checkpoint_dir_falls_back_to_the_bundle_at_its_own_pin() {
        let c = FakeCache::new();
        let lock = fake_lock();
        c.snapshot("convaiinnovations/laya", "zzz", Some("multilingual"));
        let e = checkpoint_dir_in(&c.0, &lock, Some("multilingual"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("convaiinnovations/laya-multilingual"), "{e}");
        let bundled = c.snapshot("convaiinnovations/laya", "aaa", Some("multilingual"));
        assert_eq!(
            checkpoint_dir_in(&c.0, &lock, Some("multilingual")).unwrap(),
            bundled
        );
    }

    /// Neither the pinned snapshot nor the bundle: the error names the repo, the pinned
    /// revision, the lock entry and the snapshots that are cached instead.
    #[test]
    fn a_missing_pinned_snapshot_names_the_pin() {
        let c = FakeCache::new();
        let lock = fake_lock();
        let e = checkpoint_dir_in(&c.0, &lock, Some("multilingual"))
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("convaiinnovations/laya-multilingual at pinned revision bbb")
                && e.contains("\"multilingual\"")
                && e.contains("cached snapshots: []")
                && e.contains("revision=\"bbb\""),
            "{e}"
        );
        c.snapshot("convaiinnovations/laya-multilingual", "ccc", None);
        let e = checkpoint_dir_in(&c.0, &lock, Some("multilingual"))
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("at pinned revision bbb") && e.contains("cached snapshots: [ccc]"),
            "{e}"
        );
        let e = checkpoint_dir_in(&c.0, &lock, None)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("convaiinnovations/laya at pinned revision aaa"),
            "{e}"
        );
    }

    #[test]
    fn decisions_follow_compare_py() {
        let choice =
            json!({"type": "choice", "choice": "b", "probabilities": {"a": 0.4, "b": 0.6}});
        assert_eq!(decision(&choice), Some(json!("b")));
        // Without `choice`, the argmax of the probabilities; first key wins a tie.
        let tie = json!({"type": "choice", "probabilities": {"a": 0.5, "b": 0.5}});
        assert_eq!(decision(&tie), Some(json!("a")));
        assert_eq!(
            decision(&json!({"type": "noul", "noul": 0.495})),
            Some(json!(false))
        );
        assert_eq!(
            decision(&json!({"type": "noul", "noul": 0.5})),
            Some(json!(true))
        );
        let score =
            json!({"type": "score", "score": 1.7, "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}});
        assert_eq!(decision(&score), Some(json!(2)));
        assert_eq!(
            decision(&json!({"type": "score", "score": 2.0})),
            Some(json!(2))
        );
        assert_eq!(decision(&json!({"type": "score", "score": 1.7})), None);
        assert_eq!(decision(&json!({"type": "other"})), None);
    }

    /// A flipped noul or a changed score level counts as a mismatch even when every number
    /// stays within a small tolerance, and a changed type or a missing answer counts too.
    #[test]
    fn mismatches_count_every_decision_type() {
        let want = json!({
            "c": {"type": "choice", "choice": "a", "probabilities": {"a": 0.51, "b": 0.49}},
            "n": {"type": "noul", "noul": 0.495},
            "s": {"type": "score", "score": 1.5, "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5}},
            "m": {"type": "noul", "noul": 0.9},
        });
        let got = json!({
            "c": {"type": "choice", "choice": "b", "probabilities": {"a": 0.49, "b": 0.51}},
            "n": {"type": "noul", "noul": 0.505},
            "s": {"type": "score", "score": 1.5, "probabilities": {"0": 0.0, "1": 0.49, "2": 0.51}},
        });
        let (prob_max, mismatches) = compare_answers(&got, &want).unwrap();
        assert_eq!(mismatches, 4);
        assert!((prob_max - 0.02).abs() < 1e-9, "{prob_max}");
        let (prob_max, mismatches) = compare_answers(&want, &want).unwrap();
        assert_eq!((prob_max, mismatches), (0.0, 0));
        // A changed type is a mismatch; the fixture's numbers the answer lacks are an error.
        let other_type = json!({"c": {"type": "noul", "noul": 1.0}});
        let want_c = json!({"c": {"type": "choice", "choice": "a"}});
        assert_eq!(compare_answers(&other_type, &want_c).unwrap().1, 1);
        let e = compare_answers(&other_type, &json!({"c": want["c"]}))
            .unwrap_err()
            .to_string();
        assert!(e.contains("c.probabilities.a is null"), "{e}");
    }

    /// A NaN from the backend must fail the case, not vanish in a `max`: the drift helper
    /// errors on it, and so does the answer comparison on the `act_probability` it becomes
    /// (`round_dp(NaN)` serializes as `null`), while the decision itself is unchanged.
    #[test]
    fn a_non_finite_backend_output_fails_the_case() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let e = max_abs_diff("c", "pooled", &[0.0, bad], &[0.0, 0.0])
                .unwrap_err()
                .to_string();
            assert!(
                e.contains("pooled[1]") && e.contains("not a finite difference"),
                "{e}"
            );
        }
        assert_eq!(
            max_abs_diff("c", "pooled", &[0.5, 1.0], &[0.0, 0.0]).unwrap(),
            1.0
        );

        // Decode one noul row whose pooled state holds a NaN: the logits are fine, so the
        // decision is the fixture's, but `act_probability` is null.
        let qs = parse_questions(&json!({"q": {"type": "noul", "instructions": "x"}})).unwrap();
        let batch = crate::sequence::collate(
            &[crate::sequence::EncodedItem {
                ids: vec![0, 1, 2],
                markers: vec![1, 2],
                qtype: crate::QType::Noul,
            }],
            0,
        );
        let out = BackendOutput {
            logits: vec![0.0, 1.0],
            pooled: vec![0.25, f32::NAN],
        };
        let head = ActHead::from_tensors(
            vec![1, 6],
            vec![0.1; 6],
            vec![0.0],
            vec![2, 1],
            vec![0.5, -0.5],
            vec![0.0; 2],
        )
        .unwrap();
        let temps = Temperatures {
            by_type: [1.0; 3],
            by_options: serde_json::Map::new(),
            rejected: vec![],
        };
        let got = decode_answers(&out, &batch, &head, &temps, &qs, 0).unwrap();
        assert!(got["q"]["action"]["act_probability"].is_null(), "{got}");
        assert_eq!(got["q"]["noul"], 0.7311);
        let want = json!({"q": {"type": "noul", "noul": 0.7311, "confidence": 0.7311,
                                 "answer_confidence": 0.7311, "action": {"act_probability": 0.4}}});
        assert!(same_decision(&got["q"], &want["q"]));
        let e = compare_answers(&got, &want).unwrap_err().to_string();
        assert!(
            e.contains("q.action.act_probability is null") && e.contains("fixture has 0.4"),
            "{e}"
        );
        // With a finite pooled state the same row compares, act_probability included.
        let out = BackendOutput {
            logits: vec![0.0, 1.0],
            pooled: vec![0.25, 0.25],
        };
        let got = decode_answers(&out, &batch, &head, &temps, &qs, 0).unwrap();
        let (prob_max, mismatches) = compare_answers(&got, &want).unwrap();
        let act = got["q"]["action"]["act_probability"].as_f64().unwrap();
        assert!(((act - 0.4).abs() - prob_max).abs() < 1e-12, "{got}");
        assert_eq!(mismatches, 0);
    }

    #[test]
    fn answer_confidence_is_dropped_only_for_old_fixtures() {
        let old = json!({"q": {"type": "noul", "noul": 0.7, "confidence": 0.7}});
        let mut got = json!({"q": {"type": "noul", "noul": 0.7, "confidence": 0.7, "answer_confidence": 0.7}});
        drop_answer_confidence_if_absent(&mut got, &old);
        assert_eq!(got, old);
        let new = json!({"q": {"type": "noul", "noul": 0.7, "confidence": 0.7, "answer_confidence": 0.7}});
        let mut got = new.clone();
        drop_answer_confidence_if_absent(&mut got, &new);
        assert_eq!(got, new);
    }

    /// The synthetic backend depends only on a row's own tokens and markers.
    #[test]
    fn synthetic_backend_is_padding_and_batch_independent() {
        let b = SyntheticBackend {
            hidden: 4,
            pad_to: 8,
        };
        let mk = |rows: Vec<Vec<u32>>, len: usize| Batch {
            n: rows.len(),
            len,
            kmax: 2,
            input_ids: rows
                .iter()
                .flat_map(|r| {
                    let mut v = r.clone();
                    v.resize(len, 0);
                    v
                })
                .collect(),
            attention_mask: rows
                .iter()
                .flat_map(|r| {
                    let mut v = vec![1u32; r.len()];
                    v.resize(len, 0);
                    v
                })
                .collect(),
            seq_lens: rows.iter().map(Vec::len).collect(),
            marker_pos: rows.iter().flat_map(|_| [1, 3]).collect(),
            marker_count: rows.iter().map(|_| 2).collect(),
            qtype: rows.iter().map(|_| 0).collect(),
        };
        let alone = b.forward(&mk(vec![vec![5, 6, 7, 8, 9]], 5)).unwrap();
        let padded = b
            .forward(&mk(vec![vec![1, 2], vec![5, 6, 7, 8, 9]], 16))
            .unwrap();
        assert_eq!(alone.logits, padded.logits[2..4]);
        assert_eq!(alone.pooled, padded.pooled[4..8]);
        assert_ne!(padded.logits[..2], padded.logits[2..4]);
        assert_eq!(b.padded_len(5, 1), 8);
        assert_eq!(b.padded_len(16, 1), 16);
        // A mask that disagrees with seq_lens is an error, not a silent wrong answer.
        let mut bad = mk(vec![vec![1, 2, 3]], 4);
        bad.attention_mask[3] = 1;
        assert!(b.forward(&bad).is_err());
    }
}
