//! Parity against the Python package, using fixtures dumped by the fork's `scripts/ref_dump.py`
//! into `runtime/tests/fixtures/<name>.json`. The fixtures are not in the repository and the
//! checks need the checkpoints in the HF cache, so the tests are ignored by default. Run them
//! from the repository root with
//! `cargo test --manifest-path runtime/Cargo.toml -p laya-core --test fixtures_parity -- --ignored`;
//! a missing fixture or checkpoint then fails instead of passing silently.

use laya_core::backend::{Backend, BackendOptions, BackendOutput, Batch};
use laya_core::decode::{act_features, ActHead, Temperatures};
use laya_core::{parse_questions, Agent, Weights};
use serde_json::Value;
use std::path::PathBuf;

fn fixtures(name: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(format!("{name}.json"));
    let bytes = std::fs::read(&p).unwrap_or_else(|e| {
        panic!(
            "fixture {} is missing ({e}); generate it with scripts/ref_dump.py from the laya-r-mlx fork",
            p.display()
        )
    });
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("fixture {} is not valid JSON: {e}", p.display()))
}

fn model_dir(subfolder: Option<&str>) -> PathBuf {
    laya_core::resolve::resolve_model_dir("convaiinnovations/laya", subfolder)
        .unwrap_or_else(|e| panic!("checkpoint for the parity fixtures is not available: {e}"))
}

/// A backend that replays the Python head outputs stored in the fixture.
struct ReplayBackend {
    logits: Vec<f32>,
    pooled: Vec<f32>,
}

impl Backend for ReplayBackend {
    fn name(&self) -> String {
        "replay".into()
    }
    fn forward(&self, _b: &Batch) -> laya_core::Result<BackendOutput> {
        Ok(BackendOutput {
            logits: self.logits.clone(),
            pooled: self.pooled.clone(),
        })
    }
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

fn check_checkpoint(name: &str, subfolder: Option<&str>) {
    let (fx, dir) = (fixtures(name), model_dir(subfolder));
    let weights = Weights::open(&dir).unwrap();
    let act_head = ActHead::load(&weights).unwrap();
    // Agent with a dummy backend just for tokenization/config.
    let agent = Agent::load(
        &dir,
        &BackendOptions::default(),
        Box::new(|_, _, _| {
            Ok(Box::new(ReplayBackend {
                logits: vec![],
                pooled: vec![],
            }) as Box<dyn Backend>)
        }),
    )
    .unwrap();
    let sp = &fx["special"];
    assert_eq!(agent.tokenizer.cls_id as u64, sp["cls"].as_u64().unwrap());
    assert_eq!(agent.tokenizer.sep_id as u64, sp["sep"].as_u64().unwrap());
    assert_eq!(agent.tokenizer.mask_id as u64, sp["mask"].as_u64().unwrap());
    assert_eq!(agent.tokenizer.pad_id as u64, sp["pad"].as_u64().unwrap());
    assert_eq!(
        agent.tokenizer.mask_token,
        sp["mask_token"].as_str().unwrap()
    );

    let temps = Temperatures::from_config(&agent.cfg.agent);
    for case in fx["cases"].as_array().unwrap() {
        let cname = case["name"].as_str().unwrap();
        let qs = parse_questions(&case["questions"]).unwrap();
        let items = agent.encode(&case["state"], &qs).unwrap();
        let want_items = case["items"].as_array().unwrap();
        assert_eq!(
            items.len(),
            want_items.len(),
            "{name}/{cname}: encoded {} items, the fixture has {}",
            items.len(),
            want_items.len()
        );
        for (it, want) in items.iter().zip(want_items) {
            let want_ids: Vec<u32> = want["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            let want_markers: Vec<u32> = want["markers"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(
                it.ids, want_ids,
                "{name}/{cname}/{}: token ids differ",
                want["qid"]
            );
            assert_eq!(
                it.markers, want_markers,
                "{name}/{cname}/{}: markers differ",
                want["qid"]
            );
            assert_eq!(it.qtype.index() as u64, want["qtype"].as_u64().unwrap());
        }
        // Decode from the Python head outputs and compare the full result object.
        let batch = agent.collate(&items);
        assert_eq!(batch.kmax as u64, case["kmax"].as_u64().unwrap());
        let out = BackendOutput {
            logits: flat_f32(&case["masked_logits"]),
            pooled: flat_f32(&case["pooled"]),
        };
        // act features/probabilities vs the Python act softmax
        let want_act = case["act_probs"].as_array().unwrap();
        for (r, want_row) in want_act.iter().enumerate().take(batch.n) {
            let d = act_head.d_in - 4;
            let feats = act_features(
                &out.logits[r * batch.kmax..(r + 1) * batch.kmax],
                batch.marker_count[r],
            );
            let p = act_head.probs(&out.pooled[r * d..(r + 1) * d], feats);
            let want: Vec<f32> = want_row
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            for (a, b) in p.iter().zip(&want) {
                assert!(
                    (a - b).abs() < 2e-4,
                    "{name}/{cname} row {r}: act probs {p:?} vs {want:?}"
                );
            }
        }
        let mut answers =
            laya_core::decode::decode_answers(&out, &batch, &act_head, &temps, &qs, 0).unwrap();
        let want = &case["result"]["answers"];
        // `scripts/ref_dump.py` fixtures from a laya older than 0.3.21 have no
        // `answer_confidence`. They need the Python package to regenerate, so compare without
        // the key instead when the fixture lacks it.
        let fixture_has_key = want
            .as_object()
            .is_some_and(|m| m.values().any(|a| a.get("answer_confidence").is_some()));
        if !fixture_has_key {
            eprintln!("{name}/{cname}: fixture predates answer_confidence; comparing without it");
            for a in answers.as_object_mut().unwrap().values_mut() {
                a.as_object_mut().unwrap().remove("answer_confidence");
            }
        }
        assert_eq!(
            &answers, want,
            "{name}/{cname}: decoded answers differ\n got: {answers}\nwant: {want}"
        );
    }
}

#[test]
#[ignore = "needs runtime/tests/fixtures/laya.json from scripts/ref_dump.py and the checkpoint in the HF cache"]
fn english_checkpoint_parity() {
    check_checkpoint("laya", None);
}

#[test]
#[ignore = "needs runtime/tests/fixtures/multilingual.json from scripts/ref_dump.py and the checkpoint in the HF cache"]
fn multilingual_checkpoint_parity() {
    check_checkpoint("multilingual", Some("multilingual"));
}
