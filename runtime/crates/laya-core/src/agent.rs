//! The high-level runtime: `Agent.predict` / `Agent.predict_batch` over any [`Backend`].
//!
//! Changed in sys1rust from laya-r-mlx 914c9a7: `Timing` records the batch shape; `collate`
//! pads to the backend's `padded_len`.

use crate::backend::{Backend, BackendOptions, Batch};
use crate::config::ModelConfig;
use crate::decode::{decode_answers, ActHead, Temperatures};
use crate::question::{parse_questions, Question};
use crate::sequence::{collate_to, encode_state, EncodedItem};
use crate::tokenizer::LayaTokenizer;
use crate::weights::Weights;
use crate::{Result, MODEL_NAME};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// Builds a backend for a checkpoint. Implemented by each backend crate.
pub type BackendFactory<'a> =
    dyn FnOnce(&Weights, &ModelConfig, &BackendOptions) -> Result<Box<dyn Backend>> + 'a;

pub struct Agent {
    pub model_dir: PathBuf,
    pub cfg: ModelConfig,
    pub tokenizer: LayaTokenizer,
    pub temperatures: Temperatures,
    act_head: ActHead,
    backend: Box<dyn Backend>,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("model_dir", &self.model_dir)
            .field("backend", &self.backend.name())
            .finish()
    }
}

/// Timing of the phases of one `predict_batch` call, for benchmarking.
#[derive(Debug, Clone, Default)]
pub struct Timing {
    pub encode_us: u128,
    pub forward_us: u128,
    pub decode_us: u128,
    /// Padded sequence length and row count of the last forward.
    pub batch_len: usize,
    pub batch_rows: usize,
}

impl Agent {
    /// Load a checkpoint directory (see [`crate::resolve::resolve_model_dir`]) with a backend.
    pub fn load(
        model_dir: &Path,
        opts: &BackendOptions,
        make_backend: Box<BackendFactory<'_>>,
    ) -> Result<Self> {
        let cfg = ModelConfig::load(model_dir)?;
        let tokenizer = LayaTokenizer::load(model_dir)?;
        let weights = Weights::open(model_dir)?;
        weights.verify()?;
        let act_head = ActHead::load(&weights)?;
        let temperatures = Temperatures::from_config(&cfg.agent);
        let backend = make_backend(&weights, &cfg, opts)?;
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            cfg,
            tokenizer,
            temperatures,
            act_head,
            backend,
        })
    }

    pub fn backend_name(&self) -> String {
        self.backend.name()
    }

    pub fn backend(&self) -> &dyn Backend {
        self.backend.as_ref()
    }

    /// Tokenize one state against validated questions (exposed for parity tests).
    pub fn encode(&self, state: &Value, questions: &[Question]) -> Result<Vec<EncodedItem>> {
        encode_state(
            &self.tokenizer,
            state,
            questions,
            self.cfg.agent.max_len,
            self.cfg.agent.head_max_len,
        )
    }

    pub fn collate(&self, items: &[EncodedItem]) -> Batch {
        let len = items.iter().map(|it| it.ids.len()).max().unwrap_or(0);
        let len = self.backend.padded_len(len, items.len());
        collate_to(items, self.tokenizer.pad_id, len)
    }

    /// `Agent.predict` / `system_one`: one state, a `{qid: definition}` object.
    pub fn predict(&self, state: &Value, questions: &Value) -> Result<Value> {
        Ok(self
            .predict_batch(std::slice::from_ref(state), questions, None)?
            .remove(0))
    }

    /// `Agent.predict_batch`: the same questions over many states, packed into shared forward passes.
    pub fn predict_batch(
        &self,
        states: &[Value],
        questions: &Value,
        batch_size: Option<usize>,
    ) -> Result<Vec<Value>> {
        Ok(self.predict_batch_timed(states, questions, batch_size)?.0)
    }

    pub fn predict_batch_timed(
        &self,
        states: &[Value],
        questions: &Value,
        batch_size: Option<usize>,
    ) -> Result<(Vec<Value>, Timing)> {
        let mut timing = Timing::default();
        if states.is_empty() {
            return Ok((Vec::new(), timing));
        }
        let qs = parse_questions(questions)?;
        if qs.is_empty() {
            let empty = json!({ "model": MODEL_NAME, "answers": {}, "usage": { "input_tokens": 0, "output_tokens": 0 } });
            return Ok((vec![empty; states.len()], timing));
        }
        let chunk = match batch_size {
            Some(b) if b > 0 => b,
            _ => states.len(),
        };
        let mut results = Vec::with_capacity(states.len());
        for part in states.chunks(chunk) {
            let t0 = std::time::Instant::now();
            let mut items = Vec::with_capacity(part.len() * qs.len());
            for st in part {
                items.extend(self.encode(st, &qs)?);
            }
            let batch = self.collate(&items);
            timing.encode_us += t0.elapsed().as_micros();
            timing.batch_len = batch.len;
            timing.batch_rows = batch.n;

            let t1 = std::time::Instant::now();
            let out = self.backend.forward(&batch)?;
            timing.forward_us += t1.elapsed().as_micros();

            let t2 = std::time::Instant::now();
            let nq = qs.len();
            for (si, _) in part.iter().enumerate() {
                let offset = si * nq;
                let n_tokens: usize = batch.seq_lens[offset..offset + nq].iter().sum();
                let answers = decode_answers(
                    &out,
                    &batch,
                    &self.act_head,
                    &self.temperatures,
                    &qs,
                    offset,
                )?;
                results.push(json!({
                    "model": MODEL_NAME,
                    "answers": answers,
                    "usage": { "input_tokens": n_tokens, "output_tokens": 0 },
                }));
            }
            timing.decode_us += t2.elapsed().as_micros();
        }
        Ok((results, timing))
    }

    /// JSON-string convenience for FFI: `state` may be any JSON value (a bare string is text).
    pub fn predict_json(&self, state_json: &str, questions_json: &str) -> Result<String> {
        let state: Value = serde_json::from_str(state_json)?;
        let questions: Value = serde_json::from_str(questions_json)?;
        Ok(serde_json::to_string(&self.predict(&state, &questions)?)?)
    }
}
