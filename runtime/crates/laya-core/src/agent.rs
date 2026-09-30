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

/// Rows (state and question pairs) per forward pass when `predict_batch` gets no batch size.
/// 128 rows at `max_len` 512 keep the fp16 attention scores of a 12-head encoder under 1 GiB.
/// sys1d never reaches this: it passes one state per request, bounded by its request limits.
pub const DEFAULT_MAX_ROWS: usize = 128;

/// States per forward pass for `n_questions` questions each, staying under [`DEFAULT_MAX_ROWS`].
pub fn default_chunk(n_questions: usize) -> usize {
    (DEFAULT_MAX_ROWS / n_questions.max(1)).max(1)
}

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
        let backend = make_backend(&weights, &cfg, opts)?;
        let mut agent = Self::from_parts(cfg, tokenizer, act_head, backend)?;
        agent.model_dir = model_dir.to_path_buf();
        Ok(agent)
    }

    /// Assemble an agent from already loaded parts. The action head must read the encoder
    /// hidden size plus its 4 scalar features, or the rows of the backend's pooled output
    /// would be sliced at the wrong width, and it must have `len(act_costs) + 1` output
    /// classes, the shape upstream builds it with and loads strictly.
    pub fn from_parts(
        cfg: ModelConfig,
        tokenizer: LayaTokenizer,
        act_head: ActHead,
        backend: Box<dyn Backend>,
    ) -> Result<Self> {
        act_head.check_pooled_width(cfg.hidden_size())?;
        act_head.check_n_act(cfg.n_act())?;
        let temperatures = Temperatures::from_config(&cfg.agent);
        Ok(Self {
            model_dir: PathBuf::new(),
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

    /// `Agent.predict_batch`: the same questions over many states, packed into shared forward
    /// passes of `batch_size` states each, or [`default_chunk`] states when it is `None` or 0.
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
        // Without an explicit batch size, cap the rows per forward pass so a large call does not
        // allocate ids, masks and activations for every state at once. One state is never split.
        let chunk = match batch_size {
            Some(b) if b > 0 => b,
            _ => default_chunk(qs.len()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AgentConfig, EncoderConfig};
    use crate::testing::SyntheticBackend;
    use serde_json::json;

    const HIDDEN: usize = 8;

    /// A whitespace word-level tokenizer over a few words; anything else is `[UNK]`.
    fn test_tokenizer() -> LayaTokenizer {
        let words = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "question", ":", "choice", "score",
            "noul", "yes", "no", "level", "0", "1", "2", "3", "refund", "billing", "the", "a",
            "Which", "team", "?", "How", "urgent", "Is", "money", "involved",
        ];
        let vocab: serde_json::Map<String, Value> = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.to_string(), Value::from(i as u32)))
            .collect();
        let spec = json!({
            "version": "1.0",
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
            "pre_tokenizer": {"type": "Whitespace"},
        });
        let tok = tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&spec).unwrap()).unwrap();
        LayaTokenizer::from_tokenizer(tok, &Value::Null).unwrap()
    }

    /// An action head with small, varied weights so `act_probability` depends on the row.
    fn test_act_head(d_in: usize) -> ActHead {
        test_act_head_with_classes(d_in, 2)
    }

    fn test_act_head_with_classes(d_in: usize, n_act: usize) -> ActHead {
        let h = 6;
        let w0 = (0..h * d_in)
            .map(|i| ((i * 7 % 11) as f32 - 5.0) / 10.0)
            .collect();
        let b0 = (0..h).map(|i| i as f32 / 10.0).collect();
        let w2 = (0..n_act * h)
            .map(|i| ((i * 5 % 7) as f32 - 3.0) / 10.0)
            .collect();
        let b2 = (0..n_act).map(|i| 0.1 - 0.2 * i as f32).collect();
        ActHead::from_tensors(vec![h, d_in], w0, b0, vec![n_act, h], w2, b2).unwrap()
    }

    fn test_config() -> ModelConfig {
        let encoder = EncoderConfig::from_value(&json!({
            "model_type": "modernbert", "vocab_size": 32, "hidden_size": HIDDEN,
            "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 16,
        }))
        .unwrap();
        let agent: AgentConfig = serde_json::from_value(json!({
            "encoder": "test", "max_len": 64, "head_max_len": 32,
            "act_costs": {"escalate": 1.0}, "temperature": [1.2, 0.8, 1.0],
        }))
        .unwrap();
        ModelConfig { agent, encoder }
    }

    /// An agent over [`SyntheticBackend`] with batches padded to a multiple of `pad_to`.
    fn test_agent(pad_to: usize) -> Agent {
        let backend = SyntheticBackend {
            hidden: HIDDEN,
            pad_to,
        };
        Agent::from_parts(
            test_config(),
            test_tokenizer(),
            test_act_head(HIDDEN + 4),
            Box::new(backend),
        )
        .unwrap()
    }

    fn questions() -> Value {
        json!({
            "route": {"type": "choice", "instructions": "Which team ?",
                      "criteria": {"billing": "refund the a", "tech": null, "other": ""}},
            "urgency": {"type": "score", "instructions": "How urgent ?",
                        "criteria": ["none", "low", "high", "now"]},
            "money": {"type": "noul", "instructions": "Is money involved ?"},
        })
    }

    /// States of different token lengths, including a dict and a list (truncated from the
    /// left), so a batch pads rows differently from a single-state call.
    fn states() -> Vec<Value> {
        vec![
            json!("refund the billing"),
            json!("a a a the the no yes level 0 1 2 3 refund billing question"),
            json!({"from": "a", "text": "the refund"}),
            json!([
                "yes",
                "no",
                "the a refund billing yes no yes no the a the a the a"
            ]),
            json!(""),
        ]
    }

    /// Collating several states into one forward, with the backend's padding, gives every
    /// state the answers it gets on its own: the same decisions, probabilities, confidences,
    /// `act_probability` and token counts.
    #[test]
    fn a_batch_answers_like_one_state_at_a_time() {
        let (questions, states) = (questions(), states());
        for pad_to in [1, 16] {
            let agent = test_agent(pad_to);
            let single: Vec<Value> = states
                .iter()
                .map(|s| agent.predict(s, &questions).unwrap())
                .collect();
            // Every state in one forward, then chunks of two and of one.
            for batch_size in [None, Some(2), Some(1)] {
                let (batched, timing) = agent
                    .predict_batch_timed(&states, &questions, batch_size)
                    .unwrap();
                assert_eq!(
                    batched, single,
                    "pad_to {pad_to}, batch_size {batch_size:?}"
                );
                assert_eq!(
                    timing.batch_len % pad_to,
                    0,
                    "padded to the backend's bucket"
                );
            }
            let (_, timing) = agent
                .predict_batch_timed(&states, &questions, None)
                .unwrap();
            assert_eq!(
                timing.batch_rows,
                states.len() * 3,
                "one forward for all rows"
            );
            // The order of states in the batch does not leak between rows.
            let reversed: Vec<Value> = states.iter().rev().cloned().collect();
            let batched = agent.predict_batch(&reversed, &questions, None).unwrap();
            assert_eq!(batched, single.iter().rev().cloned().collect::<Vec<_>>());
            // The backend is not constant: different states get different answers.
            assert_ne!(single[0]["answers"], single[1]["answers"]);
            assert_ne!(single[0]["usage"], single[1]["usage"]);
            let route = &single[0]["answers"]["route"];
            assert!(["billing", "tech", "other"].contains(&route["choice"].as_str().unwrap()));
        }
    }

    #[test]
    fn from_parts_rejects_an_action_head_of_another_width() {
        let e = Agent::from_parts(
            test_config(),
            test_tokenizer(),
            test_act_head(HIDDEN + 5),
            Box::new(SyntheticBackend {
                hidden: HIDDEN,
                pad_to: 1,
            }),
        )
        .err()
        .map(|e| e.to_string())
        .unwrap();
        assert!(
            e.contains("input width 13") && e.contains("hidden size 8"),
            "{e}"
        );
    }

    /// `test_config` lists one act cost, so the head must have two classes: a three-class
    /// head is refused at assembly, before any `act_probability` is read from it.
    #[test]
    fn from_parts_rejects_an_action_head_of_another_class_count() {
        let e = Agent::from_parts(
            test_config(),
            test_tokenizer(),
            test_act_head_with_classes(HIDDEN + 4, 3),
            Box::new(SyntheticBackend {
                hidden: HIDDEN,
                pad_to: 1,
            }),
        )
        .err()
        .map(|e| e.to_string())
        .unwrap();
        assert!(
            e.contains("3 output classes") && e.contains("lists 1 act_costs"),
            "{e}"
        );
        assert_eq!(test_config().n_act(), 2);
    }

    #[test]
    fn default_chunk_bounds_rows_and_keeps_whole_states() {
        assert_eq!(default_chunk(1), DEFAULT_MAX_ROWS);
        assert_eq!(default_chunk(3), DEFAULT_MAX_ROWS / 3);
        assert!(default_chunk(3) * 3 <= DEFAULT_MAX_ROWS);
        // More questions than the row cap still runs one state per pass.
        assert_eq!(default_chunk(DEFAULT_MAX_ROWS + 1), 1);
        assert_eq!(default_chunk(0), DEFAULT_MAX_ROWS);
    }
}
