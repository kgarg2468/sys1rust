//! Backend-independent runtime for Laya System 1 decision models.
//!
//! The crate reproduces the Python `laya` package's pipeline step for step:
//! question validation and rendering, `json.dumps`-compatible state serialization,
//! HF-tokenizer tokenization, the `[CLS] <type> question [SEP] [MASK] opt ... [SEP] state [SEP]`
//! sequence layout, batch collation, temperature-scaled decoding and the action head.
//! Only the transformer forward (encoder + decision head transformer + scorer) is delegated
//! to a [`Backend`] implementation (MLX, candle, ...).

pub mod agent;
pub mod backend;
pub mod config;
pub mod decode;
pub mod error;
pub mod pyjson;
pub mod question;
pub mod resolve;
pub mod sequence;
pub mod testing;
pub mod tokenizer;
pub mod weights;

pub use agent::Agent;
pub use backend::{Backend, BackendOptions, BackendOutput, Batch, Device};
pub use config::{AgentConfig, EncoderConfig, ModelConfig};
pub use error::Error;
pub use question::{parse_questions, QType, Question};
pub use tokenizer::LayaTokenizer;
pub use weights::Weights;

pub type Result<T> = std::result::Result<T, Error>;

/// Value the Python package reports in `result["model"]`.
pub const MODEL_NAME: &str = "laya-rl-agent";
