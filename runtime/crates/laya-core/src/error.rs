#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Config(String),
    #[error("{0}")]
    Question(String),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("weights: {0}")]
    Weights(String),
    #[error("backend: {0}")]
    Backend(String),
}

impl From<safetensors::SafeTensorError> for Error {
    fn from(e: safetensors::SafeTensorError) -> Self {
        Error::Weights(e.to_string())
    }
}
