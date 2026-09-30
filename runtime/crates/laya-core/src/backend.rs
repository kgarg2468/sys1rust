//! The contract between the runtime and a model implementation.
//!
//! Changed in sys1rust from laya-r-mlx 914c9a7: `BackendOptions::tuning`.
//!
//! A backend owns the encoder, the decision-head transformer layers, the type embedding and
//! the scorer. It receives a collated batch and returns the masked scorer logits plus the
//! pooled `[CLS]` hidden state of the head output; the action head runs on the CPU in
//! [`crate::decode`].

use crate::Result;

/// Which compute device a backend should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Auto,
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Default)]
pub struct BackendOptions {
    pub device: Device,
    /// Run the transformer in f32 instead of the checkpoint's f16 (slower, closer to the
    /// Python CPU/MPS numerics).
    pub f32: bool,
    /// Backend-specific settings as a comma list of `key=value` (laya-mlx: see its `Knobs`).
    /// `None` falls back to the `SYS1_MLX` environment variable.
    pub tuning: Option<String>,
}

/// A padded batch, row-major. Row `i` covers `input_ids[i*len .. (i+1)*len]`.
#[derive(Debug, Clone)]
pub struct Batch {
    pub n: usize,
    pub len: usize,
    pub kmax: usize,
    pub input_ids: Vec<u32>,
    /// 1 for real tokens, 0 for padding (`n * len`).
    pub attention_mask: Vec<u32>,
    /// Unpadded length of each row.
    pub seq_lens: Vec<usize>,
    /// Marker token positions (`n * kmax`, zero-filled past `marker_count`).
    pub marker_pos: Vec<u32>,
    /// Valid markers per row.
    pub marker_count: Vec<usize>,
    /// Question type index per row (0 choice, 1 score, 2 noul).
    pub qtype: Vec<u32>,
}

impl Batch {
    pub fn marker_mask(&self, row: usize, k: usize) -> bool {
        k < self.marker_count[row]
    }
    pub fn total_tokens(&self) -> usize {
        self.seq_lens.iter().sum()
    }
}

/// Backend output for one batch.
#[derive(Debug, Clone)]
pub struct BackendOutput {
    /// `n * kmax` scorer logits, `-1e4` where the marker mask is false.
    pub logits: Vec<f32>,
    /// `n * hidden` head-output hidden state at position 0, in f32.
    pub pooled: Vec<f32>,
}

pub trait Backend: Send + Sync {
    /// Human-readable backend/device description, e.g. `mlx(gpu,f16)`.
    fn name(&self) -> String;
    fn forward(&self, batch: &Batch) -> Result<BackendOutput>;
    /// Sequence length to pad a batch of `rows` rows whose longest row is `len` tokens to.
    /// Backends that keep a fixed set of shapes round `len` up to a bucket.
    fn padded_len(&self, len: usize, _rows: usize) -> usize {
        len
    }
    /// Hint that the backend may want to warm up compiled kernels for a shape.
    fn warmup(&self) -> Result<()> {
        Ok(())
    }
    /// Debug hook: the encoder's `last_hidden_state` for a batch (`n * len * hidden`, f32),
    /// used by the parity harness against `encoder_hidden_item0` fixtures.
    fn encoder_hidden(&self, _batch: &Batch) -> Result<Option<Vec<f32>>> {
        Ok(None)
    }
}
