//! Memory-mapped `model.safetensors` access shared by every backend.

use crate::{Error, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use std::path::Path;

pub struct Weights {
    mmap: Mmap,
    path: std::path::PathBuf,
}

impl std::fmt::Debug for Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Weights({})", self.path.display())
    }
}

impl Weights {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("model.safetensors");
        if !path.exists() {
            return Err(Error::Weights(format!(
                "'model.safetensors' not found in {}",
                model_dir.display()
            )));
        }
        let file = std::fs::File::open(&path)?;
        // SAFETY: the checkpoint file is treated as read-only for the life of the mapping.
        let mmap = unsafe { Mmap::map(&file)? };
        let w = Self { mmap, path };
        w.view()?; // validate the header eagerly
        Ok(w)
    }

    /// Parse the safetensors header. Cheap; call per use rather than caching a self-borrow.
    pub fn view(&self) -> Result<SafeTensors<'_>> {
        Ok(SafeTensors::deserialize(&self.mmap)?)
    }

    pub fn names(&self) -> Result<Vec<String>> {
        Ok(self
            .view()?
            .names()
            .into_iter()
            .map(|s| s.to_string())
            .collect())
    }

    pub fn tensor_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let st = self.view()?;
        let t = st
            .tensor(name)
            .map_err(|e| Error::Weights(format!("{name}: {e}")))?;
        Ok((t.shape().to_vec(), to_f32(&t)?))
    }

    /// Verify the checkpoint carries the parameter families the decision model needs.
    pub fn verify(&self) -> Result<()> {
        let names = self.names()?;
        for prefix in ["encoder.", "type_emb.", "scorer.", "act_head."] {
            if !names.iter().any(|n| n.starts_with(prefix)) {
                return Err(Error::Weights(format!(
                    "checkpoint is missing '{prefix}' parameters; expected an RL Agent decision model"
                )));
            }
        }
        Ok(())
    }
}

/// Convert any float tensor view to f32.
pub fn to_f32(t: &TensorView<'_>) -> Result<Vec<f32>> {
    let data = t.data();
    Ok(match t.dtype() {
        Dtype::F32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::F16 => data
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::BF16 => data
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::F64 => data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        other => return Err(Error::Weights(format!("unsupported dtype {other:?}"))),
    })
}

/// Convert any float tensor view to f16.
pub fn to_f16(t: &TensorView<'_>) -> Result<Vec<f16>> {
    let data = t.data();
    Ok(match t.dtype() {
        Dtype::F16 => data
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]))
            .collect(),
        _ => to_f32(t)?.into_iter().map(f16::from_f32).collect(),
    })
}
