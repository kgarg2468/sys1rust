//! Checkpoint configuration: `rl_agent_config.json` and `encoder/config.json`.

use crate::{Error, Result};
use serde::Deserialize;
use serde_json::{Map, Value};
use std::path::Path;

/// `rl_agent_config.json` as shipped with a Laya checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct AgentConfig {
    pub encoder: String,
    #[serde(default = "d_head_layers")]
    pub head_layers: usize,
    #[serde(default = "d_max_len")]
    pub max_len: usize,
    #[serde(default = "d_head_max_len")]
    pub head_max_len: usize,
    #[serde(default)]
    pub act_costs: Map<String, Value>,
    #[serde(default = "d_temperature")]
    pub temperature: Vec<Value>,
    #[serde(default)]
    pub temperature_by_options: Map<String, Value>,
    #[serde(default)]
    pub amp_dtype: Option<String>,
    #[serde(default)]
    pub model_name: Option<String>,
}

fn d_head_layers() -> usize {
    2
}
fn d_max_len() -> usize {
    512
}
fn d_head_max_len() -> usize {
    192
}
fn d_temperature() -> Vec<Value> {
    vec![Value::from(1.0), Value::from(1.0), Value::from(1.0)]
}

impl AgentConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let p = dir.join("rl_agent_config.json");
        if !p.exists() {
            return Err(Error::Config(format!(
                "Incompatible model: {} does not contain 'rl_agent_config.json'. That file ships with the weights of a Laya checkpoint.",
                dir.display()
            )));
        }
        let cfg: AgentConfig = serde_json::from_slice(&std::fs::read(p)?)?;
        Ok(cfg)
    }

    /// Number of action classes: `len(act_costs) + 1`.
    pub fn n_act(&self) -> usize {
        self.act_costs.len() + 1
    }
}

/// The ModernBERT encoder architecture (`encoder/config.json`), normalized across the
/// transformers 4.x (`global_rope_theta`) and 5.x (`rope_parameters`) config layouts.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub norm_eps: f64,
    pub global_attn_every_n_layers: usize,
    /// Sliding window width in tokens (a query attends keys within `local_attention / 2`).
    pub local_attention: usize,
    pub global_rope_theta: f64,
    pub local_rope_theta: f64,
    pub pad_token_id: u32,
    pub max_position_embeddings: usize,
    /// Per layer: `true` when the layer uses sliding-window attention.
    pub layer_is_local: Vec<bool>,
}

impl EncoderConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let p = dir.join("encoder").join("config.json");
        if !p.exists() {
            return Err(Error::Config(format!(
                "missing encoder config {}",
                p.display()
            )));
        }
        let v: Value = serde_json::from_slice(&std::fs::read(p)?)?;
        Self::from_value(&v)
    }

    pub fn from_value(v: &Value) -> Result<Self> {
        let get_u = |k: &str| -> Result<usize> {
            v.get(k)
                .and_then(Value::as_u64)
                .map(|x| x as usize)
                .ok_or_else(|| Error::Config(format!("encoder config: missing integer '{k}'")))
        };
        let get_f = |k: &str, d: f64| v.get(k).and_then(Value::as_f64).unwrap_or(d);
        if v.get("model_type").and_then(Value::as_str) != Some("modernbert") {
            return Err(Error::Config(format!(
                "unsupported encoder model_type {:?}; only 'modernbert' is implemented",
                v.get("model_type")
            )));
        }
        let num_hidden_layers = get_u("num_hidden_layers")?;
        let every = get_u("global_attn_every_n_layers").unwrap_or(3);
        let (mut g_theta, mut l_theta) = (
            get_f("global_rope_theta", 160000.0),
            get_f("local_rope_theta", 10000.0),
        );
        if let Some(rp) = v.get("rope_parameters") {
            if let Some(t) = rp
                .pointer("/full_attention/rope_theta")
                .and_then(Value::as_f64)
            {
                g_theta = t;
            }
            if let Some(t) = rp
                .pointer("/sliding_attention/rope_theta")
                .and_then(Value::as_f64)
            {
                l_theta = t;
            }
        }
        let layer_is_local: Vec<bool> = match v.get("layer_types").and_then(Value::as_array) {
            Some(types) if types.len() == num_hidden_layers => types
                .iter()
                .map(|t| t.as_str() == Some("sliding_attention"))
                .collect(),
            _ => (0..num_hidden_layers).map(|i| i % every != 0).collect(),
        };
        Ok(Self {
            vocab_size: get_u("vocab_size")?,
            hidden_size: get_u("hidden_size")?,
            num_hidden_layers,
            num_attention_heads: get_u("num_attention_heads")?,
            intermediate_size: get_u("intermediate_size")?,
            norm_eps: get_f("norm_eps", get_f("layer_norm_eps", 1e-5)),
            global_attn_every_n_layers: every,
            local_attention: get_u("local_attention").unwrap_or(128),
            global_rope_theta: g_theta,
            local_rope_theta: l_theta,
            pad_token_id: v.get("pad_token_id").and_then(Value::as_u64).unwrap_or(0) as u32,
            max_position_embeddings: get_u("max_position_embeddings").unwrap_or(8192),
            layer_is_local,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Everything a backend needs to build the full `DecisionModel`.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub agent: AgentConfig,
    pub encoder: EncoderConfig,
}

impl ModelConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        Ok(Self {
            agent: AgentConfig::load(dir)?,
            encoder: EncoderConfig::load(dir)?,
        })
    }
    pub fn hidden_size(&self) -> usize {
        self.encoder.hidden_size
    }
    /// Heads of the decision-head transformer layers: `max(1, d // 64)`.
    pub fn head_nheads(&self) -> usize {
        (self.encoder.hidden_size / 64).max(1)
    }
    pub fn head_ffn(&self) -> usize {
        4 * self.encoder.hidden_size
    }
    pub fn n_act(&self) -> usize {
        self.agent.n_act()
    }
}
