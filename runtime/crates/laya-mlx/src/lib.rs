//! MLX (Apple silicon) backend for the Laya decision model.
//!
//! Implements [`laya_core::Backend`] with [`mlx_rs`]: the ModernBERT encoder, the two
//! `nn.TransformerEncoderLayer` decision-head layers and the scorer, exactly as described in
//! `docs/MODEL.md`. Weights stay resident as MLX arrays (f16 by default, f32 with
//! [`BackendOptions::f32`]); the whole forward pass is built lazily and evaluated once.
//!
//! Changed in sys1rust from laya-r-mlx 914c9a7: `Knobs` (settings from
//! `BackendOptions::tuning` or `SYS1_MLX`), the `f16gelu` fix for the f16 -> f32 promotion in
//! GELU, length buckets with warm-up at load, MLX cache and wired limits, optional boolean
//! masks, and diagnostic timing. Work-reduction knobs, off unless asked for (measured in
//! `results/SPEED.md`): `dense_upto` (the dense/chunked local-attention switch), `headprune`
//! (last head layer only on the rows the scorer reads) and `unpad` (hidden states packed to
//! the real tokens outside attention). Experiments that gained nothing (`split`, `rope1`,
//! `splitk`) were removed after commit 0495800; that commit has their code.

use laya_core::weights::{to_f16, to_f32};
use laya_core::{
    Backend, BackendOptions, BackendOutput, Batch, Device, Error, ModelConfig, Result, Weights,
};
use mlx_rs::error::Exception;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::compile::compile;
use mlx_rs::{fast, nn, ops, transforms, Array, Dtype, Stream};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::sync::Mutex;

/// Finite "minus infinity" for additive attention masks (safe in f16, no NaN rows).
const MASK_NEG: f32 = -1e4;
/// Logit value reported for masked marker slots.
const LOGIT_MASKED: f32 = -1e4;

/// Backend settings, read once at load from `BackendOptions::tuning` or else `SYS1_MLX` (comma
/// list, e.g. `f16gelu,mask=bool`). Defaults reproduce laya-r-mlx 914c9a7.
#[derive(Debug, Clone)]
struct Knobs {
    /// `compiled`: split + GELU + gate compiled per shape (default). `plain`: no compile.
    /// `shapeless`: split outside, GELU * gate compiled once for all shapes.
    geglu: String,
    /// Fuse residual adds into the gemm with `addmm` (default) or use matmul + add.
    addmm: bool,
    /// Boolean attention masks (as Python laya-mlx) instead of additive f16 masks.
    bool_mask: bool,
    /// Chunked sliding-window attention for long inputs (default) or dense masks always.
    windowed: bool,
    /// `mlx::clear_cache` after every forward.
    clear: bool,
    /// Print graph-build and eval times of every forward to stderr.
    trace: bool,
    /// Evaluate after every encoder layer and print per-layer times (slows the forward).
    layers: bool,
    /// Evaluate after every op of the encoder layers and print summed times per op.
    ops: bool,
    /// Linear weights: `view` keeps the host-loaded `[out, in]` array behind a transposed view
    /// (default); `gpu` copies it on the GPU first; `t` stores a GPU-written contiguous `[in, out]`.
    wcopy: String,
    /// Keep GELU in the compute dtype (fixes the upstream f16 -> f32 promotion).
    f16gelu: bool,
    /// Pad the sequence length up to the first of these that fits (`buckets=128:256:512`).
    buckets: Vec<usize>,
    /// Otherwise (or past the last bucket) pad the length to a multiple of this.
    pad: usize,
    /// Row counts to run one forward for at every bucket length at load (`warm=1:4:10`).
    warm: Vec<usize>,
    /// MLX buffer-cache limit in MiB. MLX's default lets freed buffers pile up to about the
    /// size of RAM when request shapes vary, which pushes the machine into swap.
    cache_mb: Option<usize>,
    /// MLX wired-memory limit in MiB (keeps the weights resident).
    wired_mb: Option<usize>,
    /// Local attention runs dense (full `len x len` masks) while the padded length is at most
    /// this, chunked above it (`dense_upto=512`). `None` is the upstream `4 * window`.
    /// `windowed=0` still means never chunk.
    dense_upto: Option<usize>,
    /// Last head layer: queries, out_proj, norm2 and the FFN only for the rows the scorer
    /// reads (position 0 and every marker slot); keys and values still from every token.
    /// The other positions of that layer are never computed, so its output is a
    /// [`ScorerRows`] and nothing else: a consumer that needs per-token head output has to
    /// run with `headprune=0`.
    headprune: bool,
    /// Keep hidden states packed as `[T, d]` (real tokens only) through embeddings, norms,
    /// linears and GeGLU; expand to `[n, len, ...]` only around attention.
    unpad: bool,
}

impl Knobs {
    fn from_spec(tuning: Option<&str>) -> Self {
        let mut k = Knobs {
            geglu: "compiled".into(),
            addmm: true,
            bool_mask: false,
            windowed: true,
            clear: false,
            trace: false,
            layers: false,
            ops: false,
            wcopy: "view".into(),
            f16gelu: false,
            buckets: Vec::new(),
            pad: 1,
            warm: Vec::new(),
            cache_mb: None,
            wired_mb: None,
            dense_upto: None,
            headprune: false,
            unpad: false,
        };
        let list = |v: &str| -> Vec<usize> { v.split(':').filter_map(|x| x.parse().ok()).collect() };
        let spec = match tuning {
            Some(t) => t.to_string(),
            None => std::env::var("SYS1_MLX").unwrap_or_default(),
        };
        for kv in spec.split(',').filter(|s| !s.is_empty()) {
            let (key, val) = kv.split_once('=').unwrap_or((kv, "1"));
            let on = val != "0";
            match key {
                "geglu" => k.geglu = val.to_string(),
                "addmm" => k.addmm = on,
                "mask" => k.bool_mask = val == "bool",
                "windowed" => k.windowed = on,
                "clear" => k.clear = on,
                "trace" => k.trace = on,
                "layers" => k.layers = on,
                "ops" => k.ops = on,
                "wcopy" => k.wcopy = val.to_string(),
                "f16gelu" => k.f16gelu = on,
                "buckets" => {
                    k.buckets = list(val);
                    k.buckets.sort_unstable();
                }
                "pad" => k.pad = val.parse().unwrap_or(1).max(1),
                "warm" => k.warm = list(val),
                "cache" => k.cache_mb = val.parse().ok(),
                "wired" => k.wired_mb = val.parse().ok(),
                "dense_upto" => k.dense_upto = val.parse().ok(),
                "headprune" => k.headprune = on,
                "unpad" => k.unpad = on,
                _ => eprintln!("SYS1_MLX: unknown knob {key}"),
            }
        }
        k
    }
}

/// Convert an MLX exception into a `laya_core::Error::Backend`.
trait Lx<T> {
    fn lx(self) -> Result<T>;
}
impl<T> Lx<T> for std::result::Result<T, mlx_rs::error::Exception> {
    fn lx(self) -> Result<T> {
        self.map_err(|e| Error::Backend(e.to_string()))
    }
}

/// A compiled MLX function `&[Array] -> Vec<Array>`.
type CompiledFn = Box<dyn for<'a> FnMut(&'a [Array]) -> std::result::Result<Vec<Array>, Exception>>;

/// `gelu_erf(input) * gate` over `x = [input | gate]` (torch `chunk(2, dim=-1)` order), fused
/// into a single Metal kernel by `mlx_rs::transforms::compile`.
///
/// Unfused, the split + five elementwise passes cost ~0.8 ms per layer at 4x512 tokens;
/// compiled they cost ~0.08 ms. The compiled state is not thread-safe, hence the mutex.
struct GeGlu {
    mode: String,
    keep: bool,
    f: Mutex<CompiledFn>,
}

// SAFETY: the compiled closure owns no thread-affine resources; calls are serialised by the
// mutex and MLX's scheduler is thread-safe.
unsafe impl Send for GeGlu {}
unsafe impl Sync for GeGlu {}

impl GeGlu {
    fn new(mode: &str, keep: bool) -> Self {
        let f: CompiledFn = if mode == "shapeless" {
            Box::new(compile(
                move |a: &[Array]| -> Vec<Array> {
                    vec![ops::multiply(gelu_erf_as(&a[0], keep).expect("gelu"), &a[1]).expect("geglu gate")]
                },
                true,
            ))
        } else {
            Box::new(compile(
                move |a: &[Array]| -> Vec<Array> {
                    let ig = a[0].split_equal(2, -1).expect("geglu split");
                    vec![ops::multiply(gelu_erf_as(&ig[0], keep).expect("gelu"), &ig[1]).expect("geglu gate")]
                },
                // Not shapeless: `split` needs concrete shapes; MLX caches one trace per input shape.
                false,
            ))
        };
        Self {
            mode: mode.to_string(),
            keep,
            f: Mutex::new(f),
        }
    }

    fn apply(&self, x: &Array) -> Result<Array> {
        if self.mode == "plain" {
            let ig = x.split_equal(2, -1).lx()?;
            return ops::multiply(gelu_erf_as(&ig[0], self.keep).lx()?, &ig[1]).lx();
        }
        // A panic inside the compiled closure (mlx-rs re-raises it after MLX returns) would
        // poison the lock; recover the guard so one failed request cannot fail every later one.
        let mut f = self.f.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut out = if self.mode == "shapeless" {
            let ig = x.split_equal(2, -1).lx()?;
            f(ig.as_slice()).lx()?
        } else {
            f(std::slice::from_ref(x)).lx()?
        };
        Ok(out.remove(0))
    }
}

/// Exact GELU, `0.5 * x * (1 + erf(x / sqrt 2))` (torch `nn.GELU()` default).
///
/// Upstream (laya-r-mlx 914c9a7) multiplies by f32 scalar arrays, which promotes f16 input to
/// f32: from the first MLP on, the residual stream and every gemm run in f32 with the f16
/// weights cast on each forward. `keep_dtype` casts the scalars to the input dtype instead.
fn gelu_erf_as(x: &Array, keep_dtype: bool) -> std::result::Result<Array, Exception> {
    let c = |v: f32| -> std::result::Result<Array, Exception> {
        let a = Array::from_f32(v);
        if keep_dtype {
            a.as_dtype(x.dtype())
        } else {
            Ok(a)
        }
    };
    let e = ops::erf(ops::multiply(x, c(std::f32::consts::FRAC_1_SQRT_2)?)?)?;
    ops::multiply(ops::multiply(x, c(0.5)?)?, ops::add(&e, c(1.0)?)?)
}

/// Chunk-local band mask `[1, 1, 1, S, 3S]`: query `i` of a chunk may see gathered key slot
/// `j` (global offset `j - S` relative to the chunk start) iff `|S + i - j| <= S`.
fn band_mask(s: usize, dtype: Dtype) -> Result<Array> {
    let ks = 3 * s;
    let mut vals = vec![MASK_NEG; s * ks];
    for i in 0..s {
        for j in i..=i + 2 * s {
            vals[i * ks + j] = 0.0;
        }
    }
    let m = Array::from_slice(&vals, &[1, 1, 1, s as i32, ks as i32])
        .as_dtype(dtype)
        .lx()?;
    m.eval().lx()?;
    Ok(m)
}

/// Cast to f32 and force a row-contiguous layout so the result can be read from the host.
fn to_f32_contiguous(a: &Array) -> Result<Array> {
    a.as_dtype(Dtype::Float32).lx()?.contiguous().lx()
}

/// Copy an evaluated, row-contiguous f32 array to the host.
fn host_f32(a: &Array) -> Result<Vec<f32>> {
    a.try_as_slice::<f32>()
        .map(|s| s.to_vec())
        .map_err(|e| Error::Backend(format!("host copy: {e}")))
}

/// Build the MLX backend from a checkpoint. See [`laya_core::Backend`].
pub fn make_backend(
    w: &Weights,
    cfg: &ModelConfig,
    opts: &BackendOptions,
) -> Result<Box<dyn Backend>> {
    Ok(Box::new(MlxBackend::new(w, cfg, opts)?))
}

/// Factory for [`laya_core::testing::run_parity`].
pub fn factory() -> laya_core::testing::Factory {
    Box::new(make_backend)
}

/// `y = x @ W^T (+ b)`; `wt` is the transposed weight view `[in, out]`.
struct Linear {
    wt: Array,
    b: Option<Array>,
}

impl Linear {
    fn apply(&self, x: &Array) -> Result<Array> {
        match &self.b {
            Some(b) => ops::addmm(b, x, &self.wt, None, None).lx(),
            None => ops::matmul(x, &self.wt).lx(),
        }
    }

    /// `residual + x @ W^T (+ b)`, with the residual add fused into the gemm epilogue.
    fn apply_add(&self, x: &Array, residual: &Array) -> Result<Array> {
        let y = ops::addmm(residual, x, &self.wt, None, None).lx()?;
        match &self.b {
            Some(b) => ops::add(&y, b).lx(),
            None => Ok(y),
        }
    }
}

/// LayerNorm over the last axis, optionally with bias.
struct Norm {
    w: Array,
    b: Option<Array>,
    eps: f32,
}

impl Norm {
    fn apply(&self, x: &Array) -> Result<Array> {
        fast::layer_norm(x, &self.w, self.b.as_ref(), self.eps).lx()
    }
}

struct EncoderLayer {
    /// `None` for layer 0 (identity).
    attn_norm: Option<Norm>,
    wqkv: Linear,
    wo: Linear,
    mlp_norm: Norm,
    wi: Linear,
    wo2: Linear,
    rope_theta: f32,
    local: bool,
}

struct HeadLayer {
    norm1: Norm,
    in_proj: Linear,
    /// `in_proj` cut into its q rows and its k|v rows, for the pruned last layer (`headprune`).
    split_proj: Option<(Linear, Linear)>,
    out_proj: Linear,
    norm2: Norm,
    linear1: Linear,
    linear2: Linear,
}

/// Reads tensors from the safetensors view into MLX arrays of the compute dtype.
struct Loader<'a> {
    st: SafeTensors<'a>,
    dtype: Dtype,
    wcopy: String,
}

impl Loader<'_> {
    fn get(&self, name: &str) -> Result<Array> {
        let t = self
            .st
            .tensor(name)
            .map_err(|e| Error::Weights(format!("{name}: {e}")))?;
        let shape: Vec<i32> = t.shape().iter().map(|&d| d as i32).collect();
        let arr = match self.dtype {
            Dtype::Float32 => Array::from_slice(&to_f32(&t)?, &shape),
            _ => Array::from_slice(&to_f16(&t)?, &shape),
        };
        Ok(arr)
    }

    fn linear(&self, w: &str, b: Option<&str>) -> Result<Linear> {
        let b = b.map(|n| self.get(n)).transpose()?;
        self.linear_from(self.get(w)?, b)
    }

    /// Output rows `rows` of a linear layer as a layer of their own.
    fn linear_rows(&self, w: &str, b: Option<&str>, rows: std::ops::Range<i32>) -> Result<Linear> {
        let w = self.get(w)?.index((rows.clone(), ..));
        let b = b.map(|n| self.get(n)).transpose()?.map(|b| b.index(rows));
        self.linear_from(w, b)
    }

    fn linear_from(&self, w: Array, b: Option<Array>) -> Result<Linear> {
        let wt = match self.wcopy.as_str() {
            "t" => ops::transpose(&w).lx()?.contiguous().lx()?,
            "gpu" => {
                let one = Array::from_f32(1.0).as_dtype(self.dtype).lx()?;
                ops::transpose(ops::multiply(&w, &one).lx()?).lx()?
            }
            _ => ops::transpose(&w).lx()?,
        };
        Ok(Linear { wt, b })
    }

    fn norm(&self, w: &str, b: Option<&str>, eps: f32) -> Result<Norm> {
        Ok(Norm {
            w: self.get(w)?,
            b: b.map(|n| self.get(n)).transpose()?,
            eps,
        })
    }
}

/// The resident model plus per-shape caches.
pub struct MlxBackend {
    cpu: bool,
    dtype: Dtype,
    hidden: usize,
    n_heads: usize,
    head_dim: usize,
    head_nheads: usize,
    window: usize,
    tok_emb: Array,
    emb_norm: Norm,
    layers: Vec<EncoderLayer>,
    final_norm: Norm,
    type_emb: Array,
    head: Vec<HeadLayer>,
    scorer_norm: Norm,
    scorer1: Linear,
    scorer3: Linear,
    /// Chunk-local band mask `[1, 1, 1, S, 3S]` for the windowed attention path.
    band: Array,
    caches: Mutex<Caches>,
    geglu: GeGlu,
    knobs: Knobs,
}

/// Per-shape constants reused across forwards.
///
/// The dense window masks are one array each, for the longest length seen so far; shorter
/// lengths take a view of it (see [`MlxBackend::window_mask`]). One mask per distinct length
/// would pile up: with `dense_upto=1024` and no length buckets, a server seeing every length
/// up to 1,024 would hold about 700 MB of masks that nothing frees.
#[derive(Default)]
struct Caches {
    /// Dense sliding-window additive mask `[1, 1, L, L]` for the longest `L` so far.
    window_mask: Option<Array>,
    /// Flat key gather indices `[n * H * n_chunks * 3S]` into `[n * H * len, hd]` rows for
    /// the windowed path, keyed by `(n, len)`.
    key_idx: HashMap<(usize, usize), Array>,
    /// Boolean sliding-window mask `[1, 1, L, L]` for the longest `L` so far (`mask=bool`).
    window_bool: Option<Array>,
}

/// How local (sliding-window) layers attend for the current batch shape.
enum LocalAttn {
    /// Full `len x len` attention with an additive pad + window mask `[n, 1, len, len]`.
    Dense(Array),
    /// Chunked attention: every 64-query chunk attends to the 192 keys that can fall inside
    /// its window (previous, own and next chunk); see [`MlxBackend::windowed_attention`].
    Windowed(Windowed),
}

/// Constants of the chunked sliding-window attention for one batch shape.
struct Windowed {
    /// Flat key gather indices `[n * H * n_chunks * 3S]` into `[n * H * len, hd]` rows.
    key_idx: Array,
    /// Additive pad + band mask `[n * H, n_chunks, S, 3S]`.
    mask: Array,
    n_chunks: i32,
}

/// Masks shared by every encoder layer of one forward pass.
struct AttnCtx {
    /// Additive key-padding mask `[n, 1, 1, len]`.
    pad: Array,
    local: LocalAttn,
}

/// Decision-head output as the scorer reads it.
///
/// Every consumer has to match all three variants: with `headprune` the last head layer is
/// computed at the scorer's rows only, so `Rows` holds no per-token output and cannot be
/// turned into one. Anything that needs every token's head output (per-token embeddings,
/// say) must run with `headprune=0` and take `Full` or `Packed`.
enum HeadOut {
    /// Every token, `[n, len, d]`.
    Full(Array),
    /// Every real token, `[T, d]` in [`Packing`] order (`unpad`).
    Packed(Array),
    /// The scorer's rows only (`headprune`).
    Rows(ScorerRows),
}

/// The last head layer at the rows the scorer reads and nowhere else: position 0 followed by
/// the `kmax` marker slots of each row, `[n, 1 + kmax, d]`. Padded marker slots hold position
/// 0's output (their logits are masked afterwards, as in the full layer). The array is private
/// so that it cannot be mistaken for a `[n, len, d]` per-token output.
struct ScorerRows(Array);

impl ScorerRows {
    /// Position 0 of every row, `[n, d]`.
    fn pooled(&self) -> Array {
        self.0.index((.., 0, ..))
    }

    /// The marker slots of every row, `[n * kmax, d]`.
    fn markers(&self, n: i32, kmax: i32, d: i32) -> Result<Array> {
        self.0.index((.., 1.., ..)).reshape(&[n * kmax, d]).lx()
    }
}

/// Token packing for `unpad`: hidden states live as `[T, d]` over the real tokens only and
/// are expanded to the padded `[n, len, ...]` layout just for attention. Every row has at
/// least one real token (its `[CLS]`), which the padding positions borrow.
struct Packing {
    /// `[T]` padded positions `r * len + pos` of the packed tokens, row-major.
    pack: Array,
    /// `[n * len]` packed index of every padded position. Padding points at its row's token 0,
    /// so expanded values stay finite; they are masked as keys and dropped as queries.
    unpack: Array,
    /// `[T]` row of every packed token.
    row: Array,
    /// Packed index of position 0 of every row.
    offsets: Vec<u32>,
}

impl Packing {
    fn new(batch: &Batch) -> Self {
        let mut pack: Vec<u32> = Vec::with_capacity(batch.total_tokens());
        let mut unpack: Vec<u32> = Vec::with_capacity(batch.n * batch.len);
        let mut row: Vec<u32> = Vec::with_capacity(batch.total_tokens());
        let mut offsets = Vec::with_capacity(batch.n);
        for r in 0..batch.n {
            let first = pack.len() as u32;
            offsets.push(first);
            for pos in 0..batch.len {
                let flat = r * batch.len + pos;
                if batch.attention_mask[flat] == 1 {
                    unpack.push(pack.len() as u32);
                    pack.push(flat as u32);
                    row.push(r as u32);
                } else {
                    unpack.push(first);
                }
            }
        }
        Self {
            pack: Array::from_slice(&pack, &[pack.len() as i32]),
            unpack: Array::from_slice(&unpack, &[unpack.len() as i32]),
            row: Array::from_slice(&row, &[row.len() as i32]),
            offsets,
        }
    }

    /// `[T, c] -> [n, len, c]`.
    fn expand(&self, x: &Array, n: i32, len: i32) -> Result<Array> {
        x.take_axis(&self.unpack, 0).lx()?.reshape(&[n, len, x.dim(-1)]).lx()
    }

    /// `[n, len, c] -> [T, c]`.
    fn compact(&self, x: &Array, n: i32, len: i32) -> Result<Array> {
        x.reshape(&[n * len, x.dim(-1)]).lx()?.take_axis(&self.pack, 0).lx()
    }
}

/// Encoder output for one batch.
struct Encoded {
    /// `last_hidden_state`: `[n, len, d]`, or `[T, d]` when `packing` is set.
    h: Array,
    /// Additive key-padding mask `[n, 1, 1, len]`.
    pad: Array,
    packing: Option<Packing>,
}

// SAFETY: `mlx_rs::Array` is a reference-counted handle to immutable, already-evaluated
// data; MLX's scheduler is thread-safe and every forward selects its own stream. The
// mask cache is behind a `Mutex`.
unsafe impl Sync for MlxBackend {}

impl MlxBackend {
    fn new(w: &Weights, cfg: &ModelConfig, opts: &BackendOptions) -> Result<Self> {
        let cpu = matches!(opts.device, Device::Cpu);
        let dtype = if opts.f32 {
            Dtype::Float32
        } else {
            Dtype::Float16
        };
        let enc = &cfg.encoder;
        let eps = enc.norm_eps as f32;
        let knobs = Knobs::from_spec(opts.tuning.as_deref());
        let loader = Loader {
            st: w.view()?,
            dtype,
            wcopy: knobs.wcopy.clone(),
        };

        if std::env::var_os("SYS1_MLX").is_some() {
            eprintln!("laya-mlx knobs: {knobs:?}");
        }
        if let Some(mb) = knobs.cache_mb {
            mlx_rs::memory::set_cache_limit(mb << 20).lx()?;
        }
        if let Some(mb) = knobs.wired_mb {
            mlx_rs::memory::set_wired_limit(mb << 20).lx()?;
        }
        let stream = if cpu { Stream::cpu() } else { Stream::gpu() };
        mlx_rs::with_stream(&stream, || -> Result<Self> {
            let mut layers = Vec::with_capacity(enc.num_hidden_layers);
            for i in 0..enc.num_hidden_layers {
                let p = format!("encoder.layers.{i}");
                let local = enc.layer_is_local[i];
                layers.push(EncoderLayer {
                    attn_norm: if i == 0 {
                        None
                    } else {
                        Some(loader.norm(&format!("{p}.attn_norm.weight"), None, eps)?)
                    },
                    wqkv: loader.linear(&format!("{p}.attn.Wqkv.weight"), None)?,
                    wo: loader.linear(&format!("{p}.attn.Wo.weight"), None)?,
                    mlp_norm: loader.norm(&format!("{p}.mlp_norm.weight"), None, eps)?,
                    wi: loader.linear(&format!("{p}.mlp.Wi.weight"), None)?,
                    wo2: loader.linear(&format!("{p}.mlp.Wo.weight"), None)?,
                    rope_theta: if local {
                        enc.local_rope_theta
                    } else {
                        enc.global_rope_theta
                    } as f32,
                    local,
                });
            }
            let mut head = Vec::with_capacity(cfg.agent.head_layers);
            let d = enc.hidden_size as i32;
            for j in 0..cfg.agent.head_layers {
                let p = format!("head.layers.{j}");
                let (in_w, in_b) = (
                    format!("{p}.self_attn.in_proj_weight"),
                    format!("{p}.self_attn.in_proj_bias"),
                );
                head.push(HeadLayer {
                    norm1: loader.norm(
                        &format!("{p}.norm1.weight"),
                        Some(&format!("{p}.norm1.bias")),
                        1e-5,
                    )?,
                    in_proj: loader.linear(&in_w, Some(&in_b))?,
                    split_proj: if knobs.headprune && j + 1 == cfg.agent.head_layers {
                        Some((
                            loader.linear_rows(&in_w, Some(&in_b), 0..d)?,
                            loader.linear_rows(&in_w, Some(&in_b), d..3 * d)?,
                        ))
                    } else {
                        None
                    },
                    out_proj: loader.linear(
                        &format!("{p}.self_attn.out_proj.weight"),
                        Some(&format!("{p}.self_attn.out_proj.bias")),
                    )?,
                    norm2: loader.norm(
                        &format!("{p}.norm2.weight"),
                        Some(&format!("{p}.norm2.bias")),
                        1e-5,
                    )?,
                    linear1: loader.linear(
                        &format!("{p}.linear1.weight"),
                        Some(&format!("{p}.linear1.bias")),
                    )?,
                    linear2: loader.linear(
                        &format!("{p}.linear2.weight"),
                        Some(&format!("{p}.linear2.bias")),
                    )?,
                });
            }
            let this = Self {
                cpu,
                dtype,
                hidden: enc.hidden_size,
                n_heads: enc.num_attention_heads,
                head_dim: enc.head_dim(),
                head_nheads: cfg.head_nheads(),
                window: enc.local_attention / 2,
                tok_emb: loader.get("encoder.embeddings.tok_embeddings.weight")?,
                emb_norm: loader.norm("encoder.embeddings.norm.weight", None, eps)?,
                layers,
                final_norm: loader.norm("encoder.final_norm.weight", None, eps)?,
                type_emb: loader.get("type_emb.weight")?,
                head,
                scorer_norm: loader.norm("scorer.0.weight", Some("scorer.0.bias"), 1e-5)?,
                scorer1: loader.linear("scorer.1.weight", Some("scorer.1.bias"))?,
                scorer3: loader.linear("scorer.3.weight", Some("scorer.3.bias"))?,
                band: band_mask(enc.local_attention / 2, dtype)?,
                caches: Mutex::new(Caches::default()),
                geglu: GeGlu::new(&knobs.geglu, knobs.f16gelu),
                knobs: knobs.clone(),
            };
            // Materialise every weight (and transposed view) once, up front.
            let all: Vec<&Array> = this.all_params();
            transforms::eval(all).lx()?;
            this.warm_buckets()?;
            Ok(this)
        })
    }

    /// One forward per (bucket length, warm row count), so the first real request of each
    /// shape finds its masks, gather indices, compiled traces and buffers ready.
    fn warm_buckets(&self) -> Result<()> {
        for &len in &self.knobs.buckets {
            for &n in &self.knobs.warm {
                let kmax = 2;
                let batch = Batch {
                    n,
                    len,
                    kmax,
                    input_ids: vec![0; n * len],
                    attention_mask: vec![1; n * len],
                    seq_lens: vec![len; n],
                    marker_pos: (0..n).flat_map(|_| [1u32, 2]).collect(),
                    marker_count: vec![kmax; n],
                    qtype: vec![0; n],
                };
                self.forward(&batch)?;
            }
        }
        Ok(())
    }

    fn all_params(&self) -> Vec<&Array> {
        fn push_lin<'a>(v: &mut Vec<&'a Array>, l: &'a Linear) {
            v.push(&l.wt);
            v.extend(l.b.as_ref());
        }
        fn push_norm<'a>(v: &mut Vec<&'a Array>, n: &'a Norm) {
            v.push(&n.w);
            v.extend(n.b.as_ref());
        }
        let mut v = vec![
            &self.tok_emb,
            &self.emb_norm.w,
            &self.final_norm.w,
            &self.type_emb,
        ];
        for l in &self.layers {
            if let Some(n) = &l.attn_norm {
                push_norm(&mut v, n);
            }
            push_lin(&mut v, &l.wqkv);
            push_lin(&mut v, &l.wo);
            push_norm(&mut v, &l.mlp_norm);
            push_lin(&mut v, &l.wi);
            push_lin(&mut v, &l.wo2);
        }
        for h in &self.head {
            push_norm(&mut v, &h.norm1);
            push_lin(&mut v, &h.in_proj);
            if let Some((q, kv)) = &h.split_proj {
                push_lin(&mut v, q);
                push_lin(&mut v, kv);
            }
            push_lin(&mut v, &h.out_proj);
            push_norm(&mut v, &h.norm2);
            push_lin(&mut v, &h.linear1);
            push_lin(&mut v, &h.linear2);
        }
        push_norm(&mut v, &self.scorer_norm);
        push_lin(&mut v, &self.scorer1);
        push_lin(&mut v, &self.scorer3);
        v
    }

    fn stream(&self) -> Stream {
        if self.cpu {
            Stream::cpu()
        } else {
            Stream::gpu()
        }
    }

    /// Additive key-padding mask `[n, 1, 1, len]` (0 for tokens, `MASK_NEG` for padding).
    fn pad_mask(&self, batch: &Batch) -> Result<Array> {
        let vals: Vec<f32> = batch
            .attention_mask
            .iter()
            .map(|&m| if m == 1 { 0.0 } else { MASK_NEG })
            .collect();
        let a = Array::from_slice(&vals, &[batch.n as i32, 1, 1, batch.len as i32]);
        a.as_dtype(self.dtype).lx()
    }

    /// Masks for this batch: dense window masks for short inputs, chunked gather indices and
    /// masks once the sequence is long enough for windowing to pay off.
    fn attn_ctx(&self, batch: &Batch) -> Result<AttnCtx> {
        let has_local = self.layers.iter().any(|l| l.local);
        let s = self.window;
        let dense_upto = self.knobs.dense_upto.unwrap_or(4 * s);
        let windowed = self.knobs.windowed && batch.len > dense_upto;
        if self.knobs.bool_mask && !windowed {
            let (n, len) = (batch.n as i32, batch.len as i32);
            let valid = Array::from_slice(&batch.attention_mask, &[n, len])
                .as_dtype(Dtype::Bool)
                .lx()?;
            let pad = valid.reshape(&[n, 1, 1, len]).lx()?;
            let local = if has_local {
                // Padded queries may see every valid key so no softmax row is fully masked
                // (as Python laya-mlx); they are never used as keys or outputs.
                let pad_q = ops::logical_not(&valid.reshape(&[n, 1, len, 1]).lx()?).lx()?;
                let band = self.window_mask_bool(batch.len)?;
                ops::logical_and(&ops::logical_or(&band, &pad_q).lx()?, &pad).lx()?
            } else {
                pad.clone()
            };
            return Ok(AttnCtx {
                pad,
                local: LocalAttn::Dense(local),
            });
        }
        let pad = self.pad_mask(batch)?;
        let local = if !has_local {
            LocalAttn::Dense(pad.clone())
        } else if windowed {
            let (n, len) = (batch.n, batch.len);
            let nc = len.div_ceil(s);
            let ks = 3 * s;
            let rows = n * self.n_heads;
            let mut caches = self.lock_caches()?;
            let key_idx = match caches.key_idx.get(&(n, len)) {
                Some(a) => a.clone(),
                None => {
                    let idx: Vec<u32> = (0..rows)
                        .flat_map(|r| (0..nc).map(move |c| (r, c)))
                        .flat_map(|(r, c)| {
                            (0..ks).map(move |j| (r, (c as i64 - 1) * s as i64 + j as i64))
                        })
                        .map(|(r, pos)| (r * len) as u32 + pos.clamp(0, len as i64 - 1) as u32)
                        .collect();
                    let a = Array::from_slice(&idx, &[(rows * nc * ks) as i32]);
                    a.eval().lx()?;
                    caches.key_idx.insert((n, len), a.clone());
                    a
                }
            };
            drop(caches);
            // Key validity per (row, chunk, key slot): in range and not padding.
            let mut vals = vec![MASK_NEG; n * nc * ks];
            for b in 0..n {
                for c in 0..nc {
                    for j in 0..ks {
                        let pos = (c as i64 - 1) * s as i64 + j as i64;
                        if pos >= 0
                            && (pos as usize) < len
                            && batch.attention_mask[b * len + pos as usize] == 1
                        {
                            vals[(b * nc + c) * ks + j] = 0.0;
                        }
                    }
                }
            }
            let (n, nc, ks, s) = (n as i32, nc as i32, ks as i32, s as i32);
            let valid = Array::from_slice(&vals, &[n, 1, nc, 1, ks])
                .as_dtype(self.dtype)
                .lx()?;
            let h = self.n_heads as i32;
            let mask = ops::add(&valid, &self.band).lx()?;
            let mask = ops::broadcast_to(&mask, &[n, h, nc, s, ks])
                .lx()?
                .reshape(&[n * h, nc, s, ks])
                .lx()?;
            LocalAttn::Windowed(Windowed {
                key_idx,
                mask,
                n_chunks: nc,
            })
        } else {
            LocalAttn::Dense(ops::add(&pad, &self.window_mask(batch.len)?).lx()?)
        };
        Ok(AttnCtx { pad, local })
    }

    fn lock_caches(&self) -> Result<std::sync::MutexGuard<'_, Caches>> {
        self.caches
            .lock()
            .map_err(|_| Error::Backend("mask cache poisoned".into()))
    }

    /// Boolean sliding-window mask `[1, 1, len, len]` (true = may attend); see
    /// [`Self::window_mask`] for the caching.
    fn window_mask_bool(&self, len: usize) -> Result<Array> {
        let mut caches = self.lock_caches()?;
        if caches.window_bool.as_ref().is_none_or(|m| (m.dim(2) as usize) < len) {
            let mut vals = vec![false; len * len];
            for i in 0..len {
                for j in 0..len {
                    vals[i * len + j] = i.abs_diff(j) <= self.window;
                }
            }
            let m = Array::from_slice(&vals, &[1, 1, len as i32, len as i32]);
            m.eval().lx()?;
            caches.window_bool = Some(m);
        }
        Ok(Self::window_view(caches.window_bool.as_ref().expect("set above"), len))
    }

    /// `residual + x @ W^T (+ b)`, fused into the gemm unless `addmm=0`.
    fn lin_add(&self, l: &Linear, x: &Array, residual: &Array) -> Result<Array> {
        if self.knobs.addmm {
            return l.apply_add(x, residual);
        }
        ops::add(&l.apply(x)?, residual).lx()
    }

    /// Additive sliding-window mask `[1, 1, len, len]`. Whether `i` may see `j` depends on
    /// `|i - j|` alone, so the mask for `len` is the top-left corner of any longer one: one
    /// array for the longest length so far is kept and shorter lengths get a view of it.
    fn window_mask(&self, len: usize) -> Result<Array> {
        let mut caches = self.lock_caches()?;
        if caches.window_mask.as_ref().is_none_or(|m| (m.dim(2) as usize) < len) {
            let mut vals = vec![0f32; len * len];
            for i in 0..len {
                for j in 0..len {
                    if i.abs_diff(j) > self.window {
                        vals[i * len + j] = MASK_NEG;
                    }
                }
            }
            let m = Array::from_slice(&vals, &[1, 1, len as i32, len as i32])
                .as_dtype(self.dtype)
                .lx()?;
            m.eval().lx()?;
            caches.window_mask = Some(m);
        }
        Ok(Self::window_view(caches.window_mask.as_ref().expect("set above"), len))
    }

    /// The `[1, 1, len, len]` corner of a `[1, 1, L, L]` window mask, `L >= len`.
    fn window_view(m: &Array, len: usize) -> Array {
        if m.dim(2) as usize == len {
            m.clone()
        } else {
            m.index((.., .., ..len as i32, ..len as i32))
        }
    }

    /// Sliding-window attention by chunks. Queries `[n, H, len, hd]` are padded to a multiple
    /// of `S = window` and viewed as `[n*H, chunks, S, hd]`; keys/values are gathered into
    /// `[n*H, chunks, 3S, hd]` (chunks c-1, c, c+1, clamped, masked when out of range), so
    /// each chunk runs a small fused attention instead of a `len x len` one.
    fn windowed_attention(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        w: &Windowed,
        scale: f32,
    ) -> Result<Array> {
        let (n, h, len, hd) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3));
        let s = self.window as i32;
        let n_chunks = w.n_chunks;
        let lp = n_chunks * s;
        let q = if lp != len {
            ops::pad(
                q,
                &[(0, 0), (0, 0), (0, lp - len), (0, 0)],
                None::<Array>,
                None::<ops::PadMode>,
            )
            .lx()?
        } else {
            q.clone()
        };
        let q = q.reshape(&[n * h, n_chunks, s, hd]).lx()?;
        // Gather whole `hd` rows from the flattened `[n*H*len, hd]` view: several times faster
        // than `take_axis` on axis 2.
        let gather = |x: &Array| -> Result<Array> {
            x.reshape(&[n * h * len, hd])
                .lx()?
                .take_axis(&w.key_idx, 0)
                .lx()?
                .reshape(&[n * h, n_chunks, 3 * s, hd])
                .lx()
        };
        let (k, v) = (gather(k)?, gather(v)?);
        let att =
            fast::scaled_dot_product_attention(&q, &k, &v, scale, &w.mask, None::<&Array>).lx()?;
        let att = att.reshape(&[n, h, lp, hd]).lx()?;
        Ok(if lp != len {
            att.index((.., .., ..len, ..))
        } else {
            att
        })
    }

    /// `[n, len, heads * hd] -> [n, heads, len, hd]`.
    fn split_heads(&self, x: &Array, n: i32, len: i32, heads: usize) -> Result<Array> {
        let hd = x.dim(-1) / heads as i32;
        x.reshape(&[n, len, heads as i32, hd])
            .lx()?
            .transpose_axes(&[0, 2, 1, 3])
            .lx()
    }

    /// `[n, heads, len, hd] -> [n, len, d]`.
    fn merge_heads(&self, x: &Array, n: i32, len: i32) -> Result<Array> {
        x.transpose_axes(&[0, 2, 1, 3])
            .lx()?
            .reshape(&[n, len, self.hidden as i32])
            .lx()
    }

    /// Encoder `qkv [n, len, 3d]` -> roped `q`, roped `k` and `v`, each `[n, H, len, hd]`.
    fn qkv_rope(&self, qkv: &Array, n: i32, len: i32, theta: f32) -> Result<(Array, Array, Array)> {
        let rope = |x: &Array| -> Result<Array> {
            fast::rope(x, self.head_dim as i32, false, theta, 1.0, 0, None::<&Array>).lx()
        };
        let parts = qkv.split_equal(3, -1).lx()?;
        let q = self.split_heads(&parts[0], n, len, self.n_heads)?;
        let k = self.split_heads(&parts[1], n, len, self.n_heads)?;
        let v = self.split_heads(&parts[2], n, len, self.n_heads)?;
        Ok((rope(&q)?, rope(&k)?, v))
    }

    /// Token index of position 0 of every row: `r * len` in the padded layout, the packing
    /// offsets when the hidden states are packed.
    fn row_starts(batch: &Batch, packing: Option<&Packing>) -> Vec<u32> {
        match packing {
            Some(p) => p.offsets.clone(),
            None => (0..batch.n as u32).map(|r| r * batch.len as u32).collect(),
        }
    }

    /// Flat token indices of the rows the scorer reads: position 0 of every row followed by
    /// its `kmax` marker slots (padded slots point at position 0, as in `Batch`).
    fn scorer_rows(batch: &Batch, starts: &[u32]) -> Array {
        let kmax = batch.kmax;
        let flat: Vec<u32> = (0..batch.n)
            .flat_map(|r| {
                let base = starts[r];
                std::iter::once(base)
                    .chain(batch.marker_pos[r * kmax..(r + 1) * kmax].iter().map(move |&p| base + p))
            })
            .collect();
        Array::from_slice(&flat, &[flat.len() as i32])
    }

    /// ModernBERT encoder; returns `last_hidden_state [n, len, d]` and the pad mask.
    fn encode(&self, batch: &Batch) -> Result<Encoded> {
        let (n, len) = (batch.n as i32, batch.len as i32);
        let d = self.hidden as i32;
        // `unpad` only pays off when there is padding; a batch without any runs the plain path.
        let packing = if self.knobs.unpad && batch.total_tokens() < batch.n * batch.len {
            Some(Packing::new(batch))
        } else {
            None
        };
        let mut h = match &packing {
            Some(_) => {
                let ids: Vec<u32> = batch
                    .input_ids
                    .iter()
                    .zip(&batch.attention_mask)
                    .filter(|(_, &m)| m == 1)
                    .map(|(&id, _)| id)
                    .collect();
                let ids = Array::from_slice(&ids, &[ids.len() as i32]);
                self.tok_emb.take_axis(&ids, 0).lx()?
            }
            None => {
                let ids = Array::from_slice(&batch.input_ids, &[batch.n as i32 * len]);
                self.tok_emb.take_axis(&ids, 0).lx()?.reshape(&[n, len, d]).lx()?
            }
        };
        h = self.emb_norm.apply(&h)?;

        let ctx = self.attn_ctx(batch)?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut lt: Vec<u128> = Vec::new();
        if self.knobs.layers {
            let t = std::time::Instant::now();
            transforms::eval([&h, &ctx.pad]).lx()?;
            if let LocalAttn::Dense(m) = &ctx.local {
                m.eval().lx()?;
            }
            lt.push(t.elapsed().as_micros());
        }

        let mut op_us: Vec<(&str, u128)> = Vec::new();
        let mut t_op = std::time::Instant::now();
        let mut mark = |name: &'static str, arrs: &[&Array]| -> Result<()> {
            if self.knobs.ops {
                transforms::eval(arrs.iter().copied()).lx()?;
                let us = t_op.elapsed().as_micros();
                match op_us.iter_mut().find(|(k, _)| *k == name) {
                    Some(e) => e.1 += us,
                    None => op_us.push((name, us)),
                }
                t_op = std::time::Instant::now();
            }
            Ok(())
        };
        mark("emb+masks", &[&h])?;
        for layer in &self.layers {
            let t = std::time::Instant::now();
            let a = match &layer.attn_norm {
                Some(norm) => norm.apply(&h)?,
                None => h.clone(),
            };
            mark("attn_norm", &[&a])?;
            let qkv = layer.wqkv.apply(&a)?;
            // Packed: back to `[n, len, 3d]` for attention.
            let qkv = match &packing {
                Some(p) => p.expand(&qkv, n, len)?,
                None => qkv,
            };
            mark("wqkv", &[&qkv])?;
            let (q, k, v) = self.qkv_rope(&qkv, n, len, layer.rope_theta)?;
            mark("split+rope", &[&q, &k, &v])?;
            let att = match (&ctx.local, layer.local) {
                (LocalAttn::Windowed(w), true) => self.windowed_attention(&q, &k, &v, w, scale)?,
                (LocalAttn::Dense(mask), true) => {
                    fast::scaled_dot_product_attention(&q, &k, &v, scale, mask, None::<&Array>)
                        .lx()?
                }
                (_, false) => {
                    fast::scaled_dot_product_attention(&q, &k, &v, scale, &ctx.pad, None::<&Array>)
                        .lx()?
                }
            };
            mark(if layer.local { "sdpa_local" } else { "sdpa_global" }, &[&att])?;
            let att = self.merge_heads(&att, n, len)?;
            let att = match &packing {
                Some(p) => p.compact(&att, n, len)?,
                None => att,
            };
            h = self.lin_add(&layer.wo, &att, &h)?;
            mark("merge+wo", &[&h])?;

            let m = layer.mlp_norm.apply(&h)?;
            mark("mlp_norm", &[&m])?;
            let wi = layer.wi.apply(&m)?;
            mark("wi", &[&wi])?;
            let act = self.geglu.apply(&wi)?;
            mark("geglu", &[&act])?;
            h = self.lin_add(&layer.wo2, &act, &h)?;
            mark("wo2", &[&h])?;
            if self.knobs.layers {
                h.eval().lx()?;
                lt.push(t.elapsed().as_micros());
            }
        }
        if self.knobs.ops {
            eprintln!("ops_us n={} len={} h_dtype={:?} {:?}", batch.n, batch.len, h.dtype(), op_us);
        }
        if self.knobs.layers {
            eprintln!("layers_us n={} len={} emb+masks={} layers={:?}", batch.n, batch.len, lt[0], &lt[1..]);
        }
        Ok(Encoded {
            h: self.final_norm.apply(&h)?,
            pad: ctx.pad,
            packing,
        })
    }

    /// Decision-head transformer layers on top of the encoder output. With `headprune` the
    /// last layer is only computed at the scorer's rows, and the result says so in its type
    /// (see [`HeadOut`]).
    fn head_forward(&self, batch: &Batch, enc: &Encoded) -> Result<HeadOut> {
        let (n, len) = (batch.n as i32, batch.len as i32);
        let d = self.hidden as i32;
        let packing = enc.packing.as_ref();
        let qtype = Array::from_slice(&batch.qtype, &[n]);
        let te = self.type_emb.take_axis(&qtype, 0).lx()?;
        let mut h = match packing {
            // Packed tokens take their row's type embedding row by row.
            Some(p) => ops::add(&enc.h, &te.take_axis(&p.row, 0).lx()?).lx()?,
            None => ops::add(&enc.h, &te.reshape(&[n, 1, d]).lx()?).lx()?,
        };
        let scale = 1.0 / 8.0; // head_dim 64
        let sdpa = |q: &Array, k: &Array, v: &Array| -> Result<Array> {
            fast::scaled_dot_product_attention(q, k, v, scale, &enc.pad, None::<&Array>).lx()
        };
        // Around attention only: `[T, c] -> [n, len, c]` and back (identity when not packed).
        let expand = |x: Array| -> Result<Array> {
            match packing {
                Some(p) => p.expand(&x, n, len),
                None => Ok(x),
            }
        };
        let compact = |x: Array| -> Result<Array> {
            match packing {
                Some(p) => p.compact(&x, n, len),
                None => Ok(x),
            }
        };
        for layer in &self.head {
            let x = layer.norm1.apply(&h)?;
            if let Some((wq, wkv)) = &layer.split_proj {
                // `headprune` (last layer only): every token still supplies a key and a value,
                // but queries, out_proj, norm2 and the FFN run only for the rows the scorer
                // reads. A token's output depends on the other tokens only through attention,
                // so the picked rows come out the same as in the full layer.
                let kv = expand(wkv.apply(&x)?)?;
                let parts = kv.split_equal(2, -1).lx()?;
                let k = self.split_heads(&parts[0], n, len, self.head_nheads)?;
                let v = self.split_heads(&parts[1], n, len, self.head_nheads)?;
                let r = 1 + batch.kmax as i32;
                let idx = Self::scorer_rows(batch, &Self::row_starts(batch, packing));
                let pick = |a: &Array| -> Result<Array> {
                    a.reshape(&[-1, d]).lx()?.take_axis(&idx, 0).lx()?.reshape(&[n, r, d]).lx()
                };
                let q = self.split_heads(&wq.apply(&pick(&x)?)?, n, r, self.head_nheads)?;
                let att = self.merge_heads(&sdpa(&q, &k, &v)?, n, r)?;
                let hs = self.lin_add(&layer.out_proj, &att, &pick(&h)?)?;
                let x = layer.norm2.apply(&hs)?;
                let x = nn::relu(&layer.linear1.apply(&x)?).lx()?;
                return Ok(HeadOut::Rows(ScorerRows(self.lin_add(&layer.linear2, &x, &hs)?)));
            }
            let qkv = expand(layer.in_proj.apply(&x)?)?;
            let parts = qkv.split_equal(3, -1).lx()?;
            let q = self.split_heads(&parts[0], n, len, self.head_nheads)?;
            let k = self.split_heads(&parts[1], n, len, self.head_nheads)?;
            let v = self.split_heads(&parts[2], n, len, self.head_nheads)?;
            let att = compact(self.merge_heads(&sdpa(&q, &k, &v)?, n, len)?)?;
            h = self.lin_add(&layer.out_proj, &att, &h)?;

            let x = layer.norm2.apply(&h)?;
            let x = nn::relu(&layer.linear1.apply(&x)?).lx()?;
            h = self.lin_add(&layer.linear2, &x, &h)?;
        }
        Ok(match packing {
            Some(_) => HeadOut::Packed(h),
            None => HeadOut::Full(h),
        })
    }

    /// Scorer logits `[n * kmax]` (unmasked) and pooled `[n, d]` as f32 arrays.
    fn score(&self, batch: &Batch, h: &HeadOut, packing: Option<&Packing>) -> Result<(Array, Array)> {
        let (n, len, kmax) = (batch.n as i32, batch.len as i32, batch.kmax as i32);
        let d = self.hidden as i32;
        let (pooled, m) = match h {
            HeadOut::Full(h) => {
                let pooled = h
                    .take_axis(Array::from_slice(&[0u32], &[1]), 1)
                    .lx()?
                    .reshape(&[n, d])
                    .lx()?;
                let flat: Vec<u32> = (0..batch.n)
                    .flat_map(|r| (0..batch.kmax).map(move |k| (r, k)))
                    .map(|(r, k)| r as u32 * len as u32 + batch.marker_pos[r * batch.kmax + k])
                    .collect();
                let idx = Array::from_slice(&flat, &[n * kmax]);
                (pooled, h.reshape(&[n * len, d]).lx()?.take_axis(&idx, 0).lx()?)
            }
            HeadOut::Packed(h) => {
                let starts = Self::row_starts(batch, packing);
                let pooled = h.take_axis(Array::from_slice(&starts, &[n]), 0).lx()?;
                let flat: Vec<u32> = (0..batch.n)
                    .flat_map(|r| (0..batch.kmax).map(move |k| (r, k)))
                    .map(|(r, k)| starts[r] + batch.marker_pos[r * batch.kmax + k])
                    .collect();
                (pooled, h.take_axis(Array::from_slice(&flat, &[n * kmax]), 0).lx()?)
            }
            HeadOut::Rows(rows) => (rows.pooled(), rows.markers(n, kmax, d)?),
        };
        let s = self.scorer_norm.apply(&m)?;
        let s = gelu_erf_as(&self.scorer1.apply(&s)?, self.knobs.f16gelu).lx()?;
        let logits = self.scorer3.apply(&s)?.reshape(&[n * kmax]).lx()?;
        Ok((to_f32_contiguous(&logits)?, to_f32_contiguous(&pooled)?))
    }
}

impl Backend for MlxBackend {
    fn name(&self) -> String {
        let dev = if self.cpu { "cpu" } else { "gpu" };
        let dt = if self.dtype == Dtype::Float32 {
            "f32"
        } else {
            "f16"
        };
        format!("mlx({dev},{dt})")
    }

    fn padded_len(&self, len: usize, _rows: usize) -> usize {
        if let Some(&b) = self.knobs.buckets.iter().find(|&&b| b >= len) {
            return b;
        }
        len.div_ceil(self.knobs.pad) * self.knobs.pad
    }

    fn forward(&self, batch: &Batch) -> Result<BackendOutput> {
        let stream = self.stream();
        mlx_rs::with_stream(&stream, || {
            let t0 = std::time::Instant::now();
            let enc = self.encode(batch)?;
            let h = self.head_forward(batch, &enc)?;
            let (logits, pooled) = self.score(batch, &h, enc.packing.as_ref())?;
            let t1 = std::time::Instant::now();
            transforms::eval([&logits, &pooled]).lx()?;
            if self.knobs.trace {
                eprintln!(
                    "forward n={} len={} build_us={} eval_us={}",
                    batch.n,
                    batch.len,
                    (t1 - t0).as_micros(),
                    t1.elapsed().as_micros()
                );
            }
            let mut logits = host_f32(&logits)?;
            for r in 0..batch.n {
                for k in batch.marker_count[r]..batch.kmax {
                    logits[r * batch.kmax + k] = LOGIT_MASKED;
                }
            }
            let pooled = host_f32(&pooled)?;
            if self.knobs.clear {
                mlx_rs::memory::clear_cache().lx()?;
            }
            Ok(BackendOutput { logits, pooled })
        })
    }

    fn encoder_hidden(&self, batch: &Batch) -> Result<Option<Vec<f32>>> {
        let stream = self.stream();
        mlx_rs::with_stream(&stream, || {
            let enc = self.encode(batch)?;
            let h = match &enc.packing {
                // Back to `[n, len, d]`; padding positions repeat their row's token 0.
                Some(p) => p.expand(&enc.h, batch.n as i32, batch.len as i32)?,
                None => enc.h,
            };
            let h = to_f32_contiguous(&h)?;
            h.eval().lx()?;
            Ok(Some(host_f32(&h)?))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knobs_default_off_and_parse() {
        let k = Knobs::from_spec(Some(""));
        assert_eq!(k.dense_upto, None);
        assert!(!k.headprune && !k.unpad);
        let k = Knobs::from_spec(Some("dense_upto=512,headprune,unpad"));
        assert_eq!(k.dense_upto, Some(512));
        assert!(k.headprune && k.unpad);
        let k = Knobs::from_spec(Some("headprune=0,unpad=0"));
        assert!(!k.headprune && !k.unpad);
    }

    /// The padding rows of a batch go through the packing as their row's token 0, and every
    /// scorer row (position 0 and the marker slots) lands on the right packed token.
    #[test]
    fn packing_indexes_real_tokens_and_scorer_rows() {
        let batch = Batch {
            n: 3,
            len: 4,
            kmax: 2,
            input_ids: vec![1, 2, 0, 0, 3, 4, 5, 6, 7, 0, 0, 0],
            attention_mask: vec![1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            seq_lens: vec![2, 4, 1],
            marker_pos: vec![1, 0, 1, 3, 0, 0],
            marker_count: vec![1, 2, 0],
            qtype: vec![0, 1, 2],
        };
        let p = Packing::new(&batch);
        assert_eq!(p.offsets, vec![0, 2, 6]);
        assert_eq!(p.pack.as_slice::<u32>(), &[0, 1, 4, 5, 6, 7, 8]);
        assert_eq!(p.unpack.as_slice::<u32>(), &[0, 1, 0, 0, 2, 3, 4, 5, 6, 6, 6, 6]);
        assert_eq!(p.row.as_slice::<u32>(), &[0, 0, 1, 1, 1, 1, 2]);
        // Padded layout: row start `r * len`; packed: the offsets. Padded marker slots point
        // at position 0 in both.
        let padded = MlxBackend::scorer_rows(&batch, &MlxBackend::row_starts(&batch, None));
        assert_eq!(padded.as_slice::<u32>(), &[0, 1, 0, 4, 5, 7, 8, 8, 8]);
        let packed = MlxBackend::scorer_rows(&batch, &MlxBackend::row_starts(&batch, Some(&p)));
        assert_eq!(packed.as_slice::<u32>(), &[0, 1, 0, 2, 3, 5, 6, 6, 6]);
    }

    /// One window mask serves every shorter length as a view with the same values.
    #[test]
    fn window_view_is_the_corner_of_the_longer_mask() {
        let full = window_mask_values(8, 2);
        let m = Array::from_slice(&full, &[1, 1, 8, 8]);
        assert_eq!(MlxBackend::window_view(&m, 8).as_slice::<f32>(), &full[..]);
        let corner = MlxBackend::window_view(&m, 5).contiguous().unwrap();
        assert_eq!(corner.shape(), &[1, 1, 5, 5]);
        assert_eq!(corner.as_slice::<f32>(), &window_mask_values(5, 2)[..]);
    }

    /// Reference values of the additive window mask for `len` and window `w`.
    fn window_mask_values(len: usize, w: usize) -> Vec<f32> {
        let mut vals = vec![0f32; len * len];
        for i in 0..len {
            for j in 0..len {
                if i.abs_diff(j) > w {
                    vals[i * len + j] = MASK_NEG;
                }
            }
        }
        vals
    }
}
