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
//! masks, and diagnostic timing.

use laya_core::weights::{to_f16, to_f32};
use laya_core::{
    Backend, BackendOptions, BackendOutput, Batch, Device, Error, ModelConfig, Result, Weights,
};
use mlx_rs::error::Exception;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::compile::compile;
use mlx_rs::{fast, nn, ops, transforms, Array, Dtype, Stream};
use safetensors::SafeTensors;
use std::collections::HashSet;
use std::sync::Mutex;

/// Finite "minus infinity" for additive attention masks (safe in f16, no NaN rows).
const MASK_NEG: f32 = -1e4;
/// Logit value reported for masked marker slots.
const LOGIT_MASKED: f32 = -1e4;
/// Distinct input shapes the per-shape GeGLU (`geglu=compiled`) compiles before new shapes take
/// the shapeless trace instead. MLX keeps every per-shape trace for the life of the process, so
/// this count bounds that memory; 64 covers a few length buckets times the row counts a server
/// sees, and the shapeless trace serves everything past it.
const GEGLU_MAX_SHAPES: usize = 64;

/// Backend settings, read once at load from `BackendOptions::tuning` or else `SYS1_MLX` (comma
/// list, e.g. `f16gelu,mask=bool`). Defaults reproduce laya-r-mlx 914c9a7.
///
/// Parsing is strict: an unknown setting or a value the setting cannot take is an
/// [`Error::Config`] naming it, so a run cannot be labeled with a setting that was never
/// applied. Flags take no value (`f16gelu`) or `=0`/`=1`; everything else needs `key=value`.
#[derive(Debug, Clone)]
struct Knobs {
    /// `shapeless` (default): split outside, GELU * gate compiled once for all shapes.
    /// `compiled`: split + GELU + gate compiled per input shape, one trace per distinct
    /// `(rows, len)` that MLX never frees, so at most [`GEGLU_MAX_SHAPES`] shapes are compiled
    /// this way and every new shape after that runs the shapeless trace; for experiments only.
    /// `plain`: no compile.
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
    /// Keep GELU in the compute dtype (fixes the upstream f16 -> f32 promotion). Off by
    /// default like every knob here, so that `BackendOptions::default()` reproduces upstream's
    /// numerics and the bench's `mlx-fp16` control variant measures the promotion. Everything
    /// that serves answers (`sys1d`, the other bench variants) sets `f16gelu`.
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
}

impl Knobs {
    /// The settings of `tuning`, or of `SYS1_MLX` when `tuning` is `None`.
    fn from_spec(tuning: Option<&str>) -> Result<Self> {
        let spec = match tuning {
            Some(t) => t.to_string(),
            None => std::env::var("SYS1_MLX").unwrap_or_default(),
        };
        Self::parse(&spec)
    }

    /// Parse a comma-separated spec. The last mention of a setting wins.
    fn parse(spec: &str) -> Result<Self> {
        let mut k = Knobs {
            geglu: "shapeless".into(),
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
        };
        for kv in spec.split(',').filter(|s| !s.is_empty()) {
            let (key, val) = match kv.split_once('=') {
                Some((key, val)) => (key, Some(val)),
                None => (kv, None),
            };
            let bad = |what: String| Error::Config(format!("mlx settings: `{kv}`: {what}"));
            // A flag is on when bare or `=1`, off when `=0`; nothing else.
            let flag = || match val {
                None | Some("1") => Ok(true),
                Some("0") => Ok(false),
                Some(v) => Err(bad(format!("`{v}` is not 0 or 1"))),
            };
            let value = || val.ok_or_else(|| bad("needs a value (`key=value`)".into()));
            let one_of = |allowed: &[&str]| -> Result<String> {
                let v = value()?;
                if allowed.contains(&v) {
                    Ok(v.to_string())
                } else {
                    Err(bad(format!("`{v}` is not one of {}", allowed.join(", "))))
                }
            };
            let number = |v: &str| -> Result<usize> {
                v.parse().map_err(|_| bad(format!("`{v}` is not a whole number")))
            };
            let list = || -> Result<Vec<usize>> { value()?.split(':').map(number).collect() };
            match key {
                "geglu" => k.geglu = one_of(&["shapeless", "compiled", "plain"])?,
                "addmm" => k.addmm = flag()?,
                "mask" => k.bool_mask = one_of(&["bool", "additive"])? == "bool",
                "windowed" => k.windowed = flag()?,
                "clear" => k.clear = flag()?,
                "trace" => k.trace = flag()?,
                "layers" => k.layers = flag()?,
                "ops" => k.ops = flag()?,
                "wcopy" => k.wcopy = one_of(&["view", "gpu", "t"])?,
                "f16gelu" => k.f16gelu = flag()?,
                "buckets" => {
                    k.buckets = list()?;
                    k.buckets.sort_unstable();
                }
                "pad" => {
                    k.pad = number(value()?)?;
                    if k.pad == 0 {
                        return Err(bad("pad must be at least 1".into()));
                    }
                }
                "warm" => k.warm = list()?,
                "cache" => k.cache_mb = Some(number(value()?)?),
                "wired" => k.wired_mb = Some(number(value()?)?),
                _ => return Err(Error::Config(format!("mlx settings: unknown setting `{key}` in `{spec}`"))),
            }
        }
        Ok(k)
    }
}

/// Check a settings spec (`BackendOptions::tuning`, `SYS1_MLX`) without loading a model: `Err`
/// names the first unknown setting or bad value, as loading with it would. Callers that label
/// a run with its settings (sys1-bench, sys1-probe) call this at startup.
pub fn check_settings(spec: &str) -> Result<()> {
    Knobs::parse(spec).map(|_| ())
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
///
/// The default is the shapeless trace: GELU * gate is elementwise, so one trace serves every
/// input shape and the split (which needs concrete shapes) happens outside as two views. Paired
/// A/B on the timing workload against the per-shape trace, `f16gelu,cache=512,wired=2048`:
/// 1.002 on typed-decisions, identical answers. The per-shape mode (`geglu=compiled`) stays
/// for experiments; MLX keeps one trace per distinct input shape for the life of the process,
/// so the mode compiles at most [`GEGLU_MAX_SHAPES`] shapes and hands every new shape after
/// that to the shapeless trace ([`ShapeBudget`]).
struct GeGlu {
    mode: String,
    keep: bool,
    traces: Mutex<GeGluTraces>,
}

/// The compiled traces behind [`GeGlu`]'s mutex.
struct GeGluTraces {
    /// GELU * gate over the two halves; one trace for every shape.
    shapeless: CompiledFn,
    /// Split + GELU * gate, one trace per input shape (`geglu=compiled` only).
    per_shape: Option<CompiledFn>,
    /// The shapes `per_shape` has compiled.
    shapes: ShapeBudget,
}

/// The distinct shapes a per-shape cache may hold, at most `cap` of them.
struct ShapeBudget {
    cap: usize,
    seen: HashSet<Vec<i32>>,
}

impl ShapeBudget {
    fn new(cap: usize) -> Self {
        Self { cap, seen: HashSet::new() }
    }

    /// Whether `shape` may take the per-shape path: yes when it is already cached, or when the
    /// cache has room (the shape is then counted); no once `cap` shapes are cached.
    fn admit(&mut self, shape: &[i32]) -> bool {
        if self.seen.contains(shape) {
            return true;
        }
        if self.seen.len() >= self.cap {
            return false;
        }
        self.seen.insert(shape.to_vec());
        true
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.seen.len()
    }
}

// SAFETY: the compiled closures own no thread-affine resources; calls are serialised by the
// mutex and MLX's scheduler is thread-safe.
unsafe impl Send for GeGlu {}
unsafe impl Sync for GeGlu {}

impl GeGlu {
    fn new(mode: &str, keep: bool) -> Self {
        Self::with_cap(mode, keep, GEGLU_MAX_SHAPES)
    }

    /// `cap` is the number of shapes `geglu=compiled` compiles per shape (tests pass a small one).
    fn with_cap(mode: &str, keep: bool, cap: usize) -> Self {
        let shapeless: CompiledFn = Box::new(compile(
            move |a: &[Array]| -> Vec<Array> {
                vec![ops::multiply(gelu_erf_as(&a[0], keep).expect("gelu"), &a[1]).expect("geglu gate")]
            },
            true,
        ));
        let per_shape: Option<CompiledFn> = (mode == "compiled").then(|| -> CompiledFn {
            Box::new(compile(
                move |a: &[Array]| -> Vec<Array> {
                    let ig = a[0].split_equal(2, -1).expect("geglu split");
                    vec![ops::multiply(gelu_erf_as(&ig[0], keep).expect("gelu"), &ig[1]).expect("geglu gate")]
                },
                // Not shapeless: `split` needs concrete shapes; MLX caches one trace per input
                // shape and never drops one, which is why this is not the default and why
                // `apply` stops sending new shapes here after `cap` of them.
                false,
            ))
        });
        Self {
            mode: mode.to_string(),
            keep,
            traces: Mutex::new(GeGluTraces {
                shapeless,
                per_shape,
                shapes: ShapeBudget::new(cap),
            }),
        }
    }

    fn apply(&self, x: &Array) -> Result<Array> {
        if self.mode == "plain" {
            let ig = x.split_equal(2, -1).lx()?;
            return ops::multiply(gelu_erf_as(&ig[0], self.keep).lx()?, &ig[1]).lx();
        }
        // A panic inside the compiled closure (mlx-rs re-raises it after MLX returns) would
        // poison the lock; recover the guard so one failed request cannot fail every later one.
        let mut guard = self.traces.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let t = &mut *guard;
        let mut out = match &mut t.per_shape {
            Some(f) if t.shapes.admit(x.shape()) => f(std::slice::from_ref(x)).lx()?,
            _ => {
                let ig = x.split_equal(2, -1).lx()?;
                (t.shapeless)(ig.as_slice()).lx()?
            }
        };
        Ok(out.remove(0))
    }

    /// How many shapes the per-shape trace has compiled (0 unless `geglu=compiled`).
    #[cfg(test)]
    fn per_shape_traces(&self) -> usize {
        self.traces.lock().unwrap_or_else(std::sync::PoisonError::into_inner).shapes.len()
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
        let w = self.get(w)?;
        let wt = match self.wcopy.as_str() {
            "t" => ops::transpose(&w).lx()?.contiguous().lx()?,
            "gpu" => {
                let one = Array::from_f32(1.0).as_dtype(self.dtype).lx()?;
                ops::transpose(ops::multiply(&w, &one).lx()?).lx()?
            }
            _ => ops::transpose(&w).lx()?,
        };
        let b = b.map(|n| self.get(n)).transpose()?;
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

/// Constants reused across forwards. Nothing here is keyed by request shape.
///
/// The dense window masks are one array each, for the longest length seen so far; shorter
/// lengths take a view of it (see [`MlxBackend::window_mask`]). One mask per distinct length
/// would pile up: without length buckets, a server seeing every length up to 1,024 would hold
/// about 700 MB of masks that nothing frees. The gather indices of the windowed path are built
/// inside the forward graph instead ([`window_key_idx`]), so they need no cache at all.
#[derive(Default)]
struct Caches {
    /// Dense sliding-window additive mask `[1, 1, L, L]` for the longest `L` so far.
    window_mask: Option<Array>,
    /// Boolean sliding-window mask `[1, 1, L, L]` for the longest `L` so far (`mask=bool`).
    window_bool: Option<Array>,
}

/// Flat key gather indices `[rows * nc * 3S]` into the `[rows * len, hd]` view of the keys for
/// the windowed path: chunk `c` of row `r` gathers positions `(c - 1) * S + j` for `j < 3S`,
/// clamped into `0..len` (the mask hides the clamped slots), as `r * len + pos`. Two small host
/// tables and one broadcast add on the device, part of the forward graph: no per-shape cache
/// and no early evaluation.
fn window_key_idx(s: usize, rows: usize, len: usize, nc: usize) -> Result<Array> {
    let ks = 3 * s;
    let pos: Vec<u32> = (0..nc)
        .flat_map(|c| (0..ks).map(move |j| (c as i64 - 1) * s as i64 + j as i64))
        .map(|p| p.clamp(0, len as i64 - 1) as u32)
        .collect();
    let pos = Array::from_slice(&pos, &[1, nc as i32, ks as i32]);
    let base: Vec<u32> = (0..rows as u32).map(|r| r * len as u32).collect();
    let base = Array::from_slice(&base, &[rows as i32, 1, 1]);
    ops::add(&base, &pos).lx()?.reshape(&[(rows * nc * ks) as i32]).lx()
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
        let knobs = Knobs::from_spec(opts.tuning.as_deref())?;
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
            for j in 0..cfg.agent.head_layers {
                let p = format!("head.layers.{j}");
                head.push(HeadLayer {
                    norm1: loader.norm(
                        &format!("{p}.norm1.weight"),
                        Some(&format!("{p}.norm1.bias")),
                        1e-5,
                    )?,
                    in_proj: loader.linear(
                        &format!("{p}.self_attn.in_proj_weight"),
                        Some(&format!("{p}.self_attn.in_proj_bias")),
                    )?,
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
        let windowed = self.knobs.windowed && batch.len > 4 * s;
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
            let key_idx = window_key_idx(s, n * self.n_heads, len, nc)?;
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
    ///
    /// The new mask is evaluated here, on purpose: the cached array is shared by every later
    /// forward and must hold data, not a graph node those forwards would all point into. This
    /// happens once per new longest length, a few times in the life of a server.
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

    /// `[n, len, d] -> [n, heads, len, hd]`.
    fn split_heads(&self, x: &Array, n: i32, len: i32, heads: usize) -> Result<Array> {
        let hd = (self.hidden / heads) as i32;
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

    /// ModernBERT encoder; returns `last_hidden_state [n, len, d]` and the pad mask.
    fn encode(&self, batch: &Batch) -> Result<(Array, Array)> {
        let (n, len) = (batch.n as i32, batch.len as i32);
        let d = self.hidden as i32;
        let ids = Array::from_slice(&batch.input_ids, &[batch.n as i32 * len]);
        let mut h = self
            .tok_emb
            .take_axis(&ids, 0)
            .lx()?
            .reshape(&[n, len, d])
            .lx()?;
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
            mark("wqkv", &[&qkv])?;
            let parts = qkv.split_equal(3, -1).lx()?;
            let q = self.split_heads(&parts[0], n, len, self.n_heads)?;
            let k = self.split_heads(&parts[1], n, len, self.n_heads)?;
            let v = self.split_heads(&parts[2], n, len, self.n_heads)?;
            let q = fast::rope(
                &q,
                self.head_dim as i32,
                false,
                layer.rope_theta,
                1.0,
                0,
                None::<&Array>,
            )
            .lx()?;
            let k = fast::rope(
                &k,
                self.head_dim as i32,
                false,
                layer.rope_theta,
                1.0,
                0,
                None::<&Array>,
            )
            .lx()?;
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
        Ok((self.final_norm.apply(&h)?, ctx.pad))
    }

    /// Decision-head transformer layers on top of the encoder output.
    fn head_forward(&self, batch: &Batch, enc: &Array, pad: &Array) -> Result<Array> {
        let (n, len) = (batch.n as i32, batch.len as i32);
        let qtype = Array::from_slice(&batch.qtype, &[n]);
        let te = self
            .type_emb
            .take_axis(&qtype, 0)
            .lx()?
            .reshape(&[n, 1, self.hidden as i32])
            .lx()?;
        let mut h = ops::add(enc, &te).lx()?;
        let scale = 1.0 / 8.0; // head_dim 64
        for layer in &self.head {
            let x = layer.norm1.apply(&h)?;
            let qkv = layer.in_proj.apply(&x)?;
            let parts = qkv.split_equal(3, -1).lx()?;
            let q = self.split_heads(&parts[0], n, len, self.head_nheads)?;
            let k = self.split_heads(&parts[1], n, len, self.head_nheads)?;
            let v = self.split_heads(&parts[2], n, len, self.head_nheads)?;
            let att =
                fast::scaled_dot_product_attention(&q, &k, &v, scale, pad, None::<&Array>).lx()?;
            let att = self.merge_heads(&att, n, len)?;
            h = self.lin_add(&layer.out_proj, &att, &h)?;

            let x = layer.norm2.apply(&h)?;
            let x = nn::relu(&layer.linear1.apply(&x)?).lx()?;
            h = self.lin_add(&layer.linear2, &x, &h)?;
        }
        Ok(h)
    }

    /// Scorer logits `[n * kmax]` (unmasked) and pooled `[n, d]` as f32 arrays.
    fn score(&self, batch: &Batch, h: &Array) -> Result<(Array, Array)> {
        let (n, len, kmax) = (batch.n as i32, batch.len as i32, batch.kmax as i32);
        let d = self.hidden as i32;
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
        let m = h.reshape(&[n * len, d]).lx()?.take_axis(&idx, 0).lx()?;
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
            let (enc, pad) = self.encode(batch)?;
            let h = self.head_forward(batch, &enc, &pad)?;
            let (logits, pooled) = self.score(batch, &h)?;
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
            let (enc, _) = self.encode(batch)?;
            let enc = to_f32_contiguous(&enc)?;
            enc.eval().lx()?;
            Ok(Some(host_f32(&enc)?))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_parse_every_kind_of_value() {
        let k = Knobs::parse("f16gelu,addmm=0,mask=bool,geglu=compiled,wcopy=t,buckets=512:128:256,pad=8,warm=1:4,cache=512,wired=2048").unwrap();
        assert!(k.f16gelu && !k.addmm && k.bool_mask);
        assert_eq!((k.geglu.as_str(), k.wcopy.as_str()), ("compiled", "t"));
        assert_eq!((k.buckets, k.pad, k.warm), (vec![128, 256, 512], 8, vec![1, 4]));
        assert_eq!((k.cache_mb, k.wired_mb), (Some(512), Some(2048)));
        // Empty spec and empty items are the defaults; the last mention wins.
        let d = Knobs::parse("").unwrap();
        assert!(!d.f16gelu && d.addmm && d.geglu == "shapeless" && d.cache_mb.is_none());
        assert!(Knobs::parse(",f16gelu,,").unwrap().f16gelu);
        assert!(!Knobs::parse("f16gelu,f16gelu=0").unwrap().f16gelu);
        assert!(!Knobs::parse("mask=bool,mask=additive").unwrap().bool_mask);
    }

    /// An unknown setting or a bad value is an error that names the item, never a silent default.
    #[test]
    fn settings_reject_unknown_keys_and_bad_values() {
        let err = |spec: &str| Knobs::parse(spec).err().map(|e| e.to_string()).unwrap_or_else(|| panic!("{spec} was accepted"));
        assert!(err("f16gelu,cache=512x").contains("`cache=512x`"), "{}", err("f16gelu,cache=512x"));
        assert!(err("f16gelu,chache=512").contains("unknown setting `chache`"), "{}", err("f16gelu,chache=512"));
        assert!(err("geglu=fast").contains("`fast` is not one of shapeless, compiled, plain"));
        assert!(err("mask=1").contains("`mask=1`"));
        assert!(err("wcopy=copy").contains("`wcopy=copy`"));
        assert!(err("f16gelu=yes").contains("`yes` is not 0 or 1"));
        assert!(err("buckets=128:abc").contains("`abc` is not a whole number"));
        assert!(err("buckets=").contains("`buckets=`"));
        assert!(err("pad=0").contains("pad must be at least 1"));
        assert!(err("pad=-1").contains("`-1` is not a whole number"));
        assert!(err("cache").contains("needs a value"));
        assert!(err("wired=2048,f16gelu,cache=512 ").contains("`cache=512 `"));
        assert_eq!(check_settings("f16gelu,cache=512,wired=2048").ok(), Some(()));
        assert!(check_settings("f16gelu,cache=512x").is_err());
    }

    /// The per-shape cache admits `cap` distinct shapes, then only the shapes it already holds.
    #[test]
    fn shape_budget_caps_distinct_shapes() {
        let mut b = ShapeBudget::new(3);
        assert!(b.admit(&[1, 4]) && b.admit(&[2, 4]) && b.admit(&[3, 4]));
        assert_eq!(b.len(), 3);
        assert!(!b.admit(&[4, 4]), "a fourth shape is refused");
        assert!(b.admit(&[2, 4]), "a cached shape stays admitted");
        assert!(!b.admit(&[4, 4]), "the refused shape is not counted");
        assert_eq!(b.len(), 3);
        assert!(!ShapeBudget::new(0).admit(&[1]));
    }

    /// `geglu=compiled` compiles at most `cap` shapes; past the cap new shapes run the shapeless
    /// trace, and every path computes the same values as the plain (uncompiled) GeGLU.
    #[test]
    fn per_shape_geglu_falls_back_past_the_cap() {
        let g = GeGlu::with_cap("compiled", true, 2);
        let plain = GeGlu::new("plain", true);
        for rows in [1i32, 2, 3, 1, 4] {
            let x = Array::from_iter((0..rows * 8).map(|i| i as f32 * 0.25 - 4.0), &[rows, 8]);
            let (got, want) = (g.apply(&x).unwrap(), plain.apply(&x).unwrap());
            assert_eq!(got.shape(), &[rows, 4]);
            let d = ops::abs(&ops::subtract(&got, &want).unwrap()).unwrap().max(None).unwrap();
            assert!(d.item_exact::<f32>() < 1e-6, "rows={rows}: max diff {d}");
        }
        assert_eq!(g.per_shape_traces(), 2, "shapes [1,8] and [2,8] compiled per shape, [3,8] and [4,8] fell back");
        assert_eq!(GeGlu::new("shapeless", true).per_shape_traces(), 0);
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

    /// The device-built gather indices are the ones the host loop used to build and cache:
    /// `r * len + clamp((c - 1) * S + j, 0, len - 1)`, row-major over `(r, c, j)`.
    #[test]
    fn window_key_idx_matches_the_host_formula() {
        for (s, rows, len) in [(2usize, 3usize, 7usize), (4, 1, 4), (64, 2, 130)] {
            let nc = len.div_ceil(s);
            let want: Vec<u32> = (0..rows)
                .flat_map(|r| (0..nc).map(move |c| (r, c)))
                .flat_map(|(r, c)| (0..3 * s).map(move |j| (r, (c as i64 - 1) * s as i64 + j as i64)))
                .map(|(r, pos)| (r * len) as u32 + pos.clamp(0, len as i64 - 1) as u32)
                .collect();
            let got = window_key_idx(s, rows, len, nc).unwrap();
            assert_eq!(got.shape(), &[(rows * nc * 3 * s) as i32]);
            assert_eq!(got.as_slice::<u32>(), &want[..], "s={s} rows={rows} len={len}");
        }
    }
}
