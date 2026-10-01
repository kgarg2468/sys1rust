//! `fuserope`: the encoder's qkv split, head reshape, RoPE of q and k, and the `unpad` expand
//! in one Metal kernel (`kernels/split_rope.metal`), in place of MLX's `split_equal`, three
//! reshape copies, two `fast::rope` launches and the `take_axis` gather. The kernel computes
//! what those ops compute, with MLX's own rope arithmetic, and [`mlx_path`] is that chain, kept
//! here as the reference the kernel is checked against: once at model load on small inputs
//! (every pipeline a request can hit), and in this module's tests at several shapes.
//!
//! Instantiations are bounded: the template arguments are the element type, the head count,
//! the head dim and whether the batch is packed, all fixed for a loaded model. MLX also picks
//! the address space of each input by its element count (fewer than 8 elements: `constant`,
//! otherwise `device`; `max_constant_array_size` in `mlx/backend/common/metal_kernel.cpp` of
//! MLX 0.32.2) and names the pipeline after that choice. The qkv rows, `dims` and `lbase`
//! never cross that limit; the packing's `unpack` index does. So a model has the 3 pipelines
//! of [`Variant`], and the load-time check runs all 3. The batch shape travels in the `dims`
//! input, never in a template argument.

use crate::metal_kernels::{MetalKernel, TemplateArg};
use crate::Lx;
use laya_core::{Error, Result};
use mlx_rs::{fast, ops, transforms, Array, Dtype};

const SOURCE: &str = include_str!("kernels/split_rope.metal");
/// Pairs of elements each thread rotates; the kernel loads and stores `vec<T, 4>`.
const PAIRS_PER_THREAD: i32 = 4;
const THREADGROUP: i32 = 256;
/// MLX passes an input with fewer elements than this in the `constant` address space, as a
/// pipeline of its own (`max_constant_array_size`, `mlx/backend/common/metal_kernel.cpp`).
const MLX_CONSTANT_LIMIT: usize = 8;

/// The MLX chain the kernel replaces: `qkv [n, len, 3 * heads * hd]` -> roped `q`, roped `k`
/// and `v`, each `[n, heads, len, hd]`.
pub(crate) fn mlx_path(qkv: &Array, n: i32, len: i32, heads: i32, hd: i32, theta: f32) -> Result<(Array, Array, Array)> {
    let heads_of = |x: &Array| -> Result<Array> {
        x.reshape(&[n, len, heads, hd]).lx()?.transpose_axes(&[0, 2, 1, 3]).lx()
    };
    let rope = |x: &Array| -> Result<Array> { fast::rope(x, hd, false, theta, 1.0, 0, None::<&Array>).lx() };
    let parts = qkv.split_equal(3, -1).lx()?;
    let q = heads_of(&parts[0])?;
    let k = heads_of(&parts[1])?;
    let v = heads_of(&parts[2])?;
    Ok((rope(&q)?, rope(&k)?, v))
}

/// `[log2(theta)]` as the kernel's `lbase` input, computed as MLX's rope does it (`log2` of the
/// base as an f32).
fn log2_base(theta: f32) -> Array {
    Array::from_slice(&[theta.log2()], &[1])
}

/// The pipelines one model can launch. Which one a request hits depends on the batch: the
/// padded layout, or the packed layout with its `unpack` index below or at MLX's `constant`
/// limit. The load-time check runs every one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    /// No packing: the kernel reads the padded rows in place.
    Unpacked,
    /// Packed, and `unpack` has fewer than [`MLX_CONSTANT_LIMIT`] elements.
    PackedConstant,
    /// Packed, and `unpack` has at least [`MLX_CONSTANT_LIMIT`] elements.
    PackedDevice,
}

impl Variant {
    const ALL: [Variant; 3] = [Variant::Unpacked, Variant::PackedConstant, Variant::PackedDevice];

    /// The pipeline a launch with this `unpack` input hits, by MLX's rule.
    fn of(unpack: Option<&Array>) -> Variant {
        match unpack {
            None => Variant::Unpacked,
            Some(u) if u.size() < MLX_CONSTANT_LIMIT => Variant::PackedConstant,
            Some(_) => Variant::PackedDevice,
        }
    }
}

/// `pack` (padded index of every real token) and `unpack` (packed index of every padded
/// position, padding borrowing its row's token 0), as `Packing` builds them, for `n` rows of
/// `len` with the given real lengths.
fn packing(n: i32, len: i32, lens: &[i32]) -> (Vec<u32>, Vec<u32>) {
    assert_eq!(lens.len(), n as usize);
    let (mut pack, mut unpack) = (Vec::new(), Vec::new());
    for (r, &l) in lens.iter().enumerate() {
        let first = pack.len() as u32;
        for pos in 0..len {
            if pos < l {
                unpack.push(pack.len() as u32);
                pack.push((r as i32 * len + pos) as u32);
            } else {
                unpack.push(first);
            }
        }
    }
    (pack, unpack)
}

/// One batch of the load-time check: its pipeline, its shape, the real length of every row
/// (ignored for the padded layout) and which rope base to use.
struct CheckCase {
    variant: Variant,
    n: i32,
    len: i32,
    lens: &'static [i32],
    local: bool,
}

impl CheckCase {
    /// The `unpack` index of this case, `None` for the padded layout.
    fn index(&self) -> Option<Array> {
        match self.variant {
            Variant::Unpacked => None,
            _ => {
                let (_, unpack) = packing(self.n, self.len, self.lens);
                Some(Array::from_slice(&unpack, &[unpack.len() as i32]))
            }
        }
    }
}

/// The batches the load-time check runs, one per [`Variant`]: 6 padded positions; 6 packed
/// positions with 5 real tokens (an index of 6 elements, `constant`); 10 packed positions with
/// 8 real tokens (an index of 10 elements, `device`). `load_check_covers_every_kernel_variant`
/// checks that each case hits the pipeline it is listed for.
const CHECK_CASES: [CheckCase; 3] = [
    CheckCase { variant: Variant::Unpacked, n: 2, len: 3, lens: &[3, 3], local: false },
    CheckCase { variant: Variant::PackedConstant, n: 2, len: 3, lens: &[3, 2], local: true },
    CheckCase { variant: Variant::PackedDevice, n: 2, len: 5, lens: &[5, 3], local: false },
];

/// The fused kernel for one model's head layout and rope bases.
pub(crate) struct SplitRope {
    kernel: MetalKernel,
    heads: i32,
    hd: i32,
    /// Placeholder for the `unpack` input when the batch is not packed; the kernel never reads
    /// it (`PACKED` is false), and MLX passes it in the `constant` address space.
    no_index: Array,
    /// The global and the local rope base, and their `lbase` inputs, indexed by the layer's
    /// `local` flag.
    theta: [f32; 2],
    lbase: [Array; 2],
}

impl SplitRope {
    /// Build the kernel and run every pipeline of [`Variant`] once on a small batch against
    /// [`mlx_path`], so that a kernel that does not compile, or does not reproduce the MLX ops
    /// on this MLX build, is an `Err` here at load and never inside a request. The run also
    /// compiles the pipelines before the first request. `hd` must be a multiple of 8 (the
    /// kernel's vector width times two halves).
    pub(crate) fn new(heads: usize, hd: usize, dtype: Dtype, global_theta: f32, local_theta: f32) -> Result<Self> {
        if hd == 0 || !hd.is_multiple_of(2 * PAIRS_PER_THREAD as usize) {
            return Err(Error::Config(format!("fuserope needs a head dim that is a multiple of 8, this model has {hd}")));
        }
        let kernel = MetalKernel::new("sys1_split_rope", &["qkv", "unpack", "dims", "lbase"], &["q", "k", "v"], SOURCE).lx()?;
        let lbase = [log2_base(global_theta), log2_base(local_theta)];
        transforms::eval([&lbase[0], &lbase[1]]).lx()?;
        let this = Self {
            kernel,
            heads: heads as i32,
            hd: hd as i32,
            no_index: Array::from_slice(&[0u32], &[1]),
            theta: [global_theta, local_theta],
            lbase,
        };
        this.self_check(dtype)?;
        Ok(this)
    }

    /// Every case of [`CHECK_CASES`] in `dtype`, each compared with [`mlx_path`] element for
    /// element, and each checked to hit the pipeline it is listed for.
    fn self_check(&self, dtype: Dtype) -> Result<()> {
        for v in Variant::ALL {
            if !CHECK_CASES.iter().any(|c| c.variant == v) {
                return Err(Error::Backend(format!("fuserope self-check: no case for the {v:?} launch")));
            }
        }
        let width = 3 * self.heads * self.hd;
        for case in &CHECK_CASES {
            let (n, len) = (case.n, case.len);
            let vals: Vec<f32> = (0..n * len * width).map(|i| ((i * 7919) % 2003) as f32 / 1001.0 - 1.0).collect();
            let padded = Array::from_slice(&vals, &[n, len, width]).as_dtype(dtype).lx()?;
            let unpack = case.index();
            if Variant::of(unpack.as_ref()) != case.variant {
                return Err(Error::Backend(format!("fuserope self-check: the {:?} case builds a {:?} launch", case.variant, Variant::of(unpack.as_ref()))));
            }
            // The rows the kernel reads and the padded layout the MLX ops read.
            let (rows, expanded) = match &unpack {
                None => (padded.clone(), padded),
                Some(unpack) => {
                    let (pack, _) = packing(n, len, case.lens);
                    let rows = padded.reshape(&[n * len, width]).lx()?.take_axis(Array::from_slice(&pack, &[pack.len() as i32]), 0).lx()?;
                    let expanded = rows.take_axis(unpack, 0).lx()?.reshape(&[n, len, width]).lx()?;
                    (rows, expanded)
                }
            };
            let dims = Array::from_slice(&[n, len], &[2]);
            let (q, k, v) = mlx_path(&expanded, n, len, self.heads, self.hd, self.theta[case.local as usize])?;
            let got = self.apply(&rows, unpack.as_ref(), &dims, case.local, n, len)?;
            for (name, want, have) in [("q", &q, &got.0), ("k", &k, &got.1), ("v", &v, &got.2)] {
                let same = ops::eq(want, have).lx()?.all(None).lx()?;
                if !same.try_item_exact::<bool>().map_err(|e| Error::Backend(e.to_string()))? {
                    return Err(Error::Backend(format!("fuserope self-check: the kernel's {name} differs from the MLX ops ({:?} launch)", case.variant)));
                }
            }
        }
        Ok(())
    }

    /// `qkv` as `[R, 3 * heads * hd]` rows (or `[n, len, 3 * heads * hd]`, reshaped for free)
    /// -> roped `q`, roped `k` and `v`, each `[n, heads, len, hd]`. `unpack` is the packing's
    /// `[n * len]` index when the rows are packed, `dims` is `[n, len]` i32, and `local` picks
    /// the layer's rope base.
    pub(crate) fn apply(&self, qkv: &Array, unpack: Option<&Array>, dims: &Array, local: bool, n: i32, len: i32) -> Result<(Array, Array, Array)> {
        let width = 3 * self.heads * self.hd;
        let rows = qkv.reshape(&[-1, width]).lx()?;
        let shape = [n, self.heads, len, self.hd];
        let threads = n * len * self.heads * (self.hd / 2 / PAIRS_PER_THREAD);
        let dtype = rows.dtype();
        let mut out = self
            .kernel
            .apply(
                &[&rows, unpack.unwrap_or(&self.no_index), dims, &self.lbase[local as usize]],
                &[(&shape, dtype), (&shape, dtype), (&shape, dtype)],
                &[
                    ("T", TemplateArg::Dtype(dtype)),
                    ("H", TemplateArg::Int(self.heads)),
                    ("D", TemplateArg::Int(self.hd)),
                    ("PACKED", TemplateArg::Bool(unpack.is_some())),
                ],
                [threads, 1, 1],
                [THREADGROUP, 1, 1],
            )
            .lx()?;
        let v = out.pop();
        let k = out.pop();
        let q = out.pop();
        match (q, k, v) {
            (Some(q), Some(k), Some(v)) => Ok((q, k, v)),
            _ => Err(Error::Backend("fuserope: the kernel returned fewer than 3 outputs".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::indexing::IndexOp;

    const THETA: f32 = 10_000.0;

    /// Deterministic values in about `[-2, 2]` with the spread of a real projection.
    fn values(count: usize) -> Vec<f32> {
        let mut x: u32 = 12345;
        (0..count)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (x >> 8) as f32 / (1u32 << 24) as f32 * 4.0 - 2.0
            })
            .collect()
    }

    /// Exact equality of two arrays, compared through f32 (lossless for f16).
    fn assert_same(what: &str, want: &Array, got: &Array) {
        assert_eq!(want.shape(), got.shape(), "{what}: shape");
        assert_eq!(want.dtype(), got.dtype(), "{what}: dtype");
        let w = want.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
        let g = got.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
        let (w, g) = (w.as_slice::<f32>(), g.as_slice::<f32>());
        let worst = w.iter().zip(g).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert_eq!(worst, 0.0, "{what}: max abs diff {worst}");
    }

    /// A batch of `n` rows padded to `len`, with the given real lengths: the packed rows, the
    /// packing's `unpack` index and the expanded padded layout the MLX reference reads.
    fn batch(n: i32, len: i32, lens: &[i32], width: i32, dtype: Dtype) -> (Array, Array, Array) {
        let padded = Array::from_slice(&values((n * len * width) as usize), &[n, len, width]).as_dtype(dtype).unwrap();
        let (pack, unpack) = packing(n, len, lens);
        let rows = padded.reshape(&[n * len, width]).unwrap();
        let packed = rows.take_axis(Array::from_slice(&pack, &[pack.len() as i32]), 0).unwrap();
        let unpack = Array::from_slice(&unpack, &[unpack.len() as i32]);
        let expanded = packed.take_axis(&unpack, 0).unwrap().reshape(&[n, len, width]).unwrap();
        (packed, unpack, expanded)
    }

    /// The kernel against the MLX chain, padded and packed, at several shapes: one row and
    /// many, lengths that are not multiples of anything, the model's 16 x 64 heads and two
    /// smaller head dims, both element types, a packed index of fewer than 8 elements (MLX's
    /// `constant` variant) and the longest position the published models allow (8,192).
    #[test]
    fn kernel_matches_the_mlx_ops() {
        // (n, len, real lengths, heads, head dim, dtype)
        let cases: Vec<(i32, i32, Vec<i32>, i32, i32, Dtype)> = vec![
            (1, 7, vec![7], 4, 64, Dtype::Float16),
            (3, 13, vec![13, 5, 1], 4, 64, Dtype::Float16),
            (2, 3, vec![3, 2], 2, 8, Dtype::Float16),
            (2, 5, vec![4, 5], 3, 32, Dtype::Float32),
            (10, 226, vec![226, 197, 150, 226, 64, 3, 99, 226, 180, 17], 16, 64, Dtype::Float16),
            (1, 8192, vec![8192], 16, 64, Dtype::Float16),
            (2, 8192, vec![8192, 4097], 2, 64, Dtype::Float16),
        ];
        let mut kernels: Vec<((i32, i32, Dtype), SplitRope)> = Vec::new();
        let mut seen = Vec::new();
        for (n, len, lens, heads, hd, dtype) in cases {
            let what = format!("n={n} len={len} lens={lens:?} heads={heads} hd={hd} {dtype:?}");
            let key = (heads, hd, dtype);
            if !kernels.iter().any(|(k, _)| *k == key) {
                kernels.push((key, SplitRope::new(heads as usize, hd as usize, dtype, THETA, THETA).unwrap()));
            }
            let kernel = &kernels.iter().find(|(k, _)| *k == key).unwrap().1;
            let width = 3 * heads * hd;
            let (packed, unpack, expanded) = batch(n, len, &lens, width, dtype);
            let dims = Array::from_slice(&[n, len], &[2]);
            // Padded layout: the expanded rows straight through, as a forward without `unpad`.
            let (q, k, v) = mlx_path(&expanded, n, len, heads, hd, THETA).unwrap();
            let (gq, gk, gv) = kernel.apply(&expanded, None, &dims, false, n, len).unwrap();
            assert_same(&format!("{what} padded q"), &q, &gq);
            assert_same(&format!("{what} padded k"), &k, &gk);
            assert_same(&format!("{what} padded v"), &v, &gv);
            seen.push(Variant::Unpacked);
            // Packed rows with the unpack index: the `unpad` forward.
            if packed.dim(0) < n * len {
                let (gq, gk, gv) = kernel.apply(&packed, Some(&unpack), &dims, false, n, len).unwrap();
                assert_same(&format!("{what} packed q"), &q, &gq);
                assert_same(&format!("{what} packed k"), &k, &gk);
                assert_same(&format!("{what} packed v"), &v, &gv);
                seen.push(Variant::of(Some(&unpack)));
            }
        }
        for v in Variant::ALL {
            assert!(seen.contains(&v), "no case launched {v:?}");
        }
    }

    /// Each load-time case hits the pipeline it is listed for under MLX's rule, and every
    /// pipeline is listed, so the check cannot silently stop covering one.
    #[test]
    fn load_check_covers_every_kernel_variant() {
        for v in Variant::ALL {
            let case = CHECK_CASES.iter().find(|c| c.variant == v).unwrap_or_else(|| panic!("{v:?} is not in the load check"));
            assert_eq!(Variant::of(case.index().as_ref()), v, "the {v:?} case builds another launch");
        }
        assert_eq!(Variant::of(Some(&Array::from_slice(&[0u32; 7], &[7]))), Variant::PackedConstant);
        assert_eq!(Variant::of(Some(&Array::from_slice(&[0u32; 8], &[8]))), Variant::PackedDevice);
    }

    /// The two rope bases of the published models, at a late position, stay exact, each
    /// through its own `lbase` input.
    #[test]
    fn both_rope_bases_match() {
        let (n, len, heads, hd) = (1, 1025, 2, 64);
        let kernel = SplitRope::new(heads as usize, hd as usize, Dtype::Float16, 10_000.0, 160_000.0).unwrap();
        let width = 3 * heads * hd;
        let (_, _, expanded) = batch(n, len, &[len], width, Dtype::Float16);
        let dims = Array::from_slice(&[n, len], &[2]);
        for (theta, local) in [(10_000.0f32, false), (160_000.0, true)] {
            let (q, k, _) = mlx_path(&expanded, n, len, heads, hd, theta).unwrap();
            let (gq, gk, _) = kernel.apply(&expanded, None, &dims, local, n, len).unwrap();
            assert_same(&format!("theta={theta} q"), &q, &gq);
            assert_same(&format!("theta={theta} k"), &k, &gk);
        }
    }

    /// A head dim that is not a multiple of 8 is refused at build time, with the number.
    #[test]
    fn odd_head_dim_is_a_config_error() {
        let err = SplitRope::new(4, 36, Dtype::Float16, THETA, THETA).err().expect("36 is not a multiple of 8");
        assert!(err.to_string().contains("multiple of 8") && err.to_string().contains("36"), "{err}");
        assert!(SplitRope::new(4, 0, Dtype::Float16, THETA, THETA).is_err());
    }

    /// The reference chain itself: `v` is the plain head split and `q`, `k` are rotated, so a
    /// test that compares the kernel to it compares against real rope output.
    #[test]
    fn mlx_path_rotates_q_and_k_only() {
        let (n, len, heads, hd) = (1, 4, 2, 8);
        let x = Array::from_slice(&values((n * len * 3 * heads * hd) as usize), &[n, len, 3 * heads * hd]);
        let (q, k, v) = mlx_path(&x, n, len, heads, hd, THETA).unwrap();
        let parts = x.split_equal(3, -1).unwrap();
        let heads_of = |p: &Array| p.reshape(&[n, len, heads, hd]).unwrap().transpose_axes(&[0, 2, 1, 3]).unwrap();
        assert_same("v", &heads_of(&parts[2]), &v);
        let moved = |a: &Array, b: &Array| !ops::eq(a, b).unwrap().all(None).unwrap().item_exact::<bool>();
        assert!(moved(&heads_of(&parts[0]), &q) && moved(&heads_of(&parts[1]), &k));
        // Position 0 is never rotated.
        assert_same("q at position 0", &heads_of(&parts[0]).index((.., .., ..1, ..)), &q.index((.., .., ..1, ..)));
    }
}
