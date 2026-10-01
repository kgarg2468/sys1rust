// Body of the `sys1_split_rope` kernel; MLX writes the signature around it (see
// metal_kernels.rs) with the template arguments `T` (the element type, half or float), `H`
// (heads), `D` (head dim) and `PACKED`.
//
// Inputs: qkv [R, 3*H*D], the encoder's qkv projection rows, where R is n*len or, with
// PACKED, the packed token count; unpack [n*len] u32, the packed index of every padded
// position (a 1-element placeholder without PACKED, never read); dims [n, len] i32; lbase [1]
// f32, log2 of the rope base. Outputs: q and k rotated, v copied, each [n, H, len, D].
//
// One thread per (b, p, h) and 4 pairs (i, i + D/2) with i = 4m, so D must be a multiple of
// 8: then every vec<T, 4> load and store is inside its row and aligned (4 elements from a
// row start that is itself a multiple of 4 elements). The grid is exactly the work, and the
// row guard below covers any thread past it. The angle is MLX's rope arithmetic (exp2 of the
// log2 base, fast cos and sin, in float), so the result matches `fast::rope` bit for bit.
using V = metal::vec<T, 4>;
const int LEN = dims[1];
const int W = qkv_shape[1];
constexpr int HALF = D / 2;
constexpr int PAIRS = HALF / 4;
const int g = int(thread_position_in_grid.x);
const int i = (g % PAIRS) * 4;
const int h = (g / PAIRS) % H;
const int p = (g / PAIRS / H) % LEN;
const int b = g / PAIRS / H / LEN;
if (b >= dims[0]) return;
const int slot = b * LEN + p;
const int srow = PACKED ? int(unpack[slot]) : slot;
const size_t src = (size_t)srow * W + h * D + i;
const size_t dst = (((size_t)b * H + h) * LEN + p) * D + i;
float c[4];
float s[4];
#pragma clang loop unroll(full)
for (int j = 0; j < 4; j++) {
    const float inv_freq = metal::exp2(-(float(i + j) / float(HALF)) * lbase[0]);
    const float theta = float(p) * inv_freq;
    c[j] = metal::fast::cos(theta);
    s[j] = metal::fast::sin(theta);
}
{
    const V x1 = *(const device V*)(qkv + src);
    const V x2 = *(const device V*)(qkv + src + HALF);
    V r1, r2;
    #pragma clang loop unroll(full)
    for (int j = 0; j < 4; j++) {
        r1[j] = T(float(x1[j]) * c[j] - float(x2[j]) * s[j]);
        r2[j] = T(float(x1[j]) * s[j] + float(x2[j]) * c[j]);
    }
    *(device V*)(q + dst) = r1;
    *(device V*)(q + dst + HALF) = r2;
}
{
    const V x1 = *(const device V*)(qkv + src + H * D);
    const V x2 = *(const device V*)(qkv + src + H * D + HALF);
    V r1, r2;
    #pragma clang loop unroll(full)
    for (int j = 0; j < 4; j++) {
        r1[j] = T(float(x1[j]) * c[j] - float(x2[j]) * s[j]);
        r2[j] = T(float(x1[j]) * s[j] + float(x2[j]) * c[j]);
    }
    *(device V*)(k + dst) = r1;
    *(device V*)(k + dst + HALF) = r2;
}
*(device V*)(v + dst) = *(const device V*)(qkv + src + 2 * H * D);
*(device V*)(v + dst + HALF) = *(const device V*)(qkv + src + 2 * H * D + HALF);
