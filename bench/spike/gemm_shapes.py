"""fp16 GEMM throughput at the typed-decisions model's shapes, as the Rust backend runs them.

Weights are stored [out, in] and used through a transposed view (x @ W.T), like laya-mlx's
`Linear`. Each timing evaluates REPS independent matmuls in one `mx.eval`, so launch and sync
costs are spread out. Run under different MLX_METAL_GPU_ARCH values to compare GEMM tiles.
"""
import os
import sys
import time

import mlx.core as mx

REPS = 10
ITERS = 15
# (name, K, N): encoder Wqkv, Wo, Wi (GeGLU in), Wo2 (GeGLU out); head FFN.
LAYERS = [("wqkv", 1024, 3072), ("wo", 1024, 1024), ("wi", 1024, 5248), ("wo2", 2624, 1024),
          ("head_l1", 1024, 4096), ("head_l2", 4096, 1024)]
MS = [int(m) for m in sys.argv[1:]] or [184, 606, 2048, 5680]


def bench(m, k, n):
    x = mx.random.normal((1, m, k)).astype(mx.float16)
    w = mx.random.normal((n, k)).astype(mx.float16)
    wt = w.T
    mx.eval(x, w)
    for _ in range(3):
        mx.eval([x @ wt for _ in range(REPS)])
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        mx.eval([x @ wt for _ in range(REPS)])
        ts.append((time.perf_counter() - t0) / REPS)
    ts.sort()
    t = ts[len(ts) // 2]
    return t * 1e3, 2 * m * k * n / t / 1e12


arch = os.environ.get("MLX_METAL_GPU_ARCH", "default")
tot = {}
for m in MS:
    row = []
    for name, k, n in LAYERS:
        ms, tf = bench(m, k, n)
        tot[m] = tot.get(m, 0) + (ms if not name.startswith("head") else 0)
        row.append("%s %.3fms %.1fTF" % (name, ms, tf))
    print("arch=%s M=%d | %s | encoder-layer gemm sum %.3f ms" % (arch, m, " | ".join(row), tot[m]), flush=True)
