# mlx-rs spike: results (2026-09-29)

This is step 1 of the plan in `SYNTHESIS.md`. It runs the typed-decisions model on mlx-rs in fp16 and checks it against the bar: p50 of 19 / 60 / 454 ms for 1 question at 128 tokens, 1 at 512 and 10 at 512, with p95 close to p50.

## Short answer

The Rust runtime in `sys1rust/runtime` now runs at p50 18.5 / 50 / 479 ms, with p95 10 to 19% above p50 and no spikes. Over 5 minutes it sustains 7.45 requests/s with no slowdown. The best 5-minute number in the bake-off was 4.83.

- One-question requests beat the bar.
- Ten questions at 512 tokens is 5% slower than the bar. That was measured on battery, while the bar was set on AC power.
- Two fixes did it. One fixes a bug in the Rust port. The other is an MLX setting.
- Python MLX with the same setting runs within about 10% of Rust. Rust brings a single binary with no Python, not a faster engine.

## Conditions

- The laptop was on battery for the whole spike, with a video call and a browser running. Load average was 2.4 to 7.5 at the start of each run.
- macOS already had swap in use: 7.6 GB at one check and 16.2 GB at another.
- Every comparison here is between runs from the same session. The bar comes from the AC-powered bake-off, so compare against it with care.
- All runs use the saved harness, the typed-decisions model at the pinned commit, and the same MLX build.
  - That build is libmlx 0.32.2 from the mlx wheel. Its metallib includes the M5 Neural Accelerator (NAX) kernels.
  - A 2048³ fp16 matmul from Rust reaches 11.7 TFLOP/s at best and 9.8 at the median.

## Finding 1: laya-r-mlx's fp16 path ran in fp32

The fork started 2 to 2.7x slower than Python laya-mlx on the same MLX binary. The per-op profile put all of the gap in the matrix multiplies.

- **Cause.** The port's GELU multiplies by `Array::from_f32(0.5)` and similar scalar arrays. MLX then promotes the fp16 activations to fp32. From the first MLP on, the residual stream and every matrix multiply run in fp32, and the fp16 weights are cast to fp32 again on every request.
- **Fix.** The `f16gelu` setting casts those scalars to the input dtype. On the short workload (69 to 92 tokens, one question) p50 drops from 32.3 to 12.0 ms, the same as Python laya-mlx fp16 (12.0).
- **Accuracy.** 1,498 of 1,500 answers on the correctness workload agree with the reference (99.9%). Both differences are near ties. Gold accuracy is 75.8% for both.

## Finding 2: the MLX spikes are the buffer cache filling RAM

MLX keeps freed GPU buffers in a cache so it can reuse them. By default the cache can grow to about the size of RAM. In the timing workload almost every request has a new sequence length, so buffer sizes rarely repeat and the cache keeps growing.

- In one pass of the timing workload the cache reached 23.4 GB on this 32 GB laptop.
- macOS then swapped, and requests waited on page-ins.
- The harness's `peak_rss` reported about 1 to 2 GB because Metal buffers do not count toward RSS. sys1-bench now logs MLX's active, cache and peak memory on every result line.

This explains what the bake-off saw:

- **Padding to 16 tokens cut the spikes from 11% to 2.5%.** Fewer distinct lengths means more buffer reuse and a smaller cache.
- **Compiled Python MLX spiked on 10.4% of requests in a 1-pass run and 22.7% in a 2-pass run.** The cache is bigger on the second pass.
- **The compiled mode lost throughput after its first 30 seconds** (6.8 down to 3.3 to 4.3 requests/s in the bake-off). With the cache capped it holds 6.6 to 6.8 requests/s for 5 minutes.

MPS and Core ML barely spiked in the bake-off (0 to 0.8%). PyTorch's MPS allocator releases cached buffers under memory pressure, which may be why. This spike did not test that.

The fix is one call, `set_cache_limit`. Timing workload, one pass, same binary:

| Rust fp16-fix | 1q @128 p50/p95 | 1q @512 p50/p95 | 10q @512 p50/p95 | requests over 2x median | largest MLX cache |
|---|---|---|---|---|---|
| default cache | 40 / 887 | 123 / 892 | 617 / 1423 | 24.2% | 23.4 GB |
| cache capped at 256 MiB | 18 / 20 | 51 / 60 | 474 / 586 | 0.0% | 0.3 GB |
| cache capped at 512 MiB | 19 / 22 | 50 / 55 | 484 / 547 | 0.0% | 0.6 GB |
| cache capped at 1024 MiB | 20 / 29 | 51 / 78 | 483 / 550 | 0.0% | 1.1 GB |

## Finding 3: once the cache is capped, the other planned fixes add nothing

The synthesis guessed that the spikes came from per-shape kernel specialization, and planned warmed fixed buckets and whole-graph compile. The cache cap removes the spikes without those. Each option below was tested with the 512 MiB cap, on the timing workload, for one pass:

| option | 1q @128 p50/p95 | 1q @512 p50/p95 | 10q @512 p50/p95 |
|---|---|---|---|
| cap only | 19 / 22 | 50 / 55 | 484 / 547 |
| + 2 GiB wired limit | 19 / 21 | 49 / 55 | 466 / 515 |
| + lengths padded to a multiple of 16 | 19 / 22 | 52 / 59 | 478 / 545 |
| + 12 length buckets, warmed at load for 1, 4 and 10 rows | 18 / 23 | 49 / 56 | 476 / 574 |

Warming the buckets makes the model load take 7.1 s instead of about 0.2 s.

In Python, compile gains 1 to 3% with the cap: laya-mlx fp16 against compiled fp16, both capped. The Rust port was not compiled.

## Results against the bar

Typed-decisions timing workload, interleaved shapes, 2 passes unless marked. Values are p50 / p95 / p99 in ms.

| runtime | stage | 1q @128 | 1q @512 | 10q @512 | over 2x median |
|---|---|---|---|---|---|
| bar (laya-mlx compiled fp16, bake-off, AC) | | 19.3 / 35 / 44 | 59.8 / 474 / 1695 | 453.5 / 2053 / 2284 | 9.2% |
| Rust fork as is | A | 117.5 / 845 / 906 | 135.5 / 1126 / 1797 | 1017 / 1218 / 1283 | 20.2% |
| Rust + f16gelu | B | 49.2 / 524 / 786 | 221 / 1207 / 1359 | 663 / 1155 / 1773 | 25.2% |
| **Rust mlx-fp16-fast** (f16gelu, 512 MiB cache, 2 GiB wired) | C | **18.5 / 22 / 23** | **50.2 / 59 / 59** | **479 / 525 / 544** | **0.0%** |
| Python laya-mlx fp16 | A | 30.8 / 606 / 770 | 88.5 / 1065 / 1221 | 561 / 769 / 1545 | 23.5% |
| Python laya-mlx compiled fp16, 1 pass | A | 19.3 / 133 / 488 | 52.8 / 734 / 994 | 474 / 1071 / 1254 | 10.4% |
| Python laya-mlx compiled fp16 | B | 32.8 / 371 / 456 | 120 / 2106 / 2269 | 659 / 3212 / 3809 | 22.7% |
| Python laya-mlx fp16, 512 MiB cache | C | 19.8 / 22 / 25 | 50.1 / 57 / 59 | 464 / 527 / 536 | 0.0% |
| Python laya-mlx compiled fp16, 512 MiB cache | C | 19.1 / 22 / 23 | 48.9 / 52 / 57 | 448 / 491 / 502 | 0.0% |

p95 within 10% of p50 was the MPS standard. mlx-fp16-fast is at 10 to 19%. Part of that spread is real length variation inside each shape. The 1q @128 requests are 149 to 197 tokens long, and the 1q @512 requests 510 to 618.

Other stages for mlx-fp16-fast:

- **5 minutes, sustained.**
  - Rust: 7.55, 7.45, 7.55, 7.45 and 7.18 requests/s by minute, with 0.04% spikes. p50/p95 was 19/21, 50/57 and 480/560 ms. The MLX cache stayed between 513 and 573 MiB.
  - Python laya-mlx compiled with the cap, right after: 6.63 to 6.80 requests/s, with 0.10% spikes. p50/p95 was 21/26, 54/72 and 537/631 ms.
- **Short workload** (69 to 92 tokens, one question): p50 12.9 ms. Python laya-mlx fp16 was 12.0 and compiled with the cap was 12.4.
- **Cold start.** Time from process start to first answer was 185, 196 and 307 ms for Rust against 245, 251 and 260 ms for Python. The model files were already in the OS file cache.

## Rust against Python MLX

With both fixes, Rust and Python MLX run the same MLX kernels at the same speed.

- The per-request cost outside the GPU is under 1.5 ms in Rust: tokenizing, building the graph and decoding.
- Python laya-mlx is about as lean.
- Neither is consistently ahead.
  - Python compiled was 6% faster at 10q @512 in the paired timing runs.
  - Rust was 11% ahead over the 5-minute runs, and it ran first.

The Rust runtime's value is packaging: one binary with no Python, ready to embed in a server or an app. That matches the synthesis ("Rust gives packaging, not speed").

## What is left

- **Ten questions at 512 tokens is limited by compute.** The median request pads to 6,110 tokens. At 0.75 to 0.85 GFLOP per token that is 4.6 to 5.2 TFLOP, or 390 to 440 ms at the best measured 11.7 TFLOP/s, against 479 ms measured. No runtime change can reach cbjev's 198 ms. That needs the packed model (step 3 in the synthesis).
- **One question at about 185 tokens** takes 18.5 ms against a compute floor of about 12 to 13 ms. Kernel work could gain at most about 1.5x.
- **The warm-up and bucket code** stays in the runtime as options, but the default does not need it.

## Files

- **Code.** `sys1rust/runtime`, commits 75454f9 (fp16 fix, settings, probe) and a30e807 (cache cap, buckets, `mlx-fp16-fast`).
  - Settings go through `BackendOptions::tuning`, or the `SYS1_MLX` environment variable (see `Knobs` in `crates/laya-mlx/src/lib.rs`).
  - `sys1-probe` gives steady-state latency per shape and raw MLX microbenchmarks.
- **Results.** `raw/spike/bench/results/` holds stages A, B and C as JSONL. Stage B's Rust runs say `9b44075-dirty`, which matches 75454f9 apart from comments. `raw/spike/explore/` holds the single-pass option runs.
- **Scripts.** `raw/spike/scripts/` has the stage runners, `analyze.py`, `quick.py`, `sus.py`, the Python probes, the stage logs, and the Python adapter with the `-c512` variants added for this spike.
