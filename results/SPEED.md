# Faster than Python MLX: results (2026-09-29)

`SERVER.md` found that Python laya-mlx, with MLX's buffer cache capped, runs as fast as the Rust runtime. Both call the same MLX kernels. This round looked for speed that Python laya-mlx does not have, without changing the answers.

## Short answer

- Three exact changes make the Rust runtime 1.12x faster than Python laya-mlx at its fastest (compiled, cache capped) on the timing workload. The runtime is faster on all 12 shapes, by 4 to 36%. Answers do not change: 1,498 of 1,500 agree with the reference, as before.
- On short single questions (69 to 92 tokens) the two tie. The Rust p50 was 11.6 to 12.2 ms and Python's 12.6 to 13.1 over three alternating runs.
- The three changes:
  - dense attention in the local layers
  - the last decision-head layer computed only where the scorer reads it
  - no computing on padding
- MLX's matmuls already run at 13 to 14 TFLOP/s on most of the model's shapes, about the published fp16 peak of this chip. Kernel settings, split-K and RoPE layout changes gained nothing.
- The one route to a much larger gain is int8 matmuls, which run about 2x fp16 on the M5 GPU. A simulation shows int8 on the MLP matmuls alone passes the 99% agreement gate, but narrowly (99.3%). It needs custom Metal kernels, because MLX has no int8 matmul.
- Round 2 (2026-09-30, below): one custom Metal kernel for the encoder's qkv split, RoPE and unpad expand. Bit-identical answers, 0.962 of the stage G default's time in a paired run, and 1.17x over Python laya-mlx in the two-round harness. It is the `sys1d` default. Int8 was measured with a real kernel and is not shipped: it fails the agreement gate on multilingual.

## Conditions

- AC power, load average 1.5 to 4.2. Runtime code is e758bfa plus the changes later committed as 0495800.
- Paired comparisons: `sys1-probe --ab` loads two engines in one process and alternates requests between them, so drift in the machine hits both. Separate runs on this laptop vary by about 5%, which is too much to see a 3% change.
- Harness comparisons against Python ran in two rounds in opposite order (stage E).

## Where the time goes

Cost per token is flat from 184 to 6,000 tokens (95 to 111 µs), so even one short question fills the GPU.

- **Matmuls.** At 10 rows of 568 tokens they take about 70% of the forward. Measured alone at the model's shapes, most run at 13 to 14 TFLOP/s. The exceptions are the two projections with a long inner dimension and 1,024 outputs (encoder MLP output, K 2624, and head FFN output, K 4096). They run at 6.7 to 10.7 TFLOP/s, because 1,024 outputs make few tiles for the GPU's 10 cores.
- **Everything else.** Local attention is 9%, RoPE with its copies 6%, GeGLU 6% and LayerNorms 5%.
- **Padding.** On multi-question requests every row pads to the longest. That is 29% of tokens at 10 questions over a 64-token state, 15% at 128, 11% at 256 and 6.4% at 512.

## What was tested

Paired A/B against the current default (`f16gelu,cache=512,wired=2048`), timing workload, 10 pairs per shape. Ratios below 1 are faster.

| change | setting | geo-mean time ratio | range over shapes | answers |
|---|---|---|---|---|
| dense local attention up to 1,024 tokens | `dense_upto=1024` | 0.968 | 0.91 to 1.01 | identical |
| last head layer only at scorer positions | `headprune` | 0.983 | 0.96 to 1.02 | max diff 0.0001 |
| no computing on padding | `unpad` | 0.953 | 0.79 to 1.02 | identical |
| all three | `dense_upto=1024,headprune,unpad` | **0.899** | 0.77 to 0.99 | max diff 0.0001, no answer changes |
| split rows into groups by length | `split=auto` | 0.970 (smoke) | | score and noul differ in the 4th decimal |
| one RoPE call for q and k | `rope1` (3 layouts) | 1.00 to 1.06 | | identical |
| split-K for the K 2624 and K 4096 projections | `splitk=2`, `splitk=4` | 1.00, 1.01 | | score differs in the 4th decimal |

- **Dense attention** wins at every length, including rows of 1,024 tokens (0.95 to 0.98). The chunked path was a leftover from laya-r-mlx. Python laya-mlx already uses dense masks, so this change only catches up with it.
- **Unpadding** keeps hidden states as one packed list of real tokens through the embeddings, LayerNorms, matmuls and GeGLU. It moves them into the padded layout only for attention. It pays most where padding is highest: 0.79 at 10 questions over a 64-token state, and about 1.0 at 512 tokens.
- **Head pruning.** The scorer reads only the CLS token and the option markers of the last head layer. That layer now runs its queries, output projection and FFN on those rows only. It saves about 2.8% of the FLOPs.
- **Split-K** was 1.3 to 2.2x faster on the two slow projections in a microbenchmark at 184 and 606 tokens, and slower from 2,048 up. In the model it gained nothing. The extra reshape, batched matmul and sum cost what the matmul saved.
- **MLX settings with no code change.** Raising or lowering the command-buffer limits (`MLX_MAX_OPS_PER_BUFFER`, `MLX_MAX_MB_PER_BUFFER`) and `MLX_METAL_FAST_SYNCH` stayed within noise. `MLX_METAL_GPU_ARCH=applegpu_g17s` selects the bigger chips' matmul tile. It was 3% faster at 606 tokens and 7% slower at 5,680.

## Against Python laya-mlx

Stage E, two rounds each, p50 ms. "Rust new" is `f16gelu,cache=512,wired=2048,dense_upto=1024,headprune,unpad`.

| shape | Python laya-mlx compiled, cache capped | Rust current default | Rust new | Python / Rust new |
|---|---|---|---|---|
| 1q, 64-token state | 21.0 | 18.9 | 19.1 | 1.10 |
| 4q, 64 | 62.1 | 58.3 | 50.8 | 1.22 |
| 10q, 64 | 144.0 | 132.5 | 105.6 | 1.36 |
| 1q, 128 | 18.9 | 17.3 | 17.6 | 1.08 |
| 4q, 128 | 68.1 | 61.1 | 61.1 | 1.12 |
| 10q, 128 | 164.6 | 150.2 | 142.0 | 1.16 |
| 1q, 256 | 28.5 | 28.7 | 26.8 | 1.06 |
| 4q, 256 | 103.6 | 105.3 | 98.9 | 1.05 |
| 10q, 256 | 258.2 | 268.1 | 235.2 | 1.10 |
| 1q, 512 | 49.4 | 48.2 | 45.6 | 1.08 |
| 4q, 512 | 173.2 | 174.3 | 166.1 | 1.04 |
| 10q, 512 | 453.2 | 453.0 | 422.1 | 1.07 |
| **geo-mean** | 83.7 | 80.3 | 74.9 | **1.12** |

- The mean request took 117 ms against Python's 130, so one client gets about 11% more requests per second.
- p95 at 10 questions over 512 tokens was 466 to 471 ms against Python's 496 to 518. No request in any run took over 2x its shape's median.
- The correctness run with the new settings agrees with the reference on 1,498 of 1,500 answers (99.9%), with gold accuracy 75.8%. Both match the current default.
- Short workload (40 requests, 15 passes, alternating runs): Rust new 11.64, 12.22 and 11.84 ms p50. Rust current default 11.82 and 11.68. Python 13.10 and 12.58. At about 80 tokens the forward is 11.1 to 11.4 ms of GPU time plus 0.2 ms to build the graph, in both.

## Int8 matmuls

MLX has no int8 × int8 matmul. Its quantized matmuls turn the weights back into fp16 first, so they cannot speed up matmuls that are limited by compute, as these are. Published measurements for the base M5 put int8 × int8 at 29.5 TOPS against 14.2 TFLOP/s for fp16.

To check accuracy before writing kernels, `fakeq_adapter.py` simulates int8 in Python laya-mlx. It rounds the weights to int8 per output channel and, for W8A8, the activations to int8 per token, then computes in fp16. Correctness workload, 1,500 answers:

| simulated | agreement | gold accuracy | gate (99%) |
|---|---|---|---|
| none (fp16) | 99.9% | 75.8% | pass |
| int8 weights, all 120 matmuls | 99.5% | 76.0% | pass |
| int8 weights and activations, all 120 matmuls | 98.5% | 75.9% | fail |
| int8 weights and activations, encoder MLP only (56 matmuls, 66% of matmul FLOPs) | 99.3% | 75.4% | pass |

Weight-only int8 passes but brings no speed on this GPU. The MLP-only W8A8 variant is the one that could pay off. If its matmuls ran at 1.8 to 2x, the forward would be about 1.2 to 1.3x faster. That estimate is not measured. It needs an int8 matmul kernel for the M5's matmul hardware, a kernel that turns activations into int8, and a check with more data than one correctness workload, given the thin margin.

## What the changes cost

- **Answers.** Nothing. The skipped work produced values nothing reads. Padding positions are masked out as keys and their outputs are dropped, and `score` reads the last head layer only at the CLS token and the option markers. Dense attention does more arithmetic than the chunked path, but in fewer, larger GPU calls, and its masked scores come out of the softmax as exact zeros. Head pruning moves probabilities by at most 0.0001, because matmuls with fewer rows can add in a different order in fp16. No answer changed.
- **Memory.** Nothing. On 1,024-token inputs (`long.jsonl`, battery), MLX peak memory was 1,671 MB with the new settings against 1,765 MB with the old ones, and the process footprint 2,379 MB against 2,458 MB. In stage E the footprint was 2,382 to 2,472 MB against 2,478 to 2,505 MB.
- **Speed on some shapes.** Unpadding alone was up to 2% slower where padding is low, because the copies in and out of the padded layout around attention cost more than the padding saved. With all three settings on, no shape was slower.
- **An assumption.** Head pruning is only correct while nothing reads the other positions of the last head layer. A future model or API feature that does, such as per-token outputs or embeddings, must turn it off. Since the review, the pruned output is its own type (`ScorerRows`), which only hands out the CLS and marker rows, so code that wants every position cannot read the wrong rows by mistake.

## Review, new default and the other Laya models (stage G)

A second agent reviewed the code (commit 0495800) line by line before the settings became the default.

- **One bug fixed.** The runtime kept one dense window mask per distinct input length and never freed them. With dense attention up to 1,024 tokens, a long-running server could collect about 700 MB of masks. It now keeps one mask for the longest length seen and uses its top-left corner for shorter inputs. A unit test checks that the corner equals a mask built for the shorter length.
- **Removed.** `split`, `rope1` and `splitk` gave no gain and are gone, about 600 lines. They stay in git history at 0495800.
- **New tests.** `crates/laya-mlx/tests/settings.rs` compares each setting, and all three together, against the old path on every cached Laya model. It covers:
  - 1, 2 and 20 options;
  - an input cut at the length limit;
  - heavy padding;
  - mixed answer types;
  - a request with no padding;
  - the smoke workload.
  All pass. The largest probability change is 0.0005 (multilingual, head pruning), with no answer changed.
- **Default.** `sys1d` now runs `f16gelu,cache=512,wired=2048,dense_upto=1024,headprune,unpad` (round 2 adds `fuserope`, below). `sys1-bench` has a new variant, `mlx-fp16-lean`, with the same settings; `mlx-fp16-fast` keeps its old meaning. The `http-fp16-fast` bench variant uses the server default, so it now runs the new settings.

All three published Laya models now run in the Rust runtime. Before this round only typed-decisions had been tried. Stage G, on the reviewed build, battery power:

| model | agreement with upstream reference | paired time, new / old (geo-mean) | range over shapes |
|---|---|---|---|
| typed-decisions | 1,498 of 1,500 on correctness, in-process and over `sys1d` HTTP (as before) | 0.900 | 0.78 to 0.99 |
| multilingual (mmBERT, 322M) | 100% on correctness, smoke and short | 0.915 | 0.82 to 0.99 |
| english (base Laya) | 100% on smoke, short and cold (no correctness reference); against the old settings, 100% on correctness | 0.904 | 0.77 to 1.00 |

- The reviewed build gives byte-identical answers to the pre-review build on the typed-decisions correctness workload.
- In the paired runs, `--check` counted 3 score mismatches on multilingual. It counts any change in a score's value, and these differ by at most 0.0002. The harness rounds scores before comparing, and there multilingual agrees 100%.
- On multilingual, one question at 128 tokens takes 7.7 ms.

## Round 2: the split, RoPE and unpad expand as one Metal kernel (2026-09-30)

Round 1 left RoPE and its copies at 6% of the forward. In every encoder layer MLX ran `split_equal`, three reshape copies into `[n, H, len, hd]`, two `fast::rope` launches and, with `unpad`, the gather that expands the packed rows. A research round timed custom Metal kernels through the C API behind `mx.fast.metal_kernel` (mlx-rs 0.32 has no binding for it; `metal_kernels.rs` is a small wrapper over the vendored mlx-sys). One kernel ships, as the setting `fuserope`.

- **What it does.** One launch per layer reads the qkv rows and writes roped q, roped k and v in `[n, H, len, hd]`. With `unpad` it reads each padded position's row through the packing's index, so the expand costs nothing extra. It keeps MLX's rope arithmetic (exp2 of the log2 base, computed on the host as f32, `fast::cos` and `fast::sin` in float), so its output equals `fast::rope`'s bit for bit. One thread handles 4 pairs with 4-wide loads and stores, so the head dim must be a multiple of 8; all three models have 64.
- **Gain.** Paired A/B (`sys1-probe --ab`) against the stage G default on typed-decisions, 12 timing shapes, 5 pairs each after 2 warmups, AC power: geo-mean 0.962, range 0.928 to 1.005. Single questions gain nothing (0.994 to 1.005); 4 and 10 questions gain 4 to 7%. Paired runs on multilingual and english with 1 warmup and 2 pairs per shape: 0.948 and 0.964.
- **Answers.** On all three models, `fuserope` alone is byte-identical to the plain path on every state of the equivalence suite (`tests/settings.rs`). `sys1-probe --check` of the stage G default against the new one on typed-decisions, multilingual and english: largest probability difference 0.000000, 0 of 60 answers changed, on each. The correctness run with the new default agrees with the reference on 1,498 of 1,500 answers (99.9%), gold accuracy 75.8%, as before.
- **macOS 14 and 15.** MLX compiles the kernel from source on the user's machine. The same `sys1d` binary with `DYLD_LIBRARY_PATH` set to the macOS 14, 15 and 26 builds of MLX 0.32.2 compiled the kernel and passed its load check on each, and the new default gave the same answers as the old one on each build (0 choice flips, largest probability difference 0.0000 over the 240 timing requests). The 14 and 15 builds differ from 26 by 0.0056 with either default, as measured before this round: they have no kernels for this chip's matmul hardware and run about 3x slower. On the 26 build the sum of per-shape p50 went from 1,315 to 1,218 ms (0.93); on 14 and 15 from 4,030 to 3,962 ms (0.98), since the kernel's share is smaller where the matmuls are 3x slower.
- **Fallback.** The decision is made at load, never in a request. laya-mlx builds the kernel and runs it once on a small packed and an unpacked batch against the MLX ops. If the kernel does not compile, does not match, the backend is CPU, or the head dim is not a multiple of 8, it prints one line to stderr and the forward runs the MLX ops.
- **Tried and ruled out.**
  - A merge-side kernel (attention output back to the packed `[T, d]` rows for the output projection, with the pack gather fused): 1.02 to 1.05x slower. MLX's attention already writes its output in `[n, len, H, hd]` memory order, so the transpose it was meant to replace is a free view.
  - 1 pair per thread instead of 4: no measured difference. The 4-wide variant ships.
  - Int8 matmuls, measured with a real kernel on the MLP input projection (int8 weights and activations): 1.39x over Python on typed-decisions, but agreement with the reference fell to 99.6% there and to 95.1% on multilingual, under the 99% gate. Not shipped. The answers-unchanged rule holds for every model, and the simulation of round 1 (99.3% on typed-decisions) did not predict the multilingual result.

Stage E style, two rounds in opposite order, p50 ms. "Old default" is the stage G default, "round 2 default" adds `fuserope`.

| shape | Python laya-mlx compiled, cache capped | Rust old default | Rust round 2 default | Python / Rust round 2 |
|---|---|---|---|---|
| 1q, 64-token state | 22.1 | 18.5 | 18.4 | 1.20 |
| 4q, 64 | 59.6 | 48.7 | 46.1 | 1.29 |
| 10q, 64 | 138.4 | 101.1 | 93.4 | 1.48 |
| 1q, 128 | 18.9 | 16.7 | 16.6 | 1.14 |
| 4q, 128 | 62.1 | 58.3 | 55.6 | 1.12 |
| 10q, 128 | 157.5 | 135.5 | 126.3 | 1.25 |
| 1q, 256 | 27.2 | 25.1 | 24.8 | 1.09 |
| 4q, 256 | 96.7 | 94.2 | 88.9 | 1.09 |
| 10q, 256 | 245.4 | 224.2 | 211.7 | 1.16 |
| 1q, 512 | 46.3 | 43.9 | 44.1 | 1.05 |
| 4q, 512 | 165.3 | 158.7 | 150.1 | 1.10 |
| 10q, 512 | 421.8 | 402.5 | 379.8 | 1.11 |
| **geo-mean** | 80.1 | 71.5 | 68.6 | **1.17** (range 1.05 to 1.48) |

The old default measured 1.12x over Python in the same session (range 1.03 to 1.37), as in stage E.

## What this means

- The lead over Python laya-mlx comes from doing less work, not from Rust. Python could copy head pruning (its research notes already list it) and, with more work, unpadding. Today no released runtime does either.
- The settings are reviewed, tested on all three Laya models, and the `sys1d` default.
- Closed after review: MLX used to compile the GeGLU step once per input shape, and with `unpad` the shape is the request's total token count, so each new count paid a trace and added a cache entry. The GeGLU trace is now shapeless (one trace for every shape; the split happens outside it). Paired A/B against the per-shape trace on the timing workload with the default settings: 1.001 on typed-decisions, 1.013 and 1.004 (sides swapped) on multilingual, identical answers.

## Files

- **Code.** `runtime/crates/laya-mlx/src/lib.rs` has the settings `dense_upto`, `headprune` and `unpad`. The experiments that did not help are at commit 0495800. `crates/laya-mlx/tests/settings.rs` has the equivalence tests. `runtime/crates/sys1-bench/src/bin/sys1-probe.rs` has `--ab SPEC_A SPEC_B` and `--check`.
- **Round 2.** `runtime/crates/laya-mlx/src/metal_kernels.rs` is the wrapper for MLX custom Metal kernels, `src/split_rope.rs` and `src/kernels/split_rope.metal` are the `fuserope` kernel with its load-time check and its tests against the MLX ops, and `tests/settings.rs` runs `fuserope` alone and with the three settings. The merge and int8 kernels were not merged.
- **Stage G.** `raw/speed/review/` holds the multilingual and english runs before the review (stage F) and after it (stage G), the paired speed log `stageG-ab.log`, and `stageF.sh`. Stage F ran the `sys1-bench` in the default target directory, not the build passed to it as `BIN_DIR`, because the sys1rust adapter re-sourced `bench/env.sh` and reset `CARGO_TARGET_DIR` (since fixed). Its results record `code_version` 0495800, which names the source tree, not the binary that ran; stage G ran on the reviewed build and is unaffected.
- **Results.** `raw/speed/bench/results/` holds stage E (timing, correctness and short) and the three alternating short runs. `raw/speed/fakeq/` holds the int8 simulation outputs.
- **Scripts.** `raw/speed/scripts/` has `stageE.sh` and its log, `gemm_shapes.py` (matmul speed at the model's shapes), `fakeq_adapter.py`, `long.jsonl` (1,024-token rows), and the per-op profile taken on battery (`ops-battery.txt`).
