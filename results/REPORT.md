# System 1 runtime bake-off on a base M5

Date: 2026-09-29. Machine: MacBook Pro 14 (Mac17,2), base Apple M5, 10 CPU cores, 32 GB, macOS 26.2, on wall power.

13 local runtimes answered the same `/v1/systemone` requests. They ran 227 timed runs between 00:58 and 07:22 PT, one at a time, with `powermetrics` logging the whole time. A run's speed counts only if its answers match upstream Laya's fp32 CPU answers on at least 99% of questions. All 227 runs exited cleanly, none timed out, and none started under heavy load (1-minute load 0.8 to 6.6, median 3.4).

Full tables are in `tables.md`, per-run data in `summary.json`, and raw events in `raw/`.

## What this means for building our runtime

1. **The fastest existing path for laya-typed-decisions is MLX fp16 with compiled graphs and inputs padded to a multiple of 16 tokens.** This is the `laya-mlx` library's optimize mode. It answers one question at 128 tokens in 19 ms, one at 512 tokens in 60 ms, and ten at 512 tokens in 454 ms, with 100% agreement. That is 1.8x faster than upstream Laya on PyTorch MPS fp16 and 3.3x faster than on MPS fp32. None of the servers tested ships this mode. laya-apple's HTTP server, the best ready-made one, runs plain MLX and is about 1.3x slower.
2. **MLX wins the median but loses sustained throughput, because of latency spikes.** When input lengths vary from request to request, 7 to 21% of MLX requests take more than twice their shape's median, and the worst take 12 to 143x. PyTorch MPS and Core ML have almost none (0 to 0.8%).
   - Over 5 minutes, upstream Laya on MPS fp16 (in-process) sustains 4.83 requests/s. The MLX runtimes sustain 3.9 to 4.2.
   - The compiled MLX mode also slows down under sustained load. It does 6.8 requests/s in its first 30 seconds and 3.3 to 4.3 afterwards, with no thermal throttling.
   - Padding to 16 tokens cuts the spikes from 11% to 2.5% on a shape-grouped workload, so the spikes follow the number of distinct input lengths. Why throughput drops after the first pass is still unexplained.
3. **On a base M5, the Neural Engine no longer wins on latency.** For one short question (66 to 92 tokens):
   - typed-decisions: 12.4 ms on the ANE, and 12.4 ms on MLX fp16.
   - multilingual: 4.8 to 5.1 ms on the ANE, and 5.2 to 5.5 ms on MLX.

   At every Core ML bucket from 256 tokens up, the GPU beats the ANE:

   | bucket | GPU | ANE |
   |---|---|---|
   | 256 | 7.4 ms | 10.2 ms |
   | 512 | 13.8 ms | 30.6 ms |
   | 1024 | 30.6 ms | 100 ms |

   The ANE's only remaining advantage is power. It draws 8 to 10.5 W against 12 to 28 W on the GPU. For a short typed-decisions question that is 118 mJ against 160 mJ on MLX fp16, and for multilingual there is no saving. An earlier claim of a 2.5x ANE lead (base M4) does not hold on the M5.
4. **On multilingual, MLX's default padded batch roughly doubles the per-question cost.** Two multilingual paths tie for fastest per question:
   - fixed-shape Core ML on the GPU, at 7.4 ms up to 240 tokens, 13.8 ms up to 496 and 30.6 ms up to 1008
   - MLX fp16 with one pass per question, at 7.9 ms at 128 tokens, 11.7 ms at 256 and 22.9 ms at 512

   Both are about 2x faster than MLX's padded batch (17.1, 22.9 and 44.2 ms), which is laya-mlx's and laya-apple's default. On multilingual, one pass per question beats the padded batch at all 12 shapes. On typed-decisions the result is mixed. One pass per question is faster for one question at 128 tokens (18.6 against 30.5 ms) but slower for ten questions at 512 tokens (1248 against 556 ms).

   Core ML buckets run one question at a time, so ten questions take 10x as long. laya.cpp's batched Core ML buckets returned near-flat probabilities on GPU and ANE (38 to 42% agreement). The Core ML comparison has not been run for the 421M typed-decisions checkpoint, because nobody has converted it to fixed Core ML buckets.
5. **The biggest measured gain comes from answering all questions in one encoder pass.**
   - Going from 1 to 10 questions multiplies latency by about 10x for runtimes that run one pass per question, and by 8 to 9x for MLX's padded batch.
   - Stock `laya serve` batches the questions on MPS and gets 4.5x.
   - cbjev packs all the questions into one sequence and gets 3.0x. It answers ten questions at 512 tokens in 198 ms in fp16, against 454 ms for the best Laya runtime.
   - cbjev is also slightly more accurate on held-out typed-decisions rows: 78.7% against Laya's 76.9% on the timing rows, and 76.2% against 73.5% on the correctness rows.

   cbjev is GPL-3.0, so it is a design reference rather than a dependency. The packing needs a model trained for it.
6. **Cross-request batching and concurrency buy nothing on this laptop.** Every HTTP server did 0.93 to 1.03x the requests/s with 4 clients that it did with 1, so one request already fills the GPU. sys1's cross-request batching lost 23% (1.32 down to 1.01 requests/s).
7. **Use fp16 on the GPU. Avoid bf16, and avoid 8-bit Core ML weights on the GPU.**
   - MLX fp16 keeps 100% agreement on both checkpoints and is 2.0 to 2.5x faster than true fp32.
   - MLX's default fp32 on M5 (TF32) is only 1.6x faster and drifts more.
   - bf16 broke agreement on PyTorch MPS (96.1% typed-decisions, 90.6% multilingual) and on sys1 (95.5% and 90.8%).
   - FluidInference's 8-bit weights run at the same speed as fp16 with 99.4 to 99.7% agreement. On the GPU they push memory to 11 to 17 GB, against 150 to 190 MB on the ANE.
8. **MLX runtimes start fast and use modest memory.**
   - Start to first answer (OS file cache warm): laya-mlx 0.25 s, Core ML 0.3 to 0.7 s, PyTorch 1.9 to 2.1 s, and the ANE paths 2.5 to 3.0 s with their compile cache present.
   - Building laya-apple's ANE artifacts the first time took 21 minutes for typed-decisions.
   - Peak memory: MLX 0.9 to 1.6 GB, PyTorch 2.6 to 2.8 GB, Core ML on the ANE under 0.2 GB.
9. **A better model is available now.** On held-out typed-decisions rows, OpenDecider-nano (Apache-2.0, 395M) scores 82.2% against Laya's 76.9% on the timing rows, and 78.2% against 73.5% on the correctness rows. On MPS fp16 it is a little faster than Laya (geo-mean 125 against 150 ms). decision-modernbert (150M) is the fastest model here, at 6.8 ms for one question at 128 tokens. On these questions it scores 34 to 36%, the same level as multilingual Laya.
10. **The Rust Metal engines are the slowest GPU paths.** By geo-mean, kime is 3.1 to 3.3x slower than the compiled MLX path, and sys1 is 5.1 to 6.1x slower. kime and sys1 gain 0 to 6% from fp16, which suggests their kernels do not use fp16 math. kime's response cache answers repeated identical requests in 0.2 ms. That is a cache, not a speed-up for new requests.

### Ready for real use?

For a local `/v1/systemone` server on a base M5, laya-apple's HTTP server is usable now. It answers with 100% agreement, takes 27 ms for one question at 128 tokens and 525 ms for ten questions at 512 tokens, and sustains about 4 requests/s. Stock `laya serve` is 1.3 to 2.6x slower per request and sustains 3.54 requests/s. It runs fp32 unless a request has at least 5 questions. Laya on MPS fp16 in-process sustains 4.83 requests/s, the best typed-decisions throughput here, but no server ships that setting.

Nothing here closes the gap our method targets. No existing runtime combines:

- the compiled fp16 MLX speed without its spikes and slowdown
- no padded-batch overhead per question
- one encoder pass for all questions

## Leaderboards

p50 latency in ms, taken from `timing.jsonl` (20 requests per shape, 2 repeats, shapes interleaved). The geo-mean is taken over all 12 shapes (64/128/256/512 state tokens by 1/4/10 questions). mJ/req is the whole-SoC CPU+GPU+ANE energy from `powermetrics` during the request; the idle baseline between runs was 4.5 W. Cold start runs from a new process to the first answer, with the OS file cache warm.

### typed-decisions (421M, the checkpoint worth serving)

| runtime | variant | agreement | 1q @128 | 1q @512 | 10q @512 | geo-mean | 5-min req/s | mJ/req | peak RSS MiB | cold start s |
|---|---|---|---|---|---|---|---|---|---|---|
| laya-mlx | mlx-fp16-opt (compile, pad to 16) | 100% | 19.3 | 59.8 | 454 | 85.1 | 4.11 | 3687 | 968 | - |
| laya-apple | mlx | 100% | 24.0 | 58.9 | 518 | 101.9 | 3.90 | 4497 | 1164 | 1.04 |
| laya-apple | http (server) | 100% | 26.9 | 61.6 | 525 | 107.9 | 3.95 (120 s) | 4735 | 1438 | - |
| laya-mlx | mlx-fp16 | 100% | 30.5 | 62.4 | 556 | 120.7 | 4.16 | 5116 | 932 | 0.25 |
| jevalaya (Rust + MLX) | mlx | 100% | 30.2 | 70.5 | 542 | 124.6 | 3.86 | 5156 | 1575 | 0.84 |
| laya-upstream | mps-fp16 | 99.8% | 45.4 | 87.6 | 714 | 149.8 | 4.83 | 4837 | 2780 | 1.91 |
| laya-upstream | http (stock `laya serve`) | 99.8% | 53.9 | 157.6 | 708 | 213.2 | 3.54 (120 s) | 6662 | 2792 | - |
| kime (Rust Metal) | metal-fp16-http | 100% | 57.2 | 179.1 | 1655 | 277.8 | 2.27 | 12060 | 2636 | 0.99 |
| laya-upstream | mps-fp32 | 100% | 54.1 | 157.7 | 1679 | 280.6 | - | 12081 | 2781 | - |
| sys1 (Rust candle Metal) | metal-fp16 | 99.8% | 72.3 | 225.6 | 2856 | 434.7 | 1.31 | 18473 | 1691 | 0.33 |

Ollaya has no typed-decisions model on MLX. Its CPU build is in the CPU table below.

### multilingual (322M mmBERT)

| runtime | variant | agreement | 1q @128 | 1q @512 | 10q @512 | geo-mean | mix mean ms | mJ/req |
|---|---|---|---|---|---|---|---|---|
| laya-mlx | mlx-fp16-opt | 100% | 8.4 | 27.5 | 230 | 35.8 | 95.7 | 1782 |
| laya-mlx | mlx-fp16-perq (one pass per question) | 100% | 7.9 | 22.9 | 234 | 38.9 | 63.3 | 1529 |
| laya-apple | http | 100% | 12.4 | 32.8 | 278 | 51.0 | 166.0 | 2677 |
| jevalaya | router (ANE + MLX) | 100% | 11.3 | 47.3 | 348 | 59.1 | 177.7 | 2937 |
| laya-upstream | mps-fp16 | 99.5% | 20.4 | 45.6 | 411 | 71.5 | 107.6 | 2361 |
| fluid-coreml | fluiduse-routing (FluidUse's router) | 99.6% | 10.9 | 101.4 | 1013 | 85.1 | 191.7 | 1722 |
| kime | metal-fp32 | 100% | 21.6 | 82.3 | 823 | 110.6 | 191.6 | 5709 |
| ollaya | mlx | 100% | 20.5 | 80.9 | 930 | 121.5 | 220.5 | 5296 |
| sys1 | metal-fp16 | 99.5% | 34.7 | 133.7 | 1749 | 216.7 | 406.1 | 10552 |

The mix mean is the average over the whole interleaved workload, which is what a busy server sees. For multilingual, one pass per question on MLX has the lowest mix mean, because it pads nothing and spikes less.

### Other models (different weights, scored against gold)

| model | runtime | 1q @128 | 1q @512 | 10q @512 | geo-mean | 5-min req/s | typed-decisions test-row accuracy (timing / correctness) |
|---|---|---|---|---|---|---|---|
| decision-modernbert-base (150M, CC BY-NC) | Core ML GPU | 6.8 | 24.2 | 242 | 36.7 | 16.13 | 36.4% / 34.3% |
| cbjev-multilingual (GPL-3.0) | MPS fp16 | 13.7 | 36.3 | 107 | 40.9 | - | 68.9% / 67.6% |
| cbjev (421M, GPL-3.0) | MPS fp16 | 28.3 | 71.5 | 198 | 84.5 | 10.14 | 78.7% / 76.2% |
| OpenDecider-nano (395M, Apache-2.0) | MPS fp16 | 26.7 | 68.7 | 708 | 125.0 | 5.19 | 82.2% / 78.2% |
| laya-typed-decisions, for comparison | reference | | | | | | 76.9% / 73.5% |
| laya-multilingual, for comparison | reference | | | | | | 36.4% / 35.3% |

Accuracy uses only the typed-decisions test split (225 and 408 gold questions). The train split is in the training data of laya-typed-decisions, cbjev and OpenDecider-nano, so all-row accuracy flatters them. cbjev also trained on the support-ticket queue question. See "Gold accuracy by source and split" in `tables.md`.

## Method experiments

### Precision (agreement and speed against each runtime's own fp32)

| runtime | typed-decisions | multilingual |
|---|---|---|
| MLX fp16 (laya-mlx) | 100%, 2.50x | 100%, 2.05x |
| MLX fp16 compiled, pad to 16 | 100%, 3.55x | 100%, 3.77x |
| MLX bf16 | 99.7%, 2.67x | 99.0%, 2.27x |
| MLX fp32 default on M5 (TF32) | 99.9%, 1.60x | 99.8%, 1.56x |
| PyTorch MPS fp16 | 99.8%, 1.87x | 99.5%, 1.69x |
| PyTorch MPS bf16 | 96.1% (fails), 1.92x | 90.6% (fails), 1.71x |
| sys1 fp16 / bf16 | 99.8%, 1.06x / 95.5% (fails) | 99.5%, 1.04x / 90.8% (fails) |
| kime fp16 | 100%, 1.00x | 100%, 0.97x |
| Core ML 8-bit weights (fluid e8) | - | 99.4 to 99.7%, same speed; 11 to 17 GB RSS on GPU |
| CPU INT8: kime / upstream ONNX | 97.5% (fails) / 99.2% | - |

On M5, MLX's TF32 fp32 drifts more than fp16 does (max probability drift 0.089 against 0.014 on multilingual).

### ANE vs GPU vs all compute units, per Core ML bucket (fluid-coreml, multilingual, p50 ms for one question)

The 128 row comes from the short workload, because no timing row fits it. The other rows come from timing rows.

| bucket (max tokens) | ANE | GPU | all |
|---|---|---|---|
| 128 (112) | 5.0 | 6.2 | 5.6 |
| 256 (240) | 10.2 | 7.4 | 10.7 |
| 512 (496) | 30.6 | 13.8 | 31.1 |
| 1024 (1008) | 100.0 | 30.6 | 100.8 |

- The crossover is between 128 and 256 tokens.
- With "all" compute units, Core ML picks the ANE. FluidUse's router sends inputs over 128 tokens to "all", so on this machine it runs them 2 to 3x slower than the GPU would.
- Bucket latency does not depend on input length within a bucket, because the input is padded. It grows linearly with the number of questions.
- decision-modernbert shows the same pattern: at its 512 bucket the GPU takes 11.9 ms and the ANE 27.9 ms.
- The ANE paths only accept short rows (laya-apple ≤128 and jevalaya ≤96 tokens for the whole sequence), so none of the timing rows fit them. laya-apple's `auto` router sent every timing request to MLX.

### Multi-question handling (p50 ms at 512 state tokens)

| runtime | 1 question | 4 | 10 | 10q / 1q |
|---|---|---|---|---|
| laya-upstream in-process, one pass per question (typed-decisions) | 157.7 | 637.4 | 1679 | 10.6x |
| laya-upstream `laya serve`, questions batched on MPS | 157.6 | 636.5 | 708 | 4.5x |
| laya-mlx padded batch, fp16 | 62.4 | 267.7 | 556 | 8.9x |
| laya-mlx one pass per question, fp16 | 68.0 | 328.2 | 1248 | 18.4x |
| cbjev packed, fp32 | 186.7 | 292.1 | 563 | 3.0x |
| cbjev one pass per question, fp32 | 185.4 | 691.8 | 1750 | 9.4x |
| cbjev one pass per question, batched | 185.9 | 749.8 | 2382 | 12.8x |
| Core ML buckets (fluid L1024 GPU, multilingual) | 30.7 | 122.8 | 308 | 10.0x |

- On typed-decisions, MLX's padded batch wins once there are several questions: 556 against 1248 ms for ten questions at 512 tokens.
- On multilingual it is slower than one pass per question at every shape.
- It is also slower for a single short question: 30.5 against 18.6 ms at 128 tokens on typed-decisions. That suggests a fixed cost in its padding path.

### MLX latency spikes

This is the share of requests slower than 2x the median for their shape:

| workload | laya-mlx fp16 | laya-mlx fp16-opt | laya-apple mlx | jevalaya mlx | PyTorch MPS fp16 |
|---|---|---|---|---|---|
| interleaved shapes (timing) | 18.1% | 9.2% (one pass) | 13.3% | 16.9% | 0.2% |
| grouped shapes (correctness) | 11.0% | 2.5% | 8.5% | 9.2% | 0.0% |

With shapes grouped, MLX's p50 at 128 tokens and one question drops from 30.5 to 17.6 ms. So part of MLX's cost comes from switching between very different lengths.

### Concurrency (HTTP servers, 2 minutes each)

| server | 1 client req/s | 4 clients req/s | ratio |
|---|---|---|---|
| ollaya mlx (multilingual) | 4.67 | 4.66 | 1.00 |
| laya-apple http | 3.95 | 3.98 | 1.01 |
| jevalaya mlx | 3.99 | 3.73 | 0.93 |
| laya-upstream `laya serve` | 3.54 | 3.54 | 1.00 |
| kime metal-fp16-http | 2.27 | 2.34 | 1.03 |
| sys1 metal-fp16 (cross-request batching) | 1.32 | 1.01 | 0.77 |

### CPU paths (typed-decisions, 2 requests per shape, indicative only)

| runtime | agreement | 1q @128 | 10q @512 |
|---|---|---|---|
| ollaya ort-cpu | 100% | 117 | 4021 |
| laya-upstream ONNX INT8 | 99.2% | 112 | 4459 |
| laya-upstream PyTorch fp32 (the reference) | 100% | 170 | 3773 |
| kime cpu-fp32 / cpu-int8 | 100% / 97.5% (fails) | 192 / 117 | 4768 / 2915 |
| sys1 cpu-fp32 | 100% | 221 | 6430 |
| cbjev packed (different model) | - | 187 | 1483 |
| laya.cpp cpu-fp32 | 100% | 1012 | 31117 |

The best CPU paths are about 6x slower than the best GPU path for one question at 128 tokens (112 against 19 ms) and 8x slower for ten questions at 512 tokens. Background daemons (mediaanalysisd, Spotlight) ran throughout and affect CPU runs more than GPU runs.

## What did not work

- **laya.cpp's Core ML path.** Batched buckets gave near-flat probabilities on GPU and ANE, with 38 to 42% agreement. Batch 1 is exact but takes 270 ms for a short question. It is excluded from the rankings. laya.cpp has no ggml Metal path for this model.
- **ONNX Runtime's Core ML execution provider.** The MLProgram format fails to load. The NeuralNetwork format answers only single-question requests: the ANE takes 145 ms and the GPU 254 ms, against 58 ms on the plain CPU, and it uses 5 GB of RAM.
- **The ANE for anything but short rows.** None of the timing rows fit a 128-token bucket, because the question text counts toward the limit.
- **Ollaya on MLX for typed-decisions.** Ollaya only offers the English base and multilingual checkpoints on MLX.

## Caveats

- **One machine, one night.** Background system daemons kept the 1-minute load average between 0.8 and 6.6. All runs were flagged quiet (load at or below 7) and ran on AC power with nominal thermal pressure.
- **Energy is whole-SoC CPU+GPU+ANE power from `powermetrics` at 1-second samples.** It is not wall power, and it includes the 4.5 W idle baseline. For requests under a second, the per-request energy is the average power of that second.
- **laya-mlx fp16-opt ran with 1 repeat on the timing set.** Its tokenized-prefix cache would otherwise hit on the repeat. Its 5-minute run loops the workload, so later passes get cache hits for tokenization only. Even so, its throughput fell after the first pass.
- **kime's cache-on variant is reported separately.** Its second repeat is all cache hits.
- **Cold starts assume a warm OS file cache.** Clearing it needs root. The Core ML and ANE numbers also assume the compile cache already exists.
- **The CPU numbers are indicative.** They come from 2 requests per shape.
- **Accuracy is not a Decision Index.** It is gold agreement on this workload's questions (typed-decisions workflows plus support tickets), not the model cards' own benchmarks.
- **The English base checkpoint was not timed.** It has the same architecture as typed-decisions.

## Files

- `tables.md`: every generated table, including per-shape latency for all 227 runs, the bucket grids, tails, gold by split and run conditions.
- `summary.json`: per-run summaries, with p50/p95/p99 per shape, energy, RSS, load and driver data.
- `raw/results/`: adapter event logs, `/usr/bin/time -l` output and run metadata for every run.
- `raw/bench/`: the plan, status log, harness, timed driver and matrices, `powermetrics` logs, workloads, references, adapters and each contender's NOTES.md.

## Cleanup (done 2026-09-29 08:56 PT)

- `bench/teardown.sh --yes` deleted the 65 GB `bench/` folder, the four Core ML compile caches in the per-user Caches folder (about 20 GB), the two rustup toolchains and the temp files, and stopped caffeinate. 18 more empty onnxruntime `mat-debug-*.log` files from the benchmark window were deleted by hand.
- Before deleting, the last small files were copied here: the final STATUS.md, laya-mlx's `tools/`, jevalaya's `run-configs/` and laya-mlx's research docs (`raw/laya-mlx-docs/`). No contender source checkout had local changes.
- The power runs used a temporary passwordless sudo rule for `powermetrics`, which the operator removes after the runs.
