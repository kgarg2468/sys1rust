# Research summary (2026-09-28, re-run on base M5 laptop)

Four Fable 5.1 researchers re-checked the earlier research notes (not in this repo) against live sources. Full reports: `report-a-rust-ecosystem.md`, `report-b-models-and-api.md`, `report-c-apple-silicon.md`, `report-d-runtimes-and-gaps.md`. The orchestrator spot-checked the items marked (verified).

## What exists

- Rust: 45+ repos (about 30 load Laya weights), 24 Jev client crates on crates.io. The prior count of 17 and 11 was low.
- Ollaya: 901 stars, v0.7.5 released today. Since v0.7.1 (2026-09-26) the macOS binary runs Laya on the GPU through MLX, fp32 only (verified).
- Hybrid ANE + GPU routing has shipped in Python (tc3oliver/laya-apple, DJLougen/laya-fast) and in Rust (chriscoveries/jevalaya: ANE native through objc2-core-ml, MLX through a PyO3 bridge, no batching, 2 stars) (verified crate layout).
- Rust to Core ML for Laya: done in jevalaya and j75689/laya_demo_test.
- One pass per request for all questions: cbjev (tomek7667, GPL-3.0 code and weights). 10 questions over 500 tokens take 11.4 ms vs 75.8 ms for Laya on an RTX 4090, and it scores 0.783 vs 0.768 on typed decisions (verified). Its document encoding depends on the question set, so it cannot be cached across requests.
- Batching servers on CUDA: laya.cpp, kime, sys1, c4bbage/laya-serve (Go + TensorRT, 402 rps on one 4090). Response cache: kime.

## What does not exist yet

1. A model plus runtime that encodes the input once and answers later questions against the cached encoding. Upstream closed the proposal (issue #49, 2026-09-25): new checkpoints "not planned for the open-source models" (verified). ikken (Apache-2.0) has the attention mask but no checkpoint. kime has a spec and trainer but no shipped model. Needs training.
2. A same-machine benchmark across runtimes that only counts runs whose answers match. kime-bench has good rules but only compares kime against Laya.
3. A native (Swift or Rust) Apple engine that routes between ANE and MLX/Metal, batches across requests, and beats PyTorch MPS, MLX and Core ML. kime Metal is level with or behind MPS at batch 1, depending on which of its two reports you read.
4. Any published Laya number on a base M5.

## Facts that matter for this laptop (base M5, 32 GB, macOS 26.2)

- Closest data: laya-apple's run on a base M4, 32 GB, macOS 26.2 build 25C56 (same build as here). typed-decisions at 128 tokens: ANE 14.5 ms vs MLX GPU 37.1 ms p50, a 2.5x ANE lead (verified). On M5 Pro the lead nearly vanished (3.6 vs 3.9 ms). A small GPU makes the ANE matter more.
- MLX uses the M5 neural accelerators from MLX 0.30 on macOS 26.2+. On M5, MLX runs fp32 matmul as TF32 by default (`MLX_ENABLE_TF32=0` turns it off), and padded batched attention drifts by about 2^-11 in fp16. Tests demanding identical bits across batch sizes will fail on M5.
- The fast ANE numbers are for the multilingual checkpoint. The 421M English checkpoint fails FluidInference's ANE parity gate.
- Precision breaks answers more than anything else: fp16/bf16 and uniform INT8 broke parity in Ollaya, zerodegress and kime. Weight-only per-channel INT8 kept every measured decision.

## Models and licenses

| model | size | license | typed-decisions |
|---|---|---|---|
| Laya (English), ModernBERT-large | 421M, 842.6 MB fp16 | Apache-2.0 | 0.361 (base) |
| laya-typed-decisions | 421M | Apache-2.0 | 0.766 to 0.768 |
| laya-multilingual, mmBERT-base | 322M | Apache-2.0 | 0.352 |
| OpenDecider-nano, Ettin-400m | 395M | Apache-2.0 | 0.796 |
| cbjev | 421M / 322M | GPL-3.0 | 0.783 |
| decision-modernbert-base (Core ML only) | 150M | CC BY-NC 4.0 | not reported |

Jev: 0.727 to 0.754 depending on harness and date. "Decider" and "Lev" are decoder LLMs (2B to 35B), not Laya-shaped.

## Corrections to the prior chat

- Batching's 3.6-10x speedup applies only on GPU. On CPU it is 0.8-2.4x.
- laya-mlx has M3 Max data only, not M4 Pro. laya-apple numbers are M4 Max, not M5 Pro. All FluidInference numbers are on macOS 27.0.
- The ~270 s ANE compile applies to laya-apple only. FluidInference's precompiled buckets load in 4.9 to 8.5 s.
- kime's "10x": 13.7x at p50 for one question on a 4090, 6.9x at p95, 1.42x for ten questions, 0.80x on M4 Metal. Report D calls the 13.7x an upper bound, because the kime and Laya timings came from different sessions. In that one-question setting kime matched Laya fp32 on CUDA on only 96 of 100 answers, below the 99% agreement this project requires. Against Laya fp32 on MPS it matched all 100, and Laya on CUDA matched 96 (Report D, section 2).
- Upstream Laya server does serialize requests behind a lock (confirmed).
