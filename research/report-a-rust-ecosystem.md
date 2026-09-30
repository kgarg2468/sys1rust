# Report A: Rust ecosystem for local System One models

Checked 2026-09-29 (UTC) against live sources: GitHub REST API via `gh`, crates.io API, Hugging Face API, raw GitHub file contents, and web search. Nothing here comes from memory. Star and download counts are as of the check time and move daily; every project in this space is 6 to 12 days old.

Scope: the "Round 1: Rust ecosystem" claims in the earlier research notes (not in this repo), plus the Rust-specific items from Rounds 2 and 3 (sys1, kime, Ollaya MLX feature, zerodegress/laya-rust, mlx-rs, objc2-core-ml, sys1 binaries, kime CHANGELOG 8-client result, kime-bench numbers).

## 1. Claims table

Status key: CONFIRMED (matches source), CORRECTED (source says something different; correct value given), UNVERIFIABLE (no primary source found).

| # | Claim from prior chat | Status | Correct value / detail | Source |
|---|---|---|---|---|
| 1 | Jev released Sept 15, Laya Sept 18 | CONFIRMED | TypeSafe blog post dated Sep 15, 2026. `NandhaKishorM/laya` repo created 2026-09-18T04:46Z; HF `convaiinnovations/laya` created 2026-09-18. | https://typesafe.ai/blog/introducing-system-one-models-and-jev ; `gh api repos/NandhaKishorM/laya` ; https://huggingface.co/api/models/convaiinnovations/laya |
| 2 | At least 17 Rust projects run these models locally | CORRECTED | 45+ Rust repos found: about 30 that load Laya weights and about 15 that put another LLM behind a Jev-shaped API. See section 2. 17 is a large undercount. | `gh search repos` (queries in section 5) |
| 3 | ~11 Rust client crates for Jev's API | CORRECTED | 24 client-library crates on crates.io (21 general-purpose plus 3 framework plugins), plus at least 3 GitHub-only clients. Most downloaded: `kunobi-jev` (924), `typesafe-sdk` (712). See section 3. | crates.io search "jev", "systemone", "typesafe" |
| 4 | Jev's weights are closed | CONFIRMED | Closed weights, hosted API only, $0.042 per million input tokens. | https://en.wikipedia.org/wiki/Jev_(AI_model) ; https://jevtypesafeai.com/jev/open-source ; https://typesafe.ai/blog/introducing-system-one-models-and-jev |
| 5 | Rust projects run Laya or open models like Decider, Lev | CORRECTED | "Lev" not found anywhere. Ollaya runs `laya`, `decider`, `kev` (Jared Palmer), `winnow`, `decision`, `qwen3guard`, `nli`, `gliclass`, `von`, `clm`, `jevk5`. Most other Rust projects run only Laya. | https://github.com/ollaya-dev/ollaya README |
| 6 | Jev compatibility means they accept `/v1/systemone` | CONFIRMED with exceptions | Most serve `POST /v1/systemone`. Exceptions: `redwolf2019/laya-rs` serves `/v1/system-one` (hyphen) and states it is not a drop-in Jev replacement; `zerodegress/laya-rust` serves `/systemone`; `centillex-labs/laya-goish` serves `/v1/decide`. | READMEs of each repo |
| 7 | Ollaya: Ollama for decision models, daemon + CLI + desktop app, Jev-compatible API | CONFIRMED | `/v1/systemone`, `/v1/decisions`, `/v1/models` wire-identical to TypeSafe; desktop app for macOS, Windows, Linux; MCP server. | https://github.com/ollaya-dev/ollaya README |
| 8 | Ollaya backends: ONNX Runtime (CPU, NVIDIA) and llama.cpp (Metal) | CORRECTED (incomplete) | ONNX Runtime on CPU and CUDA; llama.cpp for GGUF models on CPU, CUDA and Metal; plus an MLX engine on the Apple GPU shipped in 0.7.1 (2026-09-26) for `laya:en`, `laya:multilingual`, `nli:modernbert-large`. A `coreml` cargo feature exists but models stay on the CPU provider. | README; `crates/ollaya-runner/Cargo.toml` features; release v0.7.1 notes |
| 9 | Ollaya fp32 build matches Python reference on 2,383/2,383 questions | CONFIRMED | README: "fp32 exports give the same decision as the PyTorch reference on 100% of 2,383 questions per checkpoint." | README |
| 10 | Ollaya 874 stars | CORRECTED | 900 stars (2026-09-29). Created 2026-09-23. Apache-2.0. | `gh api repos/ollaya-dev/ollaya` |
| 11 | sys1: pure Rust on candle, dynamic batching, Flash Attention 2/3, CPU/Metal/CUDA, `cargo install sys1` | CONFIRMED | Cargo features `cpu` (default), `metal`, `cuda`, `flash-attn-2`, `flash-attn-3`. README lists token-based dynamic batching and SDPA on CPU, Metal, CUDA. Serves `/v1/systemone` and `/v1/decide`. | https://github.com/alvarobartt/sys1 README and Cargo.toml |
| 12 | sys1 by Alvaro Bartolome (Hugging Face) | CONFIRMED | GitHub profile: "tech lead & eng @huggingface, inference + cloud". | `gh api users/alvarobartt` |
| 13 | sys1 49 stars, crate v0.0.3 | CONFIRMED | 49 stars; crate `sys1` v0.0.3, 59 downloads, published 2026-09-25. | `gh api repos/alvarobartt/sys1` ; https://crates.io/api/v1/crates/sys1 |
| 14 | sys1 v0.0.3 release has no downloadable binaries | CONFIRMED | v0.0.1, v0.0.2, v0.0.3 releases all have 0 assets. Install is source-only via cargo or Dockerfile. | `gh api repos/alvarobartt/sys1/releases` |
| 15 | kime: "10x faster than Laya" are targets, not measurements | CORRECTED (now partly measured) | The 10x table is still targets, but kime-bench has measured rows: RTX 4090 W1 p50 2.61 ms vs Laya 35.72 ms (13.7x), p95 5.26 ms (6.9x), W3 38.15 vs 54.24 ms (1.42x), M4 Metal 103.67 vs 83.44 ms (0.80x, a loss). | https://github.com/tamnd/kime-bench README ; reports/2026-09-28-kime-0.1.0-4090 |
| 16 | kime runs Laya, serves Jev-style API | CONFIRMED | Runs published Laya checkpoints on CPU, CUDA, Metal; `kime serve` answers `/v1/systemone`, `/v1/systemone/batch`, `/v1/models`, `/metrics`. Native kime models, router, SDKs not shipped. | kime README, CHANGELOG 0.0.12 |
| 17 | kime is the most downloaded crate (476) | CORRECTED | `kime` 489 downloads; the workspace crates are higher (`kime-core` 1,130, `kime-model` 953, `kime-tok` 724, `kime-cpu` 715, `kime-cuda` 681). Among Laya-runtime crates the kime family leads. Client crates `kunobi-jev` (924) and `typesafe-sdk` (712) beat the `kime` crate itself. kime has 5 GitHub stars. | crates.io API |
| 18 | jev-rs turns any local GGUF LM into a Jev-style decision engine in one pass; 15 stars | CONFIRMED with nuance | 15 stars. It does not load GGUF itself: it scores through a running `llama-server` (raw prompt path) or any OpenAI-compatible logprobs endpoint. Prefills state once, one suffix per question. Crate `jev-rs` v0.1.0, 23 downloads. Release v0.1.0 has binaries for 4 targets. | https://github.com/yijunyu/jev-rs README |
| 19 | Smaller projects exist: aovestdipaperino/laya-rust, Trystan-SA/laya-candle, GTC6244/Laya-Decision, redwolf2019/laya-rs, IAmJSD/pg-laya, bvolpato/kevala | CONFIRMED | All exist. Details in section 2. Note `GTC6244/Laya-Decision` publishes three crates (`laya-decision`, `-serve`, `-cli`) and `bvolpato/kevala` is Rust compiled to WASM with WebGPU kernels, distributed on npm. | `gh api repos/...` |
| 20 | Python original NandhaKishorM/laya ~27.7k stars | CONFIRMED | 27,915 stars. Apache-2.0. | `gh api repos/NandhaKishorM/laya` |
| 21 | Python server holds a lock, one model call at a time | CONFIRMED | `laya/serve.py`: `ThreadPoolExecutor(max_workers=1)`, an `asyncio.Lock` gate around `router.predict`, and an admission `asyncio.Semaphore` that returns 503 when full. | https://raw.githubusercontent.com/NandhaKishorM/laya/main/laya/serve.py lines 338-360, 425-495 |
| 22 | Batching gave 3.6-10x upstream | UNVERIFIABLE | Could not find "3.6-10x" in BENCHMARKS.md. Closest figures: T4 per-question cost drops from 32.8 ms (1 q) to 7.2 ms (10 q), about 4.5x; the CUDA fast path (TileLang + CUDA graphs) gives 1.2x to 5.1x end to end. "3-10x" appears only as a parity statement (fp16 is 3-10x closer to fp32 than bf16). | https://raw.githubusercontent.com/NandhaKishorM/laya/main/BENCHMARKS.md |
| 23 | CPU model compute 193-584 ms per question | CONFIRMED | AWS m7a.xlarge (4 cores), fp32, 1 question: multilingual 193 ms, english 580 ms, typed-decisions 584 ms. | BENCHMARKS.md "Server CPU: AMD EPYC 9R14" |
| 24 | Tokenization already Rust via HF tokenizers | CONFIRMED (indirect) | sys1 README uses `tokenizers`; laya-r-mlx verifies its tokenizer byte-identical to laya 0.3.10; laya-mlx docs say tokenization uses HF's Rust tokenizer. | sys1 README; laya-r-mlx README; https://aiidelist.com/blog/what-is-laya-mlx |
| 25 | Weights ~842 MB fp16 | CONFIRMED | `model.safetensors` is 842.6 MB for `laya` and `laya-typed-decisions`; `laya-multilingual` is 643.8 MB. Fanaperana/laya-rs reads it as F16 ("846 MB F16 to F32"). | HF API `?blobs=true` |
| 26 | INT8/INT4 can shift probabilities near thresholds | CONFIRMED | Ollaya: fp16/bf16 MLX fail parity (482/483, 481/483; max diff 0.085); zerodegress deleted an f16 GEMM that broke its 1e-3 tolerance; kime spec: uniform W8/W6/W4 failed fidelity, k-means W8 passed. | Ollaya docs/decisions/0001-mlx-engine.md ; zerodegress AGENTS.md ; kime spec/09-apple.md |
| 27 | Peak GPU: upstream Python/CUDA 4.6 ms per request | CONFIRMED | 4.6 ms, 1 question, 72 tokens, `laya` fast path on RTX 4070 Ti SUPER (stock 17.7 ms). Multilingual 2.8 ms. | BENCHMARKS.md "GPU fast path" |
| 28 | kime: big speedups come from changing the model; Laya re-reads whole input per question; encode once is the gain | CONFIRMED | README: engine alone worth 3-5x on Laya's weights; "Laya re-encodes the whole state once per question. kime-v1 encodes the state once, caches its keys and values." | kime README "How" |
| 29 | oxiglade/mlx-rs 377 stars, v0.32 | CONFIRMED | 377 stars; v0.32.0 released 2026-09-12; 236,638 downloads; Apache-2.0. | `gh api repos/oxiglade/mlx-rs` ; crates.io |
| 30 | mlx-rs supports MLX compile and fused attention, no custom Metal kernel support | CONFIRMED | `transforms/compile` module present; `fast.rs` exposes `rope`, `scaled_dot_product_attention`, `rms_norm`, `layer_norm`. No `metal_kernel` binding in Rust source; code search for `mlx_fast_metal_kernel` hits only `ledger/*.json`. | raw `mlx-rs/src/fast.rs`, `lib.rs`; `gh api search/code` |
| 31 | zerodegress/laya-rust (1 star) is the only Laya port on mlx-rs | CORRECTED | Three Laya ports depend on `mlx-rs 0.32`: `zerodegress/laya-rust` (1 star), `andyjusa/laya-mlx-rs` (0 stars), `tjameswilliams/laya-r-mlx` (1 star, also candle Metal, C ABI, iOS). Ollaya's MLX engine binds mlx-c directly (`ollaya-mlx-sys`) and rejected mlx-rs as "unofficial, two maintainers, nine months behind MLX until recently". | Cargo.toml of each; Ollaya docs/decisions/0001-mlx-engine.md |
| 32 | zerodegress/laya-rust: no graph optimization, no benchmarks | CORRECTED (half) | "Without graph optimisation" is the README's own wording. It does have benchmarks in AGENTS.md: M4 16 GB MLX 62.1 ms p50 for 3 questions (20.7 ms/question), CPU 1,268 ms, RTX 3060 Laptop CUDA 30.3 ms. License is MIT (GitHub shows NOASSERTION because Cargo.toml has no license field). | https://raw.githubusercontent.com/zerodegress/laya-rust/main/AGENTS.md |
| 33 | Rust can reach ANE through objc2-core-ml | CONFIRMED | Crate `objc2-core-ml` v0.3.2, 224,617 downloads, last release 2025-10-04, part of madsmtm/objc2. | crates.io |
| 34 | Nobody has done ANE from Rust | CORRECTED (partly) | `tamnd/kime` publishes `kime-ane` v0.1.4 ("Apple Neural Engine backend through Core ML with fixed shape buckets"), but its `lib.rs` is a doc header and it targets kime's unreleased native models; spec/09-apple.md is the plan. `codejunkie99/keel` (304 stars, Rust/GPUI macOS app) runs Laya "locally through Core ML" via a worker; compute unit not stated. Ollaya measured ORT's Core ML provider and rejected it. No Rust project has a shipped, measured Laya path on the ANE. Correction (2026-09-29): that last sentence is wrong. Report C (R2-14) found two Rust projects that run Laya on the ANE through objc2-core-ml: chriscoveries/jevalaya (8.8 ms p50 on an M1 Max, its own number) and j75689/laya_demo_test (about 12 ms per decision on an M3). The bake-off measured jevalaya's ANE lane on the base M5 at 5.1 ms p50, with 100% agreement on the 40 short single-question multilingual requests (`results/tables.md`). | crates.io `kime-ane`; kime spec/09-apple.md; keel README; Ollaya ADR 0001 |
| 35 | kime-bench: M4 Metal 103.67 ms vs 83.44 ms PyTorch/MPS; RTX 4090 13.7x (2.61 vs 35.72 ms) | CONFIRMED | Exact figures in kime-bench README. Report adds p95 5.26 ms (6.9x), p99 6.07 ms, W3 1.42x, 200 warmup + 2,000 measured calls, answers byte-identical to 0.0.26. | https://github.com/tamnd/kime-bench ; reports/2026-09-28-kime-0.1.0-4090/README.md |
| 36 | kime 8 clients: 43.3 s vs 67.1 s PyTorch/MPS (CHANGELOG) | CONFIRMED with caveats | CHANGELOG 0.0.24: 300 texts x 5 questions from 8 clients, Metal FP16 43.3 s (FP32 59.3 s) vs laya-serve 0.3.20 on MPS 67.1 s "in the same hour (and 50.4 s in a quieter hour yesterday)". Load average 20-58 from other work. Single client was level: 196-238 ms vs 180-217 ms. | https://raw.githubusercontent.com/tamnd/kime/main/CHANGELOG.md |
| 37 | Ollaya has an optional MLX engine behind a build feature | CORRECTED | It is behind the `mlx` cargo feature, but it shipped: release 0.7.1 (2026-09-26) links MLX into the macOS binary, installer requires macOS 14, and `ollaya-darwin-arm64-mlx` assets carry the Metal library. M4 Pro, 5 questions: `laya:en` 265 to 114 ms, `laya:multilingual` 115 to 42 ms, parity within 1.4e-4. fp32 only. | release v0.7.1 notes; scripts/install.sh; site/docs/faq.md |
| 38 | kime targets 0.72 ms/question, announced not shipped | CONFIRMED | Target table row "Per question, ten questions over a document, T4: 7.2 ms (Laya) to 0.72 ms". README: "The native kime models, the router, the caches and the SDKs are still to come." | kime README |
| 39 | Encode-once model needs retraining | CONFIRMED | kime README and spec describe kime-v1 as a new architecture (state tower + question tower with cross attention), trained by kime-train. | kime README, spec/05-model.md reference |

## 2. Rust projects found

### 2a. Runtimes that load Laya weights

Backend column lists what the repo says it compiles against. "API" is the served path. "Bench" means the repo publishes its own numbers. "Bin" means GitHub release binaries exist.

| Repo | Stars | Created / last push | License | Backend | Models | API | Bench | Bin | Crate (version, downloads) |
|---|---|---|---|---|---|---|---|---|---|
| ollaya-dev/ollaya | 900 | 09-23 / 09-28 | Apache-2.0 | ONNX Runtime CPU + CUDA; llama.cpp CPU/CUDA/Metal for GGUF; MLX (Apple GPU) since 0.7.1 | laya (en, multilingual, typed-decisions), decider, kev, winnow, decision, qwen3guard, nli, gliclass, von, clm, jevk5 | `/v1/systemone`, `/v1/decisions`, `/api/decide` | Yes (4090, M4 Pro) | Yes, 23 assets per release (Linux, macOS, Windows, deb, rpm, dmg, msi, Docker); 2,146 downloads on v0.7.5 | none |
| alvarobartt/sys1 | 49 | 09-22 / 09-28 | Apache-2.0 | candle CPU / Metal / CUDA, flash-attn 2 and 3 | laya, laya-multilingual, laya-typed-decisions | `/v1/systemone`, `/v1/decide` | No | No | `sys1` 0.0.3, 59 |
| tamnd/kime | 5 | 09-23 / 09-29 | Apache-2.0 | own kernels: CPU (FP32, INT8), CUDA (FP16/FP8/INT8, CUDA graphs), Metal (indirect command buffers), ANE crate (stub) | laya checkpoints; native kime models unreleased | `/v1/systemone`, `/batch`, `/v1/models`, `/metrics` | Yes (kime-bench, CHANGELOG) | Yes, 4 targets per release (macOS arm64, Linux x86_64/aarch64, Windows) | `kime` 0.1.4, 489; `kime-core` 1,130; 14 workspace crates |
| aovestdipaperino/laya-rust | 10 | 09-20 / 09-20 | Apache-2.0 | candle CPU / Metal / CUDA | all three Laya checkpoints | library + `laya-serve` web console | No | No | `laya` 0.1.1, 99 |
| Trystan-SA/laya-candle | 9 | 09-22 / 09-23 | Apache-2.0 | candle (Hub download) | laya en + multilingual, Router | library + CLI; `cargo bench` | No published numbers | No | `laya-candle` 0.1.1, 37 |
| GTC6244/Laya-Decision | 2 | 09-23 / 09-28 | Apache-2.0 | candle CPU + Metal ("~4x faster forward on an M4") | laya en + multilingual | `/v1/systemone` (laya-serve) | Parity 1e-4 vs PyTorch | No | `laya-decision` 0.2.3, 130; `-serve` 73; `-cli` 75 |
| redwolf2019/laya-rs | 10 | 09-23 / 09-24 | MIT | ONNX Runtime CPU, Linux x86_64/aarch64 only | laya-multilingual (fixed ONNX bundle) | `/v1/system-one` (hyphen) | docs/mvp-validation.md | Yes, v0.1.1 (Linux x86_64, aarch64, install.sh) | none |
| IAmJSD/pg-laya | 0 | 09-21 / 09-21 | Apache-2.0 | candle via Kelwing/laya-candle, pgrx, PostgreSQL 14-18 | laya en | SQL functions `laya.noul`, `laya.choice`, `laya.predict` | No | No | none |
| Kelwing/laya-candle | 0 | 09-21 / 09-21 | Apache-2.0 | candle | laya | library (used by pg-laya) | No | No | none |
| bvolpato/kevala | 8 | 09-21 / 09-28 | Apache-2.0 | Rust to WASM + WebGPU kernels, browser | laya (int8 pack 479 MB), bruv, kev, semif-qwen3.5, gemma-4 | JS API `Kevala.load().decide()` | Yes (M4 Max, Chrome: laya 23.8 ms mean) | npm package | none (npm `kevala`) |
| zerodegress/laya-rust | 1 | 09-23 / 09-23 | MIT (no Cargo license field) | CUDA (default, NVRTC + cuBLAS), CPU, MLX via mlx-rs 0.32 | laya (GGUF converted) | `POST /systemone` | Yes (AGENTS.md) | No | none (`laya-rust` on crates.io is a different, repo-less crate) |
| andyjusa/laya-mlx-rs | 0 | 09-22 / 09-22 | Apache-2.0 | mlx-rs 0.32, Metal | all three checkpoints, routing | CLI + localhost HTTP | Yes (M4 Pro: Rust MLX 42.5 ms vs Python MLX 48.5 ms encoder) | No | none |
| tjameswilliams/laya-r-mlx | 1 | 09-23 / 09-23 | Apache-2.0 | mlx-rs and candle Metal/CPU; C ABI, XCFramework, Swift, iOS | laya, laya-multilingual | library | Yes (M4 Max: 32.2 ms vs torch MPS 37.2 ms; iPhone 17 Pro 66 ms) | No | none |
| b0xtch/laya-candle | 4 | 09-20 / 09-20 | Apache-2.0 (vendored MLX kernels MIT) | candle CPU, Metal with fused kernels tuned for M1 Pro, CUDA untested | laya | CLI `predict` | No | No | none |
| centillex-labs/laya-goish | 16 | 09-22 / 09-23 | none stated | GGUF, CPU (AVX2/FMA), Linux x86-64 | laya en f16 GGUF, OpenThai-SystemOne | `/v1/decide` | No | Yes (v0.1.0 Linux) | none |
| Fanaperana/laya-rs | 2 | 09-23 / 09-23 | none | candle | laya | CLI | Yes (cold start 0.9 s vs 5.75 s Python) | No | none |
| bob-rietveld/laya-rs | repo 404 | crate 09-20 | ? | candle | laya | library, Snake demo | No | No | `laya-rs` 0.1.0, 70; `laya-core` 70; `laya-snake` 20 |
| humandebri/IC-Laya | 0 | 09-25 / 09-29 | MIT | W8A8 INT8, Internet Computer canisters | laya | canister | No | No | none |
| rookery-labs/laya-onnx-android | 0 | 09-28 / 09-28 | Apache-2.0 | ONNX Mobile runtime, Rust JNI tokenizer | laya | Android | No | No | none |
| elepedus/laya-rk3588-turingpi-rk1 | 1 | 09-27 / 09-27 | Apache-2.0 | RK3588 Mali GPU and Rockchip NPU | laya | ? | No | No | none |
| Partysun/jigor | 2 | 09-26 / 09-28 | Apache-2.0 | ort (ONNX) local for von/laya, plus remote Jev via OpenRouter | von, laya | System One wire protocol gateway | No | No | `jigor` 0.1.5, 70; `jigor-cli` 51 |
| LiteVar/system-one | 0 | 09-22 / 09-23 | MIT | llama.cpp FFI (CPU/Metal/Vulkan), managed provisioning | laya-multilingual | `/v1/systemone` | No | No | none |
| mahabodi/mahabodi | 3 | 09-24 / 09-28 | MIT | Rust crates (memory + Laya decisions), C ABI for Go and .NET | laya | library | No | No | `mahabodi` 0.1.2, 28; `-core` 42; `-ffi` 28 |
| codejunkie99/keel | 304 | 09-22 / 09-25 | MIT | Laya through Core ML worker (Rust/GPUI macOS app; a coding workspace, not a general runtime) | laya (pinned checkpoint) | internal selector | No | ? | none |
| a1re1/layad | 0 | 09-20 / 09-27 | Apache-2.0 | ? | laya | daemon | No | No | none |
| House-of-Imaginations/rsdecider-inference | 0 | 09-21 / 09-23 | none | ? | laya | ? | No | No | none |
| aryanthegamedev3465-collab/tiny-decision | 1 | 09-25 / 09-25 | MIT | in-process GGUF/ONNX | ? | ? | No | No | none |
| mcembalest/sys1 | 0 | 09-23 / 09-23 | NOASSERTION | candle (based on alvarobartt/sys1) | laya | `/v1/systemone` | No | No | none |
| codesoda/laya-rs | 0 | 09-19 / 09-22 | Apache-2.0 | "Planned" Metal runtime | laya | planned | No | No | none |
| mannlohchab/laya-candle, moecly/laya-rust, fxcl/laya-rs | 2 / 0 / 0 | 09-23 to 09-24 | none | not inspected (no description) | laya | ? | No | No | none |
| grafuls/huncho | 0 | 09-26 / 09-26 | none | "portable serving engine" | ? | ? | No | No | none |

### 2b. Jev-shaped engines that wrap another model

| Repo | Stars | Created / last push | License | Backend | Models | API | Bench | Bin | Crate |
|---|---|---|---|---|---|---|---|---|---|
| yijunyu/jev-rs | 15 | 09-21 / 09-28 | Apache-2.0 | scores via `llama-server` or OpenAI-compatible logprobs; MCP server | any GGUF decoder; adapters for trained specialists | `/v1/systemone`, `/v1/models` | `jev eval` metric stack | Yes, 4 targets | `jev-rs` 0.1.0, 23 |
| emnlmn/snap | 21 | 09-23 / 09-28 | MIT | llama.cpp, one batched `llama_decode` per request | any GGUF chat model | Jev wire-compatible | ? | Yes (releases) | none |
| bokuweb/omg | 3 | 09-19 / 09-25 | none | llama.cpp (`llama_memory_seq_cp` shared prefix), wgpu browser demo | Gemma 3/4, trained pointer heads, Kev layout | `/v1/systemone`, `/v1/models` | JGLUE eval, ~1.1 s for 5 q on M4 (E2B, browser) | ? | none |
| LuticaCANARD/L2S1 | 2 | 09-20 / 09-28 | MIT | llama.cpp | local GGUF chat models | Jev-style | No | No | `l2s1` 0.1.4, 41 |
| wnzn/carabao.rs | 0 | 09-29 / 09-29 | MIT | llama.cpp GGUF | any | ? | No | No | none |
| alitrack/jev-clone | 1 | 09-19 / 09-28 | Apache-2.0 | own readout, Chinese-first eval sets | ? | contract-compatible System One server | 195-item zh eval | No | none |
| einyx/kredo | 0 | 09-26 / 09-27 | Apache-2.0 | ONNX with parity gate (Ollaya-inspired) | own small trained heads (BERT-tiny), DistilBERT MNLI, mDeBERTa | `/v1/systemone` | acc numbers in README | ? | none |
| zojeda/llama-cpp-system-one | 0 | 09-19 / 09-21 | none | llama.cpp | DiffusionGemma | System One API | No | No | none |
| codesoda/kev-rs | 1 | 09-23 / 09-24 | Apache-2.0 | MLX (Apple) and candle CPU, parity-gated | Kev (pointer heads on Qwen) | ? | No | No | none |
| zozo123/gemma-to-jev | 0 | 09-22 / 09-27 | none | candle | Gemma 3 4B | Jev-style | No | No | none |
| lib-x/jev-bridge | 0 | 09-22 / 09-24 | MIT | proxies any OpenAI-compatible API | remote | Jev-style scoring service | No | No | `jev-bridge` 0.6.0, 104 |
| wweir/weigh | 0 | 09-24 / 09-28 | MPL-2.0 | vLLM / SGLang / OpenAI-compatible logprobs | remote | typed decisions | No | No | none |
| bhubbard/zev-rs | 1 | 09-24 / 09-27 | MIT | ? | ? | ? | No | No | `zev-rs` 0.1.0, 13 |
| ppmpreetham/vej, VakeDomen/DIY-Jev | 0 / 3 | 09-22, 09-18 | MIT | educational from-scratch | ? | ? | No | No | none |
| karminski/Jev-Quantum | 33 | 09-21 / 09-21 | MIT | none: a pseudo-random baseline that speaks the Jev protocol, for mocks and lower bounds | none | `/v1/systemone` | latency bench vs Jev | No | none |
| thusinh1969/BrighTO_Router | 17 | ? / 09-27 | Apache-2.0 | LLM gateway with System One routing | remote | gateway | No | ? | none |

Also seen: `Data-based-eng/laya-server` (0 stars, Apache-2.0, Docker CPU/CUDA/ROCm; GitHub reports no primary language, so probably not Rust). `tamnd/kime-compat` (0 stars) runs TypeSafe SDKs and Laya clients against kime.

## 3. Rust Jev API client crates

All published on crates.io between 2026-09-16 and 2026-09-28. Downloads as of 2026-09-29.

| Crate | Version | Downloads | Repo | Notes |
|---|---|---|---|---|
| kunobi-jev | 0.2.0 | 924 | kunobi-ninja/kunobi-jev | most downloaded client |
| typesafe-sdk | 0.1.2 | 712 | codeitlikemiley/typesafe-sdk-rust | 4 stars |
| jev | 0.1.2 | 151 | gitlab.com/porky11/jev | also targets self-hosted backends |
| typesafe-ai-sdk | 0.5.0 | 122 | aoprisan/typesafe-ai-rust-sdk | ships `jev-repl` |
| jev-client | 0.2.0 | 115 | shaharia-lab/jev-cli | 30-star CLI repo |
| jev-sdk | 0.1.0 | 83 | portlandhodl/jev-sdk | |
| typesafe-client | 0.1.0 | 65 | JedimEmO/typesafe-client | earliest, 2026-09-16 |
| fuzzy-jev | 0.5.0 | 64 | dsaad68/fuzzy-jev | Jev via OpenRouter decisions endpoint |
| typesafe-systemone | 0.2.0 | 45 | 2commits/typesafe-systemone | |
| typesafe-io | 0.1.0 | 41 | typesafe-ai/typesafe-sdk-rs (404) | claims official; repo does not exist; TypeSafe org has only Python and JS SDKs |
| typesafe-jev | 0.2.0 | 37 | thehumanworks/jevgrep | |
| typesafe-ai | 0.1.0 | 26 | Twister915/typesafe-ai | 14 stars, sync + async |
| jevlin | 0.2.0 | 24 | Whth/fabricatio | |
| typesafe-api | 0.0.2 | 23 | none | |
| systemone (+ -facet, -macro) | 0.1.0 | 15 / 12 / 11 | lu-zero/systemone | derive macros for criteria |
| typesafe-rust-sdk | 0.1.0 | 15 | phiat/typesafe-rust-sdk | |
| jev-api | 0.1.0 | 14 | virolea/lintus | |
| system-one | 0.1.0 | 14 | heyaozh/system-one | backend-agnostic, cost-matrix decisions |
| jevkit | 0.1.0 | 12 | viralkachhadiya/jevkit | enum-macro questions |
| jevvy | 0.1.0-alpha.1 | 9 | zliv83/jevvy | |
| jev-rust | 0.1.0 | 7 | none | |
| gaise-provider-typesafe | 3.0.1 | 47 | ikcore/gaise | framework plugin |
| polyc-judgment-systemone | 2026.9.6 | 30 | officialunofficial/polychrome | framework plugin |
| typesafe-sdk-cmd-kit | 0.8.0 | 179 | douglance/jevon | shared plumbing for `jevon` CLI |

The crate named `typesafe` (0.1.1, 35 downloads) is a typing-practice app, not a client.

Count: 24 crates (21 general-purpose clients, 3 framework plugins). GitHub-only clients not on crates.io: `abeldzan/jev-rs` (3 stars), `Shearerbeard/jev-driver`, `FuturePresentLabs/ooda`, `tanishqnalloju/typesafe-rust-sdk` (self-described "Official", not under the TypeSafe org).

Beyond clients, crates.io search "jev" returns 209 crates, most of them CLIs and agent tools built on the hosted API (`jevi` 344 downloads, `jev-harness` 221, `jev-seo` 114, `jev-bridge` 104, `jevcc` and `jev-router` 100 each).

## 4. Model weight licenses

- `convaiinnovations/laya`, `laya-multilingual`, `laya-typed-decisions`: Apache-2.0 on Hugging Face, ungated, no token needed. `model.safetensors` 842.6 MB (laya, typed-decisions) and 643.8 MB (multilingual). Model card describes base checkpoints as "a fast base to specialise, not a zero-shot decision engine."
- Ollaya README lists per-model licenses for everything it pulls: laya (Convai Innovations), decider (Mapika), kev (Jared Palmer, on Qwen3.5), decision (vLLM Semantic Router, on Qwen3.5), qwen3guard, gliclass, von, winnow (on Gemma 4), jevk5, nli:modernbert-large are Apache-2.0; nli:deberta-v3-large (Moritz Laurer) is MIT; llama.cpp is MIT. Ollaya never re-hosts weights; it ships ~3 MB ONNX graphs that read the author's safetensors pinned to a commit and verified by sha256.
- kevala downloads pinned int8 packs from `huggingface.co/bvolpato/kevala-packs` (license not checked) or converts the original Laya checkpoint in the browser.
- GTC6244/Laya-Decision: "Model checkpoints are distributed separately on the Hugging Face Hub under their own license."
- Jev: closed weights, hosted only.

## 5. New findings not in the prior chat

1. The ecosystem is roughly three times bigger than claimed. 45+ Rust repos (about 30 Laya-weight runtimes, about 15 wrappers around other LLMs) and 24 client crates, all created 2026-09-16 to 2026-09-29. GitHub search queries used: "laya language:rust", "jev language:rust", "systemone language:rust", "system one decision language:rust", "typesafe decision language:rust", "laya candle", "laya server language:rust", "laya inference language:rust", "kev decision language:rust", "jev compatible language:rust".
2. Ollaya shipped an MLX engine on 2026-09-26 (v0.7.1): `laya:en` 265 to 114 ms, `laya:multilingual` 115 to 42 ms for 5 questions on an M4 Pro, fp32 only, parity within 1.4e-4. It binds mlx-c directly and explicitly rejected mlx-rs, ORT Core ML, ORT WebGPU and an mlx-swift sidecar. Its Core ML provider test failed to initialise in MLProgram form and gave wrong outputs in NeuralNetwork form. Ollaya's own M4 Pro numbers (114 ms for 5 questions) are far slower than the laya-mlx or Core ML figures in Round 2, so its MLX path is not tuned.
3. kime now has real measurements: 13.7x over Laya at p50 on an RTX 4090 for one short question (2.61 vs 35.72 ms), but only 6.9x at p95, 1.42x for ten questions over a document, and a loss on M4 Metal (0.80x). kime also has an answer cache (blake3-keyed), API keys and rate limits, `/metrics`, a training crate (`kime-train`) and a `kime-ane` crate that is a stub. 15 workspace crates on crates.io; release binaries for 4 targets on most tags.
4. Three Laya ports on mlx-rs (not one). The strongest is `tjameswilliams/laya-r-mlx`: MLX and candle-Metal backends behind one trait, C ABI, XCFramework, Swift package, runs on iPhone 17 Pro (66 ms p50 for an email with 4 questions). M4 Max: Rust MLX 32.2 ms vs torch MPS 37.2 ms, and its README states the forward is GPU-compute-bound (22 of 32 ms in f16 GEMM), so the gain is framework overhead and load time. `andyjusa/laya-mlx-rs` matches torch FP32 exactly and beats Python MLX by about 12% (42.5 vs 48.5 ms encoder).
5. Ollaya's release assets show real distribution demand: 2,146 downloads on v0.7.5 (2026-09-28), 1,972 on v0.7.3, vs 0 or 1 downloads on every kime release and no sys1 binaries at all.
6. `codejunkie99/keel` (304 stars) is a Rust/GPUI macOS coding workspace that runs Laya through Core ML as a route selector. It is the only Rust project found using Core ML for Laya, but compute units are not stated. Correction (2026-09-29): jevalaya and j75689/laya_demo_test also run Laya through Core ML from Rust (see claim 34).
7. Rust embedding targets now covered by existing projects: PostgreSQL (pg-laya), WASM/WebGPU browser (kevala), iOS/Swift/C ABI (laya-r-mlx), Android JNI (laya-onnx-android), Internet Computer canisters (IC-Laya), Rockchip NPU (laya-rk3588), Go/.NET via C ABI (mahabodi-ffi).
8. Precision is a recurring failure point across independent projects: Ollaya (fp16/bf16 MLX fail parity), zerodegress (deleted an f16 GEMM path that broke tolerance and was slower than cuBLAS), kime (uniform W8/W6/W4 fail, k-means W8 passes; FP16 attention needs split f16 plus remainder to hit its 1.5e-2 bound). Anyone building a fast path must budget for fp32-equivalent parity tests.
9. Upstream Laya's own CUDA fast path (TileLang fused kernels + CUDA graphs) already reaches 4.6 ms per single question and 2.8 ms multilingual on an RTX 4070 Ti SUPER, with 864/864 argmax agreement in fp16. Rust engines on NVIDIA compete with that, not with the 17.7 ms stock path.
10. `laya-ai.com/runtimes/rust` (third-party page, verified 2026-09-25) attributes "about 14 ms per query on an NVIDIA RTX Pro 6000" to sys1. The sys1 repo has no benchmark file; this number is unsourced.

## 6. Open questions

- Whether `kime-ane` contains any working code beyond the doc header. Only the head of `lib.rs` was read; the crate compiles to empty on non-macOS by design.
- Whether keel's Core ML worker runs Laya on the ANE or GPU. README says "through Core ML" only.
- `bob-rietveld/laya-rs` published three crates on 2026-09-20 but the GitHub repo returns 404. Source may be gone.
- `typesafe-io` claims to be the official TypeSafe Rust SDK from `typesafe-ai/typesafe-sdk-rs`, which does not exist. No official Rust SDK was found.
- Source of the prior chat's "3.6-10x batching" figure. Not in upstream BENCHMARKS.md.
- Source of sys1's "14 ms on RTX Pro 6000".
- Four repos with empty descriptions were not inspected: `mannlohchab/laya-candle`, `moecly/laya-rust`, `fxcl/laya-rs`, `Kelwing/laya-candle` (beyond its use by pg-laya).
- Ollaya's MLX engine is fp32 only; the ADR says fp16/bf16 fail parity. Whether a tuned fp16 path with fp32 residuals (as upstream Laya and kime do) would pass is untested there.
- No Rust project has published a same-machine comparison against laya-mlx or laya-coreml. kime-bench names them as baselines but its Apple rows so far cover only PyTorch/MPS.
- The GitHub `search/code` endpoint rate-limited once during this pass (5,000/hr shared). All results above were re-checked after the reset; no claim rests on a rate-limited call.
