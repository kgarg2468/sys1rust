# laya-mlx (worker A)

Python MLX port of Laya by mizorewww, running on the GPU. It is not an official Convai release.

## Install

- Source: github.com/mizorewww/laya-mlx at `0a859518634112655cb97c745dbf04f5191aaf13` (main, 2026-09-22, "Sync with upstream v0.3.5"), cloned into `src/`. Package version 0.2.0.
- Venv: `.venv/` (uv 0.11.7, managed CPython 3.12.13 from `bench/.cache/uv-python`), made with
  `UV_PROJECT_ENVIRONMENT=$PWD/.venv uv sync --frozen --python 3.12 --managed-python` inside `src/` (the lockfile's base dependencies only, no extras).
- Versions: mlx 0.32.2 (+ mlx-metal 0.32.2), numpy 2.5.3, tokenizers 0.23.2, huggingface-hub 1.32.0.
- Weights: loaded straight from the pinned upstream repos in `bench/models.lock.json` (`convaiinnovations/laya-typed-decisions@1a793eb5…`, `convaiinnovations/laya-multilingual@e4e9ddf2…`) with `revision=<sha>`. laya-mlx converts the original `model.safetensors` to MLX in memory at load (parameter renaming plus a dtype cast). No export step ran, and the pre-converted `aac6fef/*-mlx` Hub repos are not used, so there is no third-party provenance question.
- `run` sets `HF_HUB_OFFLINE=1` by default, so the checkpoints must already be in the shared `HF_HOME`. They are, as of this setup.

## Adapter

`run` sources `bench/env.sh` and execs `.venv/bin/python adapter.py`. All variants are in-process on `Device(gpu, 0)`. `latency_ms` covers `Agent.predict(state, questions)`: prompt building, tokenization, the forward pass, the sync and the answer formatting. `load_ms` covers the MLX import plus `laya_mlx.load(...)`, which includes the weight conversion. `t_process_start` comes from `sysctl(KERN_PROC_PID)`. Because `run` uses `exec`, it is the moment `run` was launched.

| variant | dtype | multi-question | env / options |
|---|---|---|---|
| `mlx-fp16` | fp16 | one padded batch per request (`batch_size=16`) | library default |
| `mlx-fp16-opt` | fp16 | one padded batch | `compile=True, pad_to_multiple=16, cache_prompts=True` (the `laya-snake --optimize` path) |
| `mlx-fp16-perq` | fp16 | one forward per question (`batch_size=1`, no cross-question padding) | |
| `mlx-bf16` | bf16 | one padded batch | not in laya-mlx's published validation |
| `mlx-fp32` | fp32 | one padded batch | MLX default on M5: fp32 matmuls run as TF32 |
| `mlx-fp32-notf32` | fp32 | one padded batch | `MLX_ENABLE_TF32=0` |
| `mlx-fp32-notf32-perq` | fp32 | one forward per question | `MLX_ENABLE_TF32=0` |

MLX reads `MLX_ENABLE_TF32` once, from a static in `mlx/utils.h`. The adapter sets it, or unsets it for the TF32-default variants, before importing mlx.

Extra fields in the meta line: `device`, `dtype`, `batch_size`, `mlx_version`, `mlx_enable_tf32_env`, `weights_source`.

## Smoke check (passed)

Workload: `bench/workloads/smoke.jsonl` (24 requests, 120 questions). Reference: `bench/reference/<model>/smoke.jsonl` (upstream laya 0.3.21, torch CPU fp32). `bench/harness/compare.py` did not exist yet, so `tools/smoke_compare.py` did the scoring. It checks the choice label, the score argmax level and the noul side of 0.5, plus the largest absolute probability difference. Raw outputs are in `bench/results/laya-mlx/smoke_<model>_<variant>.jsonl`, and the summary is in `smoke_agreement.jsonl`.

| variant | typed-decisions agree / max dp | multilingual agree / max dp |
|---|---|---|
| mlx-fp16 | 100% / 0.0014 | 100% / 0.0036 |
| mlx-fp16-opt | 100% / 0.0014 | 100% / 0.0036 |
| mlx-fp16-perq | 100% / 0.0011 | 100% / 0.0036 |
| mlx-bf16 | **99.17%** (119/120) / 0.0096 | **96.67%** (116/120) / 0.0409 |
| mlx-fp32 (TF32) | 100% / 0.0017 | 100% / 0.0078 |
| mlx-fp32-notf32 | 100% / 0.0001 | 100% / 0.0001 |
| mlx-fp32-notf32-perq | 100% / 0.0001 | 100% / 0.0001 |

Other GPU builds ran during setup, so no smoke timing here means anything. For the record, model load was 0.3 to 1.7 s per variant, and nothing needed a cold compile.

## Caveats

- The default TF32 on M5 makes "fp32" less precise than fp16 on multilingual: max dp 0.0078 against 0.0036 for fp16 and 0.0001 without TF32. Label them as distinct precisions in reports.
- bf16 is below the 99% gate on multilingual in smoke.
- laya-mlx clamps calibration temperatures to [0.5, 5.0], following upstream v0.3.5. Typed-decisions ships `choice:11+ = 0.1006`, which gets clamped with a RuntimeWarning on stderr. This does not affect the bench: no smoke, correctness or timing question has more than 10 choice options, so the `choice:11+` bucket is never used.
- laya-mlx answers do not have upstream's `answer_confidence` field (upstream 0.3.20+ adds it). The adapter passes answers through unchanged, so a harness that scores `answer_confidence` will find it missing.
- `mlx-fp16-opt` caches tokenized question prefixes (up to 128). Workloads that repeat question sets across requests, such as `--repeats` or `--duration`, get cache hits that a real single pass would not. `mx.compile` is shape-specialized, so the first request at each new padded length pays a trace. Warmup covers only the first 5 requests.
- Past the checkpoint's `max_len` (1024 for both models), input is right-truncated silently, as upstream does. No bench shape comes near that limit (longest row about 850 tokens).
- The model runs single-threaded: one request at a time, and no server mode (the adapter rejects `--concurrency` other than 1).

## Written outside bench/

Nothing that I observed. MLX's Metal kernels are JIT-compiled in-process. No `~/Library/Caches` entry appeared for these runs.
