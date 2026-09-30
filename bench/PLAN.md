# Base M5 System 1 bake-off

Goal: the first same-machine Laya runtime numbers on a base Apple M5 (MacBook Pro 14, Mac17,2, 10 CPU cores, 32 GB, macOS 26.2). Every contender answers the same requests; a run's speed only counts if its answers agree with the fp32 CPU reference. The results feed the design of our own runtime, so method experiments (precision, ANE vs GPU crossover, batching, padding) matter as much as the leaderboard.

## Ground rules for every worker

- `source bench/env.sh` (from the repo root) in every shell before installing or running anything. It redirects Hugging Face, uv, pip, cargo, npm, go and torch caches into `bench/`.
- Everything you create lives under your own `bench/contenders/<name>/` folder (clones, venvs, builds, converted weights, adapter, notes). Shared model downloads go to the shared `HF_HOME` automatically.
- SwiftPM: always `swift build $SWIFTPM_FLAGS --scratch-path <your folder>/.build`. xcodebuild: always `-derivedDataPath $XCODE_DERIVED`.
- Never use `sudo`, Homebrew installs, global `pip install`, `cargo install` without `--root <your folder>`, or install scripts that write to `/usr/local` or `~`. Download release archives into your folder instead.
- If anything writes outside `bench/` anyway (Core ML compile caches, Metal shader caches, a rustup toolchain a project pins, an app's `~/.something` folder), append the absolute path to `bench/LEAKS.txt` with a `# why` comment. Only list paths that did not exist before.
- Load the exact model revisions in `bench/models.lock.json`. For derived formats (MLX, Core ML, ONNX, GGUF), record which source sha they came from, or note that a third party's conversion has unknown provenance.
- No timing that matters happens during setup. Other workers build in parallel, so setup-time latencies are smoke checks only.

## Request format

Workloads are JSONL files in `bench/workloads/`. Each line:

```json
{"id": "s128_q4_007", "shape": {"state_tokens": 128, "n_questions": 4}, "body": {"state": "...", "questions": {"q0": {...}, "q1": {...}}}}
```

`body` is exactly the JSON body that upstream `laya serve` accepts at `POST /v1/systemone`, minus `model`. See upstream `docs/http-api.md` in github.com/NandhaKishorM/laya for the question schema (types `choice`, `score`, `noul`).

Files (made by the reference worker):
- `workloads/smoke.jsonl`: 2 requests per shape, for quick checks.
- `workloads/correctness.jsonl`: about 300 requests across shapes, with gold answers under `"gold"` where the source dataset has them.
- `workloads/timing.jsonl`: 20 requests per shape.
- Shapes: state lengths 64, 128, 256, 512 tokens (English ModernBERT tokenizer, within 10%) times 1, 4, 10 questions = 12 shapes.
- `reference/<model>/<workload>.jsonl`: upstream Laya PyTorch fp32 CPU answers in the output format below.

## Adapter interface

Each contender provides an executable `bench/contenders/<name>/run` (any language; a shell script that activates a venv is fine).

```
run --list-variants
run --variant V --model M --workload FILE --out FILE [--warmup N] [--repeats R] [--duration SECONDS] [--concurrency K]
```

- `--list-variants` prints JSON: `[{"variant": "mlx-fp32", "models": ["typed-decisions", "multilingual"], "mode": "inproc" | "http", "max_state_tokens": 512, "notes": "..."}]`.
- `--model` is a key from `models.lock.json`.
- The adapter loads the model once, runs `--warmup` requests (default 5, taken from the workload), then runs every workload request in file order, `--repeats` times (default 1). With `--duration`, it loops over the workload until the time is up instead. `--concurrency K` (http mode only) runs K client threads.
- Output JSONL, one line per event:
  - first line: `{"type": "meta", "contender": "...", "variant": "...", "model": "...", "model_sha": "...", "code_version": "<git sha or release tag>", "backend": "mlx|coreml-ane|coreml-gpu|metal|mps|cpu|...", "mode": "inproc|http", "load_ms": 1234.5, "pid": 123, "t_process_start": <unix float>}`
  - per request: `{"type": "result", "id": "...", "repeat": 0, "t_start": <unix float>, "latency_ms": 12.3, "answers": {...}}`
  - unsupported: `{"type": "result", "id": "...", "unsupported": "state longer than 128-token bucket"}`
  - errors: `{"type": "result", "id": "...", "error": "..."}`
- `answers` uses the `/v1/systemone` response `answers` shape (same keys as the request's `questions`), including probabilities where the runtime exposes them. Convert from `/predict` or library formats inside the adapter.
- `latency_ms` covers request in, answers out: tokenization, model, post-processing. It excludes reading the workload file and writing output. In http mode it is the client-side round trip on localhost with a kept-alive connection.
- Server contenders: `run` starts its own server on a free port, waits for readiness, drives it, and shuts it down on exit, including on error.
- `bench/contenders/<name>/NOTES.md`: install steps, versions, variants, known caveats, anything written outside `bench/`, and whether the smoke check passed.

## Correctness

The harness compares every result to the reference for the same model: categorical agreement (choice label, noul value, score level or value within the reference's rounding) and the largest absolute probability difference. Runs below 99% agreement are flagged in every table. Contenders that run different weights (cbjev, OpenDecider-nano, decision-modernbert) are scored against gold answers instead, and are not compared with the reference.

Smoke check before handing off: run `smoke.jsonl` and, if `reference/<model>/smoke.jsonl` exists, report agreement. If the reference is not ready yet, say "built, not validated".

## Phases

1. Reference and harness (worker P1): upstream Laya venv, pinned downloads, workloads, reference answers, harness (`bench/harness/`), upstream contender.
2. Contenders (workers A to E, parallel with P1):
   - A `laya-mlx`, `laya-apple`: Python MLX and ANE.
   - B `fluid-coreml`, `decision-modernbert`: Swift and Core ML, ANE vs GPU vs all compute units per bucket, FluidUse routing.
   - C `sys1`, `kime`, `laya-cpp` (stretch): native Metal engines.
   - D `ollaya`, `jevalaya`, `ort-coreml` (stretch): servers and routers.
   - E `cbjev`, `opendecider-nano`: different models relevant to our method (multi-question packing, smaller or better encoders).
3. Timed runs (orchestrator only, serial, on wall power, `powermetrics` logging): warm latency per shape, cold start, 5-minute sustained single client, 4-client throughput for http contenders, peak RSS, energy per request.
4. Report: `bench/results/` raw data and `sys1rust/results/REPORT.md`.

## Method experiments (variants inside contenders)

- Precision: fp32 vs fp16 vs bf16 (and weight-only per-channel INT8 where a contender offers it). Agreement against speed.
- MLX on M5: TF32 default vs `MLX_ENABLE_TF32=0`.
- ANE vs GPU vs all compute units at each Core ML bucket (crossover length on base M5).
- Multi-question handling: one batch per request vs one pass per question vs cbjev packing.
- Padding: padded batch vs per-question when question lengths differ.
