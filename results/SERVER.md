# sys1d server: results (2026-09-29)

This is the server step of the plan in `SYNTHESIS.md`, after the mlx-rs spike (`SPIKE.md`). `sys1d` serves Laya's `/v1/systemone` API from the Rust runtime, with the `mlx-fp16-fast` engine settings from the spike.

## Short answer

- `sys1d` is a 5.1 MB Rust binary. It runs no Python.
- It speaks the same protocol as upstream `laya-serve` (laya 9d955671), so Jev and Laya clients can point at it without changes. That covers request limits, error codes and `detail` strings, the `Server-Timing` and `X-Inference-Time-Ms` headers, and the response keys in the same order.
- Over HTTP it gives the same answers as the in-process engine. On the correctness workload 1,498 of 1,500 answers agree with the reference (99.9%), with 0 errors.
- HTTP adds 0.3 ms per request at p50 and 0.5 ms at p95.
- Over 5 minutes it holds 7.9 requests/s at p50 18.1 / 48.4 / 453 ms, which meets the bar on all three shapes, with no spikes.
- From process start to the first answer takes 243 to 375 ms.
- More clients do not add throughput. The GPU runs one forward pass at a time, so 4 clients get 3% more requests/s and wait about 4x longer each.

## Conditions

- AC power. Load average 3.8 to 6.3 at the start of each run, with other work on the laptop.
- Code at `runtime` commit e758bfa. Typed-decisions model at the pinned commit. Model files already in the OS file cache.
- The HTTP client matches the bake-off's HTTP contenders: Python `http.client`, one kept-alive connection per client thread. `latency_ms` is the client round trip.

## Results

Typed-decisions timing workload. Values are p50 / p95 / p99 in ms.

| run | 1q @128 | 1q @512 | 10q @512 | over 2x median |
|---|---|---|---|---|
| bar (laya-mlx compiled fp16, bake-off, AC) | 19.3 / 35 / 44 | 59.8 / 474 / 1695 | 453.5 / 2053 / 2284 | 9.2% |
| Rust in-process, 2 passes | 19.0 / 22 / 24 | 51.3 / 61 / 63 | 492.7 / 595 / 615 | 0% |
| sys1d over HTTP, 2 passes | 19.5 / 22 / 31 | 53.0 / 63 / 66 | 496.4 / 566 / 622 | 0% |
| Python laya-mlx compiled, 512 MiB cache, 2 passes | 20.0 / 24 / 31 | 53.0 / 59 / 79 | 482.7 / 550 / 564 | 0% |
| **sys1d over HTTP, 5 minutes** | **18.1 / 19 / 20** | **48.4 / 52 / 53** | **453.3 / 497 / 508** | **0%** |
| sys1d over HTTP, 4 clients, 1 pass | 238 / 253 / 261 | 478 / 494 / 498 | 989 / 1055 / 1065 | 0% |

Other measurements:

- **HTTP overhead.** Each response carries the server's own inference time in `X-Inference-Time-Ms`. The client round trip minus that time is 0.34 ms at p50 and 0.53 ms at p95 on the timing workload, 0.31 / 0.42 ms over 5 minutes, and 0.23 / 0.31 ms on the short workload. 5 of 3,353 requests took more than 1 ms extra, the most 7.7 ms.
- **Short workload** (69 to 92 tokens, one question). p50 is 13.4 ms over HTTP and 13.5 ms in-process.
- **5 minutes, sustained.** 7.80, 7.80, 7.98, 7.95 and 7.97 requests/s by minute, 2,370 requests in all. The in-process run in `SPIKE.md` held 7.45 on battery.
- **4 clients.** 7.57 requests/s against 7.33 for one client over the same pass. The server queues requests for its single inference thread, which is what upstream does too. The extra wait shows up as latency.
- **Cold start.** Three fresh processes. The model loaded in 164 to 294 ms and warm-up took 58 ms. The ready line came 227 to 357 ms after spawn. The first request took 13.7 to 14.4 ms, and the first answer arrived 243 to 375 ms after the client process started.
- **Memory.** MLX holds 806 MB of weights and caps its buffer cache at 512 MiB. Its peak is 1.6 GB and the process's largest RSS was 1.8 GB.
- **CPU.** Kernel time is about 15 s per 69 s timing run for Rust in-process, Rust over HTTP and Python alike. It comes from MLX and Metal, not from the server.

The paired timing runs put 10q @512 at 483 to 496 ms on AC, the same as the spike's 479 on battery. The 5-minute run reached 453. At this load the gap to the bar is run-to-run spread, not power.

## Correctness checks

- `cargo test -p sys1d` runs 8 unit tests and 19 HTTP tests against a fake model. They cover auth, every limit, every error code, admission (`503` with `Retry-After: 1`), a panic in inference, and a client that disconnects while its request waits.
- A live test (`cargo test -p sys1d --test live -- --ignored`) sends 24 requests to the real model. The HTTP answers are byte-identical to in-process `predict`. All 120 answers have the reference's key order, the largest probability difference is 0.0008, and 39 of 39 choices match.
- The correctness workload over HTTP: 300 requests with 0 errors. 1,498 of 1,500 answers agree with the reference, and gold accuracy is 75.8%, the same as in-process.

Building the server found two bugs in the runtime, fixed in e758bfa.

- Answers lacked `answer_confidence`. laya-r-mlx 914c9a7 predates the field. Every answer type now has it, between `confidence` and `action`, as upstream does.
- A panic inside the compiled GELU left its lock poisoned, so every later request failed with a `500`. The lock now recovers.

## Running it

```bash
source bench/env.sh
hf download convaiinnovations/laya-typed-decisions   # once; sys1d never downloads
$CARGO_TARGET_DIR/release/sys1d --model typed-decisions --port 8000
```

`sys1d --help` lists the options. They use upstream's environment variables (`LAYA_PORT`, `LAYA_API_KEY`, `LAYA_MAX_CONCURRENT` and so on). It binds 127.0.0.1 by default where upstream binds 0.0.0.0. When ready, it prints one JSON line on stdout with the address, model, revision, load time and warm-up time. SIGINT or SIGTERM lets requests in flight finish before it exits.

## What is left

- **Packaging.** The binary finds `libmlx.dylib` and `mlx.metallib` through an absolute rpath into the Python MLX wheel under `bench/`. It runs no Python, but it breaks if `bench/` is removed and cannot be copied to another Mac as it is. Shipping it needs those two files next to the binary with an `@executable_path` rpath, or MLX built from source.
- **Model download.** `sys1d` only reads the local Hugging Face cache. A user has to run `hf download` first.
- **Throughput.** 10q @512 is compute-bound, as `SPIKE.md` found. More clients cannot help on one GPU. The packed model is still the lever.

## Files

- **Code.** `sys1rust/runtime`, commit e758bfa: `crates/sys1d`, plus the `answer_confidence` and lock fixes in `laya-core` and `laya-mlx`.
- **Results.** `raw/server/bench/results/` holds every stage D run as JSONL, with the harness's `.run.json` and `.time.txt` and each server's log.
- **Scripts.** `raw/server/scripts/` has `stageD.sh` and its log, `run_stage.sh`, the HTTP bench adapter `http_adapter.py`, and `sys1rust-run`, which sends `http*` variants to it.
