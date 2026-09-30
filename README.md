# sys1rust

Run Laya System 1 decision models locally on Apple silicon. sys1rust is a Rust runtime on Apple's MLX with no Python at run time. Its server, `sys1d`, speaks the same `/v1/systemone` API as upstream `laya serve`, so Jev and Laya clients can point at it without changes.

## Status

Measured on a base M5 MacBook, typed-decisions model (details in `results/`):

- Answers match upstream Laya. On the 1,500-answer correctness workload, 1,498 agree with the upstream PyTorch fp32 reference, the same as Python MLX.
- It is 1.12x faster than Python laya-mlx at its fastest (compiled fp16, buffer cache capped), and faster on all 12 benchmark shapes. One question over a 128-token state takes 17.6 ms at p50. Ten questions over 512 tokens take 422 ms.
- The tail stays close to the median. In the timing runs no request took more than twice the median for its shape, where Python MLX without a capped cache had 9% of requests over that line.
- `sys1d` is a 5 MB binary. HTTP adds 0.3 ms per request at p50. From process start to the first answer takes 243 to 375 ms.

Limits today:

- Building needs a prebuilt MLX 0.32.2, which currently comes from the `mlx` Python wheel. The built binary links it by path. A self-contained release is the next step.
- The runtime finds models in the local Hugging Face cache. It does not download them.
- The GPU runs one forward pass at a time, so more clients wait longer instead of getting more throughput.

## Supported models

| name | Hugging Face repo | encoder |
|---|---|---|
| `typed-decisions` (default) | `convaiinnovations/laya-typed-decisions` | ModernBERT-large |
| `multilingual` | `convaiinnovations/laya-multilingual` | mmBERT-base |
| `english` | `convaiinnovations/laya` | ModernBERT-large |

All three give the upstream answers in this runtime. A local checkpoint directory also works.

## Build

Requirements: an Apple silicon Mac, Rust 1.85 or newer, CMake, and Xcode command line tools.

```sh
# 1. A prebuilt MLX 0.32.2 (the Python wheel ships libmlx and its CMake files).
python3 -m venv .mlx && .mlx/bin/pip install mlx==0.32.2
export MLX_SYS_PREBUILT_DIR="$PWD/.mlx/lib/python3.12/site-packages/mlx"   # match your Python version

# 2. Build.
cd runtime && cargo build --release
```

`source bench/env.sh` sets `MLX_SYS_PREBUILT_DIR` to the bench's own MLX and keeps every cache (Cargo, Hugging Face, pip) under `bench/`.

## Run

```sh
hf download convaiinnovations/laya-typed-decisions   # the huggingface_hub CLI
runtime/target/release/sys1d --model typed-decisions --port 8000
```

```sh
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "{\"subject\": \"Login\", \"body\": \"Tasks vanished after I logged out and back in.\"}",
  "questions": {"ticket_type": {"type": "choice",
    "instructions": "What kind of support ticket is this?",
    "criteria": {"Incident": "something is broken or not working",
                 "Request": "asks for information or a new service"}}}
}'
```

The reply (usage and routing fields cut):

```json
{"model":"laya-rl-agent","answers":{"ticket_type":{"type":"choice","choice":"Incident",
  "probabilities":{"Incident":0.8703,"Request":0.1297},"confidence":0.4434,
  "answer_confidence":0.8703,"action":{"act_probability":1.0}}}, ...}
```

Flags take the same environment variables as `laya serve`: `LAYA_HOST`, `LAYA_PORT`, `LAYA_API_KEY` and `LAYA_MAX_CONCURRENT`. `--model` (`SYS1_MODEL`) picks the checkpoint. `sys1d --help` lists the rest.

## Layout

- `runtime/`: the Rust workspace.
  - `laya-core`: request parsing, tokenization, sequence layout and answer decoding, with no GPU code.
  - `laya-mlx`: the forward pass on MLX through mlx-rs.
  - `sys1d`: the HTTP server.
  - `sys1-bench`: the benchmark adapter and `sys1-probe`.
  - `vendor/mlx-sys`: mlx-sys 0.6.0 with a build that can link a prebuilt MLX.
- `bench/`: the benchmark harness, workloads and upstream reference answers (`bench/PLAN.md`).
- `results/`: measured write-ups, from the bake-off of existing runtimes to the speed round.
- `research/`: sourced reports on the models, runtimes and hardware.

## License

Apache-2.0. `laya-core` and `laya-mlx` started as a fork of tjameswilliams/laya-r-mlx, and `vendor/mlx-sys` is from oxiglade/mlx-rs. See `NOTICE`.
