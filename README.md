# sys1rust

Run Laya System 1 decision models locally on Apple silicon. sys1rust is a Rust runtime on Apple's MLX with no Python at run time. Its server, `sys1d`, speaks the same `/v1/systemone` API as upstream `laya serve`, so Jev and Laya clients can point at it without changes. Unlike `laya serve`, which listens on 0.0.0.0, `sys1d` listens on 127.0.0.1 by default. Clients on other machines need the `--host` setting in [Run](#run).

## Status

Measured on a base M5 MacBook, typed-decisions model (details in `results/`):

- Answers match upstream Laya. On the 1,500-answer correctness workload, 1,498 agree with the upstream PyTorch fp32 reference (99.9%, above the 99% gate). The two that differ are near ties.
- It is 1.12x faster than Python laya-mlx at its fastest (compiled fp16, buffer cache capped), and faster on all 12 benchmark shapes. One question over a 128-token state takes 17.6 ms at p50. Ten questions over 512 tokens take 422 ms.
- The tail stays close to the median. In the timing runs no request took more than twice the median for its shape, where Python MLX without a capped cache had 9% of requests over that line.
- `sys1d` is a 5 MB binary. HTTP adds 0.3 ms per request at p50. From process start to the first answer takes 243 to 375 ms.

Limits today:

- Building needs a prebuilt MLX 0.32.2, which currently comes from the `mlx` Python wheel. The built binary links it by path. A self-contained release is the next step.
- The runtime finds models in the local Hugging Face cache. It does not download them.
- The GPU runs one forward pass at a time, so more clients wait longer instead of getting more throughput.

## Supported models

| name | Hugging Face repo | encoder | agreement with upstream fp32 answers |
|---|---|---|---|
| `typed-decisions` (default) | `convaiinnovations/laya-typed-decisions` | ModernBERT-large | 1,498 of 1,500 on correctness |
| `multilingual` | `convaiinnovations/laya-multilingual` | mmBERT-base | 100% on correctness, smoke and short |
| `english` | `convaiinnovations/laya` | ModernBERT-large | 100% on smoke, short and cold; no upstream correctness reference exists |

A local checkpoint directory also works.

## Build

Requirements: an Apple silicon Mac, Rust 1.85 or newer, CMake, Xcode command line tools, and Python 3.10 or newer. Run these from the repository root.

```sh
# 1. A prebuilt MLX 0.32.2 (the Python wheel ships libmlx and its CMake files),
#    and the Hugging Face CLI (hf) to download models.
python3 -m venv .mlx && .mlx/bin/pip install mlx==0.32.2 huggingface_hub
export MLX_SYS_PREBUILT_DIR="$(.mlx/bin/python -c 'import mlx.core, os; print(os.path.dirname(mlx.core.__file__))')"

# 2. Build. The binaries go to runtime/target/release/.
cargo build --release --manifest-path runtime/Cargo.toml
```

## Run

```sh
.mlx/bin/hf download convaiinnovations/laya-typed-decisions   # once; sys1d never downloads
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

`sys1d` listens on 127.0.0.1 by default, so only programs on the same Mac can reach it. To serve clients on other machines, listen on all interfaces with `--host 0.0.0.0` or `LAYA_HOST=0.0.0.0`. The server then accepts connections from anyone who can reach this Mac on the network, so set an API key too. With `LAYA_API_KEY` set, `/v1/systemone` answers only requests that send `Authorization: Bearer <key>`.

```sh
export LAYA_API_KEY=replace-with-a-secret
runtime/target/release/sys1d --model typed-decisions --host 0.0.0.0 --port 8000
```

## Build and run inside the benchmark setup

Use this instead of the steps above when working on the benchmark. `bench/env.sh` takes MLX from the laya-mlx contender's venv, so set that venv up first (`bench/contenders/laya-mlx/NOTES.md`). The script also keeps the Cargo output and the Hugging Face cache under `bench/`, so the binary is at `$CARGO_TARGET_DIR/release/sys1d`. The commands download and serve the typed-decisions revision pinned in `bench/models.lock.json`, the one the results used.

```sh
source bench/env.sh
cargo build --release --manifest-path runtime/Cargo.toml
REV=1a793eb568e6718f15941d08f85432581df534e3   # typed-decisions sha in bench/models.lock.json
bench/contenders/laya-mlx/.venv/bin/hf download convaiinnovations/laya-typed-decisions --revision $REV
$CARGO_TARGET_DIR/release/sys1d --model typed-decisions --revision $REV --port 8000
```

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
