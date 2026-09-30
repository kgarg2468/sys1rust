#!/bin/bash
# Stage F: other Laya checkpoints (multilingual, english) through the Rust runtime, old default
# (mlx-fp16-fast) vs the speed settings (mlx-env). Correctness only; the machine is busy.
# Usage: stageF.sh STAMP BIN_DIR "model workload warmup repeats" ...
# BIN_DIR is a cargo target dir; the sys1rust adapter runs BIN_DIR/release/sys1-bench.
STAMP=$1; BIN_DIR=$(cd "$2" && pwd) || exit 1; shift 2
export CARGO_TARGET_DIR=$BIN_DIR
export SYS1_MLX="f16gelu,cache=512,wired=2048,dense_upto=1024,headprune,unpad"
specs=()
for spec in "$@"; do
  set -- $spec
  for v in mlx-fp16-fast mlx-env; do
    specs+=("sys1rust $v $1 $2 $3 $4")
  done
done
exec "$(dirname "${BASH_SOURCE[0]}")/run_stage.sh" "$STAMP" "${specs[@]}"
