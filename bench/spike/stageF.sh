#!/bin/bash
# Stage F: other Laya checkpoints (multilingual, english) through the Rust runtime, old default
# (mlx-fp16-fast) vs the speed settings (mlx-env). Correctness only; the machine is busy.
# Usage: stageF.sh STAMP BIN_DIR "model workload warmup repeats" ...
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd $BENCH_ROOT
STAMP=$1; export CARGO_TARGET_DIR=$2; shift 2
export SYS1_MLX="f16gelu,cache=512,wired=2048,dense_upto=1024,headprune,unpad"
for spec in "$@"; do
  set -- $spec
  for v in mlx-fp16-fast mlx-env; do
    echo "$(date +%T) run: $v $spec"
    harness/.venv/bin/python harness/runner.py --contender sys1rust --variant $v --model $1 --workload $2 --warmup $3 --repeats $4 --timestamp $STAMP --no-compare 2>&1 | grep runner:
  done
done
echo "$(date +%T) STAGE DONE"
