#!/bin/bash
# Needs a runtime built from PR #17 or later. PR #17 adds the dense_upto, headprune and unpad
# settings of SYS1_MLX, and earlier builds reject them.
# Stage E: the exact speed settings (dense local attention, last head layer pruned, unpadding)
# against Python laya-mlx at its fastest (compiled, cache capped) and against our current default.
# Two rounds in opposite order, so drift in machine state hits every contender.
cd "$(dirname "${BASH_SOURCE[0]}")"
# The recorded run used STAMP=20260929T140500. By default a new run gets a fresh stamp, so it
# cannot overwrite the recorded results.
S=${STAMP:-$(date +%Y%m%dT%H%M%S)}
fail=0
export SYS1_MLX="f16gelu,cache=512,wired=2048,dense_upto=1024,headprune,unpad"
./run_stage.sh $S-E \
  "sys1rust mlx-env typed-decisions correctness 5 1" \
  "sys1rust mlx-env typed-decisions timing 12 2" \
  "laya-mlx mlx-fp16-opt-c512 typed-decisions timing 12 2" \
  "sys1rust mlx-fp16-fast typed-decisions timing 12 2" || fail=1
./run_stage.sh $S-E2 \
  "sys1rust mlx-fp16-fast typed-decisions timing 12 2" \
  "laya-mlx mlx-fp16-opt-c512 typed-decisions timing 12 2" \
  "sys1rust mlx-env typed-decisions timing 12 2" \
  "sys1rust mlx-env typed-decisions short 5 5" \
  "laya-mlx mlx-fp16-opt-c512 typed-decisions short 5 5" || fail=1
exit $fail
