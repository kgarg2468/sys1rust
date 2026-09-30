#!/bin/bash
# Stage C: the candidate Rust variant against cache-capped Python MLX controls.
cd "$(dirname "${BASH_SOURCE[0]}")"
S=20260929T095500
./run_stage.sh $S-C \
  "sys1rust mlx-fp16-fast typed-decisions correctness 5 1" \
  "sys1rust mlx-fp16-fast typed-decisions timing 12 2" \
  "laya-mlx mlx-fp16-opt-c512 typed-decisions timing 12 2" \
  "laya-mlx mlx-fp16-c512 typed-decisions timing 12 2" \
  "sys1rust mlx-fp16-fast typed-decisions short 5 5" \
  "laya-mlx mlx-fp16-opt-c512 typed-decisions short 5 5"
for i in 1 2 3; do
  ./run_stage.sh $S-cold$i "sys1rust mlx-fp16-fast typed-decisions cold 0 1" "laya-mlx mlx-fp16-opt-c512 typed-decisions cold 0 1"
done
./run_stage.sh $S-sus \
  "sys1rust mlx-fp16-fast typed-decisions timing 12 1 300" \
  "laya-mlx mlx-fp16-opt-c512 typed-decisions timing 12 1 300"
