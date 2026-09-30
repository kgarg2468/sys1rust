#!/bin/bash
# explore.sh WORKLOAD REPEATS "knobs" ... : sys1-bench mlx-env runs into tmp/explore, one summary line each.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "$BENCH_ROOT" || exit 1
mkdir -p tmp/explore
W=$1; R=$2; shift 2
for k in "$@"; do
  out="tmp/explore/$W-r$R-$(echo "$k" | tr ',:=' '_-.').jsonl"
  t0=$(date +%s)
  if ! SYS1_MLX="$k" "$CARGO_TARGET_DIR/release/sys1-bench" --variant mlx-env --model typed-decisions --workload "workloads/$W.jsonl" --out "$out" --warmup 12 --repeats "$R" 2>"$out.err"; then
    echo "$(date +%T) [$k] sys1-bench failed, see $out.err"
    continue
  fi
  echo "$(date +%T) $(($(date +%s)-t0))s load=$(sysctl -n vm.loadavg | awk '{print $2}') swap=$(sysctl -n vm.swapusage | awk '{print $6}') [$k] $(harness/.venv/bin/python spike/quick.py "$out")"
  sleep 5
done
