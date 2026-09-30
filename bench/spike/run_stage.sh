#!/bin/bash
# Serial runs: run_stage.sh STAMP "contender variant model workload warmup repeats [duration|-] [concurrency]" ...
# Each run prints the runner's status line and the compare total. The stage exits non-zero if a
# run fails or the compare total is flagged LOW (below 99% agreement with the reference).
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "$BENCH_ROOT" || exit 1
STAMP=$1; shift
failed=0
for spec in "$@"; do
  set -- $spec
  extra=""; [ -n "$7" ] && [ "$7" != "-" ] && extra="--duration $7"; [ -n "$8" ] && extra="$extra --concurrency $8"
  echo "$(date +%T) load=$(sysctl -n vm.loadavg | awk '{print $2}') power=$(pmset -g batt | head -1 | sed "s/.*'\(.*\)'.*/\1/") run: $spec"
  out=$(harness/.venv/bin/python harness/runner.py --contender $1 --variant $2 --model $3 --workload $4 --warmup $5 --repeats $6 $extra --timestamp $STAMP 2>&1 | grep -E '^runner:|^\| all \|')
  rc=$?
  [ -n "$out" ] && echo "$out"
  if [ $rc -ne 0 ]; then
    echo "$(date +%T) FAILED (exit $rc): $spec"; failed=$((failed + 1))
  elif grep -q '^| all |.*LOW' <<<"$out"; then
    echo "$(date +%T) LOW AGREEMENT: $spec"; failed=$((failed + 1))
  fi
  sleep 10
done
if [ $failed -ne 0 ]; then
  echo "$(date +%T) STAGE FAILED: $failed run(s)"
  exit 1
fi
echo "$(date +%T) STAGE DONE"
