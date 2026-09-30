#!/bin/bash
# Serial runs: run_stage.sh STAMP "contender variant model workload warmup repeats [duration|-] [concurrency]" ...
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd $BENCH_ROOT
STAMP=$1; shift
for spec in "$@"; do
  set -- $spec
  extra=""; [ -n "$7" ] && [ "$7" != "-" ] && extra="--duration $7"; [ -n "$8" ] && extra="$extra --concurrency $8"
  echo "$(date +%T) load=$(sysctl -n vm.loadavg | awk '{print $2}') power=$(pmset -g batt | head -1 | sed "s/.*'\(.*\)'.*/\1/") run: $spec"
  harness/.venv/bin/python harness/runner.py --contender $1 --variant $2 --model $3 --workload $4 --warmup $5 --repeats $6 $extra --timestamp $STAMP --no-compare 2>&1 | grep runner:
  sleep 10
done
echo "$(date +%T) STAGE DONE"
