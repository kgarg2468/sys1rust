#!/bin/bash
# Serial runs: run_stage.sh STAMP "[gold-only] contender variant model workload warmup repeats [duration|-] [concurrency]" ...
# Each run prints the runner's status line and the compare total. The stage exits non-zero if a
# run fails, writes no results, has any request that ended in an error (the adapters exit 0 and
# write an error line when a request fails), the compare total is flagged LOW (below 99% agreement
# with the reference), or a correctness run has no agreement figure because no upstream reference
# exists for it.
# A leading "gold-only" opts one correctness run out of that last check, for a model that has no
# upstream correctness reference and is scored against gold labels only. Such a run still fails
# on request errors, and it fails if compare has no gold accuracy for it. Otherwise the stage
# prints NO REFERENCE with the gold accuracy and counts the run in the final line, so it never
# reads as a passed agreement check.
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "$BENCH_ROOT" || exit 1
STAMP=$1; shift
failed=0; noref=0
for spec in "$@"; do
  gold_only=0
  case "$spec" in "gold-only "*) gold_only=1; spec=${spec#gold-only } ;; esac
  set -- $spec
  extra=""; [ -n "$7" ] && [ "$7" != "-" ] && extra="--duration $7"; [ -n "$8" ] && extra="$extra --concurrency $8"
  echo "$(date +%T) load=$(sysctl -n vm.loadavg | awk '{print $2}') power=$(pmset -g batt | head -1 | sed "s/.*'\(.*\)'.*/\1/") run: $spec"
  out=$(harness/.venv/bin/python harness/runner.py --contender $1 --variant $2 --model $3 --workload $4 --warmup $5 --repeats $6 $extra --timestamp $STAMP 2>&1 | grep -E '^runner:|^\| all \|')
  rc=$?
  [ -n "$out" ] && echo "$out"
  # The errors, agreement and gold accuracy cells of the compare total row. compare prints "-" for
  # agreement when there is no reference and for gold accuracy when no question has a gold label.
  # runner.py prints no row when the run wrote no results.
  read -r errors agreement gold <<<"$(awk -F'|' '/^\| all \|/ { gsub(/ /, ""); print $6, $8, $12 }' <<<"$out")"
  if [ $rc -ne 0 ]; then
    echo "$(date +%T) FAILED (exit $rc): $spec"; failed=$((failed + 1))
  elif [ -z "$errors" ]; then
    echo "$(date +%T) NO RESULTS: $spec wrote no results"; failed=$((failed + 1))
  elif [ "$errors" != 0 ]; then
    echo "$(date +%T) REQUEST ERRORS: $errors request(s) failed in $spec"; failed=$((failed + 1))
  elif grep -q '^| all |.*LOW' <<<"$out"; then
    echo "$(date +%T) LOW AGREEMENT: $spec"; failed=$((failed + 1))
  elif [ "$4" = correctness ] && [ $gold_only -eq 1 ] && [ "$agreement" = "-" ] && [ "$gold" = "-" ]; then
    echo "$(date +%T) NO GOLD RESULT: $spec has no upstream reference and no gold accuracy"; failed=$((failed + 1))
  elif [ "$4" = correctness ] && [ $gold_only -eq 1 ] && [ "$agreement" = "-" ]; then
    echo "$(date +%T) NO REFERENCE: $spec has no upstream reference, so agreement was not checked (gold accuracy $gold)"
    noref=$((noref + 1))
  elif [ "$4" = correctness ] && [ "$agreement" = "-" ]; then
    echo "$(date +%T) NO AGREEMENT: agreement was not measured for $spec"; failed=$((failed + 1))
  fi
  sleep 10
done
if [ $failed -ne 0 ]; then
  echo "$(date +%T) STAGE FAILED: $failed run(s)"
  exit 1
fi
if [ $noref -ne 0 ]; then
  echo "$(date +%T) STAGE DONE: $noref gold-only run(s) had no upstream reference and no agreement check"
else
  echo "$(date +%T) STAGE DONE"
fi
