#!/usr/bin/env python3
"""Run one contender variant on one workload through its `run` adapter.

Usage:
  runner.py --contender laya-upstream --variant mps-fp32 --model typed-decisions --workload smoke
            [--warmup N] [--repeats R] [--duration S] [--concurrency K] [--power] [--no-compare]

`--workload` is a name in bench/workloads/ (smoke, correctness, timing) or a path.
The adapter runs under `/usr/bin/time -l` (peak RSS, peak memory footprint). Output:
  bench/results/<contender>/<variant>/<model>/<workload>-<timestamp>.jsonl   adapter events
  ...-<timestamp>.time.txt                                                  /usr/bin/time -l output
  ...-<timestamp>.run.json                                                  command, exit code, wall, RSS
  ...-<timestamp>.power.jsonl                                               with --power only
Then prints compare.py's table if a reference exists (skip with --no-compare).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def parse_time_l(path):
    out = {}
    if not os.path.exists(path):
        return out
    txt = open(path).read()
    m = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", txt)
    if m:
        out.update(real_s=float(m.group(1)), user_s=float(m.group(2)), sys_s=float(m.group(3)))
    for key, label in (("max_rss_bytes", "maximum resident set size"),
                       ("peak_footprint_bytes", "peak memory footprint"),
                       ("instructions_retired", "instructions retired")):
        m = re.search(r"(\d+)\s+%s" % label, txt)
        if m:
            out[key] = int(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contender", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--duration", type=float)
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--power", action="store_true", help="log powermetrics for the run (sudo -n)")
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--timestamp", help="override the timestamp in output names")
    args = ap.parse_args()

    run = os.path.join(BENCH, "contenders", args.contender, "run")
    if not os.access(run, os.X_OK):
        sys.exit("no executable adapter at %s" % run)
    wl_path = args.workload if os.path.exists(args.workload) else os.path.join(BENCH, "workloads", args.workload + ".jsonl")
    wl_name = os.path.splitext(os.path.basename(wl_path))[0]
    ts = args.timestamp or time.strftime("%Y%m%dT%H%M%S")
    out_dir = os.path.join(BENCH, "results", args.contender, args.variant, args.model)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, "%s-%s" % (wl_name, ts))
    out, time_txt, run_json, power_out = stem + ".jsonl", stem + ".time.txt", stem + ".run.json", stem + ".power.jsonl"

    cmd = [run, "--variant", args.variant, "--model", args.model, "--workload", wl_path, "--out", out]
    for flag in ("warmup", "repeats", "duration", "concurrency"):
        v = getattr(args, flag)
        if v is not None:
            cmd += ["--" + flag, str(v)]
    full = ["/usr/bin/time", "-l", "-o", time_txt] + cmd

    power = None
    if args.power:
        import power as pw
        _, status = pw.start(power_out)
        power = (pw, status)
    t0 = time.time()
    try:
        rc = subprocess.call(full)
    finally:
        t1 = time.time()
        if power:
            power[0].stop(power_out)
    info = {"cmd": cmd, "returncode": rc, "t_start": t0, "t_end": t1, "wall_s": round(t1 - t0, 3),
            "workload": wl_path, "out": out, "time": parse_time_l(time_txt),
            "power": power_out if power else None, "power_status": power[1] if power else None}
    n_res = n_err = n_unsup = n_bad = 0
    if os.path.exists(out):
        for l in open(out):
            if not l.strip():
                continue
            try:
                obj = json.loads(l)
            except ValueError:
                n_bad += 1
                continue
            if not isinstance(obj, dict) or obj.get("type") != "result":
                continue
            n_res += 1
            if "unsupported" in obj:
                n_unsup += 1
            elif "error" in obj:
                n_err += 1
    info.update(results=n_res, errors=n_err, unsupported=n_unsup, unparsed_lines=n_bad)
    with open(run_json, "w") as f:
        json.dump(info, f, indent=1)
    rss = info["time"].get("max_rss_bytes")
    print("runner: rc=%d results=%d unsupported=%d errors=%d unparsed=%d wall=%.1fs peak_rss=%s -> %s"
          % (rc, n_res, n_unsup, n_err, n_bad, t1 - t0, "%.0f MiB" % (rss / 2 ** 20) if rss else "?", out), file=sys.stderr)
    if not args.no_compare and os.path.exists(out) and n_res:
        import compare
        rep = compare.compare(out)
        print(compare.markdown(rep))
    return rc


if __name__ == "__main__":
    sys.exit(main())
