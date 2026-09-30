#!/usr/bin/env python3
"""Summarize result files into markdown tables.

Usage:
  summarize.py RESULT.jsonl [RESULT.jsonl ...] [--power SAMPLES.jsonl] [--gold-only] [--out TABLE.md]

Per shape: request count, unsupported requests and coverage (supported / requests), errors,
latency p50/p95/p99 (ms, linear interpolation over every answered result, all repeats; unsupported
and error lines are left out), agreement with the reference over answered requests and its flag
(compare.py), and energy per request. Per run: load time, peak RSS and peak memory footprint (from the runner's
.run.json), average combined CPU+GPU+ANE power, energy per request, and for runs of two minutes
or more, first-minute vs last-minute throughput.

Energy: power samples come from the result's sibling <stem>.power.jsonl (runner --power) or
--power. For serial runs, each request's energy is combined power integrated over
[t_start, t_start + latency] (samples are 1 s, so a short request gets the average power of the
second it ran in). With --concurrency > 1 windows overlap, so only the run-level figure
(energy over the whole measured span / completed requests) is reported.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare  # noqa: E402
import power as pw  # noqa: E402


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def f1(v, nd=1):
    return "-" if v is None else ("%.*f" % (nd, v))


def summarize(path, power_path=None, gold_only=False):
    rows = compare.read_jsonl(path)
    meta = next((r for r in rows if r.get("type") == "meta"), {})
    res = [r for r in rows if r.get("type") == "result"]
    stem = path[:-len(".jsonl")]
    run_info = json.load(open(stem + ".run.json")) if os.path.exists(stem + ".run.json") else {}
    power_path = power_path or (stem + ".power.jsonl" if os.path.exists(stem + ".power.jsonl") else None)
    samples = pw.load_samples(power_path)
    concurrency = meta.get("concurrency") or 1
    cmp = compare.compare(path, gold_only=gold_only)
    wl_path = os.path.join(compare.BENCH, "workloads", cmp["workload"] + ".jsonl")
    shape_of = {}
    if os.path.exists(wl_path):
        for w in compare.read_jsonl(wl_path):
            sh = w["shape"]
            shape_of[w["id"]] = "s%s_q%s" % (sh["state_tokens"], sh["n_questions"])

    by_shape = {}
    ok = [r for r in res if "latency_ms" in r and "error" not in r and "unsupported" not in r]
    n_unsup = sum(1 for r in res if "unsupported" in r)
    for r in res:
        key = shape_of.get(r["id"], "?")
        s = by_shape.setdefault(key, {"lat": [], "energy": [], "errors": 0, "unsupported": 0, "n": 0})
        s["n"] += 1
        if "unsupported" in r:
            s["unsupported"] += 1
            continue
        if "latency_ms" not in r or "error" in r:
            s["errors"] += 1
            continue
        s["lat"].append(r["latency_ms"])
        if samples and concurrency <= 1:
            e = pw.energy_mj(samples, r["t_start"], r["t_start"] + r["latency_ms"] / 1000.0)
            if e is not None:
                s["energy"].append(e)

    shapes = []
    for key in sorted(by_shape, key=compare.shape_order):
        s = by_shape[key]
        c = cmp["shapes"].get(key, {})
        shapes.append({"shape": key, "n": s["n"], "unsupported": s["unsupported"],
                       "coverage": (s["n"] - s["unsupported"]) / s["n"] if s["n"] else None,
                       "errors": s["errors"],
                       "p50": pct(s["lat"], 50), "p95": pct(s["lat"], 95), "p99": pct(s["lat"], 99),
                       "agreement": c.get("agreement"), "max_drift": c.get("max_drift"),
                       "gold_acc": c.get("gold_acc"), "flag": c.get("flag", ""),
                       "energy_mj": (sum(s["energy"]) / len(s["energy"])) if s["energy"] else None})

    run = {"file": path, "contender": meta.get("contender"), "variant": meta.get("variant"),
           "model": meta.get("model"), "workload": cmp["workload"], "mode": meta.get("mode"),
           "backend": meta.get("backend"), "concurrency": concurrency, "load_ms": meta.get("load_ms"),
           "max_rss_mib": None, "footprint_mib": None, "avg_power_w": None, "energy_per_req_mj": None,
           "thr_first_min": None, "thr_last_min": None, "agreement": cmp["total"].get("agreement"),
           "flag": cmp["total"].get("flag", ""), "requests": len(res), "unsupported": n_unsup,
           "coverage": cmp["total"].get("coverage"), "errors": len(res) - len(ok) - n_unsup}
    t = run_info.get("time", {})
    if t.get("max_rss_bytes"):
        run["max_rss_mib"] = t["max_rss_bytes"] / 2 ** 20
    if t.get("peak_footprint_bytes"):
        run["footprint_mib"] = t["peak_footprint_bytes"] / 2 ** 20
    if ok:
        t0 = min(r["t_start"] for r in ok)
        t1 = max(r["t_start"] + r["latency_ms"] / 1000.0 for r in ok)
        if samples:
            e = pw.energy_mj(samples, t0, t1)
            if e is not None and t1 > t0:
                run["avg_power_w"] = e / 1000.0 / (t1 - t0)
                run["energy_per_req_mj"] = e / len(ok)
        if t1 - t0 >= 120:
            ends = [r["t_start"] + r["latency_ms"] / 1000.0 for r in ok]
            run["thr_first_min"] = sum(1 for x in ends if x <= t0 + 60) / 60.0
            run["thr_last_min"] = sum(1 for x in ends if x >= t1 - 60) / 60.0
    return run, shapes


def markdown(items):
    out = ["## Runs", "",
           "| contender | variant | model | workload | mode | conc | requests | unsupported | coverage | errors | load ms | peak RSS MiB | footprint MiB | avg W | mJ/req | req/s first min | req/s last min | agreement | flag |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, _ in items:
        out.append("| %s | %s | %s | %s | %s | %s | %d | %d | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            run["contender"], run["variant"], run["model"], run["workload"], run["mode"], run["concurrency"],
            run["requests"], run["unsupported"], compare.fmt(run["coverage"], True), run["errors"], f1(run["load_ms"], 0), f1(run["max_rss_mib"], 0),
            f1(run["footprint_mib"], 0), f1(run["avg_power_w"], 2), f1(run["energy_per_req_mj"], 0),
            f1(run["thr_first_min"], 2), f1(run["thr_last_min"], 2),
            compare.fmt(run["agreement"], True), run["flag"]))
    out += ["", "## Per shape", "",
            "| contender | variant | model | workload | shape | n | unsupported | coverage | errors | p50 ms | p95 ms | p99 ms | agreement | max drift | gold acc | mJ/req | flag |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, shapes in items:
        for s in shapes:
            out.append("| %s | %s | %s | %s | %s | %d | %d | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                run["contender"], run["variant"], run["model"], run["workload"], s["shape"], s["n"],
                s["unsupported"], compare.fmt(s["coverage"], True), s["errors"],
                f1(s["p50"]), f1(s["p95"]), f1(s["p99"]), compare.fmt(s["agreement"], True),
                compare.fmt(s["max_drift"]), compare.fmt(s["gold_acc"], True), f1(s["energy_mj"], 0), s["flag"]))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--power")
    ap.add_argument("--gold-only", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--json")
    args = ap.parse_args()
    items = [summarize(p, args.power, args.gold_only) for p in args.results]
    md = markdown(items)
    print(md)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md + "\n")
    if args.json:
        with open(args.json, "w") as f:
            json.dump([{"run": r, "shapes": s} for r, s in items], f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
