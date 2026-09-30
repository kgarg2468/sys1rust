#!/usr/bin/env python3
"""Compact comparison of result files: per-shape p50/p95/p99, geo-mean p50, spike share, agreement.

analyze.py RESULT.jsonl ...   (paths or globs)
"""
import glob, json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import compare, summarize

SHOW = ["s128_q1", "s512_q1", "s512_q10"]
paths = [p for a in sys.argv[1:] for p in sorted(glob.glob(a))]
print("| run | agreement | geo p50 | " + " | ".join(f"{s} p50/p95/p99" for s in SHOW) + " | spikes >2x | mean ms |")
print("|---|---|---|" + "---|" * len(SHOW) + "---|---|")
for p in paths:
    run, shapes = summarize.summarize(p)
    sh = {s["shape"]: s for s in shapes}
    rows = [r for r in compare.read_jsonl(p) if r.get("type") == "result" and "latency_ms" in r]
    wl = {w["id"]: "s%s_q%s" % (w["shape"]["state_tokens"], w["shape"]["n_questions"])
          for w in compare.read_jsonl(os.path.join(compare.BENCH, "workloads", run["workload"] + ".jsonl"))}
    by = {}
    for r in rows:
        by.setdefault(wl[r["id"]], []).append(r["latency_ms"])
    med = {k: summarize.pct(v, 50) for k, v in by.items()}
    spikes = sum(1 for r in rows if r["latency_ms"] > 2 * med[wl[r["id"]]]) / max(1, len(rows))
    p50s = [s["p50"] for s in shapes if s["p50"]]
    geo = math.exp(sum(math.log(x) for x in p50s) / len(p50s)) if p50s else None
    cells = []
    for s in SHOW:
        x = sh.get(s)
        cells.append("%.1f / %.0f / %.0f" % (x["p50"], x["p95"], x["p99"]) if x and x["p50"] else "-")
    mean = sum(r["latency_ms"] for r in rows) / max(1, len(rows))
    name = "%s %s %s %s" % (run["contender"], run["variant"], run["workload"], os.path.basename(p).split("-", 1)[1][:-6])
    print("| %s | %s | %s | %s | %.1f%% | %.1f |" % (name, compare.fmt(run["agreement"], True),
          "%.1f" % geo if geo else "-", " | ".join(cells), 100 * spikes, mean))
