"""Sustained-run summary: req/s per minute, p50/p95 per shape, spikes, MLX cache over time."""
import os, json, statistics as st, sys
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "")
wl = {}
for l in open(B + "workloads/timing.jsonl"):
    w = json.loads(l); wl[w["id"]] = "s%s_q%s" % (w["shape"]["state_tokens"], w["shape"]["n_questions"])
for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f)]
    rows = [r for r in rows[1:] if "latency_ms" in r]
    t0 = rows[0]["t_start"]
    per_min = [0] * 6
    for r in rows:
        per_min[min(5, int((r["t_start"] - t0) // 60))] += 1
    by = {}
    for r in rows: by.setdefault(wl[r["id"]], []).append(r["latency_ms"])
    med = {k: st.median(v) for k, v in by.items()}
    pct = lambda v, q: sorted(v)[min(len(v) - 1, int(q * len(v)))]
    sp = sum(r["latency_ms"] > 2 * med[wl[r["id"]]] for r in rows) / len(rows)
    caches = [r.get("mlx_mb", {}).get("cache", 0) for r in rows]
    print(f.split("results/")[-1])
    print("  requests %d, req/s by minute %s" % (len(rows), " ".join("%.2f" % (n / 60) for n in per_min[:5])))
    print("  " + " ".join("%s %.0f/%.0f" % (s, pct(by[s], .5), pct(by[s], .95)) for s in ["s128_q1", "s512_q1", "s512_q10"]),
          "spikes %.2f%%" % (100 * sp), "cache first/max/last %s/%s/%s" % (caches[0], max(caches), caches[-1]))
