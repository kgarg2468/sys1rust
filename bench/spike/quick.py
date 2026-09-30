"""One-line summary of a result file: p50/p95 for three shapes, geo p50, spike share, MLX memory, load ms."""
import os, json, math, statistics as st, sys
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "")
rows = [json.loads(l) for l in open(sys.argv[1])]
meta, rows = rows[0], [r for r in rows[1:] if "latency_ms" in r]
wname = sys.argv[1].split("/")[-1].split("-")[0]
wl = {}
for l in open(B + "workloads/%s.jsonl" % wname):
    w = json.loads(l); wl[w["id"]] = "s%s_q%s" % (w["shape"]["state_tokens"], w["shape"]["n_questions"])
by = {}
for r in rows: by.setdefault(wl[r["id"]], []).append(r["latency_ms"])
pct = lambda v, q: sorted(v)[min(len(v) - 1, int(q * len(v)))]
med = {k: st.median(v) for k, v in by.items()}
sp = sum(r["latency_ms"] > 2 * med[wl[r["id"]]] for r in rows) / len(rows)
geo = math.exp(sum(math.log(m) for m in med.values()) / len(med))
cells = " ".join("%s %.0f/%.0f" % (s, pct(by[s], .5), pct(by[s], .95)) for s in ["s128_q1", "s512_q1", "s512_q10"] if s in by)
mem = rows[-1].get("mlx_mb", {})
print("%s geo %.1f spikes %.1f%% cache_max %s peak %s load_ms %s" % (cells, geo, 100 * sp, max(r.get("mlx_mb", {}).get("cache", 0) for r in rows), mem.get("peak"), meta.get("load_ms")))
