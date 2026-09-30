"""Quick agreement check of adapter output against bench/reference/<model>/<workload>.jsonl.

Used only when bench/harness/compare.py does not exist. Per question: choice label, score
level (argmax of the level probabilities) and noul side (p >= 0.5) must match; also reports
the largest absolute probability difference. Only repeat 0 of each id is compared.

usage: smoke_compare.py REFERENCE.jsonl RESULT.jsonl [RESULT.jsonl ...]
"""

import json
import sys


def load(path):
    out, meta = {}, None
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("type") == "meta":
            meta = r
            continue
        if r.get("type", "result") != "result" or r.get("repeat", 0) != 0:
            continue
        out.setdefault(r["id"], r)
    return meta, out


def label(ans):
    t = ans.get("type")
    if t == "choice":
        return ("choice", ans.get("choice"))
    if t == "score":
        p = ans.get("probabilities") or {}
        if p:
            return ("score", max(p, key=lambda k: p[k]))
        return ("score", round(float(ans.get("score", 0))))
    if t == "noul":
        return ("noul", float(ans.get("noul", 0)) >= 0.5)
    return (t, None)


def probs(ans):
    if ans.get("type") == "noul":
        p = float(ans.get("noul", 0))
        return {"false": 1 - p, "true": p}
    return {str(k): float(v) for k, v in (ans.get("probabilities") or {}).items()}


def compare(ref, res):
    n = agree = 0
    maxdiff = 0.0
    unsupported = errors = missing = 0
    for rid, r in ref.items():
        got = res.get(rid)
        if got is None:
            missing += 1
            continue
        if "unsupported" in got:
            unsupported += 1
            continue
        if "error" in got or "answers" not in got:
            errors += 1
            continue
        for qid, ra in r["answers"].items():
            ga = got["answers"].get(qid)
            n += 1
            if ga is None:
                continue
            if label(ra) == label(ga):
                agree += 1
            rp, gp = probs(ra), probs(ga)
            for k in rp:
                if k in gp:
                    maxdiff = max(maxdiff, abs(rp[k] - gp[k]))
    return dict(questions=n, agree=agree, agreement=(agree / n if n else None), max_prob_diff=round(maxdiff, 4),
                requests=len(ref), unsupported=unsupported, errors=errors, missing=missing)


def main():
    _, ref = load(sys.argv[1])
    for path in sys.argv[2:]:
        meta, res = load(path)
        s = compare(ref, res)
        tag = f"{meta.get('contender')}/{meta.get('variant')}/{meta.get('model')}" if meta else path
        ag = "n/a" if s["agreement"] is None else f"{100 * s['agreement']:.2f}%"
        print(json.dumps({"run": tag, "agreement_pct": ag, **s}))


if __name__ == "__main__":
    main()
