#!/usr/bin/env python3
"""Compare contender results with the fp32 CPU reference and with gold answers, per shape.

Usage:
  compare.py RESULT.jsonl [RESULT.jsonl ...] [--reference REF.jsonl] [--workload WL.jsonl]
             [--gold-only] [--json OUT.json]

Defaults: the model comes from the result's meta line, the workload name from the result file
name (<workload>-<timestamp>.jsonl), the reference from bench/reference/<model>/<workload>.jsonl
and gold from bench/workloads/<workload>.jsonl. `--gold-only` skips the reference (contenders
that run different weights: cbjev, opendecider-nano, decision-modernbert).

Categorical answer per question:
  choice  the `choice` label (argmax of `probabilities` if `choice` is missing)
  noul    `noul` >= 0.5
  score   argmax of `probabilities` (level index); without probabilities, an integer `score`
          is taken as the level, a fractional one is compared as a value within 0.05
Probability drift per question: max |p - p_ref| over options (noul: |noul - noul_ref|).
Near tie: the reference's top-two margin is below 0.05 (noul: |2p - 1| < 0.05).
Every repeat of a request is compared. Agreement below 99% is flagged with "LOW".

An answer agrees only if its `type` matches the reference's type. Gold checks also require the
type of the workload question.

Unsupported vs error: a result line with "unsupported" (the contender declines the request by
design, for example a state longer than its fixed bucket) is counted apart from a line with
"error" (it tried and failed). `coverage` is supported requests / requests, where supported means
not "unsupported". Agreement, drift and gold accuracy leave out unsupported requests, so they
describe what the contender does on the requests it accepts; read them together with coverage.
A question the contender should have answered but did not (left out of a result, or part of a
request that ended in "error") is counted in `missing` and as a disagreement and a gold miss.
"""
import argparse
import json
import os
import re
import sys
from collections import OrderedDict, defaultdict

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEAR_TIE = 0.05
FLAG_BELOW = 0.99


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def workload_name(result_path):
    base = os.path.basename(result_path)
    m = re.match(r"(.+?)-\d{8}T\d{6}.*\.jsonl$", base)
    return m.group(1) if m else os.path.splitext(base)[0]


def argmax_key(probs):
    return max(probs, key=lambda k: probs[k]) if probs else None


def categorical(a):
    t = a.get("type")
    if t == "choice":
        return a.get("choice", argmax_key(a.get("probabilities") or {}))
    if t == "noul":
        v = a.get("noul")
        return None if v is None else bool(float(v) >= 0.5)
    if t == "score":
        p = a.get("probabilities")
        if p:
            return int(argmax_key(p))
        s = a.get("score")
        if s is not None and float(s) == int(float(s)):
            return int(float(s))
        return None
    return None


def margin(a):
    t = a.get("type")
    if t == "noul":
        return abs(2 * float(a["noul"]) - 1)
    p = sorted((a.get("probabilities") or {}).values(), reverse=True)
    return p[0] - p[1] if len(p) > 1 else 1.0


def drift(a, r):
    if a.get("type") != r.get("type"):
        return None
    if r.get("type") == "noul":
        if a.get("noul") is None:
            return None
        return abs(float(a["noul"]) - float(r["noul"]))
    pa, pr = a.get("probabilities"), r.get("probabilities")
    if not pa or not pr:
        return None
    return max(abs(float(pa.get(k, 0.0)) - float(v)) for k, v in pr.items())


def agree(a, r):
    # Without this check a score level 1 would equal a noul True, since 1 == True in Python.
    if a.get("type") != r.get("type"):
        return False
    ca, cr = categorical(a), categorical(r)
    if a.get("type") == "score" and ca is None and a.get("score") is not None:
        return abs(float(a["score"]) - float(r["score"])) <= 0.05
    return ca is not None and ca == cr


def new_stats():
    return {"requests": 0, "errors": 0, "unsupported": 0, "questions": 0, "agree": 0,
            "near_tie_disagree": 0, "max_drift": 0.0, "sum_drift": 0.0, "n_drift": 0,
            "gold_n": 0, "gold_ok": 0, "ref_gold_ok": 0, "missing": 0}


def compare(result_path, reference_path=None, workload_path=None, gold_only=False):
    rows = read_jsonl(result_path)
    meta = next((r for r in rows if r.get("type") == "meta"), {})
    results = [r for r in rows if r.get("type") == "result"]
    wname = workload_name(result_path)
    model = meta.get("model")
    if workload_path is None:
        workload_path = os.path.join(BENCH, "workloads", wname + ".jsonl")
    wl = {r["id"]: r for r in read_jsonl(workload_path)} if os.path.exists(workload_path) else {}
    ref = {}
    if not gold_only:
        if reference_path is None and model:
            reference_path = os.path.join(BENCH, "reference", model, wname + ".jsonl")
        if reference_path and os.path.exists(reference_path):
            ref = {r["id"]: r for r in read_jsonl(reference_path)
                   if r.get("type") == "result" and "answers" in r}
        else:
            reference_path = None
    shapes = OrderedDict()
    total = new_stats()
    seen = set()
    for res in results:
        rid = res["id"]
        seen.add(rid)
        w = wl.get(rid, {})
        sh = w.get("shape") or {}
        key = "s%s_q%s" % (sh.get("state_tokens", "?"), sh.get("n_questions", "?"))
        for st in (shapes.setdefault(key, new_stats()), total):
            st["requests"] += 1
        if "unsupported" in res:
            for st in (shapes[key], total):
                st["unsupported"] += 1
            continue
        if "error" in res:
            for st in (shapes[key], total):
                st["errors"] += 1
            answers = {}
        else:
            answers = res.get("answers") or {}
        ref_ans = (ref.get(rid) or {}).get("answers")
        gold = w.get("gold") or {}
        wq = (w.get("body") or {}).get("questions") or {}
        expected = set(wq) | set(ref_ans or {})
        missing = expected - set(answers)
        upd = defaultdict(float)
        for qid in missing:
            upd["missing"] += 1
            if ref_ans is not None and qid in ref_ans:
                upd["questions"] += 1
            if qid in gold:
                upd["gold_n"] += 1
                if ref_ans is not None and qid in ref_ans:
                    upd["ref_gold_ok"] += categorical(ref_ans[qid]) == gold[qid]
        for st in (shapes[key], total):
            for k, v in upd.items():
                st[k] += v
        for qid, a in answers.items():
            upd = defaultdict(float)
            if ref_ans is not None and qid in ref_ans:
                r = ref_ans[qid]
                upd["questions"] += 1
                ok = agree(a, r)
                upd["agree"] += ok
                if not ok and margin(r) < NEAR_TIE:
                    upd["near_tie_disagree"] += 1
                d = drift(a, r)
                if d is not None:
                    upd["sum_drift"] += d
                    upd["n_drift"] += 1
                    upd["_drift"] = d
            if qid in gold:
                qtype = (wq.get(qid) or {}).get("type")
                upd["gold_n"] += 1
                upd["gold_ok"] += (qtype is None or a.get("type") == qtype) and categorical(a) == gold[qid]
                if ref_ans is not None and qid in ref_ans:
                    upd["ref_gold_ok"] += categorical(ref_ans[qid]) == gold[qid]
            for st in (shapes[key], total):
                for k, v in upd.items():
                    if k == "_drift":
                        st["max_drift"] = max(st["max_drift"], v)
                    else:
                        st[k] += v
    expected = set(wl) if wl else set()
    out = {"result": result_path, "meta": meta, "workload": wname, "reference": reference_path,
           "not_run": len(expected - seen) if expected else None,
           "shapes": {k: finish(v) for k, v in sorted(shapes.items(), key=lambda kv: shape_order(kv[0]))},
           "total": finish(total)}
    return out


def shape_order(key):
    m = re.match(r"s(\d+)_q(\d+)", key)
    return (int(m.group(1)), int(m.group(2))) if m else (10 ** 9, 0)


def finish(st):
    s = dict(st)
    s["supported"] = st["requests"] - st["unsupported"]
    s["coverage"] = round(s["supported"] / st["requests"], 4) if st["requests"] else None
    s["agreement"] = round(st["agree"] / st["questions"], 4) if st["questions"] else None
    s["mean_drift"] = round(st["sum_drift"] / st["n_drift"], 5) if st["n_drift"] else None
    s["max_drift"] = round(st["max_drift"], 4) if st["n_drift"] else None
    s["gold_acc"] = round(st["gold_ok"] / st["gold_n"], 4) if st["gold_n"] else None
    s["ref_gold_acc"] = round(st["ref_gold_ok"] / st["gold_n"], 4) if st["gold_n"] and st["questions"] else None
    s["flag"] = "LOW" if s["agreement"] is not None and s["agreement"] < FLAG_BELOW else ""
    for k in ("sum_drift", "n_drift"):
        s.pop(k)
    return s


def fmt(v, pct=False):
    if v is None:
        return "-"
    if pct:
        return "%.1f%%" % (100 * v)
    return str(v)


def markdown(rep):
    m = rep["meta"]
    lines = ["### %s / %s / %s on %s" % (m.get("contender"), m.get("variant"), m.get("model"), rep["workload"]),
             "",
             "reference: `%s`" % (rep["reference"] or "none (gold only)"),
             "",
             "| shape | requests | unsupported | coverage | errors | questions | agreement | near-tie flips | max drift | mean drift | gold acc | ref gold acc | flag |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in list(rep["shapes"].items()) + [("all", rep["total"])]:
        lines.append("| %s | %d | %d | %s | %d | %d | %s | %d | %s | %s | %s | %s | %s |" % (
            k, s["requests"], s["unsupported"], fmt(s["coverage"], True), s["errors"], s["questions"],
            fmt(s["agreement"], True),
            s["near_tie_disagree"], fmt(s["max_drift"]), fmt(s["mean_drift"]), fmt(s["gold_acc"], True),
            fmt(s["ref_gold_acc"], True), s["flag"]))
    if rep["total"]["missing"]:
        lines.append("")
        lines.append("%d questions have no answer; they count as disagreements and gold misses."
                     % rep["total"]["missing"])
    if rep.get("not_run"):
        lines.append("")
        lines.append("%d workload requests have no result line." % rep["not_run"])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--reference")
    ap.add_argument("--workload")
    ap.add_argument("--gold-only", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()
    reps = [compare(p, args.reference, args.workload, args.gold_only) for p in args.results]
    for rep in reps:
        print(markdown(rep))
        print()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(reps if len(reps) > 1 else reps[0], f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
