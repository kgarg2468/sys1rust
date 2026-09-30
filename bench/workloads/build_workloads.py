"""Build bench/workloads/{smoke,correctness,timing}.jsonl.

Run with the upstream venv (needs transformers + pandas):
    source bench/env.sh
    bench/contenders/laya-upstream/.venv/bin/python bench/workloads/build_workloads.py

Deterministic: same pinned datasets + same seed give byte-identical files.
See README.md in this folder for the construction.
"""
import collections
import glob
import json
import os
import random

import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BENCH, "workloads")
LOCK = json.load(open(os.path.join(BENCH, "models.lock.json")))
SEED = 20260928

TD_REPO, TD_SHA = "LocalLLaMA/typed-decisions", "f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8"
TK_REPO, TK_SHA = "Tobi-Bueck/customer-support-tickets", "ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb"
TK_FILE = "dataset-tickets-multi-lang-4-20k.csv"

BUCKETS = [64, 128, 256, 512]
NQS = [1, 4, 10]
N_SMOKE, N_CORRECT, N_TIMING = 2, 25, 20

# q10 requests borrow the 5 questions of a thematically close workflow (no gold for those).
PARTNER = {
    "agent_trace_observability": "security_incidents",
    "security_incidents": "agent_trace_observability",
    "customer_service": "invoice_processing",
    "invoice_processing": "customer_service",
}
SHORT = {"agent_trace_observability": "trace", "security_incidents": "sec",
         "customer_service": "cs", "invoice_processing": "inv"}

# Upstream research/scripts/bench_apps.py "app.support_triage" question, verbatim.
QUEUES = {"Technical Support": "technical problems, bugs, outages, integrations",
          "Product Support": "help using a product or feature",
          "Customer Service": "general account or service questions",
          "IT Support": "internal IT, devices, access, networks",
          "Billing and Payments": "invoices, charges, refunds, payment methods",
          "Returns and Exchanges": "returning or exchanging an item",
          "Service Outages and Maintenance": "downtime, outages, scheduled maintenance",
          "Sales and Pre-Sales": "pricing, quotes, buying",
          "Human Resources": "employment, payroll, leave, hiring",
          "General Inquiry": "anything else"}
QUEUE_Q = {"type": "choice", "instructions": "Which support queue should handle this ticket?",
           "criteria": QUEUES}

FORMATS = {
    "default": lambda o: json.dumps(o, ensure_ascii=False),   # == laya serialize_state(dict)
    "compact": lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":")),
    "pretty": lambda o: json.dumps(o, ensure_ascii=False, indent=2),
}
FORMAT_RANK = {"default": 0, "compact": 1, "pretty": 2}


def snapshot(key):
    e = LOCK[key]
    name = "models--" + e["repo"].replace("/", "--")
    return os.path.join(os.environ["HF_HOME"], "hub", name, "snapshots", e["sha"])


EN_TOK = AutoTokenizer.from_pretrained(os.path.join(snapshot("typed-decisions"), "tokenizer"))
ML_TOK = AutoTokenizer.from_pretrained(os.path.join(snapshot("multilingual"), "tokenizer"))


def ntok(tok, text):
    # Exactly the state ids laya/agent.py `_encode_state` builds.
    return len(tok(text.replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"])


def gold_value(qdef, g):
    if qdef["type"] == "choice":
        return str(g["label"])
    if qdef["type"] == "noul":
        return str(g["label"]).lower() == "true"
    return int(g["label"])


def load_td():
    rows = []
    for split in ("test", "train"):
        p = hf_hub_download(TD_REPO, "all/%s-00000-of-00001.parquet" % split,
                            repo_type="dataset", revision=TD_SHA)
        df = pd.read_parquet(p)
        for _, r in df.iterrows():
            qs = json.loads(r["questions"])
            g = json.loads(r["gold"])
            rows.append({"src": "typed-decisions", "id": r["id"], "split": split,
                         "workflow": r["workflow"], "obj": json.loads(r["state"]),
                         "questions": qs, "gold": {k: gold_value(qs[k], g[k]) for k in qs}})
    return rows


def load_tickets():
    p = hf_hub_download(TK_REPO, TK_FILE, repo_type="dataset", revision=TK_SHA)
    df = pd.read_csv(p)
    rows = []
    for i, r in df.iterrows():
        if r["language"] != "en" or r["queue"] not in QUEUES or not isinstance(r["body"], str):
            continue
        subj = r["subject"] if isinstance(r["subject"], str) else ""
        rows.append({"src": "support-tickets", "id": "%s:%d" % (TK_FILE, i), "split": "train",
                     "workflow": "support_tickets",
                     "obj": {"subject": subj, "body": r["body"].replace("\\n", "\n")[:3000]},
                     "questions": {"queue": QUEUE_Q}, "gold": {"queue": r["queue"]}})
    return rows


def candidates(rows, bucket, formats):
    lo, hi = 0.9 * bucket, 1.1 * bucket
    out = []
    for r in rows:
        for f in formats:
            s = FORMATS[f](r["obj"])
            n = ntok(EN_TOK, s)
            if lo <= n <= hi:
                out.append(dict(r, fmt=f, state=s, n_en=n))
                break  # one format per row, the first (preferred) that fits
    return out


def order(cands, rng):
    """test before train, default before compact before pretty, round-robin over workflows."""
    groups = collections.defaultdict(list)
    for c in cands:
        groups[(c["split"] != "test", FORMAT_RANK[c["fmt"]], c["workflow"])].append(c)
    for g in groups.values():
        rng.shuffle(g)
    out = []
    for tier in sorted({k[:2] for k in groups}):
        wfs = sorted(k[2] for k in groups if k[:2] == tier)
        lists = [groups[tier + (w,)] for w in wfs]
        while any(lists):
            for l in lists:
                if l:
                    out.append(l.pop())
    return out


def questions_for(state_row, nq, j, td_questions_by_wf):
    """Return (questions, gold) for a request with `nq` questions; rotation index j."""
    own = state_row["questions"]
    gold = state_row["gold"]
    if state_row["src"] == "typed-decisions":
        ids = list(own)
        partner = PARTNER[state_row["workflow"]]
        extra = [("x_%s_%s" % (SHORT[partner], k), v) for k, v in td_questions_by_wf[partner].items()]
        if nq == 1:
            pick = [ids[j % 5]]
            items = [(k, own[k]) for k in pick]
        elif nq == 4:
            drop = ids[j % 5]
            items = [(k, own[k]) for k in ids if k != drop]
        else:
            items = [(k, own[k]) for k in ids] + extra
    else:  # support ticket: queue (gold) + customer-service questions (+ invoice for q10)
        cs = [("x_cs_%s" % k, v) for k, v in td_questions_by_wf["customer_service"].items()]
        inv = [("x_inv_%s" % k, v) for k, v in td_questions_by_wf["invoice_processing"].items()]
        if nq == 1:
            items = [("queue", own["queue"])]
        elif nq == 4:
            items = [("queue", own["queue"])] + [cs[(j + t) % 5] for t in range(3)]
        else:
            items = [("queue", own["queue"])] + cs + [inv[(j + t) % 5] for t in range(4)]
    assert len(items) == nq, (nq, len(items))
    q = {k: v for k, v in items}
    g = {k: gold[k] for k, _ in items if k in gold}
    return q, g


def make_request(prefix_id, bucket, nq, row, j, td_q):
    q, g = questions_for(row, nq, j, td_q)
    req = {"id": prefix_id, "shape": {"state_tokens": bucket, "n_questions": nq},
           "body": {"state": row["state"], "questions": q}}
    if g:
        req["gold"] = g
    src = TD_REPO + "@" + TD_SHA[:12] if row["src"] == "typed-decisions" else TK_REPO + "@" + TK_SHA[:12]
    req["meta"] = {"source": src, "row": row["id"], "split": row["split"],
                   "workflow": row["workflow"], "state_format": row["fmt"],
                   "state_tokens_en": row["n_en"], "state_tokens_multilingual": ntok(ML_TOK, row["state"]),
                   "gold_questions": sorted(g)}
    return req


def main():
    rng = random.Random(SEED)
    td = load_td()
    td_q = {}
    for r in td:
        td_q.setdefault(r["workflow"], r["questions"])
    tickets = None
    pools = {}
    for b in BUCKETS:
        cands = candidates(td, b, ["default", "compact", "pretty"])
        if len(cands) < 60:
            if tickets is None:
                tickets = load_tickets()
            # Short buckets: add English support tickets (default JSON only, like upstream bench_apps).
            tk = candidates(tickets, b, ["default"])
            rng.shuffle(tk)
            cands = order(cands, rng) + tk[: max(0, 120 - len(cands))]
        else:
            cands = order(cands, rng)
        pools[b] = cands
        by = collections.Counter((c["src"], c["split"], c["fmt"]) for c in cands)
        print("bucket %d: %d candidate states %s" % (b, len(cands), dict(by)))

    files = {"smoke": [], "correctness": [], "timing": []}
    for b in BUCKETS:
        L = pools[b]
        for qi, nq in enumerate(NQS):
            for j in range(N_SMOKE):
                files["smoke"].append(make_request("s%d_q%d_%03d" % (b, nq, j), b, nq, L[j % len(L)], j, td_q))
            for j in range(N_CORRECT):
                row = L[(qi * N_CORRECT + j) % len(L)]
                files["correctness"].append(make_request("s%d_q%d_%03d" % (b, nq, j), b, nq, row, j, td_q))
    # Timing: shapes interleaved round-robin so every shape is hit early in a run.
    timing = collections.defaultdict(list)
    for b in BUCKETS:
        L = pools[b]
        for qi, nq in enumerate(NQS):
            for j in range(N_TIMING):
                row = L[(len(NQS) * N_CORRECT + qi * N_TIMING + j) % len(L)]
                timing[(b, nq)].append(make_request("s%d_q%d_%03d" % (b, nq, j), b, nq, row, j + 1, td_q))
    for j in range(N_TIMING):
        for b in BUCKETS:
            for nq in NQS:
                files["timing"].append(timing[(b, nq)][j])

    for name, reqs in files.items():
        with open(os.path.join(OUT, name + ".jsonl"), "w") as f:
            for r in reqs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_gold = sum(len(r.get("gold", {})) for r in reqs)
        n_q = sum(r["shape"]["n_questions"] for r in reqs)
        off = [r["id"] for r in reqs
               if abs(r["meta"]["state_tokens_en"] - r["shape"]["state_tokens"]) > 0.1 * r["shape"]["state_tokens"]]
        assert not off, off
        print("%s: %d requests, %d questions, %d with gold" % (name, len(reqs), n_q, n_gold))


if __name__ == "__main__":
    main()
