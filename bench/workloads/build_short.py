"""Build bench/workloads/short.jsonl: short single-question requests for fixed-shape contenders.

Neural Engine contenders with small fixed input shapes (jevalaya ANE: at most 96 tokens and one
question) cannot run the 64-512 token buckets. This file gives them 40 requests whose whole
Laya sequence ([CLS] head [SEP] options [SEP] state [SEP]) stays under 96 tokens for all three
pinned checkpoints, measured with upstream's own `laya.common.build_sequence`.

Run with the upstream venv:
    source bench/env.sh
    bench/contenders/laya-upstream/.venv/bin/python bench/workloads/build_short.py

Deterministic. See README.md in this folder.
"""
import json
import os
import random

import pandas as pd
from huggingface_hub import hf_hub_download
from laya.agent import Agent
from laya.common import build_sequence
from transformers import AutoTokenizer

import build_workloads as bw

N = 40
STATE_MIN, STATE_MAX = 24, 40   # English ModernBERT state tokens
SEQ_MAX = 95                    # whole sequence, every checkpoint: "under 96"
SHAPE = {"state_tokens": 32, "n_questions": 1}

# Written for this benchmark (not upstream questions). Gold comes from the ticket's own
# `type` and `priority` columns. Heads are short so that head + options + state fit in 96.
QUESTIONS = [
    ("ticket_type", {"type": "choice", "instructions": "What kind of support ticket is this?",
                     "criteria": {"Incident": "something is broken or not working",
                                  "Request": "asks for information or a new service",
                                  "Problem": "a recurring or underlying issue",
                                  "Change": "asks to change an existing setup"}},
     lambda r: r["type"]),
    ("priority", {"type": "score", "instructions": "How urgent is this ticket?",
                  "criteria": ["Low; it can wait.", "Medium; handle in the normal queue.",
                               "High; it needs attention soon."]},
     lambda r: ["low", "medium", "high"].index(r["priority"])),
    ("reports_fault", {"type": "noul",
                       "instructions": "This ticket reports a fault: something is broken, failing or not working.",
                       "criteria": {"true": "A fault is reported.",
                                    "false": "A question, request or change, with nothing broken."}},
     lambda r: r["type"] in ("Incident", "Problem")),
]

MODELS = ["typed-decisions", "multilingual", "english"]
TOKS, CFGS = {}, {}
for m in MODELS:
    snap = bw.snapshot(m)
    TOKS[m] = AutoTokenizer.from_pretrained(os.path.join(snap, "tokenizer"))
    CFGS[m] = json.load(open(os.path.join(snap, "rl_agent_config.json")))


def seq_tokens(m, state, qdef):
    q = Agent._to_internal(qdef)
    cfg = CFGS[m]
    ids, _ = build_sequence(TOKS[m], state, q, cfg.get("max_len", 512), cfg.get("head_max_len", 192))
    return len(ids)


def main():
    rng = random.Random(bw.SEED + 32)
    p = hf_hub_download(bw.TK_REPO, bw.TK_FILE, repo_type="dataset", revision=bw.TK_SHA)
    df = pd.read_csv(p)
    cands = []
    for i, r in df.iterrows():
        if (r["language"] != "en" or not isinstance(r["body"], str)
                or r["type"] not in ("Incident", "Request", "Problem", "Change")
                or r["priority"] not in ("low", "medium", "high")):
            continue
        subj = r["subject"] if isinstance(r["subject"], str) else ""
        state = bw.FORMATS["default"]({"subject": subj, "body": r["body"].replace("\\n", "\n")[:3000]})
        n = bw.ntok(bw.EN_TOK, state)
        if STATE_MIN <= n <= STATE_MAX:
            cands.append((i, r, state, n))
    rng.shuffle(cands)
    reqs, used = [], 0
    for i, r, state, n in cands:
        if len(reqs) == N:
            break
        j = len(reqs)
        qid, qdef, gold_of = QUESTIONS[j % len(QUESTIONS)]
        seq = {m: seq_tokens(m, state, qdef) for m in MODELS}
        used += 1
        if max(seq.values()) > SEQ_MAX:
            continue
        reqs.append({"id": "s32_q1_%03d" % j, "shape": dict(SHAPE),
                     "body": {"state": state, "questions": {qid: qdef}},
                     "gold": {qid: gold_of(r)},
                     "meta": {"source": bw.TK_REPO + "@" + bw.TK_SHA[:12], "row": "%s:%d" % (bw.TK_FILE, i),
                              "split": "train", "workflow": "support_tickets", "state_format": "default",
                              "state_tokens_en": n, "state_tokens_multilingual": bw.ntok(bw.ML_TOK, state),
                              "seq_tokens": seq, "gold_questions": [qid]}})
    assert len(reqs) == N, len(reqs)
    with open(os.path.join(bw.OUT, "short.jsonl"), "w") as f:
        for q in reqs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    allseq = [v for q in reqs for v in q["meta"]["seq_tokens"].values()]
    print("short: %d requests from %d candidates (%d looked at), state %d-%d en tokens, sequence %d-%d tokens"
          % (len(reqs), len(cands), used, min(q["meta"]["state_tokens_en"] for q in reqs),
             max(q["meta"]["state_tokens_en"] for q in reqs), min(allseq), max(allseq)))


if __name__ == "__main__":
    main()
