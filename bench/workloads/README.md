# Workloads

`build_workloads.py` builds `smoke`, `correctness` and `timing`, and `build_short.py` builds `short`
(deterministic, seed 20260928). No script builds `cold.jsonl` or `long.jsonl`; see "Cold and long
workloads" below.

The builders need Python with `transformers`, `pandas` and `huggingface_hub`. `build_short.py` also
imports upstream `laya` 0.3.21. The spike ran them in the venv of a `laya-upstream` contender that is
not in this repository. They download the pinned datasets themselves, but they load the tokenizers
and `rl_agent_config.json` of the pinned checkpoints from `$HF_HOME/hub` and fail if those
snapshots are missing. Rebuild with:

```
source bench/env.sh
PY=python3   # any Python with the packages above
$PY - <<'EOF'
import json, huggingface_hub as hub
lock = json.load(open("bench/models.lock.json"))
for key in ("typed-decisions", "multilingual", "english"):
    hub.snapshot_download(lock[key]["repo"], revision=lock[key]["sha"],
                          allow_patterns=["tokenizer/*", "rl_agent_config.json"])
EOF
$PY bench/workloads/build_workloads.py
cd bench/workloads && $PY build_short.py
```

| file | requests | per shape | questions | questions with gold | distinct states |
|---|---|---|---|---|---|
| `smoke.jsonl` | 24 | 2 | 120 | 80 | 8 |
| `correctness.jsonl` | 300 | 25 | 1,500 | 825 | 300 |
| `timing.jsonl` | 240 | 20 | 1,200 | 668 | 240 |
| `short.jsonl` | 40 | 40 (one shape) | 40 | 40 | 40 |
| `cold.jsonl` | 1 | 1 (one shape) | 1 | 1 | 1 |
| `long.jsonl` | 60 | 20 | 300 | 200 | 60 |

Shapes: state length 64 / 128 / 256 / 512 tokens times 1 / 4 / 10 questions. Question types across
the correctness file: 526 choice, 571 score, 403 noul. `short.jsonl` has a single extra shape,
32 state tokens with 1 question (see "Short workload" below).

`SMOKE_READY` marks that `smoke.jsonl` and both smoke references exist.

## Line format

```json
{"id": "s128_q4_007",
 "shape": {"state_tokens": 128, "n_questions": 4},
 "body": {"state": "<string>", "questions": {"<qid>": {"type": ..., "instructions": ..., "criteria": ...}}},
 "gold": {"<qid>": "label" | true/false | 2},
 "meta": {"source": "...", "row": "...", "split": "test|train", "workflow": "...", "state_format": "default|compact|pretty",
          "state_tokens_en": 121, "state_tokens_multilingual": 130, "gold_questions": ["..."]}}
```

- `body` is the `/v1/systemone` body minus `model`. Send it as is.
- `gold` holds only the questions whose label comes from the source dataset: a choice label
  string, a noul boolean, or a score level index. It is absent when no question in the request has gold.
- `meta.state_tokens_en` is the exact number of state ids upstream Laya builds with the English
  ModernBERT tokenizer (the `laya-typed-decisions` tokenizer at the locked sha):
  `len(tok(state.replace(tok.mask_token, " "), add_special_tokens=False).input_ids)`, the same call
  as `laya/agent.py::_encode_state`. Every request is within 10% of its bucket.
  `meta.state_tokens_multilingual` is the same count with the `laya-multilingual` (mmBERT) tokenizer.
- `short.jsonl` lines also carry `meta.seq_tokens`, the full sequence length per checkpoint.
- Ids repeat across files (each file has its own `s64_q1_000`). Results are matched to the reference
  of the same workload file.

## States

The state is always a JSON string, not an object. Upstream serializes an object state with
`json.dumps(state, ensure_ascii=False)` before tokenizing; sending the string removes any doubt
about separators, key order or escaping for contenders that are not Python. For the `default`
format the string is byte-identical to what upstream builds from the parsed object.

Sources:

1. `LocalLLaMA/typed-decisions` at `f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8` (Apache-2.0), the benchmark
   upstream reports its 0.766 on (`research/scripts/bench_local.py::build_typed_decisions`). 400 test and
   1,200 train states over four workflows, each with the same 5 questions per workflow and teacher gold labels.
   The test split is used first; train rows fill buckets the test split cannot. The
   `laya-typed-decisions` checkpoint was fine-tuned on the train split, so gold accuracy on train rows
   is optimistic; `meta.split` lets you separate them.
2. `Tobi-Bueck/customer-support-tickets` at `ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb`, file
   `dataset-tickets-multi-lang-4-20k.csv` (CC BY-NC 4.0, synthetic tickets), English rows only. Only the
   64-token bucket uses it, because only 2 typed-decisions states fit 58-70 tokens. The state is
   `{"subject", "body"}` and the question is upstream's own `app.support_triage` queue question from
   `research/scripts/bench_apps.py`, with the ticket's queue as gold. Both Laya checkpoints are
   zero-shot on this question.

Token lengths come from the natural length of the state plus one of three JSON layouts of the same
content, chosen in this order: `default` (upstream's serialization), `compact` (no spaces after
separators) or `pretty` (`indent=2`). The content and therefore the gold label are unchanged; only
whitespace differs. No typed-decisions state is truncated or padded. Both builders cut each support
ticket body to its first 3,000 characters when they load the CSV. No committed ticket reaches that
limit. Only tickets of 24 to 70 state tokens are selected, and the longest ticket body in any
workload file is 374 characters.

Composition by bucket (correctness file):

| bucket | English tokens | multilingual tokens | states |
|---|---|---|---|
| 64 | 58-70 | 53-76 | 73 support tickets, 2 typed-decisions customer-service threads (compact) |
| 128 | 116-139 | 117-163 | typed-decisions agent traces and short customer-service threads (default, compact, pretty), 53 test / 22 train |
| 256 | 232-281 | 246-325 | typed-decisions customer service, invoices, security alerts, all test split, default layout |
| 512 | 461-562 | 482-747 | typed-decisions long customer-service threads and invoices, mostly pretty-printed, 17 test / 58 train |

The 512 bucket is short of natural states: only 16 rows in either split reach 461 tokens in the
default layout, so most of it is pretty-printed JSON. The multilingual tokenizer produces up to 747
state tokens there; with a head of up to 256 tokens the multilingual checkpoint's `max_len` of 1024
can truncate the longest few states (the reference does the same, so agreement is unaffected).

## Questions

- 1 question: one of the state's own 5 questions, rotating by request index, so the three types appear.
- 4 questions: the state's own 5 minus one, rotating.
- 10 questions: the state's own 5 (with gold) plus the 5 questions of a related workflow
  (agent traces with security alerts, customer service with invoices), keyed `x_<workflow>_<qid>`, without gold.
- Support tickets (64 bucket): the queue question (gold) plus 3 customer-service questions for 4, and
  plus all 5 customer-service and 4 invoice questions for 10. Only `queue` has gold.

Borrowed questions are real typed-decisions questions of all three types, but their topic only
roughly fits the state, so they carry no gold.

## Short workload

`short.jsonl` is for contenders with small fixed input shapes, such as the jevalaya Neural Engine
model (at most 96 tokens, 1 question). Every request has one question, and the whole Laya sequence,
`[CLS] <type> question: <instructions> [SEP] [MASK] option ... [SEP] state [SEP]`, is 66 to 92 tokens
for each of the three pinned checkpoints. The count comes from upstream's own
`laya.common.build_sequence` with each checkpoint's tokenizer and `max_len` / `head_max_len`, and
is stored per checkpoint in `meta.seq_tokens`. The shape is recorded as
`{"state_tokens": 32, "n_questions": 1}`; the actual state is 24 to 40 English tokens.

- States: 40 English support tickets from the same `Tobi-Bueck/customer-support-tickets` file,
  `{"subject", "body"}` in the default layout, chosen at random (seeded) among the 1,674 tickets
  whose state is 24 to 40 tokens. Many have an empty subject.
- Questions: three questions written for this benchmark, rotating by request index. Upstream's
  queue question does not fit (133 tokens before the state), and every typed-decisions choice
  and score question needs 64 to 106 tokens before the state.
  - `ticket_type` (choice, 14 requests): Incident, Request, Problem or Change. Gold is the ticket's
    `type` column.
  - `priority` (score, 13 requests): low, medium, high as levels 0 to 2. Gold is the `priority` column.
  - `reports_fault` (noul, 13 requests): gold is true when `type` is Incident or Problem.
- The dataset labels are synthetic, and the questions are new to every checkpoint. The references
  score 0.800 (`typed-decisions`), 0.825 (`multilingual`) and 0.750 (`english`) on the 40 gold
  answers, but 40 decisions is a small sample. The file is meant for agreement with the reference.

## Cold and long workloads

No script in this repository builds these two files, and none was kept from the spike. They were
derived from the built files as described below, which matches the committed data exactly.

- `cold.jsonl` is the first line of `short.jsonl`, unchanged. Cold-start runs use it with `--warmup 0`.
- `long.jsonl` takes the 60 `s512` requests of `timing.jsonl` in file order. Each state is repeated
  twice, joined by a blank line (`state + "\n\n" + state`), so it is no longer a JSON object. The id
  prefix `s512` becomes `s1024` and `shape.state_tokens` becomes 1024. `gold` and `meta` are copied
  unchanged, so the `meta` token counts describe one copy of the state. The doubled states are 924 to
  1,126 English tokens and 965 to 1,495 multilingual tokens. The 1,024-token `max_len` cuts the end
  of 223 of the 300 question sequences with `typed-decisions` and all 300 with `multilingual`. There
  is no reference for this file; it was used for memory and speed checks, not for agreement.

## Order

`smoke.jsonl` and `correctness.jsonl` are grouped by shape. `timing.jsonl` interleaves the 12 shapes
round-robin (request 0 of every shape, then request 1, ...), so a run touches every shape in its first 12
requests. Use `--warmup 12` with it to warm each shape once.

## References

`bench/reference/<model>/<workload>.jsonl` is the output of
`contenders/laya-upstream/run --variant cpu-fp32 --model <model> --workload <file> --warmup 0`: upstream
Laya 0.3.21 (`9d955671415fc19f069b9cc998928075c1f255ec`), PyTorch 2.14.0 fp32 on CPU,
`Agent.system_one` per request. Answers are upstream's rounded (4 decimals) `/v1/systemone` answers.

| model | smoke | correctness | timing | short |
|---|---|---|---|---|
| `typed-decisions` | yes | yes | yes | yes |
| `multilingual` | yes | yes | yes | yes |
| `english` | yes | no | no | yes |

Token budgets come from each checkpoint's own `rl_agent_config.json`, which upstream `Agent` reads
at load: `max_len` 1024 and `head_max_len` 256 for `typed-decisions` and `multilingual`, 512 and 192
for `english`. No request overrides them. Upstream `docs/http-api.md` says the model fits options
into a `head_max_len=192` window; 192 is the `english` checkpoint's value and the code default, not
the value the other two checkpoints use. Contenders that rebuild the sequence themselves should
read the value from the checkpoint config.

The `english` checkpoint (the root of `convaiinnovations/laya`) truncates the longest states in the
512 bucket because of its 512-token `max_len`. Its reference does the same, so agreement is unaffected.

### Gold accuracy of the references

| model | typed-decisions test rows | typed-decisions train rows | support tickets (queue) | published (400 test cases) |
|---|---|---|---|---|
| `typed-decisions` | 0.735 (correctness, 408 decisions), 0.769 (timing, 225) | 0.834, 0.821 | 0.521, 0.328 | 0.766 |
| `multilingual` | 0.353 (correctness), 0.364 (timing) | 0.360, 0.332 | 0.658, 0.603 | 0.352 |
| `english` | smoke only: 0.425 over all 80 smoke gold answers | | | 0.362 |

The multilingual checkpoint is zero-shot on typed-decisions and is expected to be weak: upstream's
README reports 0.352 on the 400 test cases, below the 0.461 per-question majority baseline and
near the 0.318 random baseline. The references match that. The 0.275 on smoke is 80 decisions and
within noise. The multilingual reference was checked: it loads `laya-multilingual` at `e4e9ddf2`
(`jhu-clsp/mmBERT-base` encoder, the snapshot's own 256k-token tokenizer, `max_len` 1024,
`head_max_len` 256 from `rl_agent_config.json`), and calling upstream `Agent.system_one` directly with
the parsed state object, as upstream's `research/scripts/bench_local.py` does, gives identical
answers to the reference for the requests tried.
