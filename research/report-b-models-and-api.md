# Report B: System 1 models, the Jev API, benchmarks and quality

Research date: 2026-09-28. Every number below comes from a live source fetched today (TypeSafe docs, Hugging Face API and model cards, GitHub API, the Decision Index data file). Nothing is from memory. Claims are the ones listed in the earlier research notes (not in this repo) that fall in this shard (model, API, benchmark, quality).

## 1. Claims table

| # | Claim from prior chat | Status | Correct value or note | Source |
|---|---|---|---|---|
| 1 | Jev released Sept 15, Laya Sept 18 | CONFIRMED | Jev early access 2026-09-15. `convaiinnovations/laya` created on HF 2026-09-18T05:05Z, GitHub repo 2026-09-18T04:46Z. `laya-typed-decisions` 2026-09-18T17:45Z, `laya-multilingual` 2026-09-19T03:23Z. | typesafe.ai/blog/introducing-system-one-models-and-jev; datacamp.com/blog/system-one-models-jev; HF API `api/models?author=convaiinnovations` |
| 2 | Jev weights are closed | CONFIRMED | No weights, no parameter count, no architecture disclosed. Docs list only `jev-1.13.0` and aliases. | docs.typesafe.ai/models.md |
| 3 | Jev compatibility means accepting `/v1/systemone` requests | CONFIRMED | `POST https://api.typesafe.ai/v1/systemone`, request `{model, state, questions}`, response `{model, answers, usage}`. Laya, kime, cbjev, Kev, lev, OpenDecider, od1 all serve this shape. | docs.typesafe.ai/api.md; laya `docs/http-api.md` |
| 4 | NandhaKishorM/laya ~27.7k stars | CONFIRMED | 27,909 stars on 2026-09-28. | `gh api repos/NandhaKishorM/laya` |
| 5 | Python server holds a lock, one model call at a time | CONFIRMED | `laya/serve.py`: `ThreadPoolExecutor(max_workers=1)` plus an `asyncio.Lock` around every `router.predict`; `asyncio.Semaphore(LAYA_MAX_CONCURRENT=16)` admission, excess gets 503 with `Retry-After: 1`. No cross-request batching. | github.com/NandhaKishorM/laya/blob/main/laya/serve.py; docs/http-api.md "Concurrency model" |
| 6 | Batching gave 3.6-10x upstream | CORRECTED (qualified) | GPU only. 9-10x is `predict_batch` on an RTX 5060 Ti (~10 ms to ~1 ms per decision). 3.6x is `decide_batch` on Apple MPS, 8 tickets. CPU: 0.8x (PR #47), 1.6-2.4x (`decide_batch`, LangChain batch), 2.15x from length sorting on 10k tickets. On a 4-core EPYC, "Batching questions saves little on CPU." T4 per-question gain from 1 to 50 questions per call: 2.6x (english), 4.9x (multilingual). | README.md lines 596-598, 786; BENCHMARKS.md "Server CPU" and "Speed (Tesla T4)"; PR #47 |
| 7 | CPU model compute 193-584 ms per question | CONFIRMED | AWS m7a.xlarge (EPYC 9R14, 4 cores, fp32, 1 question): multilingual 193 ms, english 580 ms, typed-decisions 584 ms. Ryzen 9 6900HX best 329 ms (8 intra-op threads, 1 inter-op). | BENCHMARKS.md "Server CPU: AMD EPYC 9R14" and "Laptop CPU" |
| 8 | Weights ~842 MB fp16 | CONFIRMED | `model.safetensors` 842,609,210 bytes; header shows 421,293,830 params, F16 (3 F32 scalars: temperature). Multilingual 643.8 MB, 321,908,998 params F16. typed-decisions 842.6 MB, all F16. | safetensors headers via HTTP range requests on huggingface.co/convaiinnovations/* |
| 9 | INT8/INT4 can shift probabilities near thresholds | CONFIRMED (sharpened) | The quantization scheme matters more than bit width. Weight-only per-channel INT8: 0/20 and 26/26 argmax preserved, max shift 0.016-0.09. Per-tensor INT8: 3/20 flipped (noul crossing 0.5, one score level), drift 0.29. Dynamic activation INT8: argmax agreement 69%, worst shift 0.99. | laya PR #498; huggingface.co/nvkudva/laya-web-q8; alexander-voronkov/laya-web-poc PR #16 |
| 10 | Upstream Python/CUDA 4.6 ms per request | CONFIRMED (context added) | 4.6 ms is the TileLang fast path (`laya[fast]`, fused kernels plus CUDA graphs) on an RTX 4070 Ti SUPER, English checkpoint, 1 question, 72 tokens. Stock PyTorch path on the same card: 17.7 ms. Multilingual fast path: 2.8 ms. Decision Index lab (RTX PRO 6000): 5.8 ms median fast path, 19.8 ms stock. | BENCHMARKS.md "GPU fast path"; Decision Index `data/index.json` Laya `latency_path` |
| 11 | Laya re-reads the whole input once per question; encode once is the big gain | CONFIRMED | Each question is its own sequence `[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`; N questions means N copies of the state through the encoder in one batch. cbjev's shared layout measured 6.7x on 10 questions over 500 tokens (RTX 4090). | `rl_common.py` `build_sequence`; laya issue #49; dev.to/_tomek7667 |
| 12 | kime "10x faster" figures are targets, not measurements | CONFIRMED | README: "The table below is what the project has to hit, not what it does today." 0.72 ms per question is the W3 target on a T4 against Laya's 7.2 ms. | github.com/tamnd/kime README; spec/13-benchmarks.md |
| 13 | Encode-once needs retraining because Laya attention is bidirectional | CONFIRMED | Maintainer: "the encoder currently reads the question and the state together, so the question conditions every state token ... needs a newly trained checkpoint." Issue closed 2026-09-25: new checkpoints "not planned for the open-source models." | github.com/NandhaKishorM/laya/issues/49 |
| 14 | kime encode-once model announced, not shipped | CONFIRMED | Milestone "M3: kime-v1 students" (issue #4, opened 2026-09-23): 0/10 issues done, gated on M2 teachers. Releases v0.1.0-v0.1.4 (Sept 28-29) ship the engine running Laya's weights only. | github.com/tamnd/kime/issues/4; `gh api repos/tamnd/kime/releases` |
| 15 | Nobody has shipped an encode-once model (implied) | CORRECTED | cbjev (HF `0010101010-1/cbjev`, 2026-09-24) ships a Laya fine-tune with a packed multi-question layout: typed-decisions 0.783 vs Laya 0.768, 11.4 ms vs 75.8 ms for 10 questions over 500 tokens. Weights and code GPL-3.0. Its document block still attends to all questions, so the state encoding is not cacheable across different question sets. ikken (Apache-2.0) has the exact state-only mask but no trained checkpoint. | huggingface.co/0010101010-1/cbjev; github.com/johnmofficial16-prog/ikken |
| 16 | Base Laya 0.361 on typed-decisions | CONFIRMED | 0.361 in BENCHMARKS.md table, 0.362 in README and model card (rounding of the same run). | BENCHMARKS.md; huggingface.co/convaiinnovations/laya |
| 17 | typed-decisions checkpoint 0.766 | CONFIRMED | 0.766 accuracy, soft accuracy 0.471, Brier 0.061, ECE 0.213, score MAE 0.242. Fine-tuned on the benchmark's own 1,200-case train split. Teacher ceiling 0.735, majority class 0.461. | BENCHMARKS.md; huggingface.co/convaiinnovations/laya-typed-decisions |
| 18 | multilingual 0.352 on typed-decisions | CONFIRMED (with a documented discrepancy) | Committed JSON says 0.3515. Repo tables said 0.342 until PR #313 (merged 2026-09-24) fixed them to 0.352. The `laya-multilingual` HF model card still prints the stale 0.342 row. | laya issue #300, PR #313; huggingface.co/convaiinnovations/laya-multilingual |
| 19 | OpenDecider Nano ~400M params, 0.796 | CONFIRMED | Ettin-encoder-400m (fully fine-tuned) plus MLP head; `model.safetensors` 789.6 MB bf16 (about 395M params). typed-decisions 0.796 vs Laya-td 0.766 (+0.030, 95% CI +0.014 to +0.044) and Jev 0.754 measured through TypeSafe's API. Apache-2.0. Released 2026-09-27. | huggingface.co/manjunathshiva/opendecider-nano; huggingface.co/blog/manjunathshiva/opendecider-beats-laya-and-jev |
| 20 | decision-modernbert-base 149.6M params, higher Decision Index than Laya (self-reported) | CONFIRMED (license added) | 149.6M, ModernBERT-base fine-tune. Decision Index 0.2 self-run 20.12 (raw 39.74) vs Laya 5.51 on the same edition; not an official board entry. License CC BY-NC 4.0, non-commercial only. Only a Core ML repo exists; no PyTorch weights repo found. | huggingface.co/FluidInference/decision-modernbert-base-coreml |
| 21 | The 3.6 ms ANE headline is the multilingual checkpoint, scoring 0.352 vs 0.766 | CONFIRMED (score part) | Multilingual 0.352 zero-shot; typed-decisions 0.766 after fine-tuning. Latency part belongs to the runtime shard. | BENCHMARKS.md |
| 22 | Rust projects run "open models like Decider, Lev" | CORRECTED (disambiguated) | "Decider" is three different things: meraGPT Decider 1 (proprietary hosted API), Mapika decider-2b/4b/35b-a3b (open Qwen3.5 fine-tunes, 1.9B to 35B), and mvbalaji od1 "Open Decider" (Qwen3.5-4B). "Lev" is interfaze-ai/lev (Qwen3.5-4B LoRA) or franckverrot/lev-350m (LFM2.5 prototype). All are decoder LLM backbones, not ModernBERT encoders, so a Laya-shaped runtime does not run them. | HF API; see section 5 |

Claims in the prior notes that are outside this shard (runtime latencies for laya-mlx, Core ML, kime-bench, mlx-rs, ANE limits) were not verified here.

## 2. Task 1: what a System 1 model is, and the `/v1/systemone` API

### Definition (TypeSafe)

TypeSafe's blog defines System One models as "frontier models built to make fast, structured decisions that software can use directly." The docs page `concepts/system-one.md` adds that they "evaluate a state and return typed answers and probabilities," are "trained for calibrated decisions: their probabilities are optimized against outcomes," and "do not write replies, produce code, or generate explanations." Training is "Reinforcement Learning for Calibrated Decisions (RLCD)". TypeSafe claims 70-500 ms end-to-end latency and "40x-200x faster" than frontier LLMs; these are vendor numbers with no independent reproduction. The blog mentions "a new model architecture" with a "parallel sampler" and nothing else about architecture. Jev is the only model; `jev-latest` and `jev-preview` both resolve to `jev-1.13.0`.

Sources: typesafe.ai/blog/introducing-system-one-models-and-jev, docs.typesafe.ai/concepts/system-one.md, docs.typesafe.ai/models.md, datacamp.com/blog/system-one-models-jev.

### Request

```
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <API_KEY>
Content-Type: application/json

{
  "model": "jev-latest",
  "state": <string | object | array>,
  "questions": {
    "<your_id>": { "type": "noul" | "choice" | "score", "instructions": ..., "criteria": ... },
    ...
  }
}
```

`state` is the content to judge: a plain string, or a JSON object or array (chat logs, records, app state). `questions` is a map keyed by ids you choose; the key is not shown to the model. Answers come back under the same keys. Questions in one request are independent and evaluated against the same state; if a later judgment depends on an earlier answer you must make a second request. `instructions` and `criteria` values may be strings, objects or arrays. The advanced page adds path references into structured state (for example `ticket.messages[0].text`).

Source: docs.typesafe.ai/api.md, docs.typesafe.ai/primitives.md, docs.typesafe.ai/primitives/advanced.md.

### Question types (exactly three)

| type | criteria | answer fields | limits |
|---|---|---|---|
| `noul` (yes/no) | optional `{"true": ..., "false": ...}` clarifying each side | `type`, `noul` (probability of yes, from 0 to 1) | none stated |
| `choice` | map `option -> description or null` | `type`, `choice` (argmax option), `probabilities` (sum to 1), `confidence` | up to 255 options |
| `score` | ordered list of level descriptions | `type`, `score`, `legend` (index to level text), `probabilities`, `confidence` | 2 to 10 levels |

There is no fourth type. `noul` answers carry no `confidence` field. `confidence` for choice and score is derived from the distribution: the docs give "(3 x largest probability - 1) / 2" for three options, which is the general form `(n * p_max - 1) / (n - 1)` that Laya's docs also restate as Jev's definition. Recommended thresholds in the docs: above about 0.9 act automatically, 0.5 to 0.9 confirm or flag, below 0.5 route to a human, tuned per action.

Sources: docs.typesafe.ai/api.md, docs.typesafe.ai/confidence.md, laya docs/http-api.md.

### Response

```
{
  "model": "jev-1.13.0",
  "answers": {
    "is_urgent": { "type": "noul", "noul": 0.87 },
    "category": { "type": "choice", "choice": "billing", "probabilities": {"billing": 0.81, ...}, "confidence": 0.72 },
    "severity": { "type": "score", "score": 2, "legend": {"0": "low", ...}, "probabilities": {...}, "confidence": 0.6 }
  },
  "usage": { "input_tokens": 1234, "output_tokens": 0 }
}
```

Errors: 401 (key), 422 (validation), 429 (rate limit), 529 (overloaded). Limits: 64k tokens per request, 32k for state plus the longest question; 1,200 requests per minute and 250k tokens per second, "adjusting dynamically." Pricing $0.042 per million input tokens, output free. Text only.

### Multi-question requests

The docs say adding questions "barely changes the response time and costs only the tokens for the extra questions." The parallel-questions cookbook measured 13 questions (8 noul, 2 choice, 3 score) over a 54 KB GDPR article: 0.27 s in one call versus 2.71 s as 13 calls ("10.0x faster, 12.2x cheaper"). No maximum question count is documented. Each question is scored on its own; answers did not change with batching.

Source: docs.typesafe.ai/cookbooks/parallel_questions.md, docs.typesafe.ai/patterns/fan-out.md.

### Streaming

No streaming or SSE mode is documented anywhere in the API reference, the primitives pages, the docs index (`llms.txt`), or the Python SDK usage page. The response is one JSON object. The Python SDK exposes `TypeSafeClient.system_one(state, questions)` and `AsyncTypeSafeClient.system_one`, with `RetryPolicy(max_retries, backoff_max, timeout)`, `base_url` override, `response_model`, and `extra_body`; there is no stream method. Any local server that speaks this protocol only needs a single JSON response.

Source: docs.typesafe.ai/llms.txt, docs.typesafe.ai/sdk/python/usage.md.

### Where Laya's wire format deviates

Laya's `laya-serve` returns the same `answers` and `usage` keys plus a `routing` block and two extra per-answer fields: `answer_confidence` (probability mass on the reported answer, on every type) and `action.act_probability` (its act/escalate head). Laya's `confidence` is `1 - H(p)/log(k)` on choice and score and `max(p_yes, p_no)` on noul, which is not Jev's formula, so a Jev-tuned threshold does not transfer. Laya's HTTP server returns 413 above 64 questions per request, 100 options per choice, 32 levels per score, 512 options across all questions, 50,000 state characters or a 2 MiB body, and 422 when option texts overflow `head_max_len`. TypeSafe documents none of these caps for Jev. Jev 1.13 known issues (docs `model-jaggedness/jev-1.13.md`): literal reading of instructions, arithmetic and counting, date logic, multi-hop, degradation with large irrelevant state, and `P(noul) != 1 - P(not noul)`.

## 3. Task 2: Laya architecture and checkpoints

### Backbones and head

All three checkpoints share one design (`rl_common.py` `DecisionModel`): a fully fine-tuned bidirectional encoder, then a head trained from scratch: a 2-layer `nn.TransformerEncoder` (pre-norm, `nhead = d/64`, FFN `4d`) over the full sequence, an option-marker scorer that reads the hidden state at each option's `[MASK]` token, an act/escalate head (2 outputs), and a 3-entry temperature buffer (one per question type) plus a `temperature_by_options` table in `rl_agent_config.json` (buckets like `choice:3-5`, `noul:2`, `choice:11+`).

| checkpoint | encoder | layers / hidden / heads / FFN | vocab | params (from safetensors header) | file | `max_len` / `head_max_len` | license |
|---|---|---|---|---|---|---|---|
| `convaiinnovations/laya` | ModernBERT-large (`answerdotai/ModernBERT-large`) | 28 / 1024 / 16 / 2624 | 50,368 | 421,293,830 | 842.6 MB F16 | 512 / 192 | Apache-2.0 |
| `convaiinnovations/laya-multilingual` | mmBERT-base (`jhu-clsp/mmBERT-base`) | 22 / 768 / 12 / 1152 | 256,000 | 321,908,998 | 643.8 MB F16 (+34.4 MB tokenizer.json) | 1024 (encoder to 8,192) / 256 | Apache-2.0 |
| `convaiinnovations/laya-typed-decisions` | ModernBERT-large | 28 / 1024 / 16 / 2624 | 50,368 | 421,293,830 | 842.6 MB F16 | 1024 / 256 | Apache-2.0 |

The model card's "395M encoder + head = 421M" and "307M + head = 322M" match the headers. The multilingual embedding table alone is 256,000 x 768 = 196.6M params, 61% of that checkpoint (kime's benchmark spec calls this out as the one row that cannot reach a 10x memory target). `amp_dtype` is bf16 in every config; the stored weights are F16. The `laya` repo root also mirrors the other two checkpoints in `multilingual/` and `typed-decisions/` subfolders (2.37 GB total).

Sources: safetensors headers read via HTTP range requests; `encoder/config.json`, `rl_agent_config.json` in each HF repo; huggingface.co/convaiinnovations/laya model card "Architecture".

### Attention pattern

From `encoder/config.json` (both backbones): `global_attn_every_n_layers: 3`, `local_attention: 128`, `layer_types` alternating `full_attention, sliding_attention, sliding_attention`, `max_position_embeddings: 8192`, RoPE theta 160,000 on global layers and 10,000 on local layers, no attention bias, no causal mask. So for ModernBERT-large: 10 global layers (0, 3, 6, ..., 27) and 18 local layers with a 128-token sliding window; for mmBERT-base: 8 global and 14 local. The head's 2 layers are full attention with a key-padding mask. Laya's fast path implements the local layers as sliding-window flash attention ("16x faster than SDPA with a dense mask at L=1024").

### How a question is scored

`build_sequence` in `rl_common.py`:

```
[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
```

Each option is `[MASK]` plus up to 48 tokens of option text. Options share the `head_max_len` budget; if they overflow, each is trimmed to `max(4, (head_max_len - 16) // n_options)`. The state is appended after the head and the whole sequence is cut at `max_len`, so the English checkpoint leaves about 320 tokens for state and the 1024 checkpoints about 768. The encoder and the 2-layer head run over the sequence; the scorer turns the hidden state at each `[MASK]` into one logit; logits are softmaxed over that question's options after temperature scaling. `noul` is a 2-option question; `score` is a k-level question that also reports an expected value.

This is a cross-encoder over (question + options, state). Every question in a request becomes its own row containing a full copy of the state; `_encode_state` builds N rows, `collate_items` pads them into one batch, and `_forward` runs one forward pass over the batch. The card's "every question in a call is answered in one single forward pass" is true at the batch level, but encoder FLOPs scale linearly with the number of questions. Since PR #109 (merged 2026-09-24) the state is tokenized once and the token ids reused; encoder work is unchanged.

Sources: huggingface.co/convaiinnovations/laya/blob/main/rl_common.py; laya/agent.py `_encode_state`, `predict_batch`, `system_one`; PR #109.

### Scores on typed-decisions (test split, 400 cases, 2,000 decisions)

| model | accuracy | soft acc | Brier | ECE | score MAE |
|---|---|---|---|---|---|
| `laya-typed-decisions` | 0.766 | 0.471 | 0.061 | 0.213 | 0.242 |
| `laya` | 0.361 (0.362 in README/card) | 0.332 | 0.316 | 0.175 | 0.694 |
| `laya-multilingual` | 0.352 (JSON 0.3515) | 0.328 | 0.463 | 0.314 | 0.760 |
| Jev 1.13.0, published | 0.727 | 0.580 | 0.148 | 0.144 | 0.391 |
| teacher ceiling | 0.735 | | | | |
| majority class | 0.461 | | | | |
| random | 0.318 | | | | |

Per workflow for the fine-tuned checkpoint: invoice processing 0.804, security incidents 0.766, customer service 0.764, agent-trace observability 0.730. By primitive: noul 0.857, choice 0.733, score 0.723.

Both base checkpoints are below the majority-class baseline; BENCHMARKS.md says "All of the capability on this benchmark comes from fine-tuning." The 0.342 row for multilingual that circulated for a week matched no committed result (issue #300); PR #313 replaced it with 0.352 in the repo on 2026-09-24, and the `laya-multilingual` model card still shows 0.342.

Other headline numbers (BENCHMARKS.md, T4 Colab and CPU sweep): MASSIVE intent over 51 languages, 20 options: `laya` macro 0.2269, `laya-multilingual` 0.3661 (English alone 0.783 vs 0.657). XNLI other languages 0.521 vs 0.731. AG News 0.950 / 0.930 / 0.953 (Jev published 0.910). DAIR Emotion 0.595 / 0.530 / 0.600 (Jev 0.480). banking77 0.425 / 0.425 / 0.492 against Jev's 0.870, which the authors attribute to the option token budget (77 labels get 3-4 tokens each). Both base checkpoints ship over-confident; refitting temperatures moves ECE 0.466 to 0.081 (`laya`) and 0.314 to 0.106 (multilingual). Option-order flip rate 0.15 (laya) and 0.23 (multilingual) on 20-option MASSIVE, against Jev's 0.13.

On the Decision Index (multimodalart, 120,340 requests, 38 scored benchmarks, chance-corrected 0-100), Laya is at 6.04 on the live 0.2.1 board (data generated 2026-09-28) and was 5.51 on 0.2; Jev is 57.91 (0.2.1) and 51.67 (0.2). Laya's coverage there is 0.7594 and its calibration sample shows accuracy 0.377 at mean confidence 0.517, ECE 0.140. The board ran Laya's `Agent.accelerate(use_graphs=True)` fast path on an RTX PRO 6000: 5.8 ms median, stock 19.8 ms.

Sources: BENCHMARKS.md; laya issue #300 and PR #313; multimodalart-jev-decision-index.static.hf.space/data/index.json; systemonemodels.org/models/laya/.

### Jev's own typed-decisions number is not one number

Laya cites Jev at 0.727 (published). OpenDecider measured Jev through the API at 0.754 (choice 0.737, score 0.701, yes/no 0.843). The od1 card reports jev-1.13.0 at 0.741 measured 2026-09-25. The Decision Index notes Jev's API answers "drifted since the board run" (20 choices changed on a latency sample). Treat Jev's typed-decisions accuracy as 0.73 to 0.75 depending on harness and date.

## 4. Task 3: upstream performance and the Python server

### BENCHMARKS.md numbers (all confirmed present in the file on 2026-09-28)

CPU, AWS m7a.xlarge (EPYC 9R14, 4 physical cores, fp32, `OMP_NUM_THREADS=4`, laya 0.3.20, torch 2.14):

| checkpoint | 1 q | 5 q | 10 q | 50 q | cold load |
|---|---|---|---|---|---|
| english | 580 ms | 3,072 ms | 6,244 ms | 35,969 ms | 4.4 s |
| multilingual | 193 ms | 912 ms | 1,842 ms | 11,157 ms | 2.5 s |
| typed-decisions | 584 ms | 2,819 ms | 6,031 ms | 35,653 ms | 0.5 s |

p95 within 2% of p50. About 600 ms per question on the ModernBERT-large checkpoints and 185 ms on mmBERT-base, roughly linear in question count. Peak RSS with up to five checkpoints loaded: 9.3 GiB. The Ryzen 9 6900HX row shows the thread trap: torch defaults gave 9,396 ms p50 on a 3-question HTTP call; `set_num_threads(8)` plus `set_num_interop_threads(1)` gave 783 ms; single question in-process 329 ms best.

GPU, stock PyTorch: T4 `laya` 39.5 / 84.5 / 158.6 / 771.3 ms for 1 / 5 / 10 / 50 questions; `laya-multilingual` 32.8 / 40.1 / 72.3 / 337.4 ms (103-332 questions/s batched). GB10 (DGX Spark) 100.2 ms for one question with about 93 ms fixed per-call overhead and 7.0 ms per extra question. Intel Arc B390 XPU 29.7 ms (1 q) versus 288.2 ms CPU on the same laptop.

GPU, fast path (`pip install laya[fast]`, TileLang fused GEMM+epilogue, GEGLU, residual+LayerNorm, in-place RoPE, sliding-window flash attention, bf16 weights, one CUDA graph per (batch, length) bucket), RTX 4070 Ti SUPER, `agent.predict()` end to end including tokenization:

| checkpoint | case | stock | fast | speedup |
|---|---|---|---|---|
| laya | 1 q, 72 tok | 17.7 ms | 4.6 ms | 3.8x |
| laya | 3 q, 72 tok | 18.9 | 6.6 | 2.9x |
| laya | 30 q, 72 tok | 43.2 | 35.7 | 1.2x |
| laya | 30 q, 512 tok | 327.5 | 232.1 | 1.4x |
| multilingual | 1 q, 72 tok | 14.1 | 2.8 | 5.1x |
| multilingual | 30 q, 966 tok | 320.7 | 187.6 | 1.7x |

The authors' diagnosis: small requests are launch-overhead bound ("about 200 kernels from Python per call"), which the CUDA graph removes; large batches are GEMM bound at about 80 TFLOPS, on par with cuBLAS, so the gain there is from fused epilogues and the sliding-window kernel. Parity: fast bf16 stays within 0.046 of fp32 per option probability; fp16 within 0.009 and 864/864 argmax agreement with fp16 stock.

Batching (`predict_batch`, many states, same questions): RTX 5060 Ti about 10 ms to about 1 ms per decision (9-10x, PR #47, merged 2026-09-23); CPU about 0.8x in that PR; length sorting on CPU 2.15x on 10,000 tickets with zero decision changes; `decide_batch` on Apple MPS 3.6x (8 tickets) and 1.6x through a Router; Router-level grouping 3.4x end to end (PR #166). So the prior "3.6-10x" spans MPS and one NVIDIA card; on CPU the honest range is 0.8x to 2.4x.

Sources: BENCHMARKS.md sections "Server CPU", "Laptop CPU", "Speed (Tesla T4)", "NVIDIA GB10", "Intel Arc B390", "GPU fast path"; README.md lines 596-607 and 784-790; research/README.md "Length batching"; PR #47.

### Does the reference server serialize requests?

Yes. `laya/serve.py` (FastAPI):

- `pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya-infer")`, with the comment "One worker, because one forward pass at a time is what a single CPU or GPU Agent wants."
- `gate: asyncio.Lock`, created on first request; every `/v1/systemone` call does `async with gate: result = await loop.run_in_executor(pool, lambda: router.predict(...))`.
- `admission: asyncio.Semaphore(LAYA_MAX_CONCURRENT)` (default 16) checked before the body is read; when full, the server answers 503 "server busy, try again later" with `Retry-After: 1` rather than queueing.
- `docs/http-api.md`: "requests are handed to a single-worker executor, which means one forward pass at a time." "There is no OpenAI-compatible endpoint and no batch endpoint; run several questions in one request instead."

Concurrent clients therefore never share a forward pass; throughput under load equals single-request latency. The `Agent` itself is left unguarded so in-process callers can batch (`predict_batch`), but the HTTP server does not use that. kime's CHANGELOG number in the prior notes (8 clients, 43.3 s vs 67.1 s) is consistent with this but was not re-verified here.

## 5. Task 4: competing open models

Benchmarks in play: typed-decisions (`LocalLLaMA/typed-decisions`, Apache-2.0, 400 test cases x 5 questions = 2,000 decisions across agent-trace observability, customer service, invoice processing, security incidents; teacher-labelled, ceiling 0.735; the od1 card notes 23.5% of test states are near-duplicates of train states). Decision Index 0.2 and 0.2.1 (multimodalart; 70 models on the board on 2026-09-28; entrants need median latency under 1,000 ms per request on one RTX PRO 6000). JevBench v1.3.0 (TypeSafe's own; Jev 74.4, Laya 54.4 at rank 33). S1Bench (used by lev). Laya's own application battery.

| model | backbone | served params | weights | license | typed-decisions | Decision Index 0.2.1 (median ms on RTX PRO 6000) | notes |
|---|---|---|---|---|---|---|---|
| Laya typed-decisions | ModernBERT-large | 421M | 842.6 MB F16 | Apache-2.0 | 0.766 (fine-tuned) | Laya base: 6.04 (5.8 fast, 19.8 stock) | reference |
| OpenDecider-nano (manjunathshiva) | Ettin-encoder-400m, full fine-tune + MLP head | ~395M | 789.6 MB bf16 (2.0 GiB fp32 at run time) | Apache-2.0 | 0.796 (fine-tuned on train split); general decisions 0.680 vs Jev 0.730; Laya battery 0.656 vs Laya 0.695; ECE 0.092 | not on board | 2,048-token budget, no per-option cap (78 options in one pass); 16 ms L40S, 18 ms M4 Max, 3.8 ms/q at 50 q; distilled from Qwen3-235B-A22B and DeepSeek V4.1 Flash logprobs; English only |
| FluidInference decision-modernbert-base-coreml | ModernBERT-base | 149.6M | fp16 Core ML, fixed shapes | CC BY-NC 4.0 (non-commercial) | not reported | 20.12 self-run on 0.2 (Laya 5.51 there); not an official entry | 5.6 ms median per question on M5 Pro, p95 34 ms; up to 255 options via windowing; 499/500 answers match PyTorch fp32; no PyTorch weights repo published |
| Decider 1 (meraGPT) | undisclosed, hosted | n/a | closed | commercial API, $0.03/M | 0.768 zero-shot, KL 0.096 | not on board | 4,096-token requests, max 10 options per choice |
| decider-2b (Mapika) | Qwen3.5-2B-Base, full fine-tune, autoregressive readout | 2.27B | 3.8 GB bf16 | Apache-2.0 | not reported | 28.97 (8.1 ms with FP8 + CUDA graphs) | decider-4b 40.7 (12.6 ms); decider-35b-a3b 47.11 (101 ms) |
| od1-typed-decisions (mvbalaji, "Open Decider") | Qwen3.5-4B | ~4.7B | bf16 | Apache-2.0 | 0.796 (fine-tuned; choice 0.768, yes/no 0.867, score 0.765) | not on board | 10.7 ms single short question on H100 with CUDA graphs; od1-base zero-shot 0.582; od1-nano 0.8B 0.751 |
| lev (interfaze-ai) | Qwen3.5-4B + LoRA (~200 MB adapter) | 4.66B | adapter only | Apache-2.0 | not reported | 38.54 (70.8 ms) | S1Bench 68.9%; 69-169 ms per call on H100 |
| lev-350m (franckverrot) | LFM2.5-350M | 350M | | not checked | not reported | not on board | author calls it "a fun weekend experiment" |
| Kev (jaredpalmer) | Qwen3.5 0.8B / 4B / 9B / 27B, rank-16 LoRA + pointer head | 0.87B to 27B | adapters (45 MB at 0.8B) | Apache-2.0 | not reported | 9B 38.48, 4B 34.64, 0.8B 14.6 (41-52 ms) | trained for about $95 of H100 time |
| cbjev (tomek7667) | Laya typed-decisions and multilingual, re-fine-tuned in a packed layout | 421M / 322M | 842 MB / 643 MB | GPL-3.0 | 0.783 vs Laya 0.768 (author's run) | not on board | see task 5 |
| Verdict (heman10x/rlcd-modernbert-151m) | GLiClass ModernBERT | 151M | | not checked | not reported | 1.87 at coverage 0.31 | abstains on most rows |
| Lumma-Fev-0.1B (FrontiersMind) | Nandi-Mini-150M | 153M | 310 MB | not checked | not reported | 1.78 | smallest entrant on the board |
| Decision 1.0 Kai / Lex (llm-semantic-router) | mmBERT-base | ~300M | | "other" | not reported | 6.52 / 4.54 (30 ms) | FluidInference has Core ML ports |

Two things stand out for a local runtime. First, every model that beats Laya on typed-decisions by a clear margin at under 1B parameters (OpenDecider-nano, cbjev) is still a ModernBERT-family encoder with per-option `[MASK]` markers, so a runtime built for Laya's graph covers them with small changes (Ettin uses the same ModernBERT architecture; nano's `config.json` is `ModernBertModel` with the same layer pattern). Second, the models that score well on the broad Decision Index (Rune 57.44, AutoJev 56.4, Decider 4B 40.7, lev 38.54) are decoder LLMs of 2B to 27B parameters; encoders under 500M sit between 1.8 and 20 on that scale. Zero-shot breadth and sub-500M size do not currently coexist in any open model.

Sources: huggingface.co/manjunathshiva/opendecider-nano; huggingface.co/FluidInference/decision-modernbert-base-coreml; meragpt.com; huggingface.co/Mapika/decider-2b; huggingface.co/mvbalaji/od1-typed-decisions; huggingface.co/interfaze-ai/lev; github.com/franckverrot/lev; huggingface.co/collections/jaredpalmer/kev; Decision Index `data/index.json` (generated 2026-09-28T00:39Z); huggingface.co/datasets/LocalLLaMA/typed-decisions.

## 6. Task 5: encode the state once, reuse for every question

### Is exact reuse possible with Laya as trained? No.

Three reasons, all from the shipped code and configs:

1. Layout. The state comes after the question and options in every row. In every full-attention layer (every third layer, plus both head layers) each state token attends to the question tokens, so the state's hidden states are a function of the question. In the local layers the 128-token window straddles the head/state boundary for the first 128 state tokens.
2. Positions. RoPE is relative, so a state block that attended only to itself would be shift-invariant, but because state tokens also attend to head tokens at different relative offsets per question, even the attention pattern inside the state differs between questions.
3. Head. The 2-layer head attends across the whole row again before the markers are read.

So a KV cache of the state from one question does not produce the same logits for another question. The maintainer confirmed this on issue #49: "the encoder currently reads the question and the state together, so the question conditions every state token. Encoding the state once and reading it with per-question cross-attention needs a newly trained checkpoint." What upstream shipped instead (PR #109 for PyTorch, #343 for ONNX, both merged 2026-09-24) is tokenizing the state once per call; the issue was closed 2026-09-25 with "Changing the encoder's attention layout would need new checkpoints, which is not planned for the open-source models."

Approximate reuse (feeding cached state activations into a different question row) would change answers; nobody has published a measurement of how much, and given the near-tie sensitivity in section 7 the effect would be largest on exactly the decisions that matter.

### Who has done it

cbjev (tomek7667; code github.com/tomek7667/cbjev GPL-3.0, weights huggingface.co/0010101010-1/cbjev GPL-3.0, created 2026-09-24). Packs `[CLS] q1 | q2 | ... | q10 | document [SEP]` into one sequence with a block attention mask: questions never see each other, the document reads every question, and position ids restart for each question block so Laya's weights transfer. Fine-tuned from `laya-typed-decisions` and `laya-multilingual` on 35 public datasets over seven rounds. Author's numbers against Laya on the same RTX 4090: 10 questions over a 500-token document 11.4 ms vs 75.8 ms (6.7x); one short ticket 3.0 vs 5.4 ms; typed-decisions 0.783 vs 0.768; mean of 15 English suites 0.741 vs 0.710; MASSIVE 51 languages 0.436 vs 0.401; option-reorder flips 0.2% vs 7.8%; trails Laya on support triage (-4.0), prompt injection (-3.5), DAIR emotion (-2.5), AG News (-0.8). The runtime is a from-scratch ModernBERT forward with torch.compile fusion and one CUDA graph per 32-token shape bucket. Two caveats: the author reports that the pure "document first, cannot see the questions" variant kept losing on half the benchmarks, so cbjev's document block is conditioned on the question set, which means the encoded state cannot be cached across requests with different questions (the gain is one pass per request instead of N); and the GPL-3.0 license on both code and weights is incompatible with most product use.

ikken (johnmofficial16-prog, Apache-2.0, created 2026-09-25, 0 stars). Exact "read once, answer many" masks for ModernBERT-family encoders: state attends only to state, each question attends to the state and itself, position ids restart after the state. Parity against per-question encoding: 1.0e-7 (CPU fp32), 2.1e-4 (T4 fp16), but only for a model trained with that mask. Measured cost on Ettin-68m, CPU, 20 questions: 3.7 s vs 51.2 s at 2,048 tokens (13.7x), 1.3 s vs 11.5 s at 512 tokens (8.9x). Found a ModernBERT-specific trap relevant to any runtime that packs questions: transformers builds the 128-token local window from sequence index while RoPE uses position ids, and with restarted positions the stock window silently changed 8 of 20 answers; ikken builds the window from position ids. No trained checkpoint is released; accuracy is one synthetic seed (0.790 vs 0.752 overall, worse on numeric questions 0.467 vs 0.492).

kime-v1 (tamnd/kime, Apache-2.0, repo created 2026-09-23, 5 stars). Design: state encoder once, cache its keys and values, small cross-attention question tower; students `kime-v1-s-en` and `kime-v1-s-x`; a segment cache for agent loops that recomputes the low 9 layers on 5-10% of tokens per step. Targets in `spec/13-benchmarks.md`: 0.72 ms per question on a T4 (W3, against Laya's 7.2 ms), 3.3 ms single decision on T4, typed-decisions 0.78 zero-shot for the small student. Status: milestone M3 (issue #4) 0/10 done, gated on M2 teachers; README: "The native kime models, the router, the caches and the SDKs are still to come." Binaries v0.1.0 to v0.1.4 (Sept 28-29) run Laya's own weights.

Papers: no arXiv paper on this specific architecture surfaced. Related: "Dissecting RLCD" (Le Duc Minh, LakoreAI/sev; checkpoint `minhleduc/laya-typed-decisions-ce-1024`, Apache-2.0) shows plain soft-label cross-entropy on the teacher distribution matches or beats Laya's RLCD on typed-decisions (0.7885 at 1024 tokens; 0.782 +/- 0.004 over three seeds at 512), which matters here because retraining an encode-once student does not need the RL machinery. Older precedent for the idea is prompt-in-decoder (arXiv 2403.13112), which encodes the input once and decodes questions in parallel.

### Distilled or smaller models

No distilled small Laya student exists on Hugging Face (searched "laya" plus distil/student/small/ModernBERT-base/MiniLM). The community repos named `laya` are mirrors, ONNX, GGUF, MLX or Core ML ports of the 421M and 322M checkpoints. The smallest encoder-class System One models with published scores are decision-modernbert-base (149.6M, CC BY-NC, Decision Index 20.12 self-run), Verdict (151M, 1.87), Lumma-Fev-0.1B (153M, 1.78) and OpenDecider-nano (395M, the best sub-500M model on typed-decisions at 0.796). The planned kime-v1-s is unreleased.

## 7. Task 6: quantization and decisions near thresholds

Where Laya's decisions live: choice is the argmax over option probabilities; noul is `p_yes` against 0.5; score is the argmax level (and an expected value); user gating is on `answer_confidence` or `confidence`. A decision flips once the runner-up overtakes the top option. The top option can lose probability while the runner-up gains it, so a shift of more than half the top-two margin is enough. For noul the top-two margin is |p_yes - p_no| = |2 p_yes - 1|, so moving p_yes from 0.51 to 0.499 flips the answer with a 0.011 shift against a 0.02 margin. A worst-case shift d can therefore flip any decision whose margin is below 2d, so the question for any quantization is the size of the worst-case shift, not the mean.

| build | scheme | size | argmax parity vs fp32 | worst probability shift | source |
|---|---|---|---|---|---|
| upstream `export_onnx.py --quantize` (PR #498, merged 2026-09-27, v0.3.21), English | dynamic per-channel weight-only INT8, CPU only | 1.6 GB to 581 MB; p50 ~340 to ~250 ms | 20/20 states | 0.090 | github.com/NandhaKishorM/laya/pull/498 |
| same graph, per-tensor INT8 (rejected in that PR) | per-tensor scales | same size and speed | 17/20; "noul booleans crossing 0.5 and a score level" | 0.29 | PR #498 |
| nvkudva/laya-web-q8, English | weight-only INT8 MatMulNBits block 64, fp32 activations, fp16 embeddings and head | 1,688 to 524 MB | 26/26 (types 2-14 options, both truncation branches, non-Latin) | 0.0158, mean KL 1.8e-4 | huggingface.co/nvkudva/laya-web-q8 |
| onnxruntime `quantize_dynamic` (activations too), English | dynamic per-tensor activation INT8 | | 69% | 0.99 | alexander-voronkov/laya-web-poc PR #16 |
| yehor-oleksiuk/laya-multilingual-onnx int8 | MatMulNBits block 32, fp16 embeddings, fp32 head | ~0.5 GB, ~15% faster than fp32 | card: 120/120 on Russian; independent check: 15/16 on English + Russian, one flip ("partial" vs "complete") | 16.9 points (independent check) | huggingface.co/yehor-oleksiuk/laya-multilingual-onnx; laya-web-poc PR #16 |
| androidli/laya-multilingual-onnx-int4 | MatMulNBits block 32 symmetric INT4 plus GatherBlockQuantized on the 256k vocab | 1,288 to 199 MB (block 32) or 181 MB (block 128) | not reported; task accuracy 70.0% vs fp32 68.3% on 60 internal items (about +/-6 points noise) | not reported | huggingface.co/androidli/laya-multilingual-onnx-int4 |
| sahilchachra/Laya-TypedDecisions-MXFP8 (MLX) | MXFP8 group 32 | ~407 MB | 1 of 3 test questions flipped top-1 on a near tie | head logit max abs diff 0.646 | huggingface.co/sahilchachra/Laya-TypedDecisions-MXFP8 |
| fr0stbit3/laya-gguf Q8_0 / Q6_K / Q4_K_M (llama.cpp backbone only; head runs outside) | k-quants | | one example only | Q4_K_M 0.02-0.03; Q8_0 and Q6_K "close to f16" | huggingface.co/fr0stbit3/laya-gguf |
| upstream fast path bf16 vs fp32 (reference for what "lossless" looks like) | bf16 weights, fp32 residual | | 47/48 choice, 180/180 noul, 59-60/60 score; flips only on near ties where fast agrees with fp32 | 0.046 | BENCHMARKS.md "Same answers" |
| upstream fast path fp16 | fp16 weights, fp32 accumulation | | 864/864 vs fp16 stock; one fp32 disagreement on a 0.001 margin | 0.009 | BENCHMARKS.md "fp16" |

Findings:

- Weight-only INT8 with per-channel or blockwise scales and fp32 activations preserves every measured decision on the English checkpoint (worst shift 0.016 to 0.09, in the same range as bf16 vs fp32). The root cause of the failures is activation quantization: ModernBERT has outlier activation channels that a per-tensor dynamic scale cannot represent (the laya-web-poc ablation found quantizing MatMuls catastrophic and embeddings alone fine). Per-tensor weight scales also flipped 3/20 decisions, which is why upstream hard-codes `per_channel=True`.
- The multilingual checkpoint is more fragile under the same nominal INT8 (one flip and a 16.9-point shift in 16 questions), and the 256k embedding table is 61% of its bytes, so INT4 builds quantize it too. The only INT4 build reports task accuracy on 60 items, not parity, so INT4's effect near thresholds is unmeasured.
- The flips that come with a margin are on near ties. Upstream's fp16 path has one fp32 disagreement, on a 0.001 margin, and upstream BENCHMARKS.md and the MXFP8 card call their bf16 and MXFP8 flips near ties. The INT8 flips in the table come with no per-flip margins: 3 of 20 under per-tensor scales, one on the multilingual build and 31% under `quantize_dynamic`. Worst shifts of 0.29 and 0.169 can flip decisions with margins up to 0.58 and 0.34, and a 0.99 shift can flip any decision, so these sources do not show that those flips were near ties. The bake-off in PR #5 adds INT8 flips on typed-decisions, also without per-flip margins. Upstream's ONNX INT8 export and kime's CPU INT8 agree with the fp32 reference on 99.2% and 97.5% of the 120 smoke questions (the "CPU paths" table of `results/tables.md`). Simulated per-channel int8 weights in laya-mlx agree on 99.5% of 1,500 answers, against 99.9% for fp16 (`results/SPEED.md`). Laya's own README warns that different batch shapes cause "small floating-point differences, including near decision thresholds," so a runtime should treat sub-0.05 margins as unstable regardless of dtype. A decision with a margin above 0.05 is known to be safe only at a precision whose measured worst shift is below half that margin, so each precision needs its own flip count on a parity set.
- The shipped temperatures were fitted on fp32 logits; quantization changes pre-temperature logits, so calibration (ECE) can move even where argmax does not. None of the community cards report ECE after quantization.
- INT8 buys 1.15x to 1.35x on CPU in the two builds that report speed. The larger CPU gains in the record come from thread pinning (12x) and length sorting (2.15x), not from lower precision.

## 8. Open questions

1. Jev's typed-decisions accuracy varies 0.727 to 0.754 across harness runs and the API "drifted" between board runs. Any comparison should pin a date and harness.
2. `laya-multilingual`'s HF model card still shows the retracted 0.342 row; the repo says 0.352. Which will the authors treat as canonical, and is the 0.3515 JSON the final run?
3. Nobody has measured what approximate KV reuse does to Laya's answers. A cheap experiment: encode the state alone, splice its activations into a second question's row, and count flips on the 396-decision parity set upstream uses.
4. cbjev's document block reads all questions, so its speedup is within a request only. Is there any measurement of a true state-only-attention model at Laya scale? ikken's one-seed synthetic result (better overall, worse on numeric) is the only data point.
5. INT4 parity is unmeasured (task accuracy on 60 items only). INT8 on the multilingual checkpoint disagrees between the publisher (120/120) and an independent check (15/16). Which eval set and which decision margins should a runtime use as its gate?
6. Post-quantization calibration: does anyone refit temperatures after INT8, and how much does ECE move?
7. OpenDecider-nano runs on the same ModernBERT graph as Laya with a different head (MLP instead of a 2-layer transformer) and a 2,048-token budget. Its `head.safetensors` is 2.1 MB. Confirming that a Laya runtime loads it with a head swap would double the number of strong sub-500M checkpoints available.
8. decision-modernbert-base is the only sub-200M model with a credible Decision Index score, but it is CC BY-NC and Core ML only. Will FluidInference publish PyTorch weights or a commercially licensed variant?
9. The Decision Index requires median latency under 1,000 ms on an RTX PRO 6000 and the Laya row was measured with the TileLang fast path. Is a CPU-only or Apple-only runtime eligible, and what latency would it be listed at?
10. TypeSafe documents no maximum question count and no streaming. Does the API enforce a hidden cap on `questions`, and does the 32k "state plus longest question" rule apply per question or to the packed sequence?
