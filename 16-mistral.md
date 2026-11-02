# 16 · Mistral AI

> 8 题。出处：`README.en.md` § Mistral AI。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Pair-programming: build a service that takes a user question, enriches it with data from a third-party API, and answers via a chat-model API. How do you structure it?

**Anchor answer**

Structure for **one seam**: the model call is I/O with latency, cost, and failure; the enrichment is I/O with a different failure mode. Keep them behind interfaces you can test without network.

```
POST /answer {question}
  → validate (length, PII, auth)
  → enrich(question)     # 3rd-party API: timeout 1.5s, 1 retry, cache 10 min
  → build_prompt(question, facts)   # facts are structured, not prose
  → chat_model(stream=True)         # SSE to client
```

Decisions to state out loud, because they are the trick:

1. **Enrichment is optional.** If the third-party API fails, answer from the model and mark the answer `degraded: true`. Never let a partner timeout take down the endpoint.
2. **Facts go in as structured data, not a paragraph.** A compact JSON of the 5 relevant fields beats 2000 tokens of API dump — and it makes the prompt cache hit.
3. **Stream.** First token immediately; users tolerate slow completion, not slow start.
4. **Timeout budget.** Total 10 s, split explicitly: enrich 1.5 s, model 7 s, 1.5 s slack. A single global timeout hides which leg is slow.
5. **Idempotency + cost guard**: request hash as key, token cap per user.

Tests: fake both clients with recorded fixtures; assert degraded path, timeout path, and that a malicious question can't reach the API with injected parameters. Log the prompt version, the fact payload, and token counts — that's your only postmortem input.

**中文要点**：把"富化"和"模型调用"拆成两个可替换的 I/O 接缝：富化失败要降级而不是整体失败、事实用结构化 JSON 而非长文、流式返回、总超时按段拆分。

---

### 2. Mistral 7B shipped with grouped-query attention and sliding-window attention. What does each buy you, and what does each cost?

**Source**: https://outcomeschool.com/blog/grouped-query-attention · https://outcomeschool.com/blog/how-does-sliding-window-attention-work

**Anchor answer**

They attack two different walls. GQA attacks **memory bandwidth during decode**; SWA attacks **memory footprint of the cache**. Neither makes the model smarter.

| | GQA | SWA |
| :--- | :--- | :--- |
| Mechanism | Q heads grouped, each group shares one K/V head (Mistral 7B: 32 Q, 8 KV) | Each token attends only to the previous *W* tokens per layer (W=4096) |
| Buys | KV cache 4× smaller; decode reads less memory per step | Cache bounded by W, not by sequence length; constant per token cost |
| Costs | Slight quality loss vs full MHA; affects prefill little | Information travels at most W per layer, so a 32-layer model reaches ~128k only *indirectly* |
| Training | Can be uptrained from MHA | Needs retraining; interacts poorly with exact long-range recall |

The subtle point: SWA does not mean the model "can't see" token 100k. Receptive field compounds — layer *L* can influence tokens up to `W × L` back. What it loses is **direct, single-hop retrieval** of an exact span far away. That is why sliding-window models feel fine on summarization and weak on "find the invoice number on page 3".

The cost people miss operationally: GQA forces you to size TP by **KV heads**, not Q heads. With 8 KV heads you cannot tensor-parallel 16 ways without duplicating KV or padding — a real constraint when choosing a serving topology.

**中文要点**：GQA 省 decode 时的 KV 带宽（32Q/8KV → cache 小 4 倍），SWA 让 cache 与序列长度解耦；代价是质量微降、长距离精确检索变弱，且 TP 只能按 KV 头数切。

---

### 3. Explain how a Mixtral-style sparse mixture-of-experts model works. Why does a ~47B-parameter model run at roughly the cost of a ~13B one?

**Source**: https://outcomeschool.com/blog/mixture-of-experts

**Anchor answer**

Every transformer block's FFN is replaced by *E* experts (Mixtral: 8), plus a small router. For each token, the router scores all experts, takes **top-2**, and the output is the weighted sum of those two experts' outputs. Attention, embeddings, and norms are dense and shared.

So: `47B total ≈ 13B active per token`. Cost tracks **active** parameters because FLOPs and, crucially, weight-memory *reads* per token scale with what you actually multiply.

| Quantity | Scales with | Consequence |
| :--- | :--- | :--- |
| FLOPs / latency per token | active (~13B) | ~3.5× faster than a dense 47B |
| VRAM / HBM for weights | total (~47B) | ~94 GB bf16 — needs 2×80 GB or int4 on one |
| Quality | total | closer to 47B dense than 13B |

**Where it hurts.** (1) **Memory, not compute, is the binding constraint** — you pay for 47B of weights to get 13B of speed; on a latency-bound single-user deployment that is a bad trade. (2) Router **load imbalance**: a few experts get hot and the tail latency becomes the hot expert's queue. (3) Expert parallelism is an **all-to-all** across nodes — fine on NVLink, brutal over Ethernet. (4) Fine-tuning is less stable: router collapse and expert under-training; aux load-balance loss is not optional.

When *not* to use it: single-stream low-batch serving on one GPU, or any environment where interconnect is the bottleneck.

**中文要点**：MoE 只激活 top-2 专家，算力跟着 active ~13B 走，显存跟着 total ~47B 走；真正的痛点不是算力而是显存、专家负载不均，以及跨节点 all-to-all。

---

### 4. Estimate the KV-cache memory for serving Mistral 7B, and design the rolling-buffer cache that sliding-window attention enables.

**Source**: https://outcomeschool.com/blog/kv-cache-in-llms · https://outcomeschool.com/blog/how-does-sliding-window-attention-work

**Anchor answer**

Mistral 7B: 32 layers, 32 Q heads, 8 KV heads, head_dim 128, bf16.

`bytes/token = 2 × 32 layers × 8 kv_heads × 128 × 2 B = 131,072 B = 128 KiB/token`

| Context | Naive cache | Rolling cache (W=4096) |
| :--- | :--- | :--- |
| 4k | 512 MiB | 512 MiB |
| 32k | 4 GiB | 512 MiB (constant) |
| 128k | 16 GiB | 512 MiB (constant) |

With 24 GiB usable on an L4/A10-class card after ~14 GB of int4 weights, naive caching gives ~1 concurrent 128k session; rolling gives ~40 × 128k sessions, because each only ever holds 4k of cache.

**Design.** One fixed ring buffer per layer per sequence:

```python
slot = pos % W                      # overwrite oldest
k_cache[layer][slot] = k; v_cache[layer][slot] = v
# attention mask: causal AND (pos - token_pos) < W
allowed = (q_pos >= k_pos) & (q_pos - k_pos < W)
# position for RoPE comes from the real counter, not the slot
```

Two things break in practice: RoPE must use the **absolute** position, not `slot`, or long-context continuation degrades; and the mask must be a *diagonal band*, not a prefix — a naive `slot < n` mask silently attends to stale slots after wraparound. Prefill of a fresh 128k prompt also still costs a full 128k attention *for that pass* (the window only bounds memory, not the prefill mask if the implementation uses global attention).

**中文要点**：128 KiB/token；rolling cache 让显存按 W=4096 定额（512 MiB/序列）而非按长度增长；坑是 RoPE 要用绝对位置、mask 要写成对角带状而非前缀。

---

### 5. You need to quantize a model for a customer's hardware. How do you choose a scheme, and how do you prove quality hasn't regressed?

**Source**: https://outcomeschool.com/blog/how-does-model-quantization-work

**Anchor answer**

Choose from the hardware **up**, not from the paper down. Four questions decide it:

| Constraint | If yes | Scheme |
| :--- | :--- | :--- |
| Card has fp8 tensor cores (H100/L40S)? | decode bandwidth-bound | **FP8 (W8A8)** — near-lossless, least engineering |
| Only int8/int4 kernels, older GPU or CPU? | memory-bound | **INT8 weight-only**, or **INT4 (GPTQ/AWQ)** |
| Batch 1 latency matters most? | bandwidth-bound | weight-only INT4 wins; activation quant adds kernel overhead |
| Large batch / throughput? | compute-bound | W8A8/W4A8 helps; W4A16 does not |

Default recommendation for a customer: **AWQ or GPTQ int4, group size 128, KV cache in fp8**. It fits a 30B model on one 24 GB card.

**Proving no regression** — perplexity is not evidence. Run a tiered gate:

1. **Task suite** with the customer's own labelled set: exact-match/F1, per-slice.
2. **Format validity**: % of parses that satisfy the JSON schema — quantized models fail here first.
3. **Long-context needle test** at the customer's real context length; int4 KV degrades here before anything else.
4. **Distribution shift**: KL of next-token distribution vs fp16 on 1–2k prompts; flag slices over threshold.
5. **Refusal/hallucination rate** on a fixed prompt set — quantization moves calibration.

Then A/B on 5% traffic with the *same* prompts, comparing task success and human thumbs. Gate: no slice worse than a pre-agreed delta, and the tail (worst 5% prompts) must be inspected by hand. Report the *quantization-per-slice* table, not one aggregate number.

**中文要点**：选方案从硬件能力倒推（fp8 卡用 W8A8，老卡用 int4 weight-only，batch1 延迟优先就别量化激活）；验证看 schema 合法率、长上下文与逐切片指标，ppl 不算证据。

---

### 6. How does function calling actually work with an LLM, and how do you make it reliable enough for production agents?

**Source**: https://outcomeschool.com/blog/how-does-function-calling-work-in-llms

**Anchor answer**

Mechanically: tool definitions are serialized into the prompt (JSON schema), the model is trained to emit a structured `{name, arguments}` block, the **client** parses it, validates, executes, and appends the result as a new message. The model never runs code — it emits intent.

Reliability is mostly not a model problem. In rough order of impact:

1. **Schema design.** 5–7 tools, flat arguments, enums instead of free strings, no untyped `object`. Every optional field with a default is ambiguity the model will resolve differently on Tuesday.
2. **Constrained decoding.** Grammar/regex-constrained output guarantees parseability. It does not guarantee *semantic* validity — `start_date: "2024-13-45"` parses fine.
3. **Server-side validation + typed error feedback.** Reject with a machine-readable reason (`missing_required: customer_id`) and re-prompt once. Silently retrying the same call is the #1 production bug.
4. **Idempotency keys** on side-effecting tools, so a retry after a timeout doesn't create two tickets.
5. **Deterministic dispatch.** Validate against the *tool registry*, never against the model's string. Unknown name → error, not fuzzy match.
6. **Eval set.** 100–200 golden (prompt → expected call) cases, including adversarial ones: wrong tool exists, argument from a previous turn, injection in retrieved text.

Failure modes worth naming: models call tools when they should answer ("what's 2+2" → calculator), chain 6 calls when 1 suffices, and lose arguments across turns. Cap steps, require an explicit "no tool needed" path, and log `(tool, args, result, latency)` for every call — that log is your only debugging surface.

**中文要点**：函数调用本质是"模型输出意图，客户端校验并执行"；可靠性靠少而扁的 schema、约束解码、可机器读的报错回灌、幂等键，以及一套含反例的 golden eval。

---

### 7. After fine-tuning on a customer's task, target accuracy is up but the model got worse at everything else. What happened and what do you do?

**Source**: https://outcomeschool.com/blog/continual-learning-in-llms

**Anchor answer**

Classic **catastrophic forgetting**, and it is the expected outcome, not a bug: gradient descent on a narrow distribution overwrites the representations the base model used for everything else. The tell is that target accuracy rose *fast* in the first epoch — that is the model specializing, i.e. narrowing.

Diagnosis order:

1. **Is it forgetting or distribution shift?** Score a general canary suite (instruction following, JSON validity, refusal, short math, multi-turn) on base vs tuned. Forgetting shows as a broad drop; shift shows as a drop only on prompts unlike the training data.
2. **Over-training.** More epochs than needed and too-high LR both increase drift. LoRA with rank 8–16 and LR ~1e-4 drifts far less than full fine-tuning at 2e-5.
3. **Data mix.** If the customer set is 100% one template, the model learns the template, not the task.

Fixes, cheapest first:

| Fix | Effect | Cost |
| :--- | :--- | :--- |
| Mix 10–30% general/replay data into training | largest single win | need a replay corpus |
| Lower LR, fewer epochs, early stop on the canary suite | cheap | tuning time |
| LoRA instead of full FT; keep base frozen | structural protection | slight ceiling |
| Merge (TIES/DARE) or add a KL term to base logits | recover drift | extra compute |

Then re-run the customer eval, not just the canary suite — the two can move in opposite directions, and only both numbers together tell you if you shipped.

**中文要点**：灾难性遗忘是预期结果而非 bug，靠"10~30% 通用 replay 数据 + LoRA/低 LR/早停"压制；必须同时看通用 canary 套件和客户任务指标。

---

### 8. Design an on-prem deployment of an open-weight model for a European bank that cannot send data to any external API.

**Anchor answer**

The technical core is easy; the *operational* surface is the interview. Assume: no egress at all (not even telemetry), the bank's security team owns the network, and GPUs are what they already bought.

| Layer | Design | Bank-specific constraint |
| :--- | :--- | :--- |
| Weights | Delivered as a signed artifact inside the perimeter; hash-verified | No "call home" for license check — offline license file |
| Serving | vLLM/TensorRT-LLM behind an internal gateway; TLS with bank CA | Must run on their K8s or bare metal, not your cloud |
| Identity | OIDC against the bank's IdP; per-user rate limits | Their group policy is the only authorizer you get |
| Data | RAG index inside the bank; embeddings computed locally | Model must include the embedder — no third-party embedding API |
| Observability | Logs/metrics to their SIEM, no external SaaS | You ship a dashboard they host |
| Updates | Quarterly offline bundle: weights + eval report + CVE list | Change window per their release process |
| DR | Second air-gapped site, model + index snapshots | RPO/RTO in the contract |

Sizing: pick context length and concurrency first, then KV cache, then cards — banks always under-provision cache and over-provision FLOPs. A 30B model at 8k context, 16 concurrent, is ~2×80 GB with room; write that number down early because procurement is the long pole.

What actually bites: certificate/CA mismatch at install, model updates invalidating the index (embedder version must be pinned), and the bank's DLP scanning the *model's* output on the way to the user.

**中文要点**：银行私有化真正的难点在运维面而非模型：离线签名交付、无外联许可校验、本地 embedder+索引、日志只进行内 SIEM；容量先按 context×并发 估 KV 再选卡。

---

## 附：这 8 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 架构内部机制 | 2（GQA + SWA）、3（稀疏 MoE） |
| 显存与容量直觉 | 4（rolling KV cache） |
| 客户硬件落地能力 | 5（量化选型与回归证明）、8（私有化部署） |
| 应用层工程 | 1（富化服务结构）、6（函数调用可靠性） |
| 后训练判断力 | 7（遗忘与回滚） |
