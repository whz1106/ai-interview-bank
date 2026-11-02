# 17 · Cohere

> 10 题。出处：`README.en.md` § Cohere。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Design a token-based rate limiter for a multi-tenant LLM API. Implement the core, then tell me what changes when it's distributed.

**Anchor answer**

Token-based limiting needs **reservation**, because cost is known only after generation. Reserve `input + max_tokens`, settle on completion.

```python
class TokenBucket:
    def __init__(self, rate, burst):
        self.rate, self.burst = rate, burst
        self.tokens, self.ts = burst, time.monotonic()

    def _refill(self, now):
        self.tokens = min(self.burst, self.tokens + (now - self.ts) * self.rate)
        self.ts = now

    def reserve(self, n, now):
        self._refill(now)
        if self.tokens < n:
            return False, max(0.0, (n - self.tokens) / self.rate)   # retry_after
        self.tokens -= n
        return True, 0.0

    def settle(self, reserved, actual, now):
        self._refill(now)
        self.tokens = min(self.burst, self.tokens + (reserved - actual))
```

Charge **per tenant, not per API key**, and keep a global bucket the tenant bucket must also pass — otherwise N tenants each under their limit still melt the fleet. Weight prefill tokens ≠ decode tokens if the fleet is disaggregated.

**Distributed changes, in order of pain:**

| Problem | Fix | Residual error |
| :--- | :--- | :--- |
| Per-node buckets drift upward | Single atomic counter (Redis Lua: refill+check+decrement in one script) | one RTT per request (~0.5–1 ms) added to admission |
| RTT too expensive at high QPS | Node **leases**: hand each node `quota/N` for 100 ms; spend locally | overshoot ≤ lease size |
| Streaming never "completes" | Settle incrementally per chunk; cut the stream when the tenant hits its cap | truncated answers need a UI story |
| Worker crash mid-generation | Reservations need a TTL/heartbeat, else quota leaks until restart | leaked quota on hard kills |
| Clock skew | Never compute refill from wall clock across nodes; send `now` from the coordinator | — |

Design rule: the limiter must be **fail-closed on spend, fail-open on read** — if the counter store is down, degrade to local buckets with a conservative rate rather than 500ing every request.

**中文要点**：token 计费必须"预扣 + 结算"（按 input+max_tokens 预留，完成后退差）；分布式靠 Redis Lua 原子脚本或按节点发配额租约，崩溃要用 TTL 防配额泄漏。

---

### 2. Our flagship is a sparse MoE with ~10x more total than active parameters. Why is that architecture a good fit for private enterprise deployment, and where does it hurt?

**Source**: https://outcomeschool.com/blog/mixture-of-experts

**Anchor answer**

Because enterprise deployments are **throughput-per-GPU-fleet** problems under a fixed memory budget, and MoE decouples the two curves that usually fight: quality tracks *total* parameters, serving cost tracks *active* ones. A ~10× ratio means you buy a large model's quality and pay a mid-size model's FLOPs — and in a private deployment, the GPUs are already bought and idle most of the day.

| Fit | Why it matters on-prem |
| :--- | :--- |
| High concurrency, batch-heavy | Per-token compute is small; the fleet serves far more tokens |
| Fixed hardware budget, quality-sensitive | You spend VRAM (already paid for) instead of latency |
| Multi-tenant traffic with diverse domains | Experts specialize; a legal tenant and a support tenant stress different experts |
| On-prem economics | No per-token cloud bill; FLOPs saved are capacity you can sell internally |

**Where it hurts:**

1. **Weights must all be resident.** Total params set VRAM; a 10× ratio means the model may not fit on the customer's node at all, and quantization is your only lever. Expert weights are harder to quantize than dense ones without per-expert calibration.
2. **Interconnect sensitivity.** Expert parallelism is all-to-all. On a bank's Ethernet fabric with no NVLink domain, dispatch latency dominates and the MoE advantage evaporates.
3. **Low-batch tail latency.** With batch 1, you get no expert reuse; the router's hot expert becomes the queue. Enterprise pilots are always batch 1.
4. **Air-gapped ops.** Expert placement, rebalancing, and profiling tooling are more complex to run without vendor support.

Say the trade-off explicitly: for a 5-user pilot on two cards, a dense model is the right answer.

**中文要点**：MoE 让"质量看总参数、成本看激活参数"解耦，适合私有化下已买断的显存 + 高并发；代价是全量权重必须驻留、专家并行对互联敏感、低 batch 时尾延迟被热专家拖垮。

---

### 3. You have an embedding model and a reranker. Why sell both? Design the two-stage retrieval pipeline and tell me when the reranker earns its latency.

**Source**: https://outcomeschool.com/blog/how-does-a-reranker-work

**Anchor answer**

Because they answer different questions at different costs. A **bi-encoder** embeds query and document **independently**, so documents are precomputed and searched in an ANN index — it scales to 100M docs but never sees query and document together. A **cross-encoder** feeds `(query, doc)` through the model jointly, so it models term interaction — far more accurate, impossible to precompute.

| Stage | Model | Cost | Role |
| :--- | :--- | :--- | :--- |
| 1. Recall | bi-encoder + ANN (top 100–500) | one query embed + ANN lookup | cheap recall, must not drop the answer |
| 2. Precision | cross-encoder rerank (top 50 → 5) | 50 forward passes | fixes ordering, kills near-misses |

Key design fact: **the reranker can only reorder what retrieval returned.** If recall@100 is 0.7, your ceiling is 0.7 — reranking improves nDCG, it cannot improve recall. That is why "add a reranker" fails on some corpora and teams wrongly blame the reranker.

**When it earns its latency** (a rerank of 50 pairs is typically tens of ms on GPU, hundreds on CPU):

- Answers are high-stakes and cite sources (compliance, support with SLAs).
- Queries are short and keyword-like while documents are long — exactly where bi-encoders are weakest.
- Domain jargon/negation matters ("does *not* require approval"), where embedding similarity is nearly blind.
- Candidate set is small but must be ordered well.

**When it doesn't:** latency-critical autocomplete, huge candidate sets (cost is linear in candidates), or when the corpus is so homogeneous that top-5 ordering barely matters.

**中文要点**：双塔负责"捞得到"（可预算、可 ANN），cross-encoder 负责"排得准"（必须逐对算）；reranker 只能重排、不能提升召回，高价值+短查询+领域术语时才值得那几十毫秒。

---

### 4. An enterprise wants semantic search over ~100M documents but is balking at vector-index cost. Walk me through embedding compression options and the maths.

**Source**: https://outcomeschool.com/blog/how-does-a-vector-database-work

**Anchor answer**

Start with the raw number, because the customer is usually reacting to it: `100M × 1024 dims × 4 B = 410 GB` of vectors alone, before HNSW graph overhead (typically +30–60% for the neighbor lists) and before replicas.

| Technique | Size @100M, 1024d | Recall impact | Engineering cost |
| :--- | :--- | :--- | :--- |
| fp32 (baseline) | ~410 GB | — | none |
| fp16 | ~205 GB | ~0 | trivial |
| **int8 scalar** | ~102 GB | ~0.5–2 pt | small (per-dim scale) |
| **Matryoshka truncation 1024→256** | ~102 GB int8, ~25 GB binary-ish | 1–3 pt | needs MRL-trained model |
| **Binary (1 bit/dim)** | ~12.8 GB | 3–10 pt, recoverable by rescoring | medium |
| **PQ (m=64, 8 bit)** | ~6.4 GB | 5–15 pt without rescoring | high (train codebooks) |
| IVF + PQ, disk-resident | GBs in RAM | depends on nprobe | high ops complexity |

The winning combo is almost always: **int8 + Matryoshka 512-d + binary first-pass with int8 rescoring of top-1000**. That lands near ~25 GB in RAM with recall within a couple of points of fp32, because the rescoring stage recovers what binary quantization loses.

Two things to tell the customer: the index is not the only cost — **ingestion re-embeds 100M docs** and that is usually the bigger one-time bill; and recall must be measured, not assumed. Build a 500-query labelled set and plot recall@10 vs memory. If they cannot label 500 queries, they cannot verify any of these choices.

**中文要点**：100M×1024d fp32 ≈ 410 GB；压缩路线 int8 → Matryoshka 降维 → 二值化首轮 + int8 重排，可压到 ~25 GB 且召回只差几个点；别忘一次性重嵌入成本。

---

### 5. How would you evaluate multilingual retrieval quality when employees query in French and Korean over mostly-English documents?

**Anchor answer**

This is **cross-lingual retrieval**, and the evaluation has two independent variables that teams conflate: language (FR/KO/EN) and query type (keyword, natural question, entity). Slice by both.

**Building the labelled set** (no labels exist yet):

1. Stratified sample of 300–500 English docs across departments and lengths.
2. Native speakers (or bilingual employees) write queries **in their own language** for those docs — do not machine-translate English queries; translated queries are unnaturally literal and flatter the system.
3. Human relevance grades 0–3 per (query, doc), with at least the top-10 system results plus 5 random negatives.
4. Add ~10% **unanswerable** queries to catch systems that always return something.

| Metric | Why |
| :--- | :--- |
| Recall@50 per language | the ceiling for any downstream reranker |
| nDCG@10 per language | ordering quality, what users feel |
| Query-language vs doc-language matrix | exposes a language that silently collapses |
| Failure taxonomy on 50 misses | tokenizer, transliteration, acronym, negation |

**The trap:** if the embedder was trained mostly on English, Korean recall can drop 10–20 points while the aggregate only drops 3 — and the aggregate is what dashboards show. Always report per-language, and set the gate on the *worst* language, not the mean.

Known failure modes: Korean compound nouns split badly under the tokenizer; French accents lost in normalization cause near-duplicate but distinct matches; English BM25 fused with a multilingual dense retriever can actively hurt non-English queries, so tune fusion weights per language or drop the lexical leg for FR/KO. Also check that doc language detection isn't mislabeling mixed-language docs.

**中文要点**：跨语言检索必须按"查询语言 × 查询类型"分片报告；标注要用母语者**原生**写 query（别翻译）；门槛看最差语言而不是均值，韩语复合词与法语重音是常见塌陷点。

---

### 6. A customer 10x'd their indexed documents and reports answer quality “got noticeably worse.” Drive the investigation.

**Anchor answer**

Quality dropped while the model didn't change, so the regression is in **retrieval or context assembly**. Ten-x scale changes four things at once; isolate them in this order (cheapest to check first):

| # | Hypothesis | Check | Typical finding |
| :--- | :--- | :--- | :--- |
| 1 | ANN recall fell | Measure recall@k vs exact brute force on 100 queries | HNSW `ef_search`/`nprobe` unchanged → recall dropped 5–15 pt |
| 2 | Distractors now outrank the right chunk | Inspect 20 bad answers, look at rank of the gold chunk | Gold is at rank 40; reranker sees 50 but the window used to be enough |
| 3 | Duplicates flood the context | Count near-dup chunks per prompt | 10× docs = 10× copies of the same policy; top-5 is 4 copies |
| 4 | Chunking/metadata broke on new sources | Look at new-doc ingest stats | New source has no section headers → chunks are 3 words or 4k tokens |
| 5 | Freshness/staleness | Diff index count vs source count; check deleted docs | Old versions still indexed, both retrieved |
| 6 | Filter/tenant leakage | Auth on retrieval | New shard lost the tenant filter → cross-tenant chunks |

The single highest-yield measurement: **retrieve the same 100 queries through both the old and new index and diff the top-10**. That separates "we lost the answer" (rank 40 vs rank 2) from "the answer got crowded out" (still rank 2, but prompt is full of duplicates).

Then fix in order: pin ANN parameters to hit a target recall, dedup by document hash before assembling context, cap chunks per source document (e.g. max 2), and re-run the same eval set. Never accept "quality feels worse" — convert it into recall@k, gold-chunk rank, and duplicates-per-context, which are all measurable in an afternoon.

**中文要点**：规模 10x 后先分离四件事：ANN 召回下降、干扰项挤占、近似重复占满上下文、切块/元数据失效；最有效手段是把同样 100 条 query 在新旧索引上各跑一遍做 top-10 diff。

---

### 7. Design an agent that automates an enterprise workflow, say, drafting RFP responses from internal documents and a CRM. What does “enter-prise-grade” add?

**Anchor answer**

The agent itself is the easy half: retrieve → draft → cite. "Enterprise-grade" is the other 80%, and it is all about **being auditable and refusable**.

Pipeline: parse the RFP into requirement rows → for each row, retrieve (products docs + past responses + CRM deal facts) → draft with mandatory citation per claim → classify gaps ("we have no answer for X") → assemble into the customer's template → human review → export.

| Requirement | Concretely |
| :--- | :--- |
| **Authorization** | Retrieval must be filtered by the *requesting user's* document ACLs, at query time, not in the UI. A sales rep must not see the legal team's redlines |
| **Citation** | Every claim maps to a doc id + span. Uncited claims are flagged, not shipped |
| **Audit trail** | Prompt version, retrieved chunk ids, model version, who edited what. Regulated buyers ask for this months later |
| **Human gate** | Draft is a proposal, never a send. Approval step with diff view |
| **Determinism/replay** | Same RFP + same corpus → same draft, so an approval can be re-derived |
| **Idempotency + rate control** | One RFP retried doesn't create three CRM records or three drafts |
| **Data residency** | Tenant-pinned index and model; no cross-tenant retrieval |
| **Cost/scale** | 200-row RFP at 4k tokens/row is real money; batch, cache, and cap |
| **Evals** | Golden RFPs, requirement-coverage score, unsupported-claim rate |

Failure modes specific to this workflow: confidently answering a compliance question with a marketing doc; the "no answer" case being silently filled with plausible text (the most expensive bug in RFP automation); and stale CRM facts outranking the current price sheet. Encode "I don't know" as a first-class output.

**中文要点**：企业级不是加功能，而是"可审计、可拒绝"：检索期按用户权限过滤、每条断言必须带引用、全程留痕、草稿必须人工过闸，且"I don't know" 必须是合法输出。

---

### 8. A bank wants the whole stack (model, RAG, agents) deployed air-gapped on their own GPUs. What actually changes versus your SaaS?

**Anchor answer**

Assume zero egress — not even DNS. Then walk the stack and mark what dies:

| Component | SaaS | Air-gapped |
| :--- | :--- | :--- |
| Model weights | served by you | shipped as signed artifacts; hash verified on install |
| Embeddings | your API | must ship the embedder; **re-embedding the bank's corpus happens on their GPUs** |
| Reranker | your API | same; it is a second model with its own VRAM line item |
| Updates | continuous | versioned bundles, quarterly, through their change board |
| Telemetry | you see traffic | nothing leaves; they get a dashboard they host |
| Auth | your IdP | their IdP/AD, group policy is the whole authorization model |
| Secrets | your vault | their KMS/HSM; you cannot hold keys |
| Support | SSH/observability | remote support is impossible; you debug from logs they export |

Capacity is the number that decides the deal: total VRAM = weights + KV cache (context × concurrency) + embedder + reranker + headroom. Banks typically specify context and concurrency in the contract, so size for *that*, and state the numbers early — procurement is longer than integration.

What actually breaks in practice: the embedder and the index must be versioned **together** (upgrading the embedder invalidates 100M vectors, a multi-day re-embed job they must schedule); the agent's tool allowlist has to be a static config the bank reviews, not something the model can extend; and the model's output passes through their DLP, so any redaction you do must be invisible to them.

Also plan the failure story: no vendor phone-a-friend means you ship runbooks, health checks, and a self-test command that proves model + index + agent work, which their ops team runs after every change window.

**中文要点**：气隙部署要连带交付 embedder/reranker 权重与整套本地可观测性；容量 = 权重+KV缓存+嵌入+重排+余量；升级 embedder 必须与索引同步重建，agent 工具白名单要静态可审。

---

### 9. An enterprise customer wants to deploy your RAG system but has no labelled data. How do you evaluate it before and after launch?

**Anchor answer**

Build the labels you need — small and targeted — instead of pretending you can evaluate without any.

**Before launch (offline):**

1. **Gold set, 200–500 items.** Sample documents stratified by department and length; have 1–2 SMEs write realistic questions *and* mark the answerable chunk. Two annotators on 10% to measure agreement; if agreement is low, the task is ambiguous, not the system.
2. **Synthetic expansion.** Generate questions *from chunks* with a strong model, then discard any whose source chunk isn't retrievable by the system and any SME rejects — this filters the "answerable only because the generator saw the doc" bias.
3. **Unanswerable set (~10%).** Essential: measures whether the system says "I don't know" or hallucinates.
4. **Metrics split by stage.** Retrieval: recall@k, MRR. Generation: faithfulness (every claim supported by a cited chunk), answer relevance, citation correctness. Judge with an LLM *calibrated against the SME labels* — report agreement, never a bare judge score.
5. **Ablations.** Same questions with retrieval off — if quality is unchanged, your RAG isn't doing anything.

**After launch (online):**

| Signal | Reads as |
| :--- | :--- |
| Thumbs / rating per answer | coarse but the only broad signal |
| Citation click-through | did the answer actually get used |
| Rephrase rate, escalation to human, ticket reopened | strongest implicit failure signal |
| Retrieval-miss log (empty/no-strong-match) | coverage gaps, drives content fixes |
| Sampled human review of 50/week | catches drift and judge blindness |

Then A/B against the incumbent (search box + human) on the metric the buyer cares about — time-to-answer or deflection — not on model-side scores. Sample 5% of real queries with their retrieved chunks so the eval set can be rebuilt monthly; unlabelled production traffic is the cheapest label source you will get.

**中文要点**：没标注就先造 200~500 条金标（按部门分层、SME 写题、低一致性说明题目本身有歧义），加 10% 无答案集测拒答；上线后用重述率/转人工/引用点击和每周 50 条人工抽检做漂移监控。

---

### 10. Tell me about a time you owned an ambiguous problem end-to-end without much direction.

**Anchor answer**

Use STAR, and make the *ambiguity* the subject — the interviewer is testing whether you convert vagueness into a measurable target, not whether you shipped something.

Structure that survives scrutiny:

1. **Situation** — one sentence of context, no company lore.
2. **The ambiguity, named precisely** — "we had two hypotheses about why customers abandoned onboarding and no instrumentation to distinguish them." Naming the fork is what proves you owned it.
3. **How I cut it** — I wrote a one-page problem statement: who hurts, what number proves it's fixed, what I will *not* do. Then I got one person to agree to that number. That agreement is the whole job.
4. **Execution** — smallest experiment that moves the number; a two-week scope with a checkpoint, not a six-week plan.
5. **What I learned / what I'd do differently** — the honest part: I over-scoped the first attempt, or I picked a metric that moved for the wrong reason.

Anti-patterns that sink this answer: describing a task someone handed you with a spec (not ambiguous), claiming "I just figured it out" without a mechanism, or skipping the result. Have **two** stories ready — one where the answer was a build, one where the right answer was *not* building — because the second demonstrates judgment and almost no candidate has one.

Numbers matter: state the baseline, the target, and the delta you actually got, even if the delta was small and you say so.

**中文要点**：行为题要把"模糊"本身讲清楚：如何把模糊需求切成一句问题陈述 + 一个可验证指标 + 明确的非目标，再给基线与实际增量；准备"该不做的决定"那种故事。

---

## 附：这 10 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 后端工程基本功 | 1（token 限流 + 分布式） |
| 架构取舍 | 2（MoE 与私有化） |
| 检索深度（核心） | 3（两级检索）、4（向量压缩数学）、5（多语言评估）、6（规模回归排查） |
| 企业交付能力 | 7（工作流 agent）、8（气隙部署） |
| 评估方法论 | 9（无标注评估） |
| 价值观 / 主动性 | 10（模糊问题端到端） |
