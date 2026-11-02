# 06 · AI System Design

> 10 题。出处：`README.en.md` § AI System Design。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Design an enterprise RAG assistant over 10M documents with per-user permissions.

**Asked at**: OpenAI · Microsoft · Amazon (AWS) · Databricks · Scale AI · ByteDance · Alibaba (Taobao/Tmall, Alibaba Cloud)

**Anchor answer**

**Goal & constraints.** p95 answer under 3 s; zero cross-tenant or cross-group leakage; documents change hourly; ~10M docs × ~8 chunks ≈ 80M vectors.

| Stage | Design |
| :--- | :--- |
| Ingest | CDC/webhook from SharePoint, Drive, Confluence, S3 → parse → structure-aware chunks (400–800 tok, heading path prepended) → upsert with `doc_version`; tombstones for deletes |
| Index | Hybrid: BM25 + dense HNSW; ACL stored as group-id array, applied as a **pre-filter inside** the ANN search |
| Retrieve | Query rewrite → hybrid top-200 → cross-encoder rerank to 20 → MMR dedupe → token budget |
| Answer | Cite chunk ids; refuse when the best rerank score is below threshold |

**Key failure modes**

- **Post-filter leakage**: filtering after retrieval both collapses recall (top-k all unauthorised) and, worse, leaks if the filter runs after generation. Filter in the query.
- **Stale ACL** from a 15-minute cache: a removed user keeps reading. Invalidate on group-change events.
- Chunks spanning a permission boundary; ACL must be resolved per chunk, not per document.
- Embedding-model upgrade requires a full reindex — dual-write into a shadow index and cut over.

**How to validate**

- 200-question golden set per department: recall@20, answer faithfulness, and a **canary document** readable only by group A, asserted absent for group B on every CI run.
- Nightly job diffs index ACLs against source systems.
- Load test at 3× peak with realistic document lengths.

**中文要点**：权限必须作为 ANN 的**前置过滤**而不是检索后裁剪，否则既掉召回又会泄露；ACL 缓存要按组成员变更事件失效，并用只能 A 组看到的金丝雀文档做每次 CI 的泄露回归。

---

### 2. Design a code assistant: repo indexing, context assembly, edit application, evaluation.

**Asked at**: Cursor (Anysphere) · ByteDance · Tencent · Meituan · Alibaba (Taobao/Tmall, Alibaba Cloud)
**Source**: https://outcomeschool.com/blog/how-does-cursor-work · https://outcomeschool.com/blog/how-does-claude-code-work

**Anchor answer**

**Goal & constraints.** Sub-second inline completion, sub-5 s multi-file edit; edits must apply cleanly and compile; index must be fresh within a minute of push.

| Stage | Design |
| :--- | :--- |
| Index | git clone → tree-sitter AST chunks by function/class → embed symbols + BM25 on identifiers → import/call graph; incremental on push |
| Context | cursor → enclosing function, imports, type defs via LSP → hybrid retrieval of similar symbols → recent diffs; 8–30k budget, **stable ordering for prefix cache** |
| Edit | model emits a patch or search-replace block → AST-aware apply; on conflict fall back to full-file rewrite; validate with typecheck, then 1–2 error-feedback rounds |
| Eval | hidden unit tests, internal SWE-bench-style suite, edit accept/revert rate from telemetry |

**Key failure modes**

- **Stale index** poisons context with deleted APIs; always verify retrieved symbols exist in the current tree.
- Line-drift apply failures on large files — anchor replacements on symbol text, not line numbers.
- Prompt-cache thrash: per-keystroke cursor context placed first kills every cache hit. Volatile content goes last.
- Hallucinated library calls: force retrieval of the signature before emitting it.

**How to validate**

- Replay recorded edit sessions: accept rate, revert rate within 24 h, patch-apply failure rate.
- Offline pass@1 on the hidden test suite per repo, tracked per model version.
- p50/p95 completion latency in the IDE, measured client-side, not server-side.

**中文要点**：索引要增量且必须校验检索到的符号在当前代码里存在；编辑用 AST/符号锚定而非行号，失败就整文件重写再跑类型检查；游标等易变上下文放 prompt 末尾，否则前缀缓存全废。

---

### 3. Design a customer-support agent that can take real actions, with escalation to humans.

**Asked at**: Consumer-Scale ML Companies (Uber, Netflix, LinkedIn, Airbnb, Pinterest, Spotify) · Sierra · DiDi · JD.com · Meituan

**Anchor answer**

**Goal & constraints.** Deflect 40–70% of contacts; **zero** unauthorised money movement; clean handoff with full context.

| Stage | Design |
| :--- | :--- |
| Router | intent + risk classifier: FAQ / account action / complaint / legal |
| Tools | reads (order, refund eligibility) under user-scoped tokens; writes carry idempotency keys and per-day limits |
| Policy | eligibility computed by **deterministic code**; model only picks which check to run and how to explain |
| Escalation | low confidence, 2 failed attempts, policy block, legal/regulatory keywords, explicit request → human with structured summary |

**Key failure modes**

- Hallucinated policy — the most expensive failure. Every policy claim must cite a KB article id, else escalate.
- Duplicate refunds on retry: idempotency must be a DB unique constraint, not a cache lock.
- Escalation loops (agent re-prompts instead of handing off) and context loss at handoff.
- Prompt-injection inside pasted customer text steering a tool call.

**How to validate**

- Simulated conversations with an LLM user simulator plus a policy oracle; score containment, policy compliance, wrong-action rate (must be 0 for money).
- Shadow mode: run actions in dry-run for two weeks, diff against human decisions.
- Per-action audit replay and weekly sampling of 200 transcripts for tone and correctness.

**中文要点**：钱相关的资格判定必须由确定性代码算，模型只负责选检查项和解释话术；动作要 DB 级幂等 + 限额，升级要给人工完整结构化摘要，且用模拟对话 + 影子模式验证。

---

### 4. Design semantic search over a large product catalogue.

**Asked at**: Character.AI · JD.com
**Source**: https://outcomeschool.com/blog/how-does-semantic-search-work

**Anchor answer**

**Goal & constraints.** 10–100M SKUs, p99 under 150 ms, freshness on price/stock in seconds, multi-language queries.

| Stage | Design |
| :--- | :--- |
| Query understanding | spell fix, synonym expansion, attribute extraction ("red, 128GB, under $500") |
| Retrieval | two-tower dense ANN (HNSW/IVF-PQ) **plus** BM25 — dense models fail on SKUs, model numbers, exact part codes |
| Filter | category, price, stock applied *during* ANN traversal, not after |
| Rank | cross-encoder on top-100 → 20, then business boosts (margin, delivery) applied as a rerank, not inside the embedding |

**Key failure modes**

- **Baking volatile attributes into embeddings**: price and stock become stale the moment they change. They are filters/boosts only.
- Filtered-ANN recall collapse at high selectivity (a rare category combined with a price band leaves nothing) — widen the candidate set or use a partition per category.
- Cold-start SKUs with no behavioural signal; fall back to text/attribute similarity.
- Taxonomy change invalidating the embedding space; version embeddings per model.

**How to validate**

- Labelled query set (2–5k) for nDCG@10 and recall@100, per language and per category.
- Online A/B on CTR and conversion, plus zero-result rate and "search then leave" rate.
- Latency budget test at p99 including filter selectivity extremes.

**中文要点**：价格和库存绝不能进 embedding（一改就过期），只能做过滤和加权；稠密向量必须配 BM25 兜住型号/SKU 精确匹配；高选择性过滤下 ANN 召回会塌，要按类目分区或放宽候选。

---

### 5. Design a content-moderation system combining classifiers and LLMs.

**Asked at**: Meta (Superintelligence Labs, FAIR, Llama)

**Anchor answer**

**Goal & constraints.** 100% coverage, sub-200 ms decision on the hot path, per-category severity policy, appealable decisions.

| Tier | Component | Share of traffic |
| :--- | :--- | :--- |
| 0 | normalisation + hash match against known violations | ~5% blocked instantly |
| 1 | small multimodal classifier (<10 ms) | 100% |
| 2 | LLM judge with policy rubric | 1–5% (borderline) |
| 3 | human review queue | 0.1–1% |

Run checks both **pre-generation** (prompt) and **post-generation** (output), the latter buffering or stream-blocking rather than retracting text already shown.

**Key failure modes**

- **False-positive rate at scale**: 1% FP on a billion items is ten million angry users. Tune thresholds per category by cost asymmetry, not for a single AUC.
- Adversarial obfuscation: homoglyphs, leetspeak, text embedded in images, coded slang — normalise and add nearest-neighbour matching to known violations.
- LLM-judge drift and inconsistency across prompt versions; pin the rubric version per decision.
- Policy updates silently regrading history; version the policy with every label.
- Streaming latency: the judge cannot be in the token path for chat.

**How to validate**

- Stratified human-labelled set per category: precision at the operating threshold, recall at the recall floor, plus a red-team suite refreshed monthly.
- Appeal rate and overturn rate as the real quality signal.
- Replay a fixed corpus on every model/policy change and diff decisions.

**中文要点**：分级过滤，小模型全量、LLM 只判边界样本；阈值按各类目的误杀代价调而不是追单一 AUC；对抗样本靠归一化 + 已知违规近邻，策略要版本化并可回放重判。

---

### 6. Design a document-intelligence pipeline: scanned PDFs in, structured fields out, at 10M documents.

**Asked at**: Palantir

**Anchor answer**

**Goal & constraints.** 10M docs × ~3 pages ≈ 30M pages; per-field accuracy above a business threshold; every extracted value auditable back to a page region; cost per 1k pages known and bounded.

| Stage | Design |
| :--- | :--- |
| Ingest | content-hash dedupe, queue, per-tenant isolation; skip OCR when a text layer exists |
| Parse | deskew/rasterise → OCR or native text → layout analysis → tables |
| Extract | doc-type classifier → type-specific schema; VLM or layout model + rules; per-field confidence and bounding box |
| Validate | cross-field rules: dates ordered, line items sum to total, ID checksums, currency/units |
| HITL | only low-confidence fields (~10–20%) into a side-by-side review UI |

**Key failure modes**

- **Silent OCR failure** on rotated, skewed or low-DPI scans producing plausible wrong digits — worse than a hard error.
- Vendor layout change breaking a template with no alert; monitor per-doc-type field distributions.
- Field-level confidence that is uncalibrated, so HITL routes the wrong 15%.
- Non-idempotent retries duplicating documents and double-counting totals.
- Long-tail doc types with a handful of examples.

**How to validate**

- Stratified labelled set per doc type: per-field precision/recall, not document-level accuracy.
- Straight-through-processing rate and review-queue precision as the operating metrics.
- Confidence calibration curve (predicted vs observed correctness).
- Replay a frozen 5k-doc set on every model change; track cost per 1k pages.

**中文要点**：最大的坑是 OCR 静默失败产出"看起来合理"的错误数字，所以要逐字段置信度 + 跨字段校验（行项目求和、日期序、校验位），并按文档类型监控字段分布漂移。

---

### 7. Design a Text-to-SQL system over a warehouse with thousands of tables.

**Asked at**: Databricks · Palantir

**Anchor answer**

**Goal & constraints.** Thousands of tables cannot fit in a prompt; wrong-but-plausible numbers are worse than a refusal; queries must be read-only and cheap.

| Stage | Design |
| :--- | :--- |
| Schema retrieval | embed table/column descriptions + sample values + past queries → shortlist 10–30 tables; join paths from the FK graph |
| Grounding | DDL + retrieved few-shot pairs + certified metric definitions |
| Generate | dialect-aware single statement; parse with a SQL parser, then `EXPLAIN`/`LIMIT 0`; reject on cost estimate |
| Execute | read replica, statement timeout, row cap; LLM summarises **only returned rows** |

**Key failure modes**

- **Silent wrong joins** — the query runs, returns a plausible number, and nobody catches it. This is the dominant failure; require the model to state assumptions and show the join reasoning.
- Metric ambiguity ("revenue" gross vs net, refunds in or out). Prefer a semantic layer of certified views over free-form SQL.
- Hallucinated columns: validate every identifier against the retrieved schema before execution.
- Expensive-query DoS on the warehouse; enforce cost gates, not just timeouts.
- Injection through the natural-language question.

**How to validate**

- Execution accuracy on a golden set compared by **result rows**, not SQL string equality.
- Human-rated correctness on a weekly sample, tracked per table family.
- Query-cost distribution and rejection rate; a rising rejection rate means retrieval is drifting.

**中文要点**：头号失败是"join 错了但数字看着合理"，所以要做 schema 检索 + 标识符校验 + 语义层认证指标，并用**结果行比对**而不是 SQL 字符串比对来做评测。

---

### 8. Design a meeting assistant: recording, diarisation, summary, action items, integrations.

**Asked at**: Microsoft

**Anchor answer**

**Goal & constraints.** Live transcript within 1–2 s; summary ready minutes after the call; correct action-item **owners**; regional consent law compliance.

| Stage | Design |
| :--- | :--- |
| Capture | bot joins the call or ingests per-participant streams; echo cancellation |
| ASR | streaming transcription + speaker embedding clustering for diarisation; timestamps and per-segment confidence |
| Summarise | map-reduce over segments (long meetings exceed context); chaptering; decisions vs discussion separated |
| Actions | owner + due date extraction with the utterance as evidence; create tickets idempotently |
| Store | audio and transcript in object store, per-tenant keys, retention policy, optional PII redaction |

**Key failure modes**

- **Diarisation error assigns a task to the wrong person** — the failure users notice and never forgive. Cross-check with the calendar attendee list and require a confidence gate before creating a ticket.
- Naive truncation of a 2-hour meeting loses the decision; must map-reduce.
- Jargon, accents, numbers and currency misheard; never let ASR numbers drive financial actions without confirmation.
- Cross-talk and remote/low-bandwidth audio collapsing speaker separation.
- Consent: recording notice per region, and deletion on request.

**How to validate**

- WER per domain and diarisation error rate on a labelled meeting set.
- Action-item precision/recall and owner accuracy against human-annotated minutes.
- End-to-end latency from call end to summary, and a consent-enforcement test suite.

**中文要点**：说话人分离错会把任务派给错的人，这是最不可接受的失败，所以建单前要用参会人列表交叉校验 + 置信度闸门；两小时会议必须 map-reduce，ASR 数字不能直接驱动财务动作。

---

### 9. Design an LLM gateway: routing across providers, failover, caching, budgets and rate limits.

**Asked at**: Perplexity · Palantir · Tencent · JD.com
**Source**: https://outcomeschool.com/blog/llm-routing · https://outcomeschool.com/blog/how-does-semantic-caching-work

**Anchor answer**

**Goal & constraints.** One OpenAI-compatible internal API; model aliases so apps never name a vendor; added overhead under 20 ms; per-tenant cost attribution exact enough to reconcile an invoice.

| Concern | Design |
| :--- | :--- |
| Routing | policy + cheap classifier (task, length, tenant tier) → model; canary percentages; fallback chain per provider |
| Caching | exact hash → provider prefix cache → semantic cache **scoped per tenant** |
| Limits | token-bucket per key/user/tenant, cost accounting from usage, hard stop vs graceful degrade |
| Reliability | timeouts, jittered retries, per-provider circuit breaker, degrade to a smaller model |

**Key failure modes**

- **Semantic-cache cross-tenant leak** — the one unrecoverable bug. Cache keys must include tenant and permission scope.
- Retry storms amplifying a provider outage; breakers must be per provider *and* per model.
- Mid-stream failover duplicating or truncating text; stream failover must surface a resumable error, not silently restart.
- Budget drift: cached tokens, reasoning tokens and image tokens priced differently — reconcile against invoices monthly.
- Vendor-specific features leaking into application code, defeating the abstraction.

**How to validate**

- Chaos drill: kill a provider and measure success rate plus added latency; measure gateway p99 overhead.
- Cache hit rate and a leak test suite that asserts tenant B never sees tenant A's cached answer.
- Reconcile computed spend vs provider invoice within 1–2%; per-request traces with prompt hash and token counts.

**中文要点**：语义缓存必须带 tenant/权限作用域，跨租户泄露是不可恢复的 bug；流式失败不能静默重开（会重复输出），熔断要按供应商+模型双维度，预算要能和账单对得上。

---

### 10. Design the serving stack for a consumer chat assistant at hundreds of millions of users.

**Asked at**: Anthropic · OpenAI · Google DeepMind · Meta (Superintelligence Labs, FAIR, Llama) · xAI · ByteDance · Tencent
**Source**: https://outcomeschool.substack.com/p/inside-chatgpt-what-happens-after · https://outcomeschool.com/blog/llm-inference-optimization

**Anchor answer**

**Goal & constraints.** Global p99 TTFT under ~1 s, ITL p99 under ~100 ms; survive a 10× viral spike; cost per turn tracked and falling.

| Layer | Design |
| :--- | :--- |
| Edge | Anycast/CDN, regional gateways, auth, per-tenant rate limits, request shaping |
| Frontend | stateless: session store, prompt assembly, safety pre-check, conversation history in KV/Dynamo |
| Router | model tiering (small model for the easy majority) + **prefix-affinity routing** so a conversation lands on the worker holding its KV |
| Inference | regional prefill/decode pools, TP intra-node, continuous batching, chunked prefill, prefix cache, FP8, speculative decode for premium tier |
| Capacity | warm spares, fast model load, active-active regions, admission control with SLO-aware queueing |

**Key failure modes**

- **Thundering herd** at viral moments: queue with admission control and priority shedding beats autoscaling, which cannot cold-start a 400 GB model in seconds.
- Prefix-affinity hotspots: consistent hashing without load bounds creates one hot worker per popular system prompt; use bounded-load hashing.
- Regional failover losing in-flight streams and duplicating messages.
- KV cache thrash when a popular prompt is evicted by a long-tail conversation.
- Cost blowup from one abusive tenant; enforce per-tenant token budgets, not just request limits.

**How to validate**

- Load test at 2–3× peak with the real prompt-length distribution (mean is useless; the tail sets capacity).
- Chaos drills: region drain, provider loss, hot-tenant flood.
- SLO dashboards per region for TTFT/ITL p99, plus a capacity model of tokens/s per pool vs forecast and a cost-per-turn trend.

**中文要点**：容量必须按**真实长度分布**压测而不是均值；前缀亲和路由能救缓存但会造热点，要用有界负载哈希；模型冷启动要几分钟，所以抗突发靠准入控制和降级，不靠自动扩容。

---

## 附：这 10 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 权限与合规意识 | 1（企业 RAG 权限）、8（录音合规） |
| 会不会把"错但合理"当成功 | 7（Text-to-SQL 错 join）、6（OCR 静默错） |
| 成本与容量思维 | 10（亿级服务栈）、9（网关预算） |
| 动作类系统的风险控制 | 3（客服 agent 退款）、8（任务派错人） |
| 平台抽象能力 | 9（网关）、2（代码助手索引与编辑闭环） |
