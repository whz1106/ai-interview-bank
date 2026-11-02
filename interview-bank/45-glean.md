# 45 · Glean

> 10 题。出处：`README.en.md` § Glean。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Merge ranked results from N connector shards into a global top-k, applying a per-user permission filter. Do it efficiently

**Source**: https://outcomeschool.com/blog/rag-evaluation

**Anchor answer**

**Reference implementation**

```python
import heapq

def merge_topk(shards, user, k, *, over_fetch=3, rrf_k=60):
    """shards: list of callables(q, user, n) -> [(doc_id, score)] sorted desc.
    Permission filtering MUST happen inside the shard query (pre-filter),
    so this loop never has to discard documents."""
    n = k * over_fetch
    results = [s(user, n) for s in shards]          # fan out in parallel
    merged, seen = [], set()
    # k-way merge with per-shard cursor, early stop when safe
    cursors = [0] * len(results)
    heap = []
    for i, r in enumerate(results):
        if r:
            heapq.heappush(heap, (-r[0][1], i, 0))
    while heap and len(merged) < k:
        neg, i, j = heapq.heappop(heap)
        doc_id, score = results[i][j]
        if doc_id not in seen:                       # dedupe across connectors
            seen.add(doc_id)
            merged.append((doc_id, -neg, i))
        if j + 1 < len(results[i]):
            heapq.heappush(heap, (-results[i][j + 1][1], i, j + 1))
    return merged
```

Three things make this correct and fast:

1. **Permission filter pushed into each shard's index query** (metadata pre-filter). Filtering after a global top-k silently returns fewer than k results for restricted users — the classic bug.
2. **Score normalization**: raw scores from BM25, dense ANN, and a Jira recency ranker are not comparable. Use **RRF** (`1/(rrf_k + rank)`) when you cannot calibrate; use per-source z-score or quantile mapping only if you maintain calibration.
3. **Adaptive k per shard** — over-fetch `3k` from shards whose scores overlap, `k` from shards empirically never contributing. You need statistics on shard contribution, not a constant.

Early termination is safe only when the best remaining shard score is below the k-th merged score **after normalization** — with RRF that bound is on rank, not score, so most systems just take fixed `3k`.

**中文要点**：权限过滤必须下推到各分片的索引查询（前置过滤），否则全局 top-k 后过滤会让受限用户拿不满 k 条；跨源原始分不可比，用 RRF 按排名融合；每源用自适应过取倍数（约 3k），并做 doc 级去重。

---

### 2. Walk me through the latency budget of a query: query understanding → retrieval → rerank → LLM answer. Where do you spend and where do you cut?

**Source**: https://outcomeschool.com/blog/rag-evaluation

**Anchor answer**

Budget for a perceived-fast assistant is roughly **p95 < 1.2s to first token** and < 4s to a complete short answer. Allocate it explicitly:

| Stage | p50 / p95 (typical) | Spend or cut |
| :--- | :--- | :--- |
| Query understanding | 15 / 60 ms classifier; **150–350 ms** if an LLM rewrites/expands | Cut: LLM rewrite only for queries flagged ambiguous/long; short navigational queries skip it entirely |
| Retrieval (fan-out to shards) | 40 / 120 ms | Spend: parallel fan-out with a hard 150 ms deadline; tail-latency hedge one slow shard |
| Rerank (cross-encoder, 50–100 cands) | 80 / 250 ms | Cut hardest: prune to 40 candidates, batch on GPU, or route navigational queries past rerank |
| LLM answer (first token) | **300 / 800 ms** | Spend: this is the visible number; cache the system+retrieved prefix |
| Streaming completion | 1–3 s | Cut: cap answer length by intent |

Where the wins actually are:

1. **Parallelize retrieval and query understanding** — run the rewrite speculatively and discard if it contradicts.
2. **Rerank candidate pruning**: going 100 → 40 candidates usually costs single-digit nDCG points while halving rerank time; measure it, don't assume.
3. **Prefix caching + prompt-order discipline** (stable instructions first, volatile user context last) so the LLM's prefill is not re-paid every turn.
4. **Stream, and show retrieved sources before the answer** — perceived latency drops more from progressive disclosure than from any single stage optimization.

**中文要点**：先给每段定预算（rerank 与 LLM 首 token 是两大块）；能砍的是 LLM 改写（仅对模糊查询启用）和 rerank 候选数（100→40），能花的是并行分片与流式；把稳定指令放最前吃到前缀缓存，并先出检索结果再出答案以降低感知延迟。

---

### 3. How would you chunk and embed heterogeneous enterprise content: Slack threads, Jira tickets, Google Docs, PDFs?

**Source**: https://outcomeschool.com/blog/chunking-strategies-for-rag

**Anchor answer**

Rule: **chunk along the unit the author actually wrote**, then attach permission and provenance metadata to every chunk. Never apply one 512-token window to all four.

| Source | Retrieval unit | Keep together | Trap |
| :--- | :--- | :--- | :--- |
| Slack thread | Whole thread if < ~1.5k tokens; else thread summary + per-message chunks | Question and its accepted answer | Channel chatter dominates; needs per-channel ACL; reactions/joins are noise |
| Jira ticket | One chunk = summary + description + last N comments; separate chunk for resolution | Status, resolution, reporter/assignee | Comment threads with "@bot" noise; a ticket closed as duplicate is a false positive |
| Google Docs | Heading-path hierarchical chunks (H1→H2→H3 as prefix) | Tables, bullets with their parent heading | Repeated access-permission changes; comments are not content |
| PDF | Layout-aware parse, per section; tables as markdown | Tables, footnotes attached to their page | Scanned pages → OCR errors; multi-column order; header/footer repetition pollutes every chunk |

Shared metadata per chunk: `{source_type, source_id, url, title, section_path, author, updated_at, acl}` — `acl` is mandatory on every chunk because access is decided per document and the index is queried per user.

Embedding strategy: one embedding per chunk of the **normalized text** (title + section path + body), not raw markdown; for Slack, prefix with channel and participants, which is a strong relevance signal in enterprise search. Queries are short and acronym-heavy, so pair dense retrieval with BM25 over titles and entity names — the acronym problem is where pure embeddings lose.

**中文要点**：按作者的写作单位切：Slack 整帖、Jira 一票一文档、Docs 按标题层级带路径前缀、PDF 按版式解析且表格转 markdown；每个 chunk 必带 acl 元数据；稠密向量要与标题/实体名的 BM25 混合，否则缩写词查不到。

---

### 4. Why is RAG the right architecture for an enterprise assistant instead of fine-tuning on the company's data? Where does RAG break?

**Source**: https://www.youtube.com/watch?v=lnfWvX66FUk

**Anchor answer**

Because enterprise knowledge has four properties fine-tuning cannot satisfy:

| Property | Fine-tuning | RAG |
| :--- | :--- | :--- |
| Freshness — docs change hourly | Weight update = retrain, always stale | Index update = minutes |
| **Per-user permissions** | Weights cannot be filtered by identity | Filter at query time per principal |
| Attribution — "where did this come from" | Unrecoverable | Chunk ID + span |
| Deletion / legal hold / retention | Cannot reliably unlearn a document | Delete from index, provable |
| Cost of change | Per-tenant retrain on every content change | Shared model, per-tenant index |

That last point is decisive commercially: a fine-tuned model is one model per customer and per snapshot; a RAG system is one model plus N indexes.

**Where RAG breaks** (say these out loud — it is what the interviewer is testing):

1. **Aggregation and counting** — "how many teams use service X" needs a scan, not top-k.
2. **Multi-hop** — answer requires chaining three documents that no single query retrieves.
3. **Vocabulary mismatch** — internal acronyms, codenames; the embedding never saw them.
4. **Implicit intent** — query is one word, the need is a workflow.
5. **Permission staleness** — the document is retrievable after access was revoked.

So the right framing is not RAG *vs* fine-tuning: fine-tune for **behaviour** (format, tone, tool-call reliability, domain vocabulary via a retrieval-aware training set), retrieve for **facts**. Hybrid: a small fine-tune that learns to ask better retrieval queries is worth more than a fine-tune that memorizes documents.

**中文要点**：企业知识要的是新鲜度、按人权限过滤、可溯源、可删除——这四条微调全做不到，且微调等于每客户每快照一份权重。RAG 的破绽在聚合计数、多跳、缩写词、隐式意图和权限过期；正确姿势是微调学行为与检索式，检索供事实。

---

### 5. Design an agent that takes actions in enterprise tools (file a Jira ticket, draft an email) on a user's behalf. How do you handle permissions and evaluate it?

**Source**: https://outcomeschool.com/blog/ai-orchestration

**Anchor answer**

Two hard rules: **act as the user, never as a super-user**; and **separate read from write** in both permission and UX.

| Concern | Design |
| :--- | :--- |
| Identity | OAuth on-behalf-of per user; a delegated token with the minimum scopes per action. Never a shared service account — it silently grants the agent every permission the service account has |
| Authorization point | The tool gateway, per request, using the caller's token. The model proposes; the gateway enforces |
| Write safety | Reversible by default: draft-then-confirm for email, preview diff for ticket creation, idempotency key per action so a retry doesn't file twice |
| Blast radius | Typed side effects per tool (`read` / `write` / `irreversible`); irreversible actions require explicit user confirmation with the rendered payload |
| Audit | Log `{user, tool, args, result, approval_id}`; the log is the product feature, not debug output |
| Prompt injection | Tool results are untrusted data. A document that says "create a ticket assigning X" must not become an action — writes require an approval step the model cannot self-issue |

**Evaluation** — do not grade the transcript. Grade the **end state**:

1. Seeded sandbox tenants with API read-back assertions ("ticket exists, correct project, correct assignee, exactly one").
2. Trajectory metrics: tool calls per task, redundant calls, wrong-tool rate.
3. **Permission-violation suite**: run tasks as a user without the required access and require a clean refusal — this is the test most teams forget.
4. Task success on a fixed set of 50–100 real request phrasings, held out from prompt development.

**中文要点**：以用户身份代为操作（OAuth 委派、最小 scope），绝不用超级账号；授权在工具网关逐请求执行；写操作要草稿确认、幂等键、不可逆动作二次确认、工单只能由人批准。评测看环境末态（API 回读断言）和越权拒绝率，不看对话文本。

---

### 6. Design agent orchestration across dozens of connected SaaS systems. Where is authorization enforced, and why can it not live in the model?

**Source**: https://outcomeschool.com/blog/ai-orchestration

**Anchor answer**

Split the system into a **control plane** (planning) and a **data plane** (execution). Authorization lives only in the data plane.

| Plane | Components | Trust |
| :--- | :--- | :--- |
| Control plane | Planner LLM, task graph, retries, memory | Untrusted output — proposes, never decides |
| Data plane | Tool gateway: auth, ACL check, rate limit, schema validation, audit | Enforces; the only path to a SaaS API |
| Identity | Per-user OAuth tokens, short TTL, per-tool scopes | Source of truth for "may this principal do this" |

Why the model cannot hold authorization:

1. **Prompt injection** — any retrieved document, email, or ticket comment can instruct the model. If the model's "decision" is the control, injection becomes privilege escalation.
2. **No verifiable identity** — a model's claim "the user has access" is self-asserted text. The gateway can check a signed token; the model can only assert.
3. **ACLs change under you** — permissions are revoked mid-session; only a live check against the authz service sees that.
4. **Audit** — you must be able to answer "who authorized this action" with a token, not a transcript.

Mechanics that make it hold up: every tool is registered in a gateway with a typed schema and an explicit permission requirement; the planner receives **capability descriptors**, not raw credentials; cross-tool orchestration passes IDs, never ambient credentials; and the gateway returns `denied` as a first-class observation so the agent can replan rather than hallucinate success. Rate-limit and fan-out caps sit in the gateway too, because 40 parallel SaaS calls from one plan is how you get a tenant-wide 429.

**中文要点**：控制面（规划）与数据面（工具网关）分离，授权只在网关逐请求做，用用户自己的 OAuth token；模型不能持有授权，因为提示注入不可信、身份不可验证、ACL 会中途变化、审计要凭 token 而不是对话文本。网关把 denied 作为一等观察返回让 agent 重规划。

---

### 7. Design a connector framework that syncs content and permissions from 100+ SaaS apps into one index

**Source**: https://outcomeschool.com/blog/ai-orchestration

**Anchor answer**

Treat it as a **data pipeline with two synchronized streams** — content and principals — because a document that arrives before its ACL is an outage waiting to happen.

| Component | Responsibility |
| :--- | :--- |
| Connector SDK | One interface: `auth`, `list_changes(cursor)`, `fetch(id)`, `list_acl(id)`, `list_groups()`, rate-limit policy, `webhooks?`. 100 connectors, one contract |
| Change detection | Webhook where the SaaS offers it; otherwise cursor/poll with a per-source interval and a `modified_since` checkpoint. Full resync weekly as a safety net |
| Normalizer | Map to a unified doc model: `{doc_id, source, url, title, body, owner, acl, updated_at, hash}` |
| Indexer | Idempotent upsert keyed on `(source, doc_id, hash)`; skip unchanged via content hash |
| ACL stream | Store **group→doc edges and user→group membership separately**; resolve at query time. Never flatten a group into members — "all-hands" has 10k people |
| Deletion | Tombstone on delete event; nightly orphan sweep for docs whose parent vanished; deletes propagate to vectors, cache, and memory stores |
| Isolation | Per-connector quarantine: a broken parser or a rate-limited source must not stall the other 99 |

Failure modes that actually bite:

1. **Permission drift** — content syncs in seconds, ACL sync lags hours; a user briefly sees a confidential doc. Enforce a per-document `acl_ready` flag, and default to *not retrievable* until the ACL is present.
2. **Cursor invalidation** — SaaS APIs expire cursors; you must detect the error and fall back to a bounded re-scan, not silently stall.
3. **Deny semantics** — many systems have explicit denies and external-sharing flags that beat group grants; model them as a separate layer above grants.
4. **Backfill storms** — 100 connectors × first sync; shard by source and tenant with an ordered, resumable queue.

**中文要点**：内容流与权限流必须分开且各自可校；把 group→doc 与 user→group 分开存、查询时解析（不要把组拍平成成员）；文档在 ACL 未同步前默认不可检索；删除要有墓碑 + 夜间孤儿扫描；每个连接器隔离熔断，避免单源拖垮全量。

---

### 8. Glean's ranking leans on a knowledge graph of people, content and activity. How would you build that graph, and how does it improve retrieval beyond embedding similarity?

**Source**: https://outcomeschool.com/blog/graphrag

**Anchor answer**

Build three node types and the edges between them, then use it as a **signal and an expansion path**, not as a replacement for the vector index.

| Node | Edges (with weight/decay) |
| :--- | :--- |
| Person | authored, edited, commented on, viewed (time-decayed), manages, in-team |
| Content | links-to, references (ticket↔PR↔doc), contains-entity, duplicate-of, supersedes |
| Entity/Project | mentioned-in, owned-by, member-of-team |

Construction: (a) ingest from connectors — authorship and link structure are already there; (b) **entity resolution** is the real work: canonicalize project codenames, acronyms, aliases, and person identities across systems (same human, two `user_id`s); (c) activity edges from access logs with exponential decay, otherwise a 2021 hot document outranks this quarter's; (d) compute per-user Personalized PageRank or a cheap 2-hop expansion offline/cached.

What it buys beyond embedding similarity:

1. **Freshness and authority** — an embedding cannot tell that this doc is the current runbook and that one is the deprecated copy; the `supersedes` edge can.
2. **Entity grounding** — "billing service" resolves to the canonical project node instead of matching surface text.
3. **Multi-hop** — "the doc the payments team uses for deploy" = query → team → owned docs, which no single-query dense retrieval reaches.
4. **People as a path** — expertise ("who knows about X") emerges from authored+edited edges; that is a feature embeddings cannot provide at all.

Cost: graph freshness and entity resolution dominate. Start with the edges you get for free (authorship, links, membership) and add activity edges only after you can measure their lift with interleaving.

**中文要点**：图谱当信号与扩展路径而非替代向量检索：人/内容/实体三类节点 + 带时间衰减的活动边；实体消解（命名歧义、跨系统同一人）是主要成本；它带来时效与权威性判断、实体归一、多跳（团队→文档）和"谁懂这个"，这些向量相似度给不了。

---

### 9. You have dozens of ranking signals and a brand-new tenant with zero interaction data. How do you rank, and how do you improve?

**Source**: https://outcomeschool.com/blog/rag-evaluation

**Anchor answer**

Cold-start is a **feature availability** problem, not a model problem. Sort your signals by whether they need tenant behaviour:

| Signal class | Needs clicks? | Cold-start usable |
| :--- | :--- | :--- |
| Query–doc relevance (BM25, dense, title/entity match) | No | Yes — full weight |
| Content priors (doc length, freshness, author authority, in-degree links) | No | Yes |
| Popularity | Yes, but cross-tenant priors may transfer | Only if sharing is contractually allowed — usually it is not |
| Learned ranker weights | Yes | No — a model trained on other tenants' click logs does not transfer, and using their data is a privacy violation |
| Session behaviour | Within-session | Yes, after the first few queries |

So rank v0 with a **hand-tuned linear blend** of query–doc relevance plus content priors, then convert it:

1. **Constrained learning** — fit a small number of weights (or a GBDT on non-behavioural features) with heavy regularization; do not deploy a neural LTR on 0 samples.
2. **Explore cheaply** — interleave with a bandit over a few ranking variants; exploration must be bounded so a new tenant does not get visibly bad results.
3. **Transfer structure, not data** — the *feature set* and *training pipeline* transfer; the weights re-fit per tenant as its logs accumulate (a tenant-level intercept + shared slopes is a reasonable middle).
4. **Mine the tenant's own signals early**: query logs (even without clicks) give query reformulation and zero-result patterns; admin config (boosted sources, pinned docs) is explicit relevance data.

Measure with interleaving rather than A/B on a small tenant — you will not reach significance on clicks alone for months.

**中文要点**：冷启动是特征可得性问题：把信号分成"不依赖点击"（相关性、新鲜度、作者权威）和"依赖点击"（排序权重）；v0 用人工加权的线性/小 GBDT 模型，新租户不能直接套别的租户的点击权重（既是分布漂移也是隐私问题）；用 interleaving 做实验，因为小租户的 A/B 永远不显著。

---

### 10. Design the evaluation framework for an enterprise AI assistant when you cannot look at customer data

**Source**: https://outcomeschool.com/blog/rag-evaluation

**Anchor answer**

You evaluate on data you are allowed to see, and you **ship the harness** for the data you are not. Four tiers:

| Tier | Where it runs | What it answers |
| :--- | :--- | :--- |
| Public + synthetic corpora | Your infra | Regression: retrieval recall, groundedness, latency, cost. Synthetic tenants generated with realistic ACL graphs and acronym-heavy queries |
| Customer-side harness | Inside their trust boundary | Same metrics, only aggregates leave: they run it, post numbers, not content |
| Opt-in shadow / sampled review | With written consent, redacted | Real failure modes you cannot synthesize: internal jargon, weird permissions |
| Online product metrics | Their analytics | Task completion, escalation to human, thumbs, deflection — diff-in-diff against a holdout |

Metrics that survive "no data access":

1. **Retrieval recall@k on customer-authored golden questions** — they write 50 questions with the correct document URL; you get a number, you never see the document.
2. **Groundedness** — automated citation-entailment on the generated answer, which needs only the answer plus retrieved chunks, run in-tenant.
3. **Refusal correctness** — for questions whose answer does not exist, the rate of a clean "not found" instead of a confident invention.
4. **Permission correctness** — a synthetic tenant whose ACL graph you control, with planted canaries: any canary appearing in an answer for an unauthorized user is a hard fail.

The tractable habit: build a **customer-runnable eval binary** from day one and make every internal eval a subset of it, so results are comparable across the boundary. Otherwise your offline number and their number diverge and nobody can tell which is wrong.

**中文要点**：把评测拆四层：自有公集与合成租户（含可植入的权限金丝雀）、客户侧自跑只回聚合指标、书面同意的影子抽样、线上 diff-in-diff；关键指标是客户自写的 50 条金标问题的 recall@k、答案引用蕴含度、"查无此条"的拒答率、以及越权金丝雀泄漏率。第一天就把评测做成可交付二进制。

---

## 附：这 10 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 检索与融合的基本功（含权限前置过滤） | 1、2 |
| 异构内容与摘要/嵌入工程 | 3、4 |
| 权限与连接器系统设计（Glean 核心域） | 6、7 |
| 排序冷启动与图谱信号 | 8、9 |
| Agent 越权风险与末态评测 | 5、10 |
| 客户侧不可见数据下的评测设计 | 10 |
