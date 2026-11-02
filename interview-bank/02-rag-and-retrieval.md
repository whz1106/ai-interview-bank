# 02 · RAG and Retrieval

> 12 题。出处：`README.en.md` § RAG and Retrieval。答案为本仓库自行整理。

---

### 1. What chunking strategy would you use for a large technical documentation corpus, and why?

**Asked at**: Glean
**Source**: https://outcomeschool.com/blog/chunking-strategies-for-rag

**Anchor answer**

No single strategy — pick by document structure, and prefer **structure over fixed size**.

| Corpus | Strategy |
| :--- | :--- |
| API docs / Markdown / manuals | **heading-tree split**: chunk = a leaf section, carrying the full ancestor heading path as metadata |
| Narrative prose | recursive split on paragraph → sentence, 300–800 tokens, 10–15% overlap |
| Tables | keep the table whole; if too large, split by rows **and repeat the header**; never split a row |
| Code | split by function/class, keep signature + docstring in the same chunk |
| Legal / standards | split by clause number; citation integrity beats size uniformity |

Two production refinements:

- **Parent-child**: index small children for precise matching, but **return the parent** to the model for context. Best precision/context tradeoff.
- **Metadata is not optional**: source path, section, version, effective date, ACL. Without version/effective-date you cannot answer "which rule was in force then".

Avoid: fixed sliding window on technical docs — it cuts tables and mid-sentence definitions, and produces chunks that mean nothing out of context.

**中文要点**：按文档结构切而不是按字数切；技术文档用标题树+祖先路径，表格整块保留并重复表头，法条按条款；再叠父子块（子块检索、父块入上下文）。

---

### 2. How do you choose between a sparse retriever (BM25) and a dense retriever? When do you need both?

**Asked at**: Microsoft · Perplexity · Glean · Tencent · JD.com · Meituan · DiDi
**Source**: https://outcomeschool.com/blog/how-does-hybrid-search-work

**Anchor answer**

| | Sparse (BM25) | Dense (embedding) |
| :--- | :--- | :--- |
| Strong at | exact strings: model numbers, error codes, SKUs, clause ids, names | paraphrase, synonyms, vague natural language |
| Weak at | vocabulary mismatch | rare tokens, numeric precision, out-of-domain names |
| Cost | cheap, lexical only | embedding + ANN index |
| Explainability | high (term matches) | low |

**Enterprise corpora almost always need both**, because real queries mix natural language with exact identifiers ("为什么 SKU-8721 的毛利率降了" / "show me error E4021").

Fusion: use **RRF (reciprocal rank fusion)** rather than score weighting — BM25 and cosine scores have different scales and distributions, so weighted sums are unstable; RRF only uses ranks.

`score = Σ 1/(k + rank_i)` with k ≈ 60.

**中文要点**：稠密管语义、BM25 管精确串（型号/错误码/条款号）；企业场景基本都要，融合用 RRF 因为只依赖排名不依赖分数量纲。

---

### 3. What is a reranker, when should you use one, and what does a cross-encoder cost you?

**Asked at**: Cohere · Microsoft · Perplexity · JD.com · Alibaba
**Source**: https://outcomeschool.com/blog/how-does-a-reranker-work

**Anchor answer**

A **cross-encoder** takes `(query, document)` **jointly** and outputs a relevance score — it can model term interactions, which a bi-encoder (independent embeddings + dot product) structurally cannot.

Cost structure:

- Bi-encoder: 1 query encode + N dot products → microseconds, but each doc is encoded **without seeing the query**.
- Cross-encoder: **N forward passes**, each over `query + doc` → typically 10–20× the latency of the retrieval step, and it dominates the budget.

So the standard pipeline is two-stage: **ANN retrieve top-50~100 → cross-encoder rerank top-5**.

When *not* to use it: latency-sensitive interactive paths, very large k, or when the first-hit precision is already fine. Measure both — it's common for reranking to lift Recall@3 substantially (0.92 → 0.96) while blowing P95 from ~90 ms to ~1.7 s. **Make it a per-scenario switch, not a default.**

**中文要点**：cross-encoder 逐对打分精度高但延迟是 N 倍；标准做法是 ANN 粗召回 50~100 再精排 top-5；实测常见"提升中段召回但 P95 放大 10~20 倍"，要按场景开关而不是默认开。

---

### 4. How would you evaluate the quality of a RAG pipeline: retrieval and generation separately?

**Asked at**: Cohere
**Source**: https://outcomeschool.com/blog/llm-evaluation

**Anchor answer**

**You must separate them**, otherwise a bad answer tells you nothing about cause.

| Stage | Metrics | What it answers |
| :--- | :--- | :--- |
| Retrieval | Recall@k, Precision@k, MRR, nDCG, hit rate | Was the right evidence in the candidate set at all? |
| Generation | Faithfulness / groundedness, answer relevancy, citation accuracy, refusal correctness | Given the evidence, did it answer correctly *and only from evidence*? |

Practices that separate real work from "I ran RAGAS once":

1. **Break down by corpus shape** (long doc / table / FAQ / scanned) — global averages hide the weak slice.
2. **Measure latency per stage** — retrieval P95 vs generation TTFT.
3. **Include negative cases**: questions whose answer is *not* in the corpus. Correct refusal rate and false-refusal rate must both be reported.
4. **Freeze the eval set with version**; every change re-runs the same set.

**中文要点**：检索侧看 Recall@k/MRR/nDCG，生成侧看忠实度/相关性/引用准确率/拒答正确率；必须按语料形态分桶，并同时报"该拒答率"和"误拒答率"。

---

### 5. What is HyDE (hypothetical document embeddings) and when does it outperform standard dense retrieval?

**Source**: https://outcomeschool.com/blog/how-does-hyde-work

**Anchor answer**

HyDE: instead of embedding the **question**, ask the LLM to write a **hypothetical answer document**, then embed *that* and search with it.

Why it can work: questions and answers live in different regions of embedding space ("利息怎么算" vs a paragraph about 年化利率计算公式). A hallucinated-but-well-shaped answer sits closer to the real passage.

When it helps: short, vague, jargon-light user queries against long formal documents.
When it does not: already-specific queries (adds latency, adds noise), factual lookups where the hallucinated document drifts the wrong direction, or latency-sensitive paths (extra LLM call before you even search).

Practical: run it as a **conditional strategy** — only when the original query is short/low-confidence, and keep it measurable via A/B rather than turning it on globally.

**中文要点**：HyDE = 先让模型编一份"假答案"再拿它去检索，解决问法与文档表述不在同一语义区；但只在"短而模糊的提问 + 正式长文档"时有效，要做条件触发和 AB 验证。

---

### 6. How does agentic RAG differ from standard RAG, and when is the extra complexity justified?

**Source**: https://outcomeschool.com/blog/agentic-rag

**Anchor answer**

| | Standard RAG | Agentic RAG |
| :--- | :--- | :--- |
| Flow | one retrieval → one generation | model decides **whether/what/how many times** to retrieve, may reformulate, decompose, or call other tools |
| Handles | single-hop fact lookup | multi-hop, comparison, "insufficient evidence → go find more" |
| Cost | 1 search + 1 LLM call | variable, easily 3–10× |

Justified when:
- questions genuinely need multi-hop (compare this year's policy vs last year's, across two systems)
- the corpus is heterogeneous and routing matters (which index to hit)
- you need "refuse if evidence insufficient" instead of always answering

Not justified when: the query distribution is simple lookups. Then agentic RAG adds latency, cost, and failure modes (loops, over-retrieval) for no gain.

The honest interview answer includes: **"start with standard RAG + good retrieval, add agency only when the eval shows single-shot retrieval is the bottleneck."**

**中文要点**：Agentic RAG 的差别是"模型自己决定要不要检索、检索几次、怎么改写"；只在多跳/异构语料/需要拒答时值得，简单查找加它只是加成本加故障面。

---

### 7. What causes semantic drift in embedding search and how do you detect it?

**Asked at**: Cohere

**Anchor answer**

Four distinct causes, often conflated:

1. **Distribution drift** — the corpus or user queries shift (new product line, new slang) so the frozen embedding space no longer aligns. Detection: monitor the distribution of top-1 similarity scores; a steady downward shift is the early signal.
2. **Model/version drift** — you upgrade the embedding model but don't re-embed the whole index → query and document vectors live in **different spaces**. This is silent and catastrophic. Detection: enforce an embedding-version tag per vector and assert query/doc versions match.
3. **Chunking drift** — chunking rules changed mid-index, so half the index is structured differently.
4. **Query drift** — same words, new intent (a product name reused).

Detection in practice: fixed golden set run on a schedule + **top-k similarity histogram monitoring** + zero-result-rate and low-score-rate alerts + sampling retrieved results for human review.

**中文要点**：四种漂移——语料/问法分布漂移、**embedding 模型升级但索引没全量重建（最致命且静默）**、切分规则变更、词义漂移；靠"向量带 embedding 版本号 + 相似度分布监控 + 固定黄金集定期跑"来发现。

---

### 8. Design permission-aware retrieval: users must never see content they can't access in the source system.

**Asked at**: Microsoft · Databricks · Glean · Palantir · ByteDance

**Anchor answer**

Non-negotiable rule: **filter at retrieval time, inside the index query — not after generation, and never via prompt instructions.**

Design:

1. **ACL as metadata on every chunk**: owner, group/role ids, tenant id, sensitivity label. Recompute on ACL change (sync job from the source system, not a manual copy).
2. **Pre-filter in the vector search**, not post-filter. Post-filtering k=10 then dropping 8 leaves you 2 results and leaks nothing but performs badly; pre-filter keeps recall.
3. **Tenant isolation at the storage layer**: separate collections/partitions per tenant, so a bug in filter code cannot cross tenants.
4. **Prevent indirect leakage**: citations, source links, "similar documents", and autocomplete can all leak titles even if bodies are blocked.
5. **Verify adversarially**: build a probe set of queries that *should* return nothing for a given role, run it on every deploy, and require **0 hits**. This is the only real proof.

Also: cache keys must include the permission scope, or a cached answer will leak across users.

**中文要点**：权限必须在**检索入口**做（向量库 pre-filter + 元数据 ACL），不能生成后过滤、更不能靠提示词；缓存 key 要带权限域；用越权探针集（要求命中 0）来证明。

---

### 9. Compare HNSW, IVF-PQ and flat indexes. How do you pick, and what does recall@k cost in latency?

**Asked at**: Tencent · Meituan · JD.com
**Source**: https://outcomeschool.com/blog/how-does-approximate-nearest-neighbor-ann-search-work

**Anchor answer**

| | Flat (brute force) | HNSW | IVF-PQ |
| :--- | :--- | :--- | :--- |
| Principle | exact scan of all vectors | multi-layer navigable graph, greedy descent | coarse k-means buckets + product quantization |
| Recall | 100% | 95–99% (tunable via `efSearch`) | lower, tunable via `nprobe` |
| Latency | O(N) — fine up to ~10⁵ | very low, O(log N)-ish | low, but needs more buckets |
| Memory | N × d × 4B | **highest** (graph edges + full vectors) | **lowest** (compressed codes) |
| Build cost | none | high (graph construction) | medium (k-means training) |
| Mutation | trivial | harder (deletes degrade the graph) | easy-ish |

Selection heuristic:

- < ~100k vectors or need exactness → **flat**
- memory affordable, latency critical, high recall → **HNSW**
- 10⁸+ vectors, memory-bound → **IVF-PQ** (or HNSW on PQ codes)

**On "what recall@k costs in latency"**: ANN trades recall for speed via search-breadth parameters — HNSW's `efSearch`, IVF's `nprobe`. More breadth → higher recall, roughly linearly more latency. You must **plot the recall-vs-P95 curve on your own data** and pick the knee; there is no universal setting.

**中文要点**：flat 精确但 O(N)（10 万级以内够用）、HNSW 快但吃内存且删除会退化、IVF-PQ 省内存但精度换压缩；召回与延迟通过 `efSearch`/`nprobe` 调，必须在自己数据上画 recall–P95 曲线选拐点。

---

### 10. How do you handle tables, figures and multi-column PDFs in a retrieval pipeline?

**Anchor answer**

This is where most RAG demos die. Treat it as **layout parsing**, not text extraction.

1. **Layout-aware parsing**: use a doc parser that preserves reading order, block type (title/para/table/figure/caption), and bounding boxes (MinerU / Docling / Marker class tools, or a layout model). Plain `pdfplumber.extract_text()` destroys multi-column order.
2. **Tables**: extract to structured form (HTML/Markdown), **never split a row**, repeat the header on split, and keep the table as one retrievable unit keyed by table caption + section.
3. **Figures**: generate a **textual description** (VLM caption + OCR of any embedded text) and index that text; store the image reference for citation. Retrieval happens on text, display happens on image.
4. **Cross-page tables**: stitch by detecting continuation (same column count + repeating header + no new caption) before chunking.
5. **Citation must survive**: every chunk carries page number + bbox so the UI can highlight the source.

**中文要点**：表格/图/多栏 PDF 本质上要做**版面解析**（保留阅读顺序与块类型），表格整行为单位不切行并重复表头，图用 VLM 生成文字描述再索引文本，跨页表先拼合再切分。

---

### 11. How do you keep an index fresh when the underlying corpus changes continuously?

**Asked at**: Perplexity · Cursor · ByteDance

**Anchor answer**

Four problems, four answers:

| Problem | Solution |
| :--- | :--- |
| Detect change cheaply | **content hash** (MD5/SHA) per document; compare on scan or via filesystem/CDC events. Skip unchanged docs entirely |
| Update without full rebuild | incremental upsert keyed by `doc_id` + version; delete old vectors for that doc first |
| Deletes are expensive in vector DBs | use **partition/namespace by doc or tenant**, drop the partition instead of deleting rows; or maintain a soft-delete tombstone filtered at query time |
| Version semantics | store `effective_from` / `effective_to`; "current rule" queries filter by time, historical queries don't. Never overwrite the old version if the business needs auditability |

Plus: **debounce** the watcher (Office temp files `~$*`, editor autosaves), write to a **staging** area and swap atomically, and monitor index lag (time from source edit to retrievable).

**中文要点**：内容哈希增量 + 变更即删旧向量；删除靠**分区裁剪**而不是逐条删；带生效时间做版本过滤；watcher 要防抖并过滤临时文件；用 staging + 原子切换。

---

### 12. How do you attribute every claim in a generated answer to a specific retrieved span?

**Asked at**: Perplexity · Harvey · Abridge

**Anchor answer**

Structure the output, then **verify it mechanically**.

1. **Force structured output**: each claim as `{claim_text, chunk_id, quote}` — quotes, not just chunk ids. A chunk-level citation is too coarse to check.
2. **Verify three things server-side** (never trust the model's self-report):
   - the `chunk_id` was actually in *this* request's retrieved set (not from another session)
   - the `quote` **exists verbatim (or fuzzily) inside that chunk**
   - the chunk is inside the user's permission scope
3. **Fail closed**: a claim with no verifiable quote is either dropped, downgraded to "related but uncited", or the whole answer is marked 待核查.
4. **UI-level traceability**: clicking a citation opens the source at the right page/anchor. If you can't do that, the attribution is cosmetic.

Metric to report: **citation precision** (fraction of citations that verify) and **claim coverage** (fraction of claims that carry a verified citation). Reporting only coverage lets a system cite wrongly and still look good.

**中文要点**：要引到**片段原文（quote）**而不是只给文档 id；服务端三查——chunk 是否在本次召回集、quote 是否真在原文里、是否有权限；同时报"引用精确率"和"主张覆盖率"两个指标。

---

## 附：这 12 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| "纯 RAG 调包侠" | 4（分开评）、12（claim 归因）、2（BM25 何时要） |
| 是否做过企业落地 | 8（权限感知）、11（索引新鲜度）、10（表格/多栏 PDF） |
| 是否真调过参 | 3（reranker 代价）、9（ANN 召回-延迟曲线） |
| 是否分得清复杂度 | 6（agentic RAG 何时值得）、5（HyDE 何时有效） |
