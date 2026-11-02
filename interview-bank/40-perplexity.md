# 40 · Perplexity

> 15 题。出处：`README.en.md` § Perplexity。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Implement a client pool over multiple LLM providers with failover: providers fail, time out, or rate-limit, and callers should just get a completion.

**Source**: https://outcomeschool.com/blog/llm-routing

**Anchor answer**

**Reference implementation**

```python
import asyncio, random, time
from enum import Enum

class Outcome(Enum):
    OK, RETRYABLE, FATAL = "ok", "retryable", "fatal"

def classify(exc):
    # 429 / 5xx / timeout / connection reset -> retryable; 400 / 401 / content policy -> fatal
    if isinstance(exc, (asyncio.TimeoutError, ConnectionError)):
        return Outcome.RETRYABLE
    code = getattr(exc, "status", None)
    return Outcome.RETRYABLE if code in (429, 500, 502, 503, 504) else Outcome.FATAL

class Provider:
    def __init__(self, name, client, limit=64, cooldown=5.0):
        self.name, self.client, self.sem = name, client, asyncio.Semaphore(limit)
        self.fails, self.open_until, self.cooldown = 0, 0.0, cooldown

    def healthy(self):
        return time.monotonic() >= self.open_until

    async def call(self, req, timeout):
        async with self.sem:
            try:
                r = await asyncio.wait_for(self.client.complete(req), timeout)
                self.fails = 0
                return r
            except Exception as e:
                self.fails += 1
                if self.fails >= 3:                      # circuit breaker
                    self.open_until = time.monotonic() + self.cooldown
                raise e

class Pool:
    def __init__(self, providers, attempts=3, base_timeout=20):
        self.providers, self.attempts, self.base_timeout = providers, attempts, base_timeout

    async def complete(self, req):
        last = None
        for i in range(self.attempts):
            cands = [p for p in self.providers if p.healthy()] or self.providers
            p = random.choice(cands)                          # or weighted by cost/latency
            try:
                return await p.call(req, self.base_timeout)
            except Exception as e:
                last = e
                if classify(e) is Outcome.FATAL:
                    raise
                await asyncio.sleep(min(0.25 * 2**i, 4) * (1 + random.random()))
        raise last
```

**Key points**

- **Classify before retrying.** 429/5xx/timeout → another provider; 400/401/content-filter → re-raise immediately, because retrying a bad request burns quota on every provider.
- **Circuit breaker per provider**, not global. A dead provider should be skipped for ~5 s, not retried by every caller.
- **One global deadline per call**, not per attempt: 3 attempts × 20 s = a 60 s stall for an interactive caller. Pass the remaining budget down.
- **Streaming breaks failover.** You can only fail over *before the first token*; after that, restarting means the user sees duplicated text. Buffer the first token, then commit.
- **Hedging beats retrying** for tail latency: send a second request to another provider after p95, cancel the loser — costs more tokens, cuts p99 dramatically.
- **Idempotency / dedup key** on the request so a retry after a timeout doesn't bill twice; timeouts are ambiguous successes.

**中文要点**：先分类错误（429/5xx/超时→换家重试，400/401/内容过滤→直接抛）；每家一个熔断器；总 deadline 而不是每次尝试各自超时；流式只能在首 token 前切换；尾延迟用对冲请求而不是重试。

---

### 2. You're ingesting millions of web pages a day. Detect near-duplicates (same article, different boilerplate) efficiently.

**Anchor answer**

Two stages — cheap signature to generate candidates, exact check to confirm. Anything one-stage is either too slow or too inaccurate.

**Stage 0, and the one people skip: extract main content first.** Shingle the *article*, not the HTML. Unstripped pages differ in nav/footer, so duplicates look different, and different articles from one site look identical.

**Stage 1 — MinHash + LSH:**

- Shingle into 5-word (or 3-gram character) shingles; hash each to 64 bits.
- MinHash with `k = 128–256` permutations → a fixed-size signature per document.
- Band into `b` bands × `r` rows (`b·r = k`). Two docs collide in a band with probability `1 − (1 − s^r)^b`, so the threshold is `≈ (1/b)^(1/r)`. With 128 hashes as 32×4, the S-curve is steep around Jaccard ≈ 0.8.
- Store each band's hash in a KV/Redis set; a new doc probes `b` keys → candidate set, usually tiny.

**Stage 2 — verify:** exact Jaccard on the shingle sets for candidates only, or SimHash with Hamming distance ≤ 3 (cheaper, 64-bit, one comparison).

**Clustering and ops:** union-find over confirmed pairs, keep the canonical copy by a quality score (content length, domain authority, has images). Incremental insert only — never recompute the corpus. For exact substring dedup across crawls, a suffix array over the corpus is stronger but batch-only.

**中文要点**：先抽正文再 shingle，否则导航栏会让"同文不同页"和"同站异文"都判错。MinHash+LSH 出候选（128 哈希切 32×4 带，阈值≈0.8），再精确 Jaccard/SimHash 确认，用并查集聚类并按质量分留主副本。

---

### 3. You need to embed millions of text chunks. The embedding service takes batches with a max batch size and a max total-token limit. Write the batcher and make it fast.

**Anchor answer**

**Reference implementation**

```python
import asyncio
from collections import deque

def token_len(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))

def pack(lengths, max_items, max_tokens):
    """First-fit-decreasing: sort cheaply by length, fill each batch greedily."""
    order = sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True)
    batches, cur, cur_tok = [], [], 0
    for i in order:
        n = lengths[i]
        if cur and (len(cur) == max_items or cur_tok + n > max_tokens):
            batches.append(cur); cur, cur_tok = [], 0
        cur.append(i); cur_tok += n
    if cur:
        batches.append(cur)
    return batches

async def embed_all(texts, tokenizer, client, max_items=256, max_tokens=64_000, inflight=8):
    lengths = [token_len(t, tokenizer) for t in texts]      # tokenize ONCE
    batches = pack(lengths, max_items, max_tokens)
    sem = asyncio.Semaphore(inflight)
    out = [None] * len(texts)

    async def run(batch):
        async with sem:
            for attempt in range(4):
                try:
                    vecs = await client.embed([texts[i] for i in batch])
                    for i, v in zip(batch, vecs):
                        out[i] = v
                    return
                except Exception:
                    await asyncio.sleep(0.2 * 2**attempt)
            raise RuntimeError("embedding failed")

    await asyncio.gather(*(run(b) for b in batches))
    return out
```

**Key points**

- **Sort by length, then fill.** One 8k-token doc in an otherwise 100-token batch wastes the service's padding and your token budget; FFD keeps batches homogeneous.
- **Tokenize exactly once, in parallel**, and cache it — tokenization is often 30–50% of the wall clock at this scale, and the API tokenizes again anyway.
- **Two constraints, both binding.** `max_items` protects the service's per-request overhead; `max_tokens` protects its memory. Check *both* on every append, not one.
- **Bound in-flight requests** (`inflight ≈ 2× service concurrency`); unbounded `gather` over 10k batches is how you get 429s and a memory spike.
- **Retry per batch, not per job**, with a jittered backoff, and write results incrementally (Parquet shards / a DB) so a crash resumes instead of restarting 3 hours of GPU time.
- **Oversized single inputs**: anything above `max_tokens` must be split before batching, or it will deadlock the packer.

**中文要点**：按长度排序做 first-fit-decreasing，同时卡 `max_items` 和 `max_tokens`；token 只算一次并缓存（常占总耗时 30%+）；并发数要有上限，按批重试，结果增量落盘以便续跑。

---

### 4. Implement beam search for an autoregressive model. When would an answer engine actually use it?

**Anchor answer**

**Reference implementation**

```python
import torch

@torch.no_grad()
def beam_search(model, input_ids, beam=4, max_new=64, eos_id=2, alpha=0.6):
    B = input_ids.size(0)
    ids = input_ids.repeat_interleave(beam, dim=0)          # (B*beam, T)
    scores = torch.full((B * beam,), -1e9, device=ids.device)
    scores[:beam] = 0.0                                     # only one live hypothesis at start
    done = [[] for _ in range(B)]

    for step in range(max_new):
        logits = model(ids[:, -1:], use_cache=True).logits[:, -1, :]
        logp = torch.log_softmax(logits.float(), dim=-1)
        cand = scores.unsqueeze(1) + logp                    # (B*beam, V)
        for b in range(B):
            s = cand[b * beam:(b + 1) * beam].reshape(-1)
            top_s, top_i = s.topk(beam)
            for sc, idx in zip(top_s.tolist(), top_i.tolist()):
                tok = idx % logp.size(-1)
                prev = idx // logp.size(-1)
                if tok == eos_id:
                    # length-normalized score for fair comparison across lengths
                    done[b].append((sc / ((step + 1) ** alpha), ids[b * beam + prev].tolist()))
            # keep the few best non-EOS continuations
        # rebuild: pick global top-k among surviving beams, drop EOS rows
        ...
    return done
```

**Key points**

- **Length normalization is not optional.** Raw log-probabilities favour short sequences; without dividing by `len^alpha` the beam picks "I don't know" for every open-ended prompt.
- **Keep duplicate-aware bookkeeping**: two beams can converge on the same prefix; without dedup you waste 4× compute on one hypothesis.
- **EOS handling:** finished hypotheses leave the beam and go to a separate list; the beam must continue with the remaining ones, otherwise you terminate the whole search on the first EOS.
- **KV cache**: `beam` × batch is the memory multiplier; 8 beams × 32 concurrent users is 256 sequences of KV cache, which usually blows the budget before the model does.

**When an answer engine should use it:**

| Use beam | Don't |
| :--- | :--- |
| Short, constrained, high-stakes outputs: query rewriting, entity/JSON extraction, SQL, classification-style reformatting | Long-form answers — beams produce bland, repetitive, "safest" prose and you cannot stream them |
| Offline/precompute: generating suggested follow-ups, cached summaries | Anything interactive on a TTFT budget: beam width multiplies prefill+decode cost linearly |
| When determinism matters (eval/debug) | When diversity matters (candidate generation for reranking) |

**中文要点**：beam search 必须做长度归一化，否则偏向短句；还要处理 EOS 出束和重复前缀去重，KV cache 内存按 beam 数线性增长。答案引擎只在短、受限、可离线的输出（query 改写、抽取、JSON/SQL）用它；长文和流式场景不能用。

---

### 5. p95 time-to-first-token regressed from 1.2 s to 3 s after a release. Walk me through finding and fixing it.

**Source**: https://www.youtube.com/watch?v=XD8DD4cEHu0

**Anchor answer**

**Step 0: make sure it is real.** p50 flat + p95 tripled is a *queueing* signature, not a compute signature. Check whether the client-side metric includes retries, and whether the load mix changed (more long prompts, more concurrent users).

**Step 1: decompose TTFT.** Never debug "TTFT"; debug its parts:

`TTFT = queue wait + routing/auth + retrieval (if any) + prefill + first decode step`

Instrument each with a span. The regression lives in exactly one of them, and that already halves the search.

| Segment jumps | Cause |
| :--- | :--- |
| Queue wait | burst in arrival rate, autoscaling lag, a batch size increase that favours throughput over latency |
| Prefill | prompt got longer — new chat template, retrieval top-k raised, system prompt grew, image tokens added |
| Prefill | chunked-prefill size changed; scheduler now prioritizes decode |
| Prefill | **prefix cache miss** — the cache key changed (template, tool list, ordering) so the shared system prompt no longer hits |
| First decode | CUDA graph capture lost, speculative decoding disabled, quant kernel change |
| Routing | a feature flag or new model version re-routed traffic to a colder pool |

**Step 2: diff the release against the last good one.** Dependencies, model/weight version pin, tokenizer, templates, retrieval config, `max_tokens`, batch/scheduler flags. The most common real cause in practice: **the prompt got longer** (retrieval added context) and prefill is quadratic-ish in length, so +40% tokens ≈ +90% prefill.

**Step 3: bisect with replay.** Capture 1k real requests from Tuesday, replay against the new build with flags off/on. Confirm the segment, then fix and add a per-segment SLO to the dashboard so the next regression points at itself.

**中文要点**：先看 p50/p95 形状——p50 平而 p95 涨三倍是排队问题不是算力问题。把 TTFT 拆成"排队 + 路由 + 检索 + prefill + 首 decode"逐段打点；最常见真因是 prompt 变长（检索 top-k 或模板改动）和前缀缓存 key 变化导致 miss。

---

### 6. Discuss reranker architecture choices: cross-encoder, ColBERT, LLM-based.

**Source**: https://outcomeschool.com/blog/how-does-a-reranker-work ; https://outcomeschool.com/blog/decoding-colbert

**Anchor answer**

| | Cross-encoder | ColBERT (late interaction) | LLM-based |
| :--- | :--- | :--- | :--- |
| Scoring | joint forward pass over `[query, doc]`, one logit | `MaxSim` over precomputed token embeddings | prompt with candidates, generate a ranking/scores |
| Cost at query time | `n` forward passes (n = candidates) | `n ×` cheap MaxSim matrix ops | 1–k LLM calls |
| Docs precomputed? | no | **yes** — doc token embeddings stored offline | no |
| Typical latency | 100 candidates ≈ 100–300 ms on GPU | 1000s of candidates in ~ms | 0.5–3 s |
| Quality | best per candidate; sees full cross-attention | close to cross-encoder, better on long docs | best at reasoning/ambiguity; weakest calibration |
| Weakness | cannot scale past a few hundred | index 10–100× larger (one vector per token) | position bias, non-deterministic, expensive |

**How I'd wire it (a cascade, not a choice):**

1. Bi-encoder / BM25 recall → top 500–2000 (cheap, high recall).
2. ColBERT or a small cross-encoder → top 50–100. This is where you buy most of the nDCG.
3. Cross-encoder (or larger) → top 10–20 that actually enter the context window.
4. LLM listwise rerank only for high-value or ambiguous queries, on ≤20 candidates.

**Three things that matter more than the model choice:**

- **Train/distill for your distribution.** A public cross-encoder is trained on MS MARCO-style pairs; web answer retrieval is different (title/snippet, freshness, navigational intent). Distilling the cross-encoder into the bi-encoder is the highest-ROI step.
- **LLM listwise rerankers have position bias** — they favour candidates shown early and are sensitive to order. Always evaluate with shuffled orders, and prefer pointwise scoring over a generated permutation when you need stable output.
- **Don't rerank what you can filter.** Dedup, domain quality gates, and freshness filters should run before the reranker; they cost microseconds and remove whole classes of bad candidates.

**中文要点**：交叉编码器质量最好但每次 n 次前向；ColBERT 靠预计算 doc token 向量 + MaxSim，可扩展到上千候选但索引大 10~100 倍；LLM 重排强在推理却慢、有位置偏置。实务是级联：召回 500~2000 → ColBERT/小交叉编码器 50~100 → 交叉编码器 10~20。

---

### 7. You retrieved 50 candidate passages but the model's useful context budget is ~10. How do you choose, and how do you know your choices are good?

**Source**: https://outcomeschool.com/blog/how-does-a-reranker-work

**Anchor answer**

**Selection is not "top 10 by score".** That fails in three predictable ways: near-duplicates eat the budget, one sub-question gets all the slots, and the single passage that actually contains the answer sits at rank 12 because it is short and lacks query keywords.

**Selection recipe:**

1. **Dedup** — near-identical chunks (same page, overlapping windows) collapse to one; then re-expand the best one to its full section.
2. **Rerank** with a cross-encoder on `(query, passage)`, not bi-encoder cosine.
3. **Diversify with MMR or sub-question coverage** — split the query into facets, take the best passage per facet, then fill remaining slots by score.
4. **Budget by tokens, not passages** — 10 passages of 800 tokens is 8k tokens; 10 of 150 is 1.5k. Truncate long passages to the matching sentence window.
5. **Order matters** — put the highest-value passage first and second-best last ("lost in the middle": information in the center of a long context is measurably less used).

**How you know it is good — measure the selection, not just the final answer:**

| Metric | Question it answers |
| :--- | :--- |
| Recall@10 of the gold supporting passage | is the evidence even in the window? |
| nDCG@10 | is the ordering sane? |
| Context precision / redundancy | how many slots are wasted on duplicates? |
| Ablation: 10 vs 20 vs 5 passages | does the extra context change the answer at all? |
| Faithfulness / citation support | does the model actually use what you gave it? |

The blunt test: if adding passages 11–20 does not change answer accuracy, your selector is fine. If it *improves* accuracy, your reranker is misordering and you are just paying for a bigger context.

**中文要点**：选不是"按分数取前 10"——先去重、再交叉编码器重排、用 MMR/子问题覆盖保多样性、按 token 而非条数预算、把最有价值的放首尾。评估要看 gold 段落 Recall@10、冗余率，以及"5/10/20 条上下文"的消融。

---

### 8. Design an answer engine: a user types a question and gets a cited, streamed answer. Your end-to-end budget is 3 seconds to a complete short answer.

**Anchor answer**

**Budget it before designing it.** 3 s to a *complete short* answer means roughly: 100 ms query understanding, 400–600 ms retrieval (parallel), 100–200 ms rerank, ~300 ms TTFT, and ~1.5 s of generation at 40–70 tok/s for a 90–120-token answer. Anything that does not fit gets cut or moved off the critical path.

| Stage | Budget | Design |
| :--- | :--- | :--- |
| Query understanding | 100 ms | cache lookup first (exact + normalized); intent class (navigational / factual / comparative / fresh); rewrite to 2–3 search queries |
| Retrieval | 400–600 ms | fan out to indexes **in parallel** with hedged requests; per-source top-k; deadline per source, not global |
| Merge / dedup / rerank | 150 ms | dedup by URL + shingle; cross-encoder on top 50; select 6–10 with coverage |
| Prompt assembly | 20 ms | fixed system prefix (cached KV), stable citation ids |
| Generation | 1.5 s | stream; emit `[1][2]` markers as tokens, resolve to links client-side |

**The three decisions that make 3 s achievable:**

1. **Start generating on partial evidence.** Two of three retrieval sources returning is usually enough; the rest stream in and can extend the answer. Waiting for the slowest source is how p95 becomes 6 s.
2. **Cache aggressively at three levels:** whole-answer cache for head queries (seconds-fresh), retrieval cache (minutes), and prefix KV cache for the shared system prompt (per-request).
3. **Fixed citation numbering at retrieval time.** Assign ids when passages are selected, not after generation — post-hoc string matching is the main source of wrong attributions.

**Failure mode to plan for:** no sufficient evidence. The 3-second answer must be allowed to say "here is what I found, and it is thin" with 2 sources, rather than inventing confidence. Ship that as an explicit response type, not a fallback path.

**中文要点**：先切预算：100ms 理解 + 500ms 并行检索 + 150ms 重排 + 300ms 首 token + 1.5s 生成。三个关键决策：部分证据就先开生成、三层缓存（答案/检索/前缀 KV）、引用编号在检索时固定。必须显式支持"证据不足"这种回答类型。

---

### 9. Design the retrieval pipeline pulling from 100B web pages with sub-second latency and freshness guarantees.

**Anchor answer**

**The premise to fix first:** you never search 100B pages per query. You search an **inverted index + ANN index** that is sharded, and you accept that the tail is served from a cheaper tier.

**Tiering by temperature:**

| Tier | Contents | Storage | Update |
| :--- | :--- | :--- | :--- |
| Hot | last 7–30 days, high-authority domains | in-memory / NVMe, replicated | continuous (minutes) |
| Warm | 30 days – 2 years | SSD shards | hourly/daily batches |
| Cold | archive | object store, on-demand index | none (rebuilt on query intent) |

**Query path:** parse/rewrite → select tiers by intent (news intent hits hot first, always) → fan out to N shards × {BM25, dense ANN} → merge top-k per shard by score → global fusion (RRF) → rerank.

**Latency engineering:** shard fan-out is the whole game. 100 shards × 5 ms in parallel = fine; serialized = 500 ms. Use hedged requests (send to a replica after p95, take the first answer), cache shard results for head queries, and cap `k` per shard — retrieving 1000 from each of 100 shards is 100k candidates you cannot rerank.

**Freshness is an SLO, not a vibe.** Define it as: "for news-intent queries, ≥90% of results in the top 10 are < 60 minutes old." Then build backwards: change-rate-based recrawl scheduling (news sitemaps, RSS, publisher push), priority crawl queues, and a separate fast path that indexes a document within minutes of fetch. Measure the p99 *ingest lag* (fetch → searchable) as a first-class metric; most "we're not fresh" incidents are a queue backlog, not a crawl coverage problem.

**中文要点**：别想着一次搜 100B 页——按"热度"分层（热 7~30 天 / 温 / 冷），查询按意图选层，分片并行扇出 + RRF 融合。延迟靠并行与对冲请求、每片限 k；新鲜度要写成 SLO（news 意图 top10 里 90% < 60 分钟），并监控"抓取→可检索"的 p99 延迟。

---

### 10. Design the ranking system combining BM25, dense retrieval and LLM reranking across multiple indexes.

**Source**: https://outcomeschool.com/blog/how-does-hybrid-search-work ; https://outcomeschool.com/blog/how-does-a-reranker-work

**Anchor answer**

**Cascade with per-stage jobs — not three scores averaged together.**

| Stage | Candidates in/out | What it is for |
| :--- | :--- | :--- |
| Query understanding | 1 → 1 + n rewrites | intent class, entity extraction, filters (site:, date:) |
| Recall, per index | → 200 BM25 + 200 dense per vertical | recall only; never rank here |
| Fusion | 400 → 100 | RRF (rank-based, no calibration needed) or a trained blend |
| Light LTR | 100 → 50 | GBDT over features: bm25, cosine, recency, authority, click prior, source vertical |
| Cross-encoder | 50 → 20 | the real quality jump |
| LLM listwise | 20 → 8 | only for ambiguous/comparative queries |
| Diversity | 8 → 8 | MMR + per-domain cap so one site cannot fill the answer |

**The hard part is multi-index score calibration.** A BM25 score from a news index and a cosine from a docs index are not comparable. Two workable answers: (1) **rank-based fusion (RRF)** — `Σ 1/(k + rank_i)`, immune to scale, hard to beat, no training data needed; (2) **per-index score normalization** (min-max over the candidate set, or a learned calibration per source) before blending, which wins only when you have click data to train it.

**Vertical routing:** don't merge everything. Classify intent first — navigational goes to a title/URL index, "how do I" goes to docs/forum, breaking news goes to the hot index. Merging all verticals for every query is how you get a Reddit thread ranking above the official docs.

**Evaluation:** offline nDCG@10 on labelled queries, but ship on **interleaving** — it detects a 1–2% change with ~100× fewer queries than A/B on CTR, and it is robust to seasonality. Log every stage's candidates; a ranking bug is invisible if you only keep the final list.

**中文要点**：级联而不是加权平均：召回各 200 → RRF 融合到 100 → 轻量 LTR 到 50 → 交叉编码器 20 → 必要时 LLM listwise 8 → 多样性收尾。多索引的关键是分数不可比，用 RRF 或按源校准；上线用 interleaving 而不是直接 A/B。

---

### 11. Design Comet's hybrid browser architecture combining on-device privacy with cloud AI assistance.

**Anchor answer**

**The architecture is a data-classification problem wearing a browser costume.** Everything follows from one question: *what is allowed to leave the device, and who decides?*

| Layer | Runs where | Contents |
| :--- | :--- | :--- |
| Extraction | device | DOM → main content, a11y tree, tab state, form fields |
| Local intelligence | device | small model (classify, redact), embedding index over history/tabs, PII detector |
| Policy engine | device | per-site, per-user, per-org rules; the *only* component allowed to emit a cloud request |
| Cloud | server | frontier model, web index, long-running agents/tools, account sync |
| Sync | server | encrypted blobs only; keys never leave the device |

**The boundary contract:** the cloud never receives a raw page by default — only (a) a redacted extract, (b) an embedding/query, or (c) a task instruction with the device executing the returned actions. Every request shows the user *exactly* what will be sent, and the policy engine enforces it (not the prompt).

**Non-obvious constraints:**

- **Redaction breaks tasks.** "Summarize this page" survives redaction; "fill this form with my insurance number" does not. So you need a per-task policy class and a local fallback good enough to avoid a dead end.
- **Cross-tab context is the product.** A local index makes "what was that thing I read yesterday" work without uploading history; that is the real differentiator, and it costs only a small on-device index.
- **Enterprise DLP is the buying trigger.** Per-org policy, audit log of what was sent, and the ability to force local-only for a domain. Ship it as a first-class feature, not a settings toggle.
- **Latency:** local model calls must be < 300 ms or users will feel the seam; keep the local model small (1–3B, quantized) and reserve it for classification/redaction, not generation.

**中文要点**：本质是"什么数据可以出设备、由谁决定"。三层：设备端抽取与本地小模型、设备端策略引擎（唯一能发云请求的组件）、云端大模型。云端默认只收脱敏抽取或 embedding；跨标签页的本地索引才是真正的隐私差异点；企业 DLP 是买单理由。

---

### 12. How does an answer engine handle breaking news: a query about something that happened 20 minutes ago?

**Anchor answer**

**Detect it before retrieving.** A freshness classifier on the query (news keywords, "today/just now", an entity + event verb) plus a real-time signal layer (trending spikes, publisher push, social volume). High freshness confidence takes a *different path* — not the same pipeline with a recency boost.

**Different path:**

| Element | Design |
| :--- | :--- |
| Sources | minute-level news index, publisher sitemaps/RSS/push, wires, official accounts; social only as a *signal* |
| Retrieval | hard time filter (e.g. last 6 h), then rank by source authority + corroboration count, not by BM25 alone |
| Synthesis | lead with what is confirmed, mark what is single-source, show timestamps, use "developing" framing |
| Freshness UI | "as of 14:32 UTC, 4 sources" — the timestamp is part of the answer, not chrome |
| Degradation | if the index has nothing < 1 h old, say so explicitly instead of answering from parametric memory |

**The failure modes, in order of frequency:**

1. **Ingest lag.** Crawl-to-searchable is 20 minutes on a good day. If someone else broke the story 8 minutes ago, you genuinely cannot cite it — the honest answer names the earliest coverage you have.
2. **Parametric contamination.** The model's training cutoff makes it want to answer with the *old* state of the world ("X is the CEO"). Mitigation: when a freshness query is detected, inject "the following is current as of <timestamp>; do not rely on prior knowledge about this entity" and prefer extractive phrasing.
3. **Single-source rumor.** Corroboration counting across *independent* domains (not syndicated copies — dedupe first, or 50 reprints of one wire story look like consensus).
4. **Correction churn.** Early numbers are wrong. Cache news answers with a minutes-long TTL and re-verify the leading claim before serving.

**中文要点**：先判断"是不是时效性查询"，再走独立链路：分钟级新闻索引 + 硬时间过滤 + 按权威与独立信源印证排序 + 明示时间戳与"仍在发展中"。三大坑：抓取到可检索的延迟、模型用旧知识回答新事实、把同一通讯社的转载当成多方印证。

---

### 13. How would you evaluate answer quality for an answer engine, continuously and at scale, with both automated and human signals?

**Source**: https://outcomeschool.com/blog/llm-evaluation ; https://outcomeschool.com/blog/llm-as-a-judge

**Anchor answer**

**Three layers, each answering a different question:**

| Layer | Signals | Cadence | Catches |
| :--- | :--- | :--- | :--- |
| Implicit / online | citation CTR, dwell, follow-up (did they re-ask?), reformulation rate, thumbs, "new search" rate | real-time | satisfaction, but confounded by UI and by the answer being *plausible* |
| Automated offline | per-claim citation support (NLI/LLM judge), answer correctness vs golden answers, completeness, freshness of cited sources, format compliance | every deploy + nightly | faithfulness, regressions, long-tail |
| Human graded | 1–5 rubric on a stratified sample, pairwise vs. previous version, side-by-side vs. search | weekly, 300–1000 items | taste, tone, "actually answers the question" |

**Judge design (this is where most teams quietly fail):**

- **Binary per-claim verdicts beat 1–10 scales.** "Is this claim supported by this passage: yes/no/partial" is reproducible; "rate helpfulness 1–10" drifts by ±1.5 between judge versions.
- **Calibrate against humans and keep measuring the calibration.** Maintain 500 items with human labels; if judge-human agreement drops, the judge changed (providers update silently) — pin the judge version.
- **Swap positions and randomize** for pairwise judging; LLM judges have measurable position bias and prefer longer answers. Control length explicitly or you will optimize for verbosity.
- **A frozen canary set + a rotated fresh set.** The frozen set detects regressions; the fresh set (real queries from last week) detects "we got better at the benchmark, not at the product."

**The one metric I would put on the wall:** *unsupported-claim rate at fixed answer length.* It cannot be gamed by writing more, and it is the metric users actually experience as trust.

**中文要点**：三层——隐式在线信号（引用点击、追问率、改写率）、自动离线（逐 claim 的引用支撑度、正确性、时效）、人工评分（分层抽样 + 成对比较）。judge 要用二值判定而非 1~10 分，固定版本并持续测 judge-人一致率，配对时随机换位控长度偏置。

---

### 14. Design the citation-verification system to reduce hallucinations in generated answers. How do you ensure every claim is actually supported by its cited source?

**Anchor answer**

**Assume the generator will lie, and build three independent gates.**

| Gate | Mechanism | Guarantees |
| :--- | :--- | :--- |
| **Before** (constrain) | only in-context passages get ids `[1..n]`; output schema is `{text, citations:[ids]}`; decoding restricted to valid ids; "if no passage supports this, write it without a citation" | no fabricated URLs, no citing an unretrieved passage |
| **After** (verify) | split the answer into **atomic claims**; for each claim, run an entailment check (NLI model or LLM judge) against *only the cited passage* → supported / partial / unsupported | catches misattribution — the right link for the wrong sentence |
| **Repair** | unsupported → drop the sentence; partial → soften ("some reports suggest"); repeated failure on a sub-question → re-retrieve and regenerate that segment | the shipped answer contains only supported claims |

**Metrics that make it an SLO, not a demo:** citation **precision** (fraction of cited claims whose passage entails them), citation **recall** (fraction of factual claims that carry support), and unsupported-claim rate per 100 answers. Track them per query class, excluding no-citation answers (definitions, arithmetic) from the denominator.

**Hard parts, honestly:**

1. **Atomic claim splitting is the bottleneck.** "Revenue grew 12% to $4.1B in Q3" is two claims with different support. A bad splitter either merges them (hiding one unsupported fact) or fragments a single claim into unprovable pieces.
2. **"Partial support" is the real distribution.** Most mis-citations are *near* correct — right document, wrong year or wrong entity. Binary entailment hides this; use a three-way label and report the partial rate separately.
3. **Verify the link resolves and contains the passage.** A correct citation to a dead URL is still a broken answer; check `URL → passage hash` at render time and re-retrieve if it 404s.
4. **Latency.** Per-claim NLI over ~15 claims is 100–300 ms batched — affordable. Per-claim LLM judging is not: small NLI in the request path, LLM judge offline.

**中文要点**：三道闸：生成前只允许引用上下文里的 id（结构化输出 + 约束解码）；生成后把答案拆成原子 claim、只对着"被引段落"做蕴含判定（支持/部分/不支持）；不支持就删句或降调，反复失败就重检。指标用引用精确率、召回率和每百答的不支持率，注意 claim 切分与"部分支持"才是难点。

---

### 15. What makes a Perplexity answer great vs mediocre? Where does Perplexity lose to traditional search, and where does it win? You clearly use it: what's broken, and what would you ship to fix it?

**Anchor answer**

**A framework first, then an opinion** — a taste question is scored on whether you have a *model* of quality.

| Dimension | Great | Mediocre |
| :--- | :--- | :--- |
| Lead | answers the question in sentence one, no restatement | "That's a great question…" then context |
| Length | proportional to the question's real complexity | padded to look thorough |
| Sources | primary (docs, filings, papers), recent, and each actually supports its sentence | SEO aggregators, dead links, citation that points at a domain not a passage |
| Ambiguity | states the assumption it resolved ("assuming US pricing") | silently picks one interpretation |
| Comparison | table when the question is comparative | three paragraphs of "on the other hand" |
| Next step | anticipates the follow-up | ends flat |

**Where it loses to search:** navigational intent ("github login"), transactional/booking, local and real-time (opening hours, stock), exhaustive enumeration ("every X under $Y"), and any high-stakes query where the user wants to *read the source themselves* rather than trust a synthesis. Search wins on verifiability and on queries where the answer is a *place to go*, not a sentence.

**Where it wins:** synthesis across 5–10 sources, comparison, explanation, and multi-hop questions where the work is reading, not clicking.

**What I'd ship** (pick one, make it falsifiable): *sentence-level source confidence*. Today citations are per-answer; the failure users feel is one shaky sentence inside a good answer. Show a per-claim support indicator, and when support is weak, offer the alternative reading inline. Success metric: reduction in immediate reformulation rate on the affected query class, without raising answer length.

**中文要点**：好的答案第一句就给结论、长度匹配问题复杂度、引用一手且逐句可支撑、歧义处明说假设、比较题用表格。它输给搜索的地方是导航/交易/本地实时/穷举类查询；赢在多源综合与多跳解释。可落地的改进：把引用粒度做到"逐句支撑度"。

---

## 附：这 15 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 检索/排序真功夫 | 6（重排架构）、7（50 选 10）、9（百亿页检索）、10（多索引融合） |
| 端到端系统设计 | 8（3 秒答案引擎）、11（端云混合）、12（突发新闻） |
| 可信度与评测 | 13（持续评测）、14（引用校验）、15（产品品味） |
| 工程实现 | 1（多供应商 failover）、2（近重复）、3（批处理）、4（beam search）、5（TTFT 回归） |
