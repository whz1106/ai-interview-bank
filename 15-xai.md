# 15 · xAI

> 10 题。出处：`README.en.md` § xAI。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Build an in-memory key-value store with SET/GET/DELETE, then add transactions with BEGIN/COMMIT/ROLLBACK, including nested transactions.

**Anchor answer**

The whole problem is one decision: **undo logging per frame, not redo logging**. Each `BEGIN` pushes a log; every write records the *first* prior value of that key in the **innermost** frame; `ROLLBACK` replays only that frame in any order; `COMMIT` merges the frame into its parent with `setdefault` so the parent keeps its older undo entry.

**Reference implementation**

```python
class KVStore:
    def __init__(self):
        self._data, self._txns = {}, []

    def get(self, k):
        return self._data.get(k)

    def set(self, k, v):
        self._record(k); self._data[k] = v

    def delete(self, k):
        self._record(k); self._data.pop(k, None)

    def _record(self, k):
        if self._txns:
            log = self._txns[-1]
            if k not in log:                      # first touch in THIS frame wins
                log[k] = (k in self._data, self._data.get(k))

    def begin(self):
        self._txns.append({})

    def commit(self):
        if not self._txns:
            raise RuntimeError("no active transaction")
        log = self._txns.pop()
        if self._txns:                            # merge into parent, parent's entry wins
            parent = self._txns[-1]
            for k, prev in log.items():
                parent.setdefault(k, prev)

    def rollback(self):
        if not self._txns:
            raise RuntimeError("no active transaction")
        for k, (existed, prev) in self._txns.pop().items():
            if existed: self._data[k] = prev
            else: self._data.pop(k, None)
```

**Key points**

- `_record` must check `if k not in log`: recording the *current* value on every write makes an outer rollback restore an intermediate value — the classic bug.
- `commit` uses `setdefault`, not `update`; the parent's older snapshot is the only correct rollback target.
- Cost: O(1) amortized per op, O(distinct keys touched) per commit/rollback. Memory is bounded by live write set, not store size.
- Empty `begin`/`commit` with **no** frame should raise, not silently no-op — otherwise unbalanced code passes tests.
- Next step if asked: durable store → same log becomes a WAL with sequence numbers, and commit = fsync appends.

**中文要点**：嵌套事务的关键是每帧一张 undo log，只记录某键在该帧的**首次**旧值；commit 用 setdefault 合并入父帧，rollback 只回放当前帧。

---

### 2. Write an iterator class that lazily flattens an arbitrarily nested list of lists/integers: no generators, explicit state.

**Anchor answer**

"No generators" means the state must be **data**, not a suspended stack frame — so the iterator can be pickled, resumed across a checkpoint, or advanced by another thread. That rules out recursion and `yield`; it forces an explicit stack of `(container, next_index)`.

**Reference implementation**

```python
class FlatIter:
    def __init__(self, data):
        self._stack = [(data, 0)]          # (container, cursor)

    def __iter__(self):
        return self

    def __next__(self):
        while self._stack:
            seq, i = self._stack[-1]
            if i >= len(seq):              # exhausted -> pop, loop back to parent
                self._stack.pop()
                continue
            self._stack[-1] = (seq, i + 1)  # advance parent BEFORE descending
            item = seq[i]
            if isinstance(item, list):
                self._stack.append((item, 0))
            else:
                return item
        raise StopIteration
```

**Key points**

- Advance the parent cursor *before* pushing the child, otherwise the child list is read twice.
- `while` (not `if`) around the loop body is what makes empty nested lists — `[[], [], [1]]` — correct; the naive `if` version returns `[]` as an item or deadlocks.
- Memory is O(nesting depth), time O(1) amortized per element, and the input is never copied.
- Failure mode to mention: an iterator that raises `StopIteration` inside `__next__`'s descent and swallows it, silently truncating output.
- Extension if asked: `isinstance(item, list)` → accept any `Sequence`, plus a `depth_limit` to defend against pathological input.

**中文要点**：禁用生成器 = 状态必须是数据而不是栈帧；核心是显式 `(容器, 游标)` 栈，弹出前先推进父游标，且循环体用 `while` 才能处理空子列表。

---

### 3. Here is a scheduler class from a small LLM inference engine. One method, \_admit_requests, is a stub: no spec, no docstring, no tests. Walk me through your first thirty minutes.

**Anchor answer**

I treat it as **reverse-engineering a contract, not writing code**. The stub is a signal the grader wants to see probe order and whether I invent requirements silently.

| Minutes | Action | Output |
| :--- | :--- | :--- |
| 0–6 | Read the whole class, not the stub: fields, lock, callers of `_admit_requests`, its caller's loop | Who calls it, how often, blocking or polling |
| 6–12 | Grep for every read/write of the shared state it must respect (free blocks, token budget, max batch, queue order, per-tenant quota, deadline) | A list of invariants other methods enforce |
| 12–18 | Write the contract in 5 lines: signature semantics, return value, side effects on queue and counters, what "admit" implies for later `_schedule` | A docstring I can defend |
| 18–25 | Write 3 failing tests: empty queue, queue longer than max batch, no free KV blocks | Red tests, no code yet |
| 25–30 | Minimum implementation that passes those three; add counters; state the two questions I'd ask the owner | Green + explicit open questions |

What I explicitly refuse to do: guess an undocumented priority policy, take a lock the class doesn't own, or mutate queue order "because it looks sorted". Scheduler bugs are silent — they show up as 3% lower goodput two weeks later, not as a crash, so the invariant list is the real deliverable.

If allowed a fourth question: what is the back-pressure contract when nothing can be admitted — return empty, block, or shed?

**中文要点**：读未实现的桩函数要"逆向契约"：先看调用方与共享状态的不变量，写 5 行规格 + 3 个失败测试，再写最小实现；不许自己编调度策略或乱加锁。

---

### 4. Estimate the KV-cache memory to serve a 70B-class model at 128k context. What do you do when it doesn't fit?

**Source**: https://outcomeschool.com/blog/kv-cache-in-llms · https://outcomeschool.com/blog/kv-cache-compression

**Anchor answer**

Formula: `bytes/token = 2 (K and V) × layers × kv_heads × head_dim × dtype_bytes`. The head count that matters is **KV heads, not query heads** — a 70B model with GQA (8 KV heads, head_dim 128, 80 layers, bf16):

`2 × 80 × 8 × 128 × 2 = 327,680 B ≈ 320 KiB per token`
`× 131,072 tokens ≈ 40 GiB per 128k sequence`

So a single full-length conversation already exceeds one 80 GB card's usable memory after weights (~140 GB at bf16, so weights are already tensor-parallel across ≥2 cards). At 8 concurrent 128k sessions you are at ~320 GiB of cache — the concurrency collapse is memory, never FLOPs.

| Lever | Effect | Cost |
| :--- | :--- | :--- |
| More KV heads shared (MQA/GQA/MLA) | 4–8× smaller | quality, needs retraining |
| Paged blocks + prefix sharing | removes fragmentation, shares system prompt | allocator complexity |
| FP8 / INT4 KV | 2–4× smaller | accuracy loss grows with context |
| Sliding window / hybrid attention | constant cache | no exact far recall |
| Cap concurrency × context (admission control) | linear | product limits |
| Offload cache to host/CPU | fits | PCIe latency per step |

**中文要点**：KV cache 按 KV 头（不是 Q 头）算：320 KiB/token × 128k ≈ 40 GiB/条；装不下时按"GQA/MLA → paged+共享前缀 → FP8 KV → 滑窗 → 限流"顺序压。

---

### 5. Design a rate limiter for an LLM API where cost scales with tokens, not requests.

**Anchor answer**

Requests are the wrong unit: one 200k-token prefill costs ~1000× a 200-token chat. The unit must be **tokens**, and because output length is unknown at admission, you need **reserve-then-reconcile**.

1. **Cost model.** `cost = input_tokens + expected_output_tokens` (percentile of the tenant's history, or their `max_tokens`). Charge prefill and decode at different weights if the fleet is split.
2. **Admission.** Per-tenant token bucket (`rate`, `burst`) + a global bucket. Two buckets, both must pass — one alone lets a single tenant saturate the fleet.
3. **Reserve.** Debit `input + max_tokens` up front. Refund `max_tokens − actual` on completion.
4. **In-flight accounting.** Reservations are *not* in the bucket; they live in a "held" counter with a TTL, or a crashed worker leaks quota forever.

```python
def admit(key, tokens, now):
    b = buckets[key]
    b.refill(now)                          # tokens += rate * dt, cap at burst
    if b.tokens < tokens or global_.tokens < tokens:
        return False, retry_after(b)
    b.tokens -= tokens; global_.tokens -= tokens
    return True, 0
```

**Where it breaks.** Local per-worker buckets drift; you must push deltas to a shared counter (Redis Lua, atomic) or hand each node a *lease* of quota it may spend without asking. Leases cut round-trips but overshoot by ≤ lease size — acceptable if the lease is 1–5% of the tenant's rate. Streaming makes reconciliation mandatory: tokens are spent incrementally, so charge as you emit and cut the stream at the cap rather than detecting overrun after the fact.

**中文要点**：按 token 计费就要"预扣 + 结算"：预付 input+max_tokens，结束时退差额；租户桶 + 全局桶双闸门，分布式用 Redis 原子脚本或按节点发配额租约。

---

### 6. You're training on tens of thousands of GPUs and hardware fails constantly. How do you keep goodput high?

**Anchor answer**

At that scale the question is not MTBF, it is **MTTR**. A 10k-GPU job with a component failure every few hours and a 20-minute restart loses most of its wall clock unless restart is cheap. Optimize in this order:

| Layer | Mechanism | Why it dominates |
| :--- | :--- | :--- |
| Detection | NCCL watchdog + heartbeat, per-step loss/grad-norm diff across ranks | Fast abort beats fast repair; silent corruption is worse than a crash |
| Restart cost | Async sharded checkpoint to local NVMe, then background upload; 1–2 min cadence | Checkpoint interval, not failure rate, sets lost work |
| Topology | Elastic re-shard on hot spare, or over-provision ~2–5% nodes | Avoid re-queueing the whole job for one dead node |
| Correctness | Loss-scale overflow guard, NaN traps, periodic checksum of params, deterministic data order | Prevents "trained 3 days on corrupt weights" |
| Data | Pre-tokenized, pre-sharded, streaming from object store with retries | Data loader stalls look like hardware failures |

Two counter-intuitive points: **more frequent checkpointing is usually the cheapest win**, and **a straggler is more expensive than a dead node** — a dead node fails fast, a 30%-slow node drags every collective. Detect stragglers with per-rank step-time percentiles and evict.

Also: rehearse failure injection on a small slice. If a killed rank doesn't restart cleanly in minutes, that's the bug to fix, not the GPU vendor's.

**中文要点**：万卡集群优化的是 MTTR 而非 MTBF：快速检测(看门狗/跨 rank loss 比对) + 分钟级异步分片 ckpt + 弹性重分片热备；掉队节点比坏节点更贵。

---

### 7. Loss spikes mid-run on a large pretraining job. Walk me through your debugging process.

**Anchor answer**

Order matters: cheap and reversible first, expensive last. The goal in the first 10 minutes is **stop the bleeding without killing the run**.

1. **Freeze the signal.** Is it one rank or all? Per-rank loss, grad norm, update norm, and router/aux losses on a dashboard. All ranks → real; one rank → hardware or data shard.
2. **Roll back, then read.** Restore from the checkpoint before the spike, reproduce with the *same* data order (this is why deterministic data order exists). If it reproduces, it's data or math; if not, it's hardware/async state.
3. **Skip the batch test.** Replay the suspect shard alone. A single bad document (encoding garbage, repeated token, mislabeled length) can spike cross-entropy to NaN-adjacent values.
4. **Numerics.** fp8/bf16 overflow in attention logits or the router softmax; check max logit magnitude, loss-scale history, whether z-loss is enabled. MoE routers spike when experts collapse.
5. **Optimizer state.** LR × warmup interaction, Adam second-moment blowup after a long stable stretch, weight decay on embeddings.
6. **Hardware.** Silent corruption shows as loss *plateau* then spike; run a quick compute check on suspect nodes and compare param checksums across ranks.

Typical fix: rewind ~500 steps, skip the offending shard, keep LR unchanged. Changing three things at once means you learn nothing.

**中文要点**：先判断是单 rank 还是全 rank；能在 spike 前 ckpt 复现 → 数据/数值问题，不能 → 硬件/异步状态；顺序是数据→数值(fp8/router)→优化器→硬件。

---

### 8. Design a deduplication pipeline for a web-scale pretraining corpus. It has to run as a streaming process.

**Anchor answer**

Two properties to hold at once: **bounded memory** and **stable decisions** (a doc's fate must not depend on arrival order, or reprocessing gives a different corpus).

Pipeline:

1. **Normalize** (Unicode, whitespace, boilerplate strip) — dedup on raw bytes is useless, the same page has three headers.
2. **Exact pass.** 128-bit hash of normalized text in a sharded set. Streaming version: hash → shard → LRU-on-disk key set, or a Bloom filter per shard with a false-positive budget you can state.
3. **Near-dup pass.** MinHash (k≈128 permutations) + LSH banding (b bands of r rows), compare only within a bucket. Cost is `O(n·k)`, dominated by shingling.
4. **Long-tail.** Suffix-array or SimHash for the sub-1% pairs that LSH misses; usually offline, not streaming.
5. **Emit** the doc with a canonical-id pointer, not a delete — you want the cluster to be reproducible and the ratio auditable.

| Streaming hazard | Mitigation |
| :--- | :--- |
| Unbounded index growth | Shard by hash; per-shard index with TTL/flush; report memory per 1M docs |
| Order dependence | Keep a stable "first-seen by doc-id" winner rule, not "first in stream" |
| False positives from boilerplate | Strip nav/footer before shingling, else every page matches |
| Shingle size | 5-gram words for prose; 1-gram hashing for code |

Track dedup ratio per source and cluster size distribution — a spike in giant clusters means boilerplate leaked through, not that the web got repetitive.

**中文要点**：流式去重 = 归一化 → 精确 hash → MinHash+LSH 近似 → 长尾 SimHash；必须按 hash 分片限制内存，且"谁保留"用 doc-id 稳定规则而非流顺序。

---

### 9. Design the serving stack for a consumer chatbot with real-time search over a social-media firehose.

**Anchor answer**

Constraints: interactive p95 latency (first token < ~500 ms), traffic with huge spikes, retrieval freshness measured in seconds, and a firehose that is hostile (spam, duplicates, adversarial text).

| Layer | Design | Key decision |
| :--- | :--- | :--- |
| Ingest | Firehose → queue → extract/embed workers, per-post dedup | Embed at ingest once; never re-embed per query |
| Fresh index | Hot tier: in-memory/ANN-on-new-segments, 1–5 min visibility; warm tier: compacted hourly | Freshness vs index-build cost |
| Query understanding | Query rewrite + entity/time extraction; decide if search is needed at all | Most turns don't need retrieval — a router saves the most latency |
| Retrieval | Hybrid BM25 + dense, RRF fuse, recency boost, engagement prior | Social content is duplicate-heavy: cluster before ranking |
| Rank | Small cross-encoder on top-50, cut to 5–8 | Rerank budget is the latency knob |
| Generation | Prefill/decode disaggregation, prefix cache for system prompt, streaming | Cache the system prompt across all users |
| Safety | Injection filter on retrieved text (it is untrusted user content) | Treat web text as data, never instructions |

Failure modes to name: retrieved posts carry prompt injection; trending spikes make one entity dominate results (needs diversity/MMR); deleted posts must age out of the hot index or the bot cites ghosts; and a cold cache after deploy causes a prefill thundering herd — warm it before shifting traffic.

**中文要点**：先判断"要不要检索"再谈召回；摄取时嵌入一次 + 热/温双层索引保证秒级新鲜；社交文本是不可信输入，必须防注入 + 多样性去重。

---

### 10. You have four hours to build and demo a working AI-powered product. How do you spend them?

**Anchor answer**

The one rule: **demo-ability beats completeness**. Pick a product whose core loop fits in one screen and one model call, then spend the last 45 minutes on the demo, not the code.

| Time | Block | Deliverable |
| :--- | :--- | :--- |
| 0:00–0:15 | Choose and cut | One sentence: input → visible output. Write down the 3 things you will *not* build |
| 0:15–0:45 | Skeleton | Server + streaming endpoint + hardcoded UI shell, deployed to a URL. Deploy first, feature second |
| 0:45–2:00 | Core loop | The one prompt/tool chain that produces the magic; cache 5 canned inputs so the demo never depends on network luck |
| 2:00–2:40 | The "wow" detail | The thing that makes it memorable: live streaming, citations, an editable intermediate |
| 2:40–3:15 | Failure handling | Timeouts, a graceful "can't do that" path, input length guard. Demos die on unhandled edge cases |
| 3:15–3:45 | Demo script | 3 scripted scenarios, ordered easy → impressive; rehearse once, timed |
| 3:45–4:00 | Buffer | Reserve it. Never plan to the last second |

What I deliberately skip: auth, persistence, tests, multi-model abstraction, fine-tuning. What I do not skip: a visible URL, deterministic demo data, and one honest sentence about what is mocked. Interviewers weight shipped and demoed far above architected.

**中文要点**：四小时的唯一目标是"可演示"：先部署空壳再填核心闭环，预留 45 分钟给失败兜底和三次彩排；明确放弃鉴权/持久化/测试。

---

## 附：这 10 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 工程基本功（数据结构） | 1（事务 KV）、2（显式状态迭代器） |
| 代码库阅读与规格推断 | 3（桩函数 30 分钟） |
| 推理集群容量直觉 | 4（KV cache 估算）、5（token 限流） |
| 大规模训练运维 | 6（goodput）、7（loss spike）、8（流式去重） |
| 端到端产品与系统设计 | 9（实时搜索服务栈）、10（四小时交付） |
