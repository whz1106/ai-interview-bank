# 35 · Databricks

> 9 题。出处：`README.en.md` § Databricks。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Implement a thread-safe batching logger: many producer threads call log(msg); a background thread flushes batches of up to 100 messages every second or when full.

**Reference implementation**

```python
import threading, time
from collections import deque

class BatchLogger:
    def __init__(self, sink, max_batch=100, interval=1.0, max_buffer=10_000):
        self._sink, self._max, self._interval = sink, max_batch, interval
        self._max_buffer = max_buffer
        self._buf = deque()
        self._cv = threading.Condition()
        self._closed = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def log(self, msg):
        with self._cv:
            if len(self._buf) >= self._max_buffer:      # backpressure: block, don't grow
                self._cv.wait_for(lambda: len(self._buf) < self._max_buffer or self._closed)
            self._buf.append(msg)
            if len(self._buf) >= self._max:
                self._cv.notify()                      # size trigger

    def _run(self):
        while True:
            batch = self._take()
            if batch is None:
                return
            try:
                self._sink(batch)                      # sink runs OUTSIDE the lock
            except Exception:
                with self._cv:                          # don't lose data on transient sink error
                    self._buf.extendleft(reversed(batch))
                time.sleep(min(self._interval, 5))

    def _take(self):
        with self._cv:
            while not self._buf and not self._closed:
                self._cv.wait(self._interval)
            if not self._buf:
                return None
            out = []
            while self._buf and len(out) < self._max:
                out.append(self._buf.popleft())
            if self._buf:
                self._cv.notify()                      # wake blocked producers
            return out

    def close(self, timeout=10):
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._t.join(timeout)
```

**Key points**

- `Condition` not `Lock`: `log()` must block *and* the flusher must be woken instantly on "batch full".
- Never call the sink while holding the lock — a slow sink otherwise serialises every producer.
- Timer semantics: wait starts when the buffer goes non-empty, so a lone message flushes after ~interval, not up to 2×.
- `close()` must drain, then quit; otherwise the last <100 messages vanish on shutdown.
- Bounded buffer is the real thread-safety issue in production; unbounded `deque` turns a slow sink into an OOM.

**中文要点**：锁内只碰队列、sink 在锁外调用；满 100 或等满 1 秒就刷；必须有关闭排空和**有界缓冲**，否则慢 sink 直接 OOM。

---

### 2. You have a stream of billions of events and need the top-K most frequent keys with bounded memory. Exact is impossible: what do you do?

Say it plainly: exact top-K over a stream needs Ω(distinct keys) memory, which is exactly what "billions" forbids. So pick a **one-pass sketch** and accept bounded error.

| Method | Memory | Guarantee | Use when |
| :--- | :--- | :--- | :--- |
| Misra-Gries / **Space-Saving** | O(K) counters | candidates are true heavy hitters; estimated counts can be overestimates | default choice for top-K |
| Count-Min Sketch | O(1/ε · log 1/δ) | ε·N overestimate per query, no false negatives | need point queries too |
| **CMS + Space-Saving hybrid** | O(K + sketch) | CMS filters the "one-shot" noise before it enters the heavy-hitter table | skewed power-law traffic |
| Count exact per shard, merge | O(K) per shard | exact if shards partition keys | parallel ingestion |

Practical recipe: shard by `hash(key) % P`, keep a Space-Saving table of size ~4K per shard, merge shard tables at query time (top-K of a union needs only the union's top-K tables).

Two things that bite:

1. **Error is relative to N, not to the item count** — with N=10¹⁰ and ε=10⁻³ you're off by 10⁷ absolute. Report confidence, or do a **second exact pass over just the candidate set** (usually a few thousand keys) when the number must be auditable.
2. **Hash collisions create phantom items.** The classic failure is a heavy hitter appearing in the table with a count inflated by collisions, so ranking flips between the true #5 and a fake one — use 64-bit hashes and a decay/rotation window so yesterday's celebrity doesn't squat the table.

**中文要点**：精确 top-K 内存下界就是"不同 key 数"，只能上 sketch；Space-Saving/CMS 是标配，最终用候选集二次精确扫描把数字做成可审计的。

---

### 3. Given allowed IP ranges as CIDR blocks plus explicit deny ranges, implement is_allowed(ip) efficiently for millions of checks per second.

Convert every prefix to an integer span `[network, broadcast]`, **merge overlaps once at construction**, then each check is a binary search — O(log N) with tiny constants, no per-check allocation.

```python
import bisect, ipaddress

class Firewall:
    def __init__(self, allow, deny):
        self._allow = self._build(allow)
        self._deny  = self._build(deny)

    @staticmethod
    def _build(cidrs):
        spans = sorted(
            (int(n.network_address), int(n.broadcast_address))
            for n in (ipaddress.ip_network(c, strict=False) for c in cidrs)
        )
        merged = []
        for lo, hi in spans:
            if merged and lo <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        return merged, [lo for lo, _ in merged]

    @staticmethod
    def _hit(rule, ip_int):
        spans, starts = rule
        i = bisect.bisect_right(starts, ip_int) - 1
        return i >= 0 and ip_int <= spans[i][1]

    def is_allowed(self, ip):
        n = int(ipaddress.ip_address(ip))
        if self._hit(self._deny, n):      # deny wins, always
            return False
        return self._hit(self._allow, n)
```

**Design notes**

- **Deny-wins flattening** is correct *only* because deny is checked first; do not merge across the two lists.
- `strict=False` matters: customer lists contain `10.0.0.5/24`.
- Merge makes the span list disjoint, which is what makes binary search on starts valid — without merging, a `-1` hit can land in a gap.
- For IPv4 + IPv6 keep two tables and dispatch on the family; converting v4-mapped v6 (`::ffff:a.b.c.d`) inconsistently is a standard bypass.
- Millions/sec: 20 ns map + two `bisect` ≈ sub-µs; keep `int()` out of the hot path if callers can hand you integers.

**中文要点**：CIDR 先转整数区间并合并成不相交集合，查询退化为二分；deny 先判且优先；注意 `strict=False`、v4/v6 分表、v4-mapped 地址别被绕过。

---

### 4. A Spark job joining a 2 TB fact table to a 50 GB dimension table has one straggler task running 100x longer than the rest. Diagnose and fix it.

**"100x on one task" is almost always one partition receiving one giant key**, i.e. key skew — not a big table. Confirm before touching config: open the stage in the Spark UI, compare the **max task's input/records against the 75th percentile**, and check `spark.sql.adaptive.skewJoin.enabled`.

| Check | What it tells you |
| :--- | :--- |
| Max vs p75 shuffle-read bytes | skew, not data volume |
| GC time / spill on the slow task | one key's hash bucket doesn't fit memory |
| Key distribution (`df.groupBy(k).count().orderBy(desc)`) | the offender: `null`, a sentinel id, a "default" customer |
| Broadcasting the 50 GB side? | that's not a fix, it's an OOM — 50 GB × executors |
| `AQE` on? | if off, that alone may explain it |

Fixes, in order of preference:

1. **AQE skew join** (`skewJoin.enabled=true`, `skewedPartitionFactor≈5`, `skewedPartitionThresholdInBytes≈256MB`) — splits oversized partitions automatically. Cheapest fix, try it first.
2. **Salt the hot key** — append `rand(0, N-1)` to the key on the fact side and explode the dimension side by N, then sum. Works when the hot key is a legitimate huge entity.
3. **Isolate the hot keys**: handle the top-N keys in a separate small job and `union` the rest. Deterministic and easy to reason about.
4. **Fix the model** — if the skew comes from `null` join keys, filter them out before the shuffle. Nulls all hash to one partition; this is the single most common cause.

**中文要点**：单任务 100x 是**数据倾斜**不是数据量大；先看 max vs p75 的 shuffle 字节确认；修复优先级 AQE skew join → 加盐 → 热键单独跑 → 过滤 null key。

---

### 5. A Structured Streaming job reads Kafka and writes to a Delta table. The cluster is killed mid-batch and restarts. Does the customer get duplicate rows? Explain at the level of the checkpoint and the transaction log.

**No — provided the sink is Delta and you didn't wrap it in something non-idempotent.** This is the exactly-once contract, and it holds because **offsets and data land in the same transaction**.

The mechanics, layer by layer:

1. The **checkpoint** directory holds `offsets/` (the Kafka ranges for batch *n*), `commits/` (batches fully written) and `state/` for stateful ops. On restart, Spark lists `offsets/`, list `commits/`, and replays every batch present in offsets but missing in commits.
2. A batch writes **one Delta commit**: a single `_delta_log/<version>.json` containing `add` actions for the new parquet files **plus a `txn` action carrying `appId` + `batchId`**.
3. On replay of batch *n*, the Delta sink checks `SetTransaction(appId, n)`. If a transaction file with that version exists, **the batch is skipped** — the data is already there.

Where duplicates *do* appear, all of them real:

| Cause | Symptom |
| :--- | :--- |
| `foreachBatch` with a custom sink (JDBC upsert, HTTP call) | at-least-once; you must be idempotent yourself |
| Different `appId` after redeploy (new checkpoint dir) | replay from earliest → full duplicate |
| Writing to a non-transactional sink in parallel with Delta | commit not atomic |
| `startingOffsets` changed + fresh checkpoint | reprocess from scratch |

The interview-grade sentence: *exactly-once is not a property of Spark; it is a property of the source offset boundary, the checkpoint's batch identity, and the sink's atomic idempotent commit — remove any one and you're at-least-once.*

**中文要点**：Delta 精确一次靠"一个事务里同时写数据 + `SetTransaction(appId,batchId)`"，重启后按 checkpoint 重放并跳过已提交批次；只有 foreachBatch 自定义 sink、换 appId、换 checkpoint 才会重复。

---

### 6. When would you fine-tune instead of using RAG or prompt engineering, and if you do, LoRA or full fine-tuning?

**Source**: https://www.youtube.com/watch?v=lnfWvX66FUk · https://outcomeschool.com/blog/how-does-fine-tuning-work

The decision is about **what you're changing**, not about "how much data you have".

| The problem is | Use | Why |
| :--- | :--- | :--- |
| The model doesn't *know* a fact | RAG | facts change; retrieval is editable and auditable |
| Output *format*/schema is wrong | prompting + structured output | cheapest, instant rollback |
| Task needs a *skill*: a house style, a domain grammar, terse classification, SQL dialect quirks | LoRA / SFT | that's a distribution shift, not a lookup |
| Latency/cost per call too high for a narrow task | distill into a small model | 3B doing one task beats 70B doing everything |
| You need behaviour *removed* | preference optimisation (DPO) | SFT alone teaches what to say, not what to refuse |

LoRA by default; full fine-tune only when you must change the whole distribution (new language, new modality alignment) or you're doing continued pretraining. Practical thresholds that hold up:

- **< ~10k examples** → LoRA is enough and is the only sane option; full FT overfits.
- LoRA rank 8–32 on all linear layers matches full FT on most task-adaptation benchmarks at a fraction of the memory — the "rank 64 everywhere" habit mostly buys overfitting.
- **Merge-and-quantise** the adapter for serving; keeping the adapter hot adds a kernel and usually a few % latency.

The failure mode to name out loud: fine-tuning to inject facts. It works in eval, then silently rots — the fact is now baked into weights and every correction costs another training run. And a LoRA trained on 5k company docs will confidently invent facts it never saw; RAG at least says "no passage found".

**中文要点**：改**知识**用 RAG、改**格式**用 prompt、改**技能/风格**才微调；默认 LoRA（rank 8~32，全线性层），样本少于万级几乎不要全参；最忌讳用微调灌事实——写进权重后改一次就要重训。

---

### 7. Take a working GenAI agent prototype to production for an enterprise. What's your checklist between demo and launch?

A demo proves the happy path on 10 curated inputs. Production is four things the demo never touched: **identity, cost, failure, and evidence**.

| Gate | Concrete deliverable | The thing that actually gets skipped |
| :--- | :--- | :--- |
| **Eval set** | 100–300 real tickets with graded answers, frozen; per-run scoring in CI | a *held-out* set that no prompt iteration touches — otherwise you're tuning on the test |
| **Identity & permission** | per-user token propagation into every tool call; row-level security enforced in the data layer, not in the prompt | "the model won't ask for other tenants' data" is not a control |
| **Budgets** | per-request token cap, per-tenant daily cap, loop step cap, hard timeout | one runaway agent burning a month of quota overnight |
| **Failure behaviour** | explicit `refuse / escalate / retry` classification; fail-closed on tool error | the model improvising after a 500 |
| **Observability** | full trace per request (prompt, tools, args, results, latency, cost) with retention | you cannot debug what you only logged as a final string |
| **Data governance** | PII redaction before the model, no-train contract with the vendor, regional pinning | prompt contains customer email and goes to a US endpoint |
| **Rollback** | model + prompt + index versioned together, one-flag revert | index refreshed independently of prompt = irreproducible regressions |
| **HITL** | approval queue for consequential actions, approver ≠ requester | "auto-approve for speed" in week one |
| **Change control** | canary 5% → 50% with the eval set as the gate, not vibes | |

The one-line version for the interviewer: *the launch checklist is not a list of features, it's a list of the ways the system can be wrong and how fast you'd find out.*

**中文要点**：demo 只证明了 happy path；上线九道闸门里最常被跳过的是**冻结的评测集、数据层强制的行级权限、预算上限、可回滚的版本绑定**，以及"贯穿到工具调用的用户身份"。

---

### 8. A customer insists on fine-tuning an open model on their support tickets because “we want our own model.” You think RAG solves it. What do you do?

Don't fight the stated solution — **find the job to be done behind it**, then let a small experiment choose. "We want our own model" usually decodes to one of three different asks, and only one of them is a fine-tune.

| What they actually mean | Real fix |
| :--- | :--- |
| "It should know our products/policies" | RAG + a fresh index; tickets are a terrible knowledge base (they contain the *wrong* answers customers were told) |
| "It should sound like us / follow our format" | LoRA on 500–2k examples, or a style prompt — cheap either way |
| "Our data must never leave" | this is a **deployment** question (VPC/self-host), not a training question |

Run a **two-week bake-off on their data**, and say what you're measuring before you start: accuracy on 200 held-out tickets, hallucination rate, latency, cost per resolved ticket. Build both. Whoever loses, they keep the eval set — that's usually worth more than the model.

Be honest about the trap: fine-tuning on tickets teaches the model to *imitate past agent replies including the bad ones*, and it cannot cite a source, so the customer trades auditability for tone. Also, a 100k-ticket fine-tune needs a refresh pipeline every quarter; RAG just needs the docs re-indexed.

Frame it commercially, not technically: "we'll do whichever wins the bake-off, and here's what each costs to maintain in year two."

**中文要点**：别正面反驳，先拆"要自己的模型"背后是补知识、改风格还是数据不出域（那是部署问题）；用两周 A/B 拿他们的数据实测，并先说清指标；点出微调在工单上会学到坏答案且无法引用来源。

---

### 9. An agent you shipped four months ago runs on a base model being deprecated in 60 days. How do you swap the model without regressing quality, and what had to be in place beforehand?

**The 60-day clock is fine; the problem is everything you should already have.** A swap is mechanically `model="new"` and a config diff. What you can't create in 60 days is the evidence.

What must already exist (and if it doesn't, build it in week 1, not week 6):

1. **A frozen golden set** — 200+ real traces with graded expected outcomes, versioned. Without it you are shipping on vibes.
2. **Trace-level replay**: re-run *recorded* inputs (prompts, tool results) against the new model without hitting live tools. This is the only way to diff behaviour without touching customers.
3. **Prompt/tool-schema portability**: no reliance on model-specific quirks (a tool-call format, a magic system-prompt phrase, "the model remembers because it's the same conversation"). Those are the things that break.

The swap sequence that actually works:

| Week | Action |
| :--- | :--- |
| 1 | Replay golden set on old + new model; classify diffs into *格式* (fixable in prompt), *能力* (capability gap), *行为* (tool-use pattern changed) |
| 2 | Fix prompts/schemas for the mechanical diffs; re-measure. Investigate the top 10 regressions by hand |
| 3 | Shadow traffic: run new model on 100% of real requests, serve old answers; compare |
| 4–8 | Canary 5% → 25% → 100%, with automatic rollback on the eval metrics, not on error rate alone |

Two traps people hit: **a better model changes your tool-call rate**, so cost per request moves even when quality is flat; and **prompt tweaks made for the new model silently regress the old one** — so keep one prompt per model version until the old one is retired.

**中文要点**：换模型本身只是改一行配置，60 天真正要靠的是**已有冻结评测集 + 录制重放（不碰线上工具）+ 影子流量**；重点排查格式漂移、能力差和工具调用率变化，旧模型下线前 prompt 要按版本各留一份。

---

## 附：这 9 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 真写过并发/工程代码 | 1（batching logger）、3（CIDR 二分） |
| 大数据底子是真是假 | 4（straggler 定位）、5（checkpoint 与事务日志） |
| 有没有做过 GenAI 落地 | 6（微调 vs RAG）、7（上线清单）、9（模型换版） |
| 客户场景判断力 | 8（客户坚持微调） |
| 算法常识 | 2（有界内存 top-K） |
