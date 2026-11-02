# 14 · Meta (Superintelligence Labs, FAIR, Llama)

> 26 题。出处：`README.en.md` § Meta (Superintelligence Labs, FAIR, Llama)。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Given an array nums of n integers where n > 1, return an output array (product of array except self).

**Anchor answer**

**Reference implementation**

```python
def product_except_self(nums: list[int]) -> list[int]:
    n = len(nums)
    out = [1] * n
    for i in range(1, n):            # out[i] = product of nums[0..i-1]
        out[i] = out[i - 1] * nums[i - 1]
    right = 1
    for i in range(n - 1, -1, -1):   # fold in suffix product on the fly
        out[i] *= right
        right *= nums[i]
    return out
```

**Key points**

1. **No division.** The follow-up is always "what if there are zeros?" and the division solution breaks: two or more zeros ⇒ every output is 0; exactly one zero ⇒ only that index is non-zero. Prefix/suffix handles all cases with no branching.
2. **O(n) time, O(1) extra space** (output array not counted) — that is the expected optimal answer.
3. `out[i] = out[i-1] * nums[i-1]` uses `nums[i-1]`, not `nums[i]`; this off-by-one is the usual bug.
4. **Overflow**: with large ints Python is fine; in Java/C++ use `long` for the accumulator even if the result fits, because intermediates during the suffix pass can overflow.
5. If division *is* allowed and no zeros exist, `total // nums[i]` is fine — but say why you would not ship it.

Edge cases to mention: `n == 2`, negatives (sign flips are automatic), and a single zero.

**中文要点**：前缀积 + 后缀积一遍扫，O(n) 时间 O(1) 额外空间；**不要用除法**——有零时会崩（1 个零只有该位非零，≥2 个零全为 0）。常见 off-by-one 是用了 nums[i] 而非 nums[i-1]。

---

### 2. Find the minimum window in S which will contain all the characters in T.

**Anchor answer**

**Reference implementation**

```python
from collections import Counter

def min_window(s: str, t: str) -> str:
    if not t or len(t) > len(s):
        return ""
    need = Counter(t)
    missing = len(t)                # chars still needed (with multiplicity)
    left = 0
    best = (float("inf"), 0, 0)     # (len, start, end)
    for right, ch in enumerate(s):
        if need[ch] > 0:
            missing -= 1
        need[ch] -= 1
        while missing == 0:         # window is valid -> shrink
            if right - left + 1 < best[0]:
                best = (right - left + 1, left, right + 1)
            need[s[left]] += 1
            if need[s[left]] > 0:   # the char we drop is still required
                missing += 1
            left += 1
    return "" if best[0] == float("inf") else s[best[1]:best[2]]
```

**Key points**

1. **`need` goes negative for surplus characters** — that is the trick. `need[ch] > 0` before decrement means "this occurrence is genuinely needed"; surplus copies decrement below zero and can be dropped without invalidating the window.
2. **Duplicates matter**: `t = "AAB"` needs two A's, so a `set` is wrong. Track multiplicity.
3. Sliding window is O(|S| + |T|) — each index enters and leaves once. A "re-scan the window for each left" version is O(|S|²) and is the usual downgrade.
4. Empty result returns `""`, not `None` or the whole string.
5. Unicode: `Counter` on `str` counts code points; for emoji, index grapheme clusters instead.
6. A version that re-scans the window at each shrinkage is O(|S|²); the `need`/`missing` bookkeeping exists to avoid it.

**中文要点**：滑动窗口 + 计数，`missing` 归零即收缩；关键是 `need` 允许变负——只有 `need[ch]>0` 才算真正被消耗，多余的副本可安全滑出。重复字符必须按重数统计，用 set 会错。

---

### 3. Serialize and deserialize a binary tree.

**Anchor answer**

**Reference implementation** (preorder with explicit null markers — the format must be self-describing)

```python
class Codec:
    def serialize(self, root) -> str:
        out = []
        def go(node):
            if node is None:
                out.append("#")
                return
            out.append(str(node.val))
            go(node.left)
            go(node.right)
        go(root)
        return ",".join(out)

    def deserialize(self, data: str):
        vals = iter(data.split(","))
        def go():
            v = next(vals)
            if v == "#":
                return None
            node = TreeNode(int(v))
            node.left = go()
            node.right = go()
            return node
        return go()
```

**Key points**

1. **Null markers are mandatory.** Without them a preorder sequence is ambiguous (you cannot tell whether the next value is a left child or a sibling's). This is the single most common failure.
2. **Delimiter safety**: `,` and `#` must not collide with values — clamp or escape for negative numbers / arbitrary strings. For integers, `#` is safe; for arbitrary strings, use length-prefixed encoding.
3. **Preorder + markers reconstructs greedily with a single index** — no need for a second pass or the inorder trick that only works for BSTs.
4. **Recursion depth**: a degenerate (linked-list) tree of 10⁵ nodes blows Python's stack; either raise the limit, or use an explicit stack / BFS with an index queue.
5. Alternatives worth naming: BFS (LeetCode's canonical form), or postorder — both need markers; and for a BST, preorder alone suffices.

Complexity: O(n) time, O(n) space, both directions.

**中文要点**：前序遍历 + 显式 null 标记，反序列化用单指针贪心递归；**没有 null 标记就无法唯一还原**（最常见错）。退化树会爆栈，需显式栈或 BFS；分隔符要与取值域不冲突（字符串用长度前缀）。

---

### 4. Convert a binary tree to a circular doubly linked list.

**Anchor answer**

The classic in-place inorder conversion; each node's `left` becomes `prev`, `right` becomes `next`, and the result is circular.

**Reference implementation**

```python
def tree_to_dll(root):
    if root is None:
        return None
    first, last = None, None

    def inorder(node):
        nonlocal first, last
        if node is None:
            return
        inorder(node.left)
        if last is None:                 # leftmost node becomes the head
            first = node
        else:
            last.right = node            # link prev -> current
            node.left = last             # link current -> prev
        last = node
        inorder(node.right)

    inorder(root)
    first.left = last                    # close the circle
    last.right = first
    return first
```

**Key points**

1. **Save nothing extra** — the traversal must not use `node.left`/`node.right` after overwriting them; inorder handles this because you visit `left` *before* relinking, and `right` is read from the current node at the end of the frame (still intact). Using an explicit stack requires saving `right` first.
2. **`nonlocal last`** is the whole algorithm; the state machine is "append current node to the tail".
3. **Circularity is a separate, final step** — `first.left = last; last.right = first`. Forgetting it produces a doubly linked list, which is a different answer.
4. Order must be **inorder** to get a sorted list from a BST; the question says "binary tree" so state the assumption explicitly.
5. Recursion depth again: 10⁵-node skewed tree → iterative Morris traversal if O(1) space is required.

**中文要点**：中序遍历原地改指针，`last` 记录尾节点，最后 `first.left=last; last.right=first` 闭环——忘这一步就只是双向链表。用栈时必须先存 right，因为 left 已被改写；退化树要 Morris 或显式栈。

---

### 5. Alien dictionary: determine character ordering from a sorted word list.

**Anchor answer**

Build a directed graph over characters from **adjacent word pairs only**, then topologically sort.

**Reference implementation**

```python
from collections import defaultdict, deque

def alien_order(words: list[str]) -> str:
    adj = defaultdict(set)
    indeg = {c: 0 for w in words for c in w}
    for a, b in zip(words, words[1:]):
        for ca, cb in zip(a, b):
            if ca != cb:
                adj[ca].add(cb)
                break
        else:
            if len(a) > len(b):          # "abc" before "ab" is impossible
                return ""
    for u in adj:
        for v in adj[u]:
            indeg[v] += 1
    q = deque(c for c, d in indeg.items() if d == 0)
    out = []
    while q:
        u = q.popleft()
        out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return "".join(out) if len(out) == len(indeg) else ""
```

**Key points**

1. **Prefix rule is the trap.** `["abc", "ab"]` is invalid regardless of graph structure — a longer word cannot precede its own prefix. The `for...else` above detects it; most solutions miss it and pass the sample test.
2. **Only the first differing character** yields an edge; the rest of the pair is unordered.
3. **Cycle ⇒ invalid order** ⇒ return `""`. Detect via "processed fewer nodes than exist".
4. Self-loops (`adj[c]` containing `c`) must be ignored or they inflate indegree forever.
5. Duplicates and single-word inputs are legal; characters not in any edge must still appear in the output.

Complexity: O(total chars in input) time. In practice: build the graph once — an interviewer who asks for incremental updates across thousands of reorderings is asking for a different design.

**中文要点**：相邻词首个不同字符建边，再拓扑排序；两个必查点——**前缀规则**（"abc" 不能排在 "ab" 前，与图无关）与**环检测**（输出长度不足即非法）。只取首个不同字符，自环要忽略。

---

### 6. K closest points to origin; top-k frequent elements; minimum number of conference rooms.

**Anchor answer**

Three problems, one skill: **"does the ordering of the whole set matter, or only the boundary?"**

| Problem | Best tool | Time | Why not the alternative |
| :--- | :--- | :--- | :--- |
| K closest points | max-heap of size k, or **quickselect** | O(n log k) / O(n) avg | sorting is O(n log n) for information you throw away |
| Top-k frequent | `Counter` + **bucket sort by frequency** | O(n) | freq ≤ n makes counting sort valid |
| Min conference rooms | sort starts & ends, sweep with a counter | O(n log n) | heap of end times is O(n log n) too, but more code |

```python
import heapq
from collections import Counter

def k_closest(points, k):
    return heapq.nsmallest(k, points, key=lambda p: p[0]**2 + p[1]**2)

def top_k_frequent(nums, k):
    cnt = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    for v, c in cnt.items():
        buckets[c].append(v)
    out = []
    for c in range(len(buckets) - 1, 0, -1):
        out.extend(buckets[c])
        if len(out) >= k:
            return out[:k]

def min_rooms(intervals):
    starts = sorted(s for s, _ in intervals)
    ends = sorted(e for _, e in intervals)
    rooms = best = 0
    i = j = 0
    while i < len(starts):
        if starts[i] < ends[j]:      # a new meeting starts before one frees
            rooms += 1
            i += 1
            best = max(best, rooms)
        else:
            rooms -= 1               # half-open: end == start frees the room
            j += 1
    return best
```

**Key points**

1. **`starts[i] < ends[j]` vs `<=`** is the semantic decision: back-to-back meetings (end == start) can share a room. State the convention.
2. Distances: compare squared distances, never `sqrt` — avoids float error and is faster.
3. `heapq.nsmallest` on k ≪ n is the practical answer; quickselect is the O(n) one if pushback comes.
4. Empty input → 0 rooms, empty list; validate that `len(points) >= k`.

**中文要点**：三题共性是"只关心边界不关心全序"。K 近点用 size-k 大顶堆或 quickselect（比较平方距离，别开方）；top-k 频次用按频次桶排序 O(n)；会议室用起止双指针扫描，注意 `end == start` 可复用同一间的约定。

---

### 7. Regular expression matching with '.' and '\*'.

**Anchor answer**

`'.'` matches any single char; `'*'` means **zero or more of the preceding element**. Note this is not shell globbing — `*` never stands alone.

**Reference implementation** (bottom-up DP)

```python
def is_match(s: str, p: str) -> bool:
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(2, n + 1):                  # "a*b*c*" can match ""
        if p[j - 1] == "*":
            dp[0][j] = dp[0][j - 2]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == "*":
                # zero occurrences, or one more of p[j-2]
                dp[i][j] = dp[i][j - 2] or (
                    dp[i - 1][j] and p[j - 2] in (s[i - 1], ".")
                )
            elif p[j - 1] in (s[i - 1], "."):
                dp[i][j] = dp[i - 1][j - 1]
    return dp[m][n]
```

**Key points**

1. **`dp[i][j-2]` is "zero occurrences"** — the most-forgotten transition. It is what makes `a*` match the empty string.
2. **`dp[i-1][j]` (not `j-1`) for "one more"** — you stay in the same pattern column because `x*` can consume more.
3. **`*` must be paired with the preceding element**, so valid patterns never start with `*`; guard or state the assumption.
4. Base row init loops `range(2, n+1)` — `dp[0][1]` can never be true.
5. O(mn) time and space; rolling two rows gives O(n) space. A naive recursion is exponential on inputs like `a*a*a*a*b` and is the usual wrong answer.

**中文要点**：`*` 表示"前一个元素重复 0 次或多次"，所以关键转移是 `dp[i][j-2]`（零次）；"再来一次" 用 `dp[i-1][j]` 留在同列。O(mn) DP，两行滚动省空间；朴素递归在 `a*a*a*b` 上指数爆炸。

---

### 8. Two-part warm-up: given a stream of user actions, return the k most engaged-with items. Then: why might your heap solution be the wrong choice in production?

**Anchor answer**

Part 1 is the standard heavy-hitters problem — Part 2 is the real question.

**Part 1.** Keep `Counter[item]` plus a size-k min-heap keyed by count. On each event, increment, and if the item is now above the heap minimum, push and evict. O(log k) per event, O(distinct + k) memory.

**Part 2 — why the heap is wrong:**

| Production reality | What breaks |
| :--- | :--- |
| Distinct items in the millions/billions | `Counter` is **unbounded** memory — the heap was never the problem, the map is |
| Engagement is not a count | "engagement" is weighted (dwell time, shares) and **decays**; a raw counter makes last month's viral item permanently #1 |
| Distributed ingestion | each shard's local top-k is not the global top-k; you must merge *sketches*, and you cannot merge heaps without the counts |
| Ties and staleness | heap ordering by count gives arbitrary tie-breaks and no stable recency |
| Hot-key skew | one item gets 30% of events → the counter for that key is a contention point, not a heap issue |

The fix is **bounded approximate counting with decay**: count-min sketch (or Space-Saving) per shard, exponential decay on a time bucket, then merge sketches and take the top-k of the merged estimate. If you need exactness for a small hot set, use a two-tier design: exact counters for the top ~10k, sketch for everything else.

Also worth saying: if the consumer is a dashboard refreshing every N seconds, compute top-k **on a schedule**, not per event — the incremental structure buys nothing.

**中文要点**：堆不是问题，**Counter 无界**才是：真实场景 key 上亿。且"engagement" 是加权、会衰减的，裸计数会让上月爆款永久霸榜；分布式下要合并 sketch 而非局部堆。方案：count-min/Space-Saving + 时间桶衰减，热 key 再分层精确计数。

---

### 9. Your ads CTR model shows a 2% offline AUC gain, but the online A/B is revenue-neutral with worse calibration. What is going on, and what do you do?

**Anchor answer**

AUC is **rank-only and global**; ads revenue is a **calibrated, per-impression** function. A model can improve ranking everywhere yet get the price wrong — and in an auction, the price *is* the product.

Diagnosis, most likely first:

1. **Calibration shifted.** AUC is invariant to monotone transforms, so it cannot see that predictions are now systematically high/low. Revenue depends on `pCTR × bid` in eCPM, and on pacing/budget constraints — an inflated pCTR overspends budgets early and starves the day.
2. **Train/serve skew.** Feature computed differently offline (window, join, default value). Check per-feature PSI between logged serving features and training features.
3. **Position bias / selection bias.** The new model shows different items, so the logged labels come from a different distribution. Offline eval on re-weighted logs would show the gain is partly artefact.
4. **AUC gain concentrated in the tail.** The 2% may come from rare items that never win an auction (they are never the argmax), so no revenue impact. Segment AUC by impression count — the gain is probably in the bottom decile.
5. **Label delay.** Conversions arrive hours/days later; offline uses matured labels, online serves with immature ones.

Actions: add **log-loss / calibration error** to the model gate, not just AUC; fit a **post-hoc calibrator** (isotonic on the most recent slice, Platt if data is thin); check **eCPM-weighted AUC** as the offline proxy that actually correlates with revenue; verify feature parity with a training-serving consistency test; and if the win is tail-only, don't ship it.

Rule of thumb: never ship a model on AUC alone — pair it with calibration and a revenue-weighted metric.

**中文要点**：AUC 只看排序且对单调变换不变，感知不到校准漂移；而广告收入 = pCTR×bid 参与竞价和预算节奏，校准错了收入就不动。先查训练/服务特征偏斜与位置偏差，再看增益是否只落在赢不了竞价的尾部。上线门槛必须加 log-loss/校准误差与 eCPM 加权 AUC。

---

### 10. Explain the architectural choices in a Llama-class model: why grouped-query attention, RoPE and SwiGLU instead of the vanilla 2017 Transformer?

**Source**: https://outcomeschool.com/blog/grouped-query-attention · https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding · https://outcomeschool.com/blog/feed-forward-networks-in-llms

**Anchor answer**

Each change targets a measured cost of the original Transformer — mostly **inference memory**, not training perplexity.

| Change | Problem it fixes | Mechanism | Cost/benefit |
| :--- | :--- | :--- | :--- |
| **GQA** | KV cache dominates decode memory | $G$ query heads share 1 KV head; interpolates MHA ($G{=}H$) and MQA ($G{=}1$) | KV cache shrinks by $H/G$ (e.g. 8× on 70B with 8 KV heads) at a small quality cost vs MHA |
| **RoPE** | learned absolute positions do not extrapolate and are baked into weights | rotate Q/K pairs by an angle proportional to position; the dot product depends only on the **relative** offset | no position table, relative by construction, tunable for long context (base scaling, YaRN) |
| **SwiGLU** | a plain ReLU FFN wastes parameters | $\text{SiLU}(xW_1)\odot xW_3$ then $W_2$ — a gated, multiplicative interaction | better quality **per parameter**; 3 matrices ⇒ hidden dim scaled to ~⅔ so total params match |
| **Pre-RMSNorm** | post-norm needs warmup and diverges at depth | normalise before each sublayer, no mean subtraction | cheap, stable, standard for deep stacks |
| **No biases** | small gain, extra params | drop all bias terms | negligible effect, simpler kernels |

The one-sentence answer: **GQA buys serving cost, RoPE buys position generalisation, SwiGLU buys quality per FLOP.** None of them is about making training faster; three of five are about making *decode* cheaper.

Failure mode to mention: naive RoPE extrapolation degrades past the trained length — you need base-frequency scaling or interpolation, not just a bigger context window.

**中文要点**：GQA 降 KV cache（70B 用 8 个 KV head 省约 8 倍显存）、RoPE 用旋转给出相对位置且可外推、SwiGLU 门控提升每参数质量（隐层缩到约 2/3 保参数量）、Pre-RMSNorm 保稳定。核心是**推理成本**而非训练速度；RoPE 外推需频率缩放。

---

### 11. What breaks when you scale LLM training from 8 GPUs to thousands, and how do modern stacks deal with it?

**Anchor answer**

At 8 GPUs you have one node and no failures. At thousands, **four categories break simultaneously**, and each has a standard countermeasure.

| What breaks | Why | Countermeasure |
| :--- | :--- | :--- |
| **Communication** | all-reduce/all-gather scales with world size; cross-node bandwidth is ~10–50× lower than NVLink/ICI | 3D parallelism: TP inside a node, PP/FSDP across nodes, hierarchical collectives, overlap comm with compute |
| **Memory** | activations grow with batch × seq; optimizer state is 12 bytes/param | ZeRO-3/FSDP sharding, activation checkpointing, sequence parallelism |
| **Numerics** | longer runs see more spikes; bf16 accumulation drift | bf16 with fp32 master weights, grad clipping, z-loss, QK-norm |
| **Reliability** | MTBF per GPU is hours-to-days; at 4k GPUs a failure is expected every few hours | **async sharded checkpointing** every ~30 min, elastic restart, health checks, deterministic resharding |
| **Data pipeline** | one input bottleneck starves thousands of chips | pre-tokenised shards, deterministic shuffling, enough loader workers, prefetch depth |
| **Debugging** | a bad step is 1 in 10⁶ | per-step metric logging, loss-spike detectors, ability to dump the offending batch |

The structural insight: at thousands of GPUs, **the job is a distributed system, and the ML is a client of it.** Utilization (MFU) is governed by the slowest of comms, data, and failures — which is why serious stacks invest more in checkpoint/restart and comm overlap than in the model code.

Also: global batch size changes the optimization landscape (LR must scale, warmup lengthens), and hyperparameters tuned at 8 GPUs do not transfer — expect a re-tuning phase, and log LR/grad-norm curves to catch it.

**中文要点**：从单机到千卡，通信、显存、数值、可靠性四件事同时坏。对策：TP 进节点、PP/FSDP 跨节点并重叠通信；重算 + 分片降显存；bf16+fp32 master 与梯度裁剪防尖峰；**异步分片 ckpt + 弹性重启**（千卡下几小时一挂）。本质是"训练作业变成分布式系统"，MFU 由最慢的一环决定。

---

### 12. You need to serve a Llama-class 70B+ model to hundreds of millions of assistant users. What does the serving stack look like and where does the money go?

**Source**: https://outcomeschool.com/blog/llm-inference-optimization

**Anchor answer**

Money goes to **GPU-seconds of decode**, and decode is memory-bandwidth-bound — so the entire stack is an exercise in raising batch size without blowing the KV cache.

| Layer | Job | Lever |
| :--- | :--- | :--- |
| Routing | cheap model first, escalate on difficulty | 40–70% of traffic never needs the big model |
| Prefix cache | cache the system prompt + tool schema KV | cuts prefill ~50%+ on agent traffic |
| Prefill / decode split | different hardware shapes | disaggregation raises MFU for both |
| Batching | continuous/in-flight batching | throughput scales near-linearly to the KV limit |
| KV memory | paged attention, quantization (fp8/int8 KV) | 2× batch for ~free quality |
| Weights | fp8/int8, sometimes 4-bit | halves HBM, raises batch |
| Speculative decoding | small draft model | 1.5–2× latency win when acceptance is high |

Sizing sanity check: 70B in fp8 ≈ 70 GB weights; an 80 GB H100 leaves ~10 GB — not enough. So you need **TP across ≥2 GPUs per replica** (or a smaller/quantised model), and each replica's batch is capped by KV bytes: `2 × layers × kv_heads × head_dim × seq_len × bytes`. At 8k context this is the number that decides your $/1k tokens.

Failure modes: **cache stampede** on shared prompts (coalesce identical requests), **head-of-line blocking** from one long generation (chunked prefill + max-token caps), and **routing hysteresis** (users bounce between models mid-conversation and see quality cliffs).

Where not to optimise: prefill is a small share of cost at long outputs; teams routinely over-invest in faster encoders while decode dominates 80% of the bill.

**中文要点**：钱在 decode 的 GPU 秒，而 decode 是显存带宽瓶颈——整栈的目标是"在 KV cache 撑得住的前提下把 batch 拉大"。手段：便宜模型先路由、prefix cache、prefill/decode 分离、PagedAttention + fp8 KV、投机解码。70B fp8 约 70GB，单卡放不下，必须 TP≥2。

---

### 13. You're dropped into an unfamiliar multi-file codebase with a failing behaviour and an LLM assistant available. Walk me through how you'd fix it.

**Anchor answer**

The skill tested is **not** "ask the LLM to fix it" — it is using the assistant as a fast reader while keeping reasoning anchored in evidence.

1. **Reproduce first, alone.** Get the failing command/case running and capture the exact error. Without a reproduction, every LLM answer is unfalsifiable. 10 minutes here saves hours of plausible-sounding wrong patches.
2. **Map the blast radius before reading code.** Check `git log` on the touched area, find the entry point from the stack trace, and ask the LLM: "here is the traceback and the file list — which 3 files are on the call path, and what should I read first?" Verify by opening them; the model is a search heuristic, not an oracle.
3. **Read the failing path, not the repo.** Follow the trace from entry to the throw. Ask for one-paragraph summaries of files it can see pasted in, never for whole-file rewrites of code it cannot run.
4. **State the hypothesis as a falsifiable claim** — "the cache key omits the tenant id, so tenant B reads tenant A's entry." Then write the smallest test that fails for that reason and passes after the fix.
5. **Fix narrowly, then check for siblings.** Search for the same pattern elsewhere (`grep` the anti-pattern); the bug is usually copied in 3 more places.
6. **Verify with the test suite** — an LLM patch that "looks right" and breaks two other tests is normal, not exceptional.

Failure modes: accepting an invented API; letting it refactor swathes of unfamiliar code (unreviewable diff); and skipping the reproduction, so you "fix" a symptom into silence.

**中文要点**：核心是"先自己复现，再让 LLM 当快速阅读器"。流程：复现→按调用栈定位 3 个文件→读失败路径→把假设写成可证伪命题→写最小失败测试→窄修→grep 同类问题→测试验证。别让它大范围重构陌生代码。

---

### 14. Walk me through a post-training recipe to turn a pretrained base model into a personalised assistant.

**Source**: https://outcomeschool.com/blog/decoding-instructgpt · https://outcomeschool.com/blog/reinforcement-learning-from-human-feedback-rlhf

**Anchor answer**

Split "assistant" from "personalised" — they need different mechanisms, and conflating them is the common design error.

| Stage | Data | Objective | Notes |
| :--- | :--- | :--- | :--- |
| 1. SFT | 10k–1M high-quality instruction/response pairs, assistant-style | next-token CE on responses only (mask the prompt) | quality ≫ quantity; dedupe, and cover refusals/format/tool calls |
| 2. Preference optimisation | pairwise comparisons (human or AI-judged) | DPO (or PPO/GRPO with a reward model) | fixes tone/verbosity/sycophancy that SFT cannot express |
| 3. Personalisation | per-user history, profile, corrections | **retrieval/memory first**, then per-user adapters | do not fine-tune per user at scale |

The key trade-off: personalisation via **fine-tuning** is expensive, hard to update, and leaks across users; via **memory/RAG** it is cheap, auditable, and revocable. Default to memory + a compact profile (name, units, tone preference, do-not-mention list) injected as context, and reserve LoRA for a small number of enterprise tenants.

Order matters: SFT sets format and refusal behaviour, DPO shifts preferences — running DPO on a base model without SFT usually produces degenerate outputs because the reference distribution is not yet an assistant.

For a preference stage, note the practical: pairwise labels are noisy and annotator-dependent; keep a held-out preference set and watch for **length bias** (models learn "longer = better") and reward hacking (verbosity, flattery). Evaluate against the base model on a frozen prompt suite, plus a human spot-check, before shipping.

**中文要点**：三段式：SFT（只对回答算 loss，质量优先）→ 偏好优化（DPO/PPO 治语气与啰嗦）→ 个性化**优先用记忆/RAG 而非逐用户微调**（便宜、可审计、可撤销）。顺序不能反：没 SFT 直接 DPO 会退化。偏好数据要防长度偏置与讨好型 reward hacking。

---

### 15. Design the recommendation system for Instagram Reels.

**Anchor answer**

Short video changes three things versus a feed ranker: the **label is duration-normalised**, the **candidate pool is enormous and fresh**, and **session-level satisfaction** matters more than a single click.

**Stage 1 — retrieval.** Two-tower (user: history, follows, embeddings of watched reels; item: visual/audio/text embeddings, creator, age) over an ANN index, plus multiple sources unioned: followed creators, co-engagement ("people who watched X"), trending-by-geo, and freshly uploaded (a **freshness channel** with a quota, or new creators never surface). Target ~1000–5000 candidates.

**Stage 2 — ranking.** Multi-task: P(complete), P(like), P(share), P(comment), P(skip < 2s), P(report). Combine with weights tuned on **session watch time**, not on click. Watch time must be predicted in **seconds**, then normalised by video length, or the model just prefers long videos.

**Stage 3 — re-ranking.** Dedupe by creator/audio/visual near-duplicate (embedding clustering — Reels has heavy reposting), diversity caps, seen-filter, and integrity filtering before ranking is wasted.

Specific failure modes:

- **Position bias** in the label: items shown higher get watched more, and the model learns the slot. Correct with inverse propensity or a position feature at train time, dropped at serve.
- **Cold start for new reels**: no engagement yet, so use content embeddings + a small exploration budget (ε-greedy or Thompson sampling); otherwise the catalogue ossifies.
- **Feedback loop on creators**: top creators get all the exposure, then the model has no data on the rest.

Offline: NDCG on watch-time-weighted labels, plus recall@k per channel. Online: session watch time, DAU retention, and a creator-diversity guardrail metric.

**中文要点**：短视频三处不同：标签要按视频时长归一（否则模型偏长视频）、候选池巨大且需**新内容通道与探索预算**、目标是会话级满足而非单次点击。排序多任务（完播/赞/分享/2 秒跳出/举报）加权到会话观看时长；重排先去重同音频/同创作者的搬运。

---

### 16. Design a personalised news-feed ranking system / the “next post” logic for Facebook's feed.

**Anchor answer**

Feed is a **session-level sequential decision problem**, not an independent per-item ranking. The same item is worth more or less depending on what preceded it.

**Architecture**

| Layer | Content |
| :--- | :--- |
| Candidate sources | friends' posts, groups, pages followed, suggested content, ads — each with its own quota; the union is ~1000s |
| Eligibility | integrity filters (violating content), seen-state, "hide author", time decay |
| Light ranker | cheap model on hundreds of features → top ~100–500 |
| Heavy ranker | multi-task DNN: P(click), P(like), P(comment), P(share), P(hide/report), P(dwell) |
| Value model | actions → expected utility; **negative actions get large negative weights** |
| Re-ranker | diversity (author/topic caps), freshness boost, ads insertion at fixed slots |
| Session controller | sequences ~10–20 slots greedily/with lookahead, updating a user-state vector after each |

Key design points:

1. **Dwell-based labels.** Click-through alone rewards clickbait; weight by dwell time with a saturation curve, and model "long dwell then hide" as negative.
2. **Sequential state.** "Next post" depends on the previous ones, so a context vector (recent topics, session fatigue) is fed back after each slot; per-item scoring cannot represent "user just saw three political posts".
3. **Freshness vs relevance.** An old-but-great post loses to a new mediocre one under pure expected utility; add an explicit age penalty and a per-source freshness floor so the feed does not feel stale.
4. **Integrity before ranking** — a great ranker over violating content is worse than a mediocre one over clean content.

Evaluation: offline replay with position-bias correction, online session-length and DAU/retention, with guardrails on hide/report rate and diversity. Watch for the trap of optimising click while dwell and retention fall.

**中文要点**：Feed 是**会话级序列决策**，不是逐条独立打分。分层：多来源候选→完整性过滤→轻排→多任务重排（含大量负向动作）→多样性/广告插入→会话控制器带着"最近看了什么"的状态选下一帖。标签用 dwell 且惩罚"长停留后举报"；纯点击会奖励标题党。

---

### 17. Design a recommendation system for Facebook Ads, and an evaluation framework for ads ranking.

**Anchor answer**

Ads differ from organic ranking in one decisive way: **the objective is money under budget constraints**, so the ranker feeds an auction, and calibration matters more than ordering.

**System**

1. **Candidate generation** — advertiser targeting constraints (geo, age, interest, custom audiences) prune the pool; typical candidate sets are hundreds to thousands per impression.
2. **Prediction** — separate heads for pCTR and pCVR (CVR is sparse and delayed, so it is often trained with a shared trunk and a delayed-feedback correction), plus p(report/hide) as a quality penalty.
3. **Auction** — expected value `eCPM = pCTR × pCVR × bid` (or `pCTR × bid` for CPC), then a **quality/negative-feedback adjustment**, then second-price/VCG-style pricing. A miscalibrated pCTR directly misprices: too high ⇒ overspend and bad ROI; too low ⇒ starvation.
4. **Pacing & budget** — a per-advertiser controller spreads spend over the day; without it, budget exhausts by noon and the auction loses competitiveness.
5. **Frequency capping** and diversity across advertisers.

**Evaluation framework**

| Layer | Metric | Note |
| :--- | :--- | :--- |
| Offline | log-loss / calibration error (ECE), eCPM-weighted AUC | plain AUC is the metric that lies |
| Counterfactual | IPS / DR estimator on logged data | corrects for the policy that logged the data |
| Online A/B | revenue, ROAS, advertiser ROI, CVR | randomise at user level |
| Incremental | **ghost ads / holdout** | measures true incremental conversions, not attributed ones |
| Guardrails | hide/report rate, ad load, long-term retention | revenue with rising hide-rate is a debt |

The subtle point: attributed conversions overstate incrementality, so a holdout is the only way to know if the ads caused the purchase. And long-term effects (ad load fatigue) are invisible in a two-week test — run a longer holdout cohort.

**中文要点**：广告是"带预算约束的赚钱问题"，排序喂给竞价，所以**校准比排序重要**——pCTR 偏高会超支、偏低会饿死。架构到 pacing/频控；评估必须分层：log-loss/ECE、eCPM 加权 AUC、IPS/DR 反事实、在线 A/B，以及 ghost ads 留白测**增量**，别拿归因转化当增量。

---

### 18. Design the ML components behind an Instagram Story feature.

**Anchor answer**

Stories are ephemeral (24 h), full-screen, and **consumed in rapid sequence** — so the ML problems are ranking the tray, ranking within a story, and picking the creative/effects. Take them one at a time.

| Component | Problem | Model |
| :--- | :--- | :--- |
| **Tray ordering** | which friends' stories to show first, out of hundreds | rank by P(view) × P(complete) with an explicit recency ring: unseen-first, then ranked; a few "close friends" slots are pinned |
| **Story ads** | which ad after which story | treat like feed ads but with stricter frequency caps and shorter attention: pCTR on a 1–3 s skip window |
| **Music / sticker suggestion** | which track or sticker for a draft | content-based retrieval (audio embedding + visual embedding) + popularity prior + trend freshness |
| **Audience suggestion** | close friends list | graph features: interaction recency/frequency, DM history |
| **Creative effects** | AR filter ranking | engagement-weighted retrieval, with fairness caps so small creators surface |

Shared machinery: a user-embedding service updated per session, item embeddings from content + engagement, and a two-stage retrieve-then-rank everywhere.

Story-specific failure modes:

1. **Seen-state is the dominant feature.** A story watched once is worth ~0; the ranker must receive a per-user "already seen this segment" signal, and the label must be per-segment, not per-story.
2. **Extreme class imbalance.** View rate is high (people tap through), completion is what discriminates; model `P(complete | view)` separately.
3. **Short lifetime ⇒ no engagement history.** Content features and the creator's history carry the cold start; a story has hours, not weeks, to accumulate signal.
4. **Ephemerality and privacy** — models must not persist per-story features after expiry; feature-store TTLs are a correctness requirement, not an optimisation.

**中文要点**：Stories 是 24 小时失活、全屏快滑的，所以要拆成托盘排序（未见优先+完成率）、广告、音乐/贴纸推荐、亲密好友建议、AR 特效五块。核心特征就是"已看状态"（看过≈0 价值），标签必须按 segment；冷启动只能靠内容与创作者历史，且特征 TTL 到期必须删。

---

### 19. Design an end-to-end classification pipeline for Marketplace listings.

**Anchor answer**

Marketplace listings are **user-generated, adversarial, and long-tailed** — the pipeline is as much about taxonomy and review workflow as about the model.

**Pipeline**

1. **Ingestion & validation** — image quality, duplicate detection (perceptual hash + embedding near-duplicate), price sanity vs category distribution, seller history. Reject obvious garbage before spending model compute.
2. **Understanding** — two heads: (a) **taxonomy classification** (category → subcategory → attributes, hierarchical softmax or per-level classifiers with a constrained beam so predictions are consistent down the tree), (b) **policy risk** (prohibited items, counterfeit signals, weapons, recalled goods).
3. **Attributes** — extract brand, size, condition and colour from title/description/image via a small VLM or NER; these power search filters and are where listings most often fail.
4. **Ranking/fusion** — combine text and image scores; a listing whose photo and title disagree is a fraud signal, not a modelling nuisance.
5. **Human-in-the-loop** — confidence bands: auto-approve high-confidence, auto-reject only very high-confidence policy violations, and route the ambiguous middle to reviewers. Reviewer decisions become training data (active learning).
6. **Feedback** — buyer reports, "wrong category" flags, and search-click logs feed back as weak labels.

Key trade-offs:

- **Precision/recall is asymmetric per head**: mis-categorising a sofa costs search quality; missing a prohibited weapon costs regulatory exposure. Threshold separately — never one global cut-off.
- **Long tail**: thousands of categories with <100 examples each. Use hierarchy (parent model as prior), few-shot prompts and embeddings, not a flat 10k-way classifier.
- **Class imbalance in policy**: prevalence may be <0.1%, so precision at fixed recall is the operating metric, and you must audit on a **randomly sampled** holdout, not on reviewer-flagged data (which is biased by what was flagged before).

**中文要点**：二手商品是 UGC + 对抗 + 长尾，管道重点是类目体系与审核流，不只是模型。分层：入库校验/去重 → 层级类目分类 + 政策风险 → 属性抽取（喂搜索筛选）→ 图文一致性 → 置信度分带的人审（只极高置信才自动处置）→ 用户举报回流。违禁品与错类目要用**不同阈值**；评估必须用随机抽样集，别用被举报过的数据。

---

### 20. Design a language translation model / service.

**Anchor answer**

Two decisions dominate: **architecture** and **what "quality" means operationally**.

**Architecture.** For a general service, a decoder-only model with instruction-style translation is now the default for one-shot low-resource flexibility; an encoder-decoder is still better per-FLOP for high-volume, single-language-pair translation because there is no decoder-side re-encoding of the source. For 100+ languages, a single multilingual model with language tags beats 100 pairs of bilingual models — shared parameters, cross-lingual transfer, and low-resource languages borrow from high-resource ones.

**Pipeline**

| Stage | Detail |
| :--- | :--- |
| Pre-processing | language ID, sentence splitting, **placeholder protection** for URLs/numbers/code, normalisation |
| Retrieval augmentation | glossary + translation memory lookup for names/products — the only reliable way to get brand terms right |
| Model | multilingual seq2seq, ~beam 4 or greedy + length penalty |
| Post-processing | detokenise, restore placeholders, terminology-constrained decoding |
| Quality estimation | reference-free QE score per segment → flag low-confidence for human/alternate model |

**Critical details**

1. **Terminology and named entities** are where MT fails commercially. Constrained decoding or post-hoc replacement with a glossary is worth more than a bigger model.
2. **Document context** matters: translating sentence-by-sentence destroys pronouns and formality; batch a paragraph and pass context.
3. **Metrics**: BLEU is cheap but rewards the wrong things; **COMET/chrF** correlates better with humans, and for production you need **human MQM** on a stratified sample. Track metrics per language pair — an aggregate hides that one pair regressed 5 points.
4. **Latency**: streaming/partial translations change the objective (you cannot revise earlier output), so cap segment length and expose confidence.

**中文要点**：架构选择：通用小语种用 decoder-only 指令式翻译，高量单语对仍适合 encoder-decoder；多语种共享模型 + 语言标签优于 100 个双语模型。真正决定商用质量的是**术语/实体保护与术语表约束解码**、段落级上下文、以及 COMET+MQM 分类别评测（BLEU 会骗人）。

---

### 21. How would you build the evaluation system for a Meta AI assistant before and after each model release?

**Source**: https://outcomeschool.com/blog/llm-evaluation

**Anchor answer**

An assistant spans many surfaces (chat, search, image, voice), so the eval system is a **portfolio with a release gate**, not a single score.

**Layer 1 — regression suite (must not break).** ~500–2000 frozen prompts from real logs, covering: instruction following, formatting/tool calls, refusals, multi-turn context, multilingual, and known past bugs. Every past production incident becomes a test case. This is the cheapest and highest-value component.

**Layer 2 — capability evals.** Held-out benchmarks for reasoning/code/math/knowledge, plus **contamination checks**. Report per-category, never only the average.

**Layer 3 — safety.** Harmful-compliance rate, jailbreak suites (static + adaptive), bias probes across demographics, and over-refusal rate — safety and helpfulness must be measured *together*, because every safety fix trades against refusals.

**Layer 4 — human evaluation.** Pairwise preference on a stratified sample (task type × language × difficulty), with MQM-style error tagging. LLM judges are a scaling device: validate them against humans on a subset, and watch for position/verbosity bias.

**Layer 5 — online.** Gradual rollout with guardrail metrics: thumbs-down rate, conversation abandonment, report rate, latency, and per-surface engagement. Offline gates decide *whether* to launch; online decides *how far*.

Operational requirements: versioned prompt sets and judge prompts, cached completions for re-scoring, paired bootstrap CIs for deltas, an owner per axis, and **pre-registered thresholds** — otherwise every miss becomes a negotiation. Fix the judge before the model: a flaky judge makes the whole system noise.

**中文要点**：五层：冻结回归集（历史事故全变用例，最有价值）、能力基准（查污染）、安全（有害服从率 + 越狱 + 过度拒答一起测）、人类/LLM 评审（judge 必须用人校准）、线上灰度护栏。必须有版本化 prompt、缓存输出、配对 bootstrap 置信区间和**预先注册阈值**，否则每次不达标都能谈判。

---

### 22. Design the harmful-content detection system for Facebook and Instagram uploads.

**Anchor answer**

Scale forces a **cascade**, and policy forces **human review** — the model only decides how much human attention each item gets.

**Cascade**

| Tier | Signal | Cost | Role |
| :--- | :--- | :--- | :--- |
| 0 | hash matching (known violating media), account reputation, prior strikes | ~free | instant action on known content |
| 1 | lightweight classifier on metadata + thumbnail + text | ms | bulk triage |
| 2 | multimodal model (image/video frames + audio + caption) | 100s of ms | the real detector |
| 3 | human reviewer | minutes | ambiguity, appeals, edge policy |

Video is the hard part: sample keyframes (scene-change detection), run the image model per frame, and aggregate with temporal pooling — plus a separate audio classifier for speech/music. Do not decode the whole video for every upload; tier by risk.

**Operating point, not accuracy.** Prevalence can be well under 1%, so a 99% accurate model can still be mostly wrong in its alerts. Set thresholds per policy category by **cost asymmetry**: terrorism/CSAM require near-zero false negatives (accept over-blocking and review load); spam and mild nudity tolerate more FNs to protect free expression. Report **precision at a fixed recall** and the volume of items routed to humans.

**Non-model requirements** (which interviewers weight heavily):

1. **Appeals** — every automated action must be contestable, and appeal outcomes must feed back as labels.
2. **Proactive rate** vs user reports — if most takedowns come from user reports, detection is failing.
3. **Adversarial drift** — users re-encode, crop, add text overlays; hash matching dies instantly, so keep a perceptual-hash + embedding near-duplicate channel.
4. **Language/region coverage** — a model trained on English text will under-detect in other languages; audit per-locale.
5. **Auditability** — log the model version, threshold, and evidence for every action.

**中文要点**：级联设计：哈希/账号信誉→轻量分类→多模态（视频关键帧+音频）→人工审核。核心不是准确率而是**按政策类别设定工作点**（恐袭/CSAM 近乎零漏检，轻度内容容忍更多误杀），报"固定召回下的精确率"与人审量。另外必须有申诉回流、主动发现率、对抗漂移（重编码/裁剪）与分语种审计。

---

### 23. How do modern multimodal models get image and video understanding into an LLM, and what changes for video specifically?

**Source**: https://outcomeschool.com/blog/multimodal-ai

**Anchor answer**

Two families, one trade-off between **modularity** and **token efficiency**:

| Approach | Mechanism | Cost |
| :--- | :--- | :--- |
| **Late fusion / adapter** | frozen ViT → projector (MLP or Q-Former) → soft tokens prepended to text | simple, cheap, but bottlenecked by a fixed token budget and loses fine detail |
| **Early fusion / native** | image patches tokenised and interleaved with text in one transformer, trained jointly | best quality and resolution flexibility, expensive to train |

The projector is the load-bearing piece: an MLP is enough when the encoder is strong (LLaVA-style), a Q-Former/resampler compresses to a fixed token count when you must bound cost.

**What changes for video:**

1. **Token explosion.** 1 minute at 1 fps × 256 tokens/frame = 15k tokens before audio. So video needs **temporal compression**: frame sampling by scene change, pooling across frames, or a dedicated temporal encoder — not "just feed more frames".
2. **Time must be represented.** Frames need temporal position encoding (or timestamp text tokens), otherwise the model cannot tell "before" from "after" — critical for action ordering and causality questions.
3. **Motion, not frames.** Static-frame models miss direction and speed; either add optical-flow-like features or train on clips with temporal contrastive objectives.
4. **Audio is half the signal.** Speech + non-speech events need their own encoder; video QA on silent frames is a common quality ceiling.
5. **Serving cost.** Video prefill dominates; cache per-video embeddings and re-use across questions about the same clip.

Failure mode to name: long-video models that answer from the first and last frame (a position bias), and audio-less training that makes the model hallucinate narration.

**中文要点**：两条路线：冻结 ViT + projector 接软 token（便宜、有 token 上限），或原生早融合（质量高、训练贵）。视频额外三件事：token 爆炸（60 秒≈1.5 万 token，必须抽帧/时序压缩）、必须有时序位置编码否则分不清先后、音频是半个信号；工程上要缓存整段视频的嵌入。

---

### 24. Give me an example of a project where you used data and machine learning. What obstacles did you hit?

**Anchor answer**

Pick a project with a **number that moved**, then spend most of the answer on the obstacles — that is what is being graded.

Structure:

1. **Setup (2 sentences).** Problem, scale, your role. "Ranking for a 40M-item marketplace; I owned the candidate-generation rewrite."
2. **The data problem first.** Every real project has one, and naming it signals maturity: label leakage through a join on `updated_at`, 30% duplicate listings, or a logging bug where the "impression" event fired on hover. Concretely: "we discovered 18% of positives were the same item re-uploaded by the same seller; deduping before training changed the offline–online gap from 4 points to 0.8."
3. **The modelling obstacle.** Distribution shift across locales, cold start on 60% of items, or a metric that disagreed with the business objective (offline NDCG up, GMV flat — because the gain was in queries that never convert).
4. **The operational obstacle.** Backfill of 2 years of features took 9 hours; the feature store's TTL did not match the training window; retraining weekly broke because the schema drifted.
5. **Result with a caveat.** "Contact rate +7% on the treatment slice, but only in the top-3 markets — the long-tail markets needed a separate model, which is the follow-up work."
6. **What you'd do differently.** One honest sentence: "I would have built the eval set before the model."

Avoid: a project with no obstacles ("it went smoothly" signals you did not measure), unquantified impact, and blaming teammates. The obstacle should be a *system* property, not a person.

**中文要点**：重点讲障碍而非成果。结构：背景两句→**数据问题**（标签泄漏/重复/上报 bug，带数字）→建模问题（分布漂移、冷启动、离线指标与业务背离）→工程问题（回填、TTL、schema 漂移）→带保留条件的结果→"我会先建评测集"。别说"很顺利"。

---

### 25. Tell me about a time you drove a significant result through ambiguity, and a time you were wrong.

**Anchor answer**

Two stories, opposite shapes — the second is the discriminator, because most candidates have a rehearsed failure and no evidence of what changed.

**Story A — ambiguity.** The signal is that you **created the definition of done**, not that you worked hard.
- Situation: "We had 'the ranking quality feels bad' and no metric, no owner, no labels."
- What you did: built a 500-query judgement set in a week, defined the primary metric (and the guardrail), shipped a baseline in two weeks so the team had a number to argue about. Named the two things you deliberately did *not* do.
- Result: quantified — "the judgement set became the release gate; three subsequent launches used it."
- Lesson: ambiguity is usually a **missing measurement**, not a missing plan.

**Story B — being wrong.** Structure: a belief, the evidence that killed it, the cost, the behaviour change.
- "I argued we should replace the GBDT ranker with a two-tower model. I ran the offline eval, won the argument, and shipped it. Online, latency rose 40% and revenue was flat — the gain was confined to queries where the ranker was not the bottleneck."
- Then: what you did (rolled back within a week), what it cost (one engineer-quarter), and the durable change — "since then I instrument the *serving* path before the model, and I write the rollback criterion before launch."

Avoid the fake-weakness failure ("I care too much"). The best version is a **technical** judgment that was wrong, with a real cost, and a change that persists. Expect the probe: "what did you do when you found out?"

**中文要点**：两段故事形状相反。模糊中拿结果，靠的是**你自己定义了完成的度量**（一周建 500 条标注集、两周出基线）；"我错了"要讲技术判断错、真实代价、一周内回滚，以及此后固化的行为（先测服务链路、上线前写明回滚标准）。别用"我太追求完美"这类假弱点。

---

### 26. Tell me about maintaining a production ML pipeline. Why Meta?

**Anchor answer**

Two halves; the first must sound like ownership of a **live system**, the second must be specific to Meta's actual problems.

**Maintaining a pipeline** — talk about invariants you defended, not tools you used:

| Concern | Concrete practice |
| :--- | :--- |
| Data freshness | SLA per table; alert on **lag**, not on failure — silent staleness is the dangerous case |
| Schema drift | contract tests on upstream inputs; a new column should break CI, not the model |
| Training/serving skew | one shared feature-transform code path, plus a parity check on sampled live requests |
| Silent quality decay | daily PSI on inputs and prediction distribution; label-based metrics lag by days |
| Retraining & cost | scheduled + triggered retrain, canary with one-click rollback, per-stage compute accounting |

One war story beats the table: "an upstream team changed a null-handling default; our feature silently became 0 for 12% of rows, and offline metrics did not move because training used the backfilled value. We caught it via a live-vs-batch parity check." That proves you have been on call.

**Why Meta** — three specific reasons, each tied to your evidence:
1. **Scale of the recommendation surface** — ranking decisions made billions of times a day, where the offline/online gap is the core technical problem.
2. **Open model work** — Llama-lineage research and the willingness to publish, which matters if you want your work reproducible and reused.
3. **The named team's problem** (e.g. cold start in Marketplace, integrity in ranking) and what you would want to learn there.

Avoid generic praise ("great culture, impactful work"). Tie each reason to something you have already done.

**中文要点**：前半讲"你守住了哪些不变量"：数据新鲜度按延迟告警、schema 契约测试、训练/服务同一条特征代码 + 线上-批处理一致性校验、PSI 监控、可一键回滚的 canary 与按阶段算成本。**再讲一个真实事故**（上游改空值默认导致 12% 特征变 0）。Why Meta 要三条具体理由并各自挂到你的经历上。

---

## 附：这 26 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 编码关（初筛） | 1（前缀积）、2（最小窗口）、3（序列化）、5（拓扑）、7（正则 DP） |
| 推荐系统纵深 | 15（Reels）、16（Feed 序列）、17（广告与增量评估）、18（Stories） |
| 模型能力 | 10（Llama 架构）、11（千卡训练）、12（70B 服务成本） |
| 校准与指标判断力 | 9（AUC 涨收入不涨）、17（eCPM 加权）、21（评测体系） |
| 数据与合规 | 19（Marketplace 长尾）、22（有害内容）、20（翻译术语） |
| 工程直觉 | 8（为什么堆不对）、13（陌生代码库排障）、26（生产管道） |
| 行为面（Jedi 轮） | 24（数据+ML 障碍）、25（模糊与认错）、26 后半（Why Meta） |
