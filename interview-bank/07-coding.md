# 07 · Coding and Data Structures

> 12 题。出处：`README.en.md` § Coding and Data Structures。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Implement scaled dot-product attention with a causal mask, from scratch, in NumPy or PyTorch.

**Asked at**: Anthropic · Google DeepMind · Amazon · ByteDance · Meituan · DiDi
**Source**: https://outcomeschool.com/blog/math-behind-attention-qkv and https://outcomeschool.com/blog/causal-masking-in-attention

**Anchor answer**

Attention is a soft dictionary lookup: each query scores every key, scores are scaled by `1/√d_k` so the logit variance does not grow with head dimension, softmaxed into a distribution, and used to average the values.

**Reference implementation**

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)   # shift for numerical stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(q, k, v, causal=True):
    """q: (B,H,T,D)  k,v: (B,H,S,D)  ->  (B,H,T,D), weights (B,H,T,S)"""
    d_k = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d_k)        # (B,H,T,S)
    if causal:
        t, s = scores.shape[-2:]
        allowed = np.tril(np.ones((t, s), dtype=bool), k=s - t)  # offset covers S > T
        scores = np.where(allowed, scores, -np.inf)
    weights = softmax(scores)
    return weights @ v, weights
```

**Key points**

- **Mask before the softmax, with `-inf`.** Setting weights to zero *after* softmax leaves rows that no longer sum to 1; using `-inf` makes `exp` contribute exactly 0.
- **The mask offset is `S - T`, not 0.** During cached decoding `T=1` and `S=n`; query `t` may see keys `0..t+S-T`. Using `np.tril(..., k=0)` there would blind the query to everything but key 0.
- **Every causal row keeps at least one valid key** (`j <= i + S - T`), so no row is all `-inf` — an all-masked row yields `0/0 = nan` and silently poisons the whole batch.
- Scale **before** softmax; scaling after changes the distribution rather than the temperature.
- In production use `torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)`, which picks a flash/memory-efficient kernel; write it by hand to prove you know what the kernel does, not to ship it.
- `q @ kᵀ` is `O(T·S·D)` compute and materialises a `T×S` matrix — the reason long context needs tiling.

**中文要点**：缩放→加 `-inf` 因果掩码→softmax→乘 V；掩码偏移必须是 `S-T`（否则 KV cache 单步解码会瞎），且每行至少留一个可见 key，否则 `0/0` 出 nan。

---

### 2. Implement multi-head attention, then convert it to grouped-query attention.

**Asked at**: Google DeepMind · Mistral AI · Alibaba · ByteDance · Meituan
**Source**: https://outcomeschool.com/blog/multi-head-attention-in-transformers and https://outcomeschool.com/blog/grouped-query-attention

**Anchor answer**

MHA splits the model dimension into `h` independent heads. GQA keeps the queries at `h` heads but projects keys/values to only `kv < h` heads, and each KV head is shared by `h/kv` query heads. `kv = h` is MHA, `kv = 1` is MQA.

**Reference implementation**

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, causal=True):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).split(x.shape[-1], dim=-1)
        q, k, v = (z.view(b, t, self.h, self.d).transpose(1, 2) for z in (q, k, v))
        scores = q @ k.transpose(-2, -1) / self.d ** 0.5
        if causal:
            mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~mask, float("-inf"))
        p = scores.softmax(-1)
        y = (p @ v).transpose(1, 2).reshape(b, t, self.h * self.d)
        return self.out(y)

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.h, self.kv, self.d = n_heads, n_kv_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, self.h * self.d, bias=False)
        self.k_proj = nn.Linear(d_model, self.kv * self.d, bias=False)
        self.v_proj = nn.Linear(d_model, self.kv * self.d, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, causal=True):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.h, self.d).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.kv, self.d).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.kv, self.d).transpose(1, 2)
        k = k.repeat_interleave(self.h // self.kv, dim=1)   # the cache stays small
        v = v.repeat_interleave(self.h // self.kv, dim=1)
        scores = q @ k.transpose(-2, -1) / self.d ** 0.5
        if causal:
            mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~mask, float("-inf"))
        y = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, t, self.h * self.d)
        return self.out(y)
```

**Key points**

- **Convert by surgery, not from scratch**: keep `q_proj` as is, and for each group mean-pool (or keep the first) the `h/kv` MHA key/value head slices into one. Loss jumps; a short continued-pretrain recovers most of it.
- `repeat_interleave` is the readable version. It allocates, so real kernels instead use `k.view(b, kv, 1, s, d).expand(b, kv, h//kv, s, d)` or just index inside the attention kernel.
- Parameter count: MHA ≈ `4·d²`, GQA = `2·d² + 2·d²·kv/h`. At `h=32, kv=8` that is 2.5·d² — about **37% fewer** attention parameters.
- KV cache per token: `2·h·d_h` → `2·kv·d_h`, i.e. **4× smaller** at `h/kv=4`. That is the real reason GQA ships: decode is bandwidth-bound and the cache is what you stream.
- Sanity test: with `n_kv_heads == n_heads` and copied weights, GQA must reproduce MHA bit-for-bit — the cheapest correctness check for the reshape logic.

**中文要点**：GQA = Q 保持 h 头、K/V 只投 kv 头并组内共享；改造靠权重切片/均值合并再短训；收益是 KV cache 按 h/kv 缩小（h=32,kv=8 时 4×），参数量降约 37%。

---

### 3. Implement a KV cache and single-step decode.

**Asked at**: Moonshot AI
**Source**: https://outcomeschool.com/blog/kv-cache-in-llms

**Anchor answer**

Prefill processes the prompt and stores every layer's K and V. Decode then feeds **one token** and appends its K/V to the cache, so each step costs `O(S)` attention instead of recomputing `O(T²)`. What changes is arithmetic intensity: the matmuls become skinny, so decode is memory-bandwidth-bound and the cache itself is the payload.

**Reference implementation**

```python
import torch
import torch.nn as nn

class CachedAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.d = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cache=None):
        """x: (B, T, D). cache: (k, v) each (B, H, S, Dh), or None on prefill."""
        b, t, _ = x.shape
        q, k, v = self.qkv(x).split(x.shape[-1], dim=-1)
        q, k, v = (z.view(b, t, self.h, self.d).transpose(1, 2) for z in (q, k, v))
        if cache is not None:                        # append along the sequence axis
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        s = k.shape[2]
        scores = q @ k.transpose(-2, -1) / self.d ** 0.5
        if t > 1:                                    # only prefill needs the mask
            mask = torch.tril(torch.ones(t, s, dtype=torch.bool, device=k.device),
                              diagonal=s - t)
            scores = scores.masked_fill(~mask, float("-inf"))
        y = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, t, self.h * self.d)
        return self.out(y), (k, v)
```

Decode loop:

```python
def generate(model_layer, prompt, n_new=4, cache=None):
    with torch.no_grad():
        logits, cache = model_layer(prompt, cache)     # prefill
        for _ in range(n_new):
            nxt = logits[:, -1:].argmax(-1)            # greedy, one token
            logits, cache = model_layer(nxt, cache)    # decode step
    return cache
```

**Key points**

- Cache shape is `(B, H, S, D_h)` per layer, both K and V: per token `2·layers·H·D_h·bytes`.
- Do the memory arithmetic out loud: 32 layers, 32 heads, `D_h=128`, fp16 → `2·32·32·128·2 = 512 KiB` **per token**. An 8k context is ~4 GiB for a *single* sequence; that is why GQA (→128 KiB/token), paged attention and prefix sharing exist.
- `torch.cat` per step is `O(S)` copying, so a real server preallocates `max_len` and writes by index (`cache[:, :, pos] = k`); paged attention goes further and removes the contiguity requirement.
- For `t == 1` no mask is needed — the single query is the newest position and may attend to everything.
- Off-by-one bugs live in the mask offset `S - T`; the test above (prefill-then-step must equal a full forward) catches all of them.
- Eviction (sliding window / attention sinks) is a policy on top of the cache, not part of it.

**中文要点**：prefill 建缓存、decode 单步只追加 K/V；每 token 显存 `2·层·头·d_h·字节`（32 层 32 头 fp16 ≈ 512KiB），所以真正瓶颈是带宽与缓存容量；实现要点是预分配按位置写入 + 掩码偏移 `S-T`。

---

### 4. Implement BPE training and encoding from scratch.

**Source**: https://outcomeschool.com/blog/bpe-in-llms

**Anchor answer**

BPE training is a greedy compression loop: count adjacent symbol pairs weighted by word frequency, merge the most frequent pair into a new symbol, repeat until the vocabulary budget is hit. Encoding replays the merges **in the order they were learned** — that order is the whole model, not the merge set.

**Reference implementation**

```python
from collections import Counter

def _merge(symbols, a, b):
    out, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out

def train_bpe(text, vocab_size):
    """Character-level BPE; '</w>' marks end of word. Returns (merges, vocab)."""
    words = {tuple(w) + ("</w>",): c for w, c in Counter(text.split()).items()}
    merges, vocab = [], {ch for w in words for ch in w}
    while len(vocab) < vocab_size:
        pairs = Counter()
        for w, c in words.items():
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += c          # frequency-weighted, not per-occurrence
        if not pairs:
            break
        (a, b), count = pairs.most_common(1)[0]
        if count < 2:                        # a pair seen once buys nothing
            break
        merges.append((a, b))
        vocab.add(a + b)
        words = {tuple(_merge(list(w), a, b)): c for w, c in words.items()}
    return merges, vocab

def encode(text, merges):
    tokens = []
    for word in text.split():
        symbols = list(word) + ["</w>"]
        for a, b in merges:                  # learned order matters
            symbols = _merge(symbols, a, b)
        tokens.extend(symbols)
    return tokens

def decode(tokens):
    return "".join(tokens).replace("</w>", " ").strip()
```

**Key points**

- **Training is `O(merges × corpus)`** in this shape; production implementations keep a pair→positions index and update only affected words.
- **Counting must be word-frequency weighted.** Counting raw occurrences gives the same result on a corpus of unique words but is wrong on real text where "the" appears 10⁶ times.
- **Encoding applies merges in priority order**, not as a set: `("l","o")` before `("lo","w")` yields `low`, the reverse yields `l` + `ow`.
- Determinism: `Counter.most_common` ties break by insertion order. Fix the tie-break (e.g. lexicographic) or two runs disagree — a classic reproducibility bug.
- Real tokenizers pre-tokenise with a regex first (so BPE never merges across whitespace or punctuation) and use a **byte-level** alphabet so any Unicode string is encodable; char-level BPE on OOV text has no fallback.
- `</w>` is doing real work: it stops "low" inside "lower" from being the same token as the standalone word.

**中文要点**：训练 = 按词频加权统计相邻对 → 合并最高频对 → 重建词表；编码必须按"学到的顺序"而非集合应用合并；生产版还要加正则预分词、字节级字母表和确定性 tie-break。

---

### 5. Implement top-k, top-p and temperature sampling over a logits vector.

**Asked at**: Google DeepMind · Apple
**Source**: https://outcomeschool.com/blog/how-does-temperature-control-llm-output and https://outcomeschool.com/blog/how-do-top-k-and-top-p-sampling-work

**Anchor answer**

Temperature is a **distribution reshape** (`logits / T`), top-k is a **fixed truncation**, top-p is a **mass truncation**. They compose, and the order changes results, so pick one and document it.

**Reference implementation**

```python
import numpy as np

def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def sample_next(logits, temperature=1.0, top_k=None, top_p=None, rng=None):
    rng = rng or np.random.default_rng()
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:                       # greedy: T=0 is a division by zero
        return int(logits.argmax())
    logits = logits / temperature
    if top_k is not None and top_k < logits.size:
        cutoff = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < cutoff, -np.inf, logits)
    order = np.argsort(-logits)
    probs = softmax(logits[order])
    if top_p is not None and top_p < 1.0:
        cdf = np.cumsum(probs)
        keep = cdf - probs < top_p     # shortest prefix whose mass reaches top_p
        keep[0] = True                 # never truncate everything
        probs = np.where(keep, probs, 0.0)
        probs = probs / probs.sum()
    return int(order[rng.choice(probs.size, p=probs)])
```

**Key points**

- **`top_p` uses the prefix before the cumulative sum crosses the threshold** (`cdf - p < top_p`), which keeps the token that *crosses* it. Using `cdf < top_p` drops it and truncates too aggressively.
- **`T → 0` must be a special case.** Dividing by 0 gives `inf`/`nan`; the mathematical limit is argmax, so branch to greedy.
- `top_k = 1` and `top_p → 0` both degenerate to greedy — cheap unit tests.
- **Order matters for top-p**: temperature rescales the whole distribution, so `T` then `top-p` ≠ `top-p` then `T`. Top-k is scale-invariant (monotone), so only p is sensitive.
- `np.partition` is `O(n)`; sorting everything is `O(n log n)` and pointless for a 200k-token vocabulary — that is the difference between a fast sampler and a slow one.
- Renormalise after truncation or `rng.choice` raises; and keep `-inf` (not `-1e9`) so `exp` gives a true zero in float32.
- Sampling always needs a seeded, per-request RNG for reproducibility, or you cannot reproduce a bad output in a bug report.

**中文要点**：温度是重塑分布、top-k 定截断、top-p 按累积质量截断；`top_p` 要包含"跨越阈值"那个 token，`T=0` 必须走贪心分支，且温度与 top-p 的先后顺序会改变结果。

---

### 6. Implement an LRU cache with O(1) get/put, then add TTL.

**Asked at**: OpenAI · xAI · Alibaba

**Anchor answer**

LRU needs two things in O(1): a hash map for lookup and a doubly linked list for recency order. In Python, `OrderedDict` is exactly that pair, so the whole structure is ten lines and the interview is really about the edge cases: expiry, mutation, and concurrency.

**Reference implementation**

```python
import time
from collections import OrderedDict

class LRUCache:
    """O(1) get/put. TTL is lazy: expired entries are dropped on access."""

    def __init__(self, capacity, ttl=None, clock=time.monotonic):
        self.capacity, self.ttl, self.clock = capacity, ttl, clock
        self.data = OrderedDict()

    def _fresh(self, key):
        value, expires = self.data[key]
        if expires is not None and expires <= self.clock():
            del self.data[key]
            return None
        return value

    def get(self, key):
        if key not in self.data:
            return None
        value = self._fresh(key)
        if value is None:
            return None
        self.data.move_to_end(key)          # mark as most recently used
        return value

    def put(self, key, value):
        expires = None if self.ttl is None else self.clock() + self.ttl
        self.data[key] = (value, expires)
        self.data.move_to_end(key)
        while len(self.data) > self.capacity:
            self.data.popitem(last=False)   # evict least recently used

    def __len__(self):
        return len(self.data)
```

**Key points**

- `put` on an existing key must **refresh recency and TTL**, not just overwrite; forgetting the `move_to_end` makes a hot key evictable.
- **Lazy TTL means memory is not bounded by time.** Expired entries still occupy slots until touched; with a skewed key distribution you can hold thousands of dead entries. Fix by sweeping on `put` (bounded work), or keep a min-heap of `(expires, key)` and pop lazily — `O(log n)` per expiry, which breaks the pure O(1) claim, so state the trade-off.
- `ttl=None` must mean "never expires", not `clock() + None`.
- **Clock choice matters**: `time.monotonic()` is immune to NTP steps; `time.time()` in a cache causes mass expiry after a clock jump.
- Hit rate is the only metric that matters. LRU is pessimal for a scan that is larger than capacity (it evicts exactly what you will need next) — that is the argument for 2Q/ARC/SLRU in front of a database.
- Concurrency: Python's GIL does not make this atomic. Guard with a `threading.Lock` or shard by key hash; a lock per shard beats one global lock under contention.

**中文要点**：哈希表 + 双向链表（`OrderedDict`）即可 O(1)；坑在「put 已存在 key 要刷新 recency 与 TTL」、惰性过期不回收内存（需定期扫描或小根堆）、以及用 `monotonic` 防时钟跳变。

---

### 7. Implement a token-bucket rate limiter, then make it distributed.

**Asked at**: Anthropic · OpenAI · xAI · Cohere

**Anchor answer**

Token bucket: a bucket of `burst` tokens refilled at `rate` tokens/second; a request is admitted if it can afford its cost. Two numbers give you both **average rate** and **burst tolerance**, which fixed windows cannot express — a fixed window lets 2× the limit through at the boundary.

**Reference implementation (single process)**

```python
import time

class TokenBucket:
    def __init__(self, rate, burst, clock=time.monotonic):
        self.rate, self.burst, self.clock = rate, burst, clock
        self.tokens, self.updated = float(burst), clock()

    def allow(self, n=1):
        now = self.clock()
        self.tokens = min(self.burst, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False
```

**Distributed (atomic in Redis, one round trip)**

```python
import redis

LUA = """
local rate  = tonumber(ARGV[1]) / 1000          -- tokens per millisecond
local burst = tonumber(ARGV[2])
local cost  = tonumber(ARGV[3])
local t     = redis.call('TIME')                -- server clock, no client skew
local now   = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local state = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens, ts = tonumber(state[1]), tonumber(state[2])
if tokens == nil then                           -- first request: full bucket
  tokens, ts = burst, now
end
tokens = math.min(burst, tokens + math.max(0, now - ts) * rate)
local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end
redis.call('HMSET', KEYS[1], 'tokens', tokens, 'ts', now)
redis.call('PEXPIRE', KEYS[1], math.ceil(burst / rate) + 1000)
return {allowed, math.floor(tokens)}
"""

class RedisTokenBucket:
    def __init__(self, client, rate, burst, prefix="rl"):
        self.rate, self.burst, self.prefix = rate, burst, prefix
        self._script = client.register_script(LUA)

    def allow(self, identity, cost=1):
        allowed, _ = self._script(keys=[f"{self.prefix}:{identity}"],
                                  args=[self.rate, self.burst, cost])
        return bool(allowed)
```

**Key points**

- **Read-modify-write must be atomic.** Doing `GET` then `SET` from the app races and lets N servers each admit the full burst; the Lua script executes as one Redis command.
- **Use the server clock** (`TIME` inside the script) — client clocks across a fleet drift by seconds, which is a rate-limit bypass or a self-DoS.
- `PEXPIRE` is garbage collection for idle identities; without it the keyspace grows forever.
- Key per **identity** (`user`, `tenant`, `api_key`), not global: a global bucket lets one abuser starve everyone, and per-tenant buckets are also what you bill.
- Weigh `cost` ≠ 1 for expensive endpoints — one 100k-token request should not cost the same as a ping.
- Redis becomes the bottleneck at high QPS: batch checks, or do a **local pre-check with a shared lease** (token-bucket at the edge, coarse quota in Redis).
- Alternatives and when: fixed window (simple, boundary burst), sliding log (exact, memory-heavy), leaky bucket (smooths output, adds latency). Token bucket is the default because it is O(1) memory and allows designed bursts.

**中文要点**：令牌桶同时给平均速率和突发容量；分布式必须用 Lua 保证"读-改-写"原子、用 Redis `TIME` 避免多机时钟漂移、`PEXPIRE` 回收空闲 key，且按身份分桶而不是全局单桶。

---

### 8. Write an async batch processor over an API with concurrency limits, retries with jitter and error isolation.

**Asked at**: Anthropic · Perplexity

**Anchor answer**

Three guarantees to design for independently: **backpressure** (never more than N in flight), **retry only transient failures** (with jittered backoff so retries do not synchronise into a thundering herd), and **isolation** (one bad item never cancels the batch).

**Reference implementation**

```python
import asyncio
import random

async def process_batch(items, call, *, concurrency=8,
                        max_attempts=4, base_delay=0.05):
    """Apply async `call` to every item. Returns (results, failures)."""
    sem = asyncio.Semaphore(concurrency)

    async def one(item):
        async with sem:                      # backpressure lives here
            for attempt in range(1, max_attempts + 1):
                try:
                    return True, await call(item)
                except (asyncio.TimeoutError, ConnectionError) as exc:
                    if attempt == max_attempts:
                        return False, exc
                    delay = base_delay * 2 ** (attempt - 1)
                    await asyncio.sleep(delay * (0.5 + random.random()))  # full jitter
                except Exception as exc:     # terminal: do not retry
                    return False, exc

    outcomes = await asyncio.gather(*(one(i) for i in items))
    ok = [value for good, value in outcomes if good]
    failed = [value for good, value in outcomes if not good]
    return ok, failed
```

**Key points**

- **`async with sem` inside the task, not around `gather`.** Wrapping `gather` just serialises the batch; the semaphore must gate each attempt.
- **Catch terminal errors separately.** A 400/validation error retried four times multiplies cost and delays the report; retry only timeouts, connection errors and 429/5xx.
- **Full jitter** (`delay * U(0,1)`) beats fixed backoff: without it, all clients that failed together retry together. Exponential *without* jitter is the classic outage amplifier.
- **Error isolation via return values, not exceptions.** `gather` without isolation cancels siblings; returning `(ok, value_or_exc)` keeps the batch alive. If you prefer exceptions, use `gather(..., return_exceptions=True)`.
- **Timeout per attempt** (`asyncio.wait_for(call(item), t)`) is required, otherwise one hung connection holds a semaphore slot forever.
- Real API work also needs: a token budget (not just request count), 429 `Retry-After` awareness, idempotency keys for side-effecting calls, and streaming results as they finish rather than buffering the whole batch.
- Results come back in **item order** with `gather`; pair each with its input id so failures are attributable.

**中文要点**：信号量放在单个任务内做背压；只重试超时/连接/429/5xx，其余立即失败；退避必须 full jitter；错误以返回值隔离而非抛异常中断整批，另加 per-attempt 超时。

---

### 9. Write a streaming SSE/JSON parser that handles arbitrary chunk boundaries.

**Asked at**: Cohere · DiDi · JD.com
**Source**: https://outcomeschool.com/blog/how-does-token-streaming-work

**Anchor answer**

The only correct design is a **stateful buffer plus a boundary search**. A network chunk can split anywhere: mid-field, mid-line, mid-`\r\n`, even mid-UTF-8 codepoint. Never parse a chunk; parse complete frames and keep the remainder.

**Reference implementation**

```python
import json
import re

_BOUNDARY = re.compile(r"\r\n\r\n|\n\n|\r\r")

class SSEParser:
    """Incremental Server-Sent Events parser; accepts any chunk boundary."""

    def __init__(self):
        self._buf = ""

    def feed(self, chunk):
        self._buf += chunk
        events = []
        while True:
            m = _BOUNDARY.search(self._buf)
            if not m:
                break
            block = self._buf[:m.start()]
            self._buf = self._buf[m.end():]
            event = self._parse(block)
            if event is not None:            # comment-only block dispatches nothing
                events.append(event)
        return events

    @staticmethod
    def _parse(block):
        event = {"event": "message", "data": "", "id": None, "retry": None}
        data = []
        for line in block.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            if not line or line.startswith(":"):     # comment / keep-alive
                continue
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "data":
                data.append(value)
            elif field == "event":
                event["event"] = value
            elif field == "id":
                event["id"] = value
            elif field == "retry":
                event["retry"] = int(value)
        if not data:                         # spec: no data => no dispatch
            return None
        event["data"] = "\n".join(data)
        return event

def parse_stream(chunks):
    """chunks: iterable of str pieces -> list of decoded JSON payloads."""
    parser, out = SSEParser(), []
    for chunk in chunks:
        for event in parser.feed(chunk):
            if event["data"] == "[DONE]":
                return out
            if event["data"]:
                out.append(json.loads(event["data"]))
    return out
```

**Key points**

- **Search for `\r\n\r\n | \n\n | \r\r`, not `\n\n`.** A server using bare `\r` or CRLF would never dispatch with an `\n\n`-only splitter. Do not normalise the raw buffer eagerly — a trailing `\r` might be the first half of `\r\n`.
- **Multi-line `data:` fields are joined with `\n`**, not concatenated. JSON error messages and multi-part payloads depend on it.
- **Comment lines (`:` prefix) are heartbeats, not events.** Dispatching them as empty events breaks clients that expect a payload.
- **Decode bytes incrementally** (`codecs.getincrementaldecoder("utf-8")`), otherwise splitting a 3-byte emoji across chunks raises `UnicodeDecodeError` — the bug that only shows up with non-ASCII output.
- **Never `json.loads` a partial buffer**; parse per complete data payload and let a `JSONDecodeError` fail loudly instead of silently dropping tokens.
- Keep the unfinished tail bounded: a malicious or broken server sending no delimiter grows the buffer without limit, so cap it and abort.
- `[DONE]` is OpenAI-specific, not SSE; Anthropic uses `event: message_stop`. Handle sentinels at the application layer.

**中文要点**：核心是「缓冲 + 找完整帧边界」；要同时认 `\r\n\r\n`/`\n\n`/`\r\r`，多行 data 用 `\n` 拼接，注释行不是事件，字节流必须增量 UTF-8 解码，绝不对半截 buffer 做 `json.loads`。

---

### 10. Implement a text chunker with overlap that never splits a semantic unit.

**Asked at**: Harvey
**Source**: https://outcomeschool.com/blog/chunking-strategies-for-rag

**Anchor answer**

Two invariants fight each other: **a hard size cap** and **never splitting a sentence**. You cannot always satisfy both, so the resolution is: split at the largest allowed boundary, and when a single unit exceeds the cap, keep it whole and let the chunk be oversized. A truncated table or half a code block retrieves worse than a slightly large chunk.

**Reference implementation**

```python
import re

_SENT = re.compile(r"(?<=[.!?。！？])\s+")

def _units(text):
    """Yield sentence-sized units, preserving paragraph breaks as separators."""
    for para in re.split(r"\n\s*\n", text):
        for sent in _SENT.split(para.strip()):
            sent = sent.strip()
            if sent:
                yield sent + " "

def chunk_text(text, max_chars=800, overlap_chars=120):
    units = list(_units(text))
    chunks, i, n = [], 0, len(units)
    while i < n:
        j, size = i, 0
        while j < n and (j == i or size + len(units[j]) <= max_chars):
            size += len(units[j])          # first unit always fits, even if oversized
            j += 1
        chunks.append("".join(units[i:j]).strip())
        if j >= n:
            break
        k, overlap = j, 0
        while k - 1 > i and overlap + len(units[k - 1]) <= overlap_chars:
            k -= 1                         # walk back whole sentences only
            overlap += len(units[k])
        if k < j and overlap + len(units[j]) > max_chars:
            k = j                          # overlap must not starve the next chunk
        i = k                              # i always advances, j > i
    return chunks
```

**Key points**

- **Overlap is measured in whole units, not characters.** A character-level slice of the tail re-splits a sentence and reintroduces the exact problem you were avoiding.
- **The overlap walk must stop at `k-1 > i`, and yield when it would starve the next chunk.** If the tail can reach the chunk start you re-emit the same text forever; if the tail plus one new unit busts the cap you emit a chunk that is *only* duplicated text. Both are progress bugs, not performance bugs.
- **Oversized units stay whole.** Log them; a 4000-char paragraph is a data problem, and hard-splitting it mid-table destroys retrieval quality.
- The naive `text[i:i+n]` chunker fails on: code fences, lists, tables, and any language without spaces. Pre-split fenced blocks and tables as atomic units before sentence splitting.
- Unit choice changes retrieval behaviour: sentence-level chunks retrieve precisely but lose context; paragraph-level chunks carry context but dilute embeddings. Overlap is a hedge against boundary loss, not a substitute for good boundaries.
- Real pipelines attach `{doc_id, section_path, char_span}` to every chunk — without span offsets you cannot cite or debug a bad retrieval.

**中文要点**：句/段为最小单元，超长单元整块保留（宁大勿裂）；重叠必须按整句回退且保证 `k > i` 否则死循环；代码块/表格要当原子单元，每块带 doc_id 与字符区间。

---

### 11. Implement cosine similarity search over embeddings, then explain why you would not ship it.

**Source**: https://outcomeschool.com/blog/how-does-a-vector-database-work

**Anchor answer**

Cosine is a dot product on **L2-normalised** vectors, so the whole index is one matrix multiply plus a partial sort. That is also the reason it is a demo and not a service.

**Reference implementation**

```python
import numpy as np

def normalize(m):
    m = np.asarray(m, dtype=np.float32)
    return m / np.linalg.norm(m, axis=-1, keepdims=True)

class FlatIndex:
    def __init__(self, dim):
        self.matrix = np.zeros((0, dim), dtype=np.float32)

    def add(self, embeddings):
        self.matrix = np.vstack([self.matrix, normalize(embeddings)])

    def search(self, query, k=5):
        if not len(self.matrix):
            return []
        sims = self.matrix @ normalize(query)        # cosine == dot after normalising
        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]      # O(n), then sort only k
        idx = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[i])) for i in idx]
```

**Why I would not ship it**

| Problem | Consequence |
| :--- | :--- |
| `O(N·d)` per query, full matrix in RAM | 10M × 768 floats ≈ 30 GB; latency grows linearly with corpus |
| No incremental delete/update story | `np.vstack` copies everything; a delete means rebuilding |
| No metadata filtering | "top-k within tenant X, after date D" cannot be pushed down, so you over-fetch and post-filter |
| No durability, replication or sharding | one process restart loses the index |
| No recall control | exact, but you cannot trade recall for latency as you scale |
| Normalisation is a silent contract | forget it and cosine quietly becomes dot product, ranking longer-norm vectors higher |

What replaces it: an ANN index (HNSW for low latency, IVF-PQ for memory), with **recall@k measured against this flat index as ground truth**, plus filtering pushed into the index, plus sharding by tenant. Keep the flat implementation forever as the oracle for evaluating the real one.

**Key points**

- **Normalise once at ingest**, not per query — otherwise cosine silently becomes dot product.
- `argpartition` is `O(n)`; sorting the whole corpus to take five neighbours is a common and fatal habit.
- A zero vector normalises to `nan`; guard it or one bad embedding poisons every result.
- Keep this as the **recall@k oracle** for the ANN index you actually ship.

**中文要点**：先归一化再用点积等价余弦；`argpartition` 取 top-k 再排序。不上线因为 O(N·d)、无增量更新/删除、无元数据过滤、无持久化复制分片——但它要留作 ANN 索引的 recall 基准。

---

### 12. Implement a minimal agent loop with tool dispatch, error handling and a step budget.

**Asked at**: Cognition · ByteDance · Tencent · Meituan
**Source**: https://outcomeschool.com/blog/ai-agent-loop

**Anchor answer**

The loop is small; the correctness is in three places: **tool errors become observations** (so the model can recover) instead of exceptions, **repeated calls are detected** (the model forgets it already tried), and **the budget returns a structured status** (so callers can tell "finished" from "gave up").

**Reference implementation**

```python
import json

def run_agent(llm, tools, task, max_steps=8):
    """llm(messages) -> {"content": str|None,
                         "tool_calls": [{"id","name","args"}]}"""
    messages = [{"role": "user", "content": task}]
    seen = set()
    for step in range(max_steps):
        reply = llm(messages)
        calls = reply.get("tool_calls") or []
        messages.append({"role": "assistant", "content": reply.get("content"),
                         "tool_calls": calls})
        if not calls:                                  # no tool call => final answer
            return {"status": "done", "answer": reply.get("content"),
                    "steps": step + 1}
        for call in calls:
            name, args = call["name"], call.get("args", {})
            signature = (name, json.dumps(args, sort_keys=True))
            if signature in seen:                      # idle-spin guard
                messages.append({
                    "role": "tool", "tool_call_id": call["id"],
                    "content": "error: repeated identical call; change approach or stop",
                })
                continue
            seen.add(signature)
            fn = tools.get(name)
            try:
                if fn is None:
                    raise KeyError(f"unknown tool {name}")
                content = json.dumps(fn(**args), default=str)
            except Exception as exc:                   # error as observation
                content = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": content[:4000]})
    return {"status": "budget_exhausted", "answer": None, "steps": max_steps}
```

**Key points**

- **Every step must append exactly one assistant message** even when `content` is `None`; dropping it desynchronises `tool_call_id` pairing and the next request is rejected by the API.
- **Errors are data.** `except` + JSON payload keeps the loop alive and lets the model correct course; raising ends the turn with no information.
- **Idle-spin guard is the cheapest win.** Hashing `(name, args)` catches the most common failure mode: the model re-issues the same failing call.
- **Truncate tool output** (`content[:4000]`) and persist the full result to a file, returning a path — unbounded observations are how a loop blows its context window in three steps.
- **Return a status enum,** never an empty string: `done` / `budget_exhausted` / (in production) `cancelled`, `blocked`.
- Production additions, in this order: per-tool timeouts, retryable-vs-terminal classification, cost/token budget alongside the step budget, and a wall-clock deadline.

**中文要点**：循环本体很小，正确性在三处——工具异常转成 observation、用 `(name, args)` 哈希检测空转、预算耗尽返回结构化状态；另外每步必须补一条 assistant 消息，工具输出要截断。

---

## 附：这 12 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 基础是否真会 | 1（掩码偏移）、5（top-p 边界）、4（合并顺序） |
| 推理优化经验 | 3（KV cache 显存账）、2（GQA 收益与权重改造） |
| 工程手感 | 8（jitter / 错误隔离）、9（分块边界）、6（惰性 TTL） |
| 分布式 | 7（Lua 原子性）、11（为什么不上线） |
| Agent 岗 | 12（循环三件事）、10（不裂语义的切分） |
