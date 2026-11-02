# 37 · Together AI

> 7 题。出处：`README.en.md` § Together AI。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Write the server-side handler for streaming token generation. Handle client disconnects correctly.

**Source**: https://outcomeschool.com/blog/how-does-token-streaming-work

```python
import asyncio, orjson

async def stream_tokens(request):
    gen = engine.generate(request)              # async generator of token ids
    resp = web.StreamResponse(headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",              # nginx must not buffer
    })
    await resp.prepare(request)
    try:
        async for tok_id in gen:
            if request.transport is None:       # transport gone
                break
            chunk = {"choices": [{"delta": {"content": decode(tok_id)}}]}
            await resp.write(b"data: " + orjson.dumps(chunk) + b"\n\n")
            await resp.drain()                  # backpressure: don't buffer in memory
        await resp.write(b"data: [DONE]\n\n")
    finally:
        await gen.aclose()                      # ALWAYS: frees the KV slot
    return resp
```

**What actually matters**

1. **Cancellation must be structural.** Wrap the engine call in its own task and cancel it in `finally`; a client that closes the tab otherwise leaves an entry in the batching scheduler forever, and those leaked entries are how "our GPU is full but QPS is normal" incidents happen. In vLLM/TPI terms: abort the request so the sequence is removed on the *next* scheduler step, not at the end of generation.
2. **Detect disconnects at the write**, not with a polling loop. `drain()` raises/returns immediately once the peer is gone; an explicit `asyncio.wait` on the transport closing is more reliable than sniffing `is_closing()`.
3. **Turn off every buffering layer** — proxy buffering turns a token stream into one blob and the client sees 0 tokens until the end. Also disable gzip on `text/event-stream`.
4. **Send an initial comment (`:\n\n`)** to flush headers early, so the client's TTFT measurement is honest.
5. **Flush on sentence/whitespace boundaries when the client is a tokenizer UI**, not per token — per-token flushes multiply syscalls and dominate at small batch.
6. **Account usage in the `finally`**, because on cancel you still burned prompt tokens and must bill them.

**中文要点**：断连必须**结构化取消**（`finally` 里 `aclose`/abort，让调度器下一步移除序列），否则泄漏的 KV 槽会让 GPU 满而 QPS 正常；记得关掉代理缓冲和 gzip，并在 finally 里结算已消耗的 token。

---

### 2. Design the scheduler for a continuous-batching inference engine.

**Source**: https://outcomeschool.com/blog/continuous-batching-in-llms · https://outcomeschool.com/blog/how-does-vllm-work

Continuous batching means the batch is **recomputed every decode step**: finished sequences leave, waiting ones join. So the scheduler isn't a queue — it's a per-step admission controller over a memory budget.

| Phase | Decision | Binding constraint |
| :--- | :--- | :--- |
| 1. Retire | drop finished/aborted seqs, free blocks | free KV immediately, not next step |
| 2. Admit | pull from waiting, promote to running | **prefill budget in tokens**, not requests |
| 3. Evict | preempt lowest priority, recompute or swap | KV block pool exhausted |
| 4. Build batch | running seqs + chunked prefill | `max_num_seqs`, `max_num_batched_tokens` |
| 5. Execute | one fused forward | attention metadata per seq |

Choices that separate a good scheduler from a naive one:

1. **Decouple prefill and decode budgets.** A long prompt must not stall 200 decoders. Chunked prefill (fixed token budget per step, e.g. 512–2048) keeps ITL flat while still admitting work.
2. **Paged KV is the enabler.** Fixed-size blocks (16 tokens) with a per-sequence block table → no external fragmentation, and prefix sharing becomes refcounting.
3. **Preemption choice matters.** Recompute is cheap for short contexts, swap-to-host for long ones; pick per sequence.
4. **Fairness**: pure FCFS starves long generations behind a burst of short chats. Two lanes, or deficit round-robin on tokens served.
5. **Autotune the batch ceiling** per (model, GPU, TP degree) — it's a curve whose knee moves with context length.

Failure mode to name: admission by *request count*. Two 32k-prompt requests and thirty-two 200-token ones look identical to a counter and behave nothing alike.

**中文要点**：调度器是**每步重算的准入控制器**而非队列：先退役释放 KV、按 token 预算预填、满了就抢占（短的重算/长的换出），并解耦 prefill 与 decode 预算；按请求数准入是最典型的错。

---

### 3. Explain speculative decoding. When does it help, when does it hurt, and why adapt the speculator to live traffic?

**Source**: https://outcomeschool.com/blog/speculative-decoding · https://outcomeschool.com/blog/n-gram-speculation-in-llms

A cheap drafter proposes γ tokens; the target model verifies them **in one forward pass** and accepts the longest correct prefix (rejection sampling keeps the distribution exactly the target's). Speedup ≈ `accepted / cost_of_verify` — a bet that verification is cheaper than generation.

| Regime | Effect | Why |
| :--- | :--- | :--- |
| **Low batch / single stream** | big win (commonly 2–3× reported) | verify is memory-bound; extra tokens nearly free |
| **High batch, compute-saturated** | **hurts** | drafter FLOPs become pure overhead |
| Predictable text (code edits, extract-from-context, JSON) | high accept rate | drafter can copy from the prompt (n-gram lookup) |
| Creative, high-entropy text | low accept rate | you pay γ proposals for ~1 token |
| Very long context | draft cost grows | drafter attends over the same KV |

**Why adapt to live traffic**: accept rate is a property of the *workload*, and workloads drift by tenant and time of day. A static γ or drafter is tuned on an eval set that stops resembling production within weeks.

1. **Measure accept rate per request class** (endpoint, prompt-length bucket, temperature), not globally.
2. **Tune γ to acceptance α**: expected tokens per verify ≈ `(1-α^(γ+1))/(1-α)`; at α=0.5, γ=8 is wasted work.
3. **Switch the drafter by class**: n-gram lookup for copy-heavy code edits and RAG extraction; a distilled 1B drafter for open-ended chat; disable speculation once batch size crosses saturation.
4. **Gate on served metrics** — accepted-tokens-per-step and p99 ITL — with an automatic kill switch. The classic production mistake: it wins a single-stream benchmark and is a net loss at 40 concurrent requests.

**中文要点**：投机解码是"验证比生成便宜"的赌注：低并发/可预测文本大赚，**高并发算力饱和时会亏**；accept rate 是流量属性，所以要按请求类调 γ、按场景换 drafter（n-gram 抄上下文最适合代码编辑），并用接受率做自动熔断。

---

### 4. Price a dedicated endpoint: estimate cost per million output tokens for a 70B model, and explain the throughput-latency trade.

Work it as a chain, show your assumptions, and flag which ones are typical ranges rather than measured facts.

**Step 1 — memory.** 70B in FP8 ≈ 70 GB weights. On 8×H100-80GB (640 GB) that's one replica with room for KV; in BF16 it's 140 GB, still one node but KV headroom halves.

**Step 2 — throughput.** For a chat-shaped mix (say 1k prompt / 400 output tokens), an 8×H100 node with continuous batching typically lands in the low thousands of output tokens/sec aggregate, and the number moves a lot with context length ([典型经验值] — measure, don't quote). Take 3,000 tok/s at a comfortable batch and near-saturated compute.

**Step 3 — cost.**

```
node cost      ≈ 8 GPU × ~$2.5/GPU-hr ≈ $20/hr   [典型云价]
tokens/hour    = 3,000 tok/s × 3,600    = 10.8 M
cost/M tokens  = $20 / 10.8             ≈ $1.85 at 100% utilisation
at 60% realised utilisation             ≈ $3.1 / M output tokens
```

Add overheads: idle capacity you must hold for peak (the big one), failed/retried requests, prefill tokens (billed or not), and cold-start. A realistic dedicated price is **~$3–6 per million output tokens** with a floor set by reserved capacity, not by tokens consumed.

**The trade, stated properly:** batch size is a dial that moves you along a curve — larger batch raises throughput (cheaper $/token) and raises queueing delay (worse ITL/p99). So a dedicated endpoint should be priced on **reserved throughput + a latency SLA**, with burst above the reservation priced higher, because burst is served by degrading someone's latency. Two customers with the same token volume but 50 ms vs 500 ms ITL targets cost very differently to serve — price the SLO, not the token.

**中文要点**：先算显存（70B FP8≈70GB，8×H100 一节点），再按典型聚合吞吐 ~3k output tok/s、$20/卡时推得满负荷 ~$1.85/M、实际利用率 60% 后 ~$3/M；报价要卖**预留吞吐 + 延迟 SLA**，因为 batch 加大是拿 p99 换成本。

---

### 5. A customer's distributed training job on your GPU cluster gets 55% scaling efficiency at 64 nodes. Debug it.

55% at 64 nodes means nearly half of wall-clock is not doing math. Don't guess; **measure the step breakdown first**, because the fix depends entirely on where the gap is.

| Symptom in the trace | Root cause | Fix |
| :--- | :--- | :--- |
| Long gaps *before* backward, uneven per-rank | **data-loader stall** — loader can't feed 8 GPUs | pre-tokenised, pre-shuffled dataset; deeper async prefetch |
| Time in `ncclAllReduce`, poor overlap | bucket too small / no compute overlap | raise bucket size (~25–100 MB), gradient-ready bucketing |
| All-reduce grows superlinearly with nodes | **topology mismatch** — cross-rail traffic | pin the job to a rail-aligned IB island; check fabric counters |
| Periodic stalls every N steps | **checkpoint** blocking the hot path | async/sharded checkpoint, separate writer rank |
| One rank slow at the same layer | PP bubble / unbalanced stages | rebalance by measured FLOPs, not parameter count |
| Correct but slow all-reduce | wrong algorithm for the message size | per-size selection; comm dtype = compute dtype |

**Order of investigation**:

1. Profiler on 2 ranks → is the gap compute, comm, or *waiting*, and is the wait symmetric?
2. Step time at 8 / 16 / 32 / 64 nodes: a drop at one specific size is a **network tier boundary**, not the model.
3. `dataloader` vs `gpu` time on first and last rank — the classic 55% is a loader feeding only 55% of the step.
4. Ask whether anyone ever measured at 64 nodes, or inherited the number from a different model size.

One judgment call worth stating: past a point, chasing efficiency with more nodes is worse than fixing the sequence-length mix — bad padding alone can cost ~30%.

**中文要点**：55% 说明近一半时间不在算数学，先分步打点；最常见三类是 **dataloader 喂不满、all-reduce 未与反向重叠、跨 spine 的拓扑错配**；再依次查周期性 checkpoint 停顿、PP 气泡、通信算法选择。

---

### 6. Design a serverless inference platform serving 100+ open models on a shared GPU fleet.

**Source**: https://outcomeschool.com/blog/llm-inference-optimization

The hard problem isn't serving one model — it's **packing dozens of cold-or-warm models onto a fleet with thousands of tenants, while keeping cold-start acceptable and noisy neighbours quiet.**

| Layer | Decision | Why |
| :--- | :--- | :--- |
| **Catalog** | per-model artifact: quantised weights, tokenizer, engine config, max context, ABI-versioned kernels | one model = one versioned deployable unit |
| **Placement** | bin-pack by VRAM + KV footprint, tenant anti-affinity | weights must fit; **KV** is what actually OOMs |
| **Warm pool** | top-N popular models preloaded, rest on demand | 100+ models can't all be resident |
| **Loading** | mmap weights from local NVMe, layer-wise streaming | cold start = weight transfer, not process start |
| **Routing** | queue per model, admission by token budget | one model's spike must not starve another |
| **Isolation** | per-tenant token/GPU-second quotas; KV never shared across tenants | a shared-cache bug leaks prompts |
| **Scaling** | scale-to-zero for the long tail with a documented cold-start SLO | idle models otherwise eat the fleet |

Three failure modes to volunteer:

1. **Cold-start thundering herd** — a popular model gets evicted by a burst of tiny ones, then 200 requests all try to load it. Single-flight lock per model, queue behind it.
2. **Fragmentation** — mixed sizes leave GPUs with 30 GB free that fits nothing. Reserve whole-GPU placements for large models; pack only small ones.
3. **Version skew** — a kernel update changes numerics for one family. Pin engine version per artifact.

The governing metric: **fleet goodput** — tokens served inside SLO ÷ provisioned GPU-seconds. Optimise that, not utilisation.

**中文要点**：100+ 模型不能全常驻，靠"热门预载 + 冷启按需"；关键是**冷启惊群用单飞锁串行化**、按 VRAM+KV 装箱避免碎片、租户级配额与隔离，并用"满足 SLO 的有效吞吐/已购 GPU 秒"（goodput）而非利用率当北极星。

---

### 7. A customer wants to migrate from a proprietary frontier-model API to an open model. How do you run that engagement?

Treat it as a **de-risking engagement with an exit ramp**, not a port. These projects fail when someone swaps the endpoint, the demo looks fine, and quality collapse shows up three weeks later on traffic nobody tested.

| Phase | Work | Artifact that gates the next phase |
| :--- | :--- | :--- |
| **1. Baseline** | replay 2–4 weeks of *real* traffic against the current API; grade it | frozen eval set of 200–500 cases + incumbent scores (quality, refusal rate, latency, cost) |
| **2. Shape** | classify traffic: extraction/classification vs long-context reasoning vs agentic | routing mix — usually only 60–80% of traffic fits a mid-size open model |
| **3. Candidate** | 2–3 open models × quantisations on *their* data | per-segment table, not one average — averages hide the failing segment |
| **4. Close the gap** | prompt restructure, few-shot, constrained decoding or SFT | any SFT ships with its own eval and a retraining owner |
| **5. Run** | shadow → canary → ramp; dual-run for a fixed window | rollback to the proprietary API kept warm 30 days |

What makes it go well:

- **Say which segments should stay on the frontier model.** A hybrid router — cheap open model for 70% of traffic, frontier for the hard 30% — usually beats full migration on cost *and* quality, and is an easier yes.
- **Price the ops cost, not just the token cost.** Open model means they own serving, upgrades, capacity planning and eval drift. No team for that? A managed open-model endpoint is the honest recommendation.
- **Check licensing and data residency early** — commercial terms and acceptable-use clauses need a legal read.
- **Migrate the prompt, not just the model**: system-prompt behaviour, tool-call format and stop sequences all differ, and that's where "the model got worse" comes from.

**中文要点**：这是降风险工程不是端口替换——先用真实流量冻结评测集和分段画像，常见结论是**只迁 60~80% 的流量**、留下路由混合;要一并把运维/容量/评测漂移成本算进去，并保留 30 天可回滚到闭源 API。

---

## 附：这 7 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 流式服务真写过 | 1（断连与取消） |
| 调度器设计功底 | 2（continuous batching） |
| 推理优化判断 | 3（投机解码何时亏） |
| 商业/成本直觉 | 4（专享端点定价） |
| 分布式排障 | 5（55% 扩展效率） |
| 平台架构能力 | 6（100+ 模型共享集群） |
| 客户交付能力 | 7（闭源迁开源） |
