# 36 · Groq

> 12 题。出处：`README.en.md` § Groq。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Our compiler statically schedules every instruction and every chip-to-chip transfer. What does that compiler need to know that an NVCC-style compiler does not, and what breaks when it's wrong?

An NVCC-style compiler schedules *one kernel on one SM* and lets the hardware resolve everything else at runtime: warp scheduling, cache misses, memory latency hiding, and (with NCCL) collective ordering. A statically scheduled whole-model compiler must resolve **all of that at compile time**.

What it needs that NVCC doesn't:

| Knowledge | Why it's mandatory |
| :--- | :--- |
| **Full-graph SRAM residency & liveness** | there's no HBM to spill to — if the plan doesn't fit, it doesn't run |
| **Cycle-accurate latencies** for every op, DMA and link hop | the schedule is a fixed pipeline; a 3-cycle error accumulates into a stall |
| **Exact shapes and iteration counts** | loops must be unrolled to a known trip count |
| **All-to-all / reduction routing** across chips | which chip sends what, when, over which link, with what buffering |
| **Worst-case, not average** operand arrival | no reorder buffer to hide a late tile |

What breaks when it's wrong — and this is the part interviewers probe:

1. **Silent numerical corruption, not a crash.** A mis-scheduled transfer reads a tile one generation stale; the model still emits fluent text with subtly wrong logits. You find it by hashing intermediates per layer, not by watching for exceptions.
2. **Deadlock** when two chips each wait for the other's buffer slot.
3. **Anything data-dependent is unrepresentable**: dynamic shapes, MoE routing, speculative-decoding accept length, early exit. These must be compiled to a worst-case schedule (compute all experts) or handled by a runtime fallback, and the fallback is where latency guarantees die.

The framing that lands: *on a GPU, the hardware is a scheduler; here the compiler is the scheduler, so every runtime freedom you forgot to enumerate becomes a correctness bug.*

**中文要点**：静态调度等于把硬件的乱序/缓存/路由全部搬到编译期，必须知道整图 SRAM 驻留、逐周期延迟、精确 shape 和跨片通信路由；错了不是崩溃而是**静默数值错**（读到上一代 tile），data-dependent 的 MoE/动态 shape 只能按最坏情况编译。

---

### 2. Design the IR and pass pipeline for a compiler targeting a spatial dataflow accelerator. Where does the memory-residency decision live, and why?

Four levels, each with one job — resist the urge to fuse levels, because the diagnostics get unusable.

| Level | Represents | Key invariant |
| :--- | :--- | :--- |
| **L1 Graph IR** | ops, tensors, dtypes, shapes; no layout | shape/dtype correctness |
| **L2 Tile IR** | fused regions, tile shapes, loop nests, explicit `load/store` to named buffers | every tensor has a declared home |
| **L3 Schedule IR** | cycle-stamped ops, DMA, link transfers, buffer versions, barriers | single-assignment per cycle per resource |
| **L4 Machine IR** | per-chip instruction stream + routing tables | bit-exact vs the functional model |

Pass pipeline: `canonicalize → shape/dtype infer → fuse → tile+layout select → **partition across chips** → **memory residency/assign** → schedule → route → register-allocate/verify → emit`.

**Where residency lives: between L2 and L3, and it must be solved jointly with partitioning.** The reason is that residency is a *lifetime* problem over the whole graph: whether tile T can stay in SRAM across a fusion boundary depends on what else is live and on which chip T ended up. Solve partitioning first and you get locally-optimal, globally-infeasible plans — the classic failure is a chain that fits if split at layer 40 but the partitioner split at 32 for load balance, and the KV/activation no longer fits.

Practical structure: residency as a constraint problem (graph-colouring / bin-packing over live ranges, capacity = SRAM bytes per chip), with partitioning feeding candidate cuts and a feasibility check looping back. Keep the solver **deterministic** — same graph must yield the same plan, or you can't diff two compiler versions.

**中文要点**：四层 IR（图→tile→调度→机器），**内存驻留决策必须放在 tile 与调度之间，并与跨片切分联合求解**——因为它是全图生命周期问题；先切分后定驻留会得到"局部最优、全局不可行"的方案，且求解器必须确定性。

---

### 3. Write the host-side runtime that feeds a deterministic accelerator across many chips. What is genuinely hard about it?

The runtime's contract looks trivial — "put the next batch of work on the device before the current one retires" — and every hard part comes from the device having **no ability to adapt**.

```python
class HostRuntime:
    def __init__(self, dev, depth=4):
        self.dev, self.depth = dev, depth
        self.inflight = collections.deque()   # (slot, seq, token_budget)

    def step(self):
        while len(self.inflight) < self.depth and self.dev.has_free_slot():
            req = self.scheduler.peek()       # admission control decides, not the device
            if req is None:
                break
            slot = self.dev.alloc_slot()
            self.dev.submit(slot, req.compile_program(), req.kv_pages())
            self.inflight.append((slot, req.seq, req.budget))

    def drain(self, timeout):
        for slot, seq, _ in list(self.inflight):
            tok = self.dev.read_result(slot, timeout)   # deadline, not blocking forever
            ...
```

What's genuinely hard:

1. **Jitter is the enemy, not throughput.** A GC pause or a page fault on the host introduces exactly the bubble the static schedule cannot absorb. You need a preallocated slab, no allocation on the hot path, and ideally a poll-mode loop.
2. **Admission control replaces dynamic batching.** The device can't reorder; the host must predict what fits in the next slot. Over-admit → missed deadlines; under-admit → idle silicon.
3. **Weight residency is a global resource.** Loading a second model evicts the first; the runtime must schedule *model* swaps, not just requests.
4. **Failure isolation across chips**: one chip's fault must cancel and replay a token on a spare, which means every step must be replayable from host-side state — which means the device must be side-effect free per step.
5. **Clock/latency accounting**: you need a per-step cycle budget to prove you met p99, and that budget must be measured on-device, not inferred from host timestamps.

**中文要点**：难点不在吞吐而在**抖动**——GC/缺页会直接变成静态调度无法吸收的气泡;主机必须自己做准入控制代替动态 batching、把权重驻留当全局资源调度，并保证每个 step 可重放以做坏片隔离。

---

### 4. A model passes bit-exact against the functional simulator on one chip but produces wrong output at rack scale. How do you find it?

**At one chip there is no communication, so the bug is almost certainly in the multi-chip layer** — routing, partitioning, or reduction order. Assume the math is right and the *topology* is wrong until disproved.

Bisect two axes at once:

| Axis | Procedure |
| :--- | :--- |
| **Scale** | 1 → 2 → 4 → 8 → one rack. Find the smallest divergent configuration. 2-chip failures are contraction/layout; rack-only are routing or clock. |
| **Depth** | Dump per-layer activations from sim and hardware, hash each tensor, find the **first** mismatch. A mismatch at layer k with identical input means k's inputs were mis-stamped, not computed wrongly. |

Then the things that only appear with wiring:

1. **Boundary tiles**: divergence confined to the first/last row = halo or split-K edge off-by-one.
2. **Reduction order**: split-K reduces in a different order across chips → last-bit differences that amplify over 80 layers. If it passes tolerance but fails bit-exact, this is it.
3. **Position dependence**: rotate the shard map one position and rerun. Failure follows the shard → layout bug; follows the chip → bad chip or bad link.
4. **Link health**: ECC counters, retransmits, clock skew — silent retransmission changes arrival timing.
5. **Deterministic replay** of a failing token from the host step log, then single-step the schedule.

The habit that makes this fast: emit a **per-layer per-chip hash manifest** in debug builds, so "find the first diverging tensor" is a two-file diff.

**中文要点**：单卡无通信，所以先怀疑**多卡层**（路由/切分/归约顺序）；同时沿规模（1→2→4→8→整机架）和深度（逐层 hash 找首个不一致）二分；重点看边界 tile、split-K 归约顺序、挪动分片映射后错误是否跟着走。

---

### 5. An LPU has no HBM at all, just on-die SRAM. Redo the decode roofline argument for that machine and tell me what changes.

**Source**: https://outcomeschool.com/blog/how-does-an-lpu-work

On a GPU, decode at batch 1 has arithmetic intensity ≈ 2 FLOP/byte, so it sits far below the ridge point: time per token ≈ `weights_bytes / HBM_bandwidth`. That's why batching exists — it moves you right along the x-axis until you hit the compute roof.

Remove HBM and the whole shape of the argument changes:

| Term | GPU | SRAM-only |
| :--- | :--- | :--- |
| Weight source | HBM, ~3 TB/s class | on-die SRAM, ~80 TB/s per chip (public figure for Groq's part) |
| Time per token per chip | `W/BW_hbm` | `W_slice/BW_sram` ≈ 230 MB / 80 TB/s ≈ **~3 µs** |
| Capacity | 80 GB per device, generous | **230 MB — the binding constraint, not bandwidth** |
| Ridge point | needs batch ~100+ | reached at batch ≪ 10 |

What changes, stated as three consequences:

1. **Capacity, not bandwidth, is the roofline.** A 70B model at 8-bit is ~70 GB, so it *must* be sharded across ≥300 chips. The design point is now "does the whole thing fit and route", not "how many bytes do I move".
2. **The interconnect replaces HBM as the bound.** Per-token latency becomes roughly per-chip compute (µs) plus pipeline traversal and activation exchange across the fabric — the network's latency and hop count start dominating.
3. **Batching stops paying for itself early.** Weights are resident and re-read at SRAM speed, so batch=1 already approaches the compute roof; extra batch mostly fills array utilisation, and it's capped by how much SRAM you can spare from weights for KV.

**中文要点**：去掉 HBM 后 roofline 的约束从"带宽"换成"**容量**"：70B/8bit 必须切到 300+ 卡，瓶颈变成片间互连延迟；单 chip 每 token 约 3 µs 量级（230MB / 80TB/s），batch=1 就接近算力屋顶，所以 batching 早就不划算了。

---

### 6. A 70B dense model at 8-bit weights, chips with ~230 MB of SRAM each. Walk me through the deployment and the unit economics.

**Capacity first.** 70 B params × 1 byte ≈ 70 GB of weights. Per chip 230 MB → ≥ **305 chips minimum** just to hold weights, and that's with zero SRAM left for KV cache, activations, or staging. Real deployments provision more than the floor (tensor-parallel splits must be legal, e.g. divisible head counts), so a rack-scale unit is the correct granularity to reason in.

**Then throughput.** Each token requires every chip to sweep its weight slice once; at ~80 TB/s on-chip bandwidth a 230 MB slice takes ~3 µs, so the compute-side floor per token is single-digit µs, and the *real* per-token time is dominated by pipeline depth and link traversal across ~300 chips. Publicly quoted behaviour for 70B-class models here is on the order of hundreds of tokens/s per stream — an order of magnitude, not a spec.

**Unit economics** — the model that matters is not $/token, it's $/chip-hour × chips × time:

- Cost per generated token ≈ `(chips × $/chip-hr) / (3600 × tokens_per_sec)`. With 320 chips and a few hundred tok/s, a single stream is *terrible* value: most silicon idles between tokens.
- Concurrency is what fixes it: many sequences share the same resident weights, so aggregate throughput scales with batch until compute saturates or KV runs out of SRAM — and KV is the hard cap here, since weights already ate the capacity.
- Therefore the honest pitch is **latency at fixed cost**, sold to workloads that would otherwise pay for a GPU with deep queues: real-time voice, agents, interactive codegen.

**Don't** promise GPU-style batch discounts: on this machine the batch curve flattens early, so price on streams/latency tiers and on utilisation you can actually sustain.

**中文要点**：70GB 权重 / 230MB ≈ 300+ 卡只是**容量下限**，KV 几乎没地方放；单 token 约几微秒但实际被流水与互连主导；经济性要按 `卡数 × 卡时` 算，单流很亏，靠共享常驻权重的并发把摊薄做回来——所以卖的是"固定成本下的确定性延迟"，不是批量折扣。

---

### 7. On a GPU you batch to amortise weight reads. What is the batching calculus on an SRAM-only machine, and how should that change how we price?

GPU batch-1 decode is a memory-bound GEMM: you pay `W` bytes of HBM traffic per token regardless, so adding sequences is nearly free until compute saturates. Batching is a **throughput multiplier bought with latency**.

On SRAM-only, weights never leave the die and the sweep runs at SRAM speed:

| Question | GPU | SRAM-only |
| :--- | :--- | :--- |
| Does batching amortise weight reads? | Yes — the main win | Marginally; the read is ~2 orders cheaper |
| What caps the batch? | KV in HBM (large) | **KV shares 230 MB with weights — tiny** |
| Does batch cut cost/token? | Strongly | Weakly, and only while the array is under-utilised |
| Does batch cost latency? | Yes, queueing | Yes, and it breaks the p99 you're selling |

So the calculus inverts: batching is **a utilisation fix, not a cost lever**. The interesting knob is concurrency at a *fixed* per-step budget — admit as many sequences as fit, run one deterministic step, return all their tokens. That yields GPU-like aggregate throughput with a tail that doesn't degrade with load.

**Pricing implication**: cost is `chips × time`, and one stream already sits near the compute floor, so revenue tracks sustained concurrent occupancy. Price on:

1. **Latency tier / reserved stream** — deterministic p99 *is* the product;
2. **Utilisation-aware commitments** — a customer holding 3 of 320 chips' streams is subsidised by everyone else, so discounts belong at reserved capacity;
3. **Cheap short-output interactive traffic**, where GPUs waste queue time; do not discount long batch jobs, the worst fit.

**中文要点**：SRAM 机器上权重常驻，batch 摊薄权重读的收益很小，真正的上限是**和权重抢 230MB 的 KV**；所以 batching 是"填满阵列"而非降本杠杆；定价应卖"确定性 p99 的保留流/延迟档"，而不是学 GPU 按 token 批量打折。

---

### 8. Determinism is the headline claim. What does it actually buy at p99, and why does it matter especially for agentic workloads?

**What it buys, mechanically:** with no dynamic batching, no cache-miss variance, no scheduler, and no SM-contention, the per-step time is a *constant* implied by the compiled schedule. So the latency distribution collapses: p50 and p99 sit within a few percent of each other, and p99 becomes a **number you can quote in a contract** rather than an emergent property of load.

That is worth more than raw speed for three reasons:

1. **Capacity planning becomes arithmetic.** With a fixed step time you can compute exactly how many concurrent admissions a rack supports, and know it holds at peak. With GPU p99 you buy headroom for a tail you can't predict.
2. **The output is bit-identical across runs.** Same prompt, same sampled-token sequence, same logits → reproducible bugs and reproducible A/B. On a GPU, nondeterministic kernels plus load-dependent batching make "it failed once" untriageable.
3. **Tail latency doesn't degrade with neighbours** — another tenant's 3 a.m. batch doesn't move my voice call.

**Why agentic workloads care disproportionately**: an agent is a *chain* of calls, often 10–40 per task, each with tool I/O in between. Latency compounds, so variance compounds — with a heavy tail, p99 of the chain is dominated by the worst of 40 steps, and timeouts at step 37 throw away all prior work. Determinism turns a chain's budget into `N × step` and makes **cancellation and deadline propagation** tractable.

The honest caveat to volunteer: determinism holds *given* the same batch composition and routing; admission-control decisions and tool latency are still outside it. And bit-exactness buys nothing if the tokenizer or sampler on the host is nondeterministic — that's where teams usually leak it.

**中文要点**：确定性把 p99 从"随负载涌现"变成"编译期算出来的常数"，p50/p99 只差几个百分点；agent 场景是 10~40 步的链式调用，方差会复利放大、超时就丢掉前面全部工作，所以"固定 step 时间"比峰值吞吐更值钱。

---

### 9. How would you serve a large mixture-of-experts model on a statically scheduled fabric when expert selection is data-dependent?

Name the tension first: **the compiler wants the whole graph's residency and timing known; MoE makes both a function of the input.**

The two constraints are independent, so split them:

| Constraint | Nature | Mitigation |
| :--- | :--- | :--- |
| **Capacity**: all E experts resident | static, solvable | expert parallelism; replicate hot experts; quantise experts harder than attention |
| **Routing**: which experts see which tokens | dynamic, unschedulable | capacity factor, token dropping, batched all-to-all |

What works on a fixed fabric:

1. **Reserve worst case, not average.** Compile a step accepting up to `C` tokens per expert (capacity factor ~1.25); overflow drops the lowest-gate-weight tokens (they keep the residual). This turns a data-dependent graph into a static one with a *guaranteed* step time, and the drop rate becomes a measurable quality knob.
2. **Two phases per step**: gather/scatter through a fixed permutation table, then dense matmul over reserved slots. The fabric's all-to-all runs on a compile-time slot map.
3. **Replicate hot experts** instead of load-balancing dynamically — popularity is stable per domain, so static replication removes most skew. The one place where "measure, then compile" beats "adapt at runtime".
4. **Keep the top-k decision at the host boundary**, so the compiler only sees "a permutation to apply".

Where it hurts: dropping tokens is a real quality loss on long-tail routing, and coarse capacity factors waste silicon. Fine-grained MoE is the worst case — the exchange is latency-dominated and each expert matmul is too small to fill the array.

**中文要点**：MoE 把"容量"（静态可解）和"路由"（数据相关）拆开处理：容量用专家并行+热专家复制，路由用固定 permutation 表 + capacity factor + 最低门控权重丢 token，把动态图变成**步长有界的静态图**；细粒度 MoE 最不适合固定时隙架构。

---

### 10. We pair LPX decode accelerators with NVIDIA GPUs doing prefill and attention. Design the serving path across those two machines.

**Source**: https://outcomeschool.com/blog/prefill-decode-disaggregation · https://outcomeschool.com/blog/how-does-an-lpu-work

The split exists because prefill and decode have opposite rooflines: prefill is **compute-bound** (a big GEMM, loves batched tensor cores), decode is **latency-bound and weight-read dominated** (loves residency + fixed schedule). Putting them on the same machine means one of them always runs at the other's efficiency.

Architecture, segment by segment:

| Segment | Runs on | Design point |
| :--- | :--- | :--- |
| Tokenize + prompt assembly + template | host | version prompts with the model |
| **Prefill** (full attention, all prompt tokens) | GPU pool, continuous batched, chunked-prefill on | maximise TTFT; big batches OK, nobody's watching |
| **KV transfer** | GPU → LPU network | format-compatible paged KV; this is the new bottleneck |
| **Decode loop** | LPX fabric, fixed step, resident weights | flat ITL; KV grows in LPU SRAM → caps concurrency |
| Tool calls / stop handling | host | cancellation must free KV on both sides |

The hard parts, in order:

1. **KV transport becomes the critical path.** Per token of context you move tensors on every step; the transfer must be paced so the LPU never starves. Prefix caching on the GPU side kills most of this for repeated system prompts — measure the hit rate before optimising the wire.
2. **Two schedulers, one deadline.** The GPU batch and the LPU slot admission are independent, so a prompt admitted at the LPU can wait in the GPU queue. You need a single end-to-end admission controller that reserves an LPU slot *before* scheduling the prefill, otherwise TTFT p99 is GPU queue time.
3. **Two clock domains.** The LPU's determinism only covers its own step; host-side jitter at the boundary leaks back in.
4. **Failure asymmetry**: a GPU node dies mid-prefill (recompute, cheap); an LPU chip dies mid-decode (replay the token, requires host-side step log).

**中文要点**：prefill 计算密集放 GPU、decode 权重驻留+固定时隙放 LPX，中间靠 KV 传输；真正的难点是**KV 传输成为关键路径**、两端调度器必须有一个端到端准入（先占 LPU 槽再排 prefill），否则 TTFT 的 p99 全耗在 GPU 排队上。

---

### 11. A prospective customer runs their workload on H100s. Talk me through when you would tell them not to move.

Credibility here comes from being willing to lose the deal. Lead with a qualification question — **what does your p99 and your GPU utilisation look like?** — then map their answers to a stay/move verdict.

**Tell them to stay** when:

| Signal | Why the LPU is the wrong answer |
| :--- | :--- |
| Offline/batch: no human waiting, minutes of latency acceptable | you'd be paying a latency premium for nothing; GPU $/token with deep batching wins outright |
| GPU utilisation already >70% with large batches | they've already amortised weight reads to the floor; the remaining gap is small |
| Custom kernels, fine-grained sparsity, non-standard ops, dynamic shapes | the whole value prop is a static schedule; every bespoke kernel is a compiler project |
| Huge context / huge KV (100k+ with many concurrent users) | KV must fit in SRAM; the capacity math will force absurd parallelism |
| Training or fine-tuning in the same cluster | different machine entirely — don't conflate the two budgets |
| Multi-modal pipeline with heavy pre-processing | the accelerator only replaces the decode stage; integration cost rarely pays back |

**Recommend a move** only where the number they should care about is the *tail*: real-time voice agents, interactive codegen, multi-step tool agents, anything with a contractual p95, and workloads whose GPUs sit at 15–25% utilisation because requests arrive one at a time.

Then be concrete about the migration cost so the recommendation can be trusted: re-qualification against their eval set, tokenizer/sampler parity, KV format differences, and the fact that a 2× throughput gain means nothing if their bottleneck is retrieval or tool latency — ask them to measure the fraction of end-to-end latency that is model time before signing anything.

**中文要点**：敢劝退才可信——离线批量、GPU 利用率已高、自定义 kernel/稀疏、超长上下文 KV、还要训模型，都应留在 H100；只有"人在等、卡在空转、p95 有合同"的场景才值得迁，并先量清楚端到端延迟里模型占多少。

---

### 12. Tell me about a performance optimisation you shipped. Give me the numbers, and tell me why I should believe them.

Structure the answer so the *measurement* is the story, not the win.

**The shape of a strong answer:**

1. **Baseline as a range with conditions.** "p50 42 ms, p99 61 ms at 8 concurrent requests, batch-1 decode, driver X, 3 runs of 5k requests."
2. **Where the time provably went**, from a profiler: per-step cycle counts showing stall reasons — "34% of cycles were link-stall waiting on the previous shard's tile".
3. **The change, in one sentence a compiler engineer can audit.** "Moved the transpose into the producer's DMA so the consumer never waits on a strided read."
4. **The delta with variance, plus what regressed.** "p99 61 → 38 ms across 3 runs (±1.2 ms); compile time +9%, SRAM +3 MB." The regressions are what make the number believable — a clean win with no cost is a smell.
5. **Validation**: same binary, A/B behind a flag, same input distribution, outputs still bit-identical.

**Why they should believe it** — say it explicitly: same build on the same hardware, toggled by a flag rather than across releases; the distribution reported, not the best run; the mechanism (a stall counter went to zero) given, which is falsifiable; and you know whose workload the +3 MB SRAM could break.

If you can't defend a number, say so. "I only have the fleet dashboard aggregate and it's confounded by a traffic shift" beats a confident fake.

**中文要点**：讲优化要把**测量**当主线：基线带条件与区间、用 profiler 指出时间花在哪、改动一句话可被审、给出方差和**回退项**（编译时间/SRAM 占用），并说明是同一 build 切 flag 的 A/B；拿不准的数字要承认混杂因素。

---

## 附：这 12 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 编译器真功夫 | 1（静态调度缺什么）、2（IR 与驻留决策） |
| 系统/运行时底子 | 3（host runtime 抖动）、4（整机架错在哪） |
| 懂不懂 roofline | 5（无 HBM 重推）、6（70B 单元经济） |
| 商业判断 | 7（batching 与定价）、11（何时劝客户别迁） |
| 新架构前沿 | 9（MoE 静态时隙）、10（LPX + GPU 拆分） |
| 诚实度 | 12（数字可不可信） |
| 确定性价值 | 8（p99 与 agent） |
