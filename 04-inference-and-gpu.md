# 04 · Inference, Serving and GPU Performance

> 15 题。出处：`README.en.md` § Inference, Serving and GPU Performance。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Explain the prefill and decode phases. Why is prefill compute-bound and decode memory-bandwidth-bound?

**Asked at**: Moonshot AI · NVIDIA · Together AI · ByteDance · Alibaba (Taobao/Tmall, Alibaba Cloud)
**Source**: https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization

**Anchor answer**

Prefill runs the whole prompt in one forward pass: a `[T, d] × [d, d]` GEMM where each weight byte fetched from HBM is reused `T` times. Arithmetic intensity grows with prompt length — at 8k tokens it is thousands of FLOP/byte, far above an H100's ridge point of roughly 295 FLOP/byte (BF16), so the tensor cores are the limit.

Decode emits **one token per sequence per step**: the same weights are re-read in full to produce a single token, giving ≈2 FLOP/byte. Tensor cores sit idle; tokens/sec ≈ `HBM bandwidth / weight bytes`. That is why a 70B BF16 model lands near 20–40 tok/s per GPU no matter how fast the ALUs are, and why batching is the only way back to compute density — with batch `B` the same weight read feeds `B` tokens.

| Phase | Bound by | Levers |
| :--- | :--- | :--- |
| Prefill | FLOPs (attention quadratic in T) | FP8 matmul, chunking, prefix cache |
| Decode | HBM bandwidth + KV capacity | large batch, weight quant, spec decoding |

Two practical consequences: mixing a long prefill into a decode batch stalls every streaming user (head-of-line blocking → chunked prefill), and at 100k+ context prefill stops being MLP-bound and becomes attention-bound, so FlashAttention-style kernels matter more than matmul throughput.

**中文要点**：prefill 是 GEMM、算力瓶颈；decode 每步只出一个 token、权重读一遍，2 FLOP/byte，纯显存带宽瓶颈；批处理是让 decode 重回算力密集的唯一手段。

---

### 2. What is continuous (in-flight) batching and why did it replace static batching?

**Asked at**: Anthropic · xAI · Mistral AI · NVIDIA · Together AI
**Source**: https://outcomeschool.com/blog/continuous-batching-in-llms

**Anchor answer**

Static batching collects N requests, pads them to the longest sequence, and runs until the **slowest** one finishes. With chat-like output-length spread (median a few hundred tokens, p99 over a thousand), finished sequences leave their slots idle for most of the batch's lifetime — typical slot utilisation lands around 30–50%, and padding burns compute on tokens nobody will read.

Continuous batching makes the scheduling decision **at every forward iteration**:

| Static | Continuous |
| :--- | :--- |
| batch formed once | batch re-formed each step |
| slot freed only at batch end | slot freed the moment a sequence emits EOS |
| pad to max length | no padding, per-sequence positions |
| throughput ceiling set by worst request | ceiling set by GPU, not by the tail |

It works because decode is stateless across steps apart from the KV cache: a slot is just KV blocks plus a position counter, so a new request can join mid-flight.

What you inherit with it: a KV block allocator, **preemption** (swap or recompute) when KV memory fills, admission control so long generations cannot starve short ones, and per-request SLOs instead of one queue. Continuous batching raises *throughput*; it does **not** lower single-request latency — bigger batches make TPOT worse, which is exactly why latency-sensitive tiers need a separate pool.

**中文要点**：静态批处理被最慢请求拖死、填充浪费，槽位利用率常只有三到五成；连续批处理逐步重组 batch，序列一结束立刻换新请求，换来数倍吞吐，但单请求延迟不变、甚至更差。

---

### 3. How does PagedAttention work, and what problem of KV-cache fragmentation does it solve?

**Asked at**: NVIDIA · Together AI
**Source**: https://outcomeschool.com/blog/paged-attention-in-llms · https://outcomeschool.com/blog/how-does-vllm-work

**Anchor answer**

KV cache grows with tokens actually generated, but the server must reserve for `max_model_len` because the final length is unknown. Reserving contiguous buffers per sequence gives three losses: internal fragmentation (a 300-token chat reserving 8k), external fragmentation, and the inability to share a common prefix.

PagedAttention borrows OS virtual memory: KV is stored in **fixed-size blocks** (16 tokens by default), each sequence owns a *block table* mapping logical block → physical block, and physical blocks need not be contiguous.

| Property | Effect |
| :--- | :--- |
| allocate on demand | waste ≈ one partially filled block (<4%) |
| block table indirection | no contiguous-reservation requirement |
| ref-count + copy-on-write | shared system prompt / beam branches share blocks |

The cost is a gather through the block table inside the attention kernel: reads are less coalesced than a contiguous FlashAttention buffer, and very small blocks add table-walk overhead. Block size is the knob — 8 is fine for short chats, 32–64 reduces overhead for long contexts.

The headline number: naive reservation wastes 60–80% of KV memory under real chat length distributions; paging drops it under 5%, which translates almost 1:1 into concurrent sequences. That concurrency, not the kernel, is where vLLM's throughput comes from.

**中文要点**：KV 必须按最大长度预留，导致六到八成显存被浪费；PagedAttention 用固定大小 block + block table 做虚拟内存式管理，浪费降到 4% 以下，外加前缀共享，直接换成并发数。

---

### 4. What is speculative decoding? Why is output quality preserved, and when does it not help?

**Asked at**: NVIDIA · Together AI · ByteDance
**Source**: https://outcomeschool.com/blog/speculative-decoding

**Anchor answer**

A cheap drafter (small model, n-gram lookup, or extra heads on the target like Medusa/EAGLE) proposes `k` tokens; the target model scores all `k+1` positions in **one** forward pass, which costs about the same as one decode step because decode is bandwidth-bound. The longest accepted prefix is kept.

Quality is preserved *exactly*, not approximately, because acceptance uses rejection sampling: accept draft token `x` with probability `min(1, p_target(x)/p_draft(x))`, and on rejection sample from the residual distribution `norm(max(0, p_target − p_draft))`. The composed distribution equals the target's own — greedy or sampled, the output is what the target would have produced.

Expected accepted tokens per target step is `(1 − α^(k+1))/(1 − α)` for acceptance rate `α`. Speedup ≈ that divided by drafter overhead.

Where it does not help:

- **Large batch**: the target step is compute-bound, so verifying `k` tokens costs ≈`k`× the FLOPs — the crossover is often around batch 32–64.
- **High temperature / open-ended chat**: `α` collapses; coding and summarisation tolerate drafts far better.
- **Domain mismatch** between drafter and target, or a drafter so big its own latency dominates.
- **Tight KV budget**: draft weights and extra KV eat the memory that concurrency needs.

So it is a **latency** tool for low-to-medium concurrency, not a throughput tool.

**中文要点**：小模型起草、大模型一次前向并行校验，接受率用 `min(1, p/p)` 的拒绝采样保证**分布完全一致**；大 batch 下校验成本随 k 线性增长，收益消失，所以它是低并发降延迟工具。

---

### 5. Explain prefix caching / prompt caching. When should you use it, and what invalidates a cached prefix?

**Asked at**: Moonshot AI · Character.AI · ByteDance
**Source**: https://outcomeschool.com/blog/how-does-prompt-caching-work

**Anchor answer**

Prefix caching stores the KV blocks of a token prefix, keyed by a hash chain (block hash = hash(parent hash, block tokens)). A matching prefix skips its prefill entirely: TTFT drops roughly in proportion to the reused fraction, and the freed prefill compute goes to other requests.

It pays whenever a long, **stable** span precedes variable content: a 2k system prompt, a tool schema block, a retrieved document set, or a long conversation history in multi-turn chat. Production chat routinely hits 60–90%; agent traces with mutating tool lists hit near zero.

What invalidates it — the cached object is an *exact token prefix*, not a semantic one:

| Invalidator | Why |
| :--- | :--- |
| Any edit earlier in the prompt | all downstream hashes change |
| Tool list added / reordered / re-described | tool JSON sits at the front |
| Timestamp, request id, username interpolated into the system prompt | every request is a new prefix |
| Non-deterministic JSON key order | one reordered key kills a 50k-token hit |
| Different images / audio, different tokenizer or chat template | different token ids |
| Different LoRA adapter | adapter changes the KV |
| LRU eviction under KV pressure, or TTL | blocks are gone |

Design rule: **stable content first, volatile last**, and serialise formally (sorted keys). Also note the cache is per node unless a router pins requests by prefix hash (SGLang's radix tree) or a shared KV store exists.

**中文要点**：键是"精确 token 前缀"的哈希链，不是语义；任何前置改动（系统提示、工具顺序、时间戳、JSON 键序、LoRA、分词器）都会让整段失效——稳定内容放前面是唯一有效设计。

---

### 6. Compare FP16, BF16, FP8, INT8, INT4 and FP4 for serving. What breaks at each step down?

**Asked at**: Mistral AI · Apple · NVIDIA · Together AI · Character.AI
**Source**: https://outcomeschool.com/blog/how-does-model-quantization-work

**Anchor answer**

| Format | Bytes/param | Gains | What breaks |
| :--- | :--- | :--- | :--- |
| FP16 | 2 | baseline | range overflow in large-model training (not serving) |
| BF16 | 2 | FP32 range, 8-bit mantissa | nothing measurable for inference; default choice |
| FP8 (E4M3) | 1 | ~1.8× throughput on Hopper/Blackwell | activation outliers >6σ; needs per-tensor scaling, per-token/per-block scaling for robustness |
| INT8 (W8A8) | 1 | same class as FP8 on older GPUs | without SmoothQuant the activation outliers crush accuracy; small models suffer first |
| INT4 weight-only (AWQ/GPTQ) | 0.5 | ~4× smaller, big win for bandwidth-bound decode | degrades below ~7B, hurts math/code/copying before chat quality; dequant overhead can make large-batch slower |
| FP4 (NVFP4, block scale) | 0.5 | 2× over FP8 on Blackwell | needs per-block scales + often QAT; naive FP4 visibly degrades reasoning |

Two rules that come from actually shipping it:

1. **Loss is not uniform.** First/last layers, `down_proj`, and anything requiring exact copying (JSON, IDs, numbers) break first. Measure on your task, not on perplexity.
2. **Quantise the KV cache separately.** FP8 KV roughly doubles concurrency; INT4 KV is where long-context needle retrieval quietly dies even though short-prompt evals look fine.

Also: weight-only quant speeds decode (bandwidth) but *not* prefill (compute), so a long-prompt workload may gain nothing.

**中文要点**：BF16 是无脑默认；FP8 要处理激活离群值，INT8 需 SmoothQuant，INT4 权重-only 在 7B 以下崩、且只降 decode 不降 prefill；KV cache 量化要单独验证长上下文。

---

### 7. Compare tensor, pipeline, data, sequence and expert parallelism. When do you combine them?

**Asked at**: Google DeepMind · Meta · Amazon (AWS) · NVIDIA

**Anchor answer**

| Strategy | Splits | Communication | Best fit |
| :--- | :--- | :--- | :--- |
| Data (DP) | replicas | none in forward | throughput scaling |
| Tensor (TP) | every weight matrix | 2 all-reduces per block, latency-critical | intra-node, NVLink, ≤8 ranks |
| Pipeline (PP) | layers into stages | tiny P2P at boundaries, but bubbles | across nodes, very large models |
| Sequence/context (SP/CP) | the token axis | K/V ring or all-gather | 100k+ contexts |
| Expert (EP) | MoE experts | all-to-all per layer, load-imbalance sensitive | MoE models, large clusters |

The governing rule: **put the highest-bandwidth dimension innermost**. TP over NVLink inside a node, PP over InfiniBand across nodes, DP outermost. Pushing TP past 8 ranks (out of the NVLink domain) is the classic mistake — per-layer all-reduce over 400G IB adds milliseconds per decode step and destroys ITL.

Combination example for a 405B dense-ish model on 2 nodes: TP=8 × PP=2 × DP=N, with micro-batches ≥2×PP stages so the pipeline bubble stays near `(PP−1)/(microbatches+PP−1)`.

When not to combine: PP at batch 1 is nearly all bubble — decode with small batches wants TP+DP, not PP. And EP only makes sense if routing is balanced; one hot expert serialises the whole all-to-all.

**中文要点**：带宽最高的维度放最内层——TP 关在 NVLink 单机 8 卡内，PP 跨机，DP 最外；TP 跨机做 all-reduce 是 decode 延迟的头号杀手，PP 在 batch 1 全是气泡。

---

### 8. Estimate the GPU memory needed to serve a 70B model: weights, KV cache, activations, fragmentation.

**Asked at**: NVIDIA · ByteDance
**Source**: https://outcomeschool.com/blog/kv-cache-in-llms · https://outcomeschool.com/blog/paged-attention-in-llms

**Anchor answer**

Worked example, Llama-2-70B shape (80 layers, 64 heads, 8 KV heads, head_dim 128):

| Component | Math | Result |
| :--- | :--- | :--- |
| Weights BF16 | 70B × 2 B | 140 GB |
| Weights FP8 | 70B × 1 B | 70 GB |
| KV per token | 2 × 80 × 8 × 128 × 2 B | 320 KB |
| KV at 8k ctx | 320 KB × 8192 | 2.6 GB / sequence |
| Activations | chunk × hidden × 2 B per stage + logits spike | ~1–3 GB transient |
| Fragmentation | paged (<4%) vs reserved (60–80%) | 0.05–5 GB |
| Runtime overhead | CUDA graphs, workspace, comm buffers | ~2 GB / process |

So the plan is: weights first, then `KV_budget / (per_token_kv × avg_seq_len)` sequences. On 8×H100 (640 GB) with BF16 weights: ~490 GB of KV → roughly **180 concurrent 8k-token sequences**; with FP8 weights + FP8 KV that number more than doubles.

The trap is the vocab logits spike: an 8k-token prefill materialising `[8192, 128k]` scores is ~2 GB in one allocation, and it is the allocation that actually OOMs, not the steady-state math. Second trap: KV per token scales with **KV heads**, so GQA (8 vs 64) buys 8× concurrency — model choice dominates any serving tweak.

**中文要点**：先算权重（BF16 140GB），再按 `每 token 320KB × 长度` 算 KV；真正 OOM 的往往是 prefill 的 vocab logits 瞬时分配，而 GQA 的 KV 头数直接决定并发上限。

---

### 9. What are TTFT, TPOT, ITL and throughput, and how do they trade against each other?

**Asked at**: Microsoft · Apple · Perplexity
**Source**: https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization · https://www.youtube.com/watch?v=XD8DD4cEHu0

**Anchor answer**

| Metric | Definition | Dominated by |
| :--- | :--- | :--- |
| TTFT | request → first token | queue wait + prefill |
| TPOT | (e2e − TTFT) / (n_out − 1) | per-step decode cost |
| ITL | the individual gaps between tokens | decode step + any prefill stealing the slot |
| Throughput | tokens/s or requests/s for the fleet | batch size, hardware, quantisation |

TPOT is a mean; **ITL p99 is what users feel**. A stream with 20 ms mean and one 600 ms stall feels broken, and that stall is almost always a long prefill occupying the batch — not a slow kernel.

Trades:

- **Batch ↑** → throughput ↑, TPOT ↑, TTFT ↑ (queueing). Decode is bandwidth-bound, so batch is nearly free up to the point where it becomes compute-bound.
- **Chunked prefill** → protects ITL, costs some TTFT and a few percent of throughput.
- **Speculative decoding** → lowers TPOT/ITL at low batch, neutral-to-negative at high batch.
- **Longer output caps / streaming** → no throughput change, big perceived-latency change.

Little's law ties it together: `concurrency = throughput × latency`. Capacity planning at a fixed SLO is a concurrency budget, not a QPS number.

Practical interactive targets (typical experience): TTFT 300–800 ms, TPOT 20–50 ms, ITL p99 under ~100 ms. Past ~30–40 tok/s humans stop noticing the difference — spend the headroom on TTFT instead.

**中文要点**：TPOT 是均值，用户感知的是 ITL p99，而 p99 尖刺基本都来自长 prefill 抢占 batch；批越大吞吐越高、TPOT 越差，扩容要按 Little's law 算并发而不是 QPS。

---

### 10. Do the roofline maths: how many tokens/sec can one H100 produce for a 70B model at batch size 1?

**Asked at**: NVIDIA · Together AI
**Source**: https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization

**Anchor answer**

Batch-1 decode reads every weight once per token, so:

```
tok/s ≈ HBM bandwidth × efficiency / weight bytes
```

H100 SXM: 3.35 TB/s peak, ~80–85% achievable. 70B BF16 weights = 140 GB.

`3.35e12 / 140e9 ≈ 24 tok/s` theoretical, so **~19–20 tok/s** at realistic efficiency. Arithmetic intensity is 2 FLOP/byte against a ridge point of `989e12 / 3.35e12 ≈ 295` FLOP/byte — about 0.7% of peak FLOPs. The tensor cores are irrelevant here.

Then the physical problem: 140 GB does not fit in 80 GB, so you need TP=2, each GPU holding 70 GB and reading 70 GB per token → `3.35e12 / 70e9 ≈ 48 tok/s` roofline, minus the two per-layer all-reduces. Measured Llama-2-70B on 2×H100 at batch 1 lands around 30–40 tok/s.

Scaling consequences, all linear and worth memorising:

| Change | Effect on batch-1 decode |
| :--- | :--- |
| FP8 weights | ~2× tok/s |
| INT4 weights | ~4× (until dequant overhead bites) |
| batch 8 | ~7× aggregate, TPOT slightly worse |
| batch 147 | decode reaches the ridge point; compute becomes the limit |

**中文要点**：batch=1 时 `tok/s ≈ 带宽 / 权重字节`，70B BF16 单卡理论 24、实际 20 左右，算力利用率不到 1%；70B 装不进 80GB，实际要 TP=2，roofline 约 48、实测 30~40。

---

### 11. When would you choose vLLM vs SGLang vs TensorRT-LLM vs a custom stack?

**Asked at**: NVIDIA · Together AI · Tencent · ByteDance
**Source**: https://outcomeschool.com/blog/how-does-vllm-work · https://outcomeschool.com/blog/how-does-sglang-work · https://outcomeschool.com/blog/how-does-tensorrt-llm-work

**Anchor answer**

| | vLLM | SGLang | TensorRT-LLM | Custom |
| :--- | :--- | :--- | :--- | :--- |
| Model coverage | widest (HF day-one) | broad, lags newest | narrow, curated | whatever you build |
| Signature strength | PagedAttention, LoRA multi-adapter | RadixAttention prefix tree, compressed-FSM structured output | best per-GPU kernels, FP8/FP4, in-flight batching | exotic KV transport, determinism |
| Iteration cost | minutes | minutes | engine build per (model, GPU, TP, quant) — tens of minutes | a team |
| Ops complexity | low | medium | high | extreme |

Selection rule that survives contact with production: **choose by model churn, not by benchmark tok/s.** If you ship model updates weekly or host dozens of fine-tunes, vLLM or SGLang win on operations even if a compiled engine is 20% faster. If you serve one or two frozen models on fixed hardware and need the last 20%, TensorRT-LLM pays. SGLang is the default when traffic is prefix-heavy (agents, multi-turn, long shared system prompts) or when you emit lots of constrained JSON — RadixAttention and the FSM decoder are worth 10–30% there.

Custom stacks are justified only by a hard constraint (deterministic latency, non-standard interconnect, PD disaggregation with your own KV transport). Otherwise you are funding an inference team to re-derive vLLM.

**中文要点**：按**模型迭代频率**选型而非跑分——模型常换选 vLLM/SGLang，模型冻结且要榨干硬件选 TRT-LLM；前缀重的 agent/多轮场景 SGLang 的 RadixAttention 值得 10~30%。

---

### 12. How would you cut LLM serving cost by 10x? Enumerate every lever and rank them.

**Asked at**: Microsoft · Amazon (AWS) · NVIDIA · Cursor (Anysphere)
**Source**: https://www.youtube.com/watch?v=jV2sCj4lHYk · https://outcomeschool.com/blog/llm-inference-optimization

**Anchor answer**

Ranked by return per engineering week — demand-side levers first, because they attack the numerator instead of the denominator:

| # | Lever | Typical factor | Note |
| :-: | :--- | :--- | :--- |
| 1 | Kill wasted tokens (prompt pruning, output caps, no verbose CoT, drop few-shots that do nothing) | 1.5–3× | cheapest, no infra |
| 2 | Caching: prefix/KV cache, exact cache, semantic cache for FAQ-shaped traffic | 1.5–3× | 60–90% hit on fixed system prompts |
| 3 | Routing: small/distilled model for the easy 60–80%, frontier for the tail | 3–10× | biggest single lever |
| 4 | Utilisation: continuous batching with the right batch and concurrency | 1.5–2× | fleets commonly run at 15–30% GPU utilisation |
| 5 | Quantisation: FP8 weights + FP8 KV | ~1.8× | INT4 for low-batch decode |
| 6 | Speculative decoding / MTP at low batch | 2–3× tok/s | latency lever that also cuts $/token when bandwidth-bound |
| 7 | Hardware & contracts: newer generation, spot for batch, reserved for steady | 1.5–3× | per-token providers for spiky load |
| 8 | PD disaggregation at fleet scale | 1.2–1.4× | only with fast KV transport |
| 9 | Distill to a smaller model | 3–20× | highest ceiling, slowest and riskiest |

10× is never one lever; it is `1.6 × 1.5 × 1.4 × 1.3 × 1.2`. Instrument first: split spend per 1k requests by *input* vs *output* tokens, by route and by tenant. Output tokens cost 3–5× input per token, and verbose JSON or a chatty system prompt is a cost bug, not a model choice.

**中文要点**：先砍需求侧——删无用 token、前缀/语义缓存、小模型路由，再谈供给侧的批处理、FP8、投机解码、硬件；十倍永远是五个 1.3~1.6 相乘，且先按输入/输出 token 拆分账单。

---

### 13. Your p99 latency doubled after a deploy with no model change. Walk through the diagnosis.

**Asked at**: OpenAI · Amazon (AWS) · Databricks · Perplexity

**Anchor answer**

First split by percentile: **p50 flat + p99 doubled = queueing or a tail cohort, not a slow kernel.** A kernel regression moves p50 too. If p50 moved as well, diff the engine/driver/GPU pool.

Then, in order:

1. **Traffic shape.** Input-length p99, `max_tokens` default, concurrent sessions. Queue delay explodes as utilisation ρ→0.9; a 20% traffic bump can double p99 with identical compute.
2. **Batch composition.** Long prefills interleaved with decode → head-of-line blocking. Look at per-request ITL against prompt length.
3. **Cache hit rate.** A one-line system-prompt edit, or a reordered tool list, resets prefix caching to zero and moves all that prefill cost into the latency path. This is the most common "nothing changed" deploy.
4. **Retries.** A gateway retry policy turning a 2% error rate into 20% extra load; check 429/5xx and duplicate request ids.
5. **Memory pressure.** KV block eviction/preemption (swap or recompute), which shows up as multi-second stalls for exactly the longest conversations.
6. **Deploy diff, literal.** Tokenizer version (same text, 15% more tokens), chat template, scheduler params, CUDA/driver, a new node type in the pool, MTU change.
7. **Noisy neighbour.** One degraded NVLink/IB link stalls a TP collective for everyone on the node.

Method: make every dashboard sliceable by tenant, route, input-length bucket and node; the slice that moved names the cause.

**中文要点**：先看 p50——p50 不动、p99 翻倍是排队或长尾队列问题，不是 kernel 变慢；排查顺序是流量形状、batch 组成、前缀缓存命中率、重试风暴、KV 抢占、分词器/模板版本。

---

### 14. What is chunked prefill, and why does it improve tail latency under mixed traffic?

**Anchor answer**

A prefill of `T` tokens occupies the batch for its whole duration — hundreds of milliseconds for 16k–32k prompts on a large model. If the scheduler admits it as one unit, every decoding request in that batch waits for it: ITL spikes to the prefill's duration and streaming visibly freezes. In a mixed workload (many short chats plus a few long RAG or codebase prompts) this single effect dominates p99 ITL.

Chunked prefill splits the prompt into chunks of `C` tokens (typically 512–2048) and schedules **per step under a fixed token budget**: each iteration admits some chunk tokens plus the decode tokens of all running sequences. Decode never loses its slot.

| | Whole prefill | Chunked |
| :--- | :--- | :--- |
| ITL p99 | spikes to prefill duration | smooth, bounded by budget |
| TTFT of the long request | best | worse (spread over more steps) |
| Throughput | best | a few % lower |

The throughput cost is real: each chunk re-reads the prefix's KV from HBM for attention, so very small chunks waste bandwidth and leave the GPU under-filled.

Tuning rule: token budget per step ≈ 2–4× the decode batch size. Too small → many steps, poor GEMM efficiency; too large → the ITL spikes come back. Also cap `max_num_batched_tokens` per request so one 100k-token user cannot monopolise the queue.

**中文要点**：长 prefill 整块入批会让所有解码请求的 ITL 尖刺到几百毫秒；按固定 token 预算切块混批后 ITL 变平滑，代价是长请求 TTFT 变差、吞吐降几个点，块大小要按解码批量的 2~4 倍调。

---

### 15. Explain disaggregated prefill/decode serving and when it pays for itself.

**Asked at**: Moonshot AI · Groq · ByteDance
**Source**: https://outcomeschool.com/blog/prefill-decode-disaggregation

**Anchor answer**

Prefill and decode want opposite machines: prefill is a big GEMM that wants FLOPs and short-lived memory; decode is a GEMV that wants bandwidth and enormous KV capacity. In one pool, every scheduling choice is a compromise, and long prompts keep stealing decode slots. Disaggregation runs **P workers** and **D workers** separately, with KV transferred P→D over NVLink/RDMA.

It pays when:

- prompts are long relative to outputs (RAG, codebase Q&A, document chat) — prefill is a large share of cost;
- you need tight ITL SLOs while accepting long prompts;
- the fleet is large enough that splitting pools does not lose batching efficiency;
- hardware is heterogeneous — put the expensive interconnect-local GPUs where prefill needs them.

It does not pay when the KV transfer exceeds the savings. Do the arithmetic: 70B, 8k prompt → 2.6 GB of KV. Over NVLink at ~900 GB/s that is ~3 ms (fine); over 400G IB ~50 ms; over 100G Ethernet ~210 ms (fatal for TTFT). Rule of thumb: if transfer time exceeds 10–15% of your TTFT budget, stay monolithic.

Also: small clusters lose more from reduced batching than they gain, and chunked prefill already solves the tail-latency complaint for most single-region deployments.

**中文要点**：prefill 要算力、decode 要带宽与 KV 容量，拆池后靠 KV 传输衔接；算清 `KV 字节 × 长度 / 互连带宽`——8k 提示 2.6GB，NVLink 3ms 可行、100G 以太 210ms 直接毁掉 TTFT，小集群拆分反而更亏。

---

## 附：这 15 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 只会调 API、没碰过 kernel | 1（prefill/decode 瓶颈）、10（roofline 手算） |
| 真做过线上服务 | 9（TTFT/ITL 权衡）、13（p99 排查）、14（chunked prefill） |
| 显存账算不清 | 3（PagedAttention）、8（70B 显存估算） |
| 平台选型能力 | 11（vLLM/SGLang/TRT-LLM）、15（PD 分离） |
| 成本意识 | 12（十倍降本排序）、6（量化档位代价） |
| 并行与规模 | 7（五种并行组合）、15（PD 分离） |
