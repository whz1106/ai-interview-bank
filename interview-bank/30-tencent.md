# 30 · Tencent

> 37 题。出处：`README.en.md` § Tencent。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Why do generative LLMs use decoder-only rather than encoder-decoder?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/encoder-vs-decoder-in-transformers

**Anchor answer**

Because **the generative task only ever needs a causal prefix**, and a decoder-only stack serves every task with that one objective.

Reasons, in order of weight:

| Reason | Mechanics |
| :--- | :--- |
| One objective, all tasks | next-token prediction on raw text; translation/QA/summarisation all become "continue the prompt" — no task-specific head, no paired data |
| No attention bottleneck | encoder-decoder passes the whole source through a fixed-width cross-attention interface; a decoder-only model attends to the original tokens at every layer |
| Simpler to scale | one stack, no encoder/decoder balance, no cross-attention; KV cache management is uniform (which is what made PagedAttention and prefix caching simple) |
| In-context learning works | demonstrations fit naturally in the prefix, which an encoder-decoder cannot exploit as well because the source is encoded separately |
| Training efficiency | every token of every document is a training signal; encoder-decoder wastes capacity on the encoder's reconstruction task |

Costs you accept: no bidirectional context (matters for classification/span extraction — a BERT-style model is still better and 10× cheaper there), and prefill is redundant work when the same prefix is reused (solved by prefix caching rather than architecture).

The honest note: for speech translation and some multimodal models, encoder-decoder is still the norm because the input and output modalities differ enough that a separate encoder earns its keep. Decoder-only won on *text*, not universally.

**中文要点**：生成任务只需要因果前缀，一个 next-token 目标就能覆盖所有任务、且避免固定宽度交叉注意力瓶颈、KV 管理统一；代价是无双向上下文（分类任务仍应选 BERT 类），语音/多模态仍常用编码器-解码器。

---

### 2. What is the difference between pre-norm and post-norm, and how does it affect training stability?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/encoder-vs-decoder-in-transformers

**Anchor answer**

The difference is where LayerNorm sits relative to the residual add, and it changes the **gradient path**, not just the numerics.

| | Post-norm (original) | Pre-norm (modern default) |
| :--- | :--- | :--- |
| Order | `LN(x + Sublayer(x))` | `x + Sublayer(LN(x))` |
| Residual path | passes **through** LN at every block | **identity** — a clean highway from output to input |
| Gradient at init | attenuated by `~1/√L`; needs warmup | O(1); trains without warmup |
| Depth scaling | degrades past ~12–24 layers without tricks | scales to 100+ layers |
| Final layer | no extra LN needed | needs a **final LN** before the head |
| Peak quality at small depth | slightly better | slightly worse |

Mechanism: with pre-norm the residual stream is unnormalised, so its variance grows with depth — hence the final LN, and hence why some models add a scaled residual (`x + α·Sublayer(...)`) to keep growth controlled. With post-norm, the LN inside the residual path means the gradient must pass through `LN`'s Jacobian at every block, which is why early training is fragile and why the original paper needed a long warmup.

What people get wrong: introducing pre-norm and then keeping the warmup "for safety" — it is not harmful but it hides that pre-norm's real benefit is removing the warmup *requirement*. And switching a trained post-norm model to pre-norm mid-run does not work: the weights are not interchangeable.

**中文要点**：post-norm 把 LN 放在残差相加**之后**，梯度每层都要穿过 LN 的雅可比，需长 warmup；pre-norm 残差是恒等通路，可无 warmup 训到上百层，但要加**最后一层 LN**，且残差方差随深度增长。

---

### 3. How does RoPE achieve relative position awareness using an absolute position encoding?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding

**Anchor answer**

RoPE rotates the query and key vectors by an angle **proportional to position**, in 2-D subspaces. Because a rotation is orthogonal, the dot product between `R_m q` and `R_n k` depends only on `m − n`.

Mechanics, for `d_k/2` pairs:

1. Split `q` into pairs `(q_{2i}, q_{2i+1})`.
2. Rotate pair `i` at position `m` by `θ_i = m · ω_i` with `ω_i = 10000^{-2i/d}`.
3. `q̃ = R(θ) q`; same for `k` at its own position.
4. Then `⟨R_m q, R_n k⟩ = ⟨q, R_{m−n} k⟩` — a function of relative offset only.

Practical consequences that matter more than the derivation:

| Property | Implication |
| :--- | :--- |
| No learned parameters | no position table to extend; that is why context extension is possible at all |
| Applied **after** `W_q`/`W_k`, before attention | so it must be inside the attention kernel, and it is why fused kernels take `cos/sin` tables |
| Relative only | no absolute position signal → the model cannot tell "the first token" from content alone; token-0 identity must come from another cue |
| Base `θ` controls length generalisation | raising the base (e.g. 10k → 500k) compresses high-frequency rotations and extends usable length |

Extension tricks operate on the frequency axis: position interpolation (divide `m`), NTK-aware scaling (scale `ω` non-uniformly), YaRN (per-band scaling plus attention temperature). The failure you should name: naive extrapolation past the trained length degrades sharply and *silently* — perplexity on short prompts stays fine while long-range retrieval collapses.

**中文要点**：RoPE 按位置旋转 Q/K，正交性使内积只依赖 `m−n`，因此"绝对编码"产生相对感知；无参数可外推，但需调基频/插值扩展长度；朴素外推会在长文检索上**静默**崩坏而短 prompt 看不出。

---

### 4. Why is everyone exploring MoE, and what is the core advantage over dense?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/mixture-of-experts

**Anchor answer**

MoE decouples **total parameters** from **FLOPs per token**. You can grow capacity (knowledge) without growing the cost of every forward pass.

| | Dense | MoE (`k` of `n` experts) |
| :--- | :--- | :--- |
| Params | `P` | `~n·P_expert` (much larger) |
| FLOPs/token | `~2P` | `~2·k·P_expert` + router |
| GPU memory | weights = params | **all experts must be resident** unless you offload |
| Compute/memory balance | coupled | decoupled — compute-cheap, memory-hungry |
| Training dynamics | stable | router collapse, load imbalance, expert under-training |

The core win is the **compute-per-quality** ratio: a ~8× larger model at ~1× dense FLOPs of a smaller one. DeepSeek and Mixtral both report better quality at equal activated compute.

The costs are the whole engineering story:

1. **Memory is not saved.** Every expert sits in HBM; you save FLOPs, not bytes. This inverts the roofline — MoE decode is *more* bandwidth-bound per useful FLOP.
2. **Load balancing.** Without an auxiliary loss (or DeepSeek's bias-based balancing), routing collapses onto a few experts and the rest never train.
3. **Communication.** Expert parallelism means all-to-all per layer; a bad token distribution turns a compute win into a network bottleneck.
4. **Batch sensitivity.** At small batch, per-expert batch size is tiny → poor GEMM efficiency. MoE is a large-batch, high-throughput technique; single-stream latency often gets *worse*.

When not to use it: latency-critical single-user serving, or small clusters where you cannot afford to keep experts resident.

**中文要点**：MoE 把总参数与每 token 算力解耦——容量变大而 FLOPs 几乎不变；但它**不省显存**（专家全驻留）、需负载均衡与 all-to-all 通信，小 batch 下单流延迟反而更差。

---

### 5. Take Qwen and DeepSeek as examples: introduce a model you know well and compare the two technically

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/mixture-of-experts

**Anchor answer**

Structure the answer as a table over the axes that actually differ, then take one position.

| Axis | Qwen3 (dense + MoE families) | DeepSeek (V3/R1) |
| :--- | :--- | :--- |
| Attention | GQA, standard RoPE | **MLA** with decoupled RoPE: KV compressed to a latent |
| FFN | dense and MoE variants across sizes | fine-grained MoE with shared experts + routed experts |
| Routing | standard top-k with aux loss | aux-loss-free bias balancing, plus a sequence-wise loss |
| Precision | bf16 training, fp8 for some deployments | **fp8 mixed precision training** at scale |
| Reasoning | hybrid thinking / non-thinking modes in one model | separate reasoning model (R1) with long CoT |
| MTP | — | **multi-token prediction** head as an auxiliary objective |

What each choice buys: MLA cuts KV cache roughly an order of magnitude versus MHA at the same width, which is what makes long context and high concurrency affordable — but it needs weight absorption and custom kernels. Fine-grained experts plus shared experts improve specialisation while keeping a stable "always-on" path, at the cost of more complex all-to-all.

Pick a side and justify it: for a product needing one model in both fast and reasoning modes, the hybrid-mode design is operationally simpler; for raw long-context throughput, MLA's cache compression is the bigger lever. The point is not to rank them — it is to show you know *which knob each turned and what it cost*.

**中文要点**：对比要落在具体旋钮：MLA 把 KV 压到隐向量（省一个数量级缓存，代价是吸收与自研 kernel）、细粒度 MoE + 共享专家、fp8 训练、MTP；Qwen 的混合思考模式在运维上更简单。

---

### 6. What is the core idea behind vLLM? What do PagedAttention and continuous batching each solve?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/how-does-vllm-work

**Anchor answer**

Core idea: **treat KV cache as an operating-system memory problem**, then schedule at token granularity on top of that allocator.

| Component | Problem solved | Mechanism |
| :--- | :--- | :--- |
| PagedAttention | KV had to be one contiguous block sized for max length → 60–80% wasted | fixed-size KV blocks + per-sequence block table; non-contiguous physical pages |
| Continuous batching | batch waited for the longest sequence; short ones idled the GPU | iteration-level scheduling: finished sequences leave, new ones join every step |
| Prefix caching | identical system prompts recomputed per request | hash-based block sharing; identical prefixes point to the same physical blocks |
| Chunked prefill | a long prefill stalls all decodes | split prefill into chunks, interleave with decode steps |

The dependency worth stating: paging enables sharing (prefix cache) *and* preemption (swap/recompute at block granularity); continuous batching enables the ragged batch that paging makes affordable. Neither alone gives the headline throughput.

Costs to name: custom attention kernel with block-table indirection (so you cannot use stock FlashAttention paths unchanged), higher TTFT variance for short requests under aggressive batching, and memory fragmentation at the block-table level instead of the sequence level. Also: `gpu_memory_utilization` too high → OOM under burst; too low → underused. Tune it with real traffic, not benchmarks.

**中文要点**：核心是把 KV cache 当操作系统内存问题——分页块表消除 60~80% 碎片并支持共享/抢占，连续批处理做逐 iteration 调度；两者互为前提，代价是定制 kernel 与短请求 TTFT 抖动。

---

### 7. What were the main improvements in FlashAttention v2 over v1?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/decoding-flash-attention

**Anchor answer**

Both are exact (no approximation); v2 is an **IO and parallelism** rewrite, not a math change.

| Improvement | v1 | v2 |
| :--- | :--- | :--- |
| Work partitioning | parallel over batch × heads, one block of rows per thread block | also parallel over **sequence length** — far more blocks, better occupancy on short sequences |
| Loop order | over K/V outer, Q inner → needed rescaling of the output each step | Q outer, K/V inner → **no output rescale** in the inner loop, less shared-memory traffic |
| Non-matmul FLOPs | relatively high | reduced; softmax rescale folded into the epilogue |
| Warp-level layout | split-K across warps required shared memory for the reduction | each warp owns a slice of rows → **no cross-warp communication** for the softmax |
| Result | ~2× over standard attention | ~2–4× over v1 in the reported configs; more at long sequence |

The key idea to articulate: attention is memory-bandwidth-bound because the `n×n` score matrix never needs to hit HBM. Tiling keeps it in SRAM; the online softmax (running max + rescaled accumulator) makes the tiling exact.

v3 (Hopper) adds async `wgmma` and overlapped softmax/matmul — a different mechanism (producer/consumer warp specialisation), not just a retune. Practical note: on modern stacks you rarely call these directly; `scaled_dot_product_attention` picks a backend, and varlen + causal masks are where the real-world speedup differs from the paper's numbers.

**中文要点**：v2 是 IO 与并行度重写（按序列维并行、Q 外层 K/V 内层免去输出重缩放、warp 各持行切片免跨 warp 通信），数学精确；v3 靠 Hopper 异步 wgmma 与 warp 特化，是另一套机制。

---

### 8. How do you quality-check training data before it goes into a run?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/llm-evaluation

**Anchor answer**

Run a **staged gate** where each stage is cheap and kills a different failure class; never rely on one "quality score".

| Gate | Check | Kills |
| :--- | :--- | :--- |
| Dedup | exact hash → MinHash/LSH near-dup → suffix-array for long repeats | memorisation, 10× over-weighting of scraped boilerplate |
| Heuristic | length, symbol ratio, language ID, repeated-n-gram ratio, template/boilerplate regex | nav junk, truncated HTML, "click here" pages |
| Model-based | classifier or small judge for quality/educational value; perplexity filter against a reference model | fluent but useless text |
| Contamination | n-gram overlap against every eval set (including the ones you plan to use later) | inflated benchmark numbers |
| Safety/PII | toxicity, PII regex + NER, licence metadata | legal and policy exposure |
| Manual | 200–500 sampled docs read by a human, per source | everything the above missed |

Numbers worth quoting as typical experience: dedup removes 30–60% of a raw web crawl; a quality classifier typically keeps 5–15%. The counter-intuitive part: **judge-model scores correlate with length**, so always length-normalise or you train a verbose model.

The failure mode to name: contamination checks run once and forgotten — public eval sets leak into every subsequent crawl. Keep a rolling n-gram index of all benchmarks you might ever use, and version the filtered dataset with its hash so a run is reproducible.

**中文要点**：分级闸门——精确/近似去重、启发式过滤、模型打分、与所有评测集的污染检测、PII 与授权、人工抽检；去重通常砍掉三到六成原始网页；注意质量分与长度相关，必须长度归一，否则训出"话多"模型。

---

### 9. Why do LLMs stream token by token, and what protocol sits underneath?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/how-does-token-streaming-work

**Anchor answer**

Because the model already produces one token per decode step, and **buffering them adds pure latency for no benefit**. Perceived latency is dominated by time-to-first-token, not total time.

Underneath, in order of the stack:

| Layer | Mechanism |
| :--- | :--- |
| Model | autoregressive decode; each step yields one token (or a few with speculative decoding) |
| Server | SSE (`text/event-stream`) is the default for browser clients — one-way, plain HTTP, proxy-friendly |
| Server (alt) | WebSocket when you need bidirectional (interrupt, barge-in), or HTTP chunked transfer |
| Payload | OpenAI-style `data: {"choices":[{"delta":{"content":"…"}}]}` with a `[DONE]` sentinel |
| Edge | proxies/CDNs must have buffering disabled, or streaming silently breaks |

The failure modes that actually bite in production:

1. **Proxy buffering** — Nginx `proxy_buffering on` or a CDN with compression coalesces chunks; users see nothing then everything at once.
2. **Token boundary splitting** — streaming raw token strings breaks multi-byte characters and markdown; emit UTF-8 safe boundaries (and remember a token is not a character).
3. **Mid-stream errors** — after headers are sent you cannot return a 5xx; you must send an error event inside the stream and close.
4. **Cancellation** — client disconnect must abort generation, or you burn GPU on nobody's answer. This is also where token accounting goes wrong.

**中文要点**：模型本来就逐 token 解码，缓冲只会白增延迟，而用户感知看 TTFT；底层多用 SSE（双向/打断才用 WebSocket）；坑在代理缓冲把流合并、按 token 切会切断多字节字符、流内错误无法回 5xx、断连不取消会白烧 GPU。

---

### 10. Where do you look to cut response latency in an API service?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization

**Anchor answer**

Decompose latency first (`TTFT + tokens × TPOT`), then attack the dominant term — most people optimise the wrong one.

| Term | Lever | Typical effect |
| :--- | :--- | :--- |
| Queueing | admission control, priority queues, autoscale on queue depth not CPU | often the largest single win; usually invisible in dashboards |
| TTFT | prefix caching on the shared system prompt, shorter prompts, chunked prefill | 30–70% of TTFT when the prompt is mostly boilerplate |
| TPOT | larger batch, GQA/MLA, fp8 KV, speculative decoding | decode is bandwidth-bound; batching is the lever |
| Network/egress | region co-location, connection reuse, streaming | tens of ms, free |
| Post-processing | don't wait for the full answer to render; stream | perceived, not real |

Ordering that actually works: measure p50/p95/p99 separately per endpoint, then check **queue wait vs compute**. In my experience queue wait is 40–70% of p99 in a healthy-looking service, and no kernel optimisation fixes it.

Anti-patterns to name: enabling every optimisation at once (you cannot attribute the gain, and speculative decoding may hurt under high batch); quantising weights to save memory when you are bandwidth-bound at decode and lose accuracy for nothing; and optimising mean latency when the SLO is p99.

Where to stop: if you are compute-bound with a high batch and low queue wait, further latency work is a model change, not a serving change.

**中文要点**：先把延迟拆成 `TTFT + token×TPOT` 并按 p50/p99 分别看**排队等待 vs 计算**——排队常占 p99 的 40~70%，改 kernel 无用；前缀缓存砍 TTFT，批处理砍 TPOT，别一次全上否则无法归因。

---

### 11. Compare RLHF, PPO and DPO: technical differences, trade-offs and when each applies

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/direct-preference-optimization-dpo

**Anchor answer**

They are not peers: RLHF is a **pipeline**, PPO and DPO are two ways to optimise the last stage of it.

| | RLHF (PPO-based) | DPO |
| :--- | :--- | :--- |
| Pipeline | SFT → reward model → RL | SFT → direct preference optimisation |
| Models in memory | policy + reference + reward + critic (4) | policy + reference (2) |
| Data | prompts + human comparisons for the RM, then online rollouts | the same preference pairs, used directly |
| Objective | maximise RM score − KL | closed-form: raise `log π(y_w) − log π(y_l)` relative to reference |
| Exploration | online — samples new outputs each step | offline — only the pairs you have |
| Stability | fragile (RM drift, KL tuning, advantage noise) | stable but prone to over-fitting the pairs |
| Cost | high (4 models, rollout generation) | low (2 models, one pass over data) |

Theoretically: DPO is the PPO objective's optimum reparameterised, so the RM is implicit in the policy's log-ratio. That is why it is cheap — and why it cannot improve beyond its data: no reward model means no new samples.

When each applies:

- **DPO** — limited compute, a solid SFT base, and clean preference pairs. Default choice for most teams.
- **PPO/RLHF** — you can afford online sampling and have a reliable verifier or RM; needed when the reward is not expressible as offline pairs (e.g. long-horizon agent tasks, verifiable math).
- **Neither** — if you cannot define "better" reliably, fix the data before touching the algorithm.

**中文要点**：RLHF 是流水线，PPO 与 DPO 只是最后一步的两种做法；DPO 把奖励模型隐式写进策略的对数比，省两个模型但**无法超越离线偏好数据**；有可靠验证器且能在线采样时才上 PPO。

---

### 12. Full fine-tuning versus LoRA versus QLoRA versus P-Tuning: how do you choose, and when does LoRA beat full fine-tuning in a recommendation setting?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/lora-low-rank-adaptation-of-llms

**Anchor answer**

Choose by **how far you need to move the distribution** and how many adapters you must serve.

| Method | Trainable | Memory | Drift capacity | Serve |
| :--- | :--- | :--- | :--- | :--- |
| Full FT | 100% | ~16× params (Adam fp32 states) | highest | one model per task |
| LoRA | ~0.1–1% | ~2× params (no optimiser states for base) | medium; rank caps it | hot-swappable adapters |
| QLoRA | same as LoRA, 4-bit base | ~0.5× params | slightly below LoRA (quantisation error) | dequantise or serve 4-bit |
| P-Tuning / prefix | soft prompt vectors only | tiny | lowest; steals context | one prefix per task |

The rule of thumb: LoRA matches full FT when the task is **style/format/domain adaptation** and the base already has the capability; full FT wins when you are teaching a genuinely new skill or shifting the distribution hard (new language, new modality). QLoRA is a memory decision, not a quality decision — most of its gap closes with bf16 adapters after training.

Where LoRA beats full FT in recommendation specifically: you have **many tasks or many tenants** (recall, ranking, explanation, per-tenant tone) and need one base plus N small adapters; you retrain weekly per campaign and full FT is too slow; and you need the base model frozen so a shared feature pipeline stays valid. The number that matters: LoRA adds ~1–3% latency at serving if merged, ~5–10% if unmerged, and merging loses the ability to swap per request.

Failure case: LoRA on a task needing new *facts* — adapters learn style faster than knowledge, so low-rank is the wrong shape for factual injection; use RAG instead.

**中文要点**：按"需要移动分布多远"选：LoRA 适合风格/领域适配且要多适配器热切换，全参适合学新技能；QLoRA 是显存决策不是质量决策；LoRA 学不进新事实，注事实应用 RAG。

---

### 13. How do you build reward-model training data, and how would you design the reward and advantage for GRPO?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/group-relative-policy-optimization-grpo

**Anchor answer**

Two separate problems: RM data is **pairwise and hard**, GRPO reward is **scalar and mostly rule-based**.

RM data construction:

1. **Prompt pool** from real traffic, not synthetic — RM learns the distribution you will decode from.
2. **Generate `k` (4–8) diverse responses** per prompt across models/temperatures, so pairs are close in quality.
3. **Annotate pairwise**, not absolute scores: "which is better" has far higher agreement than "score 1–5".
4. **Stratify**: include ties, near-duplicates, long/short, refusal cases. Ties are essential or the RM learns length.
5. **Adversarial pairs**: same content, different verbosity; correct but rude; fluent but wrong — these are the patches for known hacks.
6. **Hold out by prompt**, never by response, or you leak.

Metrics: pairwise accuracy alone is misleading — track accuracy on **hard pairs** (annotator disagreement), tie calibration, and length correlation of the score.

GRPO design:

- **Reward**: `r = w_v·verify + w_f·format − w_l·length_penalty`, keep each term bounded and inspectable. Rule rewards for anything checkable; a model judge only for open-ended parts, at low weight.
- **Advantage**: `Â_i = (r_i − mean(r_group)) / (std(r_group) + ε)`. The `ε` is not cosmetic — an all-identical group makes std ≈ 0 and the ratio explodes.
- **Guard**: if `std < τ`, skip the group (no signal) rather than training on noise. This single check removes most RL divergence.

**中文要点**：奖励模型数据要成对标注、同 prompt 多候选择优、刻意包含 tie 与"冗长/礼貌但错误"的对抗对；GRPO 的奖励分项有界可审，优势用组内标准化且**必须加 ε 并在标准差过小时跳过该组**，否则数值爆炸。

---

### 14. How would you improve a model's multi-turn conversation ability, across model, data, memory and decoding?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/context-engineering

**Anchor answer**

Multi-turn failure is usually **not** a model-capability problem — it is a state problem. Fix in this order.

| Layer | Action | Why it dominates |
| :--- | :--- | :--- |
| Data | train on real multi-turn transcripts, not single-turn concatenations; include corrections, topic switches, pronoun resolution, "as I said before" | most SFT data is single-turn, so the model never learns to use history |
| Data (hard) | synthesise **conflict** cases: user changes their mind; earlier fact superseded | teaches recency handling, the top multi-turn bug |
| Memory | explicit structured state (slots/facts) extracted and re-injected each turn, instead of hoping attention finds it | long histories get lost in the middle |
| Model | train on the *same* role/turn markers used at inference; mismatched templates are a silent killer | template drift is the most common regression |
| Decoding | lower temperature for slot-filling turns; avoid over-long chain-of-thought in chat (it leaks into the answer) | feels responsive |

Two concrete hooks: (1) a "state block" at the top of the context — `{user_goal, confirmed_facts, open_questions}` — refreshed each turn, which consistently beats raw history replay because it survives compaction; (2) **recency-weighted ordering**, since models attend poorly to the middle of long contexts.

Evaluation is the part people skip: build a multi-turn test set where success requires a fact from turn 1 at turn 8, and where turn 5 contradicts turn 2. Single-turn benchmarks will show no difference at all.

**中文要点**：多轮问题多是**状态管理**而非能力问题：训练数据要真实多轮（含用户改主意、前文被推翻），上下文顶部维持结构化状态块，模板必须与推理一致；评测要造"第 8 轮用到第 1 轮事实"的用例。

---

### 15. Why use both sparse and dense retrieval in RAG, and how do you tune the weights in production?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/how-does-hybrid-search-work

**Anchor answer**

Sparse and dense fail on **disjoint** query types, so their union is strictly better than either — the tuning question is only how to combine.

| Query type | Sparse (BM25) | Dense |
| :--- | :--- | :--- |
| exact code / SKU / error string | strong | weak — the vector washes out the literal |
| paraphrase, synonyms | weak | strong |
| out-of-domain vocabulary | strong (no embedding needed) | weak — OOV maps to a bad vector |
| long natural-language question | weak (term dilution) | strong |

How to tune in production:

1. **Start with RRF**, `score = Σ 1/(k + rank_i)` with `k≈60`. It needs no score calibration, so it cannot break when one retriever's scale drifts. This is the correct default and most teams should stop here.
2. **Only go to weighted score fusion if** you have per-route labelled data (which route *should* have found the gold doc). Then the weight is a coordinate you can fit; without labels, tuning weights is fitting noise.
3. **Measure per-route recall@k separately** before fusing. If sparse contributes 3% unique recall, drop it — the extra index and latency are not free.
4. **Log the route that produced the winning document** for every answer. This is what lets you re-tune when traffic shifts, and it is the number that reveals "dense carries 95% of wins now".

Guardrail: cap candidates per route (e.g. 50) so one noisy retriever cannot flood the reranker, and always rerank the fused set — fusion quality plateaus, reranking does not.

**中文要点**：稀疏与稠密的失败查询集几乎不相交，所以联合必优；默认用 RRF（无需分数校准），只有在有"哪路应召回金标"的标注时才做加权分数融合；先分别测各路 recall@k，贡献 3% 独有召回就直接砍掉。

---

### 16. What drives the choice of BGE as an embedding model, and how do you build and tune a FAISS index?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/how-does-a-vector-database-work

**Anchor answer**

BGE is chosen for **evaluation transparency plus instruction-aware retrieval**, not for being fastest.

Why teams pick it: it publishes retrieval benchmarks per size (small/base/large), so you can pick by latency budget; it supports an instruction prefix on the query side (`"Represent this sentence for searching relevant passages:"`), which measurably helps short queries; multilingual and Chinese variants exist; and it is small enough to self-host, unlike API embeddings where cost and data residency are decided for you.

Caveat that matters: **the prefix must match training**. Applying it to documents instead of queries (or omitting it) silently costs several points — this is the most common BGE bug in the wild.

FAISS index build and tuning:

| Choice | When |
| :--- | :--- |
| `IndexFlatIP` (exact) | <100k vectors, or as a recall ground truth |
| `IVF` + `PQ`/`SQ` | millions; tune `nlist ≈ 4√N`, `nprobe` for recall/latency |
| `HNSW` | low latency, high recall, memory-rich; tune `M` (16–64), `efSearch` |
| `IVF` + refine (rerank with exact) | cheap recall recovery |

Normalise vectors and use inner product (= cosine). Tune by measuring **recall@10 against exact search** on a 5k query sample, not by vibes — and re-tune after any corpus change, since `nlist` is a function of `N`. Rebuild on a schedule; incremental adds degrade IVF balance over time.

**中文要点**：选 BGE 是因为公开分档评测 + 查询侧指令前缀可自托管；坑是前缀必须只加在 query 且与训练一致；FAISS 调参用"对精确检索的 recall@10"作目标，`nlist≈4√N`，语料变化后必须重调重建。

---

### 17. How do you chunk documents, and how do you trade chunk size against completeness?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/chunking-strategies-for-rag

**Anchor answer**

Chunk size trades **retrieval precision against answer completeness**, and there is no universal value — only a value per document type.

| Strategy | Unit | Good for | Bad for |
| :--- | :--- | :--- | :--- |
| Fixed tokens + overlap | 300–500, 10–15% overlap | homogeneous text, quick baseline | tables, code, anything structured |
| Structure-aware | heading/section boundaries | manuals, policies, wikis | documents with no headings |
| Semantic | embedding-similarity breakpoints | prose with topic shifts | expensive, unstable chunk counts |
| Parent-child | small child (200) matched, parent (1k) returned | the usual best default | extra storage, needs parent lookup |
| Sentence-window | sentence hit, ±k sentences | precise factual QA | loses document-level context |

The mechanism behind the trade-off: too small ⇒ the chunk has the fact but not the condition ("applies only if the contract started before 2020"), so the model answers confidently wrong. Too large ⇒ the embedding averages many topics, so the chunk matches weakly and the relevant sentence is buried among three irrelevant pages.

Practical defaults worth stating as experience, not fact: 300–500 tokens with parent expansion; keep tables intact and never split a table row across chunks; prepend the heading path to each chunk's text before embedding (`Doc > §Section > subsection`), which typically improves retrieval noticeably at zero cost.

Evaluate the choice, don't guess: hold out questions whose evidence spans two sentences 300 tokens apart. If your chunking splits them, recall collapses and no reranker will save you.

**中文要点**：太小丢前提（"仅适用于 2020 年前签约"），太大主题被平均导致匹配弱；默认 300~500 token + 父块回填、表格不切、embedding 前拼上标题路径；用"证据跨两句相隔 300 token"的题集验证切分。

---

### 18. What if one generation has to reason across several documents?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/how-does-a-reranker-work

**Anchor answer**

Multi-document reasoning fails for a structural reason: **the evidence needed is spread across chunks that no single query retrieves well.** Retrieval optimisation alone cannot fix it.

Three workable patterns:

| Pattern | How | Cost |
| :--- | :--- | :--- |
| Iterative retrieval (multi-hop) | generate a sub-question from what you have, retrieve again, repeat ≤3 | extra latency, loop risk |
| Query decomposition | split the question into sub-questions up front, retrieve per sub-question, union | needs a good decomposer; over-splits |
| Structured map-reduce | per-document extraction into a typed table, then reason over the table | most reliable, needs a schema |

The third is underused and usually wins for "compare X across these 5 reports": extract `{entity, attribute, value, source_url}` per document into rows, then let the model reason over a small table instead of 5×4k tokens of prose. Accuracy goes up and citation becomes mechanical.

What to add regardless:

1. **Per-document quotas** in retrieval — otherwise one verbose document takes all 6 slots.
2. **Explicit contradiction handling** — when two documents disagree, the answer must surface both with dates and say which is newer; models otherwise average them into a confident falsehood.
3. **Coverage check** — before answering, verify each sub-question has at least one supporting passage; refuse or narrow if not.

Stop condition for iteration: no new document retrieved, or a step budget. Uncapped multi-hop is the most common runaway agent.

**中文要点**：多文档失败的根因是证据分散、单条 query 都召不全；最稳的是"逐文档抽取成结构化表再推理"（引用可机械化），并做每文档配额、显式冲突呈现与子问题覆盖检查。

---

### 19. Beyond truncation, what are better ways to compress a long conversation?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/how-does-context-compaction-work

**Anchor answer**

Truncation is the worst option because it removes the *oldest* tokens, which is exactly where the user's original constraints live.

Ranked alternatives:

| Method | Mechanism | Preserves |
| :--- | :--- | :--- |
| Structured state extraction | maintain `{goal, constraints, decisions, open_questions, entities}` and re-inject; drop raw turns | commitments and numbers |
| Rolling summary + pinned head | summarise the middle, keep system prompt and last `k` turns verbatim | flow and tone |
| Hierarchical summarisation | per-topic summaries, merged on demand | long projects with distinct phases |
| Retrieval over history | embed past turns, fetch the relevant ones for the current query | specific facts on demand |
| Tool-output offloading | replace big outputs with a pointer + one-line result | reasoning context |
| KV-level compression | token pruning / attention sinks / cache eviction | nothing semantic — pure engineering |

The combination that works in production: **state block (always) + rolling summary (middle) + verbatim tail (recent) + retrievable archive (everything)**. The state block is what makes it robust — it is small, structured, and survives further compaction because it is rewritten from facts rather than summarised from prose.

What must never be summarised: identifiers, numbers, permissions, the literal user instruction. And never summarise a summary repeatedly — always re-derive from the original transcript, or facts decay each pass.

**中文要点**：截断砍掉的正是最早的约束；用"状态块（目标/约束/决策/未决）+ 滚动摘要 + 逐字近期窗口 + 可检索归档"组合；ID、数字、权限、用户原话不许摘要，且摘要必须每次从原文重做。

---

### 20. Claude Code already exists, so why build your own coding agent? What differs, and how do you design layered context?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/how-does-claude-code-work

**Anchor answer**

You build your own when the value is in the **private context and the private execution environment**, not in the model loop.

Reasons that hold up:

| Reason | Detail |
| :--- | :--- |
| Code cannot leave the perimeter | regulated/source-sensitive repos; a hosted agent is a compliance dead end |
| Private toolchain | internal build system, proprietary test harness, monorepo indexing, deploy gates |
| Workflow embedding | must run inside your ticket system, CI, review flow, not a terminal |
| Cost and model choice | route cheap edits to a small model; own the token budget |
| Evaluation | you can only improve what you can measure on your own tasks |

What then differs from a generic CLI agent: your **retrieval** (repo map, symbol graph, ownership), your **verification** (real test runner, sandbox), and your **policy** (which paths are writable, which commands need approval). Those three are the moat; the loop is commodity.

Layered context design:

1. **Always-on layer** — conventions, build/test commands, do-not-touch list. Small, stable, cacheable (this becomes a prefix cache hit every request).
2. **Task layer** — the ticket, plus a plan file the agent writes and updates; the plan survives compaction.
3. **Retrieved layer** — files by symbol/dependency, not by similarity to the prompt; lazy, one file at a time.
4. **Verbatim tail** — recent edits and their diffs, tool results.

Keep layer 1 stable to maximise prefix-cache hits; mutate only layers 2–4. The failure to name: dumping the whole repo map into layer 1, which destroys caching and drowns the actual task.

**中文要点**：自研的理由是**私有上下文与私有执行环境**（代码不出域、内部工具链、工单/CI 集成、成本与模型选择、自有评测）；真正的差异在检索、验证与权限策略，循环本身是同质化能力；上下文分层里第 1 层要保持稳定以吃满前缀缓存。

---

### 21. How do you design an agent's assembly and execution chain, and how does it degrade gracefully?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/ai-orchestration

**Anchor answer**

Design for **degradation as a first-class path**: every stage declares a cheaper fallback before you write its happy path.

Assembly chain:

| Stage | Output | Fallback |
| :--- | :--- | :--- |
| Intent/route | task class, required tools | default to the safest generic path |
| Context assembly | prompt + evidence, budgeted | drop retrieved layer, keep instructions |
| Plan | ordered steps, or a direct answer | single-shot answer |
| Execute | tool calls, one at a time | read-only tools only |
| Verify | schema + tests + judge | return unverified result with a warning |
| Respond | answer + citations | "partial result + what's missing" |

Degradation ladder, triggered by explicit conditions rather than exceptions:

1. **Retrieval down** → answer from parametric knowledge, say so, no citations.
2. **Primary model down/timeout** → smaller model with a shorter prompt; accept lower quality over no answer.
3. **Tool failure** → drop that tool, continue with the rest, mark the gap.
4. **Budget exhausted** → summarise partial findings, state what was not checked.
5. **Everything down** → deterministic template response plus a human handoff link.

Two rules that make it real: the degraded response must be **visibly different** (a flag, not silent), and each rung must be **tested by fault injection** — kill the retriever in staging and see what the user gets. Agents that only work when all dependencies are up are demos; the difference is whether you wrote the ladder before the incident.

**中文要点**：把降级当一等信息设计：每个环节先写更便宜的兜底再写顺利路径；阶梯是"检索挂→直答并声明、主模型超时→小模型短提示、工具失败→剔除继续、预算耗尽→总结已查/未查、全挂→模板+人工"；降级必须显式标记并做故障注入测试。

---

### 22. A sub-agent discovers it needs an extra task mid-execution. How should that task flow and get approved?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/ai-subagents

**Anchor answer**

The sub-agent must **not** silently expand its own scope. It proposes, the orchestrator decides — this keeps the plan auditable and prevents fan-out explosions.

Flow:

1. **Detect** — the sub-agent hits a precondition it cannot satisfy (missing file, unknown schema, dependency outside its slice).
2. **Emit a request, don't act** — a structured message: `{kind: new_task, reason, proposed_scope, blocking?, est_cost}`. It then **pauses or returns partial output**; it must not spawn a sibling itself.
3. **Orchestrator validates** — checks against the original plan: is it in scope? Does it overlap an existing task? Is the budget available?
4. **Approve / deny / merge** — three outcomes, all explicit. Deny must come with an instruction ("proceed with the fallback: mark the field unknown").
5. **Record** — append to the plan with a parent link, so the final trace shows why the extra task existed.

Rules that prevent the classic failures:

| Failure | Rule |
| :--- | :--- |
| Recursive spawning | only the orchestrator can create tasks; depth cap (e.g. 2) |
| Duplicate work | dedupe by task signature before approval |
| Deadlock | requests must have a timeout; on timeout the sub-agent proceeds with its fallback |
| Silent scope creep | the final report lists every unapproved task it wanted, so reviewers see the drift |

The approval gate should differ by risk: read-only research tasks can be auto-approved within budget; anything that writes, costs money, or touches another owner's files needs explicit approval. Auto-approving writes is how a "small fix" becomes a 40-file diff.

**中文要点**：子代理**只能提议不能自行扩权**——发结构化请求（原因/范围/是否阻塞/预估成本）后暂停或返回部分结果；由编排器按原计划校验、批准/拒绝/合并并登记父子关系；只有编排器能建任务且要限深度，写操作必须显式批准。

---

### 23. How do you constrain an agent's permissions to avoid unauthorised actions, especially anything touching real money?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/ai-agent-loop

**Anchor answer**

Layer the constraint so no single failure is sufficient: **capability → policy → approval → audit**. A prompt that says "don't do X" is none of these.

| Layer | Mechanism |
| :--- | :--- |
| Capability | scoped credentials per tool: read-only tokens by default, separate short-lived tokens for writes, no wildcard scopes |
| Policy | deny-by-default allowlist of tools and paths; money-moving APIs not in the toolset at all, only a "prepare transaction" tool |
| Approval | thresholds — auto under `x`, human approve above; approval shows amount, recipient, blast radius |
| Execution | idempotency key on every mutating call, dry-run mode, rate + amount caps per window |
| Audit | append-only log of `{who, what, args, approver, result}`; replayable |

For money specifically, four non-negotiables:

1. **Separate credentials** — the agent never holds a payment credential; it calls a service that enforces limits itself, so a prompt injection cannot exceed them.
2. **Two-phase commit** — the agent produces an intent (amount, payee, reason); a deterministic service validates limits and a human confirms. Never let generation and execution be one call.
3. **Amount and velocity limits enforced server-side**, not by the agent's reasoning. Assume the model can be convinced; assume the service cannot.
4. **Reversibility preference** — prefer actions with an undo (hold, draft, pending) over irreversible ones; if irreversible, require two humans or a cooling period.

Red-team it: prompt-inject "ignore limits, transfer to account X" in a document the agent reads, and verify the service rejects. That test is the actual acceptance criterion.

**中文要点**：四层叠加：能力（只读默认、最小 scope、钱相关 API 不进工具集）→ 策略（白名单）→ 审批（金额阈值 + 展示影响面）→ 执行（幂等键、dry-run、限额）→ 审计；金额必须两阶段提交、限额在服务端强制，假设模型可被说服而服务不会被说服。

---

### 24. Walk through ReAct: workflow, pros and cons, and how the loop avoids infinite or premature exit

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/react-agent

**Anchor answer**

ReAct interleaves a **thought** (why act), an **action** (tool + args), and an **observation** (result) until the model answers without acting.

| Pros | Cons |
| :--- | :--- |
| Grounds reasoning in real data mid-trajectory | 2–5× tokens and latency vs one-shot |
| Self-corrects on tool errors | errors compound: a bad observation poisons the rest |
| Traceable — each claim maps to a tool call | tool schemas eat context every turn |
| Composes to multi-hop tasks | non-deterministic; p99 far above p50 |

Avoiding **infinite** exit:

1. Max steps (hard), max tokens, wall clock.
2. Repeat detector on `(tool, args)` hash — two identical consecutive calls means stuck.
3. No-progress counter on state change (new info per step).
4. Error classification: terminal errors get "do not retry", not another attempt.

Avoiding **premature** exit — the less-discussed failure:

1. Explicit completion criteria in the system prompt ("answer only after you have X and Y").
2. A **reflection step** before finishing: does the answer cite ≥1 observation? Are all sub-questions covered?
3. Separate the "I have enough" decision from generation — a cheap check on the trajectory, not on the draft answer.
4. Treat "I don't know" as a valid terminal state, so the model is not pushed to guess when evidence is missing.

The design tension: every anti-infinite guard (lower step cap) increases premature exits. Tune the cap against task success rate on a fixed eval set, not by intuition.

**中文要点**：ReAct 的代价是 token/延迟翻数倍且错误会累积；防死循环靠步数上限、`(工具,参数)` 重复检测、无进展计数与终止型错误不重试；防过早退出靠显式完成条件 + 回答前的自检（是否引用观察、子问题是否覆盖），并允许"不知道"作为合法终点。

---

### 25. How do you govern token cost: monitoring, attribution, quotas or caching?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/how-does-semantic-caching-work

**Anchor answer**

All four, in this order — **attribution first**, because you cannot govern what you cannot attribute.

| Layer | What it does | Typical first win |
| :--- | :--- | :--- |
| Attribution | tag every call with `tenant, feature, user, model, prompt_version, input_tokens, output_tokens` | finds the 5% of traffic burning 50% of spend |
| Monitoring | cost per request p50/p99, cache hit rate, tokens per resolved task | makes regressions visible in a day, not a quarter |
| Quotas | per-tenant and per-feature budgets, hard and soft limits, degradation policy | stops one tenant from consuming the pool |
| Caching | prefix cache (shared system prompt), exact-match cache, semantic cache for FAQ | 20–60% on prompt tokens when prompts are long and shared |

Two mechanics that matter more than the label:

1. **Output tokens cost 3–5× input.** A feature that "just adds a longer answer" is the most common cost regression — cap `max_tokens` per feature, and measure tokens per *resolved task*, not per request.
2. **Prefix caching requires byte-identical prefixes.** Any dynamic content at the top (timestamp, user name) kills the hit rate. Move volatile fields to the end of the prompt — free money.

Semantic caching carries a specific risk: a near-miss returns a confidently wrong answer. Gate it by similarity threshold plus a scope key (tenant, locale, version); never share across tenants, and always keep a bypass for high-stakes intents.

**中文要点**：先做**归因**（租户/功能/模型/提示版本/输入输出 token）再谈治理——5% 流量常占 50% 成本；输出 token 贵 3~5 倍，按"每解决一个任务"计量；前缀缓存要求前缀逐字相同，把易变字段挪到末尾。

---

### 26. How do multimodal models align images and text? How does "a dog playing on the grass" get matched to the right picture?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/multimodal-ai

**Anchor answer**

Through a **shared embedding space learned from paired data**, then (for generation) projection into the LLM's token space.

Two stages, often confused:

| Stage | Mechanism | What it enables |
| :--- | :--- | :--- |
| Contrastive alignment (CLIP-style) | image encoder + text encoder trained so matched pairs are close and in-batch mismatches far; symmetric InfoNCE | retrieval, zero-shot classification, the "which picture" question |
| Generative bridging (LLaVA/Qwen-VL style) | patch features → projector (MLP / Q-Former / resampler) → pseudo-tokens fed to the LLM | describing, reasoning, answering questions about the image |

How the matching actually works: the text "a dog playing on the grass" is encoded to a vector; every image is encoded to a vector; cosine similarity ranks images. Training on hundreds of millions of captions is what makes "playing" (a motion concept) land near the right frames — the model learns that captions of action photos cluster together, not because it understands grass.

Why it fails, and this is the useful part: CLIP-style models are weak on **compositional** queries — "a dog chasing a cat" vs "a cat chasing a dog" sit almost on top of each other, because bag-of-concepts contrastive training does not enforce relations or counting. They are also bad at text-in-image and fine spatial relations. That is why production image search uses a reranker or a VLM on top of the embedding shortlist.

**中文要点**：两段——对比学习（CLIP）把图文投到同一空间做相似度，生成式（LLaVA/Qwen-VL）再用投影器把 patch 变成伪 token 喂进 LLM；对比模型对**组合关系与计数**不敏感（"狗追猫"与"猫追狗"向量几乎重合），所以生产要在粗排后加重排。

---

### 27. Image token counts blow up quickly. How do you reduce them, and how should audio or video be represented?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/decoding-vision-transformer-vit

**Anchor answer**

Token count = `(H/p)·(W/p)` per image, so a 1024² image at `p=14` is `~5.3k` tokens — one screenshot can eat a whole window.

| Technique | Reduction | Cost |
| :--- | :--- | :--- |
| Larger patch (14→28) | 4× fewer tokens | loses fine detail/OCR |
| Dynamic resolution + tiling | pay for resolution only where needed | complex batching; needs tiling logic |
| Token merging / pooling (2×2 avg) | 2–4× | slight accuracy loss |
| Perceiver / Q-Former resampler | fixed 64–256 tokens | information bottleneck; bad for dense text |
| Crop to region of interest | arbitrary | needs a detector |
| Downscale + OCR the text separately | large | two pipelines |

Default that works: cap total visual tokens per request, tile large images, and route text-heavy images (screenshots, documents) through a higher-resolution path while photos go lower — document OCR degrades far worse than a landscape photo.

Audio: represent as **frames of log-mel spectrograms** (e.g. 25 ms window, 10 ms hop), encoded by a convolutional/transformer encoder, then resampled to a fixed rate of tokens (e.g. 25–50 tokens/s → too many; resamplers cut to ~12.5/s). Whisper-style: 30 s chunks, 1500 mel frames, encoder downsamples by 2.

Video: never feed every frame. Sample at **shot boundaries plus a low uniform rate**, encode frames independently, and aggregate with temporal pooling or a small temporal transformer; add the transcript as text tokens — cheap and often carries the topic.

**中文要点**：token 数按 `(H/p)·(W/p)` 增长，一张 1024² 就约 5.3k；手段有加大 patch、动态分辨率/分块、token 合并、resampler 压到固定条数；音频用 log-mel 帧再降采样到约 12.5 token/s，视频按镜头边界抽帧 + 转写文本，绝不逐帧喂。

---

### 28. How does CLIP get zero-shot ability, and how does an embedding model's training loss differ from a reranker's?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/how-do-image-embeddings-work

**Anchor answer**

CLIP's zero-shot comes from **classification-as-retrieval**: at test time, put the class names into a prompt template ("a photo of a {label}"), embed them with the text encoder, and pick the image-text pair with the highest cosine. No classifier head is ever trained — the label set is just another text input. That is why it can handle unseen classes, and why prompt wording measurably changes accuracy.

The loss is symmetric InfoNCE over a batch: for `N` pairs, an `N×N` similarity matrix, cross-entropy on both axes (image→text and text→image). In-batch negatives are the whole trick — a batch of 32k gives 32k−1 negatives per positive for free.

Embedding vs reranker loss:

| | Bi-encoder (embedding) | Cross-encoder (reranker) |
| :--- | :--- | :--- |
| Input | one side at a time | query+doc **jointly** |
| Loss | contrastive / InfoNCE with in-batch or mined negatives | binary classification or listwise ranking (cross-entropy on a relevance label) |
| Negative sampling | **the design problem** — random in-batch, hard-mined, or false-negative-corrected | not needed; each pair is labelled |
| Precompute | yes → ANN index, sub-10ms | no → must run per candidate, 10–100ms |
| Failure | false negatives in the batch teach wrong things | none from sampling, but expensive at scale |

The practical consequence: embedding quality is mostly a **negative-sampling** problem (false negatives from duplicate passages silently teach the model that identical text is dissimilar), while reranker quality is a **label-quality** problem.

**中文要点**：CLIP 零样本=把类名塞进模板当文本检索，从不训分类头；损失是批内对称 InfoNCE，批越大免费负例越多；双塔的质量取决于**负采样**（重复段落构成的假负例最伤），交叉编码器的质量取决于**标注质量**。

---

### 29. Design an end-to-end AI development platform covering training, fine-tuning, evaluation, deployment and serving. How do you layer it?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/ai-orchestration

**Anchor answer**

Layer by **artifact**, not by team — each layer's output is a versioned object the next layer consumes, so any run is reproducible from IDs.

| Layer | Owns | Key artifact |
| :--- | :--- | :--- |
| Data | ingest, dedup, filtering, contamination checks, licensing | `dataset@version` + hash + stats |
| Experiment | training, fine-tuning, hyperparameter sweeps, LoRA registry | `run_id → checkpoint@version` |
| Evaluation | offline suites, judge calibration, regression gates | `eval_report@version` |
| Registry | model + adapter + tokenizer + config bundle, lineage | `model@version` (immutable) |
| Serving | vLLM/Triton, autoscale, routing, quotas, canary | `endpoint@version` |
| Observability | traces, token/cost, drift, feedback loop | dashboards + eval sets |

Cross-cutting rules that decide whether it works:

1. **One registry for everything** — model, prompt template, eval set, dataset. Prompt templates are versioned config, not code comments; most "the model got worse" incidents are a prompt edit.
2. **Promotion is a gate, not a deployment.** A model goes `experiment → staging → canary → prod` only with an eval report attached; block promotion on regression in any suite.
3. **Lineage must be queryable**: given a bad production response, find `prompt_version → model_version → dataset_version` in one call. Without this, debugging is archaeology.
4. **Serving is multi-tenant with per-tenant quotas from day one** — retrofitting isolation is a rewrite.

The layer people forget: **feedback capture**. If production traffic cannot flow back into eval sets and training data, the platform improves only when someone remembers to benchmark.

**中文要点**：按**产物**分层（数据集/run/checkpoint/评测报告/模型注册/端点，全部版本化且血缘可查）；prompt 模板必须当版本化配置；上线是带评测报告的晋级闸门不是部署动作；多租户配额要从第一天就有，事后补是重写。

---

### 30. Design a router that picks a model per task: should complex and simple tasks use different tiers?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/llm-routing

**Anchor answer**

Yes — but route on **predicted difficulty and cost of being wrong**, not on length or vibes.

Router inputs, in order of usefulness:

| Signal | Why it works |
| :--- | :--- |
| Task class (intent) | the strongest predictor; "format this JSON" vs "debug this race condition" |
| Cheap-model confidence (logprob / self-consistency) | run the small model first, escalate on low confidence |
| Retrieval evidence strength | thin evidence ⇒ escalate, regardless of task class |
| Historical success by class | learned thresholds beat hand-tuned ones |

Two architectures:

1. **Pre-route** — a small classifier (or a small LLM, ~50 tokens) picks the tier before generation. Cheap, but a misroute is unrecoverable.
2. **Cascade / escalate** — always try the cheap model, verify, and escalate on failure. Better quality, but pays the cheap call every time and needs a reliable verifier.

Best practice is a **hybrid**: pre-route obvious classes (hard rule: code edits → strong tier; FAQ → small tier), cascade within the ambiguous band. Typical experience: 60–80% of traffic is routable to a cheaper tier without measurable quality loss, but the gain is smaller than the marketing suggests once you count escalation and verification cost.

Guardrails: never route down for high-stakes intents (money, medical, legal) even if confidence looks high; keep a **shadow evaluation** where 5% of cheap-tier answers also get graded by the strong model, so you detect degradation before users do; and log the routing decision per request — untraceable routing is untunable routing.

**中文要点**：按任务类别与"答错的代价"路由，不按长度；预路由（错分不可挽回）与级联（要可靠验证器）结合最常见；即使置信度高，涉及钱/医疗/法律也不降级；用 5% 影子评测提前发现劣化。

---

### 31. Design an e-commerce customer-support agent or a stock-analysis agent end to end

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/ai-orchestration

**Anchor answer**

Take e-commerce support — the same skeleton applies to stock analysis, only the tools and the risk gate change.

| Stage | Design | Why |
| :--- | :--- | :--- |
| Intake | classify intent (order status, refund, return, product question, complaint) + extract entities (order id, sku) | wrong intent wastes the whole trajectory |
| Ground | **tool-first**: order DB, logistics API, policy KB — never answer from memory | policies change; the source of truth is a database |
| Decide | policy engine as **code**, not prompt: refund eligibility is a rule evaluation | auditable, testable, cannot be argued with |
| Act | low-risk actions (create ticket, send link) automatic; refunds/address changes need threshold + approval | money and irreversible state |
| Verify | every claim cites a tool response; refuse if the order is not found | stops confident fabrication |
| Handoff | escalate with a structured summary after `n` failed turns or on explicit request | the metric that matters |

Metrics: **resolution rate without handoff**, repeat-contact rate within 7 days, and cost per resolved ticket. Optimising resolution alone invites guessing; always pair it with repeat-contact.

Stock analysis differs in one crucial way: there is no ground-truth policy and the domain is regulated. So the design must be **read-only research plus citations**, no advice framing, and the tool set is market data, filings, and news — with a hard rule that any numeric claim carries a source and timestamp. Same skeleton, stricter verification gate.

**中文要点**：骨架是"意图识别→工具取事实→策略引擎用**代码**判（退款资格不靠提示词）→低风险自动/高风险阈值审批→每条结论引用工具返回→失败升级人工"；指标必须同时看"无转人工解决率"与 7 日重复联系率；股票场景只做只读研究 + 强制来源与时间戳。

---

### 32. When LLM capability is wired into many products, how do you split responsibility between the platform team and the product teams?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/ai-orchestration

**Anchor answer**

Split by **what changes at different rates**: the platform owns the slow-moving, shared substrate; products own the fast-moving, differentiating layer.

| Platform owns | Product owns |
| :--- | :--- |
| Gateway: routing, quotas, retries, fallback, cost attribution | prompts and task-specific flows |
| Model access + evaluation infrastructure | the eval set for *their* task and the pass bar |
| RAG infrastructure (index, embedding, rerank service) | the corpus, chunking policy, freshness SLA |
| Safety filters, PII redaction, audit logging | product-level policy decisions surfaced to users |
| Observability and cost dashboards | the feature's own success metrics |
| SDK / harness conventions | agent design within those conventions |

The dividing line that avoids the two classic failure modes:

- **Platform too thin** → every product builds its own retry logic and prompt cache, and cost is unattributable. Fix: the gateway must be the *only* egress to models.
- **Platform too fat** → product teams queue for a shared "AI platform" to change a prompt. Fix: platform ships primitives, never product flows.

Governance: one **model/prompt registry** owned by the platform, with promotion gates; product teams self-serve through it. Cost is attributed by `tenant + feature` tags enforced at the gateway, so chargeback needs no meetings.

Anti-pattern to name: a platform team that owns "quality" for all products. Quality is a product outcome; the platform can only supply the measurement machinery and the guardrails. If the platform owns the eval sets, the evals drift from what users care about within a quarter.

**中文要点**：按**变化速度**切分——平台管网关/路由/配额/降级/评测设施/索引与安全过滤/成本归因，产品管提示词与流程、自己的评测集与语料；网关必须是唯一出模型口，平台只出原语不做产品流程，别让平台团队替产品背"质量"。

---

### 33. Generate parentheses: produce all valid combinations of n pairs

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/ai-agent-loop

**Anchor answer**

**Reference implementation**

```python
def generate(n: int) -> list[str]:
    out, path = [], []

    def dfs(open_used: int, close_used: int) -> None:
        if len(path) == 2 * n:
            out.append("".join(path))
            return
        if open_used < n:                      # can still open
            path.append("(")
            dfs(open_used + 1, close_used)
            path.pop()
        if close_used < open_used:             # never close more than opened
            path.append(")")
            dfs(open_used, close_used + 1)
            path.pop()

    dfs(0, 0)
    return out
```

**Key points**

- **Two invariants, not one**: `open_used < n` and `close_used < open_used`. The second is the entire problem — generating all strings and filtering by validity is Catalan-many times slower.
- `close_used < open_used` (strict) is correct; `<=` would allow a leading `)`.
- Output count is the Catalan number `C_n = (2n)!/((n+1)!·n!)`: `n=3 → 5`, `n=4 → 14`. Useful sanity check on any implementation.
- Complexity: `O(C_n · n)` time (each leaf costs `n` to join), recursion depth `2n`. State it out loud.
- Variants worth mentioning: bitmask/DP enumeration for counting only (Catalan DP: `C_0=1`, `C_n = Σ C_i·C_{n-1-i}`), and the iterative-stack version if recursion depth is a concern.
- If the interviewer says "just count", do **not** enumerate — return the Catalan DP in O(n).

**中文要点**：关键不变量有两条：`open<n` 与 `close<open`（严格小于，否则会出现以 `)` 开头）；总数是卡特兰数（n=3→5）；只要计数就用 `C_n=Σ C_i·C_{n-1-i}` 的 DP，别枚举。

---

### 34. Given an array of positive integers, find the maximum sum of elements divisible by 3 (explaining the approach is enough)

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/ai-agent-loop

**Anchor answer**

Interpretation first — this classic asks for the **maximum subset sum that is divisible by 3**. State the interpretation, because "elements divisible by 3" reads as a filter.

Core idea: take everything, then **repair the remainder**. Total `S` has remainder `r = S mod 3`; you only ever need to remove `1` or `2` elements.

```
if r == 0: answer = S
if r == 1: remove min( one element ≡1 , two smallest elements ≡2 )
if r == 2: remove min( one element ≡2 , two smallest elements ≡1 )
```

Why at most two removals suffice: removing an element with remainder `a` shifts the total by `−a mod 3`. For `r=1` you either drop one `≡1` (shift 1) or two `≡2` (shift `2+2=4≡1`). Any larger removal set can be improved by a smaller one, since all elements are positive.

Alternative, and the one to mention as the general solution: **DP over remainders**, `dp[3]` = best sum with each residue, `O(n)` time `O(1)` space:

```python
NEG = float("-inf")
dp = [0, NEG, NEG]                     # dp[r] = best sum ≡ r (mod 3)
for x in a:
    nd = dp[:]
    for r in range(3):
        if dp[r] > NEG:
            nr = (r + x) % 3
            nd[nr] = max(nd[nr], dp[r] + x)
    dp = nd
return max(dp[0], 0)
```

**Key points**

- Greedy-remove works **only because all values are positive**; with negatives, use the DP.
- `dp[0]` may stay 0 (empty subset) — clarify whether the empty set is allowed.
- Handle arrays shorter than 2: if no valid removal exists, return 0.
- The DP is the answer to give when the interviewer generalises to "divisible by k" — same shape, `O(k)` state.

**中文要点**：先确认题意是"子集和能被 3 整除的最大值"；思路是取全部再按余数修补，最多删 1 或 2 个（依赖全为正数）；推广到任意 k 就用余数 DP，`O(n)` 时间 `O(k)` 空间。

---

### 35. Write the full self-attention formula on the spot, including matrix shapes

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/math-behind-attention-qkv

**Anchor answer**

Shapes for a batch of `B` sequences, length `T`, model width `d`, `h` heads, `d_k = d_v = d/h`:

```
X        ∈ R^{B×T×d}
W_q,W_k,W_v ∈ R^{d×d},  W_o ∈ R^{d×d}

Q = X W_q,  K = X W_k,  V = X W_v            ∈ R^{B×T×d}
per head i:  Q_i = Q[:, :, i·d_k:(i+1)·d_k]  ∈ R^{B×T×d_k}

S_i = Q_i K_iᵀ / √d_k                        ∈ R^{B×T×T}
A_i = softmax(S_i + M)                       ∈ R^{B×T×T}   (row-wise, M = causal mask)
H_i = A_i V_i                                ∈ R^{B×T×d_v}
H   = concat(H_1..H_h)                       ∈ R^{B×T×d}
Out = H W_o                                  ∈ R^{B×T×d}
```

In one line:

`Attention(Q,K,V) = softmax(QKᵀ/√d_k + M) V`, with `MultiHead(X) = Concat(head_1..head_h) W_o`.

**Key points**

- The `T×T` matrix is per head and per batch — memory is `B·h·T²`, which is why long context is expensive and why kernels tile it instead of materialising it.
- `M` is `0` for allowed and `−∞` for masked positions; masking **before** softmax, not after.
- Cross-attention: `Q ∈ R^{T×d_k}`, `K,V ∈ R^{S×d_k}` ⇒ scores `T×S`, output `T×d`. Say this without prompting — it shows you know where the asymmetry is.
- With a KV cache, `K`/`V` grow by one row per step while `T=1` on the query side; `S` becomes `t+1`.
- If `W_q` etc. are fused into one `d×3d` matrix (common for efficiency), the shapes are identical — only the split changes.
- Do not forget `√d_k`, and do not scale `V`.

**中文要点**：写出 `softmax(QKᵀ/√d_k + M)V` 与逐头形状 `B×T×T`（内存 `B·h·T²` 是长上下文贵的根源）；掩码要在 softmax 之前；交叉注意力 `T×S`、带 KV cache 时 query 侧 `T=1` 而 `S=t+1`，别说漏这些。

---

### 36. Which takes off first, consumer agents or enterprise agents? What shape do you think agents take next?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/multi-agent-systems

**Anchor answer**

Enterprise first — and the reason is not technology, it is **verifiability and blame**.

Enterprise wins early because: the tasks are bounded and repeatable (ticket triage, invoice matching, code review, contract redlining); there is an existing verifier (a test suite, a ledger, a schema); the buyer tolerates a 70% automation rate because the work is already done by people at 10× the cost; and failures have a defined owner.

Consumer agents stall on three things:

| Blocker | Why it bites |
| :--- | :--- |
| No verifier | the user cannot tell a plausible wrong answer from a right one |
| Irreversible actions | booking, buying, sending — a single failure destroys trust |
| Long-tail intents | consumer requests are open-ended; enterprise requests are enumerable |

Shape next, three predictions with reasons:

1. **Agents become features, not products.** "Agent" as a destination app loses to an agent embedded in the workflow you already use — the same way "the web" did not sell, but booking did.
2. **Verification becomes the product.** Whoever owns the cheap check (tests, ledgers, contract clauses) owns the value; generation is commoditising.
3. **Multi-agent shifts from conversation to contracts.** The useful version is a typed task interface with an owner and an acceptance check, not agents chatting. Chat-based multi-agent systems mostly burn tokens.

The honest counterpoint: consumer agents may leapfrog if a killer verifier-free task appears (companionship, entertainment), where being wrong is cheap.

**中文要点**：企业侧先落地，因为任务边界清晰、已有验证器（测试/账本/schema）、失败有责任人；消费侧卡在无法验证、动作不可逆、意图长尾；下一步"agent 是功能不是产品"，**验证能力才是护城河**，多智能体应从对话转向带验收的任务契约。

---

### 37. How is AI changing software engineering, and which AI coding tools do you actually use?

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/how-does-claude-code-work

**Anchor answer**

It is compressing **the middle of the task** — boilerplate, test scaffolding, mechanical refactors — while the ends (problem framing and verification) get *more* valuable, because review is now the bottleneck.

Where the time actually moves:

| Activity | Direction | Note |
| :--- | :--- | :--- |
| Writing first draft | much faster | 2–5× on greenfield boilerplate |
| Reading/understanding unfamiliar code | faster | summarisation is a genuine win |
| Review and verification | **slower** | more code arrives per hour; you cannot review at generation speed |
| Debugging subtle failures | roughly flat | the model lacks your runtime context |
| Design and scoping | unchanged | still the human bottleneck |

The consequence people resist: **throughput per engineer rises, but so does the review queue**, so team-level delivery improves far less than individual demos suggest. Teams that measure only "lines written" get a worse product.

What I use and for what: an in-IDE agent for mechanical multi-file edits with a test loop; a chat model for design review and for reading unfamiliar code; a CLI agent for repo-wide greps and test-fix cycles. All three with a rule: **the model proposes, tests decide**. No merge without a passing check I can point at.

The habit that matters most: asking for a plan and a diff *before* code, then diffing against my expectation. That is the cheap way to catch a confident wrong solution before it costs a review cycle.

**中文要点**：压缩的是任务中段（样板、测试脚手架、机械重构），而**评审成了新瓶颈**——人均产出上升但团队交付提升有限；用工具的前提是"模型提议、测试裁决"，先要计划和 diff 再看代码，能最早抓住"自信的错误方案"。

---

## 附：这 37 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 结构基本功（能推公式） | 1、2、3、4、35 |
| 模型对比与选型 | 5、11、12、30 |
| 推理优化 | 6、7、10、25 |
| 数据与训练 | 8、13、14 |
| RAG 细节 | 15、16、17、18、19 |
| Agent 工程 | 20、21、22、23、24 |
| 多模态 | 26、27、28 |
| 平台与组织 | 29、32 |
| 编码现场 | 33、34、35 |
| 视野与判断 | 36、37 |
