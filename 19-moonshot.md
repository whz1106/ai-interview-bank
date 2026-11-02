# 19 · Moonshot AI (Kimi)

> 10 题。出处：`README.en.md` § Moonshot AI (Kimi)。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Kimi's headline feature is very long context. When you push from 8K to hundreds of thousands of tokens, what actually breaks first, and why?

**Source**: https://outcomeschool.com/blog/lost-in-the-middle-problem-in-llms

**Anchor answer**

Not the thing people expect. "Attention is O(n²)" is the textbook answer, but at 256K the bill is survivable with FlashAttention and chunked prefill. What breaks first is **usefulness**, roughly in this order:

1. **Positional out-of-distribution.** Trained at 8K, the model has never seen a 200K distance between question and evidence. RoPE phases fall outside the trained range, attention scores become arbitrary. Degradation starts here, long before anything OOMs.
2. **Attention dilution / lost-in-the-middle.** Softmax over $n$ keys gives each key less mass. A fact at 40% depth of a 200K window is measurably worse than the same fact at 8K — the U-shaped accuracy curve is the signature.
3. **Decode bandwidth.** KV per token is constant while the sequence grows, so decode becomes pure HBM traffic. At 200K × batch, KV dwarfs the weights and throughput collapses even though FLOPs don't.
4. **Prefill latency.** TTFT goes to seconds. Users feel it before they see any accuracy change.
5. **Task data scarcity.** Real 200K documents *with tasks that require the whole window* are rare; needle-in-a-haystack overfits to one skill.

So the fix order is positional extension → long-context continued pretraining → memory/bandwidth engineering → task design. Getting (1) right and skipping (2) yields a model that *accepts* 256K but cannot reason across it — which is the most common way long-context launches disappoint.

**中文要点**：先崩的不是 O(n²) 而是可用性——RoPE 位置外推 OOD → 注意力稀释（lost in the middle）→ decode 变纯带宽瓶颈 → TTFT 爆炸 → 长文档任务数据稀缺。顺序是位置扩展、长文继续预训练、带宽工程、任务设计。

---

### 2. Kimi K2 uses Multi-head Latent Attention (MLA). Explain what it does and how it compares to GQA for KV-cache reduction.

**Source**: https://outcomeschool.com/blog/kv-cache-compression

**Anchor answer**

MLA compresses KV **into a low-rank latent** instead of sharing heads. Each token stores a small latent plus a decoupled RoPE key, and reconstructs per-head $K,V$ by up-projection:

$$c_t \in \mathbb{R}^{512},\quad k^{R}_t \in \mathbb{R}^{64}\;\Rightarrow\; K_t = W^{UK}c_t,\; V_t = W^{UV}c_t$$

At inference the up-projections are **absorbed into** $W^{Q}$ and $W^{O}$, so full K/V is never materialised — the cache holds only $c_t + k^R_t$, 576 numbers per token per layer instead of $2 n_h d_h$.

| | MHA (128 heads) | GQA (8 groups) | MLA |
| :--- | :--- | :--- | :--- |
| KV per token per layer | ~64 KB (BF16) | ~4 KB | ~1.1 KB |
| Quality at equal cache | baseline | small drop | ≈ MHA (reported parity or better) |
| Compute | baseline | baseline | extra up-projections, matmuls in latent space |
| Implementation | trivial | trivial | weight absorption, two RoPE paths, hostile to naive kernel fusion |

**The trade-off in one line:** GQA discards KV *heads* (a quality sacrifice you hope is small); MLA discards KV *rank* (a compression you hope is lossless), paying in kernel complexity — absorbed matrices make weight sharding, chunked prefill and prefix-cache bookkeeping noticeably fiddlier.

**Where it loses:** small models, short contexts, batch 1. At 4K the cache was never the bottleneck, so MLA's implementation cost buys nothing.

**中文要点**：MLA 不共享头而是把 KV 压成低秩 latent（512）+ 解耦 RoPE key（64），推理时把上投影吸收进 $W^Q/W^O$，缓存约 1.1KB/token/层，比 GQA-8 再小 3–4 倍且质量接近 MHA；代价是 kernel 复杂度和权重吸收带来的工程麻烦，短上下文小模型上纯亏。

---

### 3. Kimi K2 is a 1T-parameter MoE with ~32B active per token and hundreds of experts. Explain the routing and the systems cost of training it.

**Source**: https://outcomeschool.com/blog/mixture-of-experts

**Anchor answer**

**Routing.** Sigmoid affinity per expert, top-k selection (K2-style: 8 of 384 routed experts + 1 shared), weights renormalised over the selected set, and a bias term added only for selection so that balance is maintained without an auxiliary loss corrupting the LM objective. Because 384 experts are spread over many ranks, the router output is also a **communication schedule**: which rank each token must visit.

**Systems cost, ranked by how often it ruins your week:**

| Cost | Magnitude | Why it hurts |
| :--- | :--- | :--- |
| All-to-all dispatch + combine | two exchanges per MoE layer, ~60 layers | bandwidth-bound; dominates at small per-rank batch |
| Expert load skew | Zipfian; one hot expert serialises the step | stragglers stall every other rank |
| Optimizer state | Adam needs 2 moments; 1T params ⇒ terabytes even in BF16 | forces ZeRO/offload and huge checkpoints |
| Activation memory | 32B active is small, but 1T total must stay resident | constrains PP/EP degree, hence batch per GPU |
| Routing instability | logits drift → expert collapse mid-run | needs the QK-clip / bias control loop |

**The counterintuitive bit:** the FLOPs are cheap (32B active ≈ a 32B dense model per token), so the naive estimate says "1T model at 32B cost". In practice the *step time* is set by the all-to-all and by the slowest expert, not by the GEMMs. Sparse models buy compute efficiency and spend it back on communication.

**中文要点**：路由是 sigmoid 亲和度 + top-8/384 + 1 个共享专家，bias 只参与选路不进权重；真正的成本不在 FLOP（≈32B 稠密）而在每层两次 all-to-all、专家负载倾斜导致的慢 rank、TB 级优化器状态与权重常驻。

---

### 4. How do you take a model trained at 8K–32K and make it work at 128K or more?

**Source**: https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding

**Anchor answer**

Three layers, in order, and each is independently necessary.

**1. Positional scheme (no training required, but not sufficient alone)**

| Method | What it does | Cost |
| :--- | :--- | :--- |
| Position interpolation | squeeze positions into the trained range | blurs high-frequency dims, hurts local order |
| NTK / YaRN (by-parts) | interpolate low-frequency dims, keep high-frequency; scale attention temperature by $\sqrt{1/t}$ | needs a `rope_scaling` config and per-model tuned factor |
| Base-frequency bump ($10^4 \to 10^6$) | coarser rotation ⇒ longer wavelengths | cheap, but degrades short context unless mixed |

**2. Continued pretraining.** A few billion tokens of genuine long documents (books, repos, long dialogues) at the target length. This is where the model learns that token 150,000 is *relevant*, not just addressable. Data must be length-stratified — a corpus of 128K documents made by concatenating unrelated 2K pages teaches nothing about long-range dependencies.

**3. Long-context SFT + evals.** Tasks that require spanning the window (multi-doc QA, repo-level code, long summarisation). Evaluate with RULER-style multi-hop and variable-depth retrieval, not just a single needle — needle tests saturate long before real reasoning does.

**Pitfall:** always mix in short-context data (10–30%) during extension or you trade 128K ability for a measurable regression at 2–8K, and users notice that far more often.

**中文要点**：三层缺一不可——位置方案（PI/YaRN/提 base，注意温度校正）、真实长文继续预训练（长度分层、别拼接无关联短文）、长文 SFT 与 RULER 式多跳评测；扩展时必须混入 10–30% 短上下文数据，否则 128K 换来 2–8K 明显掉点。

---

### 5. Walk me through why you would disaggregate prefill and decode onto separate machines, as Mooncake does. What does that buy you and what does it cost?

**Source**: https://outcomeschool.com/blog/prefill-decode-disaggregation

**Anchor answer**

**Why they conflict.** Prefill is compute-bound (large GEMMs, high arithmetic intensity); decode is memory-bandwidth-bound (one token per sequence, reading the whole KV cache). Put them in one batch and the scheduler must choose a compromise chunk size, and every decode step inherits prefill's latency jitter.

**What disaggregation buys:**

| Gain | Mechanism |
| :--- | :--- |
| Independent scaling | prefill pool sized by TTFT SLO, decode pool by TPOT SLO — different GPU counts, different parallelism |
| No head-of-line blocking | a 100K-token prefill no longer stalls everyone's decode step |
| Cache reuse | decode nodes hold a **global KV pool** (DRAM/SSD tiered) keyed by prefix hash, so shared system prompts hit cache across requests |
| Better utilisation | each pool runs its natural batch shape |

**What it costs:**

- **KV transfer.** A 100K-token context is hundreds of MB per request; moving it over RDMA takes tens of ms and can exceed the decode it enables. Long context makes transfer the new bottleneck — hence layer-wise streaming and overlapping transfer with compute.
- **An extra hop** in the critical path plus a global scheduler that must track cache locality, or you ship tokens to a node that has to re-fetch.
- **Failure surface**: two pools, two failure domains, cache coherence on weight updates.

**When not to:** short prompts, low concurrency, single-node deployment — transfer and scheduling overhead dominate and you are slower than a colocated engine.

**中文要点**：prefill 算力密集、decode 带宽密集，混批互相拖累；分离后可独立扩缩、消除队头阻塞、做全局 KV 池前缀复用；代价是 KV 传输（长上下文下几百 MB，可能比它换来的 decode 还贵）、多一跳和全局调度复杂度。短 prompt 低并发时纯亏。

---

### 6. A chat assistant re-sends a long conversation history on every turn. How do you avoid recomputing all of it, and what are the pitfalls?

**Source**: https://outcomeschool.com/blog/how-does-prompt-caching-work

**Anchor answer**

**Mechanism: prefix KV caching keyed by content hash.**

1. Tokenize the request; split into fixed-size blocks (16–128 tokens).
2. Hash each block **including its position and all preceding blocks' hashes** — a radix tree / prefix trie over block hashes.
3. Look up the longest matching prefix; reuse its KV blocks, prefill only the suffix.
4. Append the new blocks to the cache after the step.

Multi-turn chat is the ideal case: each turn's history is a strict extension of the last, so turn $n$ prefills only the new user message + the assistant's own reply.

**Pitfalls, in order of how much they cost:**

| Pitfall | Consequence |
| :--- | :--- |
| Cache key omits model version / LoRA adapter / tokenizer | silently serving stale or wrong KV — wrong outputs, not errors |
| Block granularity | a 1-token change at position 5000 invalidates everything after it; keep the volatile part (system prompt, timestamps) **last** |
| Eviction policy | LRU on blocks fights the running batch for HBM; tier to DRAM/SSD instead of dropping |
| Prefix sharing across tenants | timing side-channel can leak prompt content; hash with a per-tenant salt for sensitive deployments |
| Prefix layout | any dynamic content injected at the top of the prompt destroys the hit rate — a measured 0% vs 80% difference in the same product |

Rule of thumb: put everything stable first, everything volatile last, and instrument **prefix hit rate** as a first-class metric next to TTFT.

**中文要点**：按块哈希（含位置与前缀哈希）建 radix 树，命中前缀直接复用 KV，只 prefill 增量；坑在于缓存键漏了模型版本/LoRA/分词器、块粒度导致一处改动全失效、淘汰策略和运行批抢显存、跨租户前缀共享有侧信道、以及把动态内容放 prompt 开头直接把命中率打到 0。

---

### 7. For a long-context assistant, when is a 1M-token context window the right tool, and when should you use retrieval instead?

**Source**: https://outcomeschool.com/blog/lost-in-the-middle-problem-in-llms

**Anchor answer**

The decision is about **corpus size, query shape, and cost per query** — not about which is more modern.

| Situation | Choose | Why |
| :--- | :--- | :--- |
| One document / one repo / one case file, per query | **long context** | whole-document reasoning, cross-references, no chunk boundaries to lose |
| Need global synthesis ("summarise themes across all 400 tickets") | **long context** (if it fits) | retrieval can't answer questions whose evidence is the *set* |
| Corpus > window, or continuously updated | **retrieval** | the window can't hold it, and re-prefilling each query is ruinous |
| Need citations / auditability | **retrieval** | every claim traces to a chunk ID |
| Needle-style fact lookup | **retrieval** | cheaper and more accurate; long context dilutes attention |
| Latency-critical, high QPS | **retrieval** | prefill is O(n) compute *per query* unless prefix-cached |

**The cost argument interviewers want:** a 1M-token window costs the same prefill as processing a million tokens — roughly a whole novel — on *every* turn that isn't cache-hit. A retrieval path reads ~10K tokens. So long context is a latency/quality purchase, not a free upgrade.

**The hybrid that actually ships:** retrieve and rerank to 50–200K tokens, then let the long window do the synthesis. That keeps citations and cost sane while preserving the cross-document reasoning that pure RAG loses at chunk boundaries.

**中文要点**：判断看语料规模、问题形态、单次查询成本——单文档/整库全局综合用长上下文，语料超窗、需引用、高 QPS 用检索；1M 窗口每轮 prefill 等于读一整本书，检索只读万级 token。落地做法：先检索重排到 50–200K，再交给长窗口做综合。

---

### 8. Training a trillion-parameter model, attention logits can blow up and destabilise the run. What is going on, and how does something like MuonClip address it?

**Anchor answer**

**Mechanism of the blow-up.** With $\text{softmax}(q^\top k/\sqrt{d})$, the magnitude of $q^\top k$ is governed by $\|q\|\|k\|$. Both grow during training — with weight decay, the *product* of two growing norms grows quadratically, not linearly. Under an optimizer like Muon, whose update is orthogonalised to a fixed RMS, the per-step scale is uniform across the matrix, so a positive feedback loop is easy: bigger $q\cdot k$ → sharper softmax → lower entropy → larger gradients on the few winning logits → bigger weights. Empirically the first symptom is a **loss spike**, then divergence; the entropy of the attention distribution hits zero and gradients vanish or explode.

**What MuonClip does.** It is a **post-step QK-clip**: after each optimizer update, compute the maximum attention logit per head over a probe batch. If it exceeds a threshold $\tau$ (order 100), rescale that head's $W_q$ and $W_k$ by $\sqrt{\tau/\max}$:

$$W_q, W_k \leftarrow W_q, W_k \cdot \sqrt{\tau / \max_{t,s} \logit_{t,s}}$$

Because the scale is applied to *both* matrices, the logit is clipped while the direction of $q$ and $k$ is untouched — unlike gradient clipping, which perturbs the update, or soft-capping logits, which changes the model's function.

**Why it matters practically:** it is a hard safety rail that costs one reduction per step, so you can keep an aggressive learning rate and Muon's fast convergence instead of backing off to a stable-but-slow regime. The reported result is a 1T-scale run with no loss spikes across trillions of tokens.

**中文要点**：logits 失控源于 $\|q\|\|k\|$ 随训练二次增长 + 优化器固定 RMS 更新形成正反馈，表现为 attention 熵塌缩与 loss spike；MuonClip 在每步后按 head 用 $\sqrt{\tau/\max}$ 同时缩放 $W_q,W_k$，只压幅度不改方向，等于给激进学习率加一道廉价护栏。

---

### 9. Kimi K1.5 scaled RL for reasoning without a process reward model or tree search. Why deliberately keep the RL recipe that simple?

**Anchor answer**

Because PRM and MCTS each buy a small quality bump for a large amount of machinery, and that machinery is where the complexity goes to die.

| Component | What it costs | What it buys |
| :--- | :--- | :--- |
| Process reward model | step-level labels or an annotator model; reward hacking against the PRM; a second model to train and serve | denser credit assignment |
| Tree search (MCTS) | $N\times$ rollout compute per sample; value estimation; non-trivial batching | better exploration |
| **Long-CoT + outcome reward** | just rollouts and a verifier | the same emergent behaviours — reflection, backtracking, self-correction |

**Why the simple version works:** for verifiable domains, correctness of the final answer is an *unbiased* reward. Long CoT gives the model room to search *inside* its own context; RL then discovers that backtracking raises expected reward. The search moves from the decoding procedure into the token stream — no tree, no value head, no PRM to be gamed.

**Deliberate design choices that make simplicity viable:**

- **Length control** via a curriculum, otherwise "think longer" is free reward and you get 30K-token rambling.
- **Curriculum over difficulty** — easy problems first, so the pass rate is never 0 (zero-variance groups give zero GRPO gradient).
- **Rejection sampling + SFT** afterwards to compress long CoT into short CoT ("long2short"), which recovers latency.

**When you *do* need a PRM:** unverifiable domains (writing quality, medical advice) where the outcome signal is noisy and step-level judgements carry most of the information.

**中文要点**：PRM 要步级标注且容易被刷，MCTS 要 $N$ 倍 rollout 与价值估计；可验证域里最终答案对错本身就是无偏奖励，长 CoT 把搜索挪进 token 流里，所以不需要树也不需要 PRM。前提是长度课程 + 难度课程（避免零方差组）+ 事后 long2short 压缩；不可验证域才值得上 PRM。

---

### 10. Kimi K2 targets agentic and coding tasks. How would you evaluate whether an agentic model is actually good, beyond a single benchmark number?

**Source**: https://outcomeschool.com/blog/ai-agent-evaluation

**Anchor answer**

One number hides four different failure modes. Decompose into a scorecard and run it on **held-out, non-public** tasks.

| Dimension | Metric | What it catches |
| :--- | :--- | :--- |
| Task success | pass@1 with hidden tests; pass^k for consistency | luck vs reliability |
| Trajectory efficiency | steps, tokens, tool calls, wall-clock per success | "solved it in 60 calls" is a product failure |
| Tool discipline | invalid-call rate, schema-repair rate, hallucinated tool names | instruction following under pressure |
| Error recovery | success after an injected failure (500, timeout, bad arg) | the actual agentic skill |
| Robustness | paraphrase / reorder / distractors; success delta | overfitting to the eval prompt |
| Safety | destructive-action rate, permission violations | the one that ends the launch |
| Cost | $ per solved task, incl. retries | decides whether it ships |

**Methodology rules that keep you honest:** sandbox with deterministic resets; run each task $n \ge 5$ times and report variance and flakiness separately from capability; keep a private suite because public benchmarks leak into training data; and include a "human baseline" on a sample so you know whether 62% is good.

**The counterintuitive finding worth saying:** models that win on single-shot benchmark accuracy often lose on pass^k, because pass@1 rewards one lucky path while production demands the *same* path succeed repeatedly. Track both, and weight consistency higher for anything user-facing.

**中文要点**：单指标会掩盖四种失败模式——用"任务成功率（含 pass^k 一致性）+ 轨迹效率 + 工具规范 + 故障注入后恢复 + 扰动鲁棒性 + 破坏性动作率 + 每任务成本"的多维记分卡，跑私有 held-out 集、每任务 ≥5 次报方差；单发准确率高的模型常常 pass^k 更差，上线看一致性。

---

## 附：这 10 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 长上下文真本事 | 1（先崩什么）、4（8K→128K）、7（窗口 vs RAG） |
| 架构与显存 | 2（MLA vs GQA）、3（1T MoE 成本） |
| 推理服务 | 5（PD 分离）、6（前缀缓存） |
| 训练稳定性 | 8（logits 爆炸 / MuonClip） |
| 认知与判断 | 9（为什么不要 PRM）、10（agentic 评测） |
