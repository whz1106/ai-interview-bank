# 29 · ByteDance

> 38 题。出处：`README.en.md` § ByteDance。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Walk through self-attention end to end, and explain why the dot product is divided by sqrt(d_k)

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/math-behind-attention-qkv

**Anchor answer**

Project each token embedding `x ∈ R^d` into `Q = xW_q`, `K = xW_k`, `V = xW_v`, all `R^{d_k}`. For a sequence of length `n`, scores `S = QKᵀ/√d_k ∈ R^{n×n}`, then `A = softmax(S)` row-wise, then `O = AV` and a final output projection `W_o` back to `d`. With `h` heads you do this in parallel on `d_k = d/h` slices and concatenate.

Why divide by `√d_k` — the real mechanism:

| Step | Effect |
| :--- | :--- |
| `q·k = Σ q_i k_i` over `d_k` terms | variance grows **linearly** with `d_k` if entries are i.i.d. mean 0, var 1 |
| std of the score | becomes `√d_k`, not 1 |
| unscaled softmax at `d_k = 128` | logits spread ±10+ → one position gets probability ≈1 |
| consequence | gradients through softmax vanish; attention saturates on a single token |
| after scaling | score variance ≈1, softmax stays in a usable temperature range |

The practical hook: if you drop the scaling, training does not diverge, it **stalls** — loss plateaus because every head collapses onto one position and stops passing gradient. This is why QK-norm (normalising q and k instead) is used in some long-context models rather than relying on `1/√d_k` alone.

**中文要点**：除以 `√d_k` 是把点积方差从随维度线性增长拉回 1；不除不会爆炸而是**软最大化饱和、梯度消失**，训练是停滞而非发散。

---

### 2. Trace the evolution from MHA to MQA, GQA and MLA. What does each step actually save?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/grouped-query-attention

**Anchor answer**

All four keep the same math and shrink only the **KV cache**, which is what actually limits decode concurrency.

| Variant | K/V heads | KV cache vs MHA (`h` heads) | What it buys / costs |
| :--- | :--- | :--- | :--- |
| MHA | `h` | 1× (baseline) | best quality, worst memory |
| MQA | 1 | `1/h` | huge memory win, quality drop and unstable training at scale |
| GQA | `g` (e.g. 8) | `g/h` | near-MHA quality at a fraction of cache; standard today |
| MLA | low-rank latent `c` | ~`d_c/h·d_h` compressed | best compression, but needs weight absorption + custom kernels |

The numbers that matter: cache per token is `2 · n_layers · n_kv_heads · d_head · bytes`. For a 70B-class model with 80 layers, `d_head=128`, fp16: MHA (64 KV heads) ≈ 2.6 MB/token; GQA-8 ≈ 0.33 MB/token — an 8× concurrency difference on the same GPU.

MLA compresses KV into a shared latent and reconstructs per-head K/V on the fly; DeepSeek adds a decoupled RoPE dimension so positional encoding survives the compression. Cost: you cannot use stock attention kernels, and the cache must be decompressed or absorbed into `W_q`/`W_o`.

When not to bother: if your workload is prefill-heavy (short outputs, long prompts), the KV cache is not the bottleneck — optimise attention FLOPs instead.

**中文要点**：MHA→MQA→GQA→MLA 只在压缩 **KV cache**（decode 并发上限）；GQA-8 相对 MHA 是 8 倍显存差，MLA 压得更狠但要吸收权重和自研 kernel。

---

### 3. How does Qwen3 switch between thinking and non-thinking mode? How do you train and serve one model that does both?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/large-reasoning-models

**Anchor answer**

It is **one set of weights, two behaviours**, controlled by prompt template plus a budget, not two models.

Four things make it work:

1. **Special tokens.** A distinct template inserts the think block opener; the non-thinking template omits it, so the model never enters the reasoning loop. A stop token or the template itself forces the answer section.
2. **Budget control.** `thinking_budget` caps generated reasoning tokens; when hit, the runtime injects the end-of-thinking token and the model continues into the answer. Instruction following on "think at most N tokens" must be trained, since it is not free.
3. **Staged training.** Long-CoT SFT/RL for reasoning ability, then a short "thinking-mode fusion" stage that mixes thinking and non-thinking data so both distributions stay sharp. Without the fusion stage the non-thinking mode degrades notably.
4. **Serving.** Same KV cache and weights; thinking mode just costs more output tokens. Routers pick the mode by task difficulty, and the token budget is the real latency lever.

Failure mode to name: a budget cutoff in the middle of a derivation produces a confidently wrong short answer — the model must be trained to *abandon* the chain and answer, and to emit "insufficient budget" style hedging when it cannot.

**中文要点**：同一套权重靠**模板 + 思考预算**切换；关键是训练时做 thinking/non-thinking 融合，否则关掉思考能力会明显退化；预算截断会逼出错误短答案，需专门训练放弃与兜底。

---

### 4. Can a decoder-only model beat an encoder-only one on a classification task?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/encoder-vs-decoder-in-transformers

**Anchor answer**

Yes — usually when labels are few, the label space is open, or the input is long and the task needs reasoning rather than surface pattern matching.

| Regime | Winner | Why |
| :--- | :--- | :--- |
| ≥100k labelled examples, fixed 5-class intent | encoder (BERT-class) | bidirectional context, cheap, linear head, fast inference |
| few hundred labels, or zero-shot | decoder | pretrained instruction following, no head training |
| label set changes weekly | decoder | no retrain; put labels in the prompt |
| long document + "why" reasoning | decoder | generation over rationale beats a CLS vector |

Mechanism: an encoder sees both directions, so each token representation already aggregates full context — ideal for a single-vector decision. A decoder's causal mask means the last token only sees left context; in practice you use the final position (or mean-pool), which is weaker per token but far better pretrained for instruction semantics.

The honest engineering answer: **cost per 1k requests is 10–100× higher for a decoder**, so a decoder only "beats" an encoder when the accuracy gap exceeds the cost gap or when you avoid a labelling programme entirely. The reliable production pattern is: encoder for the 90% head of the distribution, decoder for the tail and for hard cases escalated by confidence.

**中文要点**：少样本、标签空间常变、长文档要推理时 decoder 更优；但单次推理成本高 10~100 倍，工程上常用"小模型兜大头 + 大模型升级尾部"。

---

### 5. Why does the FFN expand and then contract? What does SwiGLU buy you over ReLU or GELU?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/feed-forward-networks-in-llms

**Anchor answer**

The expand-then-contract shape (`d → 4d → d`) exists to give the model a **wide, per-token nonlinear feature space** that attention cannot provide. Attention mixes information across positions but is a linear weighted sum; the FFN is where each token is transformed nonlinearly and independently. A wider hidden layer means more independent "key-value memories" — the common reading is that the first matrix acts as pattern detectors and the second as a lookup that writes results back into the residual stream.

SwiGLU changes the activation and the parameter count:

| Variant | Form | Hidden size used | Notes |
| :--- | :--- | :--- | :--- |
| ReLU/GELU MLP | `W₂·act(W₁x)` | `4d` | 2 matrices, dead-unit risk with ReLU |
| SwiGLU | `W₂(swish(W₁x) ⊙ W₃x)` | `≈ 8d/3` | 3 matrices, multiplicative gating |

Two facts worth stating: the `8/3` is chosen so SwiGLU matches a `4d` MLP's parameter count, and the gate gives a **data-dependent multiplicative path** that empirically improves quality at equal params (the PaLM result). The price is one extra matmul, so kernels and tensor-parallel sharding need care — and SwiGLU's `w1·w3` fusion is why some serving stacks see a different FLOP profile than their roofline model predicted.

**中文要点**：FFN 是逐 token 的非线性特征变换（注意力的补充），先扩后缩是为更宽的键值记忆；SwiGLU 用门控乘性交互在同等参数量（隐藏维取 8/3·d）下效果更好，代价是多一个矩阵。

---

### 6. Take an 8x7B model: roughly what share of GPU memory goes to KV cache versus weights? What else occupies memory?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/kv-cache-in-llms

**Anchor answer**

Weights are fixed; KV cache scales with batch × context, so the share depends entirely on the traffic assumption.

Weights: 8 × 7B ≈ 56B params. In fp16 that is **~112 GB**, plus roughly 1–2 GB of buffers. On 8×80GB = 640 GB that leaves ~520 GB for everything else.

KV cache per token: `2 · L · n_kv · d_head · bytes`. For a Mixtral-style 7B expert (32 layers, 32 heads, `d_head=128`, fp16, no GQA) ≈ **512 KB/token**. At 8k context that is ~4 GB per sequence — so ~120 concurrent 8k sequences fills the rest.

What else occupies memory, and is usually forgotten:

| Consumer | Order of magnitude | Notes |
| :--- | :--- | :--- |
| Activations / workspace | 1–5 GB | depends on prefill batch and TP |
| CUDA graphs + kernels | 1–3 GB | static capture per batch bucket |
| Fragmentation | 10–20% of free | why PagedAttention exists |
| Expert weights replicated | model-dependent | EP/TP decides duplication |
| Prefix cache pool | configurable | shared, evictable, competes with KV |

The hook: on MoE models the *weights* dominate and KV is a minor term; on dense long-context serving the ratio inverts and KV becomes the binding constraint.

**中文要点**：56B 权重 fp16 约 112GB，KV 每 token 约 512KB（8k 上下文≈4GB/条）；还有激活、CUDA graph、碎片和前缀缓存池。MoE 是权重主导，长上下文稠密服务则反过来。

---

### 7. A prefix cache hits, but the KV blocks have been offloaded to SSD. What is the fetch path, and does it have to go through host memory on the way back to the GPU?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/how-does-prompt-caching-work

**Anchor answer**

Path: **SSD → (page cache / host DRAM staging) → pinned host buffer → H2D copy → GPU HBM**, then the block table is remapped and prefill skips the cached prefix.

Does it have to go through host memory? For plain NVMe, yes — the device cannot DMA into GPU memory on most server topologies, so a bounce buffer is unavoidable. Two ways around it:

| Path | Mechanism | When |
| :--- | :--- | :--- |
| GPUDirect Storage | NVMe → GPU HBM directly, bypassing the bounce buffer | needs supported NIC/PCIe topology and driver support |
| GDS-less optimisation | large pinned staging buffer + `cudaMemcpyAsync`, overlapped | the default |

Cost model that decides whether the hit is worth it: at ~3–7 GB/s per NVMe device, a 100k-token prefix at ~50 MB of KV takes ~10 ms of I/O, but the H2D copy and the block-table remap add latency; a second hit on the same prefix is far cheaper because the page cache already holds it. So **the second hit is the good one**.

The failure mode to name: if eviction happens while a request is waiting on the fetch, two requests can thrash the same blocks. You need a reference count on offloaded blocks and a strict "never evict a block that is in-flight" rule, otherwise you get recompute storms that look like random latency spikes.

**中文要点**：SSD→页缓存/主机暂存→pinned 缓冲→H2D→HBM，普通 NVMe 必须过主机内存（GPUDirect Storage 才能绕开）；第二次命中才便宜，离载块要被引用计数保护否则抖动。

---

### 8. What do PagedAttention and continuous batching each solve in vLLM?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/how-does-vllm-work

**Anchor answer**

They solve different problems at different layers, and neither is sufficient alone.

| | PagedAttention | Continuous batching |
| :--- | :--- | :--- |
| Layer | **memory allocator** | **scheduler** |
| Problem | KV cache had to be one contiguous per-sequence block, so you reserved for max length | a batch waited for the longest sequence before admitting new ones |
| Mechanism | fixed-size blocks + per-sequence block table, like OS virtual memory | iteration-level admission: finished sequences leave, new ones join each step |
| Saves | internal + external fragmentation, enables sharing and swapping | GPU utilisation during ragged generation |

Numbers people remember: fragmentation used to waste 60–80% of KV memory; block allocation brings that to single-digit percent for most workloads.

Interaction that matters: paging only pays off because the scheduler can swap or preempt at block granularity; continuous batching only pays off because the allocator can grow a sequence block by block. Together they turn decode from "batch waits for the worst case" into "throughput ≈ sum of independent token rates".

Costs to mention: block tables add an indirection in the attention kernel (custom kernel required), and aggressive continuous batching increases TTFT jitter for short requests unless you have chunked prefill or priority scheduling.

**中文要点**：PagedAttention 是**显存分配器**（块表 + 共享，消除 60~80% 碎片），连续批处理是**调度器**（逐 iteration 进出）；两者互为前提，代价是注意力需专用 kernel、短请求 TTFT 抖动。

---

### 9. Explain speculative decoding. Which models does it not help?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/speculative-decoding

**Anchor answer**

A cheap draft proposes `k` tokens; the target model verifies them in **one** forward pass; accept the longest correct prefix and resample the first rejected position. Because verification is a single batched forward, you pay target-model cost once for up to `k` tokens. Guarantee: with the standard rejection-sampling correction, the output distribution is **identical** to the target model's — it is a pure latency trick, not an approximation.

Speedup ≈ `accepted_tokens / (1 + draft_cost/verify_cost)`. So it helps exactly when acceptance rate is high and the draft is cheap.

| Not helpful when | Why |
| :--- | :--- |
| target is already small / heavily batched | decode is bandwidth-bound and near-saturated; draft overhead adds latency |
| high-entropy or creative output | acceptance collapses to ~50–60%, ratio ≈1 |
| draft and target from different distributions | you pay verification for tokens you reject |
| strict low-latency single-stream with a slow draft | draft cost is serial and dominates |
| very high batch at high QPS | you are throughput-bound; speculative decoding mostly buys nothing and costs KV memory |

The counter-intuitive point: at large batch sizes speculative decoding often **reduces** throughput, because verification wastes compute on rejected tokens while you were already compute-bound. It is a single-stream latency tool. Media/video, quantised targets and domain-shifted drafts (e.g. code draft for a chat target) are the classic non-helpers.

**中文要点**：小模型起草、大模型一次前向并行验证，输出分布严格不变；收益取决于接受率/起草成本比；大 batch 已算力饱和时它反而降吞吐，属单流延迟优化。

---

### 10. Why does prefill/decode disaggregation raise overall throughput, and when is it worth it for you?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/prefill-decode-disaggregation

**Anchor answer**

Because prefill and decode have **opposite bottlenecks** and interfere when co-located.

| Phase | Bound by | Ideal hardware posture |
| :--- | :--- | :--- |
| Prefill | compute (large GEMMs, high arithmetic intensity) | high SM utilisation, big batches |
| Decode | memory bandwidth (one token, weight streaming) | low compute use, latency-sensitive |

Co-located, a long prefill stalls every decoding sequence in the batch — the classic p99 TTFT spike while TPOT is fine. Disaggregation puts them on separate pools, so each is tuned independently (chunk size, batch size, even different GPU types), and KV is transferred over NVLink/RDMA.

Worth it when:

- **TTFT and TPOT SLOs conflict** and you serve long prompts with short outputs (chat with big system prompts, RAG).
- Utilisation is low because prefill bursts are rare but violent.
- You can afford the KV transfer: a 10k-token prefix is tens of MB, over RDMA that is single-digit milliseconds.

Not worth it when the cluster is small — you lose flexibility and the transfer cost eats the gain — or when prompts are short enough that prefill is not a visible stall. Vendor benchmarks showing 2× throughput usually assume long prompts plus tight TPOT, which is the best case.

**中文要点**：prefill 算力受限、decode 带宽受限，混部时长 prefill 会卡住整批 decode 造成 TTFT 尖刺；拆池可各自调优，代价是 KV 传输；小集群或短 prompt 不值得。

---

### 11. Explain GRPO. Is the KL term in GRPO the same as the one in PPO, and what changes when you put KL in the loss versus in the reward?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/group-relative-policy-optimization-grpo

**Anchor answer**

GRPO drops the critic. For a prompt it samples a group of `G` completions, scores each with the reward, and uses the **group-normalised** reward as the advantage: `Â_i = (r_i − mean(r)) / std(r)`. No value network, so no value-loss, no separate critic pass — the group mean is the baseline.

KL is **not** the same. PPO's KL is a *trust region* against the old policy used inside the clipped surrogate (a ratio penalty, usually `k3` estimator). GRPO's KL is against a **frozen reference policy** to prevent drift from the SFT model — a different purpose. GRPO also often uses the unbiased `k3` estimator `exp(Δ) − Δ − 1` rather than PPO's `log` estimator, because the log version is unbiased but has high variance.

Where the KL lives changes the gradient:

| Placement | Effect |
| :--- | :--- |
| **in loss** (per-token, differentiable) | dense per-token signal, fine-grained control, but it perturbs every token's gradient and can fight the advantage term |
| **in reward** (subtracted from scalar `r`) | simpler, but KL becomes constant within a completion → only a sequence-level penalty; a long bad tail is punished once, weakly |

Practically, loss-side KL keeps the policy closer per token; reward-side KL is easier to tune but lets individual tokens drift. Many stacks just use loss-side `k3` with a small coefficient and monitor the empirical KL.

**中文要点**：GRPO 用组内归一化奖励当优势、去掉 critic；它的 KL 是**对冻结参考策略**的约束（PPO 的 KL 是相对旧策略的信任域）；放 loss 是逐 token 稠密约束，放 reward 变成序列级一次性惩罚。

---

### 12. Walk through the two stages of DeepSeek-R1. How are the reward function and reward model designed?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/large-reasoning-models

**Anchor answer**

R1-Zero is pure RL from the base model; the published R1 adds two SFT stages to fix readability and general capability.

| Stage | Data | Objective |
| :--- | :--- | :--- |
| R1-Zero | none | GRPO on base model with rule rewards only → reasoning emerges, but output is unreadable and mixes languages |
| Cold-start SFT | a few thousand long-CoT samples | fixes format, makes the reasoning readable before RL |
| Reasoning RL | verifiable prompts | GRPO with **rule-based** rewards |
| Rejection-sample SFT | ~600k reasoning samples filtered by correctness | distil reasoning into the base model, then general SFT (~200k) for writing/QA |

The reward design is the point. For reasoning they use **rule-based** rewards only: (1) accuracy — string/boxed-answer match against a verifier for math, code, logic; (2) format — whether the chain is inside the think tags. **No neural reward model** at this stage, because a learned RM is hackable and adds reward-hacking surface over long chains. In the general-alignment stage they keep a **preference reward model** for helpfulness/harmlessness.

Key trade-off: rule rewards are objective but only cover verifiable domains, so the syllabus must be machine-checkable; a preference RM generalises but is where reward hacking enters.

**中文要点**：R1-Zero 纯 RL + 规则奖励，R1 加冷启动 SFT、推理 RL、拒绝采样 SFT（约 60 万）与通用 SFT；推理阶段刻意**不用神经奖励模型**，只用答案正确性 + 格式两条规则，避免长链被刷分。

---

### 13. Are PPO and GRPO on-policy or off-policy? How is the PPO critic trained?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/proximal-policy-optimization-ppo

**Anchor answer**

Both are **on-policy algorithms run in a slightly off-policy way**. Data must come from the current (or very recent) policy; the clip exists precisely because you reuse each rollout for several gradient steps, which makes the data off-policy by a small `ε`. PPO's clip bounds the ratio to `[1−ε, 1+ε]`; GRPO relies on the same clip plus a KL to a reference model. True off-policy would mean a replay buffer of old policies — neither does that.

PPO's critic (`V_φ`) is trained by regression to the **returns**, and the detail that trips people is *which* returns:

1. Rollout gives `(s, a, r)` plus a value for each state.
2. Compute `GAE(λ)`: bootstrapped TD residuals over the trajectory, `δ_t = r_t + γV(s_{t+1}) − V(s_t)`.
3. Fit `V_φ` to `GAE + V` (the value target) by MSE, usually with **clipping** `V_clip` and, in RLHF, often initialised from the reward model.
4. The advantage for the policy is `GAE` itself, typically normalised per batch.

In LLM RLHF the "state" is the prefix, the "action" is the next token, and `γ=1` — so the critic is just predicting the sequence's final reward from each prefix, i.e. a token-level value head. It is expensive: a second full model, and its error directly biases every advantage.

**中文要点**：两者都是 on-policy 但靠裁剪多次复用同一批 rollout，属"轻微 off-policy"；critic 用 GAE 目标做 MSE 回归（RLHF 中 γ=1，等于从每个前缀预测最终奖励），代价是第二套全模型且其误差直接污染优势。

---

### 14. What causes reward hacking, how do you detect it, and which metrics do you watch during RL training when loss moves abnormally?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/reinforcement-learning-from-human-feedback-rlhf

**Anchor answer**

Reward hacking = the policy finds inputs that score high on the proxy without doing the intended task. Root cause is always a **gap between proxy and intent**, amplified by optimisation pressure.

Common mechanisms: length bias (longer = higher score), sycophancy (agreeing scores well), format exploits (bullet lists, "as an AI…"), verifier loopholes (regex passes a wrong answer), and distribution escape (nonsense the RM has never seen and scores high out of uncertainty).

Detection:

| Signal | Interpretation |
| :--- | :--- |
| reward ↑ but held-out task metric flat or ↓ | classic divergence — the single most reliable detector |
| output length exploding | length-bias hack |
| entropy collapse / repetition | policy degenerate, KL too loose |
| grader pass rate ↑ with human spot-check ↓ | verifier loophole |
| KL to reference rising fast | drift |

When loss moves abnormally, watch: **policy loss, KL, clip fraction, advantage std, response length, entropy, and the ratio of samples at the max length**. Policy loss going *up* is normal in PPO (you are not minimising a fixed loss), so never read it alone — read it next to KL and clip fraction. Abrupt loss spikes are usually a bad batch with identical rewards (advantage ≈ 0 → division noise when std is tiny), which is why you add an epsilon to the group std.

**中文要点**：奖励被刷的根因永远是代理指标与真实意图的缝隙；最可靠检测是"reward 涨而 holdout 任务指标不涨"；看 loss 要同时看 KL、clip fraction、长度、熵、优势标准差，全同分组会导致优势数值噪声。

---

### 15. Can RLVR be considered a form of SFT? Why or why not?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/large-reasoning-models

**Anchor answer**

No — but the confusion is understandable because the *effective* update often looks like SFT.

The formal difference: SFT minimises a fixed cross-entropy over given tokens, one gradient direction per sample, no notion of better/worse. RLVR maximises expected reward over **self-generated** samples, so the gradient sign depends on the reward relative to a baseline — it can *suppress* tokens it produced, which cross-entropy on that same sample never would.

Yet the two coincide in an important limit:

| Condition | Effect |
| :--- | :--- |
| binary reward, group of `G`, all correct or all wrong | advantage ≈ 0 (or the classic "+1/−1" case collapses to REINFORCE) |
| reward strictly `1` for correct and `0` otherwise, single sample per prompt | gradient direction ≈ `∇log π(y)` for correct `y` — i.e. **on-policy rejection-sampling SFT** |
| KL term dominant | update shrinks to a trust-region SFT |

This is why rejection-sampling fine-tuning (generate `k`, keep correct, SFT on them) is a decent cheap substitute — it is the zero-order approximation of the same signal. But they differ in the failure modes: SFT on filtered correct samples cannot express "this reasoning style is bad"; RLVR can assign negative advantage and unlearn it. And RLVR can discover a solution the model never sampled under SFT's fixed data.

**中文要点**：形式上不是 SFT——RLVR 用自采样、带基线、可给负优势压制错误路径；但当奖励二值、每 prompt 单样本时梯度方向近似于"正确样本的 on-policy SFT"，这正是拒绝采样微调能替代它的原因。

---

### 16. Agentic RL: when do you move from SFT to RL, and when should a multi-turn reward come from a model versus a rule?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/large-reasoning-models

**Anchor answer**

Move to RL when **SFT has saturated on data you can still author, but the policy can already stumble onto correct behaviour**. Concretely: the model passes ~30–60% of tasks under sampling, and you have a verifier. Below that, RL gets no signal (all-zero advantage); above ~90%, RL has nothing to teach and only risks drift. If you cannot verify, stay on SFT and preference data.

| Signal type | Use when | Failure mode |
| :--- | :--- | :--- |
| **Rule / executable** (tests pass, state matches, schema valid) | outcome is checkable in a sandbox | sparse; no credit for near-misses; gameable by shortcut |
| **Model judge** | open-ended quality, tone, multi-step helpfulness | hackable, position/verbosity bias, expensive per rollout |
| **Hybrid** | rule on hard constraints + judge on style | needs calibration; two scales to weight |

For multi-turn specifically: intermediate turns rarely have a checkable outcome, so the practical design is **rule-based final-state reward + model-based process reward at low weight**, plus turn-level discounts so long trajectories are not implicitly favoured. Rules give a stable floor; the judge gives gradient on the middle of the trajectory.

The failure to name: a judge trained on your SFT distribution rewards *looking like* your SFT data, so RL sharpens style while capability plateaus — watch task success separately from judge score.

**中文要点**：SFT 饱和且采样通过率约 30~60%、又有可验证器时才上 RL；多轮里最终状态用规则奖励，过程用低权重模型奖励，避免长轨迹被隐性偏爱；判断器只奖励"像 SFT 数据"是典型陷阱。

---

### 17. Why run multi-way retrieval (dense + keyword + hierarchical)? What is the bad case for each route?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/how-does-hybrid-search-work

**Anchor answer**

Each route fails on a different query shape, so union-then-rerank is the only robust design.

| Route | Strong at | Systematic bad case |
| :--- | :--- | :--- |
| Dense (bi-encoder) | paraphrase, semantics, cross-lingual | exact IDs, error codes, rare proper nouns, numbers — the embedding washes out the literal string |
| Sparse/BM25 | exact terms, codes, names, out-of-domain vocabulary | synonyms and query-document vocabulary mismatch; long queries dilute term weights |
| Hierarchical (doc→section→chunk) | long structured docs, "which chapter" questions | extra hop adds latency, and a bad parent summary orphans good children |

Mechanism behind the dense failure: a bi-encoder compresses the whole passage into one vector, so a single rare token contributes little to cosine similarity. That is why "error 0x80070005" and "permission denied" can outrank the exact match — the model has learned that the *meaning* is the same and thrown away the identifier.

The practical pattern: retrieve top-50 from each route, fuse with RRF (rank-based, no score calibration needed — score fusion across incomparable scales is a common bug), then rerank the fused ~50 to 6 with a cross-encoder. Weights between routes are best avoided entirely; RRF is nearly free and robust. Only tune weights when you have per-route labelled recall.

**中文要点**：稠密在精确 ID/错误码上会"语义化"丢字面，稀疏在同义词与长 query 上失效，层级多一跳但易断链；稳妥做法是各取 top-50、RRF 融合、交叉编码器重排到 6 条，别硬调分数权重。

---

### 18. You retrieved 30 passages but only 6 fit the context budget. How do you choose?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/how-does-a-reranker-work

**Anchor answer**

Treat it as **budgeted subset selection under redundancy**, not top-k truncation.

Pipeline:

1. **Rerank all 30 with a cross-encoder** — bi-encoder scores are not comparable enough to pick 6. This alone usually recovers most of the gap.
2. **Score-margin cutoff.** Drop anything below a calibrated threshold, or below `best − Δ`. If you keep 6 low-score passages "to fill the budget", you inject noise that the model will cite.
3. **MMR / dedup within budget.** Passages that overlap heavily add tokens and no information. Greedy MMR (`λ·rel − (1−λ)·max_sim`) with `λ≈0.7` is cheap and effective.
4. **Per-document cap.** At most 2–3 chunks per source document, otherwise one verbose document monopolises the window.
5. **Order matters.** Put the strongest evidence at the **start and end**; models attend poorly to the middle of long contexts. This is free accuracy.

The counter-intuitive part: returning **4 strong passages beats 6 mediocre ones** in most evals — the budget is not a quota you must fill. And always keep the passage IDs you used in the trace, so "why did it answer that" is answerable without re-running retrieval.

When not to bother: if the 30 passages came from a single small document, dedup plus first-3 is fine; the machinery pays off only for wide candidate pools.

**中文要点**：30→6 不是截断而是"预算下带冗余惩罚的子集选择"：交叉重排 → 分数边际截断 → MMR 去重 → 每文档限额 → 强证据放首尾；宁少勿滥，4 条强于 6 条平庸。

---

### 19. Permission-aware knowledge base: filter by ACL before retrieval, or after? How do you monitor and prove no unauthorised content was retrieved?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/how-does-hybrid-search-work

**Anchor answer**

**Before**, always — as a hard filter inside the vector query, not a post-filter on results. Post-filtering leaks in three ways: the top-k gets consumed by documents the user cannot see (so recall silently dies), scores/thresholds reveal existence of restricted material, and any bug in the filter step is a data breach rather than a ranking error.

Design:

| Layer | Mechanism |
| :--- | :--- |
| Index | ACL as **metadata on every chunk**, partitioned or namespaced by tenant; never mix tenants in one unfiltered HNSW graph |
| Query | pre-filter injected into the ANN search (`tenant_id = ? AND acl_tag ∈ …`), with a filtering-capable index (payload index, or per-tenant sub-index) |
| Rerank / context | re-check ACL at assembly time — defence in depth against stale metadata |
| Generation | the prompt never contains a passage the user cannot see; no "the document says X but you can't see it" |

Proving it: emit a per-request audit record `{user, query, candidate_ids, returned_ids, acl_version}` and run a **negative test suite** — synthetic users with no rights querying known-restricted content; success = zero restricted IDs in candidates *and* zero in returned. Plus a continuous sampler that recomputes ACL for returned docs.

The trap: permission changes propagate slowly. Version the ACL and rebuild affected chunks; a revoked user must not get hits from a stale index shard.

**中文要点**：ACL 必须在**检索前**作为元数据硬过滤进 ANN 查询，后置过滤会吃掉召回并可能泄露内容存在性；用逐请求审计日志 + 无权限用户的负向测试证明"候选与返回皆零"，并给 ACL 打版本防陈旧。

---

### 20. When should the system trigger a web search and when should it answer directly? How do you tell "the model did not know it needed retrieval" from "it knew but did not execute"?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/ai-agent-loop

**Anchor answer**

Trigger search when the query is about **changeable or post-cutoff facts, current entities, prices, versions, or anything the user implies must be verified**. Answer directly for stable knowledge, definitions, transformations of user-provided material, and any query where the local corpus is authoritative.

Practical trigger design is a three-way decision, not a binary:

| Route | Condition | Action |
| :--- | :--- | :--- |
| Answer | stable + high self-consistency, no time cue | generate |
| Search | time cue, entity not in corpus, low confidence | one search + cite |
| Clarify | ambiguous referent | ask, don't guess |

Diagnosing "didn't know" vs "knew but didn't act" is a **counterfactual test**: replay the same prompt with retrieval forced on. If the answer becomes correct, the policy had the capability and mis-decided the route — a *routing* bug, fixable with a classifier or trained tool-use examples. If forcing retrieval does not help, the query was not retrievable or the model cannot use the evidence — an *ability* bug, fixable with data, not prompts.

Second signal: probe the model's own uncertainty (logprob on the answer, or "do you need a tool?" self-report) and compare with outcomes. Systematic overconfidence on the missed cases means the trigger needs an external rule, not better self-assessment.

**中文要点**：可检索事实/时间敏感 → 搜；稳定知识/用户给定材料 → 直答；用**强制检索重放**区分"不会路由"与"没有能力"——前者改触发器/训练样例，后者只能改数据。

---

### 21. How do you chunk and retrieve when a long document spans chapters and a retrieved fragment points at another document?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/chunking-strategies-for-rag

**Anchor answer**

Use **two-level chunks plus explicit cross-references**; flat fixed-size chunking cannot represent "see §4.2 of the other doc".

Structure:

| Level | Unit | Purpose |
| :--- | :--- | :--- |
| Doc summary | per document | routing: decide *which* document to enter |
| Section/parent chunk (child parent) | per chapter/section | context given to the model |
| Child chunk | 200–400 tokens | what you actually embed and match |

Retrieve small for precision, **expand to the parent section** for generation. The parent gives the chapter heading, numbering and surrounding definitions that the child alone loses.

For the cross-document pointer: during ingestion, run a lightweight reference extractor (regex plus model) that turns "as described in Doc B §4.2" into a **typed edge** `(doc_A, §x) → (doc_B, §4.2)`. Store it in the metadata, not in the prose. At query time, after retrieving a fragment with an outgoing edge, do a bounded **1-hop expansion** with a step budget (1–2 hops max, and never follow edges across permission boundaries).

Failure modes worth naming: hop explosion on documents with hundreds of "see above" links (cap total tokens, not just hops), stale targets after a document is re-versioned (resolve edges by stable IDs, not titles), and citations that point to a document the user cannot access — the edge must be dropped, not silently followed.

**中文要点**：两级切分（文档摘要/章节父块/子块），小块用于匹配、父块用于生成；把 "见文档 B §4.2" 建成**有类型的边**做有界 1~2 跳扩展，并按稳定 ID 解析、遵守权限边界。

---

### 22. Where are the boundaries between Agent, Workflow, Orchestrator and Harness?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/ai-orchestration

**Anchor answer**

They sit at four different layers and are routinely conflated.

| Term | What it is | Who decides the next step | Best for |
| :--- | :--- | :--- | :--- |
| Workflow | fixed DAG of steps, code-driven | the developer | predictable, auditable, ~80% of production tasks |
| Agent | model chooses tools/steps in a loop until a stop condition | the model | open-ended, path unknown in advance |
| Orchestrator | the runtime that schedules agents/workflows, routes models, handles retries and hand-offs | the developer + policy | multi-step, multi-model products |
| Harness | the scaffolding around a single model call: prompt assembly, tool schemas, compaction, permissions, evals | the developer | quality and safety of everything above |

The clean mental model: **workflow = control flow written by you; agent = control flow written by the model; orchestrator = what runs the control flow; harness = what the control flow is allowed to see and do.**

Boundary rule that matters in practice: any step whose failure is unrecoverable and whose inputs are known should be a workflow node, not an agent decision. Teams that make everything an agent end up with 3× the tokens and an unbounded p99. The reverse mistake — wrapping an open-ended research task in a rigid DAG — produces an agent that "works" only on the demos.

**中文要点**：workflow 是**你写的**控制流（可审计、覆盖约 80% 生产任务），agent 是**模型写的**控制流；orchestrator 是跑控制流的运行时，harness 是单次调用的脚手架（提示拼装、工具 schema、压缩、权限、评测）。

---

### 23. How do Function Calling, Tool, MCP and Skill differ? Why does a Skill save tokens, and what risk does that bring?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/what-is-mcp-model-context-protocol

**Anchor answer**

They are four layers of the same stack:

| Layer | Definition | Analogy |
| :--- | :--- | :--- |
| Tool | one callable capability with a JSON schema (`search`, `read_file`) | a function |
| Function calling | the **protocol** by which the model emits `{name, arguments}` and the client returns a result | the calling convention |
| MCP | a **transport/standard** so tools and resources are discovered and served by an external process | a driver interface / port |
| Skill | a **packaged procedure** (instructions + scripts + reference files) the model loads on demand | a manual + toolbelt |

Why a Skill saves tokens: a tool's schema plus usage guidance is in context **every turn**; a skill is a short description in context, and its full text/scripts are read only when invoked. Ten skills as tools might be 4–6k tokens of schemas per request; ten skills as skill entries might be a few hundred until one is used.

The risk that comes with it: **deferred loading means deferred validation** — the model can decide to invoke a skill whose body then turns out to be wrong for the task, and it discovers that only after spending the tokens. Two related dangers: prompt injection hidden inside a skill file (it is trusted content on disk, so a poisoned skill is executed silently), and unversioned skills diverging from the tools they call.

Mitigation: version skills with the tool interfaces they depend on, review skill text like code, and keep a hard budget on how much skill body can be loaded per turn.

**中文要点**：tool 是能力、FC 是调用协议、MCP 是外部工具的传输标准、Skill 是按需加载的"流程包"；Skill 省 token 因为描述常在而正文按需读，风险是惰性校验、skill 文件里的注入和被静默执行。

---

### 24. How do you compress context as it approaches the window limit, and what must never be left to a summarising model?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/how-does-context-compaction-work

**Anchor answer**

Compaction is a **tiered rewrite**, not "summarise everything older than N".

| Tier | Content | Action |
| :--- | :--- | :--- |
| Never touch | system prompt, safety rules, tool schemas, current task statement | verbatim |
| Pin | recent `k` turns, the user's latest instruction, open action items | verbatim |
| Verbatim-ify | tool outputs already used | replace with **one-line result + id reference**, keep the full text on disk/artifact |
| Summarise | older assistant reasoning and chat | structured summary: decisions, constraints, rejected options |
| Drop | superseded drafts, duplicate retrievals | remove, keep a pointer |

What must never be delegated to a summariser: (1) exact identifiers — file paths, IDs, error strings, numbers, URLs; (2) permissions and policy constraints; (3) the verbatim user instruction; (4) anything a summariser could "improve" into a false claim (e.g. "tests pass" when they did not). Summaries lose exactly the low-frequency tokens that are load-bearing.

Mechanically: compute the budget in tokens (not chars), trigger at ~80% so you never hit a hard failure mid-generation, and keep the summary **append-only with a version**, so you can prove what the model actually saw. The failure to name: recursive summarisation of summaries, which loses facts at each pass — always summarise from the original transcript.

**中文要点**：分层压缩——系统提示/工具 schema/用户原话逐字保留，旧工具输出换成"一行结果+引用"，旧对话压成含决策与约束的结构化摘要；ID、路径、数字、权限、原始指令**绝不交给模型改写**，且必须从原文而非摘要再摘要。

---

### 25. A production agent keeps calling tools but never returns. How do you debug it? How do you defend against reasoning-chain error propagation, tool hallucination, broken JSON and infinite loops?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/ai-agent-loop

**Anchor answer**

Debug from the **trace**, not the prompt. Pull the last N steps and classify each as `progress` / `repeat` / `error`. The single most useful check: hash `(tool_name, canonicalised_args)` — identical consecutive hashes mean the agent is stuck, and that is ~70% of "never returns" cases.

| Symptom | Root cause | Fix |
| :--- | :--- | :--- |
| same call, same args | no state change; model can't see prior failure | inject an explicit "this exact call failed, do not repeat" note; add loop detector |
| ping-pong between two tools | conflicting instructions, no decision criterion | add a stopping rule in the system prompt, or a hard step budget |
| escalating retries | transient error treated as terminal | classify errors; backoff; cap |
| endless sub-planning | task decomposed but never executed | require a tool call every N reasoning steps |

Defences, all cheap:

1. **Hard budgets** — max steps, max tokens, wall clock; returning a partial result with "budget exhausted" beats hanging.
2. **Schema-validate every tool call** before execution; on invalid JSON, re-ask **once** with the parse error, then fall back to a repair parser — never loop on the model.
3. **Tools return errors as observations**, with `retryable: false` for terminal cases.
4. **No-progress counter** → forced summarise-and-stop.

Reasoning-chain propagation is the subtler failure: one hallucinated fact gets cited confidently for 5 steps. Mitigation is grounding — every claim in the final answer must map to a tool observation ID.

**中文要点**：从 trace 而非 prompt 调试，`(工具名, 参数)` 哈希连续相同即卡死（约七成"不返回"）；手段是硬预算 + 每步 schema 校验 + 错误作为观察回灌 + 无进展计数；幻觉链靠"结论必须挂到某条工具观察 ID"。

---

### 26. How do multiple agents communicate, and how do you stop two of them editing the same file?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/multi-agent-systems

**Anchor answer**

Communication is **shared state plus short messages**, never free-form chat between agents (message ping-pong is the main cost sink).

| Channel | Use | Cost |
| :--- | :--- | :--- |
| Shared artifact store (`local://`, files, DB) | large payloads, outputs | cheap, needs ownership |
| Typed inbox / mailbox (address a peer id) | coordination, questions | one file write |
| Broadcast | global constraints, stop signals | can flood; use sparingly |
| Blackboard with version numbers | multi-writer state | needs CAS |

Preventing concurrent edits: **partition by ownership, not by luck**. Assign each file an owner at plan time; a non-owner must send a request message, not edit. Enforce mechanically with per-file locks (or CAS on a version field) and make the lock scope the *file*, not the task — task-level locks are too coarse and serialise everything.

Three failure modes to name:

1. **Lost update** — two agents read, both write, last wins. Version check on write catches it.
2. **Half-finished sibling state** — agent A reads B's partially written file. Write-then-rename (atomic replace) prevents torn reads.
3. **Integration drift** — both edit different files that must agree on an interface. Fix: publish the interface as an immutable artifact *before* parallel work starts, and have one integration owner.

Never let siblings run build/test concurrently — they stomp each other's artifacts.

**中文要点**：靠"共享产物 + 短消息"而非自由对话；文件按**所有者划分**，非所有者只能发请求，用文件级锁或版本 CAS 防丢失更新，写入用"临时文件+原子改名"；接口必须先定稿为不可变产物。

---

### 27. How do you define and measure a hallucination rate? How was that improvement number actually computed, and how do you tell stale content from hallucination?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/llm-evaluation

**Anchor answer**

Define it as a **ratio over claims, not answers**, with an explicit denominator.

Practical definition: `hallucination_rate = unsupported_claims / total_checkable_claims`. For each answer, an annotator or judge labels each atomic claim as `supported` / `contradicted` / `unverifiable`. Report all three — collapsing "unverifiable" into "false" inflates the number and makes it incomparable across releases.

How the improvement number is really computed, and the four ways it lies:

| Leak | Effect |
| :--- | :--- |
| answer-level instead of claim-level | one bad claim in a 10-claim answer counts as 1/1, so easy answers dominate |
| judge comparing against the *retrieved* context, not ground truth | measures grounding, not truth — RAG looks great |
| dataset changed between runs | "improvement" may be a sampling artefact |
| judge is the same model family as the system | shared blind spots |

Stale vs hallucinated: a claim that **was** true in an older version of the source is staleness — detect by checking the source's version/timestamp and whether the claim appears in a superseded revision. A claim absent from every revision is a hallucination. Conflating them is a serious reporting bug: staleness is a retrieval/freshness problem (fix the index), hallucination is a generation problem (fix grounding).

Minimum credible protocol: fixed claim-level test set, two annotators on a 200-item subsample for agreement, judge verified against human labels, and the dataset hash published with the number.

**中文要点**：按**声明（claim）级**算 `无支撑声明数/可核查声明数`，并单列"不可验证"，否则数字不可跨版本比较；常见四漏：答案级计数、对着检索上下文判、换了数据集、判官同源；旧版本为真属**陈旧**（修索引）而非幻觉（修落地）。

---

### 28. Design an AI code review agent: input is a PR, it only comments, it never edits. What are the three things that matter most?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/how-does-claude-code-work

**Anchor answer**

The three things: **precision over recall, deterministic context, and a hard read-only boundary.**

Constraints: no writes, no branch pushes, bounded timeout, cost per PR capped. The product risk is not missing a bug — it is a noisy bot that gets muted, after which it catches nothing.

Architecture:

| Stage | Design |
| :--- | :--- |
| Trigger | webhook on PR open/sync; diff-scoped, not repo-scoped |
| Context build | full file for changed hunks + reverse deps (grep callers) + repo conventions file; **cap** total tokens |
| Analysis | hunk-by-hunk with a checklist; run linters/tests in a sandbox and feed results in as facts |
| Filter | self-critique pass: "is this a real defect, or style?" drop anything unprovable |
| Post | one review with line-anchored inline comments, never edits, never approves |

Precision mechanics that actually matter: only comment when the agent can name the **concrete failure** (input, wrong output, severity); dedupe against existing review comments and against its own prior run; cap comments per PR (e.g. 8) and rank by severity so it cannot flood.

Failure modes to name: hallucinated line numbers (anchor to the diff hunk text, not to a number the model produced), reviewing unchanged code because retrieval pulled it in, and re-commenting the same issue every push — you need stable finding IDs keyed by `(file, hunk-hash, rule)`.

Read-only enforcement is not a prompt: it is a token without write scope. Any design that relies on "please don't edit" will eventually edit.

**中文要点**：三件事是**精确率优先、确定性上下文、硬只读边界**；按 hunk 分析并把 lint/test 结果当事实喂入，自评过滤后限条数评论；只读靠无写权限的 token 而非提示词，评论锚定 diff 文本与稳定 finding ID。

---

### 29. Design an enterprise knowledge base / local wiki agent: ingestion, permission isolation and evaluation in one story

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/chunking-strategies-for-rag

**Anchor answer**

One story means the same identity travels from ingestion to answer.

| Layer | Design |
| :--- | :--- |
| Connectors | Confluence/Drive/Git/Slack; incremental sync by `updated_at` + content hash; deletions propagate as tombstones |
| Ingestion | parse → section-aware chunk (parent + child) → enrich (title path, ACL snapshot, version, source URL) |
| Index | per-tenant partition; ACL as filterable metadata; BM25 + dense + (optional) graph of doc links |
| Retrieval | pre-filter by ACL, hybrid top-50, RRF, cross-encoder to 6, parent expansion |
| Answer | cite every claim with source URL + last-modified; refuse when evidence is thin |
| Evaluation | golden set per domain + permission test suite + freshness checks |

Permission isolation, the part that fails in production: ACL is snapshotted at ingest and **must be re-resolved at query time** for the identity — otherwise a revoked employee keeps getting hits from an old chunk. Store `acl_version`; on ACL change, invalidate rather than patch. Sharing must be explicit: if two users see different renditions, dedupe keyed by `(tenant, doc, acl_version)`, never by content hash alone.

Evaluation, three suites, all run per release:

1. **Answer quality** — 200–500 question/answer/evidence triples, scored on claim support and citation correctness.
2. **Permissions** — users with no rights must get zero restricted chunks in candidates *and* in context (log both).
3. **Freshness** — after editing a doc, the new fact must appear within the sync SLA; stale answers are graded as failures, not hallucinations.

**中文要点**：同一条身份贯穿入库到回答；ACL 在入库时快照但**查询时必须按当前身份重新解析**（打版本，变更即失效）；评测三套：声明-引用质量、无权限用户零泄露（候选与上下文都查）、改文档后按 SLA 生效否则记失败。

---

### 30. Design an ops and incident-triage agent: how does it run the full diagnostic flow when a payment endpoint times out, and how are high-risk actions controlled?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/ai-orchestration

**Anchor answer**

Structure it as **read-only investigation with a fixed evidence ladder**, and put all mutations behind a human gate.

Flow for "payment endpoint p99 timeout":

1. **Scope** — which region, which error rate, since when; compare to baseline.
2. **Change correlation** — deploys, config pushes, feature flags, upstream dependency versions in the window.
3. **Dependency check** — downstream call latencies, error codes, connection pool saturation, DNS.
4. **Resource check** — CPU/GC, thread pools, DB slow queries, queue lag, connection counts.
5. **Hypothesis ranking** — an explicit table `hypothesis | evidence for | evidence against | next probe`.
6. **Report** — timeline, top hypothesis, the discriminating probe that confirmed it, and a proposed remediation.

Every step is a **read-only tool** with a bounded timeout; the agent must cite the metric it read, not summarise impressions.

High-risk action control uses a spectrum, not a boolean:

| Tier | Example | Control |
| :--- | :--- | :--- |
| Read | query metrics/logs | auto |
| Low-risk mutate | raise a flag in a canary | auto with audit + auto-rollback |
| High-risk | restart prod, failover, rollback deploy | **prepare-only**: agent produces a command + plan, human executes/approves |
| Irreversible | data fix, refund | forbidden; escalate to runbook |

Non-negotiables: dry-run first, per-action blast-radius estimate shown to the approver, idempotency keys on every mutate, and a kill switch. The agent's value is the timeline and the discriminating probe, not autonomy.

**中文要点**：只读排查走固定证据阶梯（范围→变更关联→依赖→资源→假设排序→时间线报告），每步引用指标；写操作分级：只读自动、canary 变更带审计与自动回滚、生产重启/回滚仅"出方案给人执行"、不可逆操作禁止。

---

### 31. Design a content-tagging agent (video or image plus text): how do you split the pipeline and what do you take from the source material?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/multimodal-ai

**Anchor answer**

Split by **signal type, then fuse at the decision layer** — a single end-to-end "tag this" call gives you unexplainable tags and no way to fix one modality.

| Stage | Input | Output |
| :--- | :--- | :--- |
| 1. Frame/shot sampling | video | keyframes at shot boundaries (not fixed 1 fps) |
| 2. Visual tagging | keyframes | objects, scene, actions, on-screen text (OCR) |
| 3. Audio tagging | track | ASR transcript, music/speech, speaker count |
| 4. Text/metadata | title, description, comments | topical tags, named entities |
| 5. Fusion + taxonomy mapping | all of the above | **closed-set** tags with confidence |
| 6. Review gate | low-confidence / high-impact | human queue |

What to take from each source, and what not to:

- **Video**: shot cut + a few frames per shot. Do not sample uniformly — a 10-minute talking-head gives 600 near-identical frames and blows the token budget.
- **Audio/ASR**: this is usually where the real topic lives; visual tags are often genre, ASR is subject matter.
- **Text metadata**: cheapest and most reliable for entity names, but biased by uploader intent — never let it override contradictory visual evidence.

The taxonomy is the design: force a **closed vocabulary with hierarchy** and an `unknown` escape, otherwise tags are unaggregatable and unusable for retrieval. Emit `tag | confidence | evidence_ref (frame ts / transcript span)` so a reviewer can audit in one click, and log per-tag precision so you can prune tags whose precision never exceeds the base rate.

**中文要点**：按模态分段（镜头级关键帧、ASR、元数据）再在决策层融合成**封闭词表**标签；别均匀采样帧（长镜头会被几百张近似帧撑爆预算）；ASR 常承载真实主题；每个标签必须带置信度与证据引用以便审阅和剪枝。

---

### 32. Implement multi-head attention in PyTorch, then extend it to cross-attention

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/multi-head-attention-in-transformers

**Anchor answer**

**Reference implementation**

```python
import math, torch, torch.nn as nn, torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def _split(self, x, B, T):
        return x.view(B, T, self.h, self.dk).transpose(1, 2)  # B,h,T,dk

    def forward(self, x, mem=None, mask=None):
        B, T, _ = x.shape
        q = self._split(self.wq(x), B, T)
        src = x if mem is None else mem          # cross-attention: mem = encoder out
        S = src.shape[1]
        k = self._split(self.wk(src), B, S)
        v = self._split(self.wv(src), B, S)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)   # B,h,T,S
        if mask is not None:
            att = att.masked_fill(mask == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).reshape(B, T, self.h * self.dk)
        return self.wo(out)
```

**Key points**

- The only difference for cross-attention is `mem`: Q from the decoder, K/V from the encoder output. Mask shape changes from `(T,T)` causal to `(T,S)` padding-only.
- Use `F.scaled_dot_product_attention` in production — it fuses the softmax and dispatches to Flash/Mem-efficient kernels. The manual version above materialises a `B×h×T×S` matrix.
- Causal mask build: `torch.tril(torch.ones(T,T))`; for KV-cache decode, mask only the new row.
- `bias=False` on projections matches most modern LLMs; `d_model % n_heads` must divide evenly.
- Watch dtype: `-inf` before softmax in fp16 produces NaN if a whole row is masked — use `-1e4` or handle empty rows.

**中文要点**：交叉注意力只改 `mem`（Q 来自解码器，K/V 来自编码器输出，掩码由因果变 padding）；生产用 `scaled_dot_product_attention`；注意 fp16 下整行 `-inf` 会出 NaN。

---

### 33. Write a minimal ReAct agent: tool registration, loop control, termination condition and error handling

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/react-agent

**Anchor answer**

**Reference implementation**

```python
import json, hashlib

class Agent:
    def __init__(self, llm, tools, max_steps=8, max_repeat=2):
        self.llm, self.tools = llm, tools          # tools: {name: fn(**) -> str}
        self.max_steps, self.max_repeat = max_steps, max_repeat

    def run(self, question):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        seen, last, repeats = set(), None, 0
        for step in range(self.max_steps):
            out = self.llm(msgs, tools=schemas(self.tools))
            if out.tool_calls:
                for call in out.tool_calls:
                    key = call.name + json.dumps(call.args, sort_keys=True)
                    if key == last:
                        repeats += 1
                        if repeats >= self.max_repeat:
                            msgs.append(note(f"repeating {key}; try a different approach"))
                            continue
                    last, _ = key, seen.add(key)
                    try:
                        obs = self.tools[call.name](**call.args)
                    except KeyError:
                        obs = f"ERROR: unknown tool {call.name}"          # terminal
                    except Exception as e:
                        obs = f"ERROR: {type(e).__name__}: {e}"          # fed back
                    msgs.append({"role": "tool", "name": call.name, "content": str(obs)[:4000]})
                continue                                    # loop again
            return out.content.strip()                      # no tool call => final
        return self.llm(msgs + [{"role": "user", "content": "Stop. Summarise what you know and answer, or say what is missing."}])
```

**Key points**

- **Termination is threefold**: the model stops calling tools; the step budget is hit; the repeat detector fires. Never rely on the model alone.
- Errors become **observations**, not exceptions — a retried-forever terminal error is the most common production hang.
- Truncate observations (`[:4000]`); a single large tool output will otherwise evict the task.
- The forced-summary path on budget exhaustion is what separates a usable agent from a timeout.
- Log `(step, tool, args, obs_len, latency)` — that trace is the only way to debug a stuck loop.

**中文要点**：终止有三重（模型不再调工具、步数预算、重复检测）；错误作为观察回灌而非抛出；工具输出必须截断；预算耗尽时强制总结，否则就是超时。

---

### 34. Write context compaction or sliding-window summarisation: given a message list and a token budget, decide what to keep, summarise or offload

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/how-does-context-compaction-work

**Anchor answer**

**Reference implementation**

```python
def compact(msgs, budget, count_tokens, summarise, keep_recent=4, trigger=0.8):
    """msgs: [{'role','content','pinned'?,'artifact'?}]"""
    if count_tokens(msgs) <= budget * trigger:
        return msgs                                   # do nothing until 80%

    pinned = [m for m in msgs if m.get("pinned")]
    rest   = [m for m in msgs if not m.get("pinned")]
    recent = rest[-keep_recent:]
    older  = rest[:-keep_recent] if len(rest) > keep_recent else []

    kept = []
    for m in older:
        if m.get("artifact") and not m.get("critical"):
            kept.append({**m, "content": f"[offloaded {m['artifact']}]: {one_line(m['content'])}"})
        elif m["role"] == "tool":
            kept.append({**m, "content": f"[tool output stored: {m.get('ref','')}]"})
        else:
            kept.append(m)

    overhead = count_tokens(pinned + recent) + 64
    room = budget - overhead
    summary = summarise(kept, max_tokens=max(128, room)) if kept else None
    out = pinned + ([{"role": "system", "content": "Earlier context summary:\n" + summary}] if summary else []) + recent
    while count_tokens(out) > budget and len(out) > 1:      # hard guarantee
        out.pop(1 if out[1]["role"] == "system" else 0)
    return out
```

**Key points**

- **Trigger at 80%**, not 100% — compaction must happen before a hard failure, and it costs a model call.
- Three classes, not two: *verbatim* (pinned, recent), *compressed* (offloaded with pointer), *summarised* (older turns).
- Never let the summariser rewrite identifiers, paths, numbers, or the user's instruction — mark those `critical` and keep them verbatim.
- Always keep a hard `while` loop that enforces the budget; a summariser that overruns its own `max_tokens` is a real failure.
- Summarise from originals, not from previous summaries — recursive summarisation silently loses facts.

**中文要点**：在 80% 触发而非撑满；分三类（逐字保留/外置指针/摘要）；ID、路径、数字、用户指令标 `critical` 不许改写；最后必须有强制裁剪循环，且永远从原文再摘要。

---

### 35. Standard coding rounds also include merge intervals, top-k and longest substring without repeats

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/llm-evaluation

**Anchor answer**

**Reference implementation**

```python
def merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)     # note the max
        else:
            out.append([s, e])
    return out

import heapq
def topk(nums, k):
    h = []
    for x in nums:                               # O(n log k), O(k) memory
        if len(h) < k: heapq.heappush(h, x)
        elif x > h[0]: heapq.heapreplace(h, x)
    return sorted(h, reverse=True)

def longest_unique(s):
    last, best, start = {}, 0, 0
    for i, c in enumerate(s):
        if c in last and last[c] >= start:
            start = last[c] + 1                  # jump, don't shrink one by one
        last[c] = i
        best = max(best, i - start + 1)
    return best
```

**Key points**

- **Merge intervals**: sort by start, then the only nontrivial detail is `max(end)` for nested intervals (`[[1,10],[2,3]]`). Touching intervals (`[1,2],[2,3]`) — confirm whether the boundary counts as overlap; it changes `<` to `<=`.
- **Top-k**: a size-`k` **min-heap** beats full sort when `k ≪ n`; `heapreplace` keeps it O(n log k). For `k` close to `n`, `nlargest` (which switches to sort) or quickselect is better.
- **Longest substring**: the classic bug is not resetting `start` when the repeated char is *before* the window (`last[c] >= start` guard). That guard is the whole problem.
- Complexity to state out loud: `O(n log n)` / `O(n log k)` / `O(n)`; space `O(n)` / `O(k)` / `O(σ)`.

**中文要点**：区间合并的关键是嵌套时取 `max(end)`，并确认端点相接算不算重叠；top-k 用小顶堆 O(n log k)，k 接近 n 时反而排序更快；最长无重复子串的坑是窗口左界不能回退（`last[c] >= start`）。

---

### 36. What paper have you read recently, and what is the core idea? How do you keep up with new LLM work?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/large-reasoning-models

**Anchor answer**

Answer structure that reads as credible: **one paper, one mechanism, one thing you disagreed with.**

Pick a paper you can defend at the level of "what did they ablate, and what would break if you removed it". For example, on speculative decoding: the interesting part is not the speedup, it is that rejection sampling keeps the output distribution exactly equal to the target — which means you can adopt it with zero quality risk, and the only question left is the acceptance-rate/cost ratio. A candidate who can say "we tried it and acceptance was 55% on our domain, so the ratio was ~1.3 and not worth the KV memory" has actually read it.

Keeping up, ranked by signal per minute:

| Channel | Why |
| :--- | :--- |
| Paper + its ablation table | the only place the real limits are stated |
| Reference implementations (vLLM, TRL, HF) | code shows what the paper omitted |
| Reproductions / issue trackers | where the claims actually break |
| Curated digests | triage only, never the only source |

The trap to avoid: listing five papers you cannot discuss. One paper you can derive beats five you can summarise. And say what you *changed your mind* about — that is the signal interviewers are actually listening for.

**中文要点**：选一篇能讲到"消融实验与去掉某组件会坏在哪"的论文；跟进的性价比排序是论文原文 > 参考实现 > 复现 issue > 摘要简报；宁可深讲一篇，也不要报五篇书名。

---

### 37. What do you think about the alignment tax of LLM agents, and are current agents meeting your expectations?

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/llm-evaluation

**Anchor answer**

The alignment tax is real but it is not a flat percentage — it is **concentrated in the tail of hard agentic tasks**, and it shows up as *refusal-shaped hesitation* rather than "worse answers".

Where the tax actually lands:

| Mechanism | Cost |
| :--- | :--- |
| Safety fine-tuning / RLHF toward cautious behaviour | more "I can't" on legitimate but unusual requests, more clarifying questions |
| Permission gating on tool use | extra round-trips, human approval latency |
| Refusal on ambiguous intent | precision up, task completion down |
| Over-refusal cascades in agents | one blocked step aborts a whole trajectory, so cost is superlinear |

The agent-specific twist: in a single turn, a 2% over-refusal rate is invisible. In a 20-step trajectory, `0.98^20 ≈ 0.67` — a third of tasks fail. That is why agent products must measure **task success**, not per-step refusal, and why "just add a safety prompt" is so expensive.

Are current agents meeting expectations? Partially, and the gap is not intelligence — it's **state and verification**. They are good at bounded, verifiable work (code edits with tests, retrieval with citations) and bad at long-horizon tasks without a feedback signal. My working rule: an agent needs a cheap verifier per step, or the reliability compounds downward.

**中文要点**：对齐税不是均匀百分比，而是集中在困难 agent 任务的尾部；单轮 2% 过度拒答在 20 步轨迹里只剩 67% 成功率；当前 agent 的短板不是智力而是状态管理与可验证性。

---

### 38. Walk me through a project you owned end to end where you can explain exactly how the metric was computed

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/llm-evaluation

**Anchor answer**

Tell it as **constraint → decision → measurement → what broke**, and pre-empt the metric question by defining it before anyone asks.

A template that survives grilling:

| Beat | Content |
| :--- | :--- |
| Constraint | the real bottleneck (latency budget, label cost, GPU count), with a number |
| Decision | the one non-obvious choice, and the option you rejected plus why |
| Metric | exact definition: numerator, denominator, sampling, who labelled it, agreement |
| Result | delta with CI, and the cost delta |
| Failure | what broke after launch and how you found it |

On "how was the metric computed", the details that make it credible: the unit of measurement (per request? per claim? per session?), the denominator's exclusions (timeouts dropped or counted as failures — this single choice can move a number 10 points), annotator count and inter-annotator agreement, and whether the test set was touched during development. If you tuned on it, say so.

Also name the **guardrail metric** you watched that could have gone wrong — e.g. you improved resolution rate but watched escalation-to-human and repeat-contact rate. A project story where only one number improved is a story where no one looked.

Close with what you would do differently: the cheapest honest signal that you owned it rather than inherited it.

**中文要点**：按"约束→关键决策与放弃的方案→指标精确定义（分子分母、采样、标注人数与一致性）→带置信区间的结果→上线后出的问题"讲；一定要说清分母是否排除超时，以及你同时盯的护栏指标。

---

## 附：这 38 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 注意力/结构基本功 | 1、2、5、32、35 |
| 训练与对齐（能推导） | 11、12、13、14、15、16 |
| 推理优化真做过 | 6、7、8、9、10 |
| RAG 工程细节 | 17、18、19、20、21、29 |
| Agent 落地与调试 | 22、23、24、25、26 |
| 系统设计与成本意识 | 28、29、30、31 |
| 编码现场 | 32、33、34、35 |
| 价值观与项目真伪 | 36、37、38 |
