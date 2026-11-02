# 09 · LLM Internals and Architecture

> 16 题。出处：`README.en.md` § LLM Internals and Architecture。

---

### 1. Explain scaled dot-product attention and why the 1/sqrt(d_k) scaling factor matters

**Source**: https://outcomeschool.com/blog/scaling-dot-product-attention · https://outcomeschool.com/blog/math-behind-attention-qkv

**Anchor answer**

Attention is a soft key-value lookup: score every key against the query, softmax the scores into weights, return the weighted sum of values.

$$\text{Attn}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

For independent, roughly unit-variance $q,k\in\mathbb{R}^{d_k}$, the dot product has variance $d_k$ — so its standard deviation grows as $\sqrt{d_k}$. At $d_k=128$ logits scatter with σ≈11. Softmax over values that wide is effectively one-hot: the largest logit wins by a huge margin.

| | Unscaled | Scaled by $1/\sqrt{d_k}$ |
| :--- | :--- | :--- |
| Logit σ | $\sqrt{d_k}$ | 1 |
| Softmax entropy | near 0 | usable range |
| Gradient through softmax | ≈0 for non-winners | healthy |
| Behaviour | hard argmax, training stalls | soft mixture |

So the scale factor is not cosmetic: it keeps attention in the regime where softmax has a gradient and can actually mix several values. The derivation assumes initialization-time statistics; trained models drift, which is why modern stacks often add **QK-norm** (normalize q,k per head) or **logit soft-capping** (Gemma 2 caps logits with a tanh at ±50) rather than trusting the fixed constant alone.

When you can drop it: if you normalize q and k to unit length, the variance argument no longer applies.

**中文要点**：q·k 的方差是 d_k，不缩放时 logit 标准差 √d_k 让 softmax 退化成 one-hot、梯度消失；1/√d_k 把方差拉回 1。实践中靠 QK-norm 或 logit soft-cap 补，而不是死守这个常数。

---

### 2. What is the KV cache, and what are its memory implications at scale? Derive the formula

**Asked at**: OpenAI · xAI · Mistral AI · Amazon · Apple · NVIDIA · Together AI · Character.AI · ByteDance · Tencent · Meituan · Alibaba (Taobao/Tmall, Alibaba Cloud)
**Source**: https://outcomeschool.com/blog/kv-cache-in-llms · https://outcomeschool.com/blog/kv-cache-compression

**Anchor answer**

Decode is autoregressive, so K and V of past tokens never change. The cache stores them once; each new step projects only the new token's q and attends over cached K/V.

$$\text{bytes}=2\cdot L\cdot H_{kv}\cdot d_{head}\cdot T\cdot B\cdot b$$

where 2 = {K,V}, $L$ layers, $H_{kv}$ KV heads, $T$ context, $B$ batch, $b$ bytes per element.

Concrete: Llama-3-8B has $L=32$, $H_{kv}=8$, $d_{head}=128$, fp16 → $2\cdot32\cdot8\cdot128\cdot2 = 131{,}072$ B = **128 KiB per token**. A 32k context is 4 GiB for *one* sequence. The 70B model ($L=80$) costs 320 KiB/token → 10 GiB at 32k.

| Implication | Effect |
| :--- | :--- |
| Linear in $T\times B$ | concurrency collapses as context grows |
| Separate from weights | weights fit, KV doesn't — the usual OOM cause |
| Read every decode step | decode becomes HBM-bandwidth-bound, not FLOP-bound |
| Allocation granularity | contiguous per-request buffers waste 20–40% to fragmentation → PagedAttention |

Levers: GQA/MLA to cut $H_{kv}$ or the per-token width; KV quantization (fp8 roughly halves it, visible quality loss past ~4-bit); prefix caching for shared system prompts; eviction/compression for long chats.

**中文要点**：公式 `2·L·H_kv·d_head·T·B·b`；Llama-3-8B 每 token 128 KiB，32k 上下文单条就 4 GiB。KV 与权重分开算，它是并发上不去的真凶，也是分页/量化/GQA 存在的理由。

---

### 3. What are Multi-Query Attention (MQA) and Grouped-Query Attention (GQA), and what do they trade away?

**Asked at**: Meta · Mistral AI · ByteDance · Meituan · DiDi
**Source**: https://outcomeschool.com/blog/grouped-query-attention

**Anchor answer**

They are answers to one question: **how many query heads must share a K/V head?**

| Variant | KV heads | KV cache | Quality | Decode speed |
| :--- | :--- | :--- | :--- | :--- |
| MHA | $H$ (one per q head) | baseline | best reference | slowest |
| GQA | $G$, $1<G<H$ | $\times G/H$ | ≈MHA | near MQA |
| MQA | 1 | $\times 1/H$ | degrades | fastest |

The trade is explicit: **you are deleting the diversity of key/value subspaces**. MQA gives one K/V pair for all heads, so heads can only differ in their query, and training is less stable — it often needs longer training or careful init to recover the lost quality. GQA is the interpolation that industry settled on: typically 8 KV heads against 32–64 query heads, i.e. an 8× cache cut with MHA-level scores.

Two practical notes:

1. **GQA is convertible.** A trained MHA model can be uptrained into GQA by mean-pooling the KV heads inside each group and continuing pretraining for a few percent of original compute — this is how most "we switched to GQA" migrations happen, rather than pretraining from scratch.
2. **The head count is a serving parameter, not just a modeling one.** Fewer KV heads means a smaller working set per step, so throughput gains show up mainly at high concurrency and long context; at batch 1 the difference is small.

**中文要点**：本质是"多少个 query head 共享一份 K/V"。MQA 共享 1 份、GQA 共享 G 份：缓存按 G/H 缩，质量接近 MHA。MQA 掉点且训练不稳；GQA 可用均值池化 KV head 再短训迁移，是工业界默认。

---

### 4. What is Multi-head Latent Attention (MLA) and why did DeepSeek introduce it?

**Asked at**: DeepSeek · Moonshot AI · ByteDance · Meituan
**Source**: https://outcomeschool.com/blog/kv-cache-compression

**Anchor answer**

MLA decouples **cache size** from **number of heads**. Instead of caching K and V, it caches a single low-rank latent per token.

Per layer, per token:

1. Down-project the hidden state to a latent $c_t\in\mathbb{R}^{512}$ and cache that.
2. At use time, up-project $c_t$ into per-head K and V.
3. Apply RoPE to a **separate small key** $k^R_t\in\mathbb{R}^{64}$ that is also cached — because rotation does not commute with the low-rank absorption, so positional information has to travel outside the latent.

Cache width becomes $512+64=576$ floats against $2\cdot H\cdot d_{head}$ for MHA (tens of thousands). DeepSeek-V2 reports ~93% KV reduction and a 5.7× generation throughput gain over its dense predecessor.

The clever part is **weight absorption**: $W^{Q}W^{UK}$ and $W^{UV}W^{O}$ can be pre-multiplied, so decode multiplies q directly against the latent and never materializes K/V at all.

Costs: extra projections in prefill, custom kernels, and the latent must be learned — unlike GQA you cannot cheaply convert an existing MHA checkpoint. Choose MLA when long context × high concurrency dominates cost; stay with GQA when you want off-the-shelf kernels.

**中文要点**：MLA 缓存一个 512 维 latent + 64 维 decoupled RoPE key，而不是 K/V；靠权重吸收把 up-projection 折进 Q/O，解码时直接和 latent 相乘。换来 MHA 级质量 + MQA 级缓存，代价是 kernel 复杂、无法从已有 MHA 便宜迁移。

---

### 5. Explain FlashAttention. It does not reduce FLOPs, so why is it faster?

**Asked at**: Together AI · Tencent
**Source**: https://outcomeschool.com/blog/decoding-flash-attention

**Anchor answer**

Because attention here is **memory-bound, not compute-bound**, and FlashAttention is an IO-aware exact algorithm.

The naive kernel materializes $S=QK^\top$ ($N\times N$) and $P=\text{softmax}(S)$ in HBM: write $N^2$, read $N^2$ for softmax, write again, read for $PV$. That is several $N^2$ round trips at ~3 TB/s while the tensor cores idle.

FlashAttention tiles Q into blocks that fit in SRAM (~100–200 KB), streams K/V blocks through, and keeps a **running max $m$ and running sum $\ell$** — online softmax — rescaling the accumulator as it goes. The $N^2$ matrix never exists. Backward recomputes tiles from the cached statistics instead of storing them.

| | Naive | FlashAttention |
| :--- | :--- | :--- |
| FLOPs | same | same |
| HBM traffic | $O(N^2)$ writes+reads | $O(N)$ |
| Activation memory | $O(N^2)$ | $O(N)$ |
| Result | exact | exact |

Rule of thumb: fp16 attention needs ≳300 FLOP/byte to be compute-bound; the materialized version sits near $O(1)$.

Where the win disappears: **decode**, where $q$ is a single token and there is no $N^2$ tile to fuse — you need split-KV (flash-decoding) instead; and head dims above 256, which FA2 does not support.

**中文要点**：FlashAttention 不省 FLOPs，省的是 HBM 往返：分块进 SRAM + online softmax，永不落地 N×N 矩阵，反向重算。所以它是 IO-aware 而非近似。解码阶段 q 只有一行，收益消失，要换 flash-decoding。

---

### 6. How does Byte Pair Encoding work, and what are its failure modes (numbers, code, non-Latin scripts)?

**Asked at**: Alibaba · Sarvam AI · Hugging Face
**Source**: https://outcomeschool.com/blog/bpe-in-llms · https://www.youtube.com/watch?v=sK2s9I84EVI

**Anchor answer**

BPE is a greedy compression scheme turned tokenizer:

1. Start from a base alphabet (bytes, or chars + `<unk>`).
2. Count adjacent pairs over the corpus.
3. Merge the most frequent pair into a new symbol.
4. Repeat until the vocabulary budget is hit.
5. Encode by replaying merges in learned rank order, left to right.

Merges are **frequency-driven**, so anything rare gets shredded. That is the whole failure surface:

| Input | Failure | Mitigation |
| :--- | :--- | :--- |
| Numbers `1234` / `1235` | different tokens, no digit alignment; arithmetic and column copying break | split digits individually (Llama-2 regex), or force `[0-9]` |
| Hindi / Tamil | UTF-8 is 3 bytes per char → 2–3 tokens/char; fertility 3–4× English | larger vocab, language-specific tokenizer, byte-fallback |
| Chinese | ~1.5–2 tokens/char → effective context shrinks | 100k+ vocab |
| Code | long identifiers split inconsistently; indentation gets its own tokens; whitespace-sensitive languages suffer | code-aware pretokenization (GPT-4 style regex) |
| Prompts | `"hello"` ≠ `" hello"` — leading space is part of the token | keep few-shot formatting byte-identical |

Also: under-trained merges produce **glitch tokens** that behave as garbage embeddings, and the vocab is frozen — swapping tokenizers invalidates every embedding matrix downstream.

**中文要点**：BPE = 按频次反复合并最常见相邻对，编码时按 merge 排名重放。频率驱动 → 稀有形态必被切碎：数字不对齐、印地语 2~3 token/字、代码标识符抖动、`" hello"` 与 `"hello"` 不同 token。词表一旦定死，换 tokenizer 等于废掉 embedding。

---

### 7. What is positional encoding in transformers, and how has it evolved (sinusoidal → learned → RoPE → ALiBi)?

**Source**: https://outcomeschool.substack.com/p/positional-embeddings-in-llms · https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding

**Anchor answer**

Self-attention is permutation-invariant, so order must be injected. The designs differ only in **where** position enters and **how** it extrapolates.

| Scheme | Where it enters | Params | Extrapolation | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Sinusoidal | added to input embedding | none | poor | fixed geometric frequencies, base 10000 |
| Learned absolute | added to input embedding | $L_{max}\times d$ | none | GPT-2; hard wall at $L_{max}$ |
| Relative bias (T5) | added to attention logits per head, bucketed | small | moderate | learned buckets, not distance-continuous |
| RoPE | rotates q and k per head | none | needs scaling | relative by construction, decays with distance |
| ALiBi | linear penalty on logits, per head slope | none | good | cheapest; fixed prior that far = less relevant |

RoPE won because it gets relative positioning *without* a bias tensor: the dot product of two rotated vectors depends only on $m-n$. That also gives a free inductive bias — scores decay as distance grows — and composes with paged/streaming kernels.

The interesting counter-current is **NoPE**: a causal decoder can infer position from the causal mask alone, and some ablations show no worse length generalization. Modern practice is RoPE plus an extension recipe, not a fixed absolute table.

**中文要点**：位置编码的差别在"注入位置"和"外推能力"：正弦/可学习绝对位置加到输入上，到 max_len 就硬墙；RoPE 旋转 q/k 使点积只依赖相对距离且无参数；ALiBi 直接给 logit 加线性惩罚，最便宜也最敢外推。

---

### 8. Explain RoPE and how position interpolation / YaRN extend context beyond the trained length

**Asked at**: Meta · Moonshot AI · Alibaba · Tencent · Meituan
**Source**: https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding

**Anchor answer**

RoPE splits each head's $d$-dim vector into $d/2$ pairs and rotates pair $i$ at position $m$ by angle $m\theta_i$, with $\theta_i=\text{base}^{-2i/d}$. Because rotation is orthogonal, $\langle R_mq, R_nk\rangle$ depends only on $m-n$: you get relative positions with zero parameters. Frequencies decay geometrically, so low-$i$ dimensions spin fast (local order) and high-$i$ dimensions spin slowly (global position).

Beyond training length, the naive fix — **position interpolation** — maps $m\mapsto m\cdot L_{train}/L_{new}$, compressing everything. It works but damages the fast dimensions: nearby tokens become nearly indistinguishable, so local syntax degrades and fine-tuning is mandatory.

| Method | What it scales | Cost |
| :--- | :--- | :--- |
| PI | positions, uniformly | breaks high-freq dims, needs full fine-tune |
| NTK-aware | the base (frequencies) | preserves high-freq, coarse |
| **YaRN** | per-dimension ramp + temperature | ~400 steps, ~0.1% of pretraining |

YaRN has two parts worth remembering: **NTK-by-parts** (dims whose wavelength < trained length are left alone; dims longer than the new context are interpolated; a ramp blends between) and an **attention temperature** factor $\sqrt{1/t}$ that compensates for the entropy collapse of attending over more tokens. The temperature is the piece people forget — without it, extended models become overconfident and lose perplexity.

**中文要点**：RoPE 旋转 q/k，使内积只依赖相对位置；高频维管局部、低频维管全局。PI 统一压缩会毁掉高频；YaRN = NTK-by-parts 按维分段插值 + 注意力温度补偿熵塌缩，只需几百步微调。

---

### 9. What do the Chinchilla scaling laws say, and how do they differ from earlier scaling intuitions?

**Asked at**: Anthropic

**Anchor answer**

Chinchilla fit $L(N,D)=E+A/N^{\alpha}+B/D^{\beta}$ and concluded that for a **fixed compute budget**, parameters and tokens should grow in equal proportion — roughly **20 tokens per parameter**. The headline result: a 70B model trained on 1.4T tokens beat a 280B model trained on 300B, at the same compute.

The earlier Kaplan-style intuition said the opposite: spend compute on parameters and keep data comparatively small. The discrepancy is largely methodological — the original fits didn't hold the learning-rate schedule constant as token count varied, which made extra data look less valuable than it is.

| | Kaplan-era | Chinchilla |
| :--- | :--- | :--- |
| Allocation | scale $N$ fast, $D$ slowly | scale $N$ and $D$ together |
| Tokens/param | ~2–5 | ~20 |
| Data assumed | effectively scarce | the binding constraint |

The modern twist: **compute-optimal is not deployment-optimal**. Parameters are paid for once, but inference cost is paid per generated token, forever. So production models are deliberately *over-trained* relative to Chinchilla — an 8B model on 15T tokens is ~1,900 tokens/param, ~100× the Chinchilla ratio — trading training FLOPs for a permanently cheaper model. Repetition past roughly 4 epochs starts to degrade quality, which is why data quality, not raw volume, becomes the constraint.

**中文要点**：Chinchilla 说固定算力下参数量与 token 数同比例涨，约 20 token/参数；Kaplan 时代偏向堆参数，部分源于学习率调度没对齐。但部署算的是终身推理成本，所以生产模型故意"超训"（8B 喂 15T），代价是重复数据超过约 4 轮会掉点。

---

### 10. What is a mixture-of-experts architecture and how does it scale capacity without scaling FLOPs?

**Asked at**: Mistral AI · Cohere · DeepSeek · Moonshot AI · Zhipu AI · Alibaba · Tencent · Meituan · Alibaba (Taobao/Tmall, Alibaba Cloud)
**Source**: https://outcomeschool.com/blog/mixture-of-experts

**Anchor answer**

MoE keeps the attention block dense and replaces each FFN with $E$ parallel experts plus a router:

$$y=\sum_{i\in\text{top-}k} g_i\,E_i(x),\qquad g=\text{softmax}(\text{top-}k(W_r x))$$

Only $k$ experts (usually 2) fire per token, so **FLOPs track active parameters while memory tracks total parameters**. Capacity — the number of distinct functions the layer can represent — grows with $E$; per-token compute does not.

| Benefit | Cost |
| :--- | :--- |
| more params at equal training FLOPs | all experts must be resident in memory |
| higher quality per FLOP | all-to-all communication in expert parallelism |
| conditional compute | routing instability, load collapse |

The real engineering is in balancing. A naive top-2 router collapses onto a few experts; fixes are an auxiliary load-balancing loss, a capacity factor with token dropping, or DeepSeek-V3's **aux-loss-free** bias adjustment on router logits. DeepSeekMoE also splits experts finer and keeps one **shared expert** always active to hold common knowledge.

Failure mode to state out loud: at small batch, an MoE can be *slower per token* than a dense model with the same active FLOPs, because each token scatters across GPUs. MoE wins only at large batch on fast interconnect.

**中文要点**：MoE 用 router 做 top-k 选择，只激活少数专家 → 参数量（容量）涨、FLOPs 不涨。真正难的是负载均衡（辅助损失 / 无辅助损失的 bias 调整）、all-to-all 带宽，以及小 batch 下反而不如稠密模型快。

---

### 11. Explain the difference between pre-training, supervised fine-tuning and preference optimisation

**Asked at**: Meta · Scale AI
**Source**: https://outcomeschool.com/blog/decoding-instructgpt · https://outcomeschool.com/blog/reinforcement-learning-from-human-feedback-rlhf

**Anchor answer**

Three stages, three different signals, three different things they can actually change.

| Stage | Data | Objective | Scale | What it installs |
| :--- | :--- | :--- | :--- | :--- |
| Pre-training | trillions of raw tokens | next-token CE | 10²²–10²⁵ FLOPs | knowledge, syntax, reasoning substrate |
| SFT | 10k–1M demonstrations | CE on response tokens only | hours–days | format, persona, tool protocol |
| Preference | chosen/rejected pairs | RM + PPO with KL, or DPO/GRPO | days | which of two valid answers is preferred |

The key asymmetry: **SFT cannot add knowledge it failed to learn in pretraining** — it surfaces and formats latent capability. Preference optimisation cannot add capability either; it reweights the distribution, which is why it is the stage that fixes tone, refusal calibration, verbosity and safety, and also the stage that *breaks* things (reward hacking, sycophancy, entropy collapse).

Ordering matters. SFT before preference, because preference methods assume the model can already produce well-formed candidates. And SFT on 1k curated examples routinely beats 100k noisy ones — the failure of large-scale SFT is style collapse, where the model parrots the annotator's voice across every domain.

**中文要点**：预训练给知识和能力，SFT 只改"怎么答"（暴露而非新增能力），偏好优化只改"两个都对时选哪个"（语气、拒答、安全），也最容易引入谄媚和奖励作弊。顺序不能反，SFT 数据质量远胜数量。

---

### 12. Compare greedy, beam search, top-k, top-p and temperature sampling. When does each fail?

**Asked at**: Google DeepMind · Apple · Perplexity
**Source**: https://outcomeschool.com/blog/how-does-temperature-control-llm-output · https://outcomeschool.com/blog/how-do-top-k-and-top-p-sampling-work

**Anchor answer**

Temperature reshapes the distribution; the truncation methods decide how much of the tail survives.

| Method | Mechanism | Fails when |
| :--- | :--- | :--- |
| Greedy | argmax | repetition loops, no diversity; misses better global continuations |
| Beam search | keep $b$ highest joint-probability prefixes | bland, repetitive in open-ended chat; good for MT and constrained decoding |
| Temperature | divide logits by $T$ | $T\!\to\!0$ degenerates to greedy; $T>1.2$ produces incoherence, and it rescales the *whole* tail |
| Top-k | keep $k$ most likely | fixed $k$ ignores distribution shape: on a flat dist it truncates valid mass, on a peaked dist it still admits junk |
| Top-p | keep smallest set with cumulative prob $p$ | at high temperature the nucleus balloons; $p$ is model-specific |
| Min-p | keep tokens with $p_i>\text{min\_p}\cdot p_{max}$ | newer, less battle-tested |

Pipeline order in most stacks: **temperature → top-k → top-p**, applied left to right on logits.

Two practical cautions. First, sampling parameters are not portable across models — a $p=0.9,T=0.7$ that works on one tokenizer's logit scale will behave differently on another. Second, reproducibility requires the seed **and** identical batch composition and kernels; floating-point nondeterminism in batched matmuls means the same seed can still drift. For JSON or code, constrain with a grammar, not with temperature games.

**中文要点**：temperature 改分布形状，top-k/top-p/min-p 截尾巴；顺序是 temp → top-k → top-p。greedy 会复读，beam 在开放对话里寡淡，top-k 的固定 k 对分布形状不敏感，高温下 top-p 的核会膨胀。结构化输出应上语法约束而不是调温度。

---

### 13. What is the lost-in-the-middle problem in long contexts and how do you address it?

**Asked at**: Moonshot AI
**Source**: https://outcomeschool.com/blog/lost-in-the-middle-problem-in-llms

**Anchor answer**

Accuracy as a function of where the gold evidence sits in the context is **U-shaped**: best when the passage is at the beginning or the end, worst in the middle. It holds even when retrieval is perfect, so it is an attention/utilisation failure, not a recall failure.

Causes, in order of how much you can act on them:

1. **Training distribution** — answers usually appear near the start of documents, so the model learns a positional prior.
2. **Attention sinks and recency** — the first tokens and the most recent tokens get disproportionate attention mass.
3. **Interference** — with 20 distractors, the middle is where the most similar-but-wrong passages sit.

Mitigations that actually move numbers:

| Tactic | Why it works |
| :--- | :--- |
| Rerank, then place best evidence first *and* restate at the end | both ends of the U |
| Force span citation before answering | makes the model attend, not skim |
| Compress chunks / summarise per chunk | shrinks the middle |
| Fewer, better chunks | less interference |
| Explicit IDs and headers | gives the model a handle to refer back |

**Measure it**: take a multi-doc QA set, permute the gold passage's position, plot accuracy. If the curve is flat, you do not have the problem; if it sags 10–20 points, no amount of "1M context" marketing saves you.

**中文要点**：金标证据放开头或结尾准确率最高、放中间明显掉点，且检索全对时依然存在——是"用不上"不是"找不到"。对策：重排后首尾各放一次、先引用再回答、压缩分块、减少干扰项；用位置置换实验量化。

---

### 14. Why is LayerNorm placed pre-block in modern transformers, and what is RMSNorm?

**Source**: https://outcomeschool.com/blog/rmsnorm-root-mean-square-layer-normalization

**Anchor answer**

The original transformer normalized **after** the residual add (post-LN): $x=\text{LN}(x+F(x))$. That puts a normalization on the residual highway, so gradients must pass through LN at every block; deep stacks need learning-rate warmup and careful init or they diverge.

Pre-LN moves the norm inside the branch: $x=x+F(\text{LN}(x))$. The residual path is now a pure identity, gradients flow straight to the embedding, and training is stable without warmup. The price: the residual stream's norm grows with depth (roughly $\sqrt{L}$), so a **final norm before the LM head is mandatory** — omit it and logits blow up as depth increases.

RMSNorm is the cheaper normalization used almost everywhere now:

$$y=\frac{x}{\sqrt{\frac{1}{d}\sum x_i^2+\epsilon}}\odot g$$

No mean subtraction, no bias, one learned gain vector. The empirical finding is that re-centering is not what makes normalization work — re-scaling is. It removes a reduction and a subtraction per element (single-digit to low-double-digit percent faster normalization) at parity quality.

Related: some stacks add **QK-norm** (RMSNorm on q and k per head) to keep attention logits bounded at high learning rates and long context — that is the same trick applied where the softmax saturates.

**中文要点**：post-LN 把归一化放在残差主路上，梯度必须穿过 LN，深网必须 warmup；pre-LN 把 LN 塞进分支，残差是恒等路径，稳定但不加最后的 norm 会导致 logits 随深度爆炸。RMSNorm 去掉减均值，只做缩放，更快且效果持平。

---

### 15. Explain SwiGLU and why gated activations replaced ReLU/GELU in modern LLM MLP blocks

**Asked at**: Meta
**Source**: https://outcomeschool.com/blog/feed-forward-networks-in-llms

**Anchor answer**

SwiGLU is a gated linear unit: one projection is passed through Swish and used as a **multiplicative gate** on a second projection.

$$\text{FFN}(x)=W_2\big(\text{Swish}(xW_1)\odot xW_3\big),\qquad \text{Swish}(z)=z\,\sigma(z)$$

Why it beats ReLU/GELU: a fixed elementwise nonlinearity is input-independent in shape, while a gate lets the network modulate each feature by another learned function of the same input — an extra multiplicative interaction at almost no depth cost. Shazeer's 2020 ablation found gated variants improve perplexity at equal parameters and compute, and that one gated layer can match roughly two dense layers in the low-depth regime.

| | ReLU/GELU MLP | SwiGLU |
| :--- | :--- | :--- |
| Matrices | 2 ($d\!\to\!d_{ff}$, $d_{ff}\!\to\!d$) | 3 (two up-projections) |
| Params at $d_{ff}=4d$ | $8d^2$ | $12d^2$ |
| Equal-param choice | $d_{ff}=4d$ | $d_{ff}=\tfrac{8}{3}d$ |

That $\tfrac{8}{3}$ is the detail people miss: implementations round it to a hardware-friendly multiple, which is why Llama uses $d_{ff}=14336$ for $d=4096$ (3.5×, not 2.67×).

Costs: an extra matmul to shard for tensor parallelism, more activation memory. For tiny edge models, plain GELU is still simpler.

**中文要点**：SwiGLU 用一条 Swish 支路做乘法门控另一条支路，多了一次乘性交互；同参数同算力下困惑度更好。代价是多一个权重矩阵，所以 d_ff 从 4d 降到 8/3·d 保持参数不变（Llama 取 14336 是对齐硬件的近似）。

---

### 16. Walk me through what happens, tensor by tensor, in one forward pass of a decoder-only transformer

**Asked at**: Anthropic
**Source**: https://outcomeschool.com/blog/decoding-transformer-architecture

**Anchor answer**

Using Llama-3-8B dimensions: $d=4096$, $L=32$, $H=32$ q-heads, $H_{kv}=8$, $d_{head}=128$, $d_{ff}=14336$, $V=128256$.

| # | Step | Shape | Note |
| :-: | :--- | :--- | :--- |
| 1 | tokenize → ids | $[B,T]$ | int64 |
| 2 | embedding lookup | $[B,T,4096]$ | pure gather, no matmul |
| 3 | RMSNorm | $[B,T,4096]$ | pre-norm |
| 4 | QKV projection | $[B,T,4096]$, $[B,T,1024]$, $[B,T,1024]$ | $H_{kv}d_{head}=1024$ |
| 5 | RoPE | same | rotate q,k pairs by position |
| 6 | causal attention | $[B,32,T,T]$ implicit | KV heads broadcast to 4 q-heads each; fused, never materialized |
| 7 | output projection + residual | $[B,T,4096]$ | |
| 8 | RMSNorm → gate/up → SiLU ⊙ → down | $[B,T,14336]\times2 \to [B,T,4096]$ | the bulk of the FLOPs |
| 9 | residual add, ×32 layers | $[B,T,4096]$ | |
| 10 | final norm → LM head | $[B,T,128256]$ | logits |
| 11 | sample | $[B,T]$ | |

Sanity numbers: params $\approx 12Ld^2+Vd\approx 8$B; FLOPs per token $\approx 2N\approx16$ GFLOP. Prefill of $T{=}4096$ is one big matmul batch ≈64 TFLOP → fractions of a second at a few hundred TFLOP/s effective. Decode is $T{=}1$: same 16 GFLOP but it must **read all 16 GB of weights**, so at 3 TB/s you get ~200 tok/s — bandwidth-bound, which is exactly why batching exists.

**中文要点**：按张量走一遍：ids → embedding → 每层（RMSNorm → QKV 投影 → RoPE → 因果注意力 → O 投影 → 残差 → RMSNorm → SwiGLU 上/门/下 → 残差）→ 末尾 norm → LM head → 采样。参数量 ≈12Ld²+Vd；prefill 是大矩阵乘，decode 是 T=1 的读权重，所以卡在带宽。

---

## 附：这 16 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 真读过论文 | 5（FlashAttention）、8（YaRN）、9（Chinchilla） |
| 真调过服务 | 2（KV 公式）、3（GQA）、4（MLA） |
| 真训过模型 | 10（MoE 均衡）、11（三阶段）、14（pre-LN/RMSNorm） |
| 只是背概念 | 7（位置编码史）、12（采样）、13（lost-in-middle） |
| 白板手推 | 1（√d_k）、16（逐张量前向） |
