# 34 · Meituan

> 34 题。出处：`README.en.md` § Meituan。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Walk through the Transformer, and explain why the encoder needs an FFN.

**Anchor answer**

Data flow for one block (encoder): `x → LayerNorm → multi-head self-attention → residual → LayerNorm → FFN → residual`. Stack N blocks, then a task head. The decoder adds masked self-attention plus cross-attention to the encoder output.

Why the FFN is not optional:

| Function | Mechanism |
| :--- | :--- |
| Per-token non-linearity | attention is a **weighted average** — a convex combination of value vectors. Without a non-linear map afterwards, stacking blocks still collapses to a linear map of the input; depth buys nothing |
| Feature transformation | attention *routes* information between positions; the FFN *processes* what arrived, at 4× the width, giving room for intermediate concepts |
| Key-value memory view | the first FFN matrix behaves like pattern detectors, the second like a lookup that writes associations back — this is the "FFN as key-value memory" reading of transformer interpretability |
| Capacity | attention params are O(d²) shared across positions; FFN is where ~2/3 of parameters live |

Practical consequence: because FFN dominates parameters, MoE replaces **FFN layers** with experts, not attention. And because attention only averages, replacing it with a linear attention variant costs quality mainly on tasks needing sharp token-to-token lookup (copying, exact recall), while the FFN stays untouched.

**中文要点**：attention 只是对 value 的凸组合（加权平均），没有 FFN 的非线性逐 token 变换，堆多少层都等价于线性映射；FFN 占约 2/3 参数，所以 MoE 换的是 FFN 而不是 attention。

---

### 2. Which attention variants do you know? Write the self-attention formula on the spot.

**Source**: https://outcomeschool.com/blog/math-behind-attention-qkv and https://outcomeschool.com/blog/self-attention-in-transformers

**Anchor answer**

$$\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V,\quad Q=XW_Q,\;K=XW_K,\;V=XW_V$$

The $\sqrt{d_k}$ keeps the logits' variance at O(1) so softmax doesn't saturate; $M$ is the additive mask (−∞ for masked positions).

| Variant | Change | Buys |
| :--- | :--- | :--- |
| MHA | baseline, h independent heads | expressiveness |
| MQA | all heads share one K/V | KV cache ÷ h, small quality loss |
| GQA | groups of heads share K/V | the practical middle ground (8 KV heads for 32 Q heads is common) |
| MLA | cache a low-rank latent, up-project | far smaller cache, quality near MHA |
| Sliding-window / local | mask beyond w | O(n·w) attention, long sequences |
| Linear / SSM hybrids | kernel trick or recurrence | O(n) but weaker exact recall |
| Sparse / top-k | attend to selected positions | long context on a budget |
| Cross-attention | K,V from encoder | multimodal / retrieval fusion |

Interview trap: saying "multi-head is h separate attentions" is fine, but you must add that heads are **concatenated then projected by $W_O$** — the mixing happens there, not in the heads.

**中文要点**：写出 softmax(QKᵀ/√d_k + M)V；√d_k 是压住 logits 方差防止 softmax 饱和；MQA/GQA 省 KV（GQA 8 组 KV 对 32 个 Q 头是常见折中），MLA 用低秩潜变量压得更多；多头最后要 concat 再乘 W_O 才混合。

---

### 3. How does MLA reduce the KV cache?

**Source**: https://outcomeschool.com/blog/kv-cache-compression

**Anchor answer**

MLA (DeepSeek-V2) stops caching per-head K and V. Instead it caches a **single low-rank latent per token**, plus a small decoupled RoPE key.

| Item | MHA | MLA |
| :--- | :--- | :--- |
| Cached per token | $2 \cdot n_h \cdot d_h$ | $d_c + d_h^{R}$ |
| Example (128 heads × 128) | ~32k values | ~576 values |
| Position handling | RoPE inside each head's K | RoPE applied to a separate small key, concatenated |

Mechanism: down-project the hidden state to $c_t = W_{DKV} h_t$, cache $c_t$; reconstruct $K = W_{UK}c_t$, $V = W_{UV}c_t$ at attention time. Because both projections are linear, they can be **absorbed** into $W_Q$ and $W_O$ at inference, so you never materialise the full per-head K/V — you attend directly against the latent.

Why the decoupled RoPE key matters: RoPE is position-dependent and multiplicative, so it cannot be folded into a low-rank linear projection without breaking the relative-position property. Carrying a small 64-dim RoPE key alongside the latent solves that at negligible cache cost.

Trade-off: more compute per step (extra projections) and a harder training setup; cache-bound serving wins, compute-bound training loses slightly. It is the right trade when concurrency, not FLOPs, is the constraint.

**中文要点**：MLA 不再按头缓存 K/V，而是每 token 缓存一个低秩潜变量 c（约 576 维 vs 3 万+），推理时把上投影吸收进 W_Q/W_O 直接对潜变量做注意力；RoPE 必须走一条小的解耦 key，因为乘法性位置编码无法折进线性低秩投影。

---

### 4. How does Qwen extrapolate to longer contexts (NTK-aware scaling, YaRN)?

**Source**: https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding

**Anchor answer**

RoPE rotates dimension pairs by angle $m\theta_i$ with $\theta_i = b^{-2i/d}$. At inference, dimension $i$ has only ever seen positions up to the training length — beyond that the rotation wraps into unseen angles and attention degrades.

Three fixes, increasing in sophistication:

| Method | What it changes | Note |
| :--- | :--- | :--- |
| Position interpolation (PI) | divide position by $s$ | uniform, blurs local resolution; needs fine-tuning |
| NTK-aware | raise the base $b \to b\cdot s^{d/(d-2)}$ | high-frequency dims barely move, low-frequency dims stretch — no training needed for short extensions |
| Dynamic NTK | recompute the base per sequence length | stays exact below the training length |
| YaRN | NTK-by-parts: ramp between interpolation and extrapolation per dimension, plus an attention temperature $1/\sqrt{t}$ | the current default; small fine-tune (hundreds of steps) fixes the rest |

Why "by parts": low-frequency dimensions encode long-range position and must be interpolated; high-frequency dimensions encode local order and must stay untouched, or local token order blurs. The temperature term compensates for the entropy increase of attention when the context lengthens.

Qwen ships YaRN configs (e.g. factor 4 on a 32k base) and applies dynamic NTK for short prompts. The honest caveat: "supported length" ≠ "effective length" — measure with a needle test at multiple depths before trusting it.

**中文要点**：RoPE 角度只在训练长度内见过；PI 均匀压缩会糊掉局部，NTK-aware 抬 base 让高频维几乎不动、低频维拉伸；YaRN 用 by-parts 斜坡 + 注意力温度补偿，配几百步微调。宣称长度 ≠ 有效长度，必须做多深度 needle 测试。

---

### 5. How does MoE work, how is load balanced, and is activation a batch-level or token-level decision? Implement a MoE in Python.

**Source**: https://outcomeschool.com/blog/mixture-of-experts

**Anchor answer**

**Token-level, by default.** Each token independently gets a gate distribution over E experts, top-k selected, outputs weighted by gate probability and summed. So different tokens in the same sequence can hit different experts — there is no "the batch uses expert 3" decision. The batch-level alternative is *expert-choice* routing (each expert picks its top-C tokens), which balances load by construction but breaks causality for autoregressive decoding, so it is rarely used in LLMs.

Load balancing, three mechanisms:

| Mechanism | How | Cost |
| :--- | :--- | :--- |
| Auxiliary loss | penalise the squared deviation of expert load from uniform (Switch/GShard) | one extra hyperparameter that fights the main objective |
| Capacity factor + drop | cap tokens per expert per batch, overflow goes residual-only | dropped tokens silently lose capacity |
| Aux-loss-free bias | nudge a per-expert routing bias up/down by observed load | no gradient interference — DeepSeek's approach |

Failure modes to watch: expert collapse (a few experts take everything, effective params shrink), routing instability across steps, and all-to-all communication dominating step time in expert parallelism.

**Reference implementation**

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, top_k):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
            for _ in range(n_experts)
        ])
        self.bias = torch.zeros(n_experts)   # aux-loss-free balancing

    def forward(self, x):                    # x: [B, T, D]
        B, T, D = x.shape
        logits = self.gate(x) + self.bias     # [B, T, E]
        probs = F.softmax(logits, dim=-1)
        w, idx = probs.topk(self.top_k, dim=-1)          # [B,T,k]
        w = w / w.sum(-1, keepdim=True)                  # renormalise
        flat = x.reshape(-1, D)
        out = torch.zeros_like(flat)
        flat_idx, flat_w = idx.view(-1, self.top_k), w.view(-1, self.top_k)
        for e, expert in enumerate(self.experts):
            mask = flat_idx == e
            tok, slot = mask.nonzero(as_tuple=True)      # tokens routed to e
            if tok.numel() == 0:
                continue
            out.index_add_(0, tok, expert(flat[tok]) * flat_w[tok, slot].unsqueeze(-1))
        return out.view(B, T, D)
```

**Key points**

- Loop over experts (not tokens): each expert runs one batched matmul on its gathered tokens.
- Renormalise top-k weights, or output scale varies with k.
- This naive version materialises gathers; production uses grouped GEMM / all-to-all with capacity limits.
- Track per-expert token counts every step — collapse shows up as a max/mean ratio far above 1.

**中文要点**：MoE 是**逐 token** 选专家（expert-choice 的批级路由破坏自回归因果性）；负载均衡靠辅助损失、容量因子丢 token、或无辅助损失的 bias 微调；实现时按专家 gather 做批量矩阵乘，并监控每个专家的 token 数比值。

---

### 6. What is the difference between bf16 and fp16, how many bits is each, and how do you choose precision in training?

**Source**: https://outcomeschool.com/blog/how-does-model-quantization-work

**Anchor answer**

Both are 16 bits. The split differs:

| | Sign | Exponent | Mantissa | Dynamic range | Precision |
| :--- | :-: | :-: | :-: | :--- | :--- |
| fp16 | 1 | 5 | 10 | ~6e-5 … 65504 | ~3 decimal digits |
| bf16 | 1 | 8 | 7 | same as fp32 (~1e-38…3e38) | ~2 decimal digits |

Consequence: fp16 **overflows and underflows** — gradients below 6e-5 flush to zero, so you need loss scaling (dynamic `GradScaler`) and you still fight NaN in attention with large logits. bf16 has fp32's range so it needs no scaling; the cost is coarser mantissa, which shows up as slower convergence on some small-model or precision-sensitive work (older results on RL and layernorm-heavy nets).

Choice:

1. **bf16 whenever the hardware supports it** (A100/H100/TPU/Apple/Recent consumer GPUs). Default for LLM training and inference.
2. **fp16 + loss scaling** only on pre-Ampere (V100/T4) — and then keep fp32 master weights in the optimizer.
3. **fp32** for small models, numerical debugging, and any layer where you see NaN (softmax logits, layernorm stats).
4. Mixed precision always: fp32 master weights, bf16 compute, fp32 optimizer states.

Practical note: for inference, fp16 vs bf16 barely matters; for training stability it matters enormously, and "switch to bf16" fixes most mystery NaN reports.

**中文要点**：都是 16 位——fp16 是 5 位指数/10 位尾数（精度高但范围窄、必须 loss scaling），bf16 是 8 位指数/7 位尾数（范围同 fp32、免缩放但尾数粗）；有 Ampere 以上就选 bf16，老卡才用 fp16+GradScaler，优化器状态永远留 fp32。

---

### 7. What memory optimisation strategies exist for attention computation?

**Source**: https://outcomeschool.com/blog/kv-cache-in-llms and https://outcomeschool.com/blog/paged-attention-in-llms

**Anchor answer**

Two different memory problems, often confused:

**A. Training / prefill — the $n^2$ score matrix**

| Technique | Effect |
| :--- | :--- |
| FlashAttention (tiling + online softmax) | never materialises the $n\times n$ matrix in HBM; O(n) memory, faster from fewer HBM round trips |
| Activation recomputation | store only block inputs, recompute in backward; ~30% more compute for much less memory |
| Sequence packing + varlen kernels | no padding waste across samples |
| Ring / context parallel attention | split the sequence across devices for very long context |

**B. Decoding — the KV cache**

| Technique | Effect |
| :--- | :--- |
| GQA / MQA | fewer KV heads → cache shrinks by the group factor |
| MLA | cache a low-rank latent instead of per-head K/V |
| PagedAttention | block-level allocation, no max-length over-reservation; enables prefix sharing |
| KV quantisation (int8/fp8) | ~2× smaller cache; measure quality, not perplexity |
| Sliding window / sparse attention | bound the cache to the last w tokens |
| Prefix cache reuse | shared system prompt computes once |

Rule of thumb for capacity planning: KV bytes/token = 2 × layers × kv_heads × head_dim × bytes. At 32 layers, 8 KV heads, 128 dim, fp16 that is ~128 KB/token — 8k context × 100 concurrent sessions ≈ 100 GB. That number, not FLOPs, decides your batch size.

**中文要点**：训练/prefill 的瓶颈是 n² 分数矩阵（FlashAttention 分块+在线 softmax 不落 HBM、激活重算）；解码的瓶颈是 KV cache（GQA/MQA、MLA、PagedAttention 分块分配、KV 量化、滑窗）；先用 KV 字节公式算容量，再决定并发。

---

### 8. What is the core difference between ZeRO-2 and ZeRO-3 in distributed training?

**Anchor answer**

ZeRO partitions optimizer state, then gradients, then parameters across data-parallel ranks.

| Stage | Sharded | Per-GPU memory (Ψ params) | Communication |
| :--- | :--- | :--- | :--- |
| DDP | nothing | 16Ψ bytes (fp16 weights + fp32 master + Adam m,v) | all-reduce grads |
| ZeRO-1 | optimizer states | ~4Ψ + 12Ψ/N | all-reduce grads |
| ZeRO-2 | + gradients | ~2Ψ + 14Ψ/N | reduce-scatter grads (cheaper) |
| ZeRO-3 | + parameters | ~16Ψ/N | reduce-scatter + **all-gather params per layer** |

The core difference: **ZeRO-2 still holds the full parameter set on every GPU**; ZeRO-3 does not, so it can train a model whose parameters don't fit on one device. The price is that parameters must be all-gathered layer-by-layer in forward and again in backward, roughly 1.5× the communication volume of ZeRO-2, and latency-sensitive small layers suffer most.

What this means in practice:

- **ZeRO-2 is usually faster** at the same model size when it fits — less communication, simpler overlap.
- **ZeRO-3 is required** past the point where weights + grads + states exceed device memory, and it pairs badly with small micro-batches (all-gather overhead amortises over batch size).
- Beyond a single node, combine with tensor/pipeline parallelism; pure ZeRO-3 across nodes is bandwidth-bound on interconnect.

**中文要点**：ZeRO-2 只切优化器状态和梯度，**每卡仍持有完整参数**；ZeRO-3 连参数一起切，能训单卡放不下的模型，代价是每层前反向各一次 all-gather、通信约 1.5 倍；装得下就选 ZeRO-2 更快。

---

### 9. Why do LLMs stream token by token, and what protocol is underneath?

**Source**: https://outcomeschool.com/blog/how-does-token-streaming-work

**Anchor answer**

Because the model already produces tokens one at a time (autoregressive decoding), and non-streaming wastes that: total latency is the same, but **time-to-first-token** becomes the whole generation time. Streaming turns a 6-second wait into 300 ms + progressive reading, which is a product win, not a compute win.

Underneath, three layers:

| Layer | Mechanism |
| :--- | :--- |
| Transport | HTTP/1.1 `Transfer-Encoding: chunked`, or HTTP/2/3 streaming; WebSocket only if you need bidirectional |
| Protocol | SSE: `text/event-stream`, repeated `data: {...}\n\n` frames, terminated by `data: [DONE]`; each frame carries a JSON delta |
| Payload | OpenAI-style `choices[0].delta.content`, plus `finish_reason` on the last chunk |

Traps that only show up in production:

1. **Detokenisation is not incremental by default** — a token can be a partial UTF-8 sequence or half of a byte-fallback token. Buffer until the decoded string is valid, or you emit `�`.
2. **Reverse proxies buffer.** Nginx needs `proxy_buffering off`; a CDN can hold the whole stream and destroy the benefit. Compression (gzip) also buffers unless flushed per chunk.
3. **Usage and error reporting arrive last** — you must reconcile token counts from the final event, and handle a stream that dies mid-flight without a `finish_reason`.
4. Keep-alive comment frames (`: ping`) are needed to survive idle timeouts during long tool calls.

**中文要点**：模型本来就是逐 token 生成，流式把"等 6 秒"变成"300ms 出首字"；底层是 chunked HTTP + SSE 的 `data:` 帧（末尾 `[DONE]`）；坑在于增量解码（半个 UTF-8 字符）、反向代理缓冲区、以及中途断流没有 finish_reason。

---

### 10. How does LoRA work, and which hyperparameters did you tune and what did you learn?

**Source**: https://outcomeschool.com/blog/lora-low-rank-adaptation-of-llms

**Anchor answer**

LoRA freezes $W$ and learns $\Delta W = BA$ with $B\in\mathbb{R}^{d\times r}$, $A\in\mathbb{R}^{r\times k}$, scaled by $\alpha/r$. Only $A,B$ get gradients and optimizer state, so memory drops from 16Ψ to ~2Ψ + tiny. At inference you can merge: $W' = W + \frac{\alpha}{r}BA$, so zero added latency.

What I actually tuned, in order of impact:

| Hyperparameter | Typical | What I learned |
| :--- | :--- | :--- |
| Target modules | all linear (q,k,v,o,gate,up,down) | attention-only is the biggest quality loss; MLP matters for knowledge-ish tasks |
| r | 16–64 | gains saturate fast; r=64 mostly overfits on <50k samples |
| α | 2r | with α=2r, effective LR per adapter path stays stable when r changes |
| LR | 1e-4 … 3e-4 | ~10× full-FT LR; too low looks like "LoRA doesn't work" |
| dropout | 0.05–0.1 | helps on small sets, hurts on 100k+ |
| epochs | 1–3 | third epoch usually overfits; watch eval loss, not train |
| init | B=0 | keeps the model identical at step 0 — a sanity check worth running |

Two traps: (1) merging a LoRA into a base that was later re-quantised loses the adapter's benefit — keep the adapter separate for quantised bases; (2) if you compare LoRA to full FT with the same LR, LoRA loses unfairly.

**中文要点**：LoRA 学 ΔW=BA 只训低秩旁路，可合并零延迟；调参顺序是 target modules（必须含 MLP）> r > α=2r > LR（约全参的 10 倍）> dropout；B 初始化为 0 保证起点等价，可做健全性检查。

---

### 11. How do you compute the SFT loss only over the answer, ignoring padding?

**Anchor answer**

Masks, two of them: prompt tokens get `-100` (ignored by cross-entropy), padding gets `-100` as well; and the loss must be a **mean over the remaining tokens**, not over all positions.

```python
labels = input_ids.clone()
labels[prompt_mask] = -100          # prompt + system + padding
loss = F.cross_entropy(
    logits[:, :-1].reshape(-1, V),  # shift
    labels[:, 1:].reshape(-1),
    ignore_index=-100,
    reduction="mean",               # mean over non-ignored tokens
)
```

Details that break silently:

1. **Shift alignment.** Logits at position $t$ predict token $t+1$; shift both, or the model learns an off-by-one and outputs garbage/echo.
2. **Do not mask the EOS token.** The single most common bug: mask it and the model never learns to stop, so it rambles until `max_new_tokens`. This is also the usual cause of "my fine-tune repeats itself".
3. **Truncation boundary.** If truncation cuts the answer, drop the sample rather than training on a truncated target.
4. **Packing.** Concatenating samples into one sequence requires resetting attention (block-diagonal mask) and not letting sample A's answer condition sample B's prompt. Also avoid BOS/EOS bleeding across boundaries.
5. **Length bias.** `reduction="mean"` weights each token equally, so long answers dominate the gradient; if that's wrong for your data, average per-sample then across samples.

**中文要点**：prompt 和 padding 都置 -100，loss 要对非忽略 token 取均值；必须做 shift 对齐；**绝不能 mask 掉 EOS**（否则模型学不会停、表现为重复）；packing 时要重置注意力掩码，别让上一样本的答案串到下一个。

---

### 12. Why GRPO in your project rather than PPO or DPO? How did you design the reward, including factuality and safety?

**Source**: https://outcomeschool.com/blog/group-relative-policy-optimization-grpo

**Anchor answer**

Because I had a **verifiable, cheap reward** and no good preference pairs. GRPO drops the critic: sample a group of G outputs per prompt, use the group's mean/std as the baseline, so advantage = $(r_i - \text{mean})/\text{std}$. That removes a model the size of the policy from memory, and it removes critic-fitting noise, which is where PPO usually goes wrong on long-form generation.

DPO was the alternative. It is cheaper but strictly offline: it cannot explore, and it optimises a fixed pair set. For tasks where the reward is checkable (format valid, answer matches a gold field, citation exists), GRPO's on-policy exploration is worth the extra compute.

Reward design, layered:

| Layer | Signal | Enforcement |
| :--- | :--- | :--- |
| Correctness | exact/rule match on the extracted answer | hard, weight 1.0 |
| Factuality | claim-level NLI against retrieved passages | graded 0/0.5/1, capped |
| Safety | policy classifier + blocklist for medical/legal/financial advice | **hard constraint in the reward, not a soft term** |
| Format | valid JSON / required sections | small bonus |
| Style/brevity | length band | clipped bonus, never unbounded |

Two lessons: filter degenerate groups (all rewards identical → zero advantage → wasted step), and validate the reward offline on frozen outputs before spending GPU-hours — a reward that can't separate good from bad by a clear margin will just amplify noise.

**中文要点**：选 GRPO 是因为有便宜可验证的奖励而没有好的偏好对——去掉 critic 省显存、免去 critic 拟合噪声；奖励分层：正确性硬权重、事实性按 claim 做 NLI 打折、安全作为硬约束而非可交换项、长度奖励必须截断；还要丢掉全同分的退化组。

---

### 13. Fine-tuning produces a model that repeats itself. What is happening and how do you debug it?

**Source**: https://outcomeschool.com/blog/continual-learning-in-llms

**Anchor answer**

Order the causes by frequency — the first one accounts for most cases.

| # | Cause | Check | Fix |
| :-: | :--- | :--- | :--- |
| 1 | **EOS label masked out** | print one training sample's labels; is the final token `-100`? | unmask EOS, it's the stop signal |
| 2 | Off-by-one / no shift | loss should start near ln(V) (~10.4) | fix logit/label alignment |
| 3 | Data duplication & low diversity | n-gram overlap within dataset, cluster count | dedupe, add diverse prompts |
| 4 | Over-training | eval loss rising while train falls | 1–2 epochs, early stop |
| 5 | Truncated targets | samples cut mid-answer | drop or extend window |
| 6 | Decoding params | temperature/top-p/repetition penalty | set `repetition_penalty` 1.05–1.1 as a bandage |
| 7 | Entropy collapse after RL | token entropy histogram | entropy bonus, lower LR |

Debug procedure: **generate from the training set itself**. If the model can't reproduce a training example, it's a label/alignment bug (1–3). If it reproduces it verbatim but loops on new prompts, it's overfitting/collapse (4–7). Then bisect with 100 samples for 50 steps — a 10-minute run tells you more than any theory.

Also check tokenizer consistency between training and serving: a different chat template can make the model repeat the prompt back.

**中文要点**：先按概率排查——**EOS 被 mask 掉**（学不会停，最常见）、shift 没对齐、数据重复/过窄、过训、目标被截断、解码参数、RL 后熵坍缩；诊断法是用训练集自身生成：复现不了是标签 bug，能复现但新输入循环是过拟合。

---

### 14. Walk through the full post-training recipe: pretraining, SFT, reward model, RL, domain adaptation.

**Source**: https://outcomeschool.com/blog/decoding-instructgpt

**Anchor answer**

| Stage | Data | Objective | What it buys | Typical scale |
| :--- | :--- | :--- | :--- | :--- |
| Pretraining | trillions of tokens, web+code+domain | next-token | world knowledge, fluency | 10¹²–10¹³ tokens |
| Continued pretraining | domain corpus (contracts, medical, logs) | next-token | vocabulary & facts of the domain | 10⁹–10¹⁰ tokens |
| SFT | instruction→answer pairs, curated | masked next-token on answer | format, following instructions, tool syntax | 10⁴–10⁵ high-quality |
| Reward model | pairwise preferences (human or AI judge) | Bradley-Terry | a scalar proxy for "better" | 10⁴–10⁵ pairs |
| RL (PPO/GRPO) | prompts + RM or verifier | maximise reward − β·KL | sharpens behaviour beyond imitation | 10³–10⁴ prompts |
| Rejection sampling / DPO | best-of-N filtered outputs | preference or SFT on winners | cheap iteration, no critic | 10⁴ |
| Domain adaptation | domain SFT + eval-driven RL | same | task accuracy | task-specific |

Sequencing rules that matter:

- **SFT sets the ceiling; RL only moves the model inside it.** If the behaviour is absent from SFT data, RL will not invent it — it will discover a shortcut instead.
- **Continued pretraining before SFT**, never after: doing it after erodes instruction-following.
- **Eval gates between every stage**, with the same held-out set, so you can attribute a regression to a stage.
- Keep a frozen copy of the previous production model to A/B against at each step; otherwise "better" is a vibe.

**中文要点**：预训练→领域继续预训练→SFT→奖励模型→RL（PPO/GRPO）→拒绝采样/DPO 迭代；关键规律是 **SFT 定上限、RL 只在上限内移动**（SFT 数据里没有的行为 RL 不会凭空发明，只会找捷径），且继续预训练必须在 SFT 之前。

---

### 15. In PPO, how do you prevent loss of generality and stop the model collapsing onto one high-reward answer?

**Anchor answer**

Collapse has one mechanism: the policy maximises the *reward model*, and the RM is a narrow proxy. Diversity dies first because the highest-reward region is usually a small, verbose, confident style.

Six levers, in the order I'd apply them:

| Lever | Setting / mechanism | Why |
| :--- | :--- | :--- |
| KL to reference | adaptive β targeting a KL band (e.g. 6–10) | anchors the policy to the SFT distribution; the main anti-collapse control |
| Entropy bonus | small coefficient, monitor token entropy | directly counteracts mode collapse |
| RM ensemble + margin | min over K RMs, or penalise RM disagreement | resists over-optimising one RM's artefacts |
| Reward clipping / normalisation | clip to ±z, length-normalise | stops "longest wins" |
| Diverse prompts | sample by intent cluster, not uniformly from logs | collapse often starts from a narrow prompt set |
| Early stopping on held-out eval | every N steps, both task metric and diversity metric | PPO degrades after the reward plateaus — the last 20% of training is net negative |

Diagnostics to watch: response length (a jump means RM hacking), distinct-n / self-BLEU across samples, entropy, and the KL curve. If KL pins at the β ceiling for many steps, the policy is fighting the anchor — either the reward is wrong or β is too high.

One more: never evaluate with the RM that trained the policy. Use a held-out judge plus a human spot check.

**中文要点**：坍缩根源是把窄代理奖励拉满——用自适应 KL 带、熵 bonus、奖励模型集成取 min、奖励裁剪与长度归一、按意图簇采 prompt、按"奖励平台期"早停；诊断看回复长度突增、distinct-n 下降、熵；评估绝不能用训练用的那个 RM。

---

### 16. How does BM25 work, and why add dense retrieval on top?

**Source**: https://outcomeschool.com/blog/how-does-hybrid-search-work

**Anchor answer**

BM25 scores a document as a sum over query terms:

$$\text{score}(q,d)=\sum_{t\in q}\text{IDF}(t)\cdot\frac{f(t,d)\,(k_1+1)}{f(t,d)+k_1\left(1-b+b\frac{|d|}{\text{avgdl}}\right)}$$

Three ideas in one formula: **IDF** (rare terms matter more), **term-frequency saturation** via $k_1$ (the 10th occurrence adds little — this is the fix over raw TF), and **length normalisation** via $b$ (long docs shouldn't win by volume). Typical $k_1\approx1.2$–2.0, $b\approx0.75$.

Why add dense retrieval:

| Query type | BM25 | Dense |
| :--- | :--- | :--- |
| Exact SKU / error code / name | strong | weak, may miss |
| Paraphrase, synonym ("cheap hotel near office") | fails | strong |
| Long natural-language question | weak (term dilution) | strong |
| Rare entity / number | strong | often wrong |
| Out-of-domain vocabulary | strong (no training needed) | degrades |

They fail in **complementary** directions, which is why hybrid plus RRF beats either. Dense also gives you a reusable embedding for clustering/dedup, and BM25 gives you an interpretable score you can threshold on — useful when you need "no answer" detection rather than a top-1 guess.

One practical note: Chinese BM25 quality is dominated by the analyser, not by $k_1$; a synonym/alias dictionary for product terms moves recall more than tuning.

**中文要点**：BM25 = IDF × TF 饱和(k1) × 长度归一(b)，解决原始 TF 的两个偏差；它强在精确词/编号/罕见实体、弱在改写同义，dense 正好相反，所以 RRF 融合优于任一；中文场景分词与同义词表比调 k1 收益大。

---

### 17. Long reviews are summarised before indexing. How do you make sure the summary does not lose information, and how do multi-level summaries plus source quotes fit?

**Anchor answer**

Do not replace the source with the summary — **route with summaries, answer with quotes**.

Architecture, three levels:

| Level | Content | Used for |
| :--- | :--- | :--- |
| Chunk | original review span, verbatim, with `review_id`, offset, timestamp | citation, exact wording |
| Aspect summary | per-review structured extraction: aspect, sentiment, concrete claim ("wait 40 min", "fish not fresh") | filtering and matching |
| Merchant summary | aggregated per aspect with counts and time window ("18 of 212 reviews mention slow service in Q3") | overview questions, routing |

Guarantees that keep information:

1. **Every summary claim carries a source list.** If a claim has no `review_id`, it is dropped — this alone removes most hallucinated summary content.
2. **Faithfulness check** — NLI/claim-level verification of summary against source; failed claims get regenerated or the summary is marked `unverified` and excluded from retrieval.
3. **Keep numbers and entities verbatim.** Summaries lose figures; extract them as fields (`wait_minutes: 40`) instead of prose.
4. **Never summarise the only copy.** The full text stays indexed; summaries are additional keys.
5. **Freshness** — summary has a window and a computed timestamp; a 2-year-old aggregate is misleading for a restaurant under new management.

Failure mode to expect: aggregation hides distribution ("mixed reviews" — from what mix? 60/40 or 95/5?). Always expose counts, not just polarity.

**中文要点**：摘要只做路由、答案必须回落到原文引用；每条摘要声明带 review_id 列表，无来源即丢弃，并做 NLI 忠实性校验；数字/实体抽成字段而非散文；原文永远保留可检索，摘要必须带时间窗口与样本计数。

---

### 18. p95 latency jumps in production. How do you debug it?

**Anchor answer**

p95 moving while p50 is flat means the **tail is being created somewhere**, not that everything got slower. That singles out queueing and heterogeneity.

Check in this order:

| Check | Signature |
| :--- | :--- |
| Batching / packing | p50 flat, p95 up: a few long sequences occupy the batch, everyone else waits (head-of-line blocking). Look at max input/output tokens per batch |
| Input length drift | p95 correlates with input length percentiles — new document type or a client sending whole files |
| Cache misses | p95 up with a new corpus/tenant; prefix-cache hit rate dropped, so prefill cost spiked |
| Retries | p95 ≈ 2–3× p50 with a step pattern: retry storm amplifying load |
| One slow shard / GPU | p95 concentrated in a subset of instances — check per-instance histograms, not the aggregate |
| GC / allocator | periodic spikes aligned with GC or CUDA allocator fragmentation |
| Timeout boundary | a spike exactly at the client timeout value means clients give up and retry |

Method: **segment the histogram** by input length bucket, tenant, instance, and cache-hit. A p95 that is "the same requests being slower" and a p95 that is "a new subset of slow requests" have completely different fixes, and the aggregate cannot tell them apart. Then reproduce with the actual slow trace, not a synthetic median request.

**中文要点**：p50 平而 p95 涨 = 尾部被某处制造出来——先按输入长度/租户/实例/缓存命中去切分直方图；常见根因是长序列造成批内队头阻塞、输入变长、前缀缓存命中下降、重试风暴、单实例慢、客户端超时值处的尖峰。

---

### 19. RAG recall is too low. What do you do, and how do you tell a retrieval problem from a generation problem?

**Source**: https://outcomeschool.com/blog/llm-evaluation

**Anchor answer**

**Separate the two first** — everything else is wasted effort until you do.

Test: take 50 failing queries, hand the model the **gold document**.
- Answers correctly with gold doc → **retrieval** problem.
- Still wrong → **generation** problem (or the gold doc doesn't actually contain the answer).
- Refuses → prompt/context-assembly problem.

If retrieval:

| Layer | Checks | Fixes |
| :--- | :--- | :--- |
| Query | is the query self-contained? pronoun/coreference? | query rewrite, HyDE, multi-query |
| Chunking | chunk too small (answer split) or too large (diluted) | heading-aware chunking, overlap, keep parent for context |
| Embedding | domain mismatch, wrong language, stale model | fine-tune embedder on in-domain pairs, or switch |
| Filter | over-strict metadata/time filter | relax, measure recall with/without filter |
| Index | missing docs, alias pointing at old index | coverage audit, staleness check |
| Fusion | one retriever dominating RRF | tune weights, check per-retriever recall@k |

Measure **recall@k at each stage** (first stage, after fusion, after rerank) so you know which stage loses the document. Most "low recall" reports are actually a filter or a coverage gap, not an embedding-quality problem — and no amount of reranking fixes a document that was never retrieved.

**中文要点**：先用"把 gold 文档直接喂给模型"这一步分诊（对→检索问题，错→生成问题）；然后逐级测 recall@k（初检/融合/重排后）；绝大多数"召回低"是元数据过滤过严或索引覆盖缺口，而不是 embedding 不行。

---

### 20. What is your understanding of function calling, and how do you train tool-use ability? What goes in the training set and where does the data come from?

**Source**: https://outcomeschool.com/blog/how-does-function-calling-work-in-llms

**Anchor answer**

Function calling is a **protocol**: tools are declared with JSON schema, the model emits `{name, arguments}`, the runtime executes, the result returns as a message. The model never runs code. Training it means teaching three separate skills: *decide whether* a tool is needed, *select which*, and *emit valid, well-typed arguments*.

Training set composition (the ratios matter more than the volume):

| Slice | Source | Share |
| :--- | :--- | :--- |
| Single-tool, clean | synthetic from schemas + self-instruct | 40% |
| Multi-tool disambiguation | near-duplicate tool pairs, same query, one correct by policy | 20% |
| Multi-turn with tool results | replay from real logs, including error results | 15% |
| No-tool-needed | chit-chat, answerable from context | 10% |
| Error recovery | call fails with a typed error → repair args | 10% |
| Adversarial | prompt injection asking for a forbidden tool, ambiguous request | 5% |

Where data comes from: (1) **synthesise from the schema** — enumerate intents per tool with an LLM, verify by executing against a mock; (2) **real traffic** — mine tool-call traces, keep the successes, and repair the failures with a stronger model; (3) **execution feedback as the label** — if the call validates and the mock returns a plausible result, it is a positive.

Non-negotiable: every synthetic call must be **executed and validated**, otherwise you train fluent-but-wrong arguments, which is worse than no tool use.

**中文要点**：函数调用是协议不是执行——要分别训"要不要用工具/用哪个/参数对不对"；数据配比约为单工具 40%、易混工具 20%、多轮工具结果 15%、无需工具 10%、错误恢复 10%、对抗 5%；来源是 schema 合成+真实 trace+执行校验，合成样本必须跑通验证。

---

### 21. Two tools can satisfy one query. How do you design the training data for that?

**Anchor answer**

Overlapping tools are a **specification** problem first, data second. Order:

1. **Fix the spec if you can.** Make them non-overlapping by argument shape ("`search_products` takes keywords; `lookup_sku` takes an exact ID"). Half of these cases disappear here.
2. **If genuinely ambiguous, define a policy** — cheapest, safest, most specific, or "ask the user". Write it down; the model cannot learn an unwritten tiebreak.
3. **Then build data that teaches the policy**, not the surface form.

Data design:

| Slice | Purpose |
| :--- | :--- |
| Same query, both tools available, one labelled correct **by policy** | the core decision |
| Same query, only one tool available | teaches that availability matters, not just wording |
| Same query, **tool order shuffled** across examples | kills position bias — the top-listed tool otherwise wins |
| Query where the other tool is correct (near-miss pairs) | forces the distinguishing feature, e.g. "exact model number" |
| Genuine tie → `clarify` action | teaches abstention instead of a coin flip |

Consistency checks: measure **flip rate** (same query, shuffled tool order, different answer) — above ~5% means position bias is still there. And measure per-tool precision/recall, not just overall accuracy, because the aggregate hides a systematic bias toward one tool.

**中文要点**：先改规格让工具不重叠，再定一条明文策略（最便宜/最安全/最具体/反问），然后才造数据；关键是**同一 query 打乱工具顺序**消除位置偏好，用近似对逼出区分特征，并测"翻转率"和每工具 precision/recall。

---

### 22. How does a lead agent dynamically pick a sub-agent, and should sub-agents run serially or in parallel?

**Source**: https://outcomeschool.com/blog/multi-agent-systems

**Anchor answer**

Selection is retrieval over **capability cards**, not a giant prompt listing all agents:

| Step | Mechanism |
| :--- | :--- |
| Describe | each sub-agent exposes `name`, `when_to_use`, input/output contract, cost, latency |
| Retrieve | embed the task, retrieve top-k cards (k≈3–5), or let the lead call `list_agents` |
| Decide | lead picks one, or decomposes into a DAG of subtasks |
| Contract | hand off with an explicit, serialisable task object — not the whole conversation |
| Verify | lead checks the returned artifact against the acceptance criteria, retries or escalates |

Serial vs parallel: **parallel when subtasks are independent and read-only** (search three sources, analyse three files) — you get wall-clock wins and isolation of context. **Serial when there is a dependency or a shared mutable state** (write to the same database, each step consumes the previous output). Practical rule: if two sub-agents can both write, run them serially or with a lock, because concurrent writers produce lost updates that are painful to debug.

Costs people forget: each parallel branch multiplies token spend and can storm a rate limit; and merging parallel outputs needs a synthesis step, which is itself an LLM call with its own failure mode (dropping one branch's finding).

**中文要点**：选择靠"能力卡"检索（name/when_to_use/契约/成本），不是把所有 agent 塞进 prompt；独立只读任务并行、有依赖或共享写入则串行；交接传可序列化的任务对象而不是整段对话，结果要按验收标准校验。

---

### 23. Why should agent capabilities be disclosed progressively?

**Anchor answer**

Because tool-selection accuracy is a function of the number of candidates, and context is a budget.

| Effect | Mechanism | Symptom |
| :--- | :--- | :--- |
| Selection degradation | N similar tools compete in one softmax; the discriminative signal dilutes | wrong tool among 40, right tool among 8 |
| Context tax | every tool schema costs tokens *every turn*, and pushes the actual task into the middle | instruction drift, higher cost |
| Permission surface | a tool that is loaded is a tool that can be called | over-broad blast radius |
| Prompt injection | a large tool list gives an attacker a larger target | unauthorised calls |

So disclosure becomes a **staged policy**: a small always-on core (search, read, ask-user), plus a `discover_capabilities(query)` call that returns a handful of relevant tools for the current subtask, unloaded when done. Skills work the same way — a 2k-token procedure enters context only when its trigger matches.

The counterintuitive part: progressive disclosure makes the *system* simpler to reason about while making the *prompt* smaller. It also gives you a natural place to enforce permissions — the discovery step is where you check the caller's scope, so an unauthorised tool is never even offered.

Cost: one extra model call (discovery), and the risk that the model fails to discover a tool it needs. Mitigate with a fallback "show all" path and by including a few always-visible canonical tools.

**中文要点**：工具越多选错越多（相似工具在同一 softmax 里竞争）、每个 schema 每轮都吃 context、且加载即拥有权限；所以分阶段披露——常驻核心 + `discover_capabilities` 按需返回少量工具，权限检查放在发现环节，工具在使用后卸载。

---

### 24. What are OpenClaw and Manus, why did they take off, and how do they differ?

**Source**: https://outcomeschool.com/blog/harness-engineering-in-ai

**Anchor answer**

Both are **general-purpose agent harnesses** — the model is a commodity, and the product is the loop around it: tools, sandbox, file system, long-running task state, and a UI that lets you watch and interrupt.

What made them land, in order of importance:

| Factor | Why it mattered |
| :--- | :--- |
| Long-horizon tasks | "go do this whole thing" instead of chat turn-by-turn — the value proposition is delegation, not answers |
| Real environment | a VM with a browser, shell and files, so tools are general rather than pre-integrated per API |
| Durability | task state survives hours, failures, and restarts |
| Observability | you can watch the plan, the actions, and the artifacts; trust needs visibility |
| Distribution | one is messenger-first/self-hosted, the other a hosted product with a shareable result |

Differences, practically: **OpenClaw** is an open-source, self-hosted personal assistant that plugs into messaging apps and runs on your own machine, so the emphasis is local execution, extensibility and community skills; **Manus** is a hosted cloud-sandbox agent with a productised plan-execute-loop, credit-based usage, and a shareable artifact page, so the emphasis is zero-setup and a controlled environment.

The shared lesson for building one: the hard parts are sandbox isolation, credential handling, and resumability — not the prompt. Anyone can demo a browser agent; making it survive a 40-minute run without leaking a token is the engineering.

**中文要点**：两者都是"通用 agent harness"——模型是商品，产品是循环：长任务、真实 VM 环境、状态持久化、可观测与可中断。OpenClaw 偏开源自托管+消息入口，Manus 偏云端沙箱+托管产品化；真正的难点是沙箱隔离、凭据管理和可恢复性。

---

### 25. Beyond tool descriptions, how do you stop an agent calling the wrong tool?

**Anchor answer**

Descriptions are one of eight controls. Ordered by leverage:

| Control | Mechanism | When it helps |
| :--- | :--- | :--- |
| **Fewer tools visible** | retrieval/gating per stage (see progressive disclosure) | the single biggest win; 40 tools → 6 |
| **Rename & reshape** | distinct verbs, enums instead of free text | overlapping names cause overlap errors |
| **Argument constraints** | enum / regex / required schema, validated before execution | blocks hallucinated IDs and free-form accounts |
| **Few-shot routing examples** | 5–10 `(query → tool, args)` pairs in the prompt | cheap, large effect for ambiguous intents |
| **Negative examples** | "do NOT use X for Y" | teaches boundaries descriptions omit |
| **Dry-run validator** | schema + DB existence check before side effect | converts a wrong call into a retryable error |
| **Post-call verification** | check the result against the intent (right entity? plausible range?) | catches semantically wrong-but-valid calls |
| **Ask instead of guess** | a `clarify` tool + policy for low confidence | the correct move for genuinely ambiguous requests |

The one that is usually missing is **post-call verification**. Most systems validate the call *shape* and stop there, so `refund(order_id=...)` on the wrong order passes every check. A cheap assertion — does this order belong to this user, is the amount within the expected band — catches the class of error descriptions can never prevent.

**中文要点**：描述只是八道控制之一，杠杆最大的是"少给工具"（按阶段检索 40→6）；其次是改名字/枚举化参数、few-shot 路由示例、负例、执行前 dry-run 校验、**调用后校验（这笔退款是不是这个用户的单）**，以及低置信度时反问。

---

### 26. RAG versus reading the file in a long context: when is each right? How does that compare with a coding agent grepping the repo?

**Source**: https://outcomeschool.com/blog/lost-in-the-middle-problem-in-llms and https://outcomeschool.com/blog/how-does-claude-code-work

**Anchor answer**

Three different bets on "how do I find the right bytes".

| Approach | Wins when | Loses when |
| :--- | :--- | :--- |
| **RAG** | large, heterogeneous, frequently changing corpus; many users; need citations and permissions per chunk | the answer needs global synthesis across the whole corpus; chunks split the reasoning unit |
| **Long context** | one or a few documents, question needs whole-document reasoning (summarise, compare, find contradictions) | corpus is bigger than the window, cost/latency per query, and "lost in the middle" makes deep recall unreliable |
| **Grep / agentic search** | exact symbols, identifiers, error strings; the source of truth is a tree you can walk | semantic questions ("where is retry logic") with no lexical anchor; huge repos where every grep is slow |

The coding-agent case is instructive: grep wins over embeddings because code has **exact, unique identifiers** and the repo is the authority — you don't need semantics, you need precision plus the ability to read the surrounding file. The agent then *verifies* by reading and running tests, which RAG cannot do.

Decision rule: if the answer requires reading a *whole* object, load it; if it requires finding *one needle among millions*, retrieve; if it requires *exact string matching plus iteration*, let the agent search. In practice the best systems mix: retrieve to locate, then read the full document once located.

**中文要点**：RAG 适合大而杂、常变、要多租户权限的语料；长上下文适合单篇/少量文档的整体推理；grep 适合代码这种有唯一标识符且可读文件验证的场景。判据是"要读整篇就整篇加载，要在千万里找一根针就检索，要精确字符串+迭代就让 agent 自己搜"。

---

### 27. A QA system goes live: break down the end-to-end time from query to answer and say where the budget goes.

**Anchor answer**

Take a 1.2 s p50 target for a grounded, ~200-token answer:

| Segment | Typical | Notes |
| :--- | ---: | :--- |
| Gateway/auth/queue | 5–20 ms | queue wait explodes under load; measure separately |
| Query rewrite (optional) | 30–80 ms | skip for short queries; it is a whole model call |
| Embed query | 5–15 ms | batch with cache; a small model is enough |
| ANN search | 10–30 ms | grows with `num_candidates` and shard fan-out |
| Rerank 50 → 8 | 60–150 ms | cross-encoder is usually the largest non-LLM cost |
| Prompt assembly | 1–3 ms | tokenisation of long context is not free |
| **LLM TTFT** | 150–400 ms | prefill of a 6k-token prompt on a loaded GPU |
| **Decode 200 tokens** | 600–1200 ms | ~5–10 ms/token at moderate batch; this is the floor |
| Post-process/citation check | 20–60 ms | can be parallelised with the last chunk |

Where the budget actually goes: **decode dominates**, then prefill, then rerank. So the levers are: shorten the answer (max_tokens, tighter prompt), shrink the context (fewer, better chunks — this cuts prefill *and* improves quality), and stream so TTFT is what the user perceives.

Anti-pattern: optimising the 15 ms embedding call while 900 ms of decode sits untouched. Measure per-stage p50 **and p95** — the p95 is usually queueing or a long-tail input, not any single stage being slow.

**中文要点**：预算里 decode 占大头（200 token ≈ 0.6~1.2s），其次 prefill，再次 rerank；所以杠杆是缩短答案、减少上下文块、用流式让用户感知的是 TTFT；别去优化 15ms 的 embedding 而放着 900ms 的解码不动。

---

### 28. Design a summarisation plus retrieval system over merchant reviews (hierarchical summaries).

**Anchor answer**

Goal: answer both "is this place good for a business dinner?" (needs aggregation) and "did someone mention the private room?" (needs a specific review).

| Layer | Unit | Built how | Serves |
| :--- | :--- | :--- | :--- |
| L0 | review (verbatim) | ingest, dedupe, language detect | citation, exact quotes |
| L1 | review → aspects | extract `{aspect, sentiment, claim, evidence_span, date}` | filtering, counting |
| L2 | merchant × aspect | aggregate over a time window, with counts and top quotes | "how's the service" |
| L3 | merchant overall | roll-up with recency weighting, plus "what changed recently" | cold-start summary, ranking features |

Mechanics that matter:

1. **Aspect taxonomy must be fixed and small** (food, service, wait, price, hygiene, ambience, delivery) or L2 aggregation becomes noise; allow a free-text "other" bucket and mine it periodically.
2. **Counts, not just polarity** — "18/212 mention slow service in Q3" is actionable; "mixed" is not.
3. **Recency weighting + change detection** — compare last 90 days against the prior year so a renovation or a new manager shows up.
4. **Drill-down must always be possible**: every L2 statement links to the L1 rows and L0 texts. If a user challenges the summary, you can show the reviews.
5. **Incremental** — recompute L2 only for merchants with new reviews, on a schedule; recomputing everything nightly does not scale past a few hundred thousand merchants.

Failure mode: summarising summaries until the numbers drift. Aggregate from L1 counts with code, and use the LLM only for phrasing.

**中文要点**：四层（原评论→方面抽取→商户×方面聚合→商户总览）；方面体系要固定且小，聚合必须给计数与时间窗而不是"褒贬不一"；支持逐层下钻到原文；数字用代码从 L1 计数算，LLM 只负责措辞，避免"摘要的摘要"漂移。

---

### 29. Design a customer-support agent for in-store or delivery, and name the core metrics.

**Anchor answer**

Scope first: **which intents are automatable**. Refund status, order modification, delivery ETA, coupon rules — automatable. Compensation decisions, food-safety complaints, legal threats — escalate immediately, always.

Architecture:

| Layer | Component |
| :--- | :--- |
| Entry | intent classifier + auth (order id, phone) before any data access |
| Policy | deterministic rules engine: eligibility, caps, SLA — the LLM proposes, the engine decides |
| Tools | `get_order`, `get_delivery_status`, `apply_coupon`, `create_refund_request`, `escalate` |
| Knowledge | RAG over policy docs, per-city rules, promotion terms |
| Guardrail | amount caps, banned promises, PII redaction, no medical/legal advice |
| Handoff | warm transfer with a structured summary + full transcript, never "sorry, transferring" |

Metrics, and what they actually tell you:

- **Containment / deflection rate** — resolved without human. Watch for the trap: agents that "contain" by refusing to help inflate this. Pair with CSAT.
- **Task success rate** (order actually changed) — the honest one.
- **Wrong-answer / re-contact rate within 24h** — the best proxy for silent failure.
- **Escalation precision** — of escalations, how many needed a human; too high means the agent gives up early.
- **AHT + cost per contact**, and **p95 first-response time**.
- **CSAT on automated sessions vs human baseline** — never compare only against other bots.

Design note: measure on a **stratified sample** per intent, otherwise high-volume easy intents hide failures in the rare expensive ones.

**中文要点**：先划清可自动化的意图（退款进度/改单/ETA/优惠规则可，赔付决定/食安投诉/法律威胁必须转人工）；架构是意图+鉴权→规则引擎定政策（LLM 只提议）→工具→知识库→护栏→带摘要的暖转接；核心指标是任务成功率、24h 重复联系率、升级精确率、分层 CSAT，警惕用"拒答"刷 containment。

---

### 30. Reverse nodes in k-group.

**Reference implementation**

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next


def reverse_k_group(head, k):
    if k <= 1 or head is None:
        return head

    dummy = ListNode(0, head)
    group_prev = dummy

    while True:
        # 1) find the k-th node of this group
        kth = group_prev
        for _ in range(k):
            kth = kth.next
            if kth is None:
                return dummy.next      # fewer than k left: leave as is

        group_next = kth.next          # node after this group

        # 2) reverse the group in place
        prev, cur = group_next, group_prev.next
        while cur is not group_next:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt

        # 3) splice: old head becomes the tail
        new_tail = group_prev.next
        group_prev.next = kth
        group_prev = new_tail
```

**Key points**

- Advance `k` nodes **before** reversing: the "leftover < k stays untouched" rule is the whole difficulty, and checking mid-reversal means you must undo work.
- Initialising `prev = group_next` makes the first node point at the next group automatically — no separate fix-up.
- O(n) time, O(1) extra space; each node is visited at most twice.
- Edge cases: `k=1`, `k > len`, empty list, `k == len` (reverse everything).
- Recursive version is shorter but O(n/k) stack frames; for a 10⁵-node list that is a real risk.

**中文要点**：先走 k 步确认够不够一整组，不够就原样返回；`prev` 初始化为 `group_next` 让组内首节点自动指到下一组；组内反转后把旧头当新尾接回。O(n) 时间 O(1) 空间，别用递归。

---

### 31. Level-order traversal of a binary tree, and zigzag level order.

**Reference implementation**

```python
from collections import deque


def level_order(root):
    if not root:
        return []
    res, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):        # level size snapshot
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(level)
    return res


def zigzag_level_order(root):
    if not root:
        return []
    res, q, left_to_right = [], deque([root]), True
    while q:
        level = deque()
        for _ in range(len(q)):
            node = q.popleft()
            if left_to_right:
                level.append(node.val)      # push back
            else:
                level.appendleft(node.val)  # push front
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(list(level))
        left_to_right = not left_to_right
    return res
```

**Key points**

- Zigzag: use a deque per level and `appendleft` instead of reversing afterwards — O(n) instead of O(n) with an extra pass, and it avoids the off-by-one of reversing on odd indices.
- Alternative: always append left-to-right, then `level[::-1]` when the flag is false. Simpler, one extra copy per level — acceptable in an interview, mention the trade-off.
- `len(q)` snapshot is what separates levels; forgetting it flattens the tree into one list.
- Both are O(n) time, O(w) space (w = max width); a complete tree is n/2 wide, so space is not O(log n).

**中文要点**：用 `len(q)` 快照分层；zigzag 用双端队列按方向 `append`/`appendleft`，比事后反转更省一次遍历；两种写法都是 O(n) 时间、O(w) 空间。

---

### 32. Implement multi-head attention or a MoE by hand (the signature coding round for algorithm roles).

**Source**: https://outcomeschool.com/blog/multi-head-attention-in-transformers

**Reference implementation**

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, causal: bool = True):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)                  # [B,T,D]
        # split heads: [B, T, h, d_k] -> [B, h, T, d_k]
        q, k, v = (t.view(B, T, self.h, self.d_k).transpose(1, 2)
                   for t in (q, k, v))

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)  # [B,h,T,T]
        if causal:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = self.drop(F.softmax(scores, dim=-1))
        y = attn @ v                                            # [B,h,T,d_k]
        y = y.transpose(1, 2).contiguous().view(B, T, D)          # concat heads
        return self.out(y), attn
```

**Key points**

- `d_k = d_model / h`, and the `1/sqrt(d_k)` scaling is on the **per-head** dimension, not `d_model` — the classic slip.
- Split heads with `view → transpose`; you need `.contiguous()` before the final `view` when re-merging.
- Causal mask must be applied **before** softmax with `-inf` (or a large negative), broadcastable to `[B,h,T,T]`.
- Cross-attention is the same code with `q` from the decoder and `k,v` from memory; T need not match.
- Complexity O(B·h·T²·d_k) — mention FlashAttention and why production never materialises `attn`.

**中文要点**：缩放因子用的是每个头的 `d_k` 而不是 `d_model`；拆头用 `view+transpose`、最后 `contiguous().view` 合并；因果 mask 要在 softmax 前填 `-inf`；复杂度 O(B·h·T²·d_k)，生产上不会真的物化 attn 矩阵。

---

### 33. Tell me about a time you actually moved a model's online results: what was the baseline and how was the metric computed?

**Anchor answer**

Answer with the measurement discipline first, the model second.

**Baseline**: the current production model, not a null. State the A/B split (e.g. 50/50 by user id hash), the run duration, and the guardrail metrics alongside the goal metric — a CTR lift that costs 3 points of return rate is not a win.

**Metric computation, precisely**:

| Term | Definition I would state |
| :--- | :--- |
| Exposure | one session where the component was rendered; dedupe per request id |
| Numerator | click events attributed to that exposure within the session window |
| Denominator | exposures, not page views — otherwise UI changes move the metric |
| Aggregation | per-user ratio then average (avoids heavy users dominating), with a bootstrap CI |
| Guardrails | latency, zero-result rate, complaint rate, GMV |

Then the honest part: **what moved it**. Usually not the model — it was one feature with clear signal (better query understanding for long-tail queries; a rerank feature that recovered exact SKU matches), plus fixing a logging gap that had been undercounting exposures.

Two things worth saying out loud: the lift decayed by ~40% in week three (novelty), and the segment that gained was new users, not power users — which changed the next quarter's roadmap more than the headline number did.

**中文要点**：先说测量纪律——基线是现网模型而非空对照，指标=点击/曝光（按用户先算比率再平均，带 bootstrap CI），并同时报护栏指标；诚实的部分往往是"动的不是模型"，且新奇效应会衰减、增益集中在新客而非重度用户。

---

### 34. How do you find the last one or two percent of hallucinations, and what can you honestly claim without an online A/B?

**Anchor answer**

Finding the tail is a **sampling and taxonomy** problem, not a metric problem. Averaged accuracy hides it by construction.

| Step | Method |
| :--- | :--- |
| Oversample failures | stratified sample by intent × length × retrieval-score-low; low retrieval score is the strongest single predictor |
| Judge + human | LLM judge screens 100%, humans audit the flagged 5% and a random 2% control |
| Taxonomy | entity, numeric, temporal, attribution (right fact, wrong source), fabricated citation, over-claim |
| Attribution | for each failure, was the fact in the retrieved context? In the corpus at all? → retrieval vs generation vs knowledge gap |
| Fix by class | missing knowledge → coverage/ingest; retrieved-but-ignored → prompt/context order; invented → abstention training |

Claiming without an A/B — be explicit about the limits:

1. **Say "on a 300-sample stratified audit with 95% CI"**, never "hallucination rate is 2%".
2. **Report the abstention trade-off**: hallucination down often means refusal up; a single number hides that.
3. **Counterfactual replay** on logged traffic gives you a defensible offline estimate — same inputs, both models, judge both, report disagreement.
4. **Never claim business impact** from offline numbers. You can claim "fewer unsupported claims on audited samples", not "higher CSAT".

**中文要点**：尾部问题靠采样与分类学而非平均值——按意图×长度×低检索分分层过采样，judge 全筛 + 人工审 5% 加 2% 随机对照，按错误类型归因（知识缺口/检索到但没用/编造）；没有 A/B 时只能报"300 条分层审计、95% CI"并同时报拒答率，不能声称业务影响。

---

## 附：这 34 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 模型底子（公式级） | 2（注意力公式）、5（MoE 手写）、32（MHA 手写）、3（MLA） |
| 训练/后训练 | 6（bf16）、8（ZeRO）、10（LoRA）、11（SFT mask）、12（GRPO）、14（全流程） |
| 排障能力 | 13（重复输出）、15（PPO 坍缩）、18（p95） |
| 检索与评测 | 16（BM25）、17（摘要索引）、19（召回低分诊）、28（分层摘要） |
| Agent | 20（工具数据）、21（两工具歧义）、22（子 agent）、23（渐进披露）、25（选错工具） |
| 系统设计 | 26（RAG vs 长上下文）、27（时延预算）、29（客服 agent） |
| 现场编码 | 30（k 组反转）、31（层序/锯齿）、32（MHA） |
| 业务诚实度 | 33（线上增益归因）、34（尾部幻觉与无 A/B 主张） |
