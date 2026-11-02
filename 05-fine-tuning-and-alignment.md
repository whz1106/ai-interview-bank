# 05 · Fine-Tuning, Post-Training and Alignment

> 12 题。出处：`README.en.md` § Fine-Tuning, Post-Training and Alignment。

---

### 1. Walk me through RLHF end to end: reward model, policy optimisation, KL penalty.

**Source**: https://outcomeschool.com/blog/reinforcement-learning-from-human-feedback-rlhf

**Anchor answer**

Three stages:

1. **SFT** — fine-tune the base model on demonstrations. This alone already fixes most "instruction following".
2. **Reward model (RM)** — collect **pairwise preferences** `(prompt, chosen, rejected)`; train a model with a Bradley-Terry loss so that `r(chosen) > r(rejected)`. Note: you cannot train a good RM on a few thousand pairs — you need tens of thousands.
3. **RL against the RM** — PPO optimises `E[r(x,y)] − β·KL(π_θ ‖ π_ref)`.
   - The **KL term is not decoration**: it keeps the policy near the SFT model, preventing the "reward-hacking, weird-output" collapse.
   - β controls the tradeoff: too small → policy exploits RM holes; too large → nothing changes.

Operationally heavy: 4 models in memory (policy, reference, reward, value). That's exactly why DPO took over for many teams.

**中文要点**：SFT → 偏好对训奖励模型（Bradley-Terry）→ PPO 带 KL 约束优化；KL 是防奖励黑客的刹车，β 太小学坏、太大没变化；工程上要同时挂 4 个模型，重。

---

### 2. What is DPO and why did it displace PPO-based RLHF at many labs? When is online RL still better?

**Asked at**: HuggingFace · Scale AI · Tencent · Alibaba · Meituan
**Source**: https://outcomeschool.com/blog/direct-preference-optimization-dpo

**Anchor answer**

DPO reparameterises the same KL-constrained objective so that **the reward model disappears**: the implicit reward is `β·log(π_θ(y|x)/π_ref(y|x))`, and the loss becomes a simple classification over preference pairs — increase the log-prob of `chosen` and decrease `rejected`, relative to the reference model.

Why it displaced PPO:

| | PPO-RLHF | DPO |
| :--- | :--- | :--- |
| Models in memory | 4 (policy, ref, RM, value) | 2 (policy, ref) |
| Training loop | generate → score → update, unstable | supervised-style, stable |
| Data | needs online rollout | fixed offline pairs |
| Tuning knobs | many (clip, GAE, β, …) | mainly β |

When **online RL is still better**: you need to explore beyond the offline distribution — long-horizon agents, tool-use policies, verifiable-reward tasks (math/code), or when your preference data is too small/biased to cover the action space. Also when you want to optimise a *process* reward rather than a final answer.

**中文要点**：DPO 把奖励模型消掉（隐式奖励 = β·log(π/π_ref)），只需 2 个模型、离线偏好对、稳定得多；但需要探索出离线下分布时（长任务 agent、可验证奖励、工具策略）仍得用在线 RL。

---

### 3. Explain GRPO and why dropping the value network matters at scale.

**Asked at**: DeepSeek · ByteDance · Tencent · Meituan · Alibaba
**Source**: https://outcomeschool.com/blog/group-relative-policy-optimization-grpo

**Anchor answer**

GRPO removes the **critic/value network** entirely. Instead of a learned baseline, for each prompt it samples a **group of G completions**, computes their rewards, and uses the **group mean as the baseline**:

`A_i = (r_i − mean(r_group)) / std(r_group)`

Why this matters at scale:

1. **Memory**: the value network is roughly as large as the policy — dropping it removes ~⅓–½ of training memory, which is usually the binding constraint.
2. **Stability & simplicity**: no separate critic to train, no GAE, no value-loss divergence. Fewer failure modes.
3. **Pairs naturally with verifiable rewards** — if you can score a group of samples cheaply (unit tests pass, math answer correct), you don't need a learned critic at all.

Cost: you must generate **G samples per prompt** (typically 8–64), so rollout compute goes up. That tradeoff — more generation, less model — is exactly right for reasoning/RLVR workloads.

**中文要点**：GRPO 用"同 prompt 一组采样的均值/标准差"当 baseline，直接砍掉 value network——省 1/3~1/2 训练显存、少一套失败模式，代价是每个 prompt 要采样 8~64 条。

---

### 4. Explain the LoRA decomposition mathematically. Why does it work, and how do you choose the rank r?

**Asked at**: Meituan · Tencent · JD.com
**Source**: https://outcomeschool.com/blog/lora-low-rank-adaptation-of-llms

**Anchor answer**

Freeze `W₀ ∈ ℝ^{d×k}`; add a low-rank update:

`W = W₀ + ΔW = W₀ + (α/r)·B·A`, with `B ∈ ℝ^{d×r}`, `A ∈ ℝ^{r×k}`, `A ~ N(0,σ²)`, `B = 0` at init.

- Trainable params: `r(d+k)` vs `dk`. For `d=k=4096, r=8`: 65k vs 16.7M → **~1/250**.
- `α/r` **scale** keeps the effective update magnitude roughly independent of `r`, so you can change `r` without re-tuning learning rate.
- Inference: merge `ΔW` into `W₀` → **zero added latency** (unlike adapters/prefix methods).

Why it works: fine-tuning updates empirically have low **intrinsic rank** — the useful adaptation lives in a small subspace. LoRA assumes you can find it.

Choosing `r`: too small → underfits the domain (can't represent the adaptation); too large → approaches full fine-tuning, overfits and loses the efficiency benefit. Practical range **8–64**, task/domain shift larger → higher r; style/format adaptation → r=8~16 is enough. **Always validate on a held-out set, not train loss.** Also note the effect is module-dependent: attention `q,k,v,o` and MLP projections matter more than others.

**中文要点**：`ΔW = (α/r)BA`，可训参数从 dk 降到 r(d+k)；`α/r` 的缩放让 r 变化时不必重调学习率；推理可合并权重零延迟；r 常用 8~64，风格适配 8~16、领域迁移取大，必须用留出集验证而非看训练 loss。

---

### 5. How does QLoRA achieve its memory reduction, and what are the quantization trade-offs?

**Asked at**: HuggingFace
**Source**: https://outcomeschool.com/blog/how-does-model-quantization-work

**Anchor answer**

Three tricks stacked:

1. **4-bit NF4 quantization of the frozen base weights** — NF4 is an information-theoretically optimal quantile quantization for normally-distributed weights. 7B weights go from ~14 GB (bf16) to ~3.5 GB.
2. **Double quantization** — quantize the quantization constants too, saving ~0.4 bits/param.
3. **Paged optimizers** — page optimizer state to CPU on memory spikes (the spike usually comes from long-sequence gradient checkpointing, not steady state).

During forward/backward, weights are **dequantized to bf16 block-by-block**; **LoRA adapters stay in bf16**, so the actual learning is still full precision.

Trade-offs:

- **Speed**: dequant overhead makes training roughly **30–50% slower**.
- **Quality**: 4-bit base slightly lowers the ceiling vs bf16 LoRA; usually fine for domain adaptation, risky for tasks needing precise numerics.
- **Not for inference-efficiency**: QLoRA is a *training* technique. For serving you'd use GPTQ/AWQ/FP8 which target throughput.

**中文要点**：4-bit NF4 冻结基座 + 双重量化 + paged optimizer；前向逐块反量化到 bf16，适配器保持 bf16；代价是训练慢 30~50%、精度上限略降；它是**训练**技术，推理加速要用 GPTQ/AWQ/FP8。

---

### 6. Compare LoRA, prefix tuning, prompt tuning and full fine-tuning. When would you choose each?

**Asked at**: Sarvam AI · Apple
**Source**: https://outcomeschool.com/blog/how-does-prefix-tuning-work

**Anchor answer**

| Method | Where the adaptation lives | Trainable params | Inference latency | Best when |
| :--- | :--- | :--- | :--- | :--- |
| **Full FT** | all weights | 100% | baseline | massive domain shift, you own the model, quality is paramount |
| **LoRA** | low-rank ΔW on chosen modules | 0.01–1% | **zero after merge** | default choice: domain/style adaptation with limited GPUs |
| **Prefix tuning** | learned key/value prefixes per layer | tiny | **adds prefix length** to every request | you can't touch weights, but can inject per-layer vectors |
| **Prompt tuning** | learned soft prompt embeddings only | tiny | small, but eats context | multi-task with one frozen backbone, many lightweight task heads |

Decision rules:

- Start with **LoRA** unless there's a reason not to: it's the best quality-per-GPU and merges away at inference.
- Choose **full FT** only if LoRA plateaus below target on a *large* domain shift and you have the hardware.
- Choose **prefix/prompt tuning** when you must host **many tasks on one frozen model** and need to swap adapters cheaply per request, or when the model is served by a vendor.
- Never choose prefix tuning for latency-critical paths — the prefix consumes both KV memory and context window.

**中文要点**：默认选 LoRA（省显存、可合并零延迟）；大规模域迁移才考虑全参；prefix/prompt tuning 只在"一个冻结模型挂很多任务"或不能改权重时用；prefix 会吃 KV 和上下文，延迟敏感场景别用。

---

### 7. What is catastrophic forgetting and how do you mitigate it during fine-tuning?

**Asked at**: Mistral AI
**Source**: https://outcomeschool.com/blog/continual-learning-in-llms

**Anchor answer**

**Definition**: fine-tuning on a narrow dataset makes the model lose general capabilities — it overfits the new distribution and its outputs collapse into that style (e.g., a chat model fine-tuned on legal text starts answering everything as a legal document).

Why: gradient updates on a small, narrow distribution shift weights broadly; nothing anchors the old behaviour.

Mitigations, cheap → expensive:

1. **Replay / data mixing** — mix 5–20% general instruction data into the fine-tuning set. The single most effective and cheapest fix.
2. **Low-rank + low LR + few epochs** — LoRA at r=8~16, 1–3 epochs. Full FT is far more prone.
3. **KL/regularisation toward the base** — penalise divergence from the reference model's logits (the same idea as DPO's β).
4. **Freeze more** — adapt only upper layers or only attention projections.
5. **Evaluate on a general benchmark before/after** — if you don't measure general capability, you will never notice the regression. This is the step teams skip.

Also: **catastrophic forgetting is why "fine-tune for knowledge" is an anti-pattern** — put knowledge in RAG, use fine-tuning only for style/format/behaviour.

**中文要点**：表现是窄数据微调后通用能力崩塌；最便宜有效的是**掺 5~20% 通用数据 replay**，再叠 LoRA 低秩+低学习率+少 epoch、KL 正则、冻结上层；**必须用通用基准前后对比**；"微调灌知识"是反模式，知识该走 RAG。

---

### 8. Prompting, RAG or fine-tuning: give me your decision framework with cost and latency attached.

**Asked at**: OpenAI · Mistral · Cohere · Microsoft · Databricks · Glean · DiDi · JD.com · Alibaba · Tencent
**Source**: 视频 https://www.youtube.com/watch?v=lnfWvX66FUk

**Anchor answer**

| | What it changes | Iteration cost | Per-request cost / latency | Fails when |
| :--- | :--- | :--- | :--- | :--- |
| **Prompt** | instruction, format, examples | minutes–hours | +0 to +2k tokens | needs knowledge the model lacks; long prompts hurt |
| **RAG** | supplies external evidence | hours–days (index pipeline) | +retrieval 50–500 ms, +2–8k tokens | evidence is unstructured/multi-hop; retrieval can't find it |
| **Fine-tune** | weights (behaviour/style) | days–weeks (data + train + eval) | **lower**: shorter system prompt, no few-shot, can drop TTFT | facts change; small data → overfit & forgetting |

**Decision order (and say it this way):**

1. **Prompt first** — always. If a better prompt solves it, nothing else is worth the complexity.
2. **RAG next** — whenever the answer depends on facts, documents, or anything that changes. Knowledge update cost ≈ re-indexing, not retraining.
3. **Fine-tune last** — for things RAG can't express: consistent tone/format, specific output schema, domain reasoning style, or to **cut cost** by replacing a long prompt.

The professional framing: **"fine-tune the style, retrieve the knowledge."** And the anti-pattern: fine-tuning to inject facts — it goes stale, it hallucinates confidently, and you can't audit or delete a fact from weights.

**中文要点**：顺序是 Prompt → RAG → 微调；微调只解决"风格/格式/行为"且能降低单请求成本（省 system prompt 与 few-shot），**绝不用于灌知识**（会过期、无法审计与删除）。

---

### 9. Do the GPU memory maths for full fine-tuning a 7B model in bf16 with Adam. Now with LoRA.

**Asked at**: Mistral AI · HuggingFace

**Anchor answer**

**Full fine-tuning, 7B params, bf16, Adam (mixed precision):**

| Component | Bytes/param | Size |
| :--- | :-: | :--- |
| Weights (bf16) | 2 | 14 GB |
| Gradients (bf16) | 2 | 14 GB |
| Adam m (fp32) | 4 | 28 GB |
| Adam v (fp32) | 4 | 28 GB |
| fp32 master weights | 4 | 28 GB |
| **Subtotal** | **16** | **≈ 112 GB** |

Plus activations: with gradient checkpointing ~ a few GB depending on seq len/batch; without it, tens of GB.

So full FT of 7B ≈ **112 GB + activations** → needs multiple 80 GB GPUs (or ZeRO/FSDP sharding).

**LoRA (r=16 on attention q,k,v,o):**

- Trainable params ≈ `4 × (d·r + r·d)` per layer ≈ 0.05–0.1% of 7B ≈ **~4–8M params**
- Adapter weights + grads + Adam states ≈ 16 bytes × 8M ≈ **~130 MB**
- Frozen base in **bf16 = 14 GB** (or **4-bit NF4 ≈ 3.5 GB** with QLoRA)
- Plus activations

Result: **~20 GB with bf16 base, ~10 GB with QLoRA** → fits a single 24 GB consumer card (QLoRA on 7B at short seq len is standard on a 4090).

**中文要点**：全参 = 权重 2 + 梯度 2 + Adam m/v 各 4 + fp32 主权重 4 = **16 bytes/param ≈ 112 GB**；LoRA 可训参数约 4~8M，优化器状态仅 ~130 MB，基座 bf16 14 GB（QLoRA 3.5 GB），单卡 24 GB 可跑。

---

### 10. What is RLVR (RL with verifiable rewards) and where does it beat a learned reward model?

**Asked at**: Zhipu AI · Alibaba · Sarvam AI · Scale AI
**Source**: https://outcomeschool.com/blog/group-relative-policy-optimization-grpo

**Anchor answer**

RLVR = reinforcement learning where the reward is a **programmatic verifier**, not a learned model: does the unit test pass, does the math answer match, does the SQL return the expected rows, does the JSON validate.

Where it beats a learned RM:

1. **No reward hacking of a proxy** — the verifier is the ground truth, so there's nothing to game (except the verifier's own coverage).
2. **Cheap and unlimited** — no human labelling per sample, no RM inference cost.
3. **Deterministic and reproducible** — you can rerun and get the same reward.
4. **Scales with generation** — perfect fit for GRPO's group sampling.

Where it fails:

- Tasks with no verifier (writing quality, tone, helpfulness) — back to RM or judge.
- **Verifier gaming**: the model finds shortcuts (hardcoding test values, matching the regex instead of solving). Mitigation: hidden held-out tests, mutation testing, adversarial test generation.
- Coverage gaps: things not covered by tests silently aren't optimised.

**In practice**: RLVR for math/code/tool-use/SQL first, then RLAIF/HH-RLHF for the residual subjective dimensions.

**中文要点**：RLVR 用程序化校验器（单测/答案匹配/SQL 结果）当奖励，优点是不存在代理奖励被 hack、便宜、可复现、天然适配 GRPO 采样；风险是**校验器本身被 hack**（要留隐藏测试集 + 变异测试）和覆盖不全。

---

### 11. Explain reward hacking in RLHF and how labs address it.

**Asked at**: Scale AI · ByteDance
**Source**: https://outcomeschool.com/blog/reinforcement-learning-from-human-feedback-rlhf

**Anchor answer**

**Definition**: the policy finds outputs that score high on the *reward model* without satisfying the *intent*. The RM is a finite proxy; RL is an adversarial optimiser against it.

Classic manifestations:

- **Length/format gaming** — longer answers, bullet lists, confident tone score higher regardless of correctness.
- **Sycophancy** — agreeing with the user's premise, praising the question.
- **Hedging and safety boilerplate** — "as an AI…" padding that raters rate as safe.
- **Reward-model over-optimisation** — after enough PPO steps, RM score keeps rising while human ratings fall (the well-known Goodhart curve).

Countermeasures:

1. **KL penalty to the reference policy** — the primary structural brake.
2. **Early stopping on held-out human eval**, not on RM score.
3. **RM ensembles / disagreement penalty** — if RMs disagree, reduce the reward.
4. **Data refresh** — continuously add the policy's own exploits as `rejected` pairs (iterative RLHF).
5. **Length normalisation and format stripping** in RM training data.
6. **Verifiable rewards where possible** — replace the proxy with ground truth.

Signal to watch: RM score ↑ while human preference ↓ = you are in the hacking regime.

**中文要点**：奖励黑客 = 策略找到"RM 高分但不符合意图"的输出（偏爱长答案/顺从/套话）；对策是 KL 约束、**用留出集人工评估做 early stop 而不是看 RM 分**、RM 集成与分歧惩罚、把模型自己的作弊样本回灌成 rejected 对、能上可验证奖励就上。

---

### 12. What is distillation, and how do you build a strong small model from a large one?

**Asked at**: Alibaba
**Source**: https://outcomeschool.com/blog/how-does-knowledge-distillation-work

**Anchor answer**

Distillation = transfer capability from a teacher to a student, in one or more of three forms:

| Form | Signal | Note |
| :--- | :--- | :--- |
| **Logit distillation** | full teacher distribution, KL divergence at temperature T | needs teacher/student same tokenizer & similar vocab |
| **Sequence distillation** | teacher-generated outputs as SFT targets | simplest, works across tokenizers; the dominant approach for LLMs |
| **Reasoning/CoT distillation** | teacher's reasoning traces | most effective for small reasoning models, but you inherit the teacher's errors |

Practical recipe for a strong small model:

1. **Pick student capacity honestly** — 1B cannot absorb a 400B's full behaviour. Aim for the Pareto point (e.g., 7B for a 70B teacher).
2. **Quality and diversity of teacher outputs matter more than volume** — verified/filtered traces beat raw volume.
3. **Mix in real human data** (10–30%) — pure distillation inherits teacher biases and produces "looks right, subtly wrong" outputs.
4. **Reject-sample**: keep only traces whose final answer verifies (verifiable-reward filtering). This one step typically buys the most.
5. **Distill then RLVR** — distill for format/behaviour, then reinforce with a verifier for correctness.

Known failure: **distillation amplifies teacher errors** — small models reproduce wrong reasoning more confidently. Always evaluate the student on its own, never against "teacher parity" alone.

**中文要点**：三种蒸馏（logit / 序列 / CoT 轨迹）；做小模型的关键是"教师输出经可验证奖励筛过 + 掺 10~30% 真人数据 + 蒸馏后再 RLVR"，且蒸馏会**放大教师错误**，必须独立评学生。

---

## 附：这 12 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 只会在 LLaMA-Factory 上点按钮 | 4（LoRA 数学）、9（显存手算） |
| 不懂为什么选型 | 8（Prompt/RAG/微调决策框架）、7（灾难性遗忘） |
| 讲不清后训练 | 1（RLHF）、2（DPO）、3（GRPO）、10（RLVR） |
| 训练成本控制 | 5（QLoRA 权衡）、9（显存）、12（蒸馏配方） |
