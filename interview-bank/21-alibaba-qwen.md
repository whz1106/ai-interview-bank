# 21 · Alibaba (Qwen)

> 10 题。出处：`README.en.md` § Alibaba (Qwen)。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Qwen2.5-Coder trains with repository-level fill-in-the-middle using tokens like <|fim_prefix|>, <|fim_suffix|>, <|repo_name|>. Write the function that formats a repo-level FIM example, and explain why repo-level beats file-level.

**Anchor answer**

**Reference implementation**

```python
import os, random

FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE = "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"
REPO_NAME, FILE_SEP, EOD = "<|repo_name|>", "<|file_sep|>", "<|endoftext|>"

def format_repo_fim(files: dict[str, str], target_file: str,
                    n_ctx: int = 4, rng: random.Random | None = None) -> str | None:
    """files: {path: content}. Returns one training string or None if unusable."""
    rng = rng or random.Random()
    body = files[target_file]
    if len(body) < 64:
        return None

    # 1. pick a random span to hold out; keep the split on line boundaries
    lines = body.splitlines(keepends=True)
    if len(lines) < 4:
        return None
    i = rng.randrange(len(lines))
    j = rng.randrange(i + 1, len(lines) + 1)
    prefix, middle, suffix = "".join(lines[:i]), "".join(lines[i:j]), "".join(lines[j:])

    # 2. context files: same-repo neighbours, truncated to fit a token budget
    others = [p for p in files if p != target_file]
    rng.shuffle(others)
    ctx, budget = [], n_ctx
    for p in others:
        if budget == 0:
            break
        snippet = files[p]
        if len(snippet) > 4096:                     # head+tail, not a blind cut
            snippet = snippet[:2048] + "\n...\n" + snippet[-2048:]
        ctx.append(f"{FILE_SEP}{p}\n{snippet}")
        budget -= 1

    repo = os.path.dirname(target_file).split(os.sep)[0] or "root"
    head = f"{REPO_NAME}{repo}\n"
    return (
        head
        + "".join(ctx)
        + f"{FILE_SEP}{target_file}<|fim_prefix|>{prefix}"
        + f"<|fim_suffix|>{suffix}<|fim_middle|>{middle}{EOD}"
    )
```

**Key points**

- Hold out a **middle span**, not a suffix — suffix-only FIM teaches "complete the end", which is a much easier and less useful task.
- Suffix must come *before* the middle in the string (that's the point of the format); getting this backwards silently trains a nonsense objective that still shows falling loss.
- Repo context first, target file last: prefix/suffix of the target must stay adjacent to the mask, or the model learns to use far context as if it were local.
- Truncate context files head+tail; a blind head-cut deletes imports and signatures, which is exactly what the model needs.
- Randomise the number of context files (0–4) per example, or inference with a different context budget goes off-distribution.

**Why repo-level beats file-level:** real completions depend on cross-file knowledge — a call site needs the callee's signature, a class needs its base class, a test needs the fixture. File-level FIM can only learn language statistics plus single-file structure, and produces code that is locally plausible and globally wrong (wrong argument order, wrong exception type). Repo-level FIM is what turns "autocomplete" into "understands the project".

**中文要点**：必须挖中间的 span 而不是后缀；suffix 要排在 middle 之前，顺序写反 loss 照样下降但学的是错目标；仓库上下文放前面、目标文件放最后以保持 mask 邻接；上下文文件头尾截断并随机化数量。仓库级 FIM 学到跨文件签名/继承/夹具，文件级只能学到通顺但全局错误。

---

### 2. Qwen uses byte-level BPE with a ~151K vocabulary, augmented for multilingual coverage and with digits split into single characters. Why those choices, and what are the trade-offs?

**Source**: https://outcomeschool.com/blog/bpe-in-llms

**Anchor answer**

**Byte-level:** nothing is unrepresentable — any Unicode string, emoji, or mojibake degrades to bytes instead of `<unk>`. That matters when you serve 29+ languages and user-generated text.

**~151K vocab:** more slots means fewer merges per word, so fertility drops in every language. At 32K, Chinese runs ~1.5–2 tokens/char and Hindi worse; at 151K you get much closer to 1 token per common word across scripts.

| Choice | Gain | Cost |
| :--- | :--- | :--- |
| Big vocab | shorter sequences ⇒ cheaper attention, longer effective context, better few-shot | embedding + LM head grow to $151\text{K} \times d$; softmax compute; rare tokens get few gradient updates |
| Byte fallback | robust to any input | bytes are the worst tokenisation for rare scripts; silent quality cliff |
| **Digits split to single chars** | `1234` can never be an opaque token; arithmetic, dates, IDs, and copying numbers become token-level operations | numbers take 2–4× more tokens than a merged encoding |

**Why split digits:** if `2024` is one token, the model can't generalise from `2024` to `2025` — they're unrelated vectors. With single digits, place-value structure is visible, and arithmetic/copying works via mechanisms over positions. The cost is sequence length on numeric-heavy text (logs, tables, finance), and a slightly higher chance of digit-level hallucination being *plausible*.

**The trade-off nobody mentions:** a large vocab makes distillation and quantisation harder (the embedding matrix dominates small-model parameter counts), and it locks you in — changing the tokenizer invalidates every fine-tune downstream.

**中文要点**：byte-level 保证任何文本不 OOV；151K 词表把各语种 fertility 压下来、序列变短；代价是 embedding/LM head 变大、softmax 更贵、稀有 token 欠训练。数字拆成单字符是为了让进位/复制/算术在 token 级可见（`2024` 整块 token 无法泛化到 `2025`），代价是数字密集文本序列变长。

---

### 3. Qwen3 unifies a thinking mode and a non-thinking mode in one model with a caller-settable thinking budget. How would you train that, and how would you serve it?

**Source**: https://outcomeschool.com/blog/large-reasoning-models

**Anchor answer**

**Training — four stages, in this order:**

1. **Long-CoT cold start.** SFT on filtered reasoning traces (verifier-checked answers only) so the thinking mode has a readable format. Cheap, and it sets the ceiling for later RL.
2. **Reasoning RL.** GRPO on verifiable math/code with a **length-aware reward**: pass/fail plus a soft penalty for exceeding the target budget. Without the length term, RL reliably discovers "think longer" as a free win and you get 20K-token answers for 3+3.
3. **Thinking-mode fusion.** SFT on a mix where the *same prompt* appears with a long CoT and with a direct answer, with an explicit mode token. This is the step that actually teaches the switch; skip it and modes leak into each other.
4. **General RL.** Preference/rule rewards over both modes for helpfulness, format, and tool use — so the non-thinking mode isn't just "truncated thinking".

**Budget control at inference** is a decode-time contract: the caller passes `max_thinking_tokens`; the server tracks the reasoning segment length and, on reaching the cap, injects the end-of-thinking token and forces the answer phase. Train with samples that hit the cap so the model knows how to wrap up abruptly — models never trained on forced termination produce incoherent final answers when cut.

**Serving implications:**

| Concern | Handling |
| :--- | :--- |
| Latency bimodality | separate queues or per-request budgets; capacity plan on the thinking mode |
| KV pressure | a 30K reasoning trace per concurrent user is real HBM; cap concurrency per GPU |
| Streaming UX | stream reasoning to a collapsible pane, answer to the main pane |
| Prefix caching | shared system prompt is the only reliable cache hit; thinking traces are per-request |

**中文要点**：四阶段——长 CoT 冷启动 SFT、带长度惩罚的推理 RL（否则奖励会退化成"想得越久越好"）、同 prompt 双答案的思考融合 SFT（真正教会开关，否则模式互渗）、双模式通用 RL；推理侧用 `max_thinking_tokens` + 到点注入结束标记强制收尾，训练时就要见过被截断的样本；服务上要按思考模式做容量规划并分开排队。

---

### 4. Qwen ships both dense and MoE models (30B with ~3B active; 235B with ~22B active). When would you pick the 30B-A3B MoE over a 32B dense?

**Source**: https://outcomeschool.com/blog/mixture-of-experts

**Anchor answer**

Same parameter *count*, completely different cost curves — that's the whole comparison.

| Dimension | 30B-A3B MoE | 32B dense |
| :--- | :--- | :--- |
| Weights in memory | ~30B (all experts must be resident) | ~32B |
| FLOPs per token | ~3B-equivalent | ~32B-equivalent |
| Large-batch throughput | **much better** (10× fewer FLOPs/token) | FLOP-bound |
| Batch-1 decode latency | better in FLOPs, but scattered expert reads and routing overhead eat much of it | predictable, dense, bandwidth-friendly |
| Fine-tuning on 1–2 GPUs | hard (expert state, imbalance) | straightforward |
| Quantisation to 4-bit | uneven (experts differ in importance) | uniform |
| Quality at equal tokens trained | usually slightly lower per-FLOP-efficiency, higher per-token | predictable scaling |

**Pick the MoE when:** you serve many concurrent users and are throughput-bound; you have ≥ 2–4 GPUs per replica so routing overhead amortises; your workload is prefill-heavy (RAG, long prompts); and you want 30B-class quality at roughly 3B-class compute cost.

**Pick the dense when:** you deploy on one GPU or at the edge; you need cheap LoRA fine-tuning for many tenants; your traffic is bursty at batch 1 and predictable latency matters more than cost per token; or you must quantise aggressively and can't tolerate uneven expert degradation.

**The measured surprise:** at batch 1 on a single GPU, the MoE's advantage mostly evaporates — the routing overhead and non-coalesced expert weight reads cancel the FLOP savings. MoE is a *batching* technology, not a latency technology.

**中文要点**：两者权重都是 30B 级，区别在每 token 算力——MoE 约 3B 稠密成本、吞吐高 10 倍，但 batch=1 时路由开销与专家权重的零散读取把优势吃光；所以并发高、prefill 重、每副本 2–4 卡选 MoE，单卡/边端、需多租户 LoRA、激进量化选稠密。MoE 是"批处理技术"不是"低延迟技术"。

---

### 5. Qwen2.5 extends context to 128K (and ~1M for Turbo) using YaRN plus Dual Chunk Attention, mostly training-free. Explain how, and why post-hoc extension is attractive.

**Source**: https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding

**Anchor answer**

**YaRN (NTK-by-parts)** — treat RoPE dimensions differently by wavelength:

| Dim band | Wavelength vs trained length | Operation |
| :--- | :--- | :--- |
| High frequency (short λ) | fine at any length | leave **unscaled** — scaling blurs local order |
| Low frequency (long λ) | far out of range | **interpolate** positions (position $t \to t/s$) |
| Middle | ambiguous | ramp factor $\gamma$ between the two |

Plus an **attention temperature** correction $\sqrt{1/t}$ that compensates for the entropy change caused by longer contexts — without it, long-context attention is too flat and quality drops even though perplexity looks fine.

**Dual Chunk Attention** — the serving trick. Split the sequence into chunks (e.g. 32K). Within a chunk: ordinary causal attention with relative positions. Across chunks: a compressed representation of each preceding chunk's keys. Result: attention cost grows roughly linearly in the number of chunks instead of quadratically in tokens, and relative position encoding stays in a *trained* range for most pairs.

**Why post-hoc extension is attractive:** you pay a few billion tokens of continued pretraining plus a config change, instead of retraining a model whose pretraining already cost 10–15T tokens. You keep the short-context quality, you can ship 128K and 1M variants from the same base, and you can tune the scaling factor per customer.

**The honest limitation:** "mostly training-free" ≠ free. Without at least a small long-context continued-pretraining phase, YaRN-extended models degrade on *reasoning* over long inputs even when retrieval tests pass — they can find the needle but not compare two needles.

**中文要点**：YaRN 按波长分段处理 RoPE——高频维不动、低频维插值、中间渐变，并做 $\sqrt{1/t}$ 温度校正；DCA 把长序列切块，块内普通因果注意力、块间用压缩表示，把 $O(n^2)$ 压成近线性且让相对位置落在训练范围内。后验扩展便宜（几 B token 续训 vs 重训 10T+），但纯免训练版本常常"能检索不会推理"。

---

### 6. Qwen3 uses strong-to-weak distillation, bootstrapping smaller models from flagship ones. How does that work and why is it cheaper?

**Source**: https://outcomeschool.com/blog/how-does-knowledge-distillation-work

**Anchor answer**

**Setup.** The flagship (teacher) already went through the full four-stage post-training — long-CoT SFT, reasoning RL, thinking fusion, general RL. The small model (student) skips RL entirely and learns from the teacher's *output distribution*.

**Mechanism:**

1. Sample prompts from the post-training distribution (mixed domains, both modes).
2. Run the teacher to get **top-k logits** (not just the argmax token) at each position.
3. Train the student with a KL divergence to those logits, plus the standard CE on the reference answer. Optionally MSE on hidden states.
4. Do it **off-policy**: the student trains on teacher-generated sequences, not its own rollouts.

**Why cheaper — three separate reasons:**

| Saving | Magnitude |
| :--- | :--- |
| No RL rollouts | RL needs many samples per prompt (GRPO: a group of 8–16) and a verifier; distillation needs **one** teacher pass, and teacher logits can be precomputed and cached on disk |
| No reward engineering | no verifier, no length curriculum, no reward hacking to debug |
| No critic / value model | same as RL-for-verifiable-rewards, but also no preference model for the general stage |

Empirically, distilling a 0.6B–8B student from a strong 30B+ teacher beats running RL on the small model directly — the small model simply cannot explore well enough to find the reasoning behaviours RL rewards.

**The catch:** the student is capped by the teacher, and *off-policy* distillation means the student never sees its own error states, so it's brittle on inputs where it diverges from the teacher. Fix: mix in a small on-policy SFT round, and keep the teacher's logits diverse (top-k = 20–100, not top-1) so the student inherits the teacher's uncertainty.

**中文要点**：学生完全跳过 RL，直接对教师 top-k logits 做 KL + 参考答案 CE，且是 off-policy（教师序列可预生成缓存），所以省掉 rollout、验证器、奖励工程与 critic 三笔开销；效果上 0.6B–8B 学生蒸馏通常胜过在小模型上直接跑 RL（小模型探索不出被奖励的行为）。代价是上限等于教师、且没见过自身错误状态，需补少量 on-policy SFT。

---

### 7. Qwen's reasoning models train with RL using verifiable rewards on maths and code. Why is that preferred over PPO with a learned reward model for these domains?

**Source**: https://outcomeschool.com/blog/group-relative-policy-optimization-grpo

**Anchor answer**

Because in maths and code, **the reward is computable** — and a learned reward model is a lossy, gameable proxy for something you can just measure.

| | Learned RM + PPO | Verifiable reward + GRPO |
| :--- | :--- | :--- |
| Reward source | human preferences → RM | unit tests, numeric answer match, compiler |
| Failure mode | **reward hacking** (longer answers, confident tone, sycophancy) | answer extraction bugs; no stylistic hacking possible |
| Extra infrastructure | RM training, RM serving during RL, preference data upkeep | a sandbox/executor and a parser |
| Signal quality | noisy, biased by annotator pool | exact, but sparse and binary |
| Variance reduction | critic / value model | group-relative baseline (no critic) |

**GRPO's specific trick** removes the value network: sample $G$ rollouts per prompt, use the group's mean reward as the baseline, and normalise advantages within the group. That halves memory (no critic the size of the policy) and removes a whole class of value-estimation bugs.

**What you give up:** coverage. RLVR only works where a verifier exists — no verifiable reward for "write a good email". And binary rewards are sparse, so you need a difficulty curriculum: if all $G$ rollouts fail, the group has zero variance and the gradient is exactly zero. That's the most common silent stall in RLVR runs.

**Failure modes I'd flag:** the model learning to exploit **weak tests** (prints the expected value, hard-codes inputs), so the test suite must be hidden and mutated; and answer-format brittleness, where a correct solution is scored wrong because `\boxed{}` parsing failed — measure the parser's false-negative rate before blaming the model.

**中文要点**：数学/代码有可计算奖励，学习型 RM 只是可被刷的有损代理（偏爱长答案、自信语气、迎合）；GRPO 再用组内均值做基线省掉 critic。代价是只覆盖可验证域、二值奖励稀疏——若一组 rollout 全错则方差为零、梯度为零，这是 RLVR 最常见的静默停滞，所以要做难度课程；同时要防模型钻弱测试的空子、并先量好答案解析器的漏判率。

---

### 8. Qwen ships open weights that top public leaderboards. As the release engineer, how do you make sure the benchmark numbers are trustworthy and not contaminated?

**Source**: https://outcomeschool.com/blog/llm-evaluation

**Anchor answer**

Treat the number as a **claim with an audit trail**, and assume contamination until proven otherwise.

**Pre-release gate, in order:**

1. **Training-data decontamination.** For every benchmark you plan to cite, extract its items and run n-gram (13-gram is the usual threshold) plus embedding-similarity overlap against the whole pretraining corpus — including the *answers*. Report the overlap rate; anything above a fraction of a percent means the score is invalid for that benchmark.
2. **Canary / held-out split.** Keep an internal benchmark whose items were created after the training cutoff and never published. This is the only number you should trust for a go/no-go.
3. **Prompt and harness parity.** Pin the eval harness version, prompt template, few-shot count, and decoding params. Publish them. Qwen-style models are especially sensitive to chat-template mismatches — a wrong template can cost 5–10 points and looks like a model regression.
4. **Suspicious-perfect check.** If a score jumps more than ~3 points over the previous generation while internal evals move less, assume leakage, not a breakthrough.

**Audit table to publish:**

| Item | Value |
| :--- | :--- |
| Benchmark + split + harness commit | pinned |
| Contamination probe result | overlap % |
| pass@1 / maj@k / both | stated |
| Decoding config, tool versions | stated |
| Private shadow score | stated |

**Why this matters commercially:** a contaminated claim that gets caught costs far more than two points of leaderboard position — and for open weights, the community *will* re-run your numbers within 48 hours.

**中文要点**：把分数当"带审计链的声明"——先对全部预训练语料做 n-gram + 向量去污染并报重叠率，再留一个截止日期后的私有 canary 集做唯一可信的发布判据，钉死 harness/模板/解码参数；分数跳升 >3 点时默认是泄漏而非突破。开源权重下社区 48 小时内就会复现你的数字。

---

### 9. Qwen2.5-VL uses a native dynamic-resolution ViT with window attention and multimodal RoPE. Why native resolution instead of fixed tiling, and what does MRoPE encode?

**Source**: https://outcomeschool.com/blog/decoding-vision-transformer-vit

**Anchor answer**

**Why native resolution.** Fixed tiling forces every image into $k$ tiles of 448×448, which means (a) resizing distorts aspect ratio, and (b) small text in a screenshot gets resampled away — fatal for document and GUI understanding. Native resolution patches the image at its own size: a 1024×300 receipt produces exactly the tokens it needs, a 4K screenshot keeps its legibility.

| | Fixed tiling | Native dynamic res |
| :--- | :--- | :--- |
| Token count | constant, wasteful on small images | proportional to area; needs a min/max cap |
| Aspect ratio | distorted or padded | preserved |
| Small text / fine detail | destroyed by downscale | preserved |
| Long images (scrolling screenshots) | needs special casing | natural, if you allow extreme ratios |

The obvious cost is **quadratic attention over variable token counts** — a 4K screenshot can be 10K+ patches. Hence **window attention**: most layers attend only within local windows, with a few global layers for long-range context, keeping cost near-linear while retaining global structure.

**MRoPE** decomposes the position into three axes — **temporal (frame index), height, width** — instead of a single flat index. Each attention head gets a slice of the rotary dimensions allocated across the three. Why it matters:

- Absolute **time** is encoded, so a video's "2 minutes in" means something, not just "frame 720".
- **2D spatial** structure survives, so a box at (x=0.9, y=0.1) is distinguishable from one at (0.1, 0.9) even at the same flat index.
- It generalises to **arbitrary resolution and duration** — the model interpolates over position rather than memorising a grid.

**The failure it fixes:** with 1D positions, two images of the same content at different aspect ratios can look identical; with MRoPE they don't.

**中文要点**：固定切块会缩放畸变、把截图小字重采样掉；原生分辨率按实际尺寸切 patch、保比例与细节，代价是 token 数随面积暴涨，所以用窗口注意力（少数全局层）把复杂度拉回近线性。MRoPE 把位置拆成时间/高/宽三轴并分给不同旋转维，既能表达"第 2 分钟"这种绝对时间，也让 2D 空间可区分，并可外推到任意分辨率与时长。

---

### 10. Alibaba open-sources Qwen under Apache 2.0 while running a commercial cloud business. Walk me through the strategy, and tell me about an ambiguous technical decision you owned end to end.

**Anchor answer**

**The strategy has three layers, and they reinforce each other:**

| Layer | Mechanism | Who captures the value |
| :--- | :--- | :--- |
| Open weights (Apache 2.0) | maximal adoption; no legal friction for enterprises; community fine-tunes become free R&D | ecosystem: Qwen is the default base for thousands of derivative models |
| Cloud/Model Studio | the *serving*, not the weights, is what enterprises actually buy: SLA, elastic GPU, private endpoints, data residency | Alibaba Cloud consumption |
| Sizes and variants | a full ladder (0.5B→235B, dense/MoE, VL, Coder, Audio) so the free tier is a funnel, not a competitor | upsell path: prototype on 7B open weights, ship on hosted 235B |

The licensing choice is the strategic bet: a restrictive licence would cap adoption at the exact moment the market is deciding which model family becomes the default. Apache 2.0 buys distribution; distribution buys the cloud bill. And it's asymmetric — the weights are copied once, the inference bill recurs.

**Risks worth naming:** commoditisation of the base model (margin migrates to serving, tooling, and vertical solutions); competitors free-riding; and the operational cost of supporting a huge variant matrix.

**On the ambiguous decision** — structure it as: the ambiguity was X (e.g. whether to ship a unified thinking/non-thinking model vs two checkpoints), the information was incomplete because Y, so I defined a decision rule up front (which metric decides, what would falsify it), ran the cheapest discriminating experiment, committed, and set a review trigger. The point an interviewer scores is that you *named the criterion before seeing the result*, not that you guessed right.

**中文要点**：开源是分发策略、云是变现路径——Apache 2.0 换来"默认基座"地位与海量衍生模型，企业真正付费的是 SLA、弹性算力、私有端点与数据驻留；全尺寸梯度（0.5B–235B + VL/Coder/Audio）让免费层成为漏斗。风险是基座商品化、利润迁移到服务与垂直方案。讲模糊决策时重点在"先定判据再跑最便宜的判别实验"。

---

## 附：这 10 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 能写能跑 | 1（仓库级 FIM 格式化） |
| 分词与多语言 | 2（151K 词表 + 数字拆分） |
| 架构与上下文 | 5（YaRN + DCA）、9（原生分辨率 + MRoPE） |
| 训练/后训练 | 3（思考预算）、6（强到弱蒸馏）、7（RLVR vs PPO） |
| 选型判断 | 4（30B-A3B vs 32B 稠密） |
| 工程严谨与商业感 | 8（榜单可信度）、10（开源策略 + 模糊决策） |
