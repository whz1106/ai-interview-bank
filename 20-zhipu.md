# 20 · Zhipu AI (GLM)

> 11 题。出处：`README.en.md` § Zhipu AI (GLM)。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. GLM's original pre-training objective is autoregressive blank infilling. How does it differ from BERT and GPT, and why did the team argue it unifies understanding and generation?

**Source**: https://outcomeschool.com/blog/encoder-vs-decoder-in-transformers

**Anchor answer**

Autoregressive blank infilling: sample spans from the document, replace each with `[MASK]`s, then **generate the spans one token at a time, autoregressively**, in an order you choose. The corrupted text is attended bidirectionally; the generated spans attend causally to themselves *and* bidirectionally to the corrupted context.

| | BERT | GPT | GLM (blank infilling) |
| :--- | :--- | :--- | :--- |
| Context visibility | bidirectional, fixed | causal, fixed | bidirectional context + causal generation |
| Mask independence | predicts masks **independently** (conditional independence assumption) | n/a | spans predicted **jointly** — span 2 sees span 1 |
| Generation | cannot | yes | yes, same head and same objective |
| Long-span structure | no notion of span | no notion of span | span length is a modelling choice |

**Why it unifies.** Understanding = recovering a *short* blank from full context (this is essentially a cloze task, which is what BERT does). Generation = recovering a *long* blank — in the limit, the whole document is one masked span, and blank infilling degenerates exactly to standard left-to-right LM. So NLU and NLG are the same objective at different span lengths, and one model needs no task-specific heads.

**The practical detail people forget:** span order is randomised (2D positional encoding). That teaches the model to condition on arbitrary subsets of the answer — which is why GLM can do infilling and later made a natural base for fill-in-the-middle code models.

**中文要点**：自回归空白填充=掩码片段逐个 token 自回归生成，被破坏文本双向可见、生成片段因果自洽；相比 BERT 消除了片段间独立性假设，相比 GPT 获得了双向上下文；生成本质是"整篇文档一个大空"，所以理解与生成是同一目标的不同跨度长度。

---

### 2. GLM-4.5 is an MoE with 355B total but 32B active parameters. Explain the economics: what does that split buy you and what does it cost?

**Source**: https://outcomeschool.com/blog/mixture-of-experts

**Anchor answer**

**What the split buys:** per-token compute is roughly that of a 32B dense model ($\approx 2 \times 32\text{B}$ FLOPs), so training cost per token and serving cost per token both drop by ~10× versus a 355B dense model — while the *capacity* to store knowledge stays at 355B. In the compute-bound regime (large batch, prefill, training) this is close to free quality.

**What it costs:**

| Cost | Detail |
| :--- | :--- |
| Memory floor | all 355B must be resident even at batch 1 — ~355 GB in FP8, so ≥ 5 H100s before KV cache |
| Bandwidth at low batch | decode reads only active experts, but the access pattern is scattered; you get neither dense-model locality nor full-model streaming |
| All-to-all latency | two exchanges per layer; at batch 1 these dominate the step |
| Fine-tuning | full FT needs optimizer state for the *whole* 355B, not the active part |
| Routing fragility | load imbalance turns one expert into the critical path |
| Quality on long-tail knowledge | total capacity is large but each rare fact lives in few experts; routing errors lose it |

**The decision rule:** MoE wins when you are throughput-bound and memory-rich (cloud serving, training); dense wins when you are memory-bound, latency-bound at batch 1, or need to fine-tune cheaply on few GPUs. The "32B active" number is a *compute* claim, never a *memory* claim — and interviewers ask precisely because candidates conflate the two.

**中文要点**：355B 总量保容量、32B 激活降每 token 算力（训练与服务都省约 10×）；代价是显存地板仍是 355B（FP8 约 355GB）、低 batch 下带宽与 all-to-all 主导、全参微调要存全量优化器状态、路由错就丢长尾知识。"32B 激活"是算力说法，不是显存说法。

---

### 3. Implement a top-k MoE router in PyTorch. Then contrast auxiliary-loss load balancing with a loss-free approach.

**Source**: https://outcomeschool.com/blog/mixture-of-experts

**Anchor answer**

**Reference implementation**

```python
import torch, torch.nn as nn

class TopKRouter(nn.Module):
    def __init__(self, d, n_experts, top_k, mode="bias", gamma=1e-3):
        super().__init__()
        self.top_k, self.mode, self.gamma = top_k, mode, gamma
        self.gate = nn.Linear(d, n_experts, bias=False)
        self.register_buffer("bias", torch.zeros(n_experts))

    def forward(self, x):                      # [T, d]
        logits = self.gate(x)                  # [T, E]
        probs = logits.sigmoid()
        sel_scores = probs + self.bias if self.mode == "bias" else probs
        idx = sel_scores.topk(self.top_k, dim=-1).indices      # [T, k]
        w = probs.gather(1, idx)
        w = w / w.sum(-1, keepdim=True)

        # fractional load per expert, and the aux loss (only in "aux" mode)
        onehot = torch.zeros_like(probs).scatter_(1, idx, 1.0)
        load = onehot.sum(0) / (x.shape[0] * self.top_k)        # [E]
        imp = probs.mean(0)                                    # [E], differentiable
        aux = (load.detach() * imp).sum() * self.n_experts if self.mode == "aux" else x.new_zeros(())
        return idx, w, aux, load

    @torch.no_grad()
    def update_bias(self, load):
        if self.mode != "bias":
            return
        target = 1.0 / self.bias.numel()
        self.bias += self.gamma * torch.sign(target - load)    # sign-only integral control
```

**Key points**

- `load` must be **detached** in the aux loss — only the importance term carries gradient; otherwise you are differentiating through an argmax, which is meaningless.
- Update `bias` with `sign`, not the raw error: a proportional update oscillates and the router chatters between experts.
- `bias` is a buffer: it must **not** receive gradient or be touched by weight decay.
- Loss-free still needs a *tiny* sequence-wise aux term to stop per-device imbalance (global balance ≠ per-rank balance).
- The aux coefficient is a real hyperparameter with a real quality cost — that is the whole motivation for the bias approach.
- In `bias` mode, return the aux term as a **zero tensor**, not `None`: a `None` slipping into the loss sum is the most common silent bug in a shared trainer.

**中文要点**：aux 损失里 load 必须 detach、只有重要性项带梯度；bias 用 sign 更新做积分控制而非比例控制（否则震荡），且不能吃梯度/权重衰减；loss-free 仍要一个极小的 sequence-wise 项管住每卡不均衡。

---

### 4. What is Multi-Token Prediction (MTP), why add an MTP layer, and how does it help at inference time?

**Anchor answer**

MTP attaches extra heads that predict tokens $t{+}2, t{+}3, \dots$ rather than only $t{+}1$. The sequential form feeds the *actual* intermediate token's embedding back in with the previous depth's hidden state, through a projection and one transformer block:

$$h_t^{(k)} = \text{Block}_k\big(\text{proj}([\,h_t^{(k-1)};\ \text{Emb}(x_{t+k})\,])\big)$$

**Why add the layer (rather than a bare linear head):** a plain linear probe off $h_t$ can't represent "what comes after I commit to token $t{+}1$" — it has no access to that commitment. The extra block is the minimum capacity to model the conditional. Sequential (not parallel) MTP also avoids training/serving mismatch, so the head can be reused as a draft.

**Inference benefit — built-in speculative decoding:**

| | Standard decode | With MTP head |
| :--- | :--- | :--- |
| Tokens per forward | 1 | 1 verified input + 1–2 drafted |
| Extra model | separate draft model to train and host | already in the checkpoint |
| Typical acceptance | 70–90% for a good draft | ~85–90% on the second token (reported) |

Because the draft is cheap (one block, not a whole model), the net effect is roughly 1.5–2× decode throughput with no extra memory beyond the heads.

**The caveat:** the gain is bounded by acceptance rate, and acceptance collapses on high-entropy output (creative writing, unusual code). It's a throughput trick for predictable text — and the MTP loss weight $\lambda/k$ has to be tuned or the shallow task degrades.

**中文要点**：MTP 加深度头预测 $t{+}2/t{+}3$，串行地把真实中间 token 的 embedding 喂回上一深度隐状态再过一个小 block，否则无法建模"承诺了 $t{+}1$ 之后"的条件分布；推理时这些头就是内置 draft，白赚约 1.5–2× 解码吞吐，但收益受接受率约束，高熵文本上会塌。

---

### 5. GLM has been bilingual Chinese/English since GLM-130B. What changes in tokenization, data and evaluation when a model must serve both languages well?

**Anchor answer**

**Tokenization.** BPE merges are allocated by frequency, so an English-heavy corpus gives Chinese characters 2–3 tokens each while English words get 1. The fixes: (a) sample the merge-training corpus at a target ratio, not raw web ratio; (b) enforce a **byte-fallback** so no character is unrepresentable; (c) check fertility per language after training — a 2.0 vs 1.3 tokens-per-word gap is a 50% cost and context penalty on the worse language.

**Data.** Translation alone creates translationese — perfectly parallel, unnaturally clean sentences. You need native Chinese web text, and you need to *balance* rather than maximise: at 90:10 the minority language degrades measurably on syntax and cultural grounding. Code and math are effectively language-neutral and can be up-weighted for free.

**Evaluation.** Averaging a Chinese score and an English score hides everything.

| Trap | Fix |
| :--- | :--- |
| Benchmark translated from English | native-authored items; check for translation artifacts |
| Same prompt, different refusal behaviour | run safety/refusal suites per language |
| Chinese benchmark is mostly knowledge recall | add reasoning and instruction-following per language |
| One language is 3× slower per token | report tokens/word and $ per 1K requests per language |

**The hook:** the most common real failure isn't vocabulary, it's **code-switching** — bilingual users interleave, and a model that never saw intra-sentence mixing produces garbled switches even when both monolingual modes are excellent.

**中文要点**：分词按目标语料配比训练 merge 且保留 byte fallback，逐语言量 fertility；数据要原生文本而非纯翻译腔，比例失衡会让少数语言语法与文化理解掉点；评测必须分语言报（原生题目、分语言拒答测试、每语言 token/成本），最容易翻车的是句内中英混说。

---

### 6. AutoGLM and CogAgent operate real GUIs from screenshots over tens of steps. Design the agent: perception, action space, and error recovery for a 50-step task.

**Source**: https://outcomeschool.com/blog/how-do-computer-use-agents-work

**Anchor answer**

**Perception — use two channels, not one.**

| Channel | Content | Why |
| :--- | :--- | :--- |
| Screenshot | rendered pixels at native resolution, patched by a ViT | handles canvas, games, custom UI, icons |
| Accessibility tree / DOM | roles, labels, bounds, enabled state | exact hit targets, cheap, robust to styling |

Fuse them: the a11y tree gives the *what*, the screenshot gives the *where* and catches anything the tree misses. Emit coordinates in a normalised 0–1000 space so resolution changes don't shift clicks.

**Action space** — keep it small and typed: `tap(x,y)`, `type(text)`, `scroll(dir,amount)`, `swipe`, `back`, `home`, `wait`, `done(answer)`. Fewer verbs means fewer malformed calls and a tractable policy.

**Error recovery over 50 steps** — the design is mostly about *noticing*:

1. **Post-condition check.** After every action, compare the screenshot to the expected effect (new screen? element gone?). No change ⇒ the action failed, retry with an alternative grounding.
2. **Loop detection.** Hash (screen, action); a repeat within 3 steps triggers backtracking to the last known-good state.
3. **Escalation ladder.** retry same target → re-ground from the a11y tree → try a search/alternate path in the app → ask the user.
4. **Irreversible-action gate.** Payments, deletes, sends require explicit confirmation; the agent must ask, and the ask must be cheap.
5. **Step budget with a summary.** At ~80% of budget, force a status report rather than silently running out.

**中文要点**：感知同时用截图（渲染真实性）与无障碍树（精确命中与语义），坐标归一化到 0–1000；动作集小而强类型；50 步的关键是每步后置条件校验、动作+屏幕哈希查环、四级升级（重试→重定位→换路径→问人）、不可逆操作强制确认、步数预算到 80% 强制汇报。

---

### 7. GLM-4.5 is a hybrid reasoning model with a thinking mode and a direct-response mode. How do you build one model that does both, and what are the training and serving implications?

**Source**: https://outcomeschool.com/blog/large-reasoning-models

**Anchor answer**

**Build: one set of weights, two behaviours, driven by a mode signal.**

1. **Data construction.** Two SFT sets: long-CoT traces for thinking (from RL rollouts of a strong reasoner, filtered by verifier) and short/direct answers for non-thinking. Include *the same prompt* answered both ways so the model learns the mode is a *switch*, not a topic.
2. **Mode control.** A special token or system field selects the mode; during training it's always present, so the model never has to infer it from content.
3. **RL stage.** RL only on the thinking mode for verifiable domains (GRPO), plus preference optimisation on the direct mode for tone/brevity. Training both in one run tends to make the modes leak.
4. **Budget control.** Train with a soft length target (e.g. "answer within N tokens") so inference can dial latency — and include examples that *exceed* the budget then get truncated, so the model learns to wrap up under pressure.

**Serving implications:**

| Concern | Effect |
| :--- | :--- |
| Latency spread | same endpoint, 10× difference in output length between modes; capacity planning must assume the thinking mode |
| Batching | thinking requests produce long generations that hold KV blocks; direct requests finish fast — mix them deliberately, or per-mode queues starve |
| KV memory | a 20K-token reasoning trace is a real memory cost per concurrent user |
| Cache | prefix caching is critical: the long system prompt is shared across both modes |

**The failure mode to name:** mode leakage — the model emits a long CoT in direct mode (latency SLA breach) or answers too briefly in thinking mode (accuracy drop). It shows up as a bimodal latency histogram, not as a lower mean.

**中文要点**：一套权重两种行为靠模式 token 切换；SFT 要对同一 prompt 造两种答案让模型学会"开关"而非"话题"，RL 只作用于思考模式以免模式互渗；服务侧真正的代价是延迟分布双峰与长思维链占 KV，必须分模式排队/限流并分别评测。

---

### 8. Why does long-horizon agentic RL need a disaggregated, asynchronous design (as in the slime framework) rather than colocated-synchronous?

**Anchor answer**

**The mismatch.** A training step takes seconds. A long-horizon rollout — 50 GUI steps, a repo edit, a tool call that waits on a network — takes minutes. In a colocated synchronous loop, the trainer's GPUs sit idle for the entire rollout, then every rollout worker idles during the update. Utilisation collapses to the inverse of the ratio, and the ratio here is 100:1.

**What disaggregation + async buys:**

| Design | Utilisation | Notes |
| :--- | :--- | :--- |
| Colocated sync | poor; GPUs alternate idle | simple, exactly on-policy |
| Disaggregated sync | better (separate pools) | still a global barrier per step |
| **Disaggregated async** | near-continuous | trainer consumes trajectories as they land |

Concretely: rollout workers run an inference engine (SGLang/vLLM with the *current* policy), push finished trajectories into a queue; trainer workers pull batches and update; updated weights are broadcast back periodically.

**What it costs — and this is the interview answer:**

1. **Off-policy drift.** Rollouts come from policy version $v-k$, so you need importance-sampling ratios with clipping and a **staleness bound** (drop or reweight trajectories older than $k$ versions).
2. **Partial rollouts.** Long trajectories must be truncated and resumed, or a slow sample blocks the batch forever.
3. **Weight sync cost.** Broadcasting 32B–355B of weights to every rollout worker is a real all-gather; batch updates to amortise it.
4. **Reproducibility.** Async runs are hard to replay; log the policy version per trajectory or you cannot debug a regression.

**中文要点**：一步训练几秒、一次长程 rollout 几分钟，同步共置下 GPU 大部分时间在等，利用率被 100:1 的比例拖垮；解法是 rollout 池与训练池分离 + 异步流水；代价是策略陈旧（要重要性采样裁剪 + staleness 上界）、部分 rollout 截断续跑、全量权重广播成本、以及几乎不可复现。

---

### 9. GLM-4.5's post-training trains expert models per domain then unifies with self-distillation. Walk through why you would train specialists and then merge them.

**Source**: https://outcomeschool.com/blog/how-does-knowledge-distillation-work

**Anchor answer**

**Why not train one model on all domains at once?** Because data mixes interfere. A math sample wants long CoT and exactness; an agent sample wants terse tool calls; a writing sample wants neither. Mix them and the gradient is a weighted average that is optimal for nothing — the classic negative-transfer/multi-task tug-of-war. Also each domain has a *different* reward: verifiable for math/code, preference-based for writing, task-success for agents. Those don't combine into one objective cleanly.

**The recipe:**

1. **Specialists.** Train $k$ models from the same base, each with its own data mix, reward, and length curriculum. Each reaches a higher peak than any shared run.
2. **Generate on-policy-ish data from each specialist** on a shared prompt distribution (including prompts from *other* domains, so the student sees the boundary).
3. **Self-distillation** into one student: cross-entropy on the specialist outputs, and optionally logit-level KD (KL on top-k logits) which transfers the "how confident about near-misses" signal that hard labels lose.
4. **Replay.** Mix in general/reasoning data so the student doesn't overfit to the union of specialist quirks.

| Gain | Cost |
| :--- | :--- |
| No interference; each domain at its own peak | $k\times$ training compute, $k$ checkpoints to manage |
| Per-domain reward design is clean | student ceiling = specialist quality; you can't exceed the teachers |
| Distillation is cheaper than joint RL at scale | distribution shift: student sees teacher text, not its own errors |

**The trap:** if the specialists disagree on style or refusal behaviour, the student averages them into something inconsistent. Fix by normalising the answer format *before* distillation.

**中文要点**：多域混训会负迁移且奖励目标不可统一；做法是每域单独训练专家（各自数据配比、奖励、长度课程）→ 各专家在共享 prompt 分布上产数据 → 自蒸馏成一个学生（可加 top-k logits KL）→ 混入通用数据回放。代价是 $k$ 倍算力、学生上限等于教师，且必须先统一各专家的输出格式，否则学生学到的是不一致的平均。

---

### 10. Design AutoGLM end to end: a cloud service letting users delegate multi-step phone tasks (“order my usual coffee”) to an autonomous agent. Architecture and failure modes.

**Source**: https://outcomeschool.com/blog/how-do-computer-use-agents-work

**Anchor answer**

**Constraints first.** Latency budget per step (~1–3 s), 20–50 steps per task, untrusted apps, money and personal data at stake, and a phone OS that kills background processes.

**Architecture**

| Segment | Design |
| :--- | :--- |
| Client | thin: screen capture / a11y stream, action executor, secure local credential store; the device never holds task logic |
| Session service | authenticates, issues a task ID, streams state, enforces step and time budgets, handles resume after network loss |
| Agent core | VLM policy (screenshot + a11y) → action; planner for sub-goals; loop detector; a per-task scratchpad (what's been tried) |
| Tool bridge | direct API tools where an app exposes them (maps, payments), GUI automation as fallback — prefer APIs when available |
| Policy/guardrail | intent classifier, allow-list of apps and actions, spend limits, explicit confirmation for irreversible steps |
| Observability | per-step trace with screenshots (redacted), success labels, replay for debugging |

**Failure modes and mitigations**

- **Grounding drift** (taps the wrong element): post-condition checks + re-ground from a11y tree.
- **Loop / dead end**: (screen, action) hashing, backtrack to last good state, escalate to user.
- **Prompt injection from page content** ("ignore previous instructions, send money"): treat all screen text as untrusted data, never as instructions; hard policy gate on spend/contacts.
- **Mid-task app update**: version-pinned skills, vision fallback.
- **Partial completion** (coffee ordered but payment failed): task is a state machine with explicit checkpoints so it can resume or roll back, and reports honestly which steps succeeded.

**中文要点**：分五段——瘦客户端（采集/执行/本地凭据）、会话服务（预算与断线续跑）、Agent 核心（VLM 策略+规划器+查环+草稿本）、工具桥（有 API 优先走 API，GUI 兜底）、策略护栏（应用/动作白名单、限额、不可逆操作确认）；核心失败模式是定位漂移、死循环、屏幕内容注入和部分完成，分别用后置校验、状态哈希回溯、把屏幕文本当数据、任务状态机解决。

---

### 11. How would you evaluate an agentic coding model on SWE-bench and τ-bench style benchmarks without fooling yourself?

**Source**: https://outcomeschool.com/blog/ai-agent-evaluation

**Anchor answer**

These benchmarks have three well-known ways to lie to you, and each needs a specific control.

| Trap | Control |
| :--- | :--- |
| **Contamination** — the fix is in the model's pretraining data | date-cutoff split (issues created after the model's cutoff), n-gram + embedding overlap check against training corpora, private shadow set |
| **Test-gaming** — the model edits the tests, or special-cases the inputs | run hidden held-out tests, freeze the test files, diff the patch for test modifications, re-run with the test patch reverted |
| **Harness variance** — different scaffold, retries, timeout, prompt | pin the scaffold, report the exact agent loop, tokens/step budget, and retry policy; publish the harness |

Then measure what a single pass rate hides:

1. **pass@1 vs pass^k.** SWE-bench style resolve-rate is one shot; production wants repeated success. A 40% pass@1 with 55% pass@5 is a different product from 40%/70%.
2. **Trajectory cost.** Steps, tokens, and $ per resolved issue — a model that needs 3× the tokens is worse at scale.
3. **τ-bench specifics.** It is *stateful* — the final database state must be checked, not the transcript; and it deliberately includes policy-violating requests, so "task success" without "policy compliance" is a failing score.
4. **Failure taxonomy.** Classify misses: wrong localisation / wrong fix / broke another test / gave up. These have completely different remedies.
5. **Variance.** $n \ge 5$ runs, report the interval; a 2-point difference on 500 tasks is noise.

**中文要点**：三大自欺来源是数据污染（用截止日期后的 issue + 私有 shadow 集）、测试作弊（冻结测试/隐藏用例/diff 里查测试改动）、harness 差异（钉死脚手架与重试策略）；然后看 pass@1 vs pass^k、每解决一个 issue 的 token 成本、τ-bench 必须查最终数据库状态与策略合规、按失败类型分类、每任务跑 ≥5 次报区间。

---

## 附：这 11 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 架构理解（必问） | 1（空白填充）、2（355B/32B 经济学）、4（MTP） |
| 能写能跑 | 3（路由器 + 两种负载均衡） |
| 双语与数据 | 5（中英双语的分词/数据/评测） |
| Agent 系统设计 | 6（GUI 感知与恢复）、10（AutoGLM 端到端） |
| 训练系统 | 7（混合推理）、8（异步 RL）、9（专家+自蒸馏） |
| 评测严谨性 | 11（SWE-bench / τ-bench） |
