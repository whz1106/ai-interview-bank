# 18 · DeepSeek

> 10 题。出处：`README.en.md` § DeepSeek。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Walk me through DeepSeekMoE. How is it different from a standard top-2 MoE like Mixtral?

**Source**: https://outcomeschool.com/blog/mixture-of-experts

**Anchor answer**

DeepSeekMoE makes two structural changes to the classic top-2 MoE.

| Axis | Mixtral-style | DeepSeekMoE |
| :--- | :--- | :--- |
| Expert granularity | 8 coarse FFN experts, top-2 | Many fine-grained experts (e.g. 64–256), top-k=6–8 |
| Shared experts | none | 1–2 experts **always** active |
| Routing signal | top-2 gate softmax | top-k gate + per-expert bias (no aux loss in V3) |

**Why fine-grained:** with coarse experts each expert must cover a wide mixture of knowledge, so its parameters stay generic. Splitting the FFN into many small experts lets each one specialise on a narrower pattern, and a top-k combination of small experts approximates the target function more precisely per FLOP.

**Why a shared expert:** common knowledge (syntax, frequent function words, general reasoning) would otherwise be redundantly relearned inside every routed expert. Isolating it in an always-on expert frees the routed experts to be genuinely specialised — the paper's ablation shows routed experts become much more domain-skewed once the shared expert absorbs the common part.

**Costs:** more routing logic, capacity factor / token-dropping decisions, and an all-to-all dispatch whose latency can exceed the expert GEMM itself at small batch. On a single GPU it's strictly worse than dense — the win only appears when you can spread experts across devices.

**中文要点**：DeepSeekMoE = 细粒度专家（多而小）+ 共享专家（常开吸收通用知识）；换来更强的专家特化，代价是路由开销与 all-to-all 通信，单卡场景反而不如稠密。

---

### 2. DeepSeek-V3 uses auxiliary-loss-free load balancing. What was wrong with the auxiliary loss, and how does the bias trick work?

**Anchor answer**

**What was wrong:** the auxiliary loss is a second objective fighting the language-modelling gradient. Set the coefficient high and you get balanced routing but a measurably worse model (the loss pushes tokens toward under-used experts regardless of whether that expert is right). Set it low and routing collapses: a handful of experts win every batch, get more gradient, get better, and win more — a rich-get-richer loop that ends with most parameters dead.

**The bias trick:** keep a per-expert scalar $b_i$. It is added to the affinity score **only when choosing top-k**, not when computing the gate weight:

$$s_{i,t} = \text{softmax}(u_t^\top e_i),\quad \text{topk over } (s_{i,t} + b_i),\quad g_{i,t}=s_{i,t}\ \text{(unbiased)}$$

After each step, update by the sign of the load error: overloaded experts get $b_i \mathrel{-}= \gamma$, underloaded get $b_i \mathrel{+}= \gamma$. Because the update is sign-based and outside the gradient path, it acts as a slow integral controller on the argmax — it cannot distort the LM objective at all.

**The trap:** bias alone fixes *global* load but not *per-sequence / per-device* load. DeepSeek-V3 therefore still keeps one tiny sequence-wise auxiliary loss purely to stop intra-batch imbalance, which is what actually causes stragglers in expert parallelism.

**中文要点**：辅助损失会和语言建模梯度打架——权重大了掉点、小了路由坍塌。bias 只在选 top-k 时加、不进 gate 权重，用符号式更新做慢速积分控制；但它只管全局负载，仍需一个极小的 sequence-wise 损失防设备间不均衡。

---

### 3. What is multi-token prediction (MTP) and why train with it?

**Source**: https://outcomeschool.com/blog/decoding-deepseek-v4

**Anchor answer**

MTP adds $k$ lightweight prediction heads that each forecast a token further ahead. At depth $k$ the head does not just see $h_t$; it sees the hidden state of the previous depth **concatenated with the embedding of the token that was actually predicted there**, passed through a projection and one transformer block:

$$h_t^{(k)} = \text{Block}_k\big(\text{proj}([\,h_t^{(k-1)};\ \text{Emb}(x_{t+k})\,])\big),\qquad P(x_{t+k+1}\mid \cdot)$$

Training loss is the sum of the per-depth cross-entropies, each weighted by $\lambda/k$.

**What it buys:**

1. **Denser signal per token** — one forward pass now yields $k{+}1$ supervised targets, so data efficiency rises without more tokens.
2. **Look-ahead planning** — to predict two tokens ahead the representation must encode the *intent* of the sentence, not just the next surface form. This is the mechanism people hand-wave as "the model plans".
3. **Free speculative decoding** — the extra heads are a built-in draft model. DeepSeek-V3 reports roughly 1.8× decode TPS with 85–90% acceptance on the second token.

**Costs:** extra parameters and activations (V3 uses sequential, not parallel, MTP for exactly this reason), and the heads are dead weight at inference unless you actually run self-speculation — in which case you must keep them resident.

**中文要点**：MTP 用"上一深度的隐状态 + 真实下一 token 的 embedding"串行预测更远的 token，得到更密的监督信号和前瞻性表征，还能直接当自推测解码的 draft；代价是额外参数与显存，推理时若不投机解码就是纯负担。

---

### 4. Implement top-k MoE routing with a shared expert in PyTorch, and point out the efficiency and correctness traps.

**Source**: https://outcomeschool.com/blog/mixture-of-experts

**Anchor answer**

**Reference implementation**

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w1, self.w2, self.w3 = nn.Linear(d, h, False), nn.Linear(h, d, False), nn.Linear(d, h, False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoELayer(nn.Module):
    def __init__(self, d, h, n_routed=64, top_k=6, n_shared=2):
        super().__init__()
        self.top_k, self.n = top_k, n_routed
        self.gate = nn.Linear(d, n_routed, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_routed), requires_grad=False)  # load-balance only
        self.routed = nn.ModuleList([Expert(d, h) for _ in range(n_routed)])
        self.shared = nn.ModuleList([Expert(d, h) for _ in range(n_shared)])

    def forward(self, x):                       # [T, d]
        T, d = x.shape
        logits = self.gate(x)                   # [T, n]
        scores = logits.sigmoid()
        idx = (scores + self.bias).topk(self.top_k, dim=-1).indices
        w = scores.gather(1, idx)
        w = w / w.sum(-1, keepdim=True)         # normalise AFTER top-k

        out = torch.zeros_like(x)
        flat = idx.reshape(-1)
        for e in range(self.n):                 # group by expert: one GEMM per expert
            tok, slot = (flat == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            src = tok // self.top_k
            y = self.routed[e](x[src]) * w[src, slot].unsqueeze(-1)
            out.index_add_(0, src, y)
        for e in self.shared:
            out = out + e(x)
        return out
```

**Key points**

- Normalise gate weights **after** top-k, not before; pre-normalising changes the relative scale between experts.
- `index_add_` accumulates in non-deterministic order — outputs differ run to run; use `float32` accumulation, or segment-sum per expert for bitwise reproducibility.
- Don't `torch.stack` per-expert outputs into a `[T, k, d]` tensor in production; it materialises $k\times$ activations and destroys the grouped-GEMM batching.
- The trap that bites: sorting tokens by expert on the host. Keep the permutation on-device, and pad each expert's bucket to a multiple (otherwise ragged GEMMs waste 20–40% of the tile).
- Shared expert must bypass the router entirely — routing it "sometimes" silently reintroduces the redundancy it exists to remove.
- Cast scores to `float32` before `topk`: in BF16 the cutoff ties are broken arbitrarily, which surfaces as training instability rather than as an error.

**中文要点**：先 top-k 再归一化；`index_add_` 是非确定性累加；别 stack 出 $[T,k,d]$，要按专家分桶做 grouped GEMM 并对齐 pad；共享专家必须完全绕过路由。

---

### 5. Sketch how you would serve a 671B-parameter MoE model with low latency under GPU-memory constraints.

**Source**: https://outcomeschool.com/blog/llm-inference-optimization

**Anchor answer**

**The arithmetic first.** 671B params in FP8 ≈ 671 GB of weights. On 80 GB cards that is ≥ 9 GPUs before a single KV byte — so the question is not "can one GPU hold it" but "how few devices, and how do I keep them busy".

| Layer | Decision | Why |
| :--- | :--- | :--- |
| Weight precision | FP8 blockwise, BF16 only for norms/gate/head | halves HBM, ~2× decode speed on Hopper |
| Parallelism | EP across nodes + MLA-compressed KV; no TP inside a node where avoidable | MoE weights shard cleanly by expert; TP adds all-reduce per layer for little gain |
| Topology | prefill pool (compute-bound) split from decode pool (bandwidth-bound) | the two phases want opposite batch shapes; colocating them wastes both |
| KV | MLA latent cache (~70 KB/token in V3-class models) + prefix caching | at 100K context KV dwarfs weights if you use MHA |
| Batching | continuous batching, chunked prefill, decode micro-batches per expert bucket | keeps the all-to-all off the critical path |
| Hot experts | replicate the top few experts on multiple ranks | routing is Zipfian; one hot expert becomes the whole step's latency |

**Failure modes to name out loud:** all-to-all dominates at small batch (so per-user latency and per-GPU throughput are in direct conflict); a single straggler rank stalls every other rank; expert replication blows up memory you already didn't have; and long prompts make TTFT large enough that you must cache prefixes or the p99 is unrecoverable.

If the SLO is interactive chat, prefer 2–4× more GPUs with a leaner per-GPU footprint over squeezing onto the minimum.

**中文要点**：671B FP8 权重就要 671GB，先算清最少卡数；再做 EP + MLA 压 KV、prefill/decode 分离、连续批处理、热专家复制；核心矛盾是小 batch 下 all-to-all 主导延迟——单用户延迟与整机吞吐天然对立，宁多给卡别硬挤。

---

### 6. R1-Zero was trained with RL and essentially no SFT first. What did that show, and why did full R1 add SFT back?

**Source**: https://outcomeschool.com/blog/large-reasoning-models

**Anchor answer**

**What R1-Zero showed:** starting from a base model and running pure outcome-reward RL (GRPO on verifiable math/code) produces long chain-of-thought *by itself* — self-verification, backtracking, and the much-quoted "aha moment" where the model re-reads its own reasoning. Nobody wrote those traces. That is the real result: **reasoning behaviour is reachable by reward pressure alone**, so behaviour cloning from a bigger reasoner is not a prerequisite.

**What it also showed:** the output is unusable as a product. Language mixing (Chinese inside English answers), unreadable formatting, no consistent answer template, and a cold-start distribution problem — early rollouts are so bad that the reward signal is nearly zero for a long time, which is both slow and fragile.

**Why R1 added SFT back (four stages):**

1. **Cold-start SFT** — a few thousand long-CoT examples to fix format and language consistency, giving RL a sane starting policy.
2. **Reasoning RL** — same verifiable-reward RL, now on a readable base.
3. **Rejection-sampling SFT** — ~600K filtered traces (including non-reasoning data) to restore general capability the RL phase eroded.
4. **Final RL across all domains** — helpfulness + safety, not just correctness.

The lesson: pure RL proves the mechanism, but format and language discipline are cheaper to install with data than to discover by reward.

**中文要点**：R1-Zero 证明纯 RL 就能自发长出长思维链（自我验证、回溯、"aha"），但可读性和语言混杂不可用；R1 用 cold-start SFT 修格式、RL 提推理、拒绝采样 SFT 找回通用能力、再全域 RL。格式用数据教比用奖励学便宜得多。

---

### 7. FP8 training at 671B scale is hard. What actually breaks in low precision, and how do you make it stable?

**Anchor answer**

FP8 has 3 mantissa bits (E4M3) and ~2 decimal digits of precision. Four distinct things break, and they need four distinct fixes.

| What breaks | Symptom | Fix |
| :--- | :--- | :--- |
| Activation outliers | a few channels are 100× larger than the rest; one tensor scale crushes everything else to zero | **blockwise/tile scaling** — 1×128 tiles for weights, 128×128 blocks for activations |
| Accumulation error | GEMM outputs drift; loss curves get noisy at long horizons | **promote every $N_C$ elements (e.g. 128) to FP32** in CUDA cores, not just at the end |
| Gradient underflow | backward pass multiplies small numbers; FP8 flushes to zero and learning stalls silently | keep the **master weights + optimizer states in BF16/FP32**, quantize only the GEMM operands |
| Sensitive ops | layernorm, embedding, router gate, output head, residual adds degrade immediately | keep those **BF16/FP16** — they are a rounding error of total FLOPs |

Plus **online scaling**: compute the per-tensor/tile amax each iteration (delayed or current-batch) rather than freezing a scale from calibration; the distribution moves during training.

**The bit people forget:** the loss curve looks fine while the model is quietly worse. You only see it in eval, weeks later, so you need an ablation at a small scale that compares FP8 vs BF16 all the way to convergence — not just 2K steps of loss parity. And FP8 gains are Hopper/Blackwell-specific; on older hardware the conversion overhead eats the win.

**中文要点**：FP8 坏在四处——激活离群值（分块缩放）、累加误差（每 128 元素升 FP32）、梯度下溢（主权重与优化器状态留 BF16/FP32）、敏感算子（norm/gate/head 保持高精度）。loss 看不出来，必须小规模跑到收敛再比对。

---

### 8. How do you build a training dataset without triggering model collapse when much of your data is synthetic?

**Anchor answer**

Model collapse is a *distribution* problem, not a data-volume problem: recursion on your own output narrows the tails, and after a few generations the model can no longer represent rare events it once handled.

| Lever | Concrete rule |
| :--- | :--- |
| Anchor with real data | never drop below ~10–30% verified human data in any mix; it holds the tails |
| Generate with a **verifier**, not free-form | rejection-sample against unit tests / numeric answers / retrieval-grounded facts; unverifiable generations are the ones that poison the set |
| Diversify the generator | multiple teachers, multiple prompts, temperature spread — one model's mode collapse becomes everyone's |
| Dedupe ruthlessly | near-dup synthetic text is what actually amplifies a mode; MinHash + embedding dedupe |
| Keep the labels human where it matters | safety, refusals, domain facts |
| Accumulate, don't replace | keep the previous real corpus in every round instead of overwriting |
| Measure drift | track n-gram/embedding divergence vs the real corpus, plus held-out human eval per round |

**The failure mode I would call out in an interview:** teams measure "does the synthetic data help on the target task" and skip "does it hurt on the long tail". The damage shows up as degraded calibration, degraded low-resource-language performance, and a model that gets *more* confident and *less* correct. So keep a "canary" eval set of rare-but-real examples and gate every data refresh on it.

**中文要点**：坍塌是分布问题不是数量问题——尾部被抹平。做法：保底 10–30% 真实数据、用验证器（单测/数值/检索）做拒绝采样而非自由生成、多教师多样本、强去重、累积而非替换、并用长尾 canary 集卡住每次数据刷新。

---

### 9. DualPipe overlaps computation and communication in training. Why is that overlap the whole game at this scale, and what is the trade-off?

**Anchor answer**

At 671B with expert parallelism, every MoE layer needs two all-to-all exchanges (dispatch tokens to experts, combine results back), plus PP/DP/EP gradients. Those exchanges are **latency- and bandwidth-bound, not FLOP-bound** — and they sit *between* compute kernels, so naively they are pure dead time, easily 30–50% of the step.

DualPipe's idea: schedule **two micro-batch directions simultaneously** (forward from the start of the pipeline and backward from the end), and inside each chunk split into attention / dispatch / MLP / combine. Each rank then has, at any instant, both a compute-heavy and a comm-heavy block in flight, so the NIC works while the tensor cores work.

| Gain | Cost |
| :--- | :--- |
| comm hidden behind compute ⇒ MFU up sharply | **~2× parameter memory**: both directions must hold their own copy of weights and activations |
| fewer pipeline bubbles (bidirectional fill/drain) | schedule is hand-written per PP degree; changing cluster shape means rewriting it |
| no need for extreme PP (which would shrink per-GPU batch) | requires NVLink intra-node + fast IB inter-node, or the overlap can't keep up |

**The honest trade-off:** DualPipe is a bet that you have memory to spare and communication you cannot avoid. On a memory-constrained cluster, or at a scale where the all-to-all already fits inside the compute window, the extra copies cost you more than the overlap saves. It is also brittle: one straggler rank and every overlap collapses into serialization.

**中文要点**：MoE 的两次 all-to-all 是纯死时间，DualPipe 用双向微批把计算块和通信块同时放进流水，把通信藏进计算里换 MFU；代价是约 2× 参数内存、需按集群形状手写调度、强依赖 NVLink/IB，一个慢 rank 就让重叠崩成串行。

---

### 10. DeepSeek claims frontier-class results at a fraction of the usual training cost. If an interviewer asks “how is that even possible,” what is your structured answer?

**Anchor answer**

Don't answer "cheap GPUs". Decompose into five independent levers, then separate *technical* savings from *accounting*.

| Lever | Mechanism | Where it shows up |
| :--- | :--- | :--- |
| Architecture | MLA shrinks KV ~5–10× vs MHA; 37B active of 671B means ~18× less compute per token than a dense model of that size | both training and serving |
| Precision | FP8 GEMMs with blockwise scaling ⇒ ~2× throughput on the same silicon | training step time |
| Scheduling | DualPipe + EP hides all-to-all; high MFU instead of idle NICs | wall-clock |
| Post-training | RL with verifiable rewards (GRPO) — no reward model to train, no preference-label pipeline | the RL stage |
| Method | small-scale ablations decide the recipe *before* the big run, so you don't pay for a failed 671B attempt | avoided cost |

Then the accounting caveat, which is the part that earns credibility: the widely quoted figure covers **one training run's GPU-hours at market rental rates**. It excludes prior research, ablation runs, failed attempts, data pipeline, salaries, and cluster capex. Architecture search is the expensive part; the final run is the cheap part.

The mature framing: "the claim is real but narrow — it's an *efficiency* result, not a claim that frontier models cost $5M to build."

**中文要点**：分五层拆——MLA 压 KV + 稀疏 MoE 降每 token 算力、FP8 提吞吐、DualPipe 藏通信提 MFU、RLVR 省掉奖励模型、小规模消融先定配方；同时点明那个数字只含单次训练的 GPU 租用费，不含前期研究、失败尝试与集群资本开支。

---

## 附：这 10 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 架构深度（必问） | 1（DeepSeekMoE）、2（无辅助损失负载均衡） |
| 能写能跑 | 4（MoE 路由实现）、9（DualPipe 取舍） |
| 训练系统 | 7（FP8 稳定性）、8（合成数据防坍塌） |
| 推理部署 | 5（671B 服务） |
| 认知与表达 | 3（MTP）、6（R1-Zero）、10（成本叙事） |
