# 13 · Google DeepMind and Google AI

> 22 题。出处：`README.en.md` § Google DeepMind and Google AI。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. You are receiving an unbounded stream of event IDs. Return the k most frequent IDs seen so far, at any point, with bounded memory.

**Anchor answer**

Exact counting is impossible with bounded memory once the key space is unbounded — you must accept approximation, and the interview answer is to name the guarantee you're buying.

| Approach | Memory | Guarantee |
| :--- | :--- | :--- |
| Hash map + heap | O(distinct) — **unbounded**, fails the requirement | exact |
| **Misra–Gries** (`k-1` counters, decrement-all-on-miss) | O(k) | **never overestimates**; true freq − estimate ≤ N/k |
| **Space-Saving** (k counters, replace the min) | O(k) | always overestimates; error ≤ N/k, and it *keeps a candidate ID*, so you can report the label |
| Count-Min Sketch + heap | O(width) | overestimates by sketch error, and heap keys can drift |

For "top-k IDs" you want **Space-Saving**: it maintains k (id, count) pairs, and on a miss evicts the minimum counter, inheriting its count + 1. Misra–Gries is a frequency *estimator*; Space-Saving is a heavy-hitter *identifier* — that distinction is the whole point.

Two traps: (1) the guarantee is on **total mass**, so with a long tail of singletons, N/k can exceed the true frequency of the k-th item and the result is genuinely meaningless — check whether N/k < f_k before trusting it. (2) "at any point" means you must not decay; if the stream is non-stationary, use a sliding window or exponential decay and say so explicitly.

**中文要点**：有界内存下精确 top-k 不可能。选 Space-Saving（保 ID、可高估、误差 ≤ N/k）而非 Misra–Gries（保计数、不高估）；先检查 N/k 是否小于第 k 名真实频次，否则结果无意义。

---

### 2. Closest key: given a dictionary with letter keys and lists of letters as values, find the closest key.

**Anchor answer**

The phrasing is deliberately under-specified; the graded move is to state your reading, then solve the cheap version first. My reading: keys are letters, values are the letters each key "covers" (a keypad/alias table), and "closest" means the key that covers the query letter, falling back to nearest by distance when nothing covers it.

**Reference implementation**

```python
import bisect
from collections.abc import Iterable

def build_index(table: dict[str, Iterable[str]]) -> dict[str, str]:
    idx = {}
    for key, covered in table.items():
        for ch in covered:
            idx.setdefault(ch, key)      # first writer wins: deterministic
    return idx

def closest_key(table, query: str) -> str | None:
    idx = build_index(table)
    if query in idx:
        return idx[query]                      # O(1) exact hit
    if not idx:
        return None
    keys = sorted(table)                       # fallback: lexicographic neighbour
    pos = bisect.bisect_left(keys, query)
    cands = [k for k in (keys[pos - 1] if pos else None,
                         keys[pos] if pos < len(keys) else None) if k]
    return min(cands, key=lambda k: (abs(ord(k[0]) - ord(query)), k))
```

**Key points**

1. **Invert once, query many times** — building `ch → key` turns a linear scan into O(1); rebuilding per call is the classic inefficiency.
2. Duplicate coverage is the real design question: if `'a'` appears under two keys, you need a documented tie-break (first, last, or error) — silently taking `setdefault` order is how keypad bugs ship.
3. If "closest" means edit distance, say that the fallback is O(len) per query and cache failures; don't silently degrade into a scan for every miss.
4. Test empty values, duplicate letters, and a query with no neighbour.

**中文要点**：先把 "closest" 定义清楚：精确命中走倒排索引 O(1)，未命中按字母距离回退。重复覆盖必须有明确 tie-break；建索引一次、查询多次。

---

### 3. Write a function to compute root-mean-square error given y_pred and y_true lists.

**Anchor answer**

**Reference implementation**

```python
def rmse(y_true, y_pred, weights=None) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("length mismatch")
    n = len(y_true)
    if n == 0:
        raise ValueError("empty input")
    if weights is None:
        s = 0.0
        for t, p in zip(y_true, y_pred):
            d = t - p
            s += d * d
        return (s / n) ** 0.5
    wsum = 0.0
    s = 0.0
    for t, p, w in zip(y_true, y_pred, weights):
        d = t - p
        s += w * d * d
        wsum += w
    return (s / wsum) ** 0.5
```

**Key points**

1. **`math.fsum` or accumulate in float64** — squaring residuals of ~1e6 overflows float32 and a naive `sum()` drifts on long lists; use `math.fsum(d*d for ...)` if you care.
2. **`sqrt(mean(d²))`, not `mean(sqrt(d²))`** — the latter is MAE. Candidates swap these under pressure.
3. **Weighted RMSE divides by Σw**, not n.
4. **Don't** use `numpy` if the question says lists and the environment is bare; do mention that vectorised numpy is what you'd ship.
5. Edge cases worth naming: NaN in `y_pred` (should it propagate or be masked?), empty input (ZeroDivisionError vs ValueError), and **denominator choice** — for a fitted model, `n - p` gives the unbiased residual variance; RMSE as a *metric* conventionally uses `n`, and mixing the two silently biases model comparisons.

Report RMSE next to MAE: RMSE punishes outliers quadratically, so a large gap between them is a diagnostic that a few points dominate your error.

**中文要点**：实现是 `sqrt(mean(d²))` 不是 `mean(|d|)`；加权版除以 Σw；长列表用 fsum 防漂移。指标用 n 作分母，估计残差方差才用 n-p，混用会悄悄带偏模型比较。

---

### 4. Parse bigrams: extract two-word phrases from strings for NLP feature engineering.

**Anchor answer**

**Reference implementation**

```python
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

def bigrams(text: str):
    toks = TOKEN_RE.findall(text.lower())
    for a, b in zip(toks, toks[1:]):
        yield f"{a}_{b}"

def top_bigrams(texts, k=20, min_count=2):
    c = Counter()
    for t in texts:
        c.update(bigrams(t))          # streaming: no full materialisation
    return [(bg, n) for bg, n in c.most_common() if n >= min_count][:k]
```

**Key points**

1. **Punctuation and sentence boundaries.** "end. Start" must not produce `end_start`. Split on sentence terminators first, or keep a sentinel token and drop bigrams containing it — this bug quietly pollutes every n-gram model.
2. **Don't `text.split()`** — it keeps `"don't,"` and `"co-op"` intact differently than you expect, and does nothing useful for CJK (no spaces, so you need character bigrams or a segmenter).
3. **Use a separator that cannot appear in tokens** (`_` after stripping `_`, or a tuple key) or `("new","york")` and `("new_york","")` collide.
4. **Streaming with `Counter.update(generator)`** keeps memory O(distinct bigrams) instead of O(corpus); with 10⁸ docs, materialising a list per document is the difference between fitting and OOM.
5. For features, `min_count` pruning plus hashing (2²⁰ buckets) beats a full vocabulary; and **TF-IDF weighting matters more than the raw count** for downstream models.

**中文要点**：先切句再切词，否则 "end. Start" 会生成假 bigram；别用 split，CJK 要字符级；key 用元组或安全分隔符避免碰撞；流式 Counter 更新，min_count 剪枝 + hashing。

---

### 5. Define the bias-variance trade-off and discuss the relationship between the two.

**Anchor answer**

Expected squared error at a point decomposes exactly:

$$\mathbb{E}[(y-\hat f)^2]=\underbrace{\sigma^2}_{\text{irreducible}}+\underbrace{(\mathbb{E}\hat f-f)^2}_{\text{bias}^2}+\underbrace{\mathbb{E}[(\hat f-\mathbb{E}\hat f)^2]}_{\text{variance}}$$

- **Bias** = error from the hypothesis class being too rigid (linear model on a curved boundary). It does not shrink with more data.
- **Variance** = error from sensitivity to the particular sample (deep tree, 1-NN). It shrinks as n grows, roughly as O(1/n) for parametric, slower for nonparametric.

They are **not independent** — that's the part most candidates miss. They are coupled through model capacity and through the *number* of effective parameters: adding capacity typically lowers bias and raises variance along a frontier, but the trade-off is not a law — modern overparameterised nets show **double descent**, where variance falls again past the interpolation threshold, and explicit regularisation (weight decay, dropout, early stopping, data augmentation, ensembling) can reduce variance *without* paying bias.

Practical reading: you cannot measure bias (you never see $f$), so you proxy it with **train error** and variance with the **train–validation gap**. A big gap = variance problem → more data/regularisation. Both high = bias problem → more capacity/features.

**中文要点**：误差 = 噪声 + 偏差² + 方差。偏差来自假设空间太窄、不随数据增长消失；方差随 n 下降。二者通过容量耦合，但过参数化会出现 double descent，正则化可只降方差——用训练误差当偏差代理、训练-验证差当方差代理。

---

### 6. What are the assumptions of linear regression?

**Anchor answer**

Split them by what they buy you, because that is the question behind the question:

| Assumption | Needed for | Violation symptom | Fix |
| :--- | :--- | :--- | :--- |
| **Linearity in parameters** | the model to be unbiased | curvature in residual plots | splines, interactions, transform y |
| **Exogeneity** $\mathbb{E}[\varepsilon\mid X]=0$ (no omitted confounders) | **causal/coefficient interpretation** | coefficients flip sign when a variable is added | instruments, panel FE, experiment |
| **No perfect multicollinearity** | $X^\top X$ invertible | singular matrix, exploding SEs | drop/combine, ridge |
| **Homoscedasticity** $\mathrm{Var}(\varepsilon\mid X)=\sigma^2$ | OLS is *efficient*; classical SEs valid | funnel shape | HC3 robust SE, WLS, log-y |
| **Independent errors** | SEs, not point estimates | autocorrelation in time series | clustered SE, GLS, AR terms |
| **Normality of residuals** | exact t/F tests and CIs in small samples | QQ plot tails | bootstrap, or rely on asymptotics |

The key insight: **only the first three affect the fitted values.** Heteroscedasticity and non-normality leave $\hat\beta$ unbiased and consistent; they corrupt *inference* (p-values, CIs). Gauss–Markov gives BLUE from linearity + exogeneity + homoscedasticity + no autocorrelation — normality is not required for BLUE, only for exact finite-sample tests.

Interview trap: claiming "residuals must be normal for regression to work." For n in the thousands, the CLT makes it irrelevant for coefficients; it matters for prediction intervals on a single point.

**中文要点**：六条假设按"影响预测还是影响推断"分组。只有线性、外生性、无完全共线性影响拟合值；异方差/非正态只毁 p 值和置信区间。Gauss-Markov 不需要正态性；系数解释必须靠外生性（无遗漏变量）。

---

### 7. Distinguish regularization from validation: when is each the right tool?

**Source**: https://outcomeschool.com/blog/regularization-in-machine-learning

**Anchor answer**

They answer different questions. **Regularisation constrains the hypothesis** (changes what the model *can* fit); **validation estimates generalisation** (changes what you *believe* about the model). One is a prior, the other is a measurement.

| | Regularisation | Validation |
| :--- | :--- | :--- |
| Effect on fit | shrinks/changes parameters (L2 pulls to 0, L1 sparsifies, dropout injects noise) | none — read-only |
| Uses data | training data | held-out data |
| Failure it prevents | overfitting *and* ill-posedness ($p \gg n$, separable data) | **self-deception**: shipping a model that only looks good |
| Failure it causes | underfitting when λ is too large; L1 unstable with correlated features | leakage → optimistic estimate; small folds → high-variance decisions |
| Chosen by | validation (λ is a hyperparameter) | nothing — it is the referee |

The dependency is the answer: **you cannot tune regularisation without validation**, so "which is right" is a false dichotomy — validation is how you *select* the regulariser. And regularisation is not only for overfitting: with 200 samples and 5,000 features, an unregularised model has no unique solution; ridge restores one, and that is a structural need, not a variance fix.

When **not** to regularise: when the bias it injects exceeds the variance it removes — typically large n, low capacity, or when you need unbiased coefficients for causal claims (regularised estimates are biased by construction, so use debiased/orthogonal methods for inference).

Validation discipline: nested CV, group/time splits, and never tuning on the test set.

**中文要点**：正则化改变模型能拟合什么，验证只测量泛化——所以不是二选一，λ 必须靠验证来选。正则化不只治过拟合，p≫n 时它让解唯一存在；但要做因果系数解释时正则化带来的偏差不可接受，需用去偏方法。

---

### 8. Derive the gradient of cross-entropy loss with softmax inputs, and explain why we fuse them numerically.

**Source**: https://outcomeschool.com/blog/math-behind-cross-entropy-loss · https://outcomeschool.com/blog/math-behind-backpropagation

**Anchor answer**

With logits $z\in\mathbb{R}^{K}$, $p_i = e^{z_i}/\sum_j e^{z_j}$, and target $y$ (one-hot, index $c$):

$$L=-\log p_c = -z_c + \log\sum_j e^{z_j}$$

$$\frac{\partial L}{\partial z_i} = -\mathbb{1}[i=c] + \frac{e^{z_i}}{\sum_j e^{z_j}} = p_i - y_i$$

So the gradient is just **prediction minus label**. That cleanliness is why every framework ships a fused op: it is the reason softmax+CE is numerically delicate on its own but trivial when fused.

Why fuse: computing $p$ first then $\log p$ loses precision. If $p_c$ underflows to 0 (logit gap of ~-90 in float32), $\log 0 = -\infty$ and the gradient becomes NaN. Fusing uses the **log-sum-exp** identity:

$$\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j-m},\quad m=\max_j z_j$$

which keeps every exponent ≤ 0, so no overflow, and the loss stays finite for arbitrarily confident wrong predictions.

| Separate | Fused |
| :--- | :--- |
| materialise $p$, risk underflow to 0 | logits shifted by max, no overflow |
| $\log(0)$ → inf/NaN at high confidence | finite loss, bounded gradient |
| extra memory traffic, extra kernel | single kernel, backward reuses saved max |

**中文要点**：softmax+交叉熵的梯度就是 `p - y`；数值上必须用 log-sum-exp（减去 max）融合计算，否则高置信度错分时 p 下溢成 0，log 0 变 NaN，梯度爆掉。

---

### 9. Explain the SVD and give two places it shows up in modern deep learning.

**Anchor answer**

For any $A\in\mathbb{R}^{m\times n}$, $A=U\Sigma V^\top$: orthonormal left/right singular vectors, non-negative singular values sorted descending. It is the generalisation of the eigendecomposition to rectangular matrices, and the best rank-$r$ approximation in Frobenius/spectral norm is $A_r=\sum_{i\le r}\sigma_i u_i v_i^\top$ (Eckart–Young). Cost O(mn·min(m,n)); randomised SVD gets most of the value in O(mn·log r).

Two places it is load-bearing in modern DL:

1. **LoRA / low-rank adapters.** The empirical finding that fine-tuning updates have low *intrinsic rank* is why $\Delta W = BA$ with $r \ll \min(d,k)$ matches full fine-tuning on many tasks; you are literally restricting the update to a rank-$r$ subspace, with ~1% of the parameters and no added inference latency once merged.
2. **Compression and analysis.** Truncated SVD of embedding matrices and of FFN weights gives low-rank factorisation; spectral normalisation uses $\sigma_{\max}$ (power iteration) to bound the Lipschitz constant of a discriminator; PCA-whitening of embeddings via SVD fixes anisotropy that otherwise ruins cosine similarity — a very common silent bug.

A third, common in research: head/representation analysis — projecting activations onto top singular directions to show that models encode a low-dimensional subspace (e.g. concept directions, refusal directions).

Caveat: SVD is *linear* and variance-optimal, not task-optimal — truncating by explained variance can destroy exactly the low-variance direction that carries the label. Use task-aware selection when the signal is subtle.

**中文要点**：SVD 是矩形矩阵的特征分解，Eckart-Young 保证截断是最优低秩近似。DL 里主要出现在 LoRA（更新本就低秩）与压缩/谱归一化/PCA 白化；注意按方差截断不等于按任务截断。

---

### 10. On average, how many fair coin flips until you see two heads in a row? Walk me through it.

**Anchor answer**

Answer: **6**. The clean way is a two-state Markov chain, states = length of current run of heads.

Let $E_0$ = expected further flips from "no active run", $E_1$ = from "one H just seen".

$$E_1 = 1 + \tfrac12(0) + \tfrac12 E_0$$
$$E_0 = 1 + \tfrac12 E_1 + \tfrac12 E_0$$

From the first, $E_1 = 1 + E_0/2$. Substitute into the second:

$$E_0 = 1 + \tfrac12(1 + E_0/2) + \tfrac12 E_0 \;\Rightarrow\; \tfrac14 E_0 = \tfrac32 \;\Rightarrow\; E_0 = 6$$

Sanity check with the general pattern-overlap formula for a fair coin and pattern of length $k$: $E = 2^{k+1}-2 = 2^3-2 = 6$. For HTH (self-overlapping prefix of length 1) it is 10, not 8 — which is the point of asking: **overlap changes the answer**, and the state machine handles it automatically because a new H can start a fresh run without going back to $E_0$.

The trap: answering "4, because 2 flips per pair" ignores that a failed attempt (HT, TT) does not necessarily reset progress — and that the two trials overlap. The other trap is stopping at $E_1$ and forgetting to weight by how often you're in state 0.

**中文要点**：答案 6。用"当前连 H 长度"两个状态建期望方程，得 E0=6；通式 2^(k+1)−2。易错点是按"每两枚一组"算成 4，忽略窗口重叠与失败后状态不一定清零。

---

### 11. When would you choose Q-learning over policy gradients, and vice versa?

**Anchor answer**

Decision rule: **Q-learning when the action space is small/discrete and you can afford to explore; policy gradients when actions are continuous, stochastic policies are needed, or only trajectories are available.**

| Dimension | Q-learning (off-policy, value-based) | Policy gradient (on-policy) |
| :--- | :--- | :--- |
| Data efficiency | **high** — replay buffer reuses old transitions | low — data from the current policy only |
| Action space | discrete/small (needs `argmax` over actions) | continuous, high-dim, structured |
| Policy type | deterministic/greedy; cannot represent stochastic optima | native stochastic policy |
| Convergence | off-policy + function approximation + bootstrapping = **deadly triad**, can diverge | local optima, but stable-ish |
| Failure mode | **Q overestimation** (max of noisy estimates) → DQN's double-Q fix | **high variance** → baselines, GAE, large batches |
| Offline data | usable (with care, + CQL/IQL for distribution shift) | not directly |

Practical notes: DQN on Atari works with a replay buffer of ~1M frames; PPO on continuous control needs on the order of millions of environment steps but tolerates reward shaping poorly. For LLM post-training you almost always use PG (PPO/GRPO) because the action space is a vocabulary and the policy must stay stochastic to explore; Q-learning appears in offline RL and in value-guided search (e.g. process reward models).

When **not** to use either: if a model of the environment exists and is cheap, use planning/DP or MCTS — sample efficiency beats both.

**中文要点**：动作空间小且离线数据可复用 → Q-learning（配 Double Q 防高估）；连续/需要随机策略/LLM 词表级动作 → 策略梯度（PPO/GRPO，配 baseline 降方差）。有可用环境模型时优先规划而非 RL。

---

### 12. You have a binary loan-approval classifier and limited access to feature weights. How do you explain a rejection?

**Anchor answer**

A loan rejection is a **regulated adverse-action notice**, not an ML demo: the applicant must be told *why*, in actionable terms, and the reason must be truthful. So the goal is not "explain the model" — it is "produce a small set of defensible reason codes."

Layer the approach:

| Level | Technique | Use |
| :--- | :--- | :--- |
| Model-native | logistic coefficients × standardised feature values | exact, cheap, defensible — **prefer this if the model is linear** |
| Local surrogate | LIME / SHAP (KernelSHAP) on the decision | model-agnostic, need a background distribution; unstable across runs |
| Counterfactual | "raise income by $X or reduce DTI below 43%" | most actionable for the applicant |
| Global constraint | monotonic constraints at training time (income ↑ ⇒ never worse) | prevents unexplainable non-monotone behaviour |

If weights are unavailable (black box, vendor model), use **SHAP with a small background set** and, crucially, **report the top 2–3 features only** — long reason lists turn into noise and, in the US, are legally insufficient (ECOA/Reg B wants specific principal reasons).

Two traps: (1) explaining the *score* but not the *threshold* — the applicant needs the gap, not just the drivers; (2) correlated features splitting credit, so "income" and "employment length" each look small — group them before ranking. Also verify **calibration**: a rejection should correspond to a real default probability, not just a low rank. And audit reason codes across protected groups; if the top reason differs systematically by group, that is a fair-lending finding.

**中文要点**：拒绝信是合规文件，要"可行动的具体理由码"而非完整解释。线性模型直接系数×标准化值；黑盒用 SHAP 取前 2-3 个并做特征分组；给反事实差距；先验校准，再按受保护群体审计理由分布。

---

### 13. Your pretraining loss suddenly diverges at step 300k of a long run. Diagnose and fix it.

**Anchor answer**

Step 300k of a long run means you are **not** in warmup: the usual suspects are data, numerics, and attention entropy — in that order, because data is free to check.

Triage sequence:

1. **Is it a spike or a climb?** A single spike that recovers in <100 steps is common; log `grad_norm`, per-layer activation norms, and loss per micro-batch. A persistent climb is a real divergence.
2. **Find the bad data.** Hash-scan the shard around that step: duplicated documents, empty/HTML-only text, wrong tokenizer version, a corpus with a different distribution mixed in. A single repeated document can cause a loss spike because the model overfits it within a few hundred steps.
3. **Numerics.** bf16 is hard to overflow but fp8/fp16 with static scaling does; check for `inf` in the loss scale, and add a **z-loss** on logits to stop logit growth.
4. **Attention pathology.** Monitor max attention logit and entropy; "massive activations" in a few dimensions are a known precursor. Fixes: **QK-norm**, logit soft-capping, or clamping.
5. **Optimiser state.** Adam's second moment is stale after a data regime change; a short LR cooldown + re-warmup usually recovers. If it does not, **roll back to the last checkpoint, skip the bad shards, resume with LR × 0.5**.

| Fix | Cost | When |
| :--- | :--- | :--- |
| skip batch / rollback | hours | spike |
| lower peak LR, longer warmup | recompute schedule | chronic instability |
| grad clipping (norm 1.0) | ~free | always on |
| QK-norm / z-loss | code change, retrain | repeated late-run spikes |

Do **not** just restart from scratch before you know which shard caused it.

**中文要点**：30 万步已过 warmup，按"数据→数值→注意力"排查：先定位 spike 附近 shard 是否重复/脏数据，再查 bf16/fp8 溢出与 logit 增长（加 z-loss、QK-norm）。修复优先级：跳批→回滚+降 LR→改结构；别无脑从头训。

---

### 14. Design the training setup for a model that doesn't fit on one accelerator, say 70B parameters on a pod.

**Anchor answer**

Start from the memory budget, because that determines the parallelism, not the other way round. For 70B with AdamW in bf16 + fp32 master:

| Item | Bytes/param | Total |
| :--- | :--- | :--- |
| weights (bf16) | 2 | 140 GB |
| gradients (bf16) | 2 | 140 GB |
| fp32 master + m + v | 12 | 840 GB |
| **subtotal (no activations)** | 16 | **~1.12 TB** |

So the state alone needs ≥14× 80 GB chips before a single activation. Layout:

1. **Shard the state — ZeRO-3 / FSDP**: parameters, gradients, optimizer states all sharded across the data-parallel group; all-gather on forward, reduce-scatter on backward. This alone makes 70B trainable on ~16–32 chips.
2. **Tensor parallelism (8-way) inside a node** where NVLink/ICI bandwidth is high; **pipeline parallelism** across nodes with interleaved 1F1B schedules to cut the bubble.
3. **Sequence/context parallelism** if you also want long context — otherwise activations, not weights, blow up.
4. **Activation checkpointing** (full or selective) to trade ~30% extra compute for a large activation-memory cut.
5. **Fused optimiser + bf16 compute**, gradient accumulation to reach the global batch.

Failure modes that matter at scale: TP is bandwidth-hungry, so keep it inside the fast fabric; PP needs enough micro-batches or the bubble eats 20%+; and **checkpointing must be asynchronous** — a 1 TB write every 30 min will otherwise stall the step time. Budget for **MFU ~35–45%** as a realistic target, and design for restart: at thousands of chips, hardware failures are a daily event, so elastic resume from the latest shard-parallel checkpoint is part of the design, not an afterthought.

**中文要点**：先算显存：70B 用 AdamW 光状态就约 1.12TB，必须 ZeRO-3/FSDP 分片；TP 放节点内走 NVLink，PP 跨节点配 1F1B。激活用重算换显存，ckpt 必须异步；千卡规模按"每天都会挂"设计弹性恢复，MFU 目标 35-45%。

---

### 15. Design the serving system for a multimodal assistant (text + image in, streaming text out) at hundreds of millions of users.

**Anchor answer**

Constraints first: **TTFT** dominates perceived quality, images are bursty and large, and output is streamed token-by-token. Design in layers:

| Layer | What it does | Key decision |
| :--- | :--- | :--- |
| Edge / CDN | image upload, resize, dedupe by hash | downscale to the model's max resolution **before** upload |
| Gateway | auth, quota, per-user rate limit, request shaping | shed load at the edge, not in the GPU pool |
| Ingest | decode, EXIF-strip, tile, vision-encoder | encode image **once**, cache by content hash |
| Prefill | prompt + image tokens + text tokens | **disaggregated prefill** from decode |
| Decode | streaming SSE/WebSocket | continuous batching, KV-cache paging |
| Safety | input + output filters | run in parallel with prefill, not serially |
| Cache | system prompt prefix cache, image-embedding cache | prefix cache is the single biggest TTFT win |

Cost structure: a 1024×1024 image is often ~1–4k visual tokens, so an image turn can cost 10–100× a text turn. That means **per-request cost is dominated by prefill**, and the money goes to: (a) vision encoders left unbatched, (b) KV cache held for long streams, (c) re-encoding the same image across turns — cache it.

Critical failure modes: **image bombs** (huge uploads → n² attention blowup) — enforce resolution caps at the edge; **cache stampede** on a viral image — request coalescing; **tail latency** from a single long decode monopolising a slot — cap max output tokens and use chunked prefill.

Verify with TTFT/TPOT p50/p95/p99 per modality split, cost per 1k requests, and a load test that replays real image sizes, not uniform ones.

**中文要点**：分边缘/网关/图像编码/预填充/解码/安全/缓存七层。图像 token 是文本的 10-100 倍，成本全在 prefill，所以图像只编码一次并按内容哈希缓存、系统提示用 prefix cache、prefill 与 decode 分离。边缘限分辨率防"图像炸弹"，并做请求合并。

---

### 16. Design a personalised recommendation system for rental listings using demographics, property metadata, amenities, price, reviews and location.

**Anchor answer**

Two constraints shape everything: the inventory is **perishable** (a listing that gets rented is useless) and **housing is legally sensitive** (fair-housing rules restrict which demographics may be used).

**Stage 1 — candidate generation.** Two-tower model: user tower (history, saved searches, price sensitivity, location affinity) and listing tower (price, beds, amenities, geo-embedding, review text embedding). Retrieve top ~500 via ANN over geo-partitioned indexes; geo is a *hard* filter (radius/commute), not a soft feature, because a perfect flat 40 km away is worthless.

**Stage 2 — ranking.** GBDT or deep model over interaction features (price vs the user's revealed budget, commute time, amenity match, listing freshness, host quality from reviews). Multi-task heads: click, save, contact, apply. Labels are position-biased → correct with inverse propensity weighting.

**Stage 3 — re-ranking.** Business rules: availability de-dup, fairness/diversity across neighbourhoods and price bands, exploration quota (10–20% of slots) so the system learns about new inventory, and per-host exposure caps.

Key failure modes:

- **Availability staleness** — a rented unit served for hours burns trust; TTL must be minutes, and the model needs an explicit availability feature whose staleness is logged.
- **Feedback loop** — you only learn about listings you showed; without exploration the long tail dies and the catalogue narrows.
- **Price non-monotonicity** — cheaper is not always better; users filter by budget, so price should enter relative to the user's inferred budget band.

Offline: NDCG@k, recall@500 for retrieval, calibration of contact rate. Online: contacts per session, time-to-lease (the real objective), and a **fairness audit by neighbourhood** before every launch.

**中文要点**：两段式：地理硬过滤 + 双塔召回 500，再排序（价格相对预算、通勤、设施、新鲜度）做多任务，最后重排加多样性与探索。三大坑：房源时效（TTL 分钟级）、反馈回路（必须留探索位）、价格非单调；上线前按社区做公平性审计。

---

### 17. Design a classifier that predicts the optimal moment to insert a commercial break in a video.

**Anchor answer**

Reframe: the "optimal moment" is not a classification label you can observe — you never see the counterfactual (what would retention have been without the break). So the honest design is **a scoring model + a constrained optimiser**, with labels from controlled experiments.

**Labels, in order of quality:**

1. **Randomised insertion** — show breaks at random offsets to a small traffic slice, measure retention delta per offset. Expensive but unbiased; this is the only clean source.
2. **Logged historical behaviour** — retention after existing breaks, but confounded by *where editors chose to break* (they pick natural scene boundaries). Debias with propensity on offset features.
3. **Proxy labels** — scene/shot boundary, silence or music transition, speaker change. Cheap, and a good prior.

**Model.** A survival/hazard model over candidate break points: for each candidate offset, predict the incremental probability of drop-off if a break occurs there, plus expected ad revenue. Features: distance to scene boundary, audio energy, remaining video length, content genre, user's historical ad tolerance, time since last break, and position within the session.

**Decision.** Do not take `argmax` per point — that clusters all breaks together and destroys retention. Solve a constrained problem: choose $k$ offsets maximising $\sum(\text{revenue} - \lambda\cdot\text{retention loss})$ subject to minimum spacing, max breaks per video, and never in the first N seconds.

**Evaluation.** Offline: AUC on drop-off is nearly useless here; use **uplift/Qini** on the randomised slice. Online: watch-time, break completion rate, and revenue per mille — watch for the trap where revenue rises while session watch time falls (you are harvesting, not optimising).

**中文要点**：最优插播点无观测标签，本质是"打分模型 + 约束优化"而非分类。干净标签只能来自随机插入实验；特征用镜头边界/音频/用户广告耐受度；决策加最小间隔与上限约束；评估用 uplift 而非 AUC，并警惕收入涨而观看时长跌的收割现象。

---

### 18. How would you improve product search results, focusing on the fraction of relevant documents retrieved (recall)?

**Source**: https://outcomeschool.com/blog/precision-vs-recall

**Anchor answer**

Recall is a **retrieval-stage** metric: `recall@k = |relevant ∩ retrieved_k| / |relevant|`. It is the ceiling on everything downstream — a ranker cannot recover a document that was never retrieved. So measure it first, separately from NDCG.

Diagnostic order (cheapest first):

| Lever | Why it moves recall | Cost |
| :--- | :--- | :--- |
| **Human-judged recall@100** on a sample of 500 queries | tells you if the ceiling is the problem at all | one labelling pass |
| **Hybrid lexical + dense**, union of candidates | BM25 catches exact SKUs/model numbers that embeddings smear; dense catches paraphrase | index size 2× |
| **Query understanding**: spelling, synonyms, attribute extraction ("red shoes size 10") | converts a vague query into a retrievable one | cheap, high ROI |
| **Hard-negative mining** for the embedding model | the model only learns to separate what it sees | retrain cycle |
| **ANN tuning** (`ef_search`, `nprobe`) | approximate indexes silently trade recall for latency | free — just measure it |
| **Index freshness / coverage** | out-of-stock or unindexed SKUs are unreachable | pipeline work |

The most common real bug is not the model: it is **index staleness and ANN parameters**. Teams routinely lose 5–15 points of recall to an `ef_search` set for p99 latency and never notice, because they only track NDCG@10 on the results they did return.

Two cautions: (1) recall@k trades against latency linearly, so pick k from the ranker's capacity, not from ambition; (2) optimising recall on a biased judgement set just optimises the judge — stratify labels by head/mid/tail query frequency. Cheap way to expand judgements: pooling from multiple retrieval systems (the TREC pooling trick).

**中文要点**：召回是检索层的天花板，必须先单独测量（人工标注 recall@100）。提升顺序：混合检索（BM25+dense）、查询理解、难负样本、ANN 参数（ef_search 常被悄悄调低丢 5-15 点）、索引新鲜度。切忌只看 NDCG@10。

---

### 19. Justify using a neural network for a given problem: what do you need to know about the network, dataset, timeline and business context?

**Anchor answer**

The question is a trap for "deep learning is better." Justification means **showing the problem has the properties neural nets exploit, and that no simpler model dominates.**

| Axis | What you must establish | Disqualifier |
| :--- | :--- | :--- |
| **Dataset** | ≥10⁴–10⁵ labelled examples per class-ish; high-dimensional, unstructured input (pixels, audio, text) where features are hard to hand-craft; iid or explicitly handled shift | 500 rows, 20 tabular features → gradient boosting wins, and it wins *today*, not in 2012 |
| **Network** | capacity vs data ratio, inductive bias fit (convolution for translation invariance, attention for long-range), inference budget (params × tokens), training budget | no GPU budget for the required epochs; latency SLA of 5 ms on CPU |
| **Timeline** | data collection + labelling + training + eval + iteration; a neural net needs 3–10× the iteration loops of a GBDT baseline | ship in two weeks with no labels |
| **Business** | is the marginal accuracy worth the cost? what is the cost of a wrong answer? is interpretability or auditability required? | regulated decision needing per-feature reason codes → linear/monotonic model |

The convincing structure is: **baseline first** (logistic regression / GBDT / heuristic), then the measured gap, then the argument that the gap comes from representation learning rather than from more capacity. If a GBDT with engineered features is within 1% and 10× cheaper to serve, the neural net is not justified.

Also name the failure modes you accept: calibration work, silent distribution shift, harder debugging, and the need for an eval set that does not leak.

**中文要点**：论证顺序是"先基线（GBDT/逻辑回归）→ 量化差距 → 说明差距来自表示学习而非容量"。四轴：数据量与是否非结构化、归纳偏置与算力预算、迭代周期、以及错误代价/可解释性要求。若 GBDT 只差 1% 且便宜十倍，神经网络就不成立。

---

### 20. Build the evaluation harness for a new frontier model release. What does it need to do?

**Source**: https://outcomeschool.com/blog/llm-evaluation

**Anchor answer**

A frontier release harness is an **evidence pipeline with a gate**, not a benchmark script. Five required capabilities:

1. **Coverage across axes** — capability (reasoning, code, math, long context, multilingual, tool use), safety (harmful compliance, jailbreak robustness, bias), and product (instruction following, refusal calibration, formatting). One number cannot represent a release.
2. **Contamination control** — n-gram overlap and embedding near-duplicate checks against pretraining corpora; canary strings in eval sets; prefer freshly-held-out or private items for the headline claims. A contaminated benchmark reports a number that looks like progress and is not.
3. **Statistical honesty** — report CIs, not point estimates; use **paired bootstrap** across items when comparing two models (the same items, so the comparison is low-variance). Fix seeds and temperature; log them per item. A 0.5-point gain on 500 items is noise.
4. **Reproducibility** — pinned harness version, prompt templates versioned as code, cached raw completions so a scoring change can be re-run without re-sampling (this alone saves days).
5. **The gate** — explicit pre-registered thresholds per axis, an owner per axis, and a **regression suite of ~200 frozen real-user prompts** that must not degrade. Blocking vs advisory must be decided before results arrive, or every result becomes negotiable.

Also: human preference eval on a stratified sample for quality axes where automatic metrics saturate, and a red-team pass whose findings are tracked to closure.

Failure mode to expect: the harness becomes a growth project of its own. Time-box it; scoring correctness beats benchmark breadth.

**中文要点**：评测框架是"带闸门的证据流水线"：能力/安全/产品三维覆盖，污染检测（n-gram + 嵌入近重复 + canary），配对 bootstrap 报置信区间，缓存原始输出保证可复现，并预先注册各轴阈值 + 200 条冻结回归集。

---

### 21. Do 1 million Seattle ride trips suffice to build an accurate ETA prediction model? How would you decide?

**Anchor answer**

The count is the wrong question — **coverage and label quality** decide, and 1M trips can be either plenty or hopeless.

Do the arithmetic first. 1M trips over a year ≈ 2,700/day in a city with ~100k road segments and thousands of OD pairs: the average cell has **a handful of samples**, and the distribution is extremely skewed (a few airport routes dominate). So:

| Test | What it tells you |
| :--- | :--- |
| **Effective sample size per stratum** (hour × OD-pair × weather) | the real constraint; if the p10 cell has <30 samples, you cannot fit a per-cell model there |
| **Sparsity structure** | % of trips covered by a route with ≥N observations; the residual mass needs a fallback model |
| **Label quality** | ETA is `dropoff_ts − pickup_ts` **minus** wait time; if the timestamp is driver-accept rather than passenger-onboard, your target has systematic bias |
| **Distribution shift** | rain, events, construction, seasonality — 1M trips from summer predicts summer |
| **Baseline** | historical mean per (route, hour) + map-routing duration. If your model does not beat the routing engine, 1M trips bought nothing |

Practical answer: **1M is enough to train a strong baseline plus a residual model**, not enough for a per-OD-pair deep model. Use a hierarchical/segment model: routing-engine time as a prior, learned multiplicative correction by time-of-day and conditions, with shrinkage toward the segment mean where data is thin.

Validate with **time-based splits** and per-segment error quantiles (p90 matters more than MAE for promises), plus coverage metrics: ETA error on the OD pairs that carry 80% of volume.

**中文要点**：问题不在 100 万这个数，而在覆盖度与标签质量。按小时×OD 对分层算有效样本，多数格子样本极少；先做"历史均值 + 路径规划时间"基线，再学乘性残差并做分层收缩；用时间切分与 p90 误差评估。

---

### 22. Tell me about a time you disagreed with a researcher or tech lead about priorities, and what happened.

**Anchor answer**

Answer with **substance disagreement, not personality**, and show that you changed the decision by changing the evidence.

Structure (STAR, compressed):

1. **Situation** — name the concrete technical conflict: e.g. "we were deciding whether to spend the quarter on a new data pipeline or on a bigger model; the lead wanted the model."
2. **Your position and *why*** — cite a measurement, not an opinion: "I had profiled the current pipeline: 40% of training examples were duplicated near-verbatim, and a 5% dedup run had already improved held-out eval by more than the last two model-size increases combined."
3. **How you disagreed** — the high-signal detail is *process*: you did not argue in the meeting; you ran a cheap experiment over three days and brought a one-page comparison with the cost and the risk of each path. You explicitly stated what evidence would change your mind.
4. **Resolution** — say what actually happened, including if you lost. "We ran the pipeline work first, with a checkpoint criterion; the model work started 6 weeks later and inherited the cleaner data."
5. **What you changed about yourself** — e.g. you now write the decision as a one-page doc *before* the meeting, because verbal disagreement escalates and written disagreement converges.

Red flags to avoid: "I was right and they were wrong"; disagreement framed as a communication problem; no measurement; no acknowledgment of the lead's constraint (they may have known about an org deadline you didn't).

If the disagreement was genuinely unresolvable, say how you **disagreed and committed** — and what guardrail you negotiated so the risk was bounded.

**中文要点**：讲"用证据改变决策"而非"我对他错"：立场要有度量（去重实验胜过两次扩模型），分歧过程是三天小实验 + 一页纸对比 + 明说"什么证据会让我改"；结局包括自己输的情况；最后加一条你因此改变的做事方式。

---

## 附：这 22 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 研究工程基本功 | 10（期望推导）、8（CE 梯度）、9（SVD） |
| 大规模训练实操 | 13（loss 发散）、14（70B 切分） |
| 服务与成本 | 15（多模态高并发）、21（数据够不够） |
| 建模判断力 | 5/6/7（偏差-方差、假设、正则 vs 验证）、19（该不该上 NN） |
| 业务落地 | 16（房源推荐）、17（广告插入点）、12（拒贷解释） |
| 评测与合规 | 20（评测框架）、18（召回）、12 |
