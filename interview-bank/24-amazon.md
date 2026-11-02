# 24 · Amazon (AWS)

> 23 题。出处：`README.en.md` § Amazon (AWS)。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Find the top-K most frequent items in a high-volume event stream with bounded memory.

**Anchor answer**

**Reference implementation** (Space-Saving, single pass, O(capacity) memory):

```python
import heapq
from collections import defaultdict

def top_k_stream(stream, k, capacity=None):
    capacity = capacity or 4 * k
    counts, delta = defaultdict(int), defaultdict(int)
    for item in stream:
        if item in counts:
            counts[item] += 1
        elif len(counts) < capacity:
            counts[item] = 1
        else:
            # evict the current minimum (heap in production: O(log capacity))
            victim = min(counts, key=counts.__getitem__)
            counts[item] = counts.pop(victim) + 1
            delta[item] = counts[item] - 1      # over-count bound
    return heapq.nlargest(k, counts, key=counts.__getitem__)
```

**Key points**

- Never keep an exact map: memory is O(cardinality), which is unbounded. Space-Saving guarantees `true_count ≤ estimated ≤ true_count + delta` with O(k) space; Misra-Gries is the other standard choice; Count-Min Sketch if you need arbitrary per-key queries.
- *Deterministic* top-k under adversarially ordered streams needs more than approximate counters — if the stream is windowed, keep one sketch per window and expire, otherwise a burst from six months ago ranks forever.
- Distributed: local sketches per shard, merge candidate sets at the coordinator, then a **second exact pass** over candidates. Sketch merges inflate error bounds — report them.
- What interviewers push on: (a) skew (one hot item starves a shard → local pre-aggregation before shuffle), (b) at-least-once delivery double counting (dedupe on event id), (c) whether "top" means count, unique users, or revenue — the answer changes the sketch.

**中文要点**：有界内存必须放弃精确计数，用 Space-Saving/Misra-Gries（O(k) 空间 + 误差上界）；分布式是"分片草图→合并候选→候选精确回算"，再加倾斜、重复投递、窗口语义三个必问追问。

---

### 2. Divide two integers without using multiplication, division or modulo. Find the number of connected components in a graph. Check balanced parentheses.

**Anchor answer**

```python
def divide(dividend: int, divisor: int) -> int:
    INT_MIN, INT_MAX = -(2**31), 2**31 - 1
    if dividend == INT_MIN and divisor == -1:      # the overflow trap
        return INT_MAX
    negative = (dividend < 0) ^ (divisor < 0)
    d, v = abs(dividend), abs(divisor)
    quotient = 0
    for shift in range(31, -1, -1):                # shift-subtract
        if (v << shift) <= d:
            d -= v << shift
            quotient += 1 << shift
    return -quotient if negative else quotient


def count_components(n: int, edges: list[tuple[int, int]]) -> int:
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]          # path halving
            x = parent[x]
        return x
    components = n
    for a, b in edges:                             # union by root; count on merge
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            components -= 1
    return components


def is_balanced(s: str) -> bool:
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack
```

**Key points**

- `divide`: shift-subtract is O(32) = O(1) for 32-bit; the two bugs that get graded are INT_MIN/-1 overflow and using `abs()` on INT_MIN in languages where it doesn't fit.
- Components: union-find is O(E α(V)); decrementing a counter on each successful union is cleaner than counting roots afterwards. DFS also works but blows the stack on deep graphs — say the iterative version.
- Balanced parentheses: empty string, `"]"` alone, and mismatched interleaving `"([)]"` are the three cases that fail naive counters. Never use a plain counter; you need ordering.
- Complexity you should volunteer: O(log n) attempts × O(1) shift for divide, O(E α(V)) union-find, O(n) time / O(n) space for the stack.

**中文要点**：除法用移位减法必须显式处理 INT_MIN/-1 溢出；连通分量用并查集并在 union 时递减计数；括号匹配必须用栈，计数器无法区分 `"([)]"`。

---

### 3. What is the closed-form solution of linear regression, and when do you use gradient descent instead?

**Source**: https://outcomeschool.com/blog/math-behind-gradient-descent

**Anchor answer**

For `y = Xw + ε` with squared loss, the normal equations give

$$\hat{w} = (X^\top X)^{-1} X^\top y$$

Derivation in one line: set `∇_w ‖Xw − y‖² = 2X^\top(Xw − y) = 0`. If `XᵀX` is singular (collinear columns, `p > n`), use the pseudo-inverse or add `λI` — ridge has the closed form `(XᵀX + λI)⁻¹Xᵀy`, which is exactly why regularization also solves the numerical problem.

**Choose gradient descent when:**

| Condition | Why |
| :--- | :--- |
| `p` large (10⁴–10⁹ features) | forming `XᵀX` is O(np²) memory and O(p³) to invert — infeasible |
| `n` huge but streamed | SGD/mini-batch touches data once per epoch; the closed form needs all data resident |
| Non-quadratic loss (logistic, Huber, quantile) | no closed form exists |
| Online / non-stationary data | you can update incrementally; refactorizing is O(p³) |

**Hook worth stating**: the normal-equation solution is not "more accurate" — `cond(XᵀX) = cond(X)²`, so forming the Gram matrix *squares* the condition number. In float32, that's a real loss of precision on poorly scaled features, which is why libraries prefer QR or SVD over literally inverting `XᵀX`. Also: for `n < p` the closed form is underdetermined, and gradient descent from zero converges to the minimum-norm solution — a different model, not an approximation of the same one.

**中文要点**：正规方程 `(XᵀX)⁻¹Xᵀy` 只在 p 小、数据能全驻内存、损失是平方时用；特征多/流式/非线性损失就上 GD。冷知识：构造 Gram 矩阵会把条件数平方，所以实践中用 QR/SVD 而不是真去求逆。

---

### 4. What are the differences between L1 and L2 regularization in logistic regression?

**Source**: https://outcomeschool.com/blog/regularization-in-machine-learning

**Anchor answer**

Objective: minimize `NLL(w) + λ‖w‖₁` (Lasso) or `+ λ‖w‖₂²` (ridge).

| | L1 | L2 |
| :--- | :--- | :--- |
| Penalty shape | diamond, corners on axes | sphere, smooth |
| Typical solution | **sparse** — some weights exactly 0 | dense, all weights shrunk |
| Geometry | constraint region has vertices → optimum lands on axes | no vertices → never exactly zero |
| Gradient | subgradient / proximal (soft-threshold) | smooth, `2λw` |
| Correlated features | picks one arbitrarily, drops the rest | splits weight between them |
| Solver | coordinate descent, LARS, proximal GD | Newton/IRLS, L-BFGS |
| Best when | `p ≫ n`, you want selection | features all somewhat useful, collinearity |

Why L1 zeroes weights: the subgradient condition at `w_j = 0` is `|∂NLL/∂w_j| ≤ λ` — a whole *interval* of gradients is consistent with zero, so small gradients don't move the parameter. L2's gradient at zero is `2λ·0 = 0`, so only exact zero stays zero.

Practical notes: **standardize features first** — L1/L2 are not scale-invariant, so an unscaled feature gets penalized more. `λ` is chosen by CV on the log scale; with L1, use 1-SE rule for a sparser model. Elastic net (`α‖w‖₁ + (1−α)‖w‖₂²`) exists because L1's arbitrary pick among correlated features hurts stability.

**中文要点**：L1 因约束域有顶点而产生精确 0（子梯度在 0 处是区间，所以不动），适合 p≫n 做选择；L2 均匀收缩、处理共线性更稳。两者都必须先标准化特征，否则惩罚随尺度失衡。

---

### 5. Write the loss function for logistic regression and prove it has a global minimum.

**Source**: https://outcomeschool.com/blog/math-behind-cross-entropy-loss · https://outcomeschool.com/blog/linear-regression-vs-logistic-regression

**Anchor answer**

With `p = σ(wᵀx)`, `σ(z) = 1/(1+e^{-z})`, the negative log-likelihood over `n` samples is

$$L(w) = -\sum_{i=1}^{n}\big[y_i\log\sigma(w^\top x_i) + (1-y_i)\log(1-\sigma(w^\top x_i))\big] = \sum_{i=1}^{n}\log\big(1+e^{-y_i w^\top x_i}\big)$$

(the second form uses labels `y ∈ {−1,+1}` — the two are algebraically identical).

**Proof sketch of convexity → global minimum:**

1. Let `z_i = y_i wᵀx_i`, affine in `w`.
2. `φ(z) = log(1+e^{-z})` has `φ'(z) = −σ(−z)` and `φ''(z) = σ(z)σ(−z) = σ(z)(1−σ(z)) ≥ 0`. So `φ` is convex (strictly, since `σ ∈ (0,1)`).
3. `L(w) = Σ φ(z_i)` is a nonnegative-weighted sum of convex functions composed with affine maps → convex.
4. Convex + any stationary point ⟹ that point is a global minimum. Gradient: `∇L = Σ (σ(wᵀx_i) − y_i) x_i`, and setting it to zero is a convex problem with no spurious local minima.

**Two honest caveats that separate a memorized answer from a real one:**

- `L` is strictly convex only if `X` has full column rank (or with L2 added). With **separable data**, the infimum is at `‖w‖ → ∞` — no finite minimizer exists, weights diverge, and the model becomes a hard step function. That's why unregularized logistic regression on separable data never "converges" and why `λ > 0` is the fix, not a nicety.
- Hessian is `XᵀDX` with `D = diag(p(1−p))` — positive semidefinite, and its smallest eigenvalue is what governs Newton/IRLS conditioning.

**中文要点**：损失写成 `Σ log(1+exp(−y wᵀx))`，二阶导 `σ(z)(1−σ(z)) ≥ 0` 保证凸性，故任一驻点即全局最优；但数据可分时最优点在无穷远（权重发散），必须靠 L2 才严格凸。

---

### 6. How is KL divergence loss different from cross-entropy loss? And from contrastive loss?

**Source**: https://outcomeschool.com/blog/math-behind-cross-entropy-loss · https://outcomeschool.com/blog/what-is-contrastive-learning

**Anchor answer**

The identity that answers the first half:

$$H(p,q) = H(p) + D_{KL}(p\|q)$$

Cross-entropy = entropy of the target + KL to the prediction. **With one-hot labels `H(p) = 0`, so minimizing CE ≡ minimizing KL.** They diverge only when `p` is soft (label smoothing, distillation): then `H(p)` is a constant you can't reduce, and KL is the "cleaner" objective — which is why distillation papers use KL.

| | Cross-entropy | KL | Contrastive (InfoNCE) |
| :--- | :--- | :--- | :--- |
| Target | explicit distribution `p` | explicit `p` | **implicit**, defined by positive/negative pairs |
| Symmetry | n/a | asymmetric (`KL(p‖q) ≠ KL(q‖p)`) | symmetric in construction |
| Needs | shared support, usually softmax over classes | same, plus `p` absolutely continuous w.r.t. `q` | only embeddings + similarity |
| Failure | overconfident on wrong labels | `KL(p‖q)` = ∞ when `q=0` where `p>0`; reverse KL is mode-seeking | batch-size sensitive; temperature `τ` controls hardness |

Contrastive loss is CE over a batch: `−log(exp(s⁺/τ) / Σ_j exp(s_j/τ))` — the "classes" are the other batch items. Two consequences worth naming: it needs **large batches or a memory queue** because the number of implicit negatives is the batch, and it maximizes a lower bound on mutual information, so it doesn't produce calibrated likelihoods — you can't read a probability off it.

**中文要点**：`H(p,q)=H(p)+KL(p‖q)`，one-hot 下两者等价，软标签/蒸馏时才分开（KL 更干净）；对比损失是"用 batch 内其他样本当负类"的 CE，靠 τ 调节难度，且只给相似度不给校准概率。

---

### 7. How do bagging and boosting differ? What is the computational difference between XGBoost and Random Forest?

**Anchor answer**

| | Bagging (Random Forest) | Boosting (XGBoost/GBDT) |
| :--- | :--- | :--- |
| Base learners | deep, low-bias, high-variance trees | shallow (depth 3–8), high-bias weak learners |
| Trained | **independently, in parallel**, on bootstrap samples + feature subsets | **sequentially**, each fits the residual/gradient of the current ensemble |
| Error reduction | variance (average decorrelated trees) | bias (then variance) |
| Overfit behaviour | more trees is *safe* — error plateaus | more trees *can* overfit — needs shrinkage, early stopping |
| Tuning knobs | `max_features`, `min_samples_leaf`, `n_estimators` | `learning_rate × n_estimators`, `max_depth`, `λ`, `γ`, `subsample` |

**Computational difference** — this is the actual question:

- **Random Forest** parallelizes across trees (embarrassingly) and within a node across features. Cost ≈ `O(T · n log n · m_try)`. No cross-tree dependency, so it scales to 1000 trees on a cluster trivially. Split criterion is the *first-order* impurity decrease (Gini/entropy), which is cheap and needs only histograms of labels.
- **XGBoost cannot parallelize across trees** — tree `t+1` needs tree `t`'s residuals. It parallelizes *within* a level: features are pre-sorted into quantile bins (weighted sketch for approximate mode), split gain uses **second-order** statistics `G²/(H+λ)`, and histograms are built in a cache-aware, column-block layout. That's why it's fast despite being sequential — but latency scales with `n_estimators`, and a 10k-tree model has a 10k-step critical path at inference-batch level (mitigate by compiling trees into arrays/predicting in bulk).

Say the trade: RF is nearly tuning-free and hard to break; XGBoost usually wins the last few points of accuracy on tabular data and gives you `feature_importance` + monotonic constraints, at the cost of careful hyperparameter work.

**中文要点**：bagging 并行降方差、树深且互相独立；boosting 串行拟合残差降偏差。计算上 RF 可跨树并行、用一阶不纯度；XGBoost 只能层内并行，靠预分桶直方图 + 二阶梯度增益 `G²/(H+λ)`，所以树的条数是推理关键路径。

---

### 8. Explain the bias-variance trade-off, cross-validation, and the curse of dimensionality.

**Anchor answer**

**Bias–variance decomposition** (squared loss, fixed `x`):

$$\mathbb{E}[(y-\hat f)^2] = \text{Bias}^2[\hat f] + \text{Var}[\hat f] + \sigma^2_\varepsilon$$

`σ²` is irreducible noise — a useful thing to say out loud, because it sets the floor and stops you from chasing it. High bias = underfit (train and val both bad, curves converge at a high loss); high variance = overfit (gap between train and val). The modern nuance: for over-parameterized models variance can *decrease* again past the interpolation threshold (double descent), so "more parameters = more overfit" is not a law.

**Cross-validation**: k-fold trades bias (each model sees `(k−1)/k` of the data) against compute. Rules that matter more than `k=5`:

- **Never leak**: fit scalers/encoders/feature selection *inside* the fold. Preprocessing on the full dataset is the most common silent leak.
- **Respect the structure**: `GroupKFold` when entities repeat (same user/patient), `TimeSeriesSplit` for temporal data — random k-fold on time series leaks the future and inflates scores by a lot.
- **Nested CV** when you both tune and report: inner loop selects hyperparameters, outer loop estimates generalization. Reporting the inner-loop best score is optimistically biased.
- Repeated k-fold when `n` is small; report mean ± std, not the best fold.

**Curse of dimensionality**: volume grows as `r^d`, so the shell of a hypersphere holds almost all the mass and pairwise distances concentrate — `(max dist − min dist)/min dist → 0`. Consequences: distance-based methods (kNN, RBF kernels, ANN indexes) degrade, you need exponentially more data to cover the space, and "nearest neighbor" stops being meaningful. Mitigate by reducing dimensionality or by using models that learn sparse local structure.

**中文要点**：`E[err] = bias² + var + σ²`，σ² 是不可约下界；CV 的关键是防泄漏（预处理放折内、时间序列/分组切分、调参与评估用嵌套 CV）；维度诅咒的本质是距离集中，会让 kNN/RBF/ANN 检索失效。

---

### 9. How do GRU cells work, and how do they address the vanishing gradient problem? How does a BiLSTM work?

**Source**: https://outcomeschool.com/blog/recurrent-neural-network

**Anchor answer**

GRU has two gates and no separate cell state:

```
r_t = σ(W_r x_t + U_r h_{t-1} + b_r)          # reset: how much past to read
z_t = σ(W_z x_t + U_z h_{t-1} + b_z)          # update: how much past to keep
h̃_t = tanh(W_h x_t + U_h (r_t ⊙ h_{t-1}) + b_h)
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

**Why this helps gradients.** Backprop through an RNN multiplies Jacobians `∂h_t/∂h_{t−1}` at every step; if their norm is <1 the product decays geometrically. GRU's update is **additive** in `h_{t-1}` with coefficient `(1−z_t)`: the path `∂h_t/∂h_{t−1} ⊇ diag(1−z_t)` plus the recurrent term, so when `z_t ≈ 0` the identity path carries gradient across many steps — a gated residual connection. LSTM does the same with `c_t = f_t ⊙ c_{t−1} + i_t ⊙ g_t`, where `f` can sit near 1 for long stretches. Neither *eliminates* vanishing; they make long-range credit assignment learnable. Honest caveat: they don't fix **exploding** gradients — that's still gradient clipping.

**BiLSTM**: run one LSTM forward and one backward over the sequence, concatenate `[h_t→ ; h_t←]` at each position. Each output sees full left and right context, which helps POS tagging, NER, and (historically) speech. The cost: it is **non-causal** — you need the whole sequence before producing any output, so it's unusable for streaming ASR or autoregressive generation, and it doubles compute. For most sequence labeling today a small transformer with relative positions beats it.

**中文要点**：GRU 用更新门做**加性**状态更新（`h_t=(1−z)h_{t−1}+z h̃`），形成类似残差的恒等通路，梯度不必连乘 Jacobian 所以能跨长距离；它不解决梯度爆炸（仍需裁剪）；BiLSTM 双向拼接但非因果，不能用于流式或自回归。

---

### 10. What is Attention in machine learning models? What happens in a neural network if you remove all the hidden layers?

**Source**: https://outcomeschool.com/blog/math-behind-attention-qkv

**Anchor answer**

**Attention** is a learned, content-based weighted average:

$$\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Three roles: `Q` = what I'm looking for, `K` = what I offer as a match, `V` = what I actually pass along. The `QKᵀ` scores are a *bilinear* similarity (not a fixed kernel), softmax normalizes over keys so weights sum to 1, and `√d_k` scaling keeps logits in a range where softmax has gradient — without it, at `d_k = 512` the dot products have std ≈ `√512`, softmax saturates, and gradients vanish. Multi-head = several attention maps in parallel subspaces, concatenated — different heads can specialize (previous token, syntactic head, coreference) without any of them being told to.

**Remove all hidden layers**, keeping embeddings → output layer:

| What you keep | Effective model |
| :--- | :--- |
| `embed → linear → softmax` | **multinomial logistic regression** on a bag of token embeddings; no interaction between positions at all |
| Same but with one attention layer | a *bilinear* model: interactions are fixed by `QKᵀ`, no learned feature composition |

Consequences: (1) no hierarchical feature composition — XOR-like or multi-hop functions become unrepresentable; (2) no non-linearity, so stacking linear maps collapses to a single linear map; (3) depth is what buys you the "universal approximation" and the induction circuits; (4) linearly separable problems still work — bag-of-embeddings + linear is a legitimate baseline, which is why the question is a good probe of whether you know *which* part does the work.

**中文要点**：注意力 = 用 QKᵀ 双线性相似度做 softmax 加权求和 V，`√d_k` 缩放是防止 softmax 饱和导致梯度消失；去掉所有隐层后网络退化为"embedding + 线性分类器"（多类逻辑回归），失去特征组合与非线性，因为线性堆叠仍是线性。

---

### 11. Discuss precision, recall and F1: when would you prioritise one over the others?

**Source**: https://outcomeschool.com/blog/precision-vs-recall

**Anchor answer**

`P = TP/(TP+FP)`, `R = TP/(TP+FN)`, `F1 = 2PR/(P+R)` (harmonic mean — dominated by the smaller of the two).

| Priority | Situation | Cost asymmetry |
| :--- | :--- | :--- |
| **Recall** | fraud pre-screening, disease screening, abuse detection, candidate sourcing | FN is catastrophic or cheap to review; a human filters the queue |
| **Precision** | auto-actions without review: auto-block, auto-send, legal hold, spam quarantine | FP harms a user directly and irreversibly |
| **F1** | balanced, no clear asymmetry, single ranking to compare models | symmetric cost assumption — usually a fiction, but a defensible default |

Better than F1 when costs are real: **cost-weighted expected loss** `c_FP·FP + c_FN·FN`, or **F_β** with `β = √(c_FN/c_FP)`. For ranking/retrieval use **PR-AUC or recall@k + precision@k** rather than a single threshold.

Things that mark real experience:

- **The threshold is a product decision, not a model output.** Fix the metric, pick the threshold from the cost curve, and report the full PR curve.
- **Under heavy imbalance, PR-AUC and ROC-AUC tell different stories** — ROC looks great because TN dominates. Choose PR.
- **Precision at fixed recall drifts with prevalence.** A model with 0.9 precision at 1 % prevalence becomes 0.09 at 0.1 % prevalence even with identical sensitivity/specificity. Any deployed threshold must be re-tuned when the base rate moves.

**中文要点**：召回优先的场景是"漏了代价致命、且有下游人工复核"（欺诈筛查、疾病筛查）；精确率优先是"模型直接自动执行、误报不可逆"。真实成本不对称时用 F_β 或加权损失，别用 F1；并记住精确率随正例率漂移，阈值必须跟着基线重调。

---

### 12. How do you handle data imbalance, collinearity, feature selection and regularization?

**Source**: https://outcomeschool.com/blog/feature-engineering · https://outcomeschool.com/blog/regularization-in-machine-learning

**Anchor answer**

**Imbalance** — fix the metric first, then the data:

1. Never use accuracy. Use PR-AUC / recall at fixed precision.
2. Try `class_weight` or threshold tuning *before* resampling — resampling changes the base rate the model learns, so the output is no longer a calibrated probability.
3. If resampling: undersample majority within the training fold only (never the validation set), or SMOTE on numeric features after scaling; oversampling duplicates leak across folds if done before splitting.
4. Consider treating it as ranking + a separate threshold, or as anomaly detection if the minority is <0.1 %.

**Collinearity**: detect via correlation matrix + VIF (>5–10); consequences are unstable coefficients and uninterpretable importances, *not* worse predictions for trees. Fix: drop one of a correlated pair using domain judgment, combine via PCA, or use L2 (which distributes weight and stabilizes). Tree models are largely immune — say this so you don't over-engineer.

**Feature selection**: filter (mutual information, variance, chi²) → embedded (L1, tree importance, permutation importance on *held-out* data) → wrapper (RFE, forward selection) only when `p` is small. Tree importance computed on training data systematically favors high-cardinality features — use permutation importance or SHAP on validation.

**Regularization**: L1 for sparsity/selection, L2 for stability and collinearity, elastic net when both. Always standardize. Tune `λ` by CV. For deep models: dropout, weight decay, early stopping, data augmentation are the analogues — but early stopping on a leaking validation set is the most common fake win.

**中文要点**：不平衡先换指标（PR-AUC）再考虑重采样，且重采样只在训练折内做、会破坏概率校准；共线性伤系数解释不伤树模型；特征选择优先 filter/embedded + 验证集上的 permutation 重要性；正则必须配合标准化。

---

### 13. Explain how you would design and evaluate an A/B test. What is a p-value and how do you interpret it here?

**Anchor answer**

Design first — most "A/B tests" fail before the analysis:

1. **One primary metric** with a pre-registered direction, plus guardrails (latency, crashes, revenue, complaints) that can kill the launch even if the primary wins.
2. **Unit of randomization** = the unit of interference. Randomize by user, not session, when the treatment has memory; randomize by cluster/geography when there's network effect (marketplace liquidity, social). 
3. **Power**: `n ≈ 16σ²/Δ²` per arm for 80 % power at α=0.05 (rule of thumb). Compute it *before* launch; if the MDE needs more traffic than you have, the honest move is to change the decision rule, not to peek.
4. **Duration**: at least one full weekly cycle, and long enough to cover novelty effects and the payback window. Fixed-horizon, not "stop when significant".
5. **Sanity checks**: SRM (sample ratio mismatch) test, pre-period covariate balance, and a guardrail that the treatment actually *was* applied (exposure logging).

**p-value**: the probability of observing data at least as extreme, *assuming the null is true and the model is correct*. It is **not** the probability the null is true, nor the probability the effect replicates. At `p = 0.03` you have evidence incompatible with "no effect" under assumptions — nothing more. Report the effect size with a confidence interval; "significant" with a 0.2 % lift on a metric with 1 % weekly noise is not a launch decision.

Traps: peeking inflates false positives (use sequential/always-valid tests if you must monitor), multiple comparisons need correction, and CUPED-style variance reduction on pre-period metrics is the cheapest power you'll ever buy.

**中文要点**：先定单一主指标 + 护栏指标、按用户/集群随机化、事前算功效与时长、查 SRM；p 值是"零假设为真时出现这么极端数据的概率"，不是效应为真的概率，必须配效应量与置信区间看。

---

### 14. What is Maximum Likelihood Estimation and how does it differ from Bayesian inference?

**Anchor answer**

**MLE**: `θ̂ = argmax_θ log p(D|θ)`. It's a point estimate with no prior; under regularity conditions it's consistent and asymptotically normal, `θ̂ ≈ N(θ*, I(θ̂)⁻¹)`, which is where Wald confidence intervals come from. Equivalently, MLE = minimizing KL from the empirical distribution to the model, which is why "minimize cross-entropy" *is* MLE for a softmax classifier.

| | MLE / MAP | Fully Bayesian |
| :--- | :--- | :--- |
| Output | point (mode) | posterior `p(θ|D)` |
| Uncertainty | asymptotic, via Fisher information | exact, including non-Gaussian, multimodal |
| Small `n` | unstable, can be at infinity (separable logistic) | prior regularizes; posterior stays proper |
| Prediction | `p(y|x,θ̂)` — plug-in, ignores parameter uncertainty | posterior predictive `∫p(y|x,θ)p(θ|D)dθ` |
| Cost | optimization | integration (MCMC / VI) — 10–1000× |
| Composition | not naturally hierarchical | hierarchical pooling falls out |

**MAP is the bridge**: `argmax p(θ|D) ∝ p(D|θ)p(θ)` — so **L2 regularization = MAP with a Gaussian prior**, L1 = Laplace prior. Naming this identity is usually the point of the question: your `λ` is a prior strength, and choosing it by CV is an empirical-Bayes-ish shortcut.

When to actually go Bayesian: small samples with structure (hierarchical/A-B testing across many segments), decisions where the *distribution* of the effect matters (expected loss under uncertainty), and sequential updating. When not to: large `n` where the posterior concentrates and MLE + bootstrap gives the same answer for a fraction of the cost.

**中文要点**：MLE 是点估计、渐近正态、用 Fisher 信息给区间；贝叶斯给后验与后验预测，小样本/分层/需要效应分布时才有必要，代价是积分。关键恒等式：L2 = 高斯先验下的 MAP、L1 = 拉普拉斯先验。

---

### 15. Why did transformers displace RNNs for language modelling, and what exactly does the KV cache buy you at inference time?

**Source**: https://outcomeschool.com/blog/how-do-rnns-and-transformers-differ · https://outcomeschool.com/blog/kv-cache-in-llms

**Anchor answer**

Three reasons, in order of importance:

1. **Path length.** An RNN connects token `i` to token `j` through `|i−j|` sequential steps; a transformer connects them in **one** attention hop. Short gradient paths mean long-range dependencies are learnable without gating heroics.
2. **Parallelism.** RNN computation is inherently sequential over time — you cannot use the sequence dimension for hardware parallelism, so training throughput is bounded by step latency. Transformer training parallelizes over all positions (teacher forcing + causal mask), which is what made training on trillions of tokens economically possible.
3. **Inductive bias.** RNN's "compress everything into one fixed vector" is a bottleneck; attention keeps all positions addressable, so retrieval of a specific earlier token is a lookup, not a reconstruction.

Cost: attention is `O(n²)` compute and `O(n)` state, versus RNN's `O(n)` compute and `O(1)` state. That's the honest trade — a transformer is worse per token at very long sequences, which is why linear-attention/SSM hybrids exist.

**KV cache**: with causal attention, token `t`'s keys and values never change once computed. Caching `K,V` for all previous positions turns per-token generation from `O(n²)` recomputation into `O(n)` attention against the cache — a decode step becomes a handful of GEMMs plus memory reads. **What it costs**: memory `= 2 · layers · kv_heads · head_dim · seq_len · batch · bytes`, which is why long context × large batch is memory-bound, and why decode is bandwidth-bound rather than compute-bound (arithmetic intensity ≈ 1). Levers: GQA/MQA (fewer kv heads), paged allocation, quantized cache, eviction/compression, prefix sharing.

**中文要点**：Transformer 赢在"任意两 token 距离为 1 + 训练可完全并行"，代价是 `O(n²)`；KV cache 把每一步解码从重算全序列降为只读缓存，本质是把计算换成显存带宽，所以长上下文/大 batch 是显存受限而非算力受限。

---

### 16. A customer's Bedrock-hosted workload costs too much. Cut inference cost dramatically without unacceptable quality loss.

**Source**: https://www.youtube.com/watch?v=jV2sCj4lHYk · https://outcomeschool.com/blog/llm-inference-optimization

**Anchor answer**

Never lead with "use a cheaper model" — you'll get "and lose quality". Lead with **measurement**, then a ladder.

**Step 0 — attribute cost.** Break spend by `model × feature × token type (input/output) × tenant`. Almost always: one or two features, dominated by **input tokens** (30–60k-token RAG contexts), and one expensive model serving a request class that doesn't need it.

**Step 1 — the cheap wins, no quality risk:**

| Lever | Typical effect |
| :--- | :--- |
| Prompt/prefix caching (stable system prompt + tool schemas + long doc prefix) | 40–90 % off cached input tokens |
| Trim retrieval (top-k 20 → 5, dedupe, drop boilerplate) | often 3–5× input reduction |
| Output cap + "answer in ≤ N bullets" | decode is the expensive side per token |
| Batch inference API for offline/async jobs | large discount for non-latency-bound work |
| Kill agentic waste: fewer steps, no redundant tool calls, stop on no-progress | 2–10× on agent workloads |

**Step 2 — the real money, with quality risk you must measure:**

1. **Cascade/routing.** Cheap model handles easy requests (classifier or logprob/short-response heuristic decides), escalate ~15–25 % to the frontier model. Evaluate *per slice*, not on the average — the damage concentrates in the escalated slice.
2. **Distill** the frontier model's behaviour into a small fine-tune on your own traffic; on narrow tasks this typically retains most quality at a fraction of price.
3. **Provisioned throughput vs on-demand**: crossover is utilization-dependent (roughly, sustained load above ~30–40 % of provisioned capacity); below it, on-demand plus caching wins. Do the arithmetic for their actual diurnal curve.
4. **Quantized self-hosted** (via SageMaker/Bedrock custom models) only if utilization is high; idle GPUs are the most expensive thing in the bill.

**Step 3 — governance**: per-tenant budgets, token quotas, cost per successful task as a first-class dashboard metric, and a regression gate tying cost changes to the same eval set that guards quality.

**中文要点**：先按"模型×功能×输入/输出 token"归因，再走阶梯：前缀缓存 + 砍检索上下文 + 限制输出 + 离线批处理是零质量风险的；级联路由与小模型蒸馏才是数量级省钱，但必须按切片评估质量；只有利用率够高才自建量化。

---

### 17. Design an agent that operates a web browser to complete multi-step tasks. How do you make it reliable enough to ship?

**Source**: https://outcomeschool.com/blog/how-do-computer-use-agents-work

**Anchor answer**

Reliability comes from **not** letting the model drive blindly. Architecture:

| Layer | Design |
| :--- | :--- |
| Perception | accessibility tree / DOM with stable ids **first**, screenshot second. Pixels are expensive, lossy, and coordinates break on any reflow |
| Action space | semantic ops: `click(node_id)`, `type(node_id, text)`, `select(option)`, `scroll(dir)`, `extract(selector)` — not `click(x=412,y=880)` |
| Grounding | resolve target by role+name+text before acting; if ≥2 matches, ask or pick the one inside the task's page scope |
| Loop | observe → decide → act → **verify the post-condition** ("did the field actually contain the value?"), not "did the click succeed" |
| State | explicit task state (`done/pending/failed`) + page fingerprint to detect that nothing changed |

**What makes it shippable:**

1. **Post-condition verification after every action.** Most browser agent failures are silent: the click landed on an ad, the form re-rendered, the submission errored. Detect by re-reading the DOM, not by trusting the model's narration.
2. **Deterministic fast paths.** For the top 20 sites, ship a scripted/selector-based path and let the model drive only the long tail. This alone turns a demo into a product.
3. **Idempotency + no-progress detection.** Hash `(url, action, target)`; a repeat means stuck → escalate or abort. Duplicate "Place order" is the failure that ends the feature.
4. **Session and credentials isolation.** Per-task browser context, no shared cookies; block non-allowlisted domains from receiving typed data.
5. **HITL for irreversible steps** (pay, send, delete, submit) with a screenshot diff.
6. **Robustness to adversarial pages**: page text is untrusted input; never execute instructions found in the DOM.
7. **Eval harness**: replayable recorded sessions + a scripted test site with injected latency, pop-ups, A/B variants. Track task success rate and *step count* (step inflation precedes failure).

**中文要点**：可靠性来自"用可访问性树的稳定 id 做语义动作 + 每步验证后置条件 + 头部站点走确定性脚本路径"；最致命的是静默失败（点到广告、表单重渲染）和重复执行不可逆动作，靠无进展检测与 HITL 兜住。

---

### 18. Design a multi-tenant inference platform that serves many foundation models to thousands of customers (Bedrock-shaped).

**Anchor answer**

**Shape it as control plane / data plane split** — this is the question's core insight.

| Plane | Components |
| :--- | :--- |
| Control plane | model catalog + version pinning, provider/region registry, tenant onboarding, IAM/policy, quota assignment, billing meters |
| Data plane | API gateway → auth → **quota/token-bucket admission** → model router → inference pool(s) → streaming response, with metering on the way out |
| Async plane | batch jobs, long-running fine-tunes, evaluation — separate pools so batch never starves interactive |

**Tenancy and isolation** (three tiers, priced differently):

1. **Shared pool, quota-isolated** — most tenants; per-tenant token buckets + concurrency caps prevent noisy neighbors. The hard part is fairness under saturation, not average throughput.
2. **Dedicated provisioned throughput** — a tenant buys reserved capacity; requires admission control that can *preempt* shared traffic when a reservation arrives.
3. **Dedicated instance / private model** — VPC isolation, customer-managed keys, single-tenant endpoints.

**Key design decisions worth stating:**

- **Model version pinning + immutable endpoints.** Customers must never get silently upgraded weights; deprecation needs a timeline and a dual-serve window.
- **Per-tenant tokens, not requests, as the quota unit.** Requests vary 100× in cost; request-count quotas let one tenant take the whole GPU.
- **Admission control with a priority lane**, and a documented behavior at saturation (429 with retry-after, never queue unboundedly).
- **Capacity model per accelerator**: continuous batching + paged KV cache means a single replica serves a *distribution* of request shapes; size on the p99 prompt length, because one 60k-token prompt evicts the effective batch.
- **Regional failover with capacity pre-reservation** — capacity is the scarce resource in an outage, not code. Warm pools or you fail over into a queue.

**Failure modes to name**: thundering herd on cold start (a new model gets 1000 tenants at once), cross-region data residency violations, token accounting drift between the meter and the gateway, and retry storms amplifying an incident — mitigate with client-side backoff hints and per-tenant jitter.

**中文要点**：控制面/数据面分离是骨架；隔离分共享池/包年吞吐/独占实例三档；配额单位必须是 **token 而不是请求数**（成本差 100 倍），饱和时的行为（429 + retry-after）与模型版本钉死是能不能对外卖的关键。

---

### 19. How would you design a recommendation system to suggest books to users? How would you model a warehouse inventory problem?

**Anchor answer**

Two different problem classes — say so first.

**A. Book recommender** (classic funnel):

| Stage | Method | Note |
| :--- | :--- | :--- |
| Candidate generation | two-tower embeddings + ANN, plus co-purchase/item-item CF, plus trending/new | reduce 10⁷ items → ~10³ cheaply |
| Ranking | GBDT/DLRM on user × item × context features | features: recency, price, series/author affinity, completion rate |
| Re-ranking | diversity (MMR), business rules, availability, duplicate editions | this is where "returned book 2 of 3" gets removed |
| Cold start | content embeddings from title/description/genre, plus popularity prior with exploration | a new book needs a prior, not a learned score |
| Eval | offline NDCG@10 / recall@100 on time-split data; online CTR + **purchase/conversion + return rate** | clicks mislead badly for books |

**B. Warehouse inventory** — this is **not** a ranking problem; it's forecasting + optimization:

- Forecast demand per SKU-location with seasonality/promotions/intermittency (Croston for slow movers).
- Decide policy: `(s, S)` / `(R, Q)` with safety stock from the **newsvendor critical ratio** `C_u/(C_u+C_o)` and lead-time variability, not from a fixed "2 weeks of cover".
- Allocate constrained supply across locations: LP/MIP maximizing fill rate subject to budget and storage; substitute across DCs before buying.
- Costs to weigh: holding, stockout (lost margin + placement penalty), expedite, obsolescence.

**The hook**: both systems are misjudged by a single aggregate metric. Recommenders look great on CTR while return rate climbs (the recommendation was wrong for the buyer); inventory looks great on stockouts while working capital balloons. Pick the metric that reflects the actual loss, and slice it.

**中文要点**：图书推荐走"两塔召回 + GBDT 精排 + 多样性重排"，冷启动靠内容向量与探索；库存问题本质是预测 + 优化（报童临界比定安全库存，LP 做供给分配），别当成排序题。两者共同陷阱是单指标好看（CTR 高但退货率涨 / 缺货低但资金占用爆炸）。

---

### 20. How would you decide an LLM-powered assistant is ready to launch to millions of customers?

**Anchor answer**

Readiness is a **set of gates with owners**, not a feeling. Four gate families:

| Gate | Evidence required |
| :--- | :--- |
| **Quality** | Pre-registered eval suite with slices (task type, language, input length, tenant). Ship threshold per slice, not on the mean. Claim-level factuality on a golden set; refusal quality on unanswerable inputs — an assistant that never says "I don't know" is not ready |
| **Safety** | Red-team results on your actual product surface (jailbreak, indirect injection via retrieved content, PII extraction, harmful advice), with residual risks written down and accepted by a named owner. Abuse rate measured on a human-labeled sample |
| **Operational** | p95 latency and cost per successful task within budget; graceful degradation defined (fallback model, cached answer, "try again"); per-tenant kill switch; rollback in minutes; capacity for 3× launch-week traffic |
| **Product** | Online experiment or canary with a primary metric and guardrails (complaint rate, escalation to human, task abandonment). A pre-committed decision rule for what result means "ship" vs "iterate" |

What separates a real answer from a checklist:

1. **Shadow/canary first.** Serve to 1 % with full logging and sampled human review. Most launch failures are distribution shift — real traffic contains the weird inputs your eval set never had.
2. **Eval on live traffic continuously.** Sample 0.1–1 % of production requests into an automated judge plus weekly human audit; the *judge–human agreement rate* is itself a metric you must maintain.
3. **Define the abort criteria in advance.** "Unsupported-claim rate above X on the finance slice for two consecutive days → auto-disable." 
4. **Cost per task, not per token.** Token dashboards hide runaway agent loops.

And say the uncomfortable part: for a customer-facing assistant, most incidents come from *confidently wrong* answers in low-volume slices that no gate covered, so the launch plan needs a fast-disable path and a feedback channel more than it needs a better average score.

**中文要点**：就绪度是四组带责任人的闸门（质量按切片、安全红队含间接注入、运维含降级与 kill switch、产品含 canary 与预设 abort 条件）；先 1% canary 再用线上流量持续抽样评测，并盯"评测器与人工的一致率"这个元指标。

---

### 21. Tell me about a time you disagreed with your team's technical direction. What did you do? (Have Backbone; Disagree and Commit)

**Anchor answer**

The LP is testing three separable behaviors — bucket the answer to show all three:

| Phase | What "good" looks like | Failure mode to avoid |
| :--- | :--- | :--- |
| **Disagree** | You argued with *evidence*: a prototype, a benchmark, a cost model, a customer quote — not seniority or taste | "I had a bad feeling" or a 40-minute opinion |
| **Escalate honestly** | You brought it to the decision owner with the tradeoff written down in one page, and you listened to the counter-argument | Going around the owner, or silently cc'ing their boss |
| **Commit** | If overruled: you built it wholeheartedly, and you *didn't* sandbag it to prove a point | Malicious compliance, "I told you so" when it broke |
| **Revisit** | You defined a tripwire in advance: "if p95 exceeds X by date Y, we revisit" | Re-litigating weekly, or never revisiting |

A shape that reads as real (fill in your own specifics): *we were choosing between extending the existing retrieval service and adopting a vector DB. I prototyped both against 3 weeks of real queries and showed the existing service, with an ANN index added, hit recall targets at a third of the operational cost. The team chose the new system for roadmap reasons outside my view. I committed, wrote the migration tooling, and added the alerting — then when the p99 latency problem I predicted appeared, the same benchmark became the argument for adding a cache layer rather than a rewrite.*

The two credibility markers interviewers listen for: (1) you can state the strongest version of the *other* side's argument, (2) you acknowledge what you got wrong in your own position. A story where you were simply right and they were simply wrong is scored as low self-awareness.

**中文要点**：LP 考三个独立行为——用证据（原型/基准/成本模型）反对、把权衡写成一页交给决策人、被否后全力执行且不消极怠工、并预先设 tripwire 触发再议；必须能讲出对方最强的理由，且承认自己判断中错的部分。

---

### 22. Tell me about your most significant failure. What happened, and what did you change afterward?

**Anchor answer**

Pick a failure with **real cost and a structural cause you owned**. Do not pick "we missed a deadline because of another team" — that's an excuse with a plot.

| Beat | Content | Interviewer is scoring |
| :--- | :--- | :--- |
| Context | the goal and the constraint, in two sentences | can you prioritize information |
| Your decision | what *you* chose, and the assumption you never validated | ownership, not shared blame |
| Impact | quantified: users, money, days, trust | honesty about magnitude |
| Detection | how it surfaced and how long it took | whether you had any instrumentation |
| Root cause | the mechanism — a missing check, an untested assumption, a metric that didn't exist | depth vs "I was too busy" |
| Change | a **mechanism** you added, with evidence it since fired | learning that compounds |

A concrete shape worth adapting: *I owned a model rollout that degraded quality for a small but high-value customer segment for eleven days because we gated on the aggregate metric and never sliced by tenant size. The aggregate actually improved.* Fixes: (1) launch gates now require per-slice thresholds, specifically the smallest economically significant slice; (2) a shadow-scoring job on a held-out tenant sample that alerts on per-slice divergence; (3) the guardrail now has an owner and an abort rule. *And the proof it worked: a later change was auto-blocked by that gate before rollout.*

Two traps:

- **Don't choose a failure whose lesson is "I worked too hard."** It reads as humble-bragging.
- **Don't over-correct.** Saying "now I'm much more careful" is not a change; naming the artifact (the gate, the runbook, the review question) is.

If your honest failure is interpersonal rather than technical, that's often stronger — engineering interviewers see plenty of "I shipped a bug" and few credible "I made a call that cost my team trust, and here's what I changed."

**中文要点**：选一个代价真实、根因可归到自己的失败；讲清当时未验证的假设、发现用了多久、以及你加的**机制**（按最小经济显著切片设闸门 + 影子评分告警）并用"后来它真的拦下了一次"证明有效；避免"我太拼了"式假失败和"以后更谨慎"式假改变。

---

### 23. Tell me about a time you saw an opportunity to do something bigger than the initial scope. (Think Big)

**Anchor answer**

The LP is not "I worked overtime". It is: **you connected the task to a customer/business outcome nobody had articulated, and you got other people to move with you.**

| Beat | What to include |
| :--- | :--- |
| The narrow ask | the ticket as scoped, with its stated success criterion |
| The observation | a signal you noticed *while doing the work* — repeated support tickets, a metric nobody owned, a manual step three teams duplicated |
| The reframe | the larger opportunity stated as a customer outcome + a size ("this affects N requests/week; fixing the class removes ~X manual hours") |
| The bet | what you did *without permission* (a spike, a prototype, a 2-page proposal) and what you explicitly did **not** do (you still shipped the original scope) |
| The enlistment | who you brought in and how — data, a demo, a shared metric — not "I convinced them" |
| Outcome + restraint | what shipped, and where you consciously stopped |

Two things that make it credible:

1. **You finished the small thing first.** The classic failure of "Think Big" stories is abandoning the committed deliverable for the shiny idea. Saying "I shipped the original scope on time, then spent 20 % of the next sprint on the spike" answers the obvious follow-up before it's asked.
2. **You name the boundary.** Big thinking without a stopping rule is scope creep; strong answers say "I stopped at a proposal because it needed a headcount decision above me" — which shows judgment about *where* initiative ends.

Adapter shape: *a ticket to add a retry to a batch job became a proposal for a shared idempotency primitive, after I found four teams independently re-implementing dedupe with different bugs. I built the library behind a flag, migrated my own job, and let the adoption data make the case.*

**中文要点**：Think Big 考的是"把任务连到没人说清的客户结果 + 拉动他人"，不是加班；必须先把承诺的小活按时交付，再在后续 sprint 里做 spike；并且说清你在哪里主动停下来（比如只做到提案），这体现的是判断力而不是收敛不足。

---

## 附：这 23 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 编码基本功 | 1（有界内存 top-k）、2（三个手写小题） |
| ML 理论是否真懂 | 5（凸性证明）、6（KL/CE/对比）、7（RF vs XGBoost 计算差异） |
| 建模判断力 | 8（维度诅咒）、11（P/R 取舍）、12（不平衡+共线性） |
| 统计学素养 | 13（A/B 与 p 值）、14（MLE vs 贝叶斯） |
| 成本与平台 | 16（Bedrock 降本）、18（多租户推理平台） |
| 系统设计广度 | 17（浏览器 agent）、19（推荐 + 库存） |
| 上线判断 | 20（放量闸门） |
| LP（Leadership Principles） | 21（Disagree）、22（Failure）、23（Think Big） |
