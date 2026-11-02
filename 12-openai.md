# 12 · OpenAI

> 31 题。出处：`README.en.md` § OpenAI。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Design and implement an in-memory key-value store supporting set, transactional begin, commit and abort.

**Asked at**: OpenAI

**Anchor answer**

The whole problem is **rollback**: `abort` must undo mutations, and nested/overlapping transactions must not see each other's uncommitted writes.

```python
class KV:
    def __init__(self):
        self.data = {}
        self.txns = []                      # stack of undo logs

    def get(self, k, default=None):
        if self.txns:
            log = self.txns[-1]
            if k in log:                     # own uncommitted write wins
                return log[k][1] if log[k][0] == "set" else default
        return self.data.get(k, default)

    def set(self, k, v):
        if self.txns:
            log = self.txns[-1]
            log.setdefault(k, ("set", self.data.get(k)))
        self.data[k] = v

    def delete(self, k):
        if self.txns:
            log = self.txns[-1]
            log.setdefault(k, ("del", self.data.get(k)))
        self.data.pop(k, None)

    def begin(self):
        self.txns.append({})

    def commit(self):
        log = self.txns.pop()
        if self.txns:                        # merge upward, keep outer view
            outer = self.txns[-1]
            for k, before in log.items():
                outer.setdefault(k, before)

    def abort(self):
        log = self.txns.pop()
        for k, (kind, before) in log.items():
            if before is None:
                self.data.pop(k, None)
            else:
                self.data[k] = before
```

**Key points**

- Undo log per transaction (`setdefault` keeps the *first* seen pre-image — that is what makes nested abort correct).
- Read-your-own-writes matters: a naive implementation reads committed data and breaks the test.
- If the spec allows concurrent transactions, this design is wrong — you need MVCC with a version per key and a commit-time conflict check.
- Say out loud whether `abort` on an empty stack is a no-op or an error; the evaluator usually tests both.

**中文要点**：核心是 undo log + 读己所写；`setdefault` 保留最早的前像才能正确嵌套 commit/abort；并发事务要换 MVCC + 提交期冲突检测。

---

### 2. Create a database ORM, step by step.

**Asked at**: OpenAI

**Anchor answer**

Build in five increments — each one is a demo you can show:

| Step | Add | Design decision |
| :--- | :--- | :--- |
| 1 | `Model` base with `__init__` from kwargs | column metadata via class attribute descriptors |
| 2 | `insert` / `get(id)` | parameterised SQL, never f-strings |
| 3 | query builder (`filter`, `order_by`, `limit`) | store fragments, compile at execution |
| 4 | relationships (`ForeignKey`, `has_many`) | lazy loader needs a session or you get N+1 |
| 5 | unit of work: identity map, dirty tracking, flush | one transaction per session |

```python
class Column:
    def __init__(self, name, ptype):
        self.name, self.ptype = name, ptype
    def __set_name__(self, owner, attr):
        self.attr = attr
    def __get__(self, obj, owner):
        return None if obj is None else obj._data.get(self.name)
    def __set__(self, obj, value):
        obj._data[self.name] = value

class Model:
    table = None
    def __init__(self, **kw):
        self._data = {}
        for k, v in kw.items(): setattr(self, k, v)
    @classmethod
    def columns(cls):
        return {n: c for n, c in vars(cls).items() if isinstance(c, Column)}
    @classmethod
    def get(cls, conn, pk):
        cols = ", ".join(cls.columns())
        row = conn.execute(f"SELECT {cols} FROM {cls.table} WHERE id = ?", (pk,)).fetchone()
        return cls(**dict(zip(cls.columns(), row))) if row else None
```

**Key points**

- Descriptors keep column metadata on the class; `__set_name__` saves passing the name twice.
- The N+1 problem is the interview trap — a relationship accessor without a session must batch.
- Always parameterise. A toy ORM that interpolates strings teaches the wrong lesson.
- Dirty tracking (snapshot on load, diff on flush) turns `insert`/`update` into `session.commit()`.

**中文要点**：分五步递进（模型/CRUD/查询构造/关系/unit of work）；坑是 N+1 与字符串拼 SQL，dirty tracking 才让 ORM 像 ORM。

---

### 3. Code a trivial web crawler using Go.

**Asked at**: OpenAI

**Anchor answer**

Go's crawler question is testing **goroutine lifecycle and dedupe safety**, not HTML parsing.

```go
package main

import (
	"net/http"
	"sync"
	"golang.org/x/net/html"
)

func crawl(root string, workers int) []string {
	seen := map[string]bool{root: true}
	var mu sync.Mutex
	var wg sync.WaitGroup
	jobs := make(chan string, 1024)
	var out []string

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for url := range jobs {
				links := fetchLinks(url)          // GET + parse
				mu.Lock()
				out = append(out, url)
				for _, l := range links {
					if !seen[l] {
						seen[l] = true            // mark before enqueue
						jobs <- l
					}
				}
				mu.Unlock()
			}
		}()
	}
	jobs <- root
	close(jobs)          // BUG: closes too early, see notes
	wg.Wait()
	return out
}
```

Correct shutdown: track outstanding work with a `sync.WaitGroup` per URL or use a counter + `cond`, because a plain `close(jobs)` after seeding kills workers mid-crawl.

**Key points**

- Mark visited **under the same mutex as the enqueue**, or two workers push the same URL.
- `close(jobs)` right after seeding is the classic bug; you need "queue empty **and** no in-flight jobs" to terminate.
- Bound concurrency (workers) and depth; unbounded fan-out on a real domain will exhaust fds.
- Use `http.Client{Timeout: ...}`; the default client has no timeout and one slow host hangs a worker forever.
- `errgroup.Group` + `context.Context` cancels the whole crawl on first fatal error — that's the idiomatic answer.

**中文要点**：Go 爬虫考的是 goroutine 生命周期——visited 与入队在同一个锁内，队列关闭要靠"空 + 无在飞任务"，并给 client 设超时。

---

### 4. Implement a UI from a mockup with provided CSS and API.

**Asked at**: OpenAI

**Anchor answer**

This round grades **process and fidelity**, not cleverness. The failure modes are always the same: you start coding before reading the API contract, then rework the layout.

Order of work:

1. **Read the API and the mockup together.** List every state: loading, empty, error, long-content, mobile.
2. **Build static structure first** with the provided CSS classes — no invented styles.
3. **Wire data** with a single fetch layer (`api.list()` / `api.update()`), so the component never `fetch`es directly.
4. **Implement the interactions** the mockup implies, not the ones you'd like: hover, disabled, optimistic update.
5. **Check the states** you listed in step 1, including a 500 and a slow response.

| What graders look for | Cheap way to score it |
| :--- | :--- |
| Reuse of provided CSS | never hardcode hex colours |
| Semantic HTML | `<button>`, not `<div onclick>` |
| Accessibility | labels, focus ring, `aria-live` on async update |
| Error handling | visible, retryable, not a console log |
| Key/list hygiene | stable keys, no index keys on reorderable lists |

**Key points**

- Say the state list out loud before coding; it's the only part interviewers can't get from the diff.
- Optimistic update + rollback on error is the "senior" signal in a CRUD UI round.
- If the CSS is provided, do **not** restyle it. Fidelity beats taste here.
- Ask one clarifying question max (e.g. mobile breakpoint), then decide and move.

**中文要点**：先读 API 与 mockup 并把状态列全（loading/empty/error/长内容/移动端），再写静态结构、单一 fetch 层、乐观更新 + 回滚；不要改提供的 CSS。

---

### 5. Refactor bad code: here are ~120 lines of working but messy code with passing tests. Improve the architecture without breaking them. What do you change first?

**Asked at**: OpenAI

**Anchor answer**

Rule: **the tests are the spec, so the first change is never structural — it's a characterization pass.**

Order of operations:

| Step | Action | Why first |
| :--- | :--- | :--- |
| 1 | Run tests, add characterization tests for untested branches | you need a safety net before moving code |
| 2 | Extract the pure core from IO | most mess is IO entangled with logic |
| 3 | Name the concepts (rename `data2`, `tmp`, `handle`) | renaming is the cheapest real improvement |
| 4 | Replace flag parameters with explicit functions/objects | kills the `if mode == 3` branches |
| 5 | Remove duplication last | premature extraction hides the real seams |

**Key points**

- Say explicitly "I'm not touching behaviour" and prove it by keeping tests green at every step.
- The most common 120-line mess is one function doing **parse → validate → transform → persist**; splitting those four is 80% of the win and is safe.
- Flag arguments (`def process(x, is_v2=True, legacy=False)`) are the highest-signal smell; convert to strategy objects.
- Don't invent an abstraction layer with one implementation — the reviewer will call it out.
- Close with what you'd *not* change yet, and what test you'd add before doing it.

**中文要点**：先补特征测试当安全网，再按"纯逻辑与 IO 分离 → 重命名 → 去 flag 参数 → 最后去重"的顺序改；每步都保持测试绿。

---

### 6. Write a Python function that displays the first n Fibonacci numbers.

**Asked at**: OpenAI

**Anchor answer**

Trivial function, so the grading is on **edge cases and stated assumptions**.

```python
def fib(n: int) -> list[int]:
    """Return the first n Fibonacci numbers: 0, 1, 1, 2, ..."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return []
    seq, a, b = [0], 0, 1
    while len(seq) < n:
        seq.append(b)
        a, b = b, a + b
    return seq
```

Variants worth naming out loud:

| Variant | Code | Note |
| :--- | :--- | :--- |
| Generator (no memory) | `yield a; a, b = b, a + b` | right answer if n is huge |
| One value only | loop with two vars, O(1) space | don't build a list |
| Matrix / fast doubling | `O(log n)` | only for `fib(k)`, not the sequence |
| Naive recursion | `f(n-1)+f(n-2)` | O(2ⁿ) — mention and reject |

**Key points**

- Fix the convention first: does "first n" start at 0 or 1? State it, because both are marked correct only if you're consistent.
- `n == 0` and negative `n` are the two cases people forget; raising on negative is clearer than returning `[]`.
- Python ints are unbounded, so no overflow — say that, it's why fast doubling isn't needed here.
- Don't use recursion for this. If asked, explain the memoised version and its stack depth limit.

**中文要点**：先声明数列从 0 开始；循环两变量 O(1) 空间，n 很大就用生成器；别用朴素递归，负数与 n=0 是必答边界。

---

### 7. Infection-spread simulation.

**Asked at**: OpenAI

**Anchor answer**

Treat it as a small simulator with an explicit model, then a grid/agent loop. Say the model out loud before coding, because "infection spread" has many.

Minimal SIR on a grid:

```python
import random

def simulate(size=50, steps=100, p_infect=0.3, p_recover=0.1, seed=0):
    rng = random.Random(seed)
    # 0 susceptible, 1 infected, 2 recovered
    grid = [[0] * size for _ in range(size)]
    grid[size // 2][size // 2] = 1
    history = []

    for _ in range(steps):
        new = [row[:] for row in grid]
        for y in range(size):
            for x in range(size):
                if grid[y][x] != 1:
                    continue
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < size and 0 <= nx < size:
                        if grid[ny][nx] == 0 and rng.random() < p_infect:
                            new[ny][nx] = 1
                if rng.random() < p_recover:
                    new[y][x] = 2
        grid = new
        history.append(counts(grid))
    return history
```

**Key points**

- **Double-buffer the grid.** Updating in place makes the spread directional (it depends on iteration order) and the result non-reproducible.
- Seed the RNG; a simulation you can't replay is not a simulation.
- Complexity is `O(steps · cells)`; for large grids use a frontier set of infected cells instead of scanning everything.
- Report the curve (infected over time) and check the peak — that's the actual output people want, not the final grid.
- If the interviewer wants continuous space, switch to agents with positions and a contact radius; same loop, different neighbour test.

**中文要点**：先声明模型（SIR/网格/邻接）；必须双缓冲否则扩散方向受遍历顺序影响，固定随机种子保证可复现，输出感染曲线而非最终网格。

---

### 8. Compute the KL divergence given different random variables.

**Asked at**: OpenAI

**Anchor answer**

Definition first: `D_KL(P‖Q) = Σ p(x) log(p(x)/q(x))`, or `∫ p log(p/q)` for continuous. It is **not symmetric** and is only defined where `q > 0` whenever `p > 0`.

```python
import numpy as np

def kl(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    mask = p > 0
    if np.any(q[mask] <= 0):
        return float("inf")            # support mismatch
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
```

Worked case — `P = [0.5, 0.5]`, `Q = [0.9, 0.1]`:

- `D(P‖Q) = 0.5·ln(0.5/0.9) + 0.5·ln(0.5/0.1) = 0.5·(−0.588) + 0.5·(1.609) = 0.511` nats
- `D(Q‖P) = 0.9·ln(0.9/0.5) + 0.1·ln(0.1/0.5) = 0.9·0.588 + 0.1·(−1.609) = 0.368` nats

| Variant | Formula | Where it appears |
| :--- | :--- | :--- |
| Forward `KL(P‖Q)` | mass-covering, mean-seeking | MLE / cross-entropy training |
| Reverse `KL(Q‖P)` | mode-seeking | variational inference, RLHF regulariser |
| `KL(P‖Q)` with Gaussians | `log(σ_q/σ_p) + (σ_p² + (μ_p−μ_q)²)/(2σ_q²) − ½` | VAEs |

**Key points**

- Zero-probability handling is the whole difficulty: `p log(p/q)` with `q=0` is `+∞`, and `0 log 0 = 0` must be skipped, not `nan`.
- The asymmetry is not a footnote — it explains mode collapse (reverse KL) vs blurry coverage (forward KL).
- Cross-entropy = `H(P) + KL(P‖Q)`; when `P` is one-hot, `H(P)=0`, which is why CE loss *is* the KL up to a constant.

**中文要点**：先处理 `q=0` 与 `0log0`；KL 不对称（正向覆盖、反向寻模），交叉熵 = H(P) + KL，one-hot 时二者等价。

---

### 9. If the accuracy of a classifier is 1, what is the lower/upper bound on the loss function for a single training example?

**Asked at**: OpenAI
**Source**: https://outcomeschool.com/blog/math-behind-cross-entropy-loss

**Anchor answer**

Accuracy 1 means the predicted class is correct on every example, so for a single example `p_true ≥ p_wrong` — but the **loss is not zero**, because confidence is unconstrained.

For cross-entropy with `K` classes and true class probability `p_t`:

- **Upper bound (worst case):** the model is correct but least confident — `p_t = 1/K`, so `L = −ln(1/K) = ln K`.
- **Lower bound (best case):** perfectly confident — `p_t → 1`, so `L → 0` (infimum, not attained in finite precision).

So `0 ≤ L ≤ ln K` for a single correctly-classified example.

| Loss | Correct-but-uncertain | Note |
| :--- | :--- | :--- |
| Cross-entropy | `ln K` | the real bound |
| 0/1 loss | `0` | why accuracy hides confidence |
| Hinge (SVM) | `0` if margin ≥ 1 | margin-based, different question |
| MSE on one-hot | `(1 − 1/K)²` | bounded, quadratic |

**Key points**

- The whole answer is the distinction between **decision correctness** and **probability quality**; accuracy=1 says nothing about calibration.
- `ln K` is the "all classes equally likely" loss — a 10-class uniform model that always guesses the argmax tie-break has not learned anything yet scores the same as random.
- If the question means "the global loss", the bound is per-example only; averaged loss can be anywhere in `[0, ln K]`.
- A well-calibrated perfect classifier still has `L > 0` on held-out data — that residual is irreducible entropy, not a bug.

**中文要点**：accuracy=1 只保证取到 argmax，交叉熵在单样本上的界是 `0 ≤ L ≤ ln K`（`p_t = 1/K` 时最大）；正确率不等于校准。

---

### 10. We have two models, 85% and 82% accuracy. Which do you pick?

**Asked at**: OpenAI

**Anchor answer**

Never answer "85%" alone. Accuracy is one number; the decision needs the **cost matrix and the deployment context.**

Questions I'd ask first:

1. **What's the class balance?** On 1% positive rate, 85% accuracy can be worse than always predicting negative.
2. **Is the gap statistically real?** With 1,000 test examples, a 3-point gap has a standard error around 1.5pp — compute the CI before believing it.
3. **What do errors cost?** FP and FN are rarely symmetric (fraud vs spam vs medical triage).
4. **Latency / cost / size?** A 3-point gain for 10× inference cost is often a bad trade.

| Situation | Pick |
| :--- | :--- |
| Balanced classes, equal error cost | 85%, if the CI excludes 0 |
| Rare positives, recall matters | compare PR-AUC and recall at fixed FPR |
| High-stakes FP | compare precision at the operating threshold |
| Latency-critical | 82% if it's 5× cheaper, with a fallback path |

**Key points**

- Say "I'd compare on the metric that matches the cost matrix" and name it: PR-AUC for imbalance, precision@k for ranking, ECE if probabilities feed a decision.
- A 3-point accuracy difference on a small eval set is frequently noise — this is the intended trap.
- Better answer than either: **ensemble or route** — use the cheap model by default, escalate low-confidence cases to the expensive one.
- Always ask whether the two models were evaluated on the same split; distribution shift between eval sets invalidates the comparison.

**中文要点**：先问类别不平衡与错误代价，再算置信区间——3 个点在千级测试集上常是噪声；最优解常是"便宜模型 + 低置信度升级"。

---

### 11. How do you handle missing data in Pandas?

**Asked at**: OpenAI

**Anchor answer**

Three decisions, in this order: **why is it missing** (MCAR / MAR / MNAR), **is the missingness informative**, **what does the model do with it**.

| Strategy | Code | Use when |
| :--- | :--- | :--- |
| Inspect first | `df.isna().mean().sort_values()` | always |
| Drop rows | `df.dropna(subset=["y"])` | label missing; tiny fraction |
| Drop columns | `df.dropna(axis=1, thresh=…)` | >50% missing and not predictive |
| Fill numeric | `df["x"].fillna(df["x"].median())` | MCAR, simple baseline |
| Fill categorical | `fillna("__missing__")` | missingness itself is a signal |
| Forward fill | `df.ffill()` | time series, last-known value |
| Indicator + fill | `df["x_na"] = df["x"].isna()` | MNAR |
| Model-based | `IterativeImputer` / `KNNImputer` | correlations strong, offline OK |

**Key points**

- Fit imputers on **train only**, then transform val/test — fitting the median on the full frame is leakage and inflates offline scores.
- `fillna(0)` on a numeric column is usually wrong: 0 is a valid value, and you've just told the model "missing = zero".
- Missingness is often the most predictive feature (e.g. "field left blank" correlates with fraud) — materialise it as an indicator instead of erasing it.
- `NaN` vs `None` vs `pd.NA` behave differently across dtypes; check `dtype` after the fill, since an int column silently becomes float.
- Time series: forward fill must be `ffill()` within group and never backwards across the split point.

**中文要点**：先看缺失比例与缺失机制（MCAR/MAR/MNAR）；填补器只在训练集 fit；缺失本身常是特征，应加 indicator 而不是简单填 0。

---

### 12. Explain self-attention. What is its computational complexity, and what are your options when contexts get long?

**Asked at**: OpenAI
**Source**: https://outcomeschool.com/blog/self-attention-in-transformers and https://outcomeschool.com/blog/how-does-sliding-window-attention-work

**Anchor answer**

Self-attention lets every token build its representation from a weighted average of all others, where weights come from `softmax(QKᵀ/√d_k)`.

For `n` tokens and dimension `d`:

| Step | Shape | Cost |
| :--- | :--- | :--- |
| `QKᵀ` | `n × n` | `O(n²d)` time, `O(n²)` memory |
| `softmax` | `n × n` | `O(n²)` |
| `AV` | `n × d` | `O(n²d)` |
| KV cache at decode | `2·L·n·d` | linear in `n`, per sequence |

So time is `O(n²d)` and the practical wall is **the `n²` score matrix**, not the weights.

Options when context grows:

1. **FlashAttention** — tiles the computation in SRAM, never materialises `n²`; same FLOPs, far less HBM traffic.
2. **Sliding-window / local attention** — each token attends to `w` neighbours, `O(n·w)`; stacking gives a receptive field of `layers × w`, but exact long-range recall is lost unless global layers are interleaved.
3. **Sparse / block attention** — fixed patterns (Longformer, BigBird) or learned.
4. **Compression** — summarise old KV (StreamingLLM sink tokens), quantise, or use GQA/MLA to shrink cache.
5. **Retrieval instead of context** — RAG sidesteps the quadratic term.
6. **SSM / linear attention hybrids** — `O(n)`, weaker exact copying.

**Key points**

- Answer the complexity as **two numbers**: `O(n²d)` compute and `O(n²)` activation memory — the second is what OOMs first.
- Sliding window is cheap but has no `n²` term only if you keep it local; a window of 4k with 32 layers reaches ~128k tokens *theoretically* and much less reliably.
- At inference, decode is `O(n·d)` per token due to the KV cache — long context hurts latency through memory bandwidth, not FLOPs.
- Say what you'd measure: needle-in-a-haystack recall at 32k/128k, not just perplexity.

**中文要点**：`O(n²d)` 计算 + `O(n²)` 中间矩阵内存，后者先 OOM；长上下文选项是 FlashAttention、滑窗/稀疏、KV 压缩、RAG、混合 SSM，验证要看长文召回而非困惑度。

---

### 13. What is the relationship between cross-entropy, KL divergence and perplexity, and why is cross-entropy the training loss for language models?

**Asked at**: OpenAI
**Source**: https://outcomeschool.com/blog/math-behind-cross-entropy-loss

**Anchor answer**

The chain is exact:

```
H(P, Q) = H(P) + D_KL(P‖Q)
PPL     = exp(H(P, Q))          # nats
```

| Quantity | Definition | Role |
| :--- | :--- | :--- |
| Entropy `H(P)` | `−Σ p log p` | irreducible uncertainty of the data; a constant w.r.t. model |
| Cross-entropy `H(P,Q)` | `−Σ p log q` | what you minimise |
| KL `D(P‖Q)` | `Σ p log(p/q)` | the *excess* cost of using `Q`; what you actually reduce |
| Perplexity | `exp(H(P,Q))` | "how many equally-likely tokens is the model choosing among" |

Why cross-entropy for LMs: next-token prediction is classification over the vocabulary with a **one-hot target**, so `H(P) = 0` and `H(P,Q) = D_KL(P‖Q)` exactly. Minimising CE = minimising KL = maximum likelihood. Its gradient is clean: `∂L/∂logits = p̂ − y`, which is well-conditioned and cheap.

**Key points**

- Since `H(P)` is constant, "minimise CE" and "minimise KL" are the same optimisation — the distinction only matters when comparing models on different data distributions.
- Perplexity is `exp` of the *mean per-token* CE; a PPL of 10 means the model is as uncertain as choosing among 10 tokens uniformly. It is not comparable across tokenizers.
- Label smoothing adds a small `H(P)` term, so CE and KL stop being equal — that's why it regularises.
- Report bits-per-byte, not PPL, when comparing models with different vocabs.

**中文要点**：`CE = H(P) + KL`，one-hot 目标下 `H(P)=0` 所以 CE 即 KL（等价于极大似然）；困惑度是 CE 的指数，跨 tokenizer 不可比，要比 bits-per-byte。

---

### 14. What is the effect of adjusting an LLM's context window size?

**Asked at**: OpenAI
**Source**: https://www.linkedin.com/posts/amit-shekhar-iitbhu_the-context-window-is-the-llms-working-memory-activity-7437754426175672320-MH9c and https://www.youtube.com/watch?v=CGIhxIaOg3M&lc

**Anchor answer**

Context window is **working memory**, and every direction has a cost.

| Direction | Gains | Costs |
| :--- | :--- | :--- |
| Bigger (32k → 200k+) | fewer retrieval round-trips, whole-repo/whole-doc reasoning, fewer "forgot earlier turn" bugs | `O(n²)` prefill, KV cache memory linear in `n`, TTFT up, attention dilution |
| Smaller | cheaper, faster, lower memory, less distraction | truncation, needle loss, more engineering (RAG, compaction) |

The non-obvious effects:

- **Lost in the middle**: retrieval accuracy for facts in the centre of a long context is measurably worse than at the edges, even at 100% "fits in window".
- **Context rot**: as unrelated tokens accumulate, instruction-following degrades. More context is not monotonically better.
- **KV cache is the real budget**: `2 · layers · heads · d_head · n · bytes` per sequence; at 128k that is tens of GB per request unless you page/quantise/offload.
- **Cost is charged on tokens you send**: a 100k prompt per turn costs the same whether or not it helps.

**Key points**

- Extending beyond the trained length (RoPE scaling / YaRN) works but degrades *precision* before it degrades fluency — always eval with needle tests before shipping.
- Practical rule: include the **most relevant 10–20%** and keep the prompt stable for prefix caching; stuffing everything is usually slower *and* worse.
- Long context and RAG are complements: long context for "reason over this document", retrieval for "find among 10M documents".
- Measure with a task that needs the middle of the context, not just summarisation.

**中文要点**：上下文变大赢在少一次检索、能整体推理，输在 `O(n²)` prefill、KV 显存、TTFT 和中段信息丢失（lost in the middle）；上下文不是越多越好。

---

### 15. You are building a production agent that calls tools (function calling). What makes the loop reliable enough to ship?

**Asked at**: OpenAI
**Source**: https://outcomeschool.com/blog/how-does-function-calling-work-in-llms and https://outcomeschool.com/blog/ai-agent-loop

**Anchor answer**

Shipping reliability is **not** prompt quality — it is the loop's guarantees.

| Guarantee | Mechanism |
| :--- | :--- |
| Termination | step budget, wall-clock deadline, cost ceiling; always a max |
| Error recovery | classify transient vs terminal; feed terminal errors back as observations |
| Idempotency | client-generated key per side-effecting call; dedupe on replay |
| State | persist the transcript + tool results; a restart resumes, not restarts |
| Validation | schema-validate arguments before execution; reject, don't patch |
| Observability | every step logged: tool, args, latency, tokens, result hash |
| Safety | allowlist tools per task, confirm destructive actions |

```python
for step in range(MAX_STEPS):
    msg = model.step(history, tools=TOOLS)
    if not msg.tool_calls:
        return msg.content
    for call in msg.tool_calls:
        args = schema.validate(call.name, call.arguments)      # hard fail → observation
        result = execute(call.name, args, idem=call.id)
        history.append(observation(call.id, result, truncated=True))
```

**Key points**

- The single biggest reliability gain: **never let a tool error raise** — return `{error, retryable}` so the model can adapt. Silent retries teach it nothing.
- No-progress detection: hash `(tool, args)`; two identical consecutive calls means the agent is stuck → break the loop with a partial answer.
- Validate arguments against JSON Schema **before** the side effect; models emit plausible-but-wrong enums constantly.
- Bound every result size. One 200k-token tool result ends the run on context overflow.
- Track success rate per tool; a tool at 60% success is a product bug, not a model bug.

**中文要点**：可上线依赖"步数/时间/成本上限 + 错误回灌不抛异常 + 幂等键 + 参数 schema 校验 + 结果截断 + 无进展检测"，而不是提示词写得好。

---

### 16. How would you build an LLM-powered enterprise search system?

**Asked at**: OpenAI

**Anchor answer**

Enterprise search fails on **permissions and freshness**, not on embedding choice. Design for those first.

| Stage | Design | Note |
| :--- | :--- | :--- |
| Ingest | connectors (Drive/Slack/Confluence/Jira), incremental sync, deletion propagation | deletes are the forgotten half |
| Parse | layout-aware extraction, tables, OCR | PDF tables are where quality dies |
| Index | hybrid: BM25 + dense, per-chunk ACL metadata | dense-only loses exact part numbers |
| Retrieve | hybrid + cross-encoder rerank, top-50 → top-5 | rerank is the highest-ROI step |
| Filter | ACL filter **before** ranking, applied at query time | never post-filter |
| Answer | cite chunk ids, abstain when evidence is weak | citations make it auditable |
| Evaluate | golden Q/A per department, recall@k + answer faithfulness | offline eval per tenant |

**Key points**

- **Permission-aware retrieval**: index ACLs with each chunk and filter in the query. Post-filtering leaks existence, and caching across users leaks content.
- Hybrid beats pure vector search on enterprise text (product codes, names, acronyms) — always keep BM25.
- Freshness: expose "last synced" in the UI; users lose trust once in ten answers is stale from an edited doc.
- Chunking: 300–800 tokens with heading context prepended, overlap ~10–15%; but measure — chunk size effects dominate embedding-model choice.
- The honest KPI is "time to correct answer", not recall@10.

**中文要点**：企业搜索的成败在权限过滤（查询期、chunk 带 ACL）与增量同步（含删除），架构上必须混合检索 + 重排 + 引用与弃答。

---

### 17. Design the serving stack for a ChatGPT-scale consumer assistant: hundreds of millions of weekly users, streaming chat, multiple model tiers.

**Asked at**: OpenAI
**Source**: https://outcomeschool.substack.com/p/inside-chatgpt-what-happens-after

**Anchor answer**

Scale math first: 300M WAU × ~10 messages/week ≈ 3B messages/week ≈ 5k QPS average, 20–50k QPS peak, each streaming for tens of seconds.

| Layer | Design | Why |
| :--- | :--- | :--- |
| Edge / CDN | static assets, TLS termination, region routing | keep bytes off origin |
| Gateway | auth, plan-based quota, abuse, request shaping | cheapest place to reject |
| Router | tier selection (small/large/reasoning), tool need, safety pre-check | 80% of traffic goes to the cheap tier |
| Context service | conversation store + prefix assembly + cache | prefix cache hit rate is the main cost lever |
| Inference | continuous batching, paged KV, prefill/decode disaggregation per tier | independent SLOs per tier |
| Stream | SSE with seq numbers, resumable | mobile networks drop |
| Post | moderation, logging, metering, feedback | async, off the critical path |

**Key points**

- **Tiering is the product's economics**: route by difficulty, not by user plan alone; escalate on user retry or explicit "think harder".
- Long-lived streams (30–120 s) mean connection counts, not QPS, size your gateway — plan for millions of concurrent sockets.
- Prefix caching across a conversation is the dominant saving; keep system prompt + tools byte-stable or you lose it.
- Regional capacity: a single-region design at this scale means a p99 that depends on transatlantic RTT.
- Failure mode to design: model tier overload. Degrade to a smaller tier with an explicit UI signal, never silently.

**中文要点**：先算 QPS/并发长连接，再分层；路由分级决定成本，前缀缓存是最大省钱杠杆，过载时显式降级到小模型而非静默。

---

### 18. Design and build a webhook delivery system that reliably delivers events to customer-registered URLs.

**Asked at**: OpenAI

**Anchor answer**

Webhooks are **at-least-once delivery to an untrusted, slow, failing endpoint.** Design for duplicates and for the customer being down for hours.

| Component | Design |
| :--- | :--- |
| Event log | append-only, per-event id, immutable payload |
| Fan-out | event → one delivery job per subscribed endpoint |
| Queue | per-endpoint queue (not a global one) so one dead customer can't starve others |
| Delivery | POST with HMAC signature + timestamp, `Idempotency-Key` = event id |
| Retry | exponential backoff with jitter, e.g. 1 m, 5 m, 30 m, 2 h, 6 h, 24 h; cap attempts |
| Timeout | short (5–10 s), never retry on 2xx |
| Circuit breaker | N consecutive failures → pause endpoint, notify owner |
| Observability | per-endpoint success rate, lag, last error; a customer-facing log |
| Ordering | per-endpoint FIFO where possible; document that global order is not guaranteed |

**Key points**

- Consumers **will** receive duplicates — that's why the signature covers the body and you send a stable event id. Saying "exactly once" here is a red flag.
- Per-endpoint queues + per-endpoint concurrency limits is the design detail that separates this from a naive retry loop.
- Signature must include the timestamp and be verified with a constant-time compare; without a timestamp, replays are trivially forgeable.
- A dead endpoint (404 for days) must be disabled automatically, or your queue grows unbounded; keep a dead-letter store for replay.
- Also design the **self-serve replay** endpoint — it removes most support tickets.

**中文要点**：语义是 at-least-once（重复必然发生，靠事件 id + 签名去重）；每 endpoint 独立队列与并发限流，退避重试 + 熔断 + 死信与自助重放。

---

### 19. Design a system to schedule jobs in a distributed environment.

**Asked at**: OpenAI

**Anchor answer**

Separate **when** (trigger) from **where** (execution), and make the boundary crash-safe.

| Component | Design | Failure it handles |
| :--- | :--- | :--- |
| Scheduler (leader-elected) | computes due jobs, writes to a queue | split-brain double-fires |
| Job store | `(job_id, due_at, cron, payload, state)` | restart recovery |
| Queue | durable, visibility timeout | worker crash mid-run |
| Workers | stateless, idempotent, heartbeat | horizontal scale |
| Lease / lock | `UPDATE ... SET locked_until = now()+30s WHERE ... AND locked_until < now()` | duplicate execution |
| Retry | attempts column, backoff, dead-letter | poison jobs |
| Coordination | etcd/ZooKeeper or DB row lock for leader | two schedulers |

**Key points**

- **Exactly-once execution does not exist here.** Pick at-least-once + idempotent jobs and enforce it with a job-level dedupe key.
- The lease must be **renewed** by a heartbeat; a fixed lock with no renewal turns a slow job into a duplicate job.
- Cron parsing is a trap: DST transitions fire twice or never. Store timezone per job and expand cron in that zone explicitly.
- Backfill and catch-up policy must be explicit — after a 2-hour outage, do you run 24 missed runs or one?
- The scheduler's own availability is the SPOF; leader election with a fencing token prevents the old leader from writing after losing leadership.

**中文要点**：调度与执行分离，靠 leader 选举 + lease 心跳 + 幂等去重；语义只能是 at-least-once，还要明确 DST 与补跑策略。

---

### 20. Design an in-memory database. / Design Slack.

**Asked at**: OpenAI

**Anchor answer**

Two prompts in one line — treat whichever you get as a **storage + fan-out** problem and state the choice out loud.

**In-memory database**

| Layer | Decision |
| :--- | :--- |
| Data | hash map for point reads + sorted structure (skiplist) for range/scan |
| TTL | heap of `(expires_at, key)`, lazy check on read |
| Persistence | append-only log + periodic snapshot; replay on boot |
| Concurrency | single-writer event loop, or sharded locks by key hash |
| Eviction | LRU list or sampling (Redis-style approximate LRU) |

**Slack**

| Component | Design |
| :--- | :--- |
| Write path | message → durable log (per channel, ordered by server seq) → fan-out |
| Read path | channel history = range scan; unread counts = per-user counter |
| Fan-out | channels are small → on-demand read; DMs/large channels → push to per-user inbox |
| Presence | ephemeral, in-memory, TTL-based, not durable |
| Search | async index (Elasticsearch-class), never on the write path |
| Realtime | WebSocket gateway + pub/sub per channel |

**Key points**

- In-memory DB: the interviewer wants the read/write path and *why in-memory* (latency, working set) plus what you lose (durability, cost per GB).
- Slack: the decisive design question is **read-time vs write-time fan-out**; pick per channel size and say the threshold (e.g. >1k members → inboxes).
- Message ordering needs a per-channel monotonic sequence, not wall-clock timestamps.
- Both: name the single point of failure and how it recovers — snapshot+log replay, or replica promotion.

**中文要点**：内存库讲读写路径/持久化（AOF+快照）/淘汰与分片锁；Slack 讲按频道规模选择读时或写时扇出、每频道单调序列、搜索异步化。

---

### 21. A customer says “the model got worse” after you upgraded model versions in their deployment. How do you verify and respond?

**Asked at**: OpenAI

**Anchor answer**

First: **is it the model, or everything else you changed?** An upgrade usually bundles a prompt template, a tokenizer, and a parameter default.

| Step | Action | Evidence produced |
| :--- | :--- | :--- |
| 1 | Pin the old version back in a test project | isolate version as the variable |
| 2 | Replay a sample of real requests (same prompts, same params) on both | side-by-side diffs |
| 3 | Score with their metric + a held-out eval, not vibes | quantified delta |
| 4 | Classify failures: format drift, refusal change, verbosity, tokenizer | cause |
| 5 | Decide: fix prompt, change param, pin old version, or roll forward | action |

**Key points**

- The most common real causes are not "the model got worse": **output format changed** (parsing broke), **temperature/top_p default changed**, **max_tokens truncation**, or **longer responses hit their token budget**.
- Always replay *their* traffic. Your benchmark improving and their task regressing is normal, and only their data answers it.
- Check the refusal rate and the verbosity separately from correctness — users report "worse" for "too long".
- Respond with a number and a plan, plus an immediate mitigation (pin the old version) — trust is lost in the hours you spend arguing.
- If it is a genuine regression, file it with the replay set; that's the artifact that makes it fixable.

**中文要点**：先做"版本单变量"回放（同 prompt 同参数跑双版本），最常见的真因是格式变化/参数默认值/截断；先回滚止损再谈量化归因。

---

### 22. An enterprise customer reports that responses from your deployed system have gotten slow. Walk me through the diagnosis.

**Asked at**: OpenAI

**Anchor answer**

Break "slow" into **TTFT vs total vs per-token**, because they have different causes and the customer means one specific one.

| Segment | Signal | Typical cause |
| :--- | :--- | :--- |
| Network / auth | p50 vs p99 by region | new proxy, VPN, DNS |
| Queue / admission | wait time before first token | concurrency limit, rate limit, tenant surge |
| Prefill (TTFT) | prompt tokens per request | context grew (RAG now returns 20 chunks) |
| Decode (TPS) | output tokens / duration | batching contention, KV cache pressure, tier change |
| Post-processing | time after last token | moderation, logging, their own callback |

Diagnostic order: dashboards per segment → compare to last week (when did it start) → correlate with a deploy/config change → reproduce with one request and a trace id.

**Key points**

- Ask for a **trace id and a timestamp of a slow request** in the first reply; without it you're guessing.
- The most common cause in deployed RAG systems is **prompt growth** — someone raised `top_k` from 5 to 20 and TTFT tripled.
- Streaming makes "slow" ambiguous: users blame the model when the real delay is the client buffering chunks. Check whether streaming is actually enabled end-to-end.
- Watch for retry storms: a downstream 429 causes silent retries that look like latency, not errors.
- Close the loop with the measurement, not the fix: "here's your p95 before/after and the segment that moved".

**中文要点**：把"慢"拆成 TTFT / 总时长 / 每 token 速度分别看；RAG 场景最常见是 prompt 变长（top_k 调大），别忽略客户端没真流式与重试风暴。

---

### 23. How do you approach GenAI safety in consumer products?

**Asked at**: OpenAI

**Anchor answer**

Consumer safety is **policy → model → product → operations**, and each layer needs its own measurable.

| Layer | Control | Metric |
| :--- | :--- | :--- |
| Policy | written usage policy, age gating, sensitive-topic policy | coverage of categories |
| Model | training-time alignment, system-prompt constraints | refusal precision/recall on eval suite |
| Product | friction on risky paths, confirmations, parental controls, rate limits | block rate, abandon rate |
| Output | classifiers (self-harm, sexual content, PII), safe-completion | false positive rate per category |
| Operations | reporting flow, appeals, incident review, red team cadence | time-to-takedown |

**Key points**

- The hardest part is **false positives**: an over-blocking filter gets disabled by the product team within a quarter. Measure both directions and tune per category.
- Safe-completion beats refusal for many categories — answer the benign core, decline the harmful edge; users route around blunt refusals to worse tools.
- Minors change the calculus: age estimation, stricter defaults, and no "just ask again" path.
- Every safety feature needs an owner and a dashboard, or it silently rots after launch.
- Evaluate on *your* traffic distribution, not a public benchmark; jailbreak suites go stale in weeks.

**中文要点**：分四层（政策/模型/产品/运营）各有指标；最难是误杀率（过严的过滤会被产品下掉），很多场景"安全补全"优于硬拒答，且要按自家流量评估。

---

### 24. How would you design safeguards for an AI system that can take actions on behalf of a user?

**Asked at**: OpenAI

**Anchor answer**

The core principle: **authority must be scoped, revocable, and no larger than the user's own.**

| Safeguard | Design | Threat it blocks |
| :--- | :--- | :--- |
| Delegated credentials | OAuth scopes, per-action consent, short-lived tokens | over-broad standing access |
| Action classification | read / write / irreversible / money-moving | uniform treatment of `rm` and `ls` |
| Confirmation gate | human approves irreversible or out-of-pattern actions | prompt injection → exfiltration |
| Limits | per-action caps, daily spend, recipient allowlist | runaway loops |
| Dry run / preview | show the exact diff, payload, recipients | "it did something I didn't read" |
| Audit | immutable action log with the triggering context | incident forensics |
| Kill switch | user-visible revoke-all + auto-pause on anomaly | ongoing damage |

**Key points**

- **Reversibility is the best safety property**: prefer trash + undo window over permanent delete; you cannot evaluate every action but you can make most of them undoable.
- Prompt injection means *any* content the agent reads is an instruction source — so the confirmation must show the action, not the model's summary of it.
- Scope tokens per action, not per session; a session-wide token granted at turn 1 is a standing liability.
- Detect anomaly by behaviour (new recipient domain, 10× normal volume), not by intent.
- Test the guardrails with red-team agents, and log the *blocked* actions — they are your best signal.

**中文要点**：权限不超过用户本人、按动作分级（读/写/不可逆/动钱）、不可逆操作必须人工确认并展示真实动作而非模型摘要、一切可撤销 + 全量审计。

---

### 25. An enterprise customer says: “We want AI to automate our claims processing.” You're the engineer in the room. What do the first two weeks look like?

**Asked at**: OpenAI

**Anchor answer**

Two weeks is for **finding the constraint**, not building the system. Claims processing is a workflow with a long tail, and the tail decides the ROI.

| Days | Activity | Output |
| :--- | :--- | :--- |
| 1–3 | Shadow 5–10 real claims end to end; count handoffs | process map with volume per step |
| 3–5 | Pull 100–200 historical claims with final outcomes | a labelled sample |
| 5–8 | Baseline the model on the sample: what % resolved correctly with citations? | a real number, not a demo |
| 8–10 | Identify the tail: which 20% of cases cause 80% of the effort | where automation actually pays |
| 10–14 | Propose the split: auto-adjudicate / assist-with-review / route-to-human; define KPIs, guardrails, integration points | a scoped pilot with acceptance criteria |

**Key points**

- Do not start with "we'll build an agent"; start with **which decision** and what its error cost is. A 2% error rate on payout decisions is not a demo metric.
- Find the humans' actual process (spreadsheets, exceptions, tribal rules) — most of the logic is undocumented, and it's the real spec.
- The believable first deliverable is **assist with citations**: extract fields, cite the policy clause. Full automation follows evidence.
- Name the compliance constraints early (audit trail, explainability, PII residency); they shape the architecture more than the model choice.
- Set the pilot's success metric with the customer in the room, in their units (cycle time, touch time, rework rate).

**中文要点**：头两周是"找到约束与长尾"，不是搭系统：跟单、抽样、跑基线、找 20% 高投入案例，再提"自动/辅助/转人工"的分工与验收指标。

---

### 26. Do you have experience working with APIs? Are you used to working with C-suite executives?

**Asked at**: OpenAI

**Anchor answer**

Two questions, both asking "can you operate at the customer boundary". Answer with concrete artifacts, not adjectives.

| Question | Weak | Strong |
| :--- | :--- | :--- |
| API experience | "Yes, I've used REST APIs" | "I built the integration for X: auth via OAuth, retries with idempotency keys, a 3k QPS webhook consumer; I owned the 429 storm after launch" |
| C-suite | "I'm comfortable presenting" | "I ran the weekly steering call for a 6-figure deployment; I cut the deck to one page: decision needed, risk, cost" |

**Key points**

- For APIs, name the hard parts: versioning, pagination, rate limits, auth refresh, error taxonomy, idempotency. Anyone can call `GET /users`.
- For executives, the currency is **decision, risk, and money** — not architecture. Say how you compressed detail and what you escalated.
- Show one moment where you pushed back on a customer/exec and won or lost, with the reasoning.
- If you lack a specific experience, say what's adjacent and offer the nearest concrete example — never claim a false one, FDE interviewers check.
- Ask them a question back: "what does the C-suite actually decide in these engagements?" — it shows you know the role.

**中文要点**：两问都在考"客户边界能力"：API 要讲版本化/限流/幂等/错误分类这类硬骨头，面对高管要讲"决策-风险-钱"并把细节压缩成一页。

---

### 27. What is your favourite product and why?

**Asked at**: OpenAI

**Anchor answer**

The trap is picking a product you love as a *user*. Pick one you can analyse **as an engineer**: what constraint it solved, what it deliberately gave up, what it got wrong.

Structure:

1. **Name it and the one-sentence thesis** of why it's good ("it made X cheap enough that people changed behaviour").
2. **The key design bet** — and the alternative it rejected.
3. **The mechanism** — one concrete technical or product detail that makes the bet work.
4. **What it sacrifices** — every good product is bad at something.
5. **What you'd change** — one specific thing, with a reason.

**Key points**

- Pick something adjacent to the role (developer tools, consumer AI, marketplaces) so the analysis is relevant to what you'd build.
- Have a **mechanism**, not a feeling: "Spotify's Discover Weekly works because it puts a constrained, ephemeral artifact in a social context — the playlist is shareable, the algorithm is invisible."
- Naming the sacrifice is what separates analysis from fan mail: "it chose latency over richness", "it chose a closed ecosystem over extensibility".
- Avoid the two clichés: iPhone (too generic, invites "why not Android?") and something you built yourself.
- Keep it to 90 seconds, then invite a follow-up — this question is a conversation starter, not a monologue.

**中文要点**：别当用户夸产品，要当工程师拆：核心赌注、被否方案、实现机制、牺牲了什么、你会改哪一点；90 秒讲完并留话头。

---

### 28. Tell me about a time you made a mistake.

**Asked at**: OpenAI

**Anchor answer**

Same discipline as any failure story: **a real, owned, specific error with a mechanism-level lesson.**

| Beat | Content | Test it passes |
| :--- | :--- | :--- |
| The mistake | a decision you made with incomplete info | ownership, not blame |
| Why it was plausible | the reasoning at the time | judgement, not carelessness |
| The impact | concrete cost (days, money, a customer) | honesty about consequences |
| Detection | how and when it surfaced | self-awareness |
| Repair | what you did, including telling people | responsibility |
| The mechanism | the check you added | learning that transfers |

**Key points**

- Choose a mistake with a **technical** dimension if you're interviewing for an engineering role; "I communicated poorly" is a weaker story than "I shipped a migration without a rollback plan".
- The most credible shape: you optimised the wrong thing (built a cache before measuring, chose a queue for a workload with no burst).
- Do not pick a mistake that hurt someone else and that you still describe defensively — interviewers read tone.
- The lesson must be a **process change**, e.g. "now every migration ships with a rollback rehearsal in staging", not "I'm more careful".
- Once you've answered, stop. Over-explaining reads as anxiety.

**中文要点**：选一个当时判断合理的技术性失误，讲清代价、发现延迟、修复动作，落点必须是一个流程机制（如迁移必须演练回滚），而不是"以后更小心"。

---

### 29. Tell me about a time you had a conflict with someone. How did you resolve it and what did you learn?

**Asked at**: OpenAI

**Anchor answer**

Conflicts worth telling are **disagreements about a technical decision**, resolved by evidence — not personality clashes.

| Beat | What to say |
| :--- | :--- |
| The disagreement | two defensible positions (not "they were wrong") |
| The stakes | what would break if the wrong one won |
| Your move | you separated the claim from the person and made it testable |
| The resolution | a measurement, a small experiment, or a scoped decision with a review date |
| The outcome | you were wrong / they were wrong / it was a false dichotomy |
| The learning | what you now do earlier |

**Key points**

- The strongest version ends with **you being partly wrong** — it proves you can be in a conflict without needing to win.
- Say how you made it cheap to disagree: "I proposed a 2-day spike instead of arguing for a week."
- Describe the other person's position **charitably and accurately**; the interviewer has heard this story before and can spot a strawman.
- Name the structural fix (design doc with a decision owner, or a written "disagree and commit" note), not just the interpersonal one.
- Avoid: a conflict with a manager where you "escalated and won" — it reads as political rather than collaborative.

**中文要点**：讲技术分歧而非人际冲突；关键动作是把主张变成可测的实验（如两天 spike），结局最好包含"我部分错了"，落点是结构性机制。

---

### 30. Tell me about a time you had conflicting priorities with stakeholders and how you secured alignment.

**Asked at**: OpenAI

**Anchor answer**

The question tests **how you make a trade-off explicit when you don't own the decision.**

| Beat | Content |
| :--- | :--- |
| The collision | two stakeholders with incompatible asks, and the deadline that made it real |
| Why it couldn't be split | the shared resource (your team, one migration, one launch slot) |
| What you did | wrote the trade-off down: options, cost, risk, who is affected |
| The forcing function | a decision meeting with an owner, or a sequenced plan with dates |
| The outcome | what shipped, what slipped, who was told and when |
| The learning | how you now surface conflicts earlier |

**Key points**

- The answer should contain a **written artifact**: a one-page options memo. Verbal alignment evaporates; a doc with a decision row does not.
- Name the actual scarce resource — usually engineer-weeks or a launch window, not "time".
- Best practice to describe: **sequence, don't average.** Say "A in Q1, B in Q2, and here's what B's delay costs", rather than trying to do 60% of each.
- Show that you escalated *early* and with a recommendation, not a complaint.
- Include the cost of the choice explicitly — alignment without a stated loss is just optimism.

**中文要点**：把稀缺资源（人周/发布窗口）和代价写成选项备忘录，找有决策权的人开会定，采用"排序而非折中"（A 先 B 后）并明确延期成本。

---

### 31. What is the project you are most proud of?

**Asked at**: OpenAI

**Anchor answer**

"Proud" invites a story about **craft**, so pick the project where the interesting part was the engineering decision, not the scale.

| Element | Content |
| :--- | :--- |
| The one-liner | what it does and for whom, in one sentence |
| Why it was hard | the constraint that made the obvious approach fail |
| Your specific contribution | the part you personally designed (be precise) |
| The decision you're proud of | one choice that was non-obvious and paid off |
| Evidence | a number you measured, and how |
| What it cost | the thing you traded away |
| What you'd do differently | one concrete change |

**Key points**

- Precision about your contribution is the whole test; "we built" for ten minutes then "I" for thirty seconds reads as inflated.
- The proud part should be a **decision**, not an outcome: "I argued for a boring single-table design and it survived 10× growth" beats "we hit 1M users".
- Include the measurement — pride with no evidence sounds like nostalgia.
- Pick something you can go three levels deep on; the follow-ups ("why not X?", "what broke?") are where this question is actually scored.
- Don't pick the most impressive project if you can't defend its details; a smaller project you owned end to end scores higher.

**中文要点**：讲"你个人的决策"而非团队成果与规模，并给出你实测的数字；选一个能连挖三层细节的项目，小而全优于大而虚。

---

## 附：这 31 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 编码基本功（事务/ORM/并发） | 1、2、3、6、7 |
| 代码品味与重构 | 5、4 |
| ML 基础是否扎实 | 8、9、10、11 |
| 模型内部理解 | 12、13、14 |
| 生产 agent 可靠性 | 15 |
| 系统设计广度 | 16、17、18、19、20 |
| 线上问题定位（最像真实工作） | 21、22 |
| 安全设计 | 23、24 |
| 客户场景（Applied / FDE / SA） | 25、26 |
| 文化轮 | 27、28、29、30、31 |
