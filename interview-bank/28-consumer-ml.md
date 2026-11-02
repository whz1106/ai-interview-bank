# 28 · Consumer-Scale ML Companies

> 14 题。出处：`README.en.md` § Consumer-Scale ML Companies (Uber, Netflix, LinkedIn, Airbnb, Pinterest, Spotify)。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Implement a streaming top-k with a bounded-memory sketch; implement a sliding-window rate counter.

**Anchor answer**

**Reference implementation**

```python
import heapq

class SpaceSaving:
    """O(k) memory, one pass. Estimate - true_count <= N/k."""
    def __init__(self, k):
        self.k = k
        self.count = {}     # item -> estimated count
        self.err = {}       # item -> max over-estimate

    def add(self, x, c=1):
        if x in self.count:
            self.count[x] += c
        elif len(self.count) < self.k:
            self.count[x], self.err[x] = c, 0
        else:
            victim = min(self.count, key=self.count.get)
            inherited = self.count.pop(victim)
            self.err.pop(victim)
            self.count[x] = inherited + c     # inherit, so we never under-count
            self.err[x] = inherited           # the price: over-estimate <= inherited

    def topk(self):
        return heapq.nlargest(self.k, self.count.items(), key=lambda kv: kv[1])


class SlidingWindowCounter:
    """Ring of per-bucket counts; error <= (bucket_size / window) of the true rate."""
    def __init__(self, window_s, n_buckets=10):
        self.n = n_buckets
        self.size = window_s / n_buckets
        self.buckets = [0] * n_buckets
        self.epoch = [-1] * n_buckets

    def _idx(self, t):
        return int(t // self.size) % self.n

    def incr(self, t, c=1):
        e, i = int(t // self.size), self._idx(t)
        if self.epoch[i] != e:
            self.epoch[i], self.buckets[i] = e, 0
        self.buckets[i] += c

    def count(self, t):
        first = int((t - self.size * self.n) // self.size)
        return sum(b for b, e in zip(self.buckets, self.epoch) if e >= first + 1)
```

**Key points**

- **Space-Saving's guarantee is one-sided**: it never under-reports a heavy hitter, which is exactly what you want for blocking/abuse decisions. Count-Min Sketch is the alternative when you need point queries, at `εN` error with more memory.
- The `k` you keep must be ≫ the `k` you return — keep 10–20k counters to return top 100, or the estimate error swamps the ranking.
- Sliding window: **bucketed is approximate**; a log of timestamps is exact but memory grows with rate. Two-bucket is the cheapest and is what most rate limiters ship.
- Concurrency: shard per key or use per-thread sketches and merge — a global lock per `add` costs more than the sketch.
- Reset by epoch, never by sweeping.

**中文要点**：Space-Saving 只保留 k 个计数器、新元素继承被逐出者的计数，保证"重尾不会低估"（误差 ≤ N/k），适合风控；滑动窗口用环形分桶换内存，误差约一个桶的比例，双桶足够多数限流器。

---

### 2. Your offline metric improved but the online A/B did not. Enumerate the reasons this happens and how you would tell them apart.

**Anchor answer**

Nine causes, grouped by where the fault is:

| Where | Cause | How to tell |
| :--- | :--- | :--- |
| **Offline eval** | proxy metric not causal (NDCG up, watch time flat) | check if the gain concentrates on items users never see |
| | **leakage** — future info, or train/eval split by random rows from the same session | rebuild eval with a strict time split; the gain often vanishes |
| | evaluation set is not the serving distribution (only logged impressions, position-biased) | reweight by propensity; compare to a randomized holdout |
| **Model** | **train/serve skew** — feature computed differently online | log served features, compare to offline values daily |
| | gain exists only in a small slice, diluted in the aggregate | pre-register segment-level analysis |
| **Experiment** | underpowered / peeking at sequential tests | compute MDE before launch, use sequential testing correctly |
| | **guardrail metric** regressed and cancelled the win | report the full metric tree, not the north star |
| | interference between overlapping experiments / network effects | check exposure ratios, run A/A, use switchback for marketplace effects |
| **Product** | novelty or primacy effect; the improvement is real but decays | look at the day-by-day treatment effect curve |

The default prior: **most "offline up, online flat" cases are leakage or a metric that does not measure what the product values.** The most useful single check is a **counterfactual replay with IPS** — it tells you whether the new policy would have been better on the *logged* distribution, separating model quality from distribution shift.

**中文要点**：先分三类——离线评估不可信（泄漏、代理指标非因果、分布偏差）、训练服务不一致、实验设计问题（功效不足、护栏指标抵消、实验间干扰）；最快的判定是 IPS 反事实回放加一次严格的按时间切分重评。

---

### 3. Explain position bias in ranking data and how you would debias training.

**Anchor answer**

Users only click what they **examine**, and examination depends on rank. Logged clicks therefore encode `P(click) = P(examine | rank) · P(relevant | examined)`. Training a ranker on raw clicks teaches it "position 1 is good", which is true only because you put things there — a self-reinforcing loop where new items can never prove themselves.

Four debiasing approaches:

| Approach | Mechanism | Cost |
| :--- | :--- | :--- |
| **Position as a feature, train-only** (PBM) | model learns `P(examine\|pos)` as a separate factor, dropped at serving | simple, needs enough randomization to identify |
| **Inverse propensity weighting (IPW)** | weight each click by `1/P(examine\|pos)`, estimated from a **randomized slice** | unbiased but high variance; clip propensities |
| **Click models** (cascade, DBN) | latent examination variable per position, EM fit | closer to reality, more moving parts |
| **Randomized exploration traffic** | show a small % of traffic with shuffled/exploration ranking; use it as unbiased labels | the only clean ground truth; costs revenue |

Practical recipe: keep a 1–5% randomized slice for propensity estimation and unbiased evaluation, train the main model with IPW on the logged data, and evaluate off-policy with **IPS/DR** plus the randomized slice. Report both.

Failure mode to name: propensities learned on one UI and reused after a layout change — the examination curve shifts and every IPS estimate is silently biased.

**中文要点**：点击 = 被看到 × 相关，直接学会把"位置 1 好"学进去并自我强化；四条解法是位置特征仅训练用（PBM）、IPW（需随机流量估倾向）、点击模型、以及保留 1~5% 随机探索切片做无偏评估；UI 改版后倾向曲线失真是最常见的坑。

---

### 4. How do you design a feature store, and what causes training/serving skew?

**Anchor answer**

A feature store is three contracts, not a database:

| Contract | Implementation | Why |
| :--- | :--- | :--- |
| **Definition once** | features declared in code, same transform DSL for batch and streaming | two code paths is the root of skew |
| **Two stores, one definition** | offline (warehouse, columnar, for training/backfill) + online (KV, ms latency, TTL) | latency vs scale |
| **Point-in-time correct join** | as-of timestamp join keyed by entity id; features never read the future | otherwise offline metrics are fantasy |

Add: versioning of feature definitions (a backfilled redefinition without a version bump silently invalidates every model), freshness SLAs, and null/default policy declared with the feature.

Causes of train/serve skew, ranked by how often they bite:

1. **Duplicate logic** — SQL in the warehouse, Java/Python in the service; they diverge on day 90.
2. **Time-travel errors** — offline window ends "now", serving window ends at request time; different aggregates.
3. **Defaulting** — missing feature → 0 offline, → global mean online (or vice versa).
4. **Cold start** — new user/item has no history; the model has never seen that all-zero row.
5. **Streaming vs batch boundaries** — 5-min tumbling vs event-time, timezone offsets, late-arriving events.
6. **Preprocessing drift** — normalization fitted on a batch snapshot, applied forever.

Detection: **log the exact feature vector served** and run a daily parity job against offline recomputation; alert on P99 absolute difference. That one job catches most of the list.

**中文要点**：特征库的核心是"定义一次、离线/在线两存、point-in-time 正确 join"加版本化；skew 六源——双份逻辑、时间窗不一致、缺失默认值不同、冷启动、批流边界/时区、归一化快照过期；最有效的检测是记录线上特征向量并每日与离线重算比对。

---

### 5. Design the ETA prediction system for a ride-hailing marketplace. What features, what model, how do you serve it in <100 ms?

**Anchor answer**

Target: `p50` and `p90` remaining time, not a single mean — riders care about the tail, and dispatch needs the distribution.

**Architecture**

| Stage | Design |
| :--- | :--- |
| Route | map service returns polyline + segment decomposition (already cached per OD pair) |
| Features | per-segment historical speed by `(road class, hour, dow, weather, holiday)`; live traffic feed; pickup wait; driver supply/demand in origin cell; driver's own historical speed; turn/intersection count |
| Model | **segment-level** GBDT (or a small MLP with segment embeddings) predicting travel time per segment, then sum; separate model for pickup ETA |
| Post | quantile heads → p50/p90; clamp against a physical speed bound; smooth across adjacent requests for the same OD |
| Serve | feature fetch from online store in **one batched multi-get**, map polyline cached, model <5 ms, budget 100 ms with 30 ms for features |

Key design points:

1. **Predict per segment, not per trip.** A 40-minute trip's error compounds; segment-level lets live traffic update the remaining part and gives you an explainable decomposition.
2. **Decompose ETA into pickup + ride.** They have different physics and different failure modes (dispatch vs traffic).
3. **Quantiles, not means.** Late delivery of a p90 promise is a product problem.
4. **Cache + fallback**: if the map service times out, fall back to the OD historical table and widen the interval — never block on it.

Monitoring: pinball loss per quantile, plus calibration per city/hour. The classic bug is a new highway interchange invalidating the road graph for a week.

**中文要点**：按**路段**预测再求和（误差不随行程长度累积，且能融合实时路况），分接驾与行程两段，输出分位数而非均值；服务端并行批量取特征 + 缓存路线，100ms 预算里 30ms 给特征，地图超时要降级而不是阻塞。

---

### 6. Design a personalised feed ranking system with a two-stage candidate generation and ranking architecture.

**Anchor answer**

| Stage | Candidates | Latency | Method |
| :--- | :--- | :--- | :--- |
| **Candidate gen** | 10^6 → 10^3–10^4 | 20–40 ms | several parallel sources: two-tower ANN (user × item embeddings), co-visitation/graph, followed authors, trending in locale, fresh items, retrieval by recency |
| **Light ranker** | 10^4 → 10^3 | 10 ms | cheap GBDT/embedding dot on ~50 features; prunes before the expensive model |
| **Heavy ranker** | 10^3 → 10^2 | 40–60 ms | DNN over hundreds of features, **multi-task**: click, dwell, like, hide, report |
| **Re-rank** | 10^2 → 50 | 5 ms | diversity (DPP / MMR), freshness quota, author dedupe, exploration slot, policy filters |

Details that decide quality:

1. **Retrieval must use hard negatives** — sampled from "shown but not clicked", not random. Random negatives make the embedding space useless.
2. **Multi-task, then combine**: separate heads with a calibrated weighted sum (or a learned fusion). Do not train one blended label; you lose the ability to change the product tradeoff without retraining.
3. **Label windows**: dwell > 5 s as a satisfaction proxy; a click alone is a clickbait signal.
4. **Position bias**: feed the ranker position as a train-only feature.
5. **Diversity is a constraint, not a bonus**: enforce per-author/per-topic caps in re-rank.

Failure modes: feedback loops (popularity lock-in), the retrieval and ranking objectives drifting apart, and a heavy ranker whose calibration is never checked — so the blend weights are meaningless.

**中文要点**：四段式——多源召回（双塔 ANN、协同、关注、热门、新内容）→ 轻排 → 多任务重排（点击/停留/负反馈分头）→ 多样性重排；召回必须用"曝光未点击"做难负样本，位置偏差要作为仅训练特征，多样性用约束而不是加权。

---

### 7. Design a content recommendation system for a streaming catalogue, including cold-start for new titles and new users.

**Anchor answer**

The catalogue is only ~10^4–10^5 titles, so **retrieval is cheap and the hard problem is cold start plus the objective**.

| Component | Design |
| :--- | :--- |
| Item representation | content embeddings: genre/tags, cast/director, **trailer + audio + subtitle text embeddings**, artwork; plus collaborative embedding once interactions exist |
| User representation | sequence model over the last N watched titles (with completion), plus long-term taste; household vs profile ambiguity is real |
| Cold title | map metadata → embedding via a learned projection trained so that `metadata_embedding ≈ collaborative_embedding` for warm titles; then it lands next to similar titles |
| Cold user | onboarding taste picks + locale/language popularity + **contextual bandit** for fast adaptation; treat first session as exploration |
| Ranking objective | not CTR — expected **watch time** and completion, with retention as the north star |
| Validation | time-based split, then **interleaving** online (far higher sensitivity than A/B for ranking changes) |

Two things interviewers push on:

1. **Cold start is a matrix-completion problem with side information**, so the design question is how you generate the first 100 impressions. Answer: guaranteed exploration slots per new title, sized to get a trustworthy estimate fast.
2. **Optimizing time-in-app creates clickbait autoplay.** Weight by completion and satisfaction (thumbs, "not for me"), and audit the correlation between predicted and realized value per title cohort.

Failure mode: new titles never get impressions because the ranker's confidence is low → they stay cold → the model is right. Fix with an explicit exploration budget outside the ranker.

**中文要点**：点播库小（1e4~1e5），难点在冷启动与目标函数：新片用元数据/预告片向量投影到协同空间并给固定探索曝光位，新用户靠引导选择 + 上下文 bandit；排序目标是完播与留存而非点击，排序改动优先用 interleaving 验证。

---

### 8. Design “people you may know” / job-recommendation ranking at a professional network's scale.

**Anchor answer**

Two products sharing one graph, with different objectives: **PYMK wants accepted invitations that stick; job rec wants applications that progress.**

**PYMK**

| Stage | Design |
| :--- | :--- |
| Candidates | graph traversal 2–3 hops (friends-of-friends), same employer/school/cohort, address-book upload, co-engagement; sampled, not exhaustive |
| Features | common connections (normalized), graph distance, mutual interaction, shared attributes, reciprocity, recency of the other person's activity |
| Model | GBDT or GNN (PinSage-style random-walk embeddings) predicting `P(invite) · P(accept) · P(retained after 90 days)` |
| Post | dedupe, hide already-invited, diversity across clusters, **"why you may know" explanation** — its absence trains users to distrust the feature |

Scale realities: 10^9 members means **precompute embeddings and neighbor lists offline**, do only a bounded traversal + ANN at request time, and cap candidates per user. The graph mutates constantly, so incremental updates and a day-old snapshot is the norm.

**Job rec** is two-sided: candidate-side ranking (`P(apply) · P(qualify) · P(interview)`) and recruiter-side (`P(response)`), with freshness and diversity constraints. The key trap is optimizing applies — a flood of unqualified applications destroys recruiter trust and degrades the market for everyone. Gate on qualification, and never let a paid signal silently enter an organic ranking without disclosure.

**中文要点**：PYMK 用 2~3 跳图遍历 + 共同好友/同公司特征 + GNN/GBDT 预测"邀请→接受→90 天留存"，并必须给"为什么推荐你认识"；职位推荐是双边市场，别只优化投递量——不合格投递会摧毁招聘方信任。

---

### 9. Design dynamic pricing / surge for a two-sided marketplace and describe the feedback loops that can go wrong.

**Anchor answer**

Objective first: **maximize completed matches**, not revenue per trip. Surge is a control signal, so the design is a closed-loop controller with a model inside.

| Element | Design |
| :--- | :--- |
| State | per geo-cell × time bucket: request rate, available drivers, idle time, expected demand from events/weather/transit |
| Model | demand elasticity (how riders drop off vs price) + **supply response** (how drivers reposition toward surge) |
| Action | multiplier per cell, bounded (`1.0–3.0`), with predictability policy (show the multiplier before booking) |
| Control | re-solve every few minutes; damp the change rate (a multiplier that jumps 20% per tick oscillates) |
| Constraints | regulatory caps, competitor parity, fairness (don't repeatedly surge low-income areas), no surge on safety-critical needs |

**Feedback loops that go wrong** — the part that gets you hired:

1. **Driver flocking**: surge attracts drivers → supply overshoots → price crashes → drivers leave → next overshoot. Fix: damp, and include the *expected future* supply in the controller.
2. **Demand learning loop**: you only observe demand at the price you charged, so a naive elasticity model learns "high prices have low demand" and never explores. Fix: explicit randomization in a small traffic slice.
3. **Competitor coupling**: two apps' surges chase each other.
4. **Data poisoning of training**: the model trains on outcomes from its own policy → self-confirming. Fix: off-policy evaluation with propensities, and log the counterfactual context.

Measure with **switchback or geo-split experiments**, not user-level A/B — supply is shared, so user-level randomization leaks treatment across arms.

**中文要点**：目标是"完成匹配量"而非客单价；必须建含司机重定位响应的控制器并做阻尼；四条会翻车的反馈环是司机潮汐、只在自己定价下观测需求导致弹性学偏、竞对互相追涨、以及用自身策略数据训练自我强化；实验要用 switchback 而不是用户级 A/B。

---

### 10. Design a visual search system: user uploads an image, you return visually similar in-catalogue items.

**Source**: https://outcomeschool.com/blog/how-do-image-embeddings-work

**Anchor answer**

Pipeline, with the decision that matters at each step:

| Stage | Choice | Why |
| :--- | :--- | :--- |
| Query prep | detector/crop to the object, background removal, product-region crop | **the single biggest win** — a photo of a sofa in a living room embeds mostly the room |
| Embedding | ViT/CNN trained with metric learning (triplet / ArcFace / contrastive) on in-domain catalogue pairs | off-the-shelf ImageNet features cluster by texture, not by product identity |
| Index | HNSW or IVF-PQ over 10^6–10^8 vectors; PQ for memory, HNSW for latency | recall@k at fixed memory |
| Filter | category/attribute model, price band, availability | visual similarity ≠ shoppable |
| Rerank | cross-encoder on (query, item) crops, or multi-vector (global + local patches) | catches attribute mismatches that a single dot product cannot |
| Dedupe | cluster near-identical SKUs, return one per cluster | twenty photos of the same shirt is a broken result page |

Numbers to reason with: 512-d float32 = 2 KB/vector → 100M items ≈ 200 GB; PQ to 64 bytes → 6.4 GB, feasible in RAM. That ratio is why quantization is not optional at catalogue scale.

Failure modes: **color/pattern dominates** (blue sofa returns blue chairs) → train with hard negatives that share color but differ in category; **background dominance** → crop; **near-duplicate flooding**; and cold items with only a studio photo — augment with generated backgrounds so the index is robust to user photos.

Evaluation: recall@k on human-labeled (query, item) pairs, plus a held-out set of real user photos — never on studio images only.

**中文要点**：视觉搜索最大的收益来自查询端裁剪去背景；嵌入要用域内度量学习（ArcFace/三元组）训练，索引用 PQ 压到 64 字节否则亿级内存爆掉；难负样本要"同色不同类"，结果页必须近重复聚类去重。

---

### 11. Design a fraud-detection system with heavy class imbalance and an adversarial opponent.

**Anchor answer**

Two constraints shape everything: **positives are ~0.1–1%**, and **the label arrives weeks later** (chargeback) while the adversary adapts.

| Layer | Design |
| :--- | :--- |
| Features | velocity (counts per card/device/IP in 1h/24h/7d), entity graph (shared device, address, card BIN), sequence of actions, amount vs user history, merchant risk, **time since account creation** |
| Model | GBDT on tabular + graph features; sequence model for session behaviour; rules kept as a parallel layer, not replaced |
| Imbalance | **cost-sensitive learning** (miss = chargeback + fee, FP = review cost), undersample the majority and **recalibrate the score afterwards** — downsampling destroys calibration if you forget |
| Labels | delayed chargebacks + human case review; use **early proxy labels** (manual blocks, confirmed fraud) with a model that handles label delay |
| Decision | threshold set by **review-queue capacity**, ranked by expected loss, not by F1 — you review the top N per day |
| Adversary | retrain on a short cadence, monitor feature distributions for evasion, keep a "recent attack" shadow model, and never let a single rule stay static long enough to be reverse-engineered |

Key points a practitioner will say out loud:

1. **PR-AUC and precision@k, not AUC** — with 0.5% positives, ROC-AUC of 0.99 is compatible with terrible precision.
2. **Downsampling correction**: `p = p_s / (p_s + (1-p_s)/β)` — forgetting this means thresholds are off by orders of magnitude.
3. **Leakage is the standard bug**: features computed after the outcome (e.g. "account was blocked") inflate offline metrics to 0.99 and collapse online.
4. Adversarial drift means the model **decays**; measure lift over rules monthly and expect it to shrink.

**中文要点**：类别极不平衡 + 标签延迟 + 对手会适应；用代价敏感学习与下采样（并记得重校准分数），按审核队列容量定阈值、看 precision@k 与 PR-AUC；最常见的坑是用了结果之后才产生的特征造成泄漏。

---

### 12. Design an LLM-powered customer-support assistant on top of an existing help centre, with escalation to humans.

**Anchor answer**

| Layer | Design | Failure it prevents |
| :--- | :--- | :--- |
| Retrieval | hybrid BM25 + dense over help-centre articles, chunked by section; **always cite** | confidently wrong policy answers |
| Intent + tools | classifier routes to: answer from docs / call backend API (order status, refund eligibility) / escalate | the model inventing account state |
| Guardrails | policy scope (no promises outside policy), PII redaction, no free-form financial commitments, output schema with `answer`, `citations`, `confidence` | the bot promising a refund it cannot issue |
| Escalation | confidence below threshold, low retrieval score, **policy exception**, angry sentiment, user asks, or 2 failed turns | infinite loops, rage |
| Handoff | transcript + summary + what was tried + retrieved sources | the customer repeats everything |
| Evaluation | groundedness (every claim in a citation), answer accuracy on a labeled set, containment rate, CSAT, **escalation precision/recall** | optimizing containment by refusing to escalate |

Serving: cache FAQ answers, small model for classification and simple retrieval answers, large model only for multi-step policy reasoning. Log every conversation for a weekly review of the escalated ones — that is your training data.

The honest tradeoff: **containment rate is a trap metric.** A bot that answers 80% but is wrong 10% of the time is worse than one that answers 60% and escalates cleanly. Track "wrong answer → later human correction" as a first-class counter and make the bot's `cannot answer` path cheap and fast.

**中文要点**：RAG + 引用强制 + 后端工具取真实状态 + 政策护栏（不得承诺退款）；升级触发要含低置信、检索差、政策例外、情绪与连续失败，交接带摘要；containment 是陷阱指标，必须同时统计"答错后被人工纠正"。

---

### 13. How do you monitor a deployed ranking model for drift, and what triggers a retrain?

**Anchor answer**

Monitor four layers, because the top ones are fast and the bottom ones are true:

| Layer | Signals | Latency |
| :--- | :--- | :--- |
| **Data health** | null rate, feature freshness, schema change, join coverage, serving-vs-offline parity | minutes |
| **Feature drift** | PSI / KL per feature vs training distribution, new-category share, embedding MMD | hours |
| **Prediction drift** | score distribution, calibration (ECE), prediction entropy | hours |
| **Outcome** | CTR / dwell / conversion by slice, and the label-based metric once labels land | days |

What should actually **trigger a retrain** — and what should not:

1. **Do**: a statistically significant outcome regression on a stable slice; feature coverage dropping; a known distribution change (new market, new UI); scheduled cadence tied to the data's natural half-life.
2. **Don't**: a raw feature-drift alert with no metric movement. Retraining on drifted data *without labels for the new regime* just relearns the old distribution faster and can amplify a feedback loop.

Guardrails for the retrain itself: shadow-evaluate against the live model on recent traffic, gate on the same slice suite, canary 1–5%, and keep a **frozen human-labeled holdout** that never enters training — otherwise your only evaluation is contaminated by the model's own choices. Book a control chart / sequential test so you do not chase noise on week 1 and miss a real 3% drop on week 5.

**中文要点**：四层监控——数据健康（新鲜度、空值、线上线下一致）→ 特征漂移（PSI/MMD）→ 预测/校准漂移 → 分片业务指标；只有"业务指标显著下滑或分布已变且新数据有标签"才重训，纯漂移告警不该自动触发，重训前必须影子评估 + 冻结人工标注集。

---

### 14. Tell me about a model you shipped that made a measurable business difference, and one that did not.

**Anchor answer**

Both halves are graded, and the second half is where most candidates collapse into vagueness. Prepare both with numbers.

For the win, four beats:

1. **Baseline → after, in a business unit**: "+1.8% session watch time, p<0.01, 3-week A/B on 12% of traffic" — not "+15% CTR" without a denominator or a significance statement.
2. **The mechanism**, one sentence: "the retrieval stage could not surface fresh titles, so we added a content-embedding source".
3. **What you had to give up**: latency +8 ms, an extra serving cluster, a re-ranking cap. A win with no cost reads as an unverified claim.
4. **What nearly killed it**: the honest near-miss — offline metric flat, the gain only appeared after fixing feature parity.

For the failure, the structure interviewers actually reward:

- **State the outcome plainly** ("we shipped it, then rolled it back in 9 days; the guardrail on unsubscribes regressed 4%").
- **Name the wrong assumption you personally held** ("we assumed dwell time meant satisfaction; it also captured confusion").
- **Show the detection**: which signal caught it, how fast, what you changed in the process (a pre-registered guardrail, an interleaving test before the full A/B).
- **Do not blame stakeholders or data.** Owning the model decision is the point of the question.

One anti-pattern to avoid: choosing a failure that is actually a success in disguise ("it didn't ship but we learned a lot"). That reads as an inability to be measured.

**中文要点**：赢的那个给"基线→提升 + 业务单位 + 显著性 + 代价"；输的那个必须说清真实结果（上线 9 天回滚、护栏指标退 4%）、你自己错在哪个假设、是哪个信号发现的、以及之后流程改了什么——别甩锅给数据或业务方。

---

## 附：这 14 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 推荐系统主战场 | 6（两段式 feed）、7（点播冷启动）、8（PYMK/职位）、10（视觉搜索） |
| 指标与实验素养 | 2（离线涨线上不涨）、3（位置偏差）、13（漂移监控与重训触发） |
| 特征工程 | 4（特征库与训练服务偏差）、5（ETA 特征与 100ms 服务） |
| 双边市场与风控 | 9（动态定价的反馈环）、11（不均衡 + 对抗） |
| LLM 落地 | 12（客服助手与升级策略） |
| 手写代码 | 1（流式 top-k + 滑动窗口限流） |
| 软素质 | 14（有业务效果的模型 / 没效果的模型） |
