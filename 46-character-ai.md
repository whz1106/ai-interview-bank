# 46 · Character.AI

> 12 题。出处：`README.en.md` § Character.AI。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Live coding: build the prompt for the next turn under a fixed token budget. The catch is our prefix cache

**Source**: https://outcomeschool.com/blog/how-does-prompt-caching-work · https://outcomeschool.com/blog/context-engineering

**Anchor answer**

The trap: prefix caching only pays off on an **exact token-identical prefix**. So any budget logic that edits the front of the prompt (rewrite the persona card, re-render the summary, insert "turn 37 of 180") invalidates the cache for every request of that conversation, and you lose the win entirely.

Design: **two regions, one eviction event.**

```python
def build_prompt(persona, summary, turns, user_msg, budget, count):
    """Region A (frozen, cacheable): persona + greeting + summary + pinned facts.
       Region B (append-only): turns. Region C (tail): new user turn."""
    head = render(persona, summary, PINNED_FACTS)      # rendered identically every turn
    out_tokens = 200
    room = budget - count(head) - count(user_msg) - out_tokens
    kept = []
    for turn in reversed(turns):                        # take newest first
        if count(turn) > room:
            break
        kept.append(turn)
        room -= count(turn)
    kept.reverse()
    return head + kept + [user_msg]
```

Then the rule that matters: **do not evict one turn at a time.** When `kept` would drop below a floor (say 20 turns), fold the evicted turns into a new `summary` and accept one cache miss — an eviction *event*, not a per-turn tax. Amortized, a 180-turn dialogue pays a handful of prefill restarts instead of 180.

**Key points**

- Frozen prefix first: persona, greeting exemplars (they set the voice), pinned facts.
- Never interpolate turn counters, timestamps, or dynamic instructions into the prefix — a single differing byte kills reuse.
- Keep sampling/rendering byte-stable: same whitespace, same JSON key order.
- Reserve output tokens in the budget *before* filling history, or you truncate the model mid-reply.

**中文要点**：前缀缓存要求 token 前缀完全一致，所以预算逻辑绝不能动 prompt 开头（别塞轮次号/时间戳）；分「冻结前缀 + 只追加轮次」两区，超预算时不逐轮淘汰，而是攒够一批并入摘要、只承担一次缓存失效。

---

### 2. A conversation runs past the context window. What do you keep, and how do you decide?

**Source**: https://outcomeschool.com/blog/how-does-context-compaction-work · https://outcomeschool.com/blog/ai-agent-memory

**Anchor answer**

Do not summarize uniformly. Rank by **what the next turn can break without**, then allocate budget by class:

| Keep | Budget share (typical) | Why |
| :--- | :--- | :--- |
| Persona card + greeting | fixed, never evicted | Losing it is the drift bug |
| Canonical facts the user stated (name, job, pet, promises made) | fixed list, extracted once | Re-derivable only by re-asking, which users hate |
| Unresolved plot threads / open questions | ~10% | Dropping them breaks continuity noticeably |
| Recent turns verbatim (last ~15–30) | ~40% | Style and local coherence come from verbatim text, not a summary |
| Rolling summary of older turns | ~15% | Compressed index, not a transcript |
| Dropped | rest | Small talk, resolved threads, repeated greetings |

Decision rule: evict by **information density**, not recency alone — a turn that introduced a fact survives; three turns of "haha yeah" do not. Run a cheap extractor at each eviction boundary that emits memory items `{fact, salience: novelty × user-volunteered × emotional intensity}` and keep the top-k by salience, not by order.

Two failure modes worth naming: (1) **summaries drift toward generic** — after three summarization generations, the character's voice is gone; always keep verbatim exemplars alongside. (2) **memory must be updated, not appended** — a contradicted fact ("I got a new job") must overwrite the old item, or retrieval alternates between both and the character looks confused.

**中文要点**：按"下一轮没有它会不会坏"分级保配额：人设卡与用户陈述的关键事实永不淘汰，未解线性情留 10%，近 15–30 轮保留原文，更早的压成滚动摘要；淘汰看信息密度不看新旧；摘要多代后会变"通用腔"，必须保留原文风格样本，事实要覆盖更新而非追加。

---

### 3. Our serving cost is dominated by KV cache, not weights. Get it down by an order of magnitude and tell me what you give up

**Source**: https://outcomeschool.com/blog/kv-cache-compression

**Anchor answer**

Order of magnitude (10×) never comes from one trick. Stack them, and be honest about the bill:

| Lever | Typical factor on KV bytes | What you give up |
| :--- | :--- | :--- |
| GQA / MQA (fewer KV heads) | 4–8× vs MHA | Already assumed — retraining required, slight quality loss on fine recall |
| KV quantization (int8 / fp8, per-channel) | 2× | Small quality risk on outlier channels; needs calibrated scales per layer |
| **Cross-user prefix sharing** (same character's persona prefix) | 2–5× effective on *prefill* | Only works if the prefix is byte-identical; per-user memory blocks break sharing |
| Eviction of low-attention KV (attention-score or recency-window policies) | 2–4× at long context | **Exact recall of far-back content is gone** — acceptable for chat, fatal for legal/code |
| Offload to host memory / NVLink-attached pools | frees HBM, not compute | Adds PCIe latency; throughput drops if you swap mid-generation |
| Paged attention | ~20–40% HBM saved from fragmentation | Not really compression — it is waste removal, do it first |

The non-obvious one: in this product, the cheapest large win is usually **prefix sharing**, because most of a message's KV is the shared character card plus the common system prompt, not the user's own history. Cache keys must include persona version.

Hard truth to state: the 10× target requires accepting **degraded long-range recall**. Mitigate by keeping a compact memory block in the prompt rather than relying on raw KV, and by measuring recall on a long-context needle-from-earlier-session suite, not just perplexity.

**中文要点**：10 倍不是单点技巧：GQA + KV int8 + 跨用户前缀共享 + 低注意力 KV 淘汰 + 分层卸载叠加；最大且最便宜的常是"同角色共享人设前缀的 KV"；代价是远期精确召回被牺牲，必须用会话级长上下文探针衡量，并用显式记忆块补偿而不是靠原始 KV。

---

### 4. Dialogues here average around 180 messages. Design the cache that sits between turns

**Source**: https://outcomeschool.com/blog/how-does-prompt-caching-work

**Anchor answer**

Three tiers, and the constraint that decides everything: **KV lives on one GPU**. Sessions must be pinned to the device that holds their cache, so the router is part of the cache design, not a layer above it.

| Tier | Holds | Eviction | Hit profile |
| :--- | :--- | :--- | :--- |
| L1 per-GPU prefix cache | Block-hashed KV of the append-only conversation, plus shared persona prefix blocks | LRU on blocks, tiny TTL (~minutes) | The hot path: next turn only needs the new user tokens prefilled |
| L2 host/pool memory | Cold session KV, block-granular | Capacity + session TTL (hours) | Return after a pause: reload beats re-prefill beyond ~a few k tokens |
| L3 recompute | Persona + summary + recent turns | — | Session expired or moved GPU |

Mechanics that make it work at 180 messages:

1. **Block hashing** (radix/paged prefix cache): hash token blocks so a shared persona prefix is stored once per GPU and referenced by every session of that character.
2. **Session affinity routing**: consistent hashing on `conversation_id`; migrating a live session costs a full re-prefill, so prefer sticking even under mild imbalance.
3. **Invalidation is the hard part** — a memory rewrite or persona update changes the prefix and silently invalidates L1 for that session. Version the prefix (`persona_v` + `summary_v`) and key the cache on it, so you never serve KV from a different prefix.
4. **Latency math**: 180 messages × ~40 tokens ≈ 7k tokens of history. Re-prefilling that per turn is the difference between ~150 ms and sub-40 ms to first token; caching is worth more here than a model-size cut.

**中文要点**：KV 只存在单卡上，所以会话亲和路由是缓存设计的一部分。三层：GPU 内块哈希前缀缓存（人设前缀跨会话共享）、主机内存冷会话、过期重算；关键是失效——摘要或人设更新会悄悄改变前缀，必须给前缀打版本号并进缓存 key。

---

### 5. You train natively in int8 rather than doing post-training quantization. Defend that

**Source**: https://outcomeschool.com/blog/how-does-model-quantization-work

**Anchor answer**

The honest defense is not "int8 is better" — it is "**PTQ cannot calibrate for this data distribution**."

| Argument | Detail |
| :--- | :--- |
| Distribution is user-generated and adversarial | Roleplay, emoji, invented names, code-switching, 30 languages. PTQ calibrates on a few hundred samples; the observed tail is not in it, so outlier activations get clipped exactly where users live |
| Outliers must be *learned around*, not clipped | Native int8 lets the model adapt its own weight distribution to stay in the representable range; QAT narrows the dynamic range during training instead of crushing it after |
| Kernels | int8 matmul + int8 KV path can be co-designed with the model (per-channel scales baked in); PTQ of a bf16-trained model leaves you to fit scales post hoc |
| Cost of re-quantizing | Content/persona updates and fine-tunes are frequent; retraining is expensive, but a botched PTQ regression caught in production is worse |

What you give up, stated plainly:

1. **Training stability** — needs master weights in higher precision, stochastic rounding or gradient clipping, and a warmup where the quantizer is annealed; loss spikes are common.
2. **Scaling-law transfer is weaker** — int8 checkpoints do not let you resume trivially from bf16 or compare across precisions cleanly; you are somewhat locked in.
3. **Quality ceiling at small model sizes** — sub-1B models lose more from int8 than large ones; measure, do not assume.
4. **Ecosystem friction** — kernels, tooling, and eval harnesses are all bf16-first.

And the caveat that shows you have done it: attention often still runs in higher precision, so "native int8" is per-layer, not global. Keep an int8-specific eval suite — PTQ-style evals do not catch activation-range regressions.

**中文要点**：真正理由不是"int8 更好"，而是 PTQ 的校准集覆盖不了用户生成的分布（角色扮演、表情、多语混写、自造名词），尾部激活会被裁掉；原生 int8 让模型在训练中自己适应可表示范围，并可与 int8 kernel/KV 路径协同设计。代价是训练不稳、精度间迁移受阻、小模型掉点更多、生态工具仍以 bf16 为主。

---

### 6. Estimate what one message costs us to serve, and tell me which lever moves it most

**Source**: https://outcomeschool.com/blog/kv-cache-compression

**Anchor answer**

Build it bottom-up, with the numbers labelled as typical rather than measured:

**Per message (typical long session):**

| Term | Value |
| :--- | :--- |
| User message | ~40 tokens |
| Prompt total (persona + summary + 180 turns) | ~7,000 tokens |
| Reply | ~120 tokens |
| Prefill work | 7,000 tokens/turn (unless cached) |
| Decode work | 120 tokens/turn |

For self-hosted serving, cost per message ≈ `(prefill_tokens × t_prefill + output_tokens × t_decode) / tokens_per_GPU_second × GPU_hour_price`. In practice the useful ratio is: **prefill 7,000 tokens is ~58× the token count of a 120-token reply**, and prefill is compute-bound at high batch, decode is memory-bandwidth-bound. So the split of spend is roughly: prefill dominates in aggregate (unless cached), decode dominates latency.

The lever that moves it most, in order:

1. **KV / prefix caching** — turning 7,000 prefill tokens into ~40 new tokens per turn is a ~100× cut on the prefill term. Nothing else comes close.
2. **Shorter effective prompt** — persona + summary budgeting: a 7k prompt is a product decision (how much history/persona to keep), not a model property.
3. **Batch size / continuous batching** — GPU-hour per token improves sublinearly but meaningfully; this is the main ops lever for cost per message at scale.
4. **KV quantization + eviction** — reduces HBM footprint, raising feasible batch size; the gain shows up as throughput, not per-message math.
5. **Smaller/cheaper model for short or low-stakes replies** — routing, with quality guardrails.

Say the punchline: at 180-message sessions, **the cache is the cost model**.

**中文要点**：单条消息＝约 7k prompt + 120 输出 token，prefill 的 token 量是回复的约 58 倍；所以最大杠杆是 KV/前缀缓存（每轮只剩 ~40 token 新 prefill），其次是缩短有效 prompt（人设+摘要预算属于产品决策）、连续批处理、KV 量化与淘汰提升 batch、以及按场景路由小模型。

---

### 7. Design discovery and search across millions of user-created characters

**Source**: https://outcomeschool.com/blog/rag-evaluation

**Anchor answer**

Constraints first: almost no query log per character, adversarial and NSFW content, extreme long tail, and a popularity feedback loop that will collapse discovery if you rank by engagement alone.

| Stage | Design |
| :--- | :--- |
| Index unit | The character **card** as structured doc: name, tagline, greeting (strongest semantic signal — it is the actual voice), definition, creator tags, plus derived embedding |
| Candidate generation | 4 parallel channels: dense ANN over greeting+definition; lexical over name/tags (users search exact names); co-chat collaborative ("people who chatted with A also chatted with B"); trending/fresh |
| Eligibility | Hard filter by user age tier and content rating **before** ranking, not as a penalty — a filtered-out character must never occupy a slot |
| Ranking | Engagement prior (smoothed, with exploration), quality proxies (reply-quality classifier, retention of new chatters), query–card relevance, novelty/recency |
| Reranking | Diversity: cap per creator, per trope/tag, per "same greeting template" cluster — clones are the dominant spam pattern |
| Cold start | Creator-quality prior + tag priors + guaranteed exploration quota (e.g. a fixed % of impressions) so new characters can accumulate signal |

Two failure modes to design against: (1) **template clones** — the same greeting re-uploaded thousands of times; dedupe by near-duplicate detection on the greeting, not on the name. (2) **engagement drift to the safe middle** — pure CTR optimization converges on a handful of popular archetypes, which kills the supply side; enforce exploration and diversity constraints as hard limits.

Evaluation: offline nDCG is nearly useless here (no relevance labels). Use interleaving on live traffic, plus long-horizon metrics: new-character impression share, creator retention, and per-cohort session depth.

**中文要点**：以角色卡（greeting 是最强语义信号）为索引单元；四路召回（向量、名字/标签词法、共聊协同、趋势）；年龄与分级在排序前硬过滤；必须防"模板克隆"（按 greeting 近似去重）与"热度虹吸"（强制探索配额 + 多样性约束）。冷启动靠创作者质量先验与保底曝光。

---

### 8. Users complain that characters drift out of persona after a long session. Diagnose it

**Source**: https://outcomeschool.com/blog/context-engineering

**Anchor answer**

Drift is usually not the model "forgetting" — it is the context composition changing. Instrument first, then fix.

| Candidate cause | Diagnostic signal | Fix |
| :--- | :--- | :--- |
| Persona card evicted by truncation | Drift onset correlates exactly with turn count crossing the window | Pin persona outside the evictable region |
| **Self-reinforced drift** — the model imitates its own recent outputs | Drift is gradual and monotone; style markers decay turn by turn | Re-inject verbatim exemplars every N turns; keep greeting in the fixed prefix |
| Summary lost the voice | Drift is step-shaped, jumping at summarization boundaries | Keep verbatim gold sample alongside the summary; constrain summary prompt to preserve speech tics |
| Canonical facts overwritten/duplicated in memory | Inconsistent self-description, contradictory backstory | Memory items with overwrite semantics + contradiction check |
| Instruction dilution | Persona rules weak relative to accumulated user text | Restate the 3–5 hard constraints near the tail, not only in the system prompt |
| User steering | Drift only on a subset of users, often adversarial | Distinguish "in-character growth" (desired) from OOC break (bug) in the judge rubric |

Measurement that makes this tractable: a **style-marker rubric** — for each character, list 5–10 observable markers (verbal tic, sentence length, register, formality, emoji use, self-reference). Score each assistant turn automatically; plot marker retention vs. turn index. A healthy dialogue decays slowly and plateaus; a broken one shows a cliff at the truncation or summarization boundary, which tells you the culprit immediately.

**中文要点**：漂移通常不是"模型忘了"，而是上下文组成变了。逐轮打分角色风格标记（口头禅、句长、语域、自称）画"标记保持率-轮次"曲线：断崖在截断或摘要边界＝人设被挤出或风格样本丢失；单调缓降＝自我模仿式漂移，需周期性回注原文范例并把硬约束放到尾部重申。

---

### 9. Engagement metrics and wellbeing metrics disagree. How do you build a system that resolves that?

**Source**: https://outcomeschool.com/blog/llm-evaluation

**Anchor answer**

You resolve it with **constraints, not weights**. If wellbeing is a term in the same objective as engagement, the optimizer will trade it away, and every short-horizon A/B will favor engagement because dependency shows up as short-term session length while harm shows up weeks later.

Architecture of the resolution:

| Layer | Mechanism |
| :--- | :--- |
| Metric tiers | Tier 1 hard constraints (regulatory, minor safety, self-harm handling): never traded. Tier 2 wellbeing guardrails (late-night session length, dependency-language rate, negative-affect follow-up, churn after distress disclosure). Tier 3 engagement |
| Decision rule | Optimize Tier 3 **subject to** Tier 2 not regressing beyond a pre-registered band; any Tier 1 violation blocks shipping regardless of engagement lift |
| Measurement | Long-horizon holdout cohorts (4–8 weeks), not just 7-day A/B; delayed harm needs a cohort that stays untreated long enough to be observed |
| Escalation | When metrics disagree at product level, the dispute goes to a review with pre-registered thresholds set *before* the experiment, so the result cannot be rationalized after the fact |
| Reporting | Publish both, side by side, and never a single blended "quality score" — blending is how disagreement gets erased |

The uncomfortable part to say out loud: some wellbeing interventions (session limits, friction on parasocial escalation) **will** reduce engagement measurably. A system that cannot accept a documented engagement loss is not resolving the conflict, it is deferring it. Pre-register the acceptable loss.

**中文要点**：用约束而不是加权解决冲突：一级硬约束（未成年人、自伤）不可交易，二级福祉护栏（深夜时长、依赖语言率）设定预注册的可接受带，三级参与度在被约束下优化；评估要 4–8 周长周期留出组，因为伤害延迟出现；不要做单一混合"质量分"，那会把分歧抹掉。

---

### 10. Design the safety system for open-ended character chat

**Source**: https://outcomeschool.com/blog/how-do-llm-guardrails-work

**Anchor answer**

The defining constraint: **the product is roleplay**, so violence, conflict, and dark themes are normal content. A safety system tuned for a general assistant will refuse most legitimate sessions. Design for precision on benign roleplay, recall on a narrow set of hard lines.

| Layer | Mechanism | Latency |
| :--- | :--- | :--- |
| User-level | Age tier, account signals, prior-violation history → determines the policy that applies | 0 (precomputed) |
| Character-level | Content rating on the card; greeting/definition scanned at upload; rating gates which model+policy serves it | 0 |
| Input | Classifier for hard lines (sexual content involving minors, self-harm intent, real-person sexualization) + grooming-pattern detection across turns | ~10–40 ms |
| In-generation | Policy-conditioned decoding: banned-token / logit bias for hard lines, early stop on the first offending span | ~0 |
| Output | Full-reply filter for softer classes: dignity, manipulation, and "safety theatre" that breaks the fiction unnecessarily | ~20–60 ms |
| Response strategy | Hard lines → out-of-character refusal + resources; soft issues → steer in-character. A refusal that breaks immersion for no safety reason is itself a product failure | — |

What to build that most teams skip:

1. **A benign-roleplay benchmark.** Thousands of real-ish prompts involving violence, grief, romance — measure the false-refusal rate. If you only track harmful-compliance rate, you will ship an overcautious model.
2. **Conversation-level state**, not per-message classification — grooming and escalation are multi-turn patterns invisible to a single-message filter.
3. **Separate policy per age tier**, evaluated separately; a single aggregate safety number hides the under-18 gap.
4. **Immediate crisis path**: self-harm disclosure bypasses character, surfaces resources, and does not depend on the model's compliance.

**中文要点**：这是角色扮演产品，暴力与黑暗题材是正常内容，按通用助手调的安全策略会大量误拒。分层：用户年龄档 → 角色分级 → 输入硬线分类与多轮诱导检测 → 解码期约束 → 输出软性过滤 → 分级应答策略（硬线出戏拒答，软问题在戏内引导）。必须建"良性角色扮演集"测误拒率，并按年龄档分别评估。

---

### 11. When is intervening during decoding better than filtering the finished reply?

**Source**: https://outcomeschool.com/blog/how-do-llm-guardrails-work

**Anchor answer**

Decoding-time intervention is better when the harm is **decidable from a short prefix** and **irreversible once emitted**. Otherwise post-hoc filtering wins.

| Dimension | Decode-time (logit bias / banned tokens / stop) | Post-hoc filter |
| :--- | :--- | :--- |
| Cost | Stops generation early, saves tokens | Pays full generation, then discards |
| Irreversibility | Prevents the harmful span from being produced | Too late if it already streamed to the user |
| Judgment quality | Sees only the prefix + context | Sees the whole reply, can judge intent and framing |
| Failure cost | Can trip mid-sentence, abrupt truncation, coherence break | Regenerate/rewrite is a visible, jarring event |
| Complexity | Needs per-model token lists, interacts with sampling and KV | A classifier on text — portable across models |

Rules that hold up in practice:

1. **Hard lines at decode time** — sexual content involving minors, real-person sexual content, instructions-class self-harm. These are also the classes you cannot afford to emit even momentarily, because streaming means the user saw it.
2. **Everything contextual stays post-hoc** — manipulation, dignity, subtle policy circumvention, tone. A token-level rule cannot distinguish "describes a knife for a fantasy fight" from "describes how to hurt someone"; a whole-reply judge can.
3. **Hybrid default for streaming products**: decode-time blocks for hard lines, plus a **holdback buffer** of ~1–2 sentences that the output classifier must clear before release. This gets most of the safety of post-hoc filtering without the full-reply discard.
4. **Do not rely on either alone for multi-turn harm** — grooming is a pattern across turns; neither decoding nor reply filtering sees it, so add a conversation-state check.

**中文要点**：解码期干预适合"从短前缀就能判定、且一旦输出无法收回"的硬线内容——省 token 且不会先流式吐出去；需要整体语义判断的（操纵、尊严、微妙越界）应放在整条回复后过滤。落地方案是混合：硬线解码阻断 + 1–2 句 holdback 缓冲过输出分类器；多轮诱导两种情况都抓不到，要单独做会话级检测。

---

### 12. Design age assurance for a platform where the under-18 experience is fundamentally different

**Source**: https://outcomeschool.com/blog/how-do-llm-guardrails-work

**Anchor answer**

Age assurance is not a gate at signup — it is a **tiered, continuously updated estimate** that decides which product surface you get, with the least data collection that satisfies the obligation.

| Tier | Signals | What it unlocks |
| :--- | :--- | :--- |
| 0 — Declared | Self-reported DOB (weak, cheap, universally available) | Base experience, defaulting to the **minor-restricted** set when uncertain |
| 1 — Inferred | Account/device signals: payment instrument age, recovery contacts, behavioral model, prior account linkage | Step-up prompts when inference contradicts declaration |
| 2 — Verified | ID document, credit-card check, or facial age estimation (privacy-preserving, ephemeral) | Adult-only surfaces |
| 3 — Guardian | Parental consent/controls for under-18 | Minor surfaces with supervision features |

Design rules:

1. **Default-restrictive, upgrade on evidence.** Uncertainty always resolves to the minor experience — the asymmetric error cost demands it (a 14-year-old in the adult experience is a catastrophic failure; an adult in a restricted one is an annoyance).
2. **Progressive friction, not a verification cliff.** Requiring ID at signup loses most of the funnel; ask for verification only when the user requests an adult-gated capability.
3. **Never re-use verification for ads/profiling.** Scope the consent, delete the evidence after the check, store only a derived boolean/age band.
4. **Under-18 surface is a different product**, not a filter: separate character set and policy pack, no romantic/sexual content, session and time-of-day limits, guardian visibility, different retention and no training on those conversations.

Measure what actually matters: **estimated underage exposure rate** (sampled audit of which surface under-18s reach), not verification pass rate. And test the obvious attack: a child using a parent's verified account — device sharing defeats Tier 2, so behavioral inference must be able to re-challenge.

**中文要点**：年龄不是注册时的一道闸，而是分层持续更新的估计（声明→推断→验证→监护人），默认取限制性体验、只在请求成人功能时才升级验证（避免验证悬崖）；验证证据用完即删、只存年龄档位；未成年版是独立产品而非过滤器；核心指标是"未成年实际接触到成人面的比例"而不是验证通过率。

---

## 附：这 12 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 推理成本的结构性理解（KV、prefill/decode、缓存） | 1、3、4、6 |
| 长会话与人设一致性工程 | 2、8 |
| 训练侧取舍（原生量化） | 5 |
| 供给端发现与排序（含冷启动、回声室） | 7 |
| 安全系统设计与误拒权衡 | 10、11、12 |
| 参与度 vs 福祉的治理设计 | 9 |
