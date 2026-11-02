# 25 · Apple

> 11 题。出处：`README.en.md` § Apple。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. You need to run a ~3B-parameter language model on a phone with tight memory and power budgets. What changes versus serving the same model in a datacenter?

**Source**: https://outcomeschool.com/blog/cloud-vs-on-device-model-deployment · https://outcomeschool.com/blog/how-does-model-quantization-work

**Anchor answer**

The regime changes completely — this is not "the same serving problem, smaller".

| Dimension | Datacenter | Phone |
| :--- | :--- | :--- |
| Batch size | 32–256 (throughput is the metric) | **1** — nothing to amortize |
| Binding constraint | GPU compute / HBM capacity | **memory bandwidth + peak power + thermal envelope** |
| Weights | bf16/fp8, 100s of GB VRAM available | int4/int8, must fit in a few GB of **unified memory shared with the OS** |
| Latency metric | TTFT p95, tokens/s aggregate | TTFT and steady-state tokens/s *for one user*, with an app-launch-class budget |
| Failure mode | OOM in a job | OS **jetsam kill** of your app; ANE/GPU memory limits are separate and smaller |
| Dispatch | one backend, mature kernels | CPU / GPU / ANE with different op support; you must shape the graph to what the NPU supports or fall back |
| Concurrency | many users | foreground app + background daemon competing; the model may be evicted mid-session |

**What actually changes in the design:**

1. **Quantize aggressively** — int4 weight-only (group-wise, per-channel scales) is the default; 3B at 4 bits ≈ 1.5–2 GB, which is the difference between fitting and being killed. Keep activations and the KV cache in higher precision if quality drops.
2. **Vocabulary and embedding size matter at batch 1**: the LM head is a `[vocab × d]` GEMM per token; a 128k vocab at 3B is a measurable fraction of decode time on a phone. Trim or use a factorized head.
3. **Context, not parameters, is the memory risk.** KV cache grows with conversation length; cap it, evict, or use sliding-window attention.
4. **Prefill is compute-bound and painful on a phone** — 1–2k tokens of prompt can cost more than 200 ms of ANE time; keep prompts short and cache the system prefix.
5. **Thermals**: sustained decode throttles. Design for bursty, cancellable generation, and never benchmark the first 10 seconds and call it the number.

**中文要点**：手机是 batch=1、受带宽/功耗/热限制，还要防系统 jetsam 杀进程；权重必须 int4（3B≈1.5–2GB）、词表大小在 decode 上很贵、真正的显存风险是 KV cache 随对话增长，且持续生成会降频。

---

### 2. Explain post-training quantization versus quantization-aware training. What breaks when you push weights to 2–4 bits, and how do you recover quality?

**Source**: https://outcomeschool.com/blog/how-does-model-quantization-work

**Anchor answer**

| | PTQ | QAT |
| :--- | :--- | :--- |
| When | after training, no gradients | during/after fine-tuning, **fake-quant** nodes in the forward pass, straight-through estimator for the backward |
| Cost | minutes to hours, calib set of a few hundred samples | full fine-tune cost, but often only a short recovery phase |
| Ceiling | good to ~4 bits weight-only; degrades fast below | reaches usable 2–3 bit, and int4 with small accuracy loss |
| Needs | calibration data that matches deployment distribution | training data + a teacher (distillation helps a lot) |

**What breaks at 2–4 bits:**

1. **Outlier channels.** A few activation/weight channels have magnitudes 10–100× the rest; a single global scale either clips them or destroys resolution for everyone. Fix: group-wise/per-channel scales, **AWQ**-style activation-aware scaling, SmoothQuant-style migration of activation outliers into weights.
2. **Low-bit rounding error compounds across layers**, so perplexity rises superlinearly — 8 bit is nearly free, 4 bit is fine with care, 3 bit needs QAT + distillation, 2 bit is a research problem for general models.
3. **The KV cache and attention are more sensitive than MLP weights**; quantizing them naively wrecks long-context behaviour even when short-prompt evals look fine.
4. **Kernel mismatch**: a format your NPU doesn't accelerate natively (e.g. 3-bit with non-standard group size) runs dequant-then-GEMM on CPU and can be *slower* than 8-bit.

**Recovery levers**: keep embeddings + LM head + first/last blocks in higher precision (mixed precision), GPTQ/AWQ for PTQ, QAT with distillation from the fp16 teacher, and evaluate on **task metrics, not perplexity** — a 0.3 perplexity hit can be a 5-point drop on structured extraction.

**中文要点**：PTQ 靠校准、便宜，8→4 bit 基本无损；QAT 用假量化 + STE 微调才能到 2–3 bit。崩的原因主要是离群通道（需分组/per-channel 缩放、AWQ/SmoothQuant）、跨层误差累积、KV cache 比权重更敏感，以及非标准位宽在 NPU 上没有原生 kernel。

---

### 3. Estimate the KV-cache memory for a 3B on-device model at 4k context, and name the levers that shrink it.

**Source**: https://outcomeschool.com/blog/kv-cache-in-llms · https://outcomeschool.com/blog/kv-cache-compression

**Anchor answer**

Formula first, then plug in a **typical** 3B config (state the assumption — the arithmetic is the answer, not the constant):

```
bytes = 2 (K and V) × layers × kv_heads × head_dim × seq_len × batch × bytes_per_element
```

| Assumption | Value |
| :--- | :--- |
| layers | 28 |
| kv_heads × head_dim (GQA) | 8 × 128 = 1024 |
| seq_len | 4096 |
| dtype | fp16 (2 B) |

`2 × 28 × 1024 × 4096 × 2 = 469,762,048 B ≈ 448 MiB`.

Sanity check that earns credit: if the same model used **full MHA** (32 kv heads instead of 8), it's 4× → ~1.75 GiB, which on a phone is not a cache, it's the whole app's budget. **GQA is what makes on-device 4k context possible at all.**

**Levers, by leverage:**

1. **Fewer KV heads** (MQA/GQA) — architectural, 4–8× on the dominant term; you can't retrofit easily, so it's a training-time decision.
2. **Fewer bits** — int8 KV is ~2×; int4 with per-group scales is ~3–4× but needs accuracy testing on long context, and per-token outliers hurt.
3. **Shorter effective context** — sliding-window attention, or evicting old turns with a running summary. Cheapest of all, and the one product teams forget until jetsam kills the app.
4. **Prefix sharing** across concurrent requests / features — one copy of the system prompt instead of N.
5. **Paged / non-contiguous allocation** — avoids fragmentation-driven over-reservation, which on a memory-capped device is as important as the raw size.
6. **Compression/eviction heuristics** (H2O, attention-sink retention) — keep the first few tokens + recent window; quality risk lives in "the user referenced something said 30 turns ago".

**中文要点**：`2×层数×kv_heads×head_dim×seq×batch×字节`，典型 3B/28 层/GQA 8×128/fp16/4k ≈ 448 MiB；换回 MHA 就是 1.75 GiB —— GQA 是端侧 4k 的前提。降的杠杆按效力排序：少 KV 头 > 低位宽 > 短有效上下文 > 共享前缀 > 分页分配。

---

### 4. Time-to-first-token for your on-device feature is 1.8 s. Walk me through diagnosing and fixing it.

**Source**: https://www.youtube.com/watch?v=XD8DD4cEHu0

**Anchor answer**

Don't guess — **segment the 1.8 s with signposts** first, because on-device the split is nothing like a server:

| Segment | Typical on-device share | What it looks like |
| :--- | :--- | :--- |
| App/UI + first-launch cold path | 100–600 ms | session setup, daemon spin-up |
| **Model load / weight paging** | 0–900 ms (bimodal!) | mmap page-in from flash, ANE compile, first-run only |
| Tokenize + prompt build | 20–80 ms | long system prompt, chat history re-serialization |
| **Prefill** | 300–900 ms | compute-bound, ∝ prompt length, ~10–50× the cost per token of decode on phone NPUs |
| Sampling + first detokenize | 10–40 ms | |
| Render / stream surface attach | 50–200 ms | first token exists but nothing is visible |

**Diagnosis order**

1. Is it bimodal? Compare cold app launch vs warm. If only the first request is 1.8 s, it's **weight paging/ANE compile**, not prefill — fix with mmap + `madvise`, keeping the model resident, and a warm-up pass at app start.
2. Log actual prompt token count. A 900-token system prompt on a phone NPU is often 600+ ms of prefill. Trim it, and **cache the prefix KV** across turns.
3. Check thermal state and background CPU contention — benchmarking right after install on a charging phone gives numbers you'll never reproduce.
4. Check whether the graph fell back to CPU. One unsupported op silently moves a whole subgraph off the NPU; you see 4× prefill with no error.

**Fixes, by cost/benefit**: cache the system-prompt prefix KV (largest single win, often 30–50 %), keep the model resident + warm up, shorten/clamp prompt, prefill in chunks and stream the UI skeleton immediately, and **decouple "useful content" from full completion** — render the first sentence as soon as it exists.

**中文要点**：1.8s 必须先按阶段打点（App 冷启/权重换页/分词/prefill/首 token 渲染），端侧常见是双峰——只有第一次慢就是权重换页与 ANE 编译；最大单体收益是缓存 system prompt 的前缀 KV + 常驻模型，别把 prefill 当成唯一嫌疑。

---

### 5. Your on-device model must emit valid, schema-conforming tool calls. How do you guarantee validity rather than hope for it?

**Source**: https://outcomeschool.com/blog/how-does-function-calling-work-in-llms

**Anchor answer**

"Guarantee" means **constrained decoding**, not prompt engineering. The model's next-token distribution is masked at each step against a state machine compiled from the JSON Schema:

| Layer | Mechanism |
| :--- | :--- |
| Grammar | JSON Schema → grammar (GBNF/regex/FSM). At each step, only tokens that keep the prefix parseable are allowed |
| Token masking | mask logits of illegal continuations before sampling — must be **token-aware**: a token can straddle a boundary (`": "` + value start), so grammar-to-token mapping is done offline per vocabulary |
| Structural guarantee | the output is **syntactically** valid by construction; enum values are enforced by allowing exactly those token sequences |
| Semantic validity | *not* guaranteed — `date: "2026-02-30"`, an id that doesn't exist, a range where `min > max`. This needs a validator + repair loop or a typed action layer |
| Fallback | if no legal token has non-zero probability (all masked), you must have a defined behavior: emit a fixed "cannot comply" token sequence, not a crash |

On-device specifics that change the design:

- **Compile the grammar once, at build time**, per tool set. Runtime grammar compilation on a phone costs tens of ms and allocate-on-first-use is a latency spike users feel.
- **Small models are worse at nested schemas.** Keep the schema shallow, prefer flat args with enums, and split a 40-parameter tool into several tools. A 3B model with a constrained decoder produces valid-but-wrong arguments far more often than invalid JSON.
- **Latency**: masking adds a per-step vocabulary scan; precompute the allowed-token bitmap per FSM state to keep it O(#allowed) instead of O(vocab).
- **Verify arguments against device state** (does the contact exist? is the album non-empty?) before dispatch — syntactic validity is the cheapest part of correctness.

**中文要点**：可靠性的来源是**约束解码**（schema→语法→token 级 logit 掩码），语法合法是构造保证的；但语义合法（不存在的联系人、非法日期）不在保证范围，要靠参数校验 + 与设备状态比对，3B 小模型更多是"合法但错"而不是 JSON 坏。

---

### 6. You have one on-device base model but a dozen features: summarization, rewriting, reply suggestions, tone adjustment. How do you specialise without shipping a dozen models?

**Source**: https://outcomeschool.com/blog/lora-low-rank-adaptation-of-llms

**Anchor answer**

Ship **one frozen base + N small adapters**, selected at runtime. LoRA makes this viable because an adapter is a tiny delta (`A ∈ R^{r×d}`, `B ∈ R^{d×r}`, `r` = 4–16), typically 1–3 % of base size — a dozen adapters cost less than one extra copy of the weights.

| Approach | Verdict |
| :--- | :--- |
| One full fine-tune per feature | 12× storage and 12× paging cost — dead on a phone |
| One model, prompting only | cheapest, but tone/format adherence is weak at 3B |
| **Multi-LoRA on one base** | storage ≈ base + N·δ; swap adapters per request; shared backbone |
| One adapter, task prefix tokens | works when tasks are similar; degrades as they diverge |
| MoE-style adapter routing | overkill at a dozen well-known tasks — the app *knows* which feature it's in |

**Design decisions that matter:**

1. **Merge vs swap.** Merge into the base at load time = zero runtime overhead but one model instance per feature (memory again). Keep adapters separate and apply them in the matmul = one base in memory, N tiny deltas, at a small kernel cost. On device, separate + fused kernel wins.
2. **Rank budget.** `r` is the quality knob; 8 is usually enough for style/format tasks, higher for genuinely new capabilities. Measure per feature — allocating `r=32` to "tone adjustment" is wasted bandwidth.
3. **Shared data, conflicting gradients.** Train adapters separately; if you must train jointly, use orthogonal constraints, otherwise adapters drift toward the dominant task.
4. **Router is trivial; the *base* is the risk.** All twelve features share the base's failure modes — a bad summarizer behaviour must be fixed in the base, and a base update invalidates all adapters unless you retrain them against the new weights. Version adapters with the base hash.

**中文要点**：一个冻结底座 + N 个小 LoRA（r=4~16，约底座 1–3% 体积），运行时按功能挂载而不是合并进权重（合并会变成 N 份模型占内存）；真正的风险在共享底座——底座一升级所有 adapter 都要按新权重重训，必须用 base hash 做版本绑定。

---

### 7. How would you improve an on-device model using signals from user devices without collecting user content?

**Anchor answer**

Split the loop into "compute where the data is" vs "aggregate what's safe to move".

| Signal (stays on device) | What it teaches | Where the update happens |
| :--- | :--- | :--- |
| Accept / edit / dismiss on a suggestion | preference + formatting errors | local reward signal for ranking, not generation |
| Edit distance between suggestion and sent text | concrete failure examples | local replay buffer for on-device adaptation |
| Retry / regenerate, abandonment | confidence calibration | router thresholds, abstain policy |
| Explicit thumbs (opt-in) | sparse but high-value labels | DP-aggregated counters only |

**Mechanisms, in increasing sophistication:**

1. **On-device evaluation, aggregate-only telemetry.** Run a small held-out suite locally; upload *counts and scores* (never text) with differential privacy noise and a minimum-bucket threshold so a small cohort can't be de-anonymized.
2. **Federated learning with secure aggregation.** Devices train locally on their own interaction data, upload clipped + noised gradient updates, server aggregates. Practical caveats worth saying: uplink budget, on-device compute cost, straggler devices, and DP-SGD's utility hit — federated systems often ship a *ranking/adapter* update, not a full model.
3. **Federated analytics for mining**: find which failure *patterns* exist without seeing them — e.g. cluster local failures into a codebook of categories, upload only the histogram.
4. **Ship the fix as a new adapter trained on public/synthetic data** that targets the discovered failure class. This is the honest path: use device signals to *find* the problem, use consented/synthetic data to *fix* it.

**The line you must not cross**: "we only send hashed text" or "we only send embeddings" is not privacy — embeddings are invertible enough, and hashes of short strings are enumerable. Say that out loud; it's the differentiator.

**中文要点**：数据留在设备上，只上传差分隐私加噪的计数/梯度并用安全聚合；端侧信号用来**发现问题类别**（本地聚类后只上报直方图），修模型用公开/合成数据训新 adapter；"只传 embedding 或哈希"不算隐私保护。

---

### 8. Design the routing layer that decides whether a user request is handled on-device, by a first-party server model, or by a third-party model.

**Source**: https://outcomeschool.com/blog/llm-routing · https://outcomeschool.com/blog/cloud-vs-on-device-model-deployment

**Anchor answer**

The router is a **policy function over request features**, and its most important output is often "clarify" or "refuse", not a model choice.

**Decision inputs**

| Class | Features |
| :--- | :--- |
| Content sensitivity | contains personal data that must not leave the device; on-screen context from other apps |
| Capability need | task type (paraphrase vs multi-step reasoning vs world knowledge vs long context), estimated difficulty |
| Context size | > device window (e.g. 8k effective) → server |
| Latency/network | offline, cellular, battery/thermal state → on-device |
| Tool need | requires server-side tools (search, maps, first-party apps) → first-party |
| Contract | third-party only for explicitly-scoped capabilities with a privacy story; never as a silent fallback for sensitive content |

**Architecture**

1. **Cheap deterministic pre-checks** (offline? sensitive? over context?) — these are rules, not a model. Roughly a third of traffic is decided here.
2. **A small classifier / difficulty estimator** for the residue: predicted success probability of the on-device model on this request. Calibrate it on-device telemetry; thresholds are product decisions (target on-device share vs quality).
3. **Fallback policy on failure**, which is where systems break: if the on-device answer is bad, do you escalate? That **re-uploads the prompt** — a privacy event the user didn't consent to. Escalation must be disclosed and gated, not automatic.
4. **Sticky decisions per session** so the model doesn't visibly switch voice mid-conversation.

**Failure modes**: router trained on English-only traffic misroutes other locales downward; on-device confidence is poorly calibrated (small models are confidently wrong); a "third-party" path leaks more context than the request; and cascading escalation turns a 1-request interaction into 2× cost. Log the *decision and reason code* per request — you can't debug routing without it.

**中文要点**：路由是"规则预判（离线/敏感/超上下文）→ 校准过的小分类器估设备端成功率 → 明确的上浮策略"，最危险的不是选错模型而是**失败上浮等于把 prompt 传上云**，必须显式告知与门控；每请求记决策与原因码。

---

### 9. A user says “send Maya the photos from Saturday's hike.” Design the on-device path from that utterance to a structured app action with resolved parameters.

**Source**: https://outcomeschool.com/blog/how-does-function-calling-work-in-llms

**Anchor answer**

This is an **entity-resolution** problem wearing a tool-calling costume. The model's job is to produce slots; the OS owns resolution.

| Stage | Output | Owner |
| :--- | :--- | :--- |
| 1. ASR / text | transcript + timestamps | system |
| 2. Intent + slot extraction | `send_media{recipient: "Maya", media: "photos", time: "Saturday", event: "hike"}` | on-device LLM, constrained decoding over a fixed schema |
| 3. Recipient resolution | candidate contacts via local index (`Maya`, `Maya Chen`, `Mom (Maya)`) | **deterministic** contacts API + ranking by interaction recency |
| 4. Time resolution | "Saturday" → concrete date range relative to *now*, in the user's timezone | date parser, not the LLM |
| 5. Media resolution | photos in that range → cluster by location/co-location → the hike cluster | Photos framework queries + on-device clustering |
| 6. Disambiguation | if 2 Mayas or 3 photo sets → ask, showing the concrete candidates | UI, with thumbnails |
| 7. Confirmation | "Send 12 photos from Sat 3 Oct (Bondi) to Maya Chen?" | UI |
| 8. Execute | `send_media(contact_id, asset_ids)` via the app's API, idempotent | app |

**Design rules that make it reliable:**

- **Never let the model emit ids.** It emits names and intents; the system maps to `contact_id`/`asset_id`. A hallucinated id is a misdirected message, which is unfixable.
- **Relative time needs an anchor.** "Saturday's hike" said on Sunday vs the following Wednesday means different Saturdays. Ask when ambiguous; never guess silently.
- **Confirmation is mandatory** for irreversible sends, and the confirmation must show *resolved* values (count, recipient, date), not the paraphrase.
- **Zero/one/many handling**: zero candidates → "I couldn't find photos from a hike on Saturday" + offer nearest match; many → picker.
- **Privacy boundary**: all resolution stays on device; only the final `send` touches the network.

**中文要点**：模型只产出槽位、绝不产出 id；联系人/时间/相册全部由确定性 API 解析（"Saturday" 必须有相对锚点），歧义就弹选择器，不可逆发送前必须用**已解析的具体值**确认（几张、给谁、哪天）。

---

### 10. You're shipping notification summarization to hundreds of millions of users in 30+ locales, and you cannot log user content. Design the evaluation and regression-detection story.

**Anchor answer**

Invert the usual architecture: **the eval harness ships to the device, only aggregates leave it.**

| Layer | Where | What moves |
| :--- | :--- | :--- |
| Golden corpus | bundled in the app: consented + synthetic notifications per locale/task type | nothing — it never leaves |
| Automated scoring | on device, using on-device models (entailment/quality classifiers, schema checks) | per-locale **scores** (DP-noised, bucketed) |
| Human eval | lab/contractor panel on consented data, per locale, before release | labels, not user content |
| Guardrail counters | on device: truncation rate, empty output, over-length, refusal rate, latency, crash | counters only |
| Regression detection | server compares DP aggregate distributions vs the previous release baseline | noise-aware deltas |

**Why it works**: you never need the raw text to detect "Spanish summaries got 4 points worse" — you need a *labeled* corpus you're allowed to have plus an aggregate score. Ship the corpus in the OS image, version it with the model.

**The four things that go wrong**

1. **Locale leakage.** Translating the English eval set is not localization — grammar, honorifics, name order, and notification length norms differ. Each locale needs native-authored items, and the *task* differs (Japanese business notifiers compress differently than English).
2. **Metric blindness to the real failure**: a summary that drops the sender's name or the amount ("your card was charged") reads fine and is useless. Score required-slot retention per notification class, not fluency.
3. **Privacy budget exhaustion.** Per-locale DP counters with a minimum cohort size; if a locale is small, use coarser buckets rather than exact counts. Rotate the reporting epoch.
4. **You can't debug what you can't see.** Build a **user-initiated** diagnostic path ("report a bad summary" uploads that one item with explicit consent), otherwise every incident becomes unanalyzable.

**中文要点**：把评测器下发到端侧（golden 语料随包、端侧打分），只上传差分隐私加噪的分桶分数与护栏计数；回归检测比的是分布而非文本；坑在于"翻译的评测集不等于本地化"、以及必须按通知类别评**关键槽位保留**（发件人、金额）而非流畅度。

---

### 11. Tell me about a time you had to make progress with incomplete information: you couldn't be told the full context of what you were building.

**Anchor answer**

The signal being tested is **how you bound risk when the spec is hidden** — not whether you were brave.

| Beat | Content that scores |
| :--- | :--- |
| The gap | what you were told, what you were explicitly not told, and why (need-to-know, unreleased product, partner dependency) |
| The move | you didn't stall and you didn't guess big — you **decomposed** the work so the unknown parts stayed behind a seam |
| The instrument | a stub/interface/prototype that let you validate everything *around* the unknown ("I built against a fake module with the documented contract and proved the rest end-to-end") |
| The question discipline | 2–3 precise questions asked once, with the *decision they unblock* stated — not a weekly "any update on requirements?" |
| The check | an explicit assumption list shared with the team, revisited at a named milestone |
| The outcome | what shipped, what changed when the information arrived, and how cheap that change was |

A shape that reads as senior: *I was building an on-device feature where I couldn't be told the target device class or the launch OS version. Rather than optimize for a guess, I built the pipeline behind a capability descriptor — memory budget, NPU availability, context limit — and made model size, quantization, and cache strategy configuration rather than constants. When the constraints arrived nine weeks in, the change was two config files plus a re-run of the perf suite, not a redesign. The one thing I couldn't de-risk was quality against real usage, so I insisted on a shadow-mode phase before ship.*

Two credibility markers: (1) you name the *specific* seam you created, (2) you say what you deliberately did **not** build, and why building it would have been a wasted guess. Avoid the version where you "just made reasonable assumptions" and got lucky — that's not a method.

**中文要点**：考的是"信息缺失时如何限定风险"：把未知隔离在一个接缝后面（能力描述符 + 配置化模型尺寸/量化/缓存），把假设列成清单在里程碑复核，只问 2–3 个能解锁决策的问题；并说清你**故意没做什么**，以及信息到位后改动为何很便宜。

---

## 附：这 11 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 端侧性能硬功底 | 1（手机 vs 数据中心）、3（KV cache 算账）、4（1.8s 定位） |
| 量化实战 | 2（PTQ/QAT、2–4 bit 崩在哪） |
| 结构约束能力 | 5（保证合法工具调用）、9（端到端槽位解析） |
| 单底座多任务 | 6（多 LoRA 与底座版本绑定） |
| 隐私工程 | 7（不上传内容的改进闭环）、10（无日志的十亿级评估） |
| 策略与隐私边界 | 8（on-device / 一方 / 三方路由） |
| 模糊性下的判断 | 11（信息不全推进） |
