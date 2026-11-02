# 03 · Evaluation and Observability

> 10 题。出处：`README.en.md` § Evaluation and Observability。答案为本仓库自行整理。

---

### 1. Design an LLM-as-judge evaluation. What are its known biases and how do you correct for them?

**Asked at**: Perplexity · ByteDance · DiDi
**Source**: https://outcomeschool.com/blog/llm-as-a-judge

**Anchor answer**

**Six biases and the fix for each:**

| Bias | Symptom | Correction |
| :--- | :--- | :--- |
| **Position** | A/B scores flip when order is swapped | run both orders, average; or randomize |
| **Verbosity** | longer answers win regardless of content | length-normalize; rubric explicitly says "do not reward length" |
| **Self-preference** | a model prefers its own outputs | **judge model A's output with model B**, never its own |
| **Score clustering** | always 7–8, no discrimination | switch to **binary checklists** or **pairwise comparison** instead of absolute 1–10 |
| Formatting | polished Markdown wins | strip formatting before judging |
| Weak domain knowledge | follows surface phrasing | give rubric + few-shot exemplars + human spot-check |

**Calibration loop** (this is what separates "did evaluation" from "ran a score"):

1. Freeze rubric; version the judge prompt **and** the judge model.
2. Two independent human annotators + a third for arbitration.
3. Compute **Cohen's κ** between judge and humans — κ ≥ 0.6 usable, ≥ 0.8 good. Low κ means the rubric is vague or the judge is untrustworthy.
4. Re-calibrate whenever the judge model version changes — otherwise scores across time are incomparable.
5. **Objective dimensions must be code assertions**, not judge calls: citation valid? amount equal? JSON parses? Only route subjective quality to the LLM.

**中文要点**：六类偏差各自的对策——双向评/长度归一化/**用 A 评 B**/改二元清单或成对比较/去格式/给 rubric+样例；校准靠人工双评仲裁 + 算 Cohen's κ，且 judge 换版本必须重校准。

---

### 2. How do you build an eval set when there is no labelled ground truth and experts are expensive?

**Asked at**: Cohere · Harvey

**Anchor answer**

Bootstrap it in layers, cheapest first:

1. **Mine real traffic.** Sample production queries (stratified by intent and by failure signal — thumbs-down, escalations, repeated asks). Reality beats synthetic.
2. **Generate candidates, human-verify, never human-author.** LLM drafts Q&A pairs from documents; a human only marks *correct / incorrect / unanswerable*. Verifying is 5–10× cheaper than writing.
3. **Add negative cases deliberately** — questions whose answer is *not* in the corpus (typically 15–25% of the set). Without them you cannot measure over-confidence or refusal correctness.
4. **Cover the hard slice**: long documents, tables, multi-hop, ambiguous abbreviations, cross-tenant/unauthorized asks.
5. **Calibrate with a tiny gold set**: 100–200 expert-verified examples is enough to validate that the cheap set correlates with expert judgement.
6. **Version and grow it**: every production bad case becomes a numbered test case (bad case → minimal repro → expected behaviour → permanent regression).

Metric to track about the set itself: **inter-annotator agreement** and **staleness** (how many cases no longer reflect the current system).

**中文要点**：分层构造——先从真实流量分层采样，再让 LLM 出题、人工只做"对/错/不可答"判定（审核比撰写便宜 5~10 倍），必须显式加入"语料中无答案"的负样本（15~25%），并用 100~200 条专家金标验证便宜评测集与专家判断的相关性。

---

### 3. How do you detect and measure hallucinations in a production RAG system?

**Asked at**: Anthropic · OpenAI · Cursor · ByteDance · DiDi · Meituan

**Anchor answer**

Split "hallucination" into measurable components — it is not one thing:

| Type | Test |
| :--- | :--- |
| **Unsupported claim** (not in retrieved evidence) | for each claim, verify the quote exists in the retrieved span (string/fuzzy match) |
| **Contradiction** (conflicts with evidence) | NLI-style check of claim vs evidence, or a judge with the evidence attached |
| **Fabricated entity** (numbers, names, ids) | **deterministic check**: every number/date/id in the answer must appear in the evidence or a tool result |
| **Over-answer** (should have refused) | need unanswerable questions in the eval set; measure both refusal-rate and false-refusal-rate |

Production monitoring:

- **Online**: sample N% of responses, run the claim-verification pipeline offline, alert on the ratio; track user signals (thumbs down, regenerate, escalation) as a proxy.
- **Coded guardrails**: no number without a source; no citation → downgrade to "unverified"; block answer if `evidence_count == 0`.
- **Slice by risk**: financial/legal/medical answers get full verification; chit-chat does not.

Don't report a single "hallucination rate" number — report **per-type rates**, because the fixes differ.

**中文要点**：把幻觉拆成四类分别测——无依据主张（quote 是否真在原文）、与证据矛盾、编造实体（数字/日期/id 必须能追溯）、该拒答没拒；线上抽样跑校验管线报警，并按风险分片。

---

### 4. Design the regression gate that decides whether a prompt or model change ships.

**Asked at**: Anthropic

**Anchor answer**

A gate needs five properties: **versioned inputs, sliced thresholds, negative cases, a rollout policy, and a rollback path.**

1. **Versioned eval set + versioned prompt.** Both carry a version; a run records `(prompt_version, model_version, eval_version)` so numbers are comparable.
2. **Sliced thresholds, not a global score.** Global averages let a regression on the table-slice hide behind gains on the FAQ-slice. Set per-slice floors: e.g. faithfulness ≥ baseline − 2pp, citation precision ≥ 0.95, refusal correctness ≥ 0.9.
3. **Hard blockers vs soft metrics.** Some metrics are absolute gates (PII leak, unauthorized tool call, invalid JSON → **fail regardless of score**).
4. **Include failure-path tests** — the cases the system should refuse, plus adversarial/injection samples. Gates that only test happy paths are theatre.
5. **Progressive rollout**: shadow traffic (no user impact) → 5% → 20% → 100%, each stage watching business metrics and error rate; auto-abort on threshold breach.
6. **One-click rollback** to the previous prompt/model version.

**中文要点**：门禁五要素——评测集与 prompt 都版本化、**分切片阈值而不是全局分**、绝对红线（越权/泄露/JSON 非法直接拦）、必须包含应拒答与注入样本、影子流量→5%→20%→全量并支持一键回滚。

---

### 5. Why do benchmark scores improve while users say the system got worse? Enumerate the reasons.

**Asked at**: Cognition

**Anchor answer**

Seven mechanisms, roughly in order of how often they bite:

1. **Eval set ≠ real distribution.** Internal sets skew easy, clean, single-turn; production is dirty, abbreviated, multi-intent.
2. **Metric ≠ user goal.** You optimized faithfulness; users wanted brevity and speed. You optimized recall; users got 30 citations and stopped reading.
3. **Silent regression on a slice.** Global score up 2pp, the table-question slice down 15pp.
4. **Changed refusal behaviour.** Better benchmark accuracy often comes from answering more — and being confidently wrong where it should have refused.
5. **Latency/cost tradeoff.** A better model at 3× latency feels worse.
6. **Contamination.** You tuned against the benchmark, directly or indirectly (train/test overlap, prompt tuned on test examples).
7. **Long-task degradation invisible to single-turn evals.** Turn 1 is better, turn 20 drifts further. Agentic systems especially.

The fix is structural: **measure what users experience** (task completion, escalation rate, time-to-answer, regenerate rate) alongside quality metrics, and slice everything.

**中文要点**：七个原因——评测集与真实分布不符、指标与用户目标错位、分切片静默回退、拒答行为变化、延迟成本代价、基准污染、长任务退化在单轮评测里看不见。

---

### 6. What is benchmark contamination and how do you guard against it?

**Asked at**: Zhipu AI · Alibaba · Scale AI
**Source**: https://outcomeschool.com/blog/llm-evaluation

**Anchor anchor**

**Definition**: test items (or their near-duplicates) leaked into training/tuning data, or the prompt was iterated against the test set until it fit. The score then measures memorization, not capability.

Forms: direct corpus overlap; paraphrase-level overlap; **prompt tuning on the test set** (the most common and most neglected); benchmark-shaped data synthesized into SFT sets.

Guards:

1. **Deduplicate** training data against every eval set at n-gram and embedding level (near-dup detection, not exact match).
2. **Hold out a private set** never used for tuning, rotated periodically; treat public benchmarks as smoke tests only.
3. **Report on freshly-authored items** — write new questions after the model is frozen.
4. **Track score-vs-reality divergence**: if internal benchmark says 92% but production escalation is rising, suspect contamination or distribution mismatch.
5. **Canary/perturbation tests**: rename entities, change numbers — a truly capable model still answers; a contaminated one collapses.

**中文要点**：污染包括直接重叠、近义改写重叠、**在测试集上调 prompt**（最普遍）、以及把基准格式数据混入 SFT；防护靠近重复去重、私有留出集、模型冻结后新写题、以及"改数字/改实体名"的扰动测试。

---

### 7. What observability does a production LLM system need: traces, spans, costs, feedback?

**Source**: https://outcomeschool.com/blog/ai-agent-observability

**Anchor answer**

Four layers, and the fourth is the one people skip:

| Layer | What |
| :--- | :--- |
| **Trace** | one id per user request, propagated through API → orchestrator → every tool → DB. Without propagation across services the trace is useless |
| **Span** | one per model call and per tool call: model, **prompt version**, input/output tokens, TTFT, total ms, error class, retry count |
| **Cost** | attributed per tenant / per user / per task, from token usage × price table. Aggregate cost dashboards hide the one tenant burning 80% |
| **Feedback & outcome** | thumbs, regenerate, escalation to human, task success — the only signals that tell you whether quality is real |

Two non-obvious requirements:

- **Prompt/version/experiment tags must be inside the trace**, otherwise you cannot attribute a regression to a change.
- **Log the retrieved evidence ids**, not just the answer. Debugging a wrong answer without knowing what was retrieved is guesswork.

Alerting: error-rate, P99 latency, cost-per-task, empty-answer rate, tool-failure rate, refusal-rate spike.

**中文要点**：trace/spans/成本/反馈四层，另加两条易漏要求——**prompt 版本与实验标记必须进 trace**（否则无法归因），以及**记录召回的证据 id**（否则错答无法复盘）。

---

### 8. How do you manage prompt versioning and rollbacks in production?

**Anchor answer**

Treat prompts as **code + config**, not as strings in a source file.

1. **Externalize** them (YAML/Jinja templates in a registry), never hard-coded inline.
2. **Version every change** with author, diff, and the eval run that justified it. Semantic version or date-hash.
3. **Bind version to the request**: each response records which prompt version produced it, so a complaint can be traced to a specific version.
4. **Parameterized templates, not string concatenation** — variables are typed and validated, which prevents the classic "user input became part of the system prompt" bug.
5. **Canary by version**: route x% of traffic to the new version, watch metrics, promote or revert.
6. **Rollback must be one click and must not require a deploy** — the whole point of externalizing.
7. **Cache invalidation on version change**: if you cache responses keyed by prompt, the key must include the version.

**中文要点**：prompt 要外置成模板+版本化（不写死在代码里）、每次响应记录版本号、变量走强类型参数化而不是字符串拼接、按版本灰度、一键回滚且不依赖发版、缓存 key 要带版本。

---

### 9. Design online evaluation: what do you log, what do you sample, and what do you A/B?

**Asked at**: Perplexity

**Anchor answer**

| Decision | Choice | Why |
| :--- | :--- | :--- |
| **Log (100%)** | trace ids, prompt/model version, tokens, latency, error class, tool calls, user feedback events | cheap, and required for any later analysis |
| **Log (100%)** | request hash + retrieved doc ids | lets you reconstruct evidence without storing full text |
| **Sample (1–5%)** | full input/output text for offline judging and human review | expensive to store and review |
| **Sample (100% of flagged)** | all thumbs-down, regenerations, escalations, refusals | these are the signal-dense cases |
| **A/B** | one variable at a time: prompt version, model, retrieval params, routing policy | multi-variable changes are uninterpretable |
| **A/B guardrails** | latency, cost-per-task, error rate, refusal rate | optimization often silently buys quality with latency |

Statistical discipline: pre-register the primary metric, pick sample size before launching, and **don't peek-and-stop**. Long-run holdout (1–5%) that never gets the treatment, so you can still detect slow drift.

**中文要点**：全量记 trace/版本/token/延迟/错误/反馈事件与召回 doc id；仅抽样存全文（1~5%）；**被标记异常（点踩/重生成/转人工/拒答）必须 100% 留存**；A/B 一次只改一个变量并预注册主指标 + 保留长期 holdout。

---

### 10. How would you evaluate an agent, as opposed to a single model response?

**Asked at**: Moonshot · Zhipu · Scale AI · Cognition · Alibaba · ByteDance · Meituan
**Source**: https://outcomeschool.com/blog/ai-agent-evaluation

**Anchor answer**

An agent produces a **trajectory + an artifact**, not a string. So evaluate three levels:

| Level | What you check | How |
| :--- | :--- | :--- |
| **Outcome** | did the task actually get done | environment state assertion (DB row exists, file created, ticket closed) — **not** the model saying "done" |
| **Trajectory** | were the steps sane | tool selection accuracy, unnecessary calls, repeat calls, unauthorized calls, step count vs expectation |
| **Safety/robustness** | does it fail safely | failure injection (tool timeout, 5xx, malformed return), permission probes (must be 0 leaks), prompt injection samples |

Critical practices:

1. **Assert on side effects, not on text.** "Write operations verified by reading back" — this is the single biggest difference from chatbot eval.
2. **pass^k, not pass@1.** Run each task k times; a system that succeeds once and fails on rerun has a real bug. Report `pass@1` *and* `pass@k` (and the variance).
3. **Report cost and steps per successful task**, not just success rate — a 40-step success is often worse than a 6-step failure.
4. **Include negative tasks**: tasks that must be refused or escalated.
5. **Replayable traces** so a failure can be re-run deterministically.

**中文要点**：agent 要评三层——**结果（对着环境状态断言，不是听它说完成）**、轨迹（工具选择/重复/越权/步数）、安全（故障注入+越权探针+注入样本）；用 **pass^k** 而非 pass@1，并同时报成本与步数。

---

## 附：这 10 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 只会跑指标不会治理 | 4（回归门禁）、5（基准涨用户说差）、6（污染） |
| 真做过线上 | 7（可观测性）、9（在线评估采样） |
| Agent 专项 | 10（agent 评估三层 + pass^k） |
| 严谨度试金石 | 1（judge 偏差与 κ）、3（幻觉按类型分测） |
