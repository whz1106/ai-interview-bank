# 01 · Agents and Tool Use

> 12 题。出处：`README.en.md` § Agents and Tool Use。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Explain the ReAct pattern and what it solves over chain-of-thought alone.

**Asked at**: ByteDance · Tencent · Meituan
**Source**: https://outcomeschool.com/blog/react-agent

**Anchor answer**

ReAct = Reasoning + Acting. It is not a model; it is a **loop pattern** around an LLM: `Thought → Action (tool call) → Observation → Thought → ...` until the model emits a final answer.

What it solves that CoT alone cannot:

| Problem in CoT | How ReAct fixes it |
| :--- | :--- |
| Reasoning is closed-world; facts come from memory | Observation injects **real external data** mid-reasoning |
| Errors are unrecoverable — the model never sees them | A failing tool returns an error the model can act on |
| Cannot do multi-hop over live data | Each hop queries a real source |
| Hallucinated facts look identical to known ones | The trace shows **which tool produced which claim** |

Costs: more tokens, more latency, and you now own loop control (budgets, retries, termination).

Note on format: the classic paper used the literal `Action: tool[input]` string. With modern APIs (Claude/GPT/Gemini) you use the native tool schema instead — the loop shape is identical, only the serialization changes.

**中文要点**：ReAct 的本质是"用工具返回的真实观察修正下一轮推理"，解决的是 CoT 闭门造车 + 错误不可恢复；代价是 token、延迟，以及你必须自己写循环控制。

---

### 2. How do you handle tool-call errors, timeouts and retries in an agentic loop?

**Asked at**: OpenAI · Cognition
**Source**: https://outcomeschool.com/blog/ai-agent-loop

**Anchor answer**

First: **classify the error before deciding to retry.**

| Class | Examples | Action |
| :--- | :--- | :--- |
| Transient | timeout, 429, 502/503 | retry with exponential backoff + jitter, cap attempts |
| Terminal | invalid args, 401/403, 404, schema violation | **do not retry** — feed the error back so the model can fix the args or give up |

Five mechanisms, all needed:

1. **Per-tool timeout + task-level deadline.** A tool that hangs forever must not eat the step budget.
2. **Circuit breaker.** N consecutive failures → fail fast instead of burning tokens.
3. **Error as observation, not exception.** Append `{tool, error, retryable}` to history so the model sees it and can self-correct. Silently retrying hides the failure and the model repeats it.
4. **Idempotency key for side-effecting tools**, so a retry doesn't double-charge or double-post.
5. **No-progress detection.** Hash `(tool_name, args)`; identical consecutive calls mean the agent is stuck.

**中文要点**：关键不是"要不要重试"，而是**先分类**——超时/429/5xx 退避重试，参数错/无权限绝不重试；错误要回灌给模型而不是抛异常终止。

---

### 3. What is the difference between structured output and function calling?

**Asked at**: Mistral AI · Apple · JD.com · Meituan · Alibaba
**Source**: https://outcomeschool.com/blog/how-does-function-calling-work-in-llms

**Anchor answer**

| | Structured output | Function calling |
| :--- | :--- | :--- |
| Constrains | **the final response format** (JSON schema / json mode) | **the interim call intent** (`name` + `arguments`) |
| Turns | single turn | multi-turn protocol |
| Who executes | nobody — it's just a shape | **the client**, then feeds the result back |
| Failure mode | schema violation | wrong tool / wrong args / infinite loop |

Relationship: function calling **relies on** structured-ish generation to produce valid `arguments`, but FC is a *protocol* while structured output is a *format constraint*.

The single most important sentence: **the model never executes code.** It only emits a call intent; your code validates the schema, authorizes, executes, and appends the result.

**中文要点**：结构化输出是"答案格式约束"，函数调用是"多轮协议"；模型从不执行代码，只生成调用意图，执行和鉴权都在客户端。

---

### 4. What is MCP (Model Context Protocol) and how does it differ from traditional function calling?

**Asked at**: Microsoft · ByteDance · JD.com · Meituan · Alibaba
**Source**: https://outcomeschool.com/blog/what-is-mcp-model-context-protocol

**Anchor answer**

| | Traditional function calling | MCP |
| :--- | :--- | :--- |
| Where tools live | inlined in every request's `tools[]` | separate **server**, discovered at runtime |
| Discovery | none — static list | `tools/list` |
| Protocol | per-vendor private JSON | **JSON-RPC 2.0** standard |
| Transport | the request itself | `stdio` (local) or **Streamable HTTP** (remote) |
| Capabilities | tools only | `tools` + `resources` + `prompts` |
| Switching framework | rewrite definitions | unchanged — any client can connect |

The value proposition is **write once, connect anywhere**: the tool stops being coupled to one agent framework, and you get dynamic discovery plus a standard auth/transport story.

Cost: an extra hop and a stateful connection to manage (short-lived clients, session routing, per-call accounting).

**中文要点**：MCP 的核心价值不是"也能调工具"，而是**工具与 agent 框架解耦 + 运行时动态发现 + 标准传输层**；代价是多了一跳连接管理。

---

### 5. How many tools is too many? How do you design tool schemas an LLM can actually use correctly?

**Asked at**: Anthropic · Cognition
**Source**: 视频 https://www.youtube.com/watch?v=lnfWvX66FUk

**Anchor answer**

There is no hard number, but selection accuracy visibly degrades past roughly **20–30 concurrently visible tools**, and worse if descriptions overlap.

Schema design rules that actually move the needle:

1. **Group with name prefixes** (`browser_*`, `shell_*`, `crm_*`) so the model can reason about families.
2. **Describe when *not* to use it**, not just what it does. "Use for X; do NOT use for Y, use `y_tool` instead."
3. **Enum > free text.** Every parameter that has a finite domain should be an enum.
4. **Few required fields.** Optional-with-default beats required-and-guessed.
5. **Return shape matters as much as input schema**: cap payload size, truncate + persist to file, return an id/path instead of 10k rows.
6. **Subset per stage** — but **mask, don't remove**: removing tools changes the prefix and invalidates the prompt cache for the whole conversation.
7. Standardize the **error contract** so the loop can classify retryable vs terminal.

**中文要点**：经验阈值 20~30 个；关键是分组前缀、"什么时候别用"的描述、枚举化参数、返回值瘦身，以及**按阶段遮挡而非移除工具**（移除会让整个 prompt cache 失效）。

---

### 6. How does multi-agent orchestration work, and when does it break down?

**Asked at**: Cognition · ByteDance · Meituan · JD.com · Alibaba
**Source**: https://outcomeschool.com/blog/multi-agent-systems

**Anchor answer**

Common topologies:

| Topology | Shape | Good for |
| :--- | :--- | :--- |
| Supervisor / orchestrator | one controller delegates to specialists | heterogeneous subtasks, clear ownership |
| Pipeline | A → B → C | staged transformations |
| Swarm / peer | agents message each other | exploration, simulation |

**Where it breaks down** (this is the part interviewers want):

1. **Unbounded delegation** — A asks B asks C asks A. Needs depth limit *and* a per-root call budget.
2. **Context divergence** — each agent has its own view; they contradict each other with no arbiter.
3. **Duplicated work** — no shared state, two agents both do the retrieval.
4. **Lost global goal** — the sub-agent optimizes its local objective and never sees the user's real ask.
5. **No single owner of the final answer** — nobody merges, or the merger invents consistency.
6. **Cost/latency blow-up** — N agents × M turns, superlinear.

Mitigations: hard depth + call caps, shared state with explicit reducers, a written handoff contract (what in, what out), a deterministic merge/verify step, convergence detection.

**Also worth saying out loud**: for many enterprise cases a **single agent with good tools beats multi-agent** — the extra complexity is often unjustified.

**中文要点**：多 agent 崩在五处——无限递归委派、上下文分叉无人仲裁、重复劳动、丢失全局目标、无人对最终答案负责；另外要敢说"很多场景单 agent 更划算"。

---

### 7. Design memory for a long-running agent: what do you store, where, and how do you retrieve it?

**Asked at**: Anthropic · JD.com · ByteDance
**Source**: https://outcomeschool.com/blog/ai-agent-memory

**Anchor answer**

Split memory by **lifetime and trust level**, not by "chat history".

| Layer | Stores | Where | Retrieval |
| :--- | :--- | :--- | :--- |
| Working | current task state: plan, done steps, pending, evidence ids | structured state object (not raw chat) | always in context |
| Short-term | last N turns + rolling summary of older turns | KV / DB per session | sliding window, summary on overflow |
| Long-term | **verified** facts & preferences | relational DB keyed by `user_id` / `tenant_id` | on demand, top-k with token budget |
| Artifacts | large outputs (reports, logs, scraped pages) | filesystem / object store | keep only path in context; read on demand |

Three rules that matter more than the storage choice:

1. **Write only verified facts.** Inferences and model guesses must never enter long-term memory — that's how you get permanent contamination.
2. **Provenance on every entry**: source, timestamp, confidence, expiry. Retrieval should be able to say "I know this because X".
3. **Forget on purpose**: superseded entries get soft-deleted and dropped from the index, not left to rot.

**中文要点**：按**生命周期与可信度**分层，不是按"聊天记录"；三条铁律——只写已验证事实、每条带来源与有效期、过期要主动遗忘。

---

### 8. How does an agent decide when to call a tool versus answer from its own knowledge?

**Anchor answer**

Three mechanisms, usually combined:

1. **Policy in the prompt**: "if the answer depends on current data, prices, inventory, user-specific records, or requires computation — call the tool; do not guess."
2. **Intent/confidence routing**: a cheap classifier (or small model) routes to `direct answer` / `tool path` / `retrieve` / `clarify`. This is much cheaper than letting the big model decide every turn.
3. **Deterministic post-check**: if the response contains a number, date, or entity that must come from a tool, assert that a tool result exists. If not, block or re-ask.

The failure to watch for: the model answers confidently from stale parametric memory when a tool was available and cheap. The fix is not better prompting alone — it's an **assertion layer** that rejects uncited facts.

**中文要点**：prompt 策略 + 小模型意图路由 + **确定性后校验**（答案里的数字/日期/实体必须能追溯到某次工具返回），三者叠加才可靠。

---

### 9. What makes an agent loop terminate correctly? How do you bound cost and steps?

**Asked at**: ByteDance · Tencent · Meituan
**Source**: https://outcomeschool.com/blog/ai-agent-loop

**Anchor answer**

**Five termination classes, all required:**

| Class | Conditions |
| :--- | :--- |
| Success | `finish_reason=stop`, no tool calls, non-empty content |
| Budget | max steps · cumulative tokens · per-step timeout + task deadline · max tool calls · monetary budget |
| Abnormal | **empty response** · **tool idle-spin** (same tool+args repeated) · **consecutive-failure circuit break** · **no progress** (state unchanged for K steps) |
| Human | user cancel · HITL rejected |
| Safety | unauthorized call / blocked action |

**Four design rules:**

1. Return a **structured reason enum**, never `return ""`. You must distinguish "finished" from "stopped unfinished".
2. **Layer the budgets**: per-step / per-task / per-session / per-tenant — one global `max_steps` is not cost control.
3. **Circuit breaking ≠ retrying.** Retry is "try again"; breaking is "stop fast after N failures".
4. **Idle-spin detection** is the one everybody forgets: the model re-issues the same call because it forgot the previous failure.

**中文要点**：五类终止条件缺一不可；终止必须返回**结构化原因**；预算要分层；熔断不等于重试；空转检测（同工具同参数）最易漏。

---

### 10. How do you make an agent's actions reversible, or at least auditable, in a production system?

**Asked at**: Palantir · JD.com · Alibaba

**Anchor answer**

Start by correcting the premise: **most consequential actions are not reversible** — sending mail, charging money, changing a budget, deleting a record. Don't claim reversibility; shrink the blast radius and make everything traceable.

| Layer | Mechanisms |
| :--- | :--- |
| **Before** | risk tiering (read-only / low / high / forbidden) · HITL for high risk with approver ≠ executor · allowlist + schema enum · **dry-run showing computed impact** · per-call and per-day quotas |
| **During** | **idempotency key** backed by a DB unique constraint (not just a Redis lock) · **intent log written before execution**, result written after · **read-back verification** (`affected_rows > 0`) · cancellation propagation |
| **After** | saga compensation (accepting it is compensating, not undoing) · audit record including **why** (which decision, which evidence, who approved) · replayable trace · short undo window for irreversible actions |

The strongest one-liner: *the goal isn't undo — it's being able to say, five minutes after an incident, which step did it, on what evidence, approved by whom.*

**中文要点**：先纠正前提——多数动作不可逆；三层是事前限权（风险分级/HITL/dry-run/限额）、事中幂等+先声明后执行+写后回读、事后补偿+含"为什么"的审计+可回放 trace。

---

### 11. Design human-in-the-loop approval for an agent that takes consequential actions.

**Asked at**: OpenAI · Palantir · Alibaba · Tencent

**Anchor answer**

Design five things:

1. **Trigger policy** — risk tier, amount threshold, low model confidence, unresolved evidence conflict, or a hard rule ("every refund").
2. **Suspend** — `interrupt` the graph and **persist a checkpoint**. Not a blocking HTTP request that dies with the connection.
3. **Resume** — on approval, **re-validate the world**: the data the plan was based on may be 3 days old. If the version changed, replan instead of executing a stale plan.
4. **The approval UI must be decision-grade**: show the concrete diff/impact ("this touches 37 accounts, total ¥120k"), the evidence, and the model's reasoning. An "approve" button with no context makes the human a rubber stamp.
5. **Fail-closed on timeout** — an approval that expires should auto-reject, never auto-approve.

Plus: separation of duties (requester ≠ approver), full audit of the decision and any edited parameters, and idempotency so a double-click doesn't execute twice.

**中文要点**：五件事——触发策略、interrupt+Checkpointer 挂起、恢复时**重新校验数据版本**、审批界面必须给影响面与证据、超时默认拒绝（fail-closed）。

---

### 12. Your agent drifts after a long run and confidently works on the wrong thing. Diagnose it.

**Asked at**: Cognition

**Anchor answer**

Six root causes, ordered by frequency:

| # | Cause | Mechanism | Fix |
| :-: | :--- | :--- | :--- |
| 1 | **Goal forgetting** | attention is U-shaped; the original objective gets pushed into the middle and stops being attended to | **Todo recap**: rewrite the task list + progress at the *end* of context every few steps |
| 2 | **Error observations accumulate as facts** | a failed/dirty tool result gets retried silently and later treated as ground truth | mark failures explicitly (`retryable:false`); never let inferences into state as facts |
| 3 | **Self-consistency bias** | autoregressive continuation makes a wrong path more confident over time | an **independent reviewer with a clean context** — self-review in the same context is worthless |
| 4 | **Context contamination** | step 20 cites "as we confirmed earlier" where step 8 was a hallucination | separate fact layer from inference layer; inferences never enter long-term memory |
| 5 | Tool-schema ambiguity | too many similar tools → selection drift | mask tool subsets per stage |
| 6 | No explicit task state | state has chat history but no `done/pending/rejected` | model an explicit state machine |

**Diagnosis order**: pull the trace → find the step where output diverged → diff that step's input against the original goal → check for failures immediately before it → check repeated call signatures → check stale state entries.

**中文要点**：六个根因（目标遗忘、错误观察累积、自我确认偏误、上下文污染、工具歧义、缺显式状态）；排障顺序是 trace → 定位偏的那步 → 对比原目标 → 查失败调用 → 查重复签名 → 查过期状态。

---

## 附：这 12 题怎么用

| 场景 | 建议挑 |
| :--- | :--- |
| 筛"伪 Agent" | 9（终止）、12（跑偏诊断）、2（错误分类） |
| 筛"平台底座" | 5（工具规模）、6（多 agent 何时崩）、7（记忆分层） |
| 筛"严肃业务" | 10（可逆可审计）、11（HITL） |
| 概念扫盲 | 1（ReAct）、3（SO vs FC）、4（MCP） |
