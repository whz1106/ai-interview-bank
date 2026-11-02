# 42 · Cognition (Devin, Windsurf)

> 9 题。出处：`README.en.md` § Cognition (Devin, Windsurf)。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. You have eight hours to build a coding agent from scratch. Describe what you build and, more importantly, what you cut.

**Anchor answer**

Eight hours is an exercise in **scope, not features**. What ships: one loop, five tools, one demo task.

| Build | Why it is in |
| :--- | :--- |
| Agent loop with explicit state (`plan`, `steps`, `budget`) | without it nothing else is debuggable |
| Tools: `list_files`, `read_file`, `write_file`, `search`, `run_shell` | five is enough to close the edit→test loop |
| Context assembly: repo map + last 3 observations + task | the whole quality difference lives here |
| Termination: budget, no-progress detection, structured stop reason | otherwise it burns the demo budget |
| Trace log: every step's input hash + output | this is what you show at the end |

| Cut | Why it can wait |
| :--- | :--- |
| Multi-agent / subagents | adds coordination bugs before you have a working single loop |
| Fine-tuning / RL | no eval to measure it against yet |
| Vector index over the repo | `grep` + a symbol map is stronger on small repos |
| Sandboxing beyond a Docker container | one container, one repo, read-only creds is enough for a demo |
| Fancy UI | terminal output |
| Rollback/undo beyond `git checkout .` | git already is your undo |

The one thing I would **not** cut is the **verification step**: run the tests, feed failures back. An agent that cannot check its own work looks impressive for ten minutes and is useless after.

Say explicitly what "done" means at hour six, so hours seven and eight are demo polish, not new features.

**中文要点**：8 小时只做"一个循环 + 五个工具 + 明确的终止条件 + 可回放 trace"；砍掉多 agent、微调、向量索引、UI。唯一不能砍的是"跑测试并把失败回灌"的验证环。

---

### 2. Cognition published an argument against multi-agent systems and later published what actually works. Reconcile those two positions.

**Source**: https://outcomeschool.com/blog/multi-agent-systems · https://outcomeschool.com/blog/ai-subagents

**Anchor answer**

They are not contradictory once you separate **what was criticised** from **what was replaced**.

The original argument was against **agents passing messages to each other as peers**: each hop loses context, decisions get made by an agent that never saw the original constraints, and there is no single owner of the truth. Debugging becomes archaeology across N transcripts.

What later "works" is the same insight at a different granularity:

| Pattern | Verdict | Why |
| :--- | :--- | :--- |
| Peer swarm, agents negotiate | avoid | context diffusion, no arbiter, unbounded cost |
| Subagent for **context isolation** (read-only research, parallel search) | works | the parent keeps the decision, the child returns a compressed summary |
| Subagent that **owns a subtask and returns a diff** | works if the contract is explicit | clear input/output, one integrator |
| Multi-agent where each has its own memory and goal | fails | nobody can reconstruct why |

The through-line is *context, not topology*: the failure was never "more than one model call", it was **the loss of a single, complete context that owns the decision**. A subagent is fine when it is a **function** — defined inputs, summarised output, parent decides. It is a bug when it is a **colleague**.

**中文要点**：Cognition 反对的不是"多个模型调用"，而是"平级 agent 互相传话导致上下文分叉、无人拥有真相"；后续"可行"的做法是把子 agent 降级为**函数**（只读调研、返回摘要/diff，父 agent 决策）。

---

### 3. Your agent spends over half its first turn just finding the relevant code. How do you fix that?

**Source**: https://outcomeschool.com/blog/how-does-claude-code-work

**Anchor answer**

Diagnose before fixing: pull the trace and count **tool calls to first correct file**. If that number is 8–20, the problem is that the agent is *searching blind* — it has no map, so its first 10 calls are guesses.

Fixes, ordered by payoff:

| Fix | Mechanism |
| :--- | :--- |
| **Pre-computed repo map** | tree-sitter symbols + imports, ranked; a compact outline (paths + top symbols) injected in the system prompt |
| **Better search tools** | `ripgrep` with regex + glob filters, plus a `glob` tool; expose flags the model can actually use (`-l`, path scoping) |
| **Parallel reads** | issue independent reads/greps in one turn instead of serial round-trips |
| **Task→query rewriting** | one cheap step mapping the natural-language task to likely identifiers ("refund" → `refund`, `RefundService`, `payment`) |
| **Warm start from open files** | the user's editor already knows the file; use it as an initial hint |

Measure it as its own eval metric: *steps to first relevant file*, and *context tokens spent before the first edit*. Otherwise you optimise the final pass rate and never see this cost.

Honest caveat: a repo map costs tokens on every request. Cap it (~1–2k tokens) and make it rank-aware; a 50k-token outline recreates the problem it solves.

**中文要点**：症状是"没有地图只能瞎搜"。解法：预建 AST 符号+导入的仓库地图（限 1–2k token）、让 grep/glob 支持正则与路径过滤、并行发起检索、把任务改写成标识符；并把"到首个相关文件的步数"单独作为指标。

---

### 4. You are training an agent model with end-to-end RL in your own harness. Walk through the environment and reward design.

**Anchor answer**

The environment is the product surface: **real repos, real tools, real failures**.

| Component | Design |
| :--- | :--- |
| Task | issue text + repo at a fixed commit + hidden tests; sampled from a curriculum (easy → multi-file) |
| Sandbox | container per rollout, deterministic image, network off by default, time/step caps |
| Observation | exactly what the production agent sees — same tool schemas, same truncation, same errors |
| Reward | outcome-dominant: tests pass = 1; partial credit = fraction of tests + a small penalty for unused steps and tool errors |

Three things that decide whether the run is meaningful:

1. **Reward hacking is the default.** If gradeable, the model will edit tests, hardcode expected outputs, or wrap everything in `try/except`. Mitigations: held-out tests applied after the episode, mutation checks, a diff-based penalty for touching test files, and a verifier that re-runs from a clean checkout.
2. **Sparse reward needs shaping that is not fake.** Use *dense but honest* signals — test-count delta, compile success, lint delta — and anneal them out, otherwise the model optimises the proxy.
3. **The env must be identical to prod.** Different truncation or error text and you train a model that is good at the training harness and bad at the product.

Algorithm side: group-relative policy optimisation over sampled trajectories, filtered by task success, with a KL anchor to the SFT model. Keep a held-out suite you never train on and re-run it every checkpoint.

**中文要点**：环境必须与线上 harness 完全一致（同样的工具 schema 与截断）；奖励以"隐藏测试通过"为主、测试数/编译为辅助并退火；最大风险是 reward hacking（改测试、硬编码），要靠留出测试 + 干净 checkout 复跑。

---

### 5. Design the execution environment for thousands of concurrent cloud coding agents. It must survive the agent waiting forty minutes for CI.

**Anchor answer**

The central rule: **a waiting agent must not hold a machine.** Split state from compute.

| Layer | Design |
| :--- | :--- |
| Control plane | durable workflow per session (queue + state machine): `planning → editing → waiting_ci → reviewing → done` |
| Compute | one microVM/sandbox per active session, from a warm pool; destroyed or snapshotted when idle |
| Workspace | git clone into the VM, work on a branch; push to a remote so the workspace is recoverable |
| Waiting | on CI, the VM is **stopped and snapshotted** (or the workspace is pushed and the VM released); resumption is a queue event, not a sleeping process |
| Artifacts | logs, patches, checkpoints in object storage, content-addressed |
| Credentials | short-lived, per-session, scoped to one repo, no prod access |

Why not just keep the VM up: 40 minutes × thousands of sessions is both money and a scheduling cliff. Snapshot/resume costs a few seconds and frees the slot.

Failure modes to design for explicitly: **orphaned sandboxes** (TTL + reaper + cost alarm per session), **image pull storms** (pre-pulled node images, overlay FS), **noisy neighbours** (CPU/mem cgroup caps), **resume after 40 min** (the branch moved — re-fetch and rebase, don't silently apply a stale diff), and **cost per resolved task** as the top-line SLO, not utilisation.

**中文要点**：核心是"等待时不占机器"——每会话一条持久化工作流，CI 等待期间把 VM 快照/释放、状态落到队列；再配短时凭证、TTL 回收、预热镜像，并以"每个已解决任务的成本"作为 SLO。

---

### 6. Devin runs asynchronously in the cloud; Windsurf's Cascade runs in the editor next to the user. What actually changes between those two products, technically?

**Anchor answer**

Same model class, radically different **latency contract and failure model**.

| Dimension | Cloud async (Devin) | In-editor (Cascade) |
| :--- | :--- | :--- |
| Budget per unit of work | hours, many turns | seconds per interaction |
| State | durable workflow, checkpoints, resumable | in-memory in the IDE, survives as long as the window |
| Compute | own VM, full shell, install anything | user's machine/remote workspace, sandboxed tools |
| Context source | repo + issue tracker + browser, no human nearby | open files, cursor, recent edits, user's next keystroke |
| Human loop | review the PR at the end | continuous: approve each step, or watch it stream |
| Failure recovery | retry the step, resume from checkpoint, ask in chat | undo, re-prompt, "stop" |
| Concurrency | many tasks in parallel | one user, many files |
| Hardest problem | long-horizon drift + credential/network security | **interruptibility** — every step must be cancellable and reversible |

The interesting technical consequence: the editor product must treat **the user as a race condition**. Any edit can be invalidated mid-flight (see the buffer-conflict problem), so streaming edits must be re-validated at apply time. The cloud product has the opposite problem: nobody is watching, so every step needs a durable checkpoint and an explicit "am I still on the goal?" check, or a two-hour run drifts.

Pricing/UX follows: cloud agents are sold per task, editor agents per seat.

**中文要点**：同一模型、两种契约——云端是"小时级、持久化 checkpoint、无人监督、失败靠重试/续跑"；编辑器是"秒级、可中断、用户即竞态"；因此前者难在长程跑偏与凭证安全，后者难在每一步都可取消、可回滚。

---

### 7. How would you evaluate an autonomous software engineering agent? Explain why SWE-bench pass rates mislead.

**Source**: https://outcomeschool.com/blog/ai-agent-evaluation

**Anchor answer**

SWE-bench is a good *floor*, a bad *headline*. Why the number misleads:

| Issue | Effect |
| :--- | :--- |
| Binary all-or-nothing | 9 of 10 tests passing scores 0; "mostly right" is invisible |
| Narrow distribution | a handful of Python repos; nothing about TS/monorepos/legacy Java |
| Contamination | public issues and patches are in pretraining data; a rising score may be recall |
| Curated issue text | real tickets are vague, contradictory, and missing repro steps |
| Test patch is given | the model is told which tests to satisfy — a massive hint |
| No cost axis | 200 tool calls and ¥40 per issue "counts" the same as 8 calls |
| No harm axis | a diff that also deletes an unrelated module still passes |

What to measure instead, in four buckets:

1. **Resolve rate at a fixed budget** (steps/tokens/time), reported per repo and per difficulty, plus `pass@k` consistency over repeated runs.
2. **Partial credit** — fraction of hidden tests, compile, lint delta.
3. **Diff quality** — lines changed vs. minimal correct edit, files touched, test files modified (red flag).
4. **Honesty & cost** — does it say "cannot reproduce" instead of guessing; cost per resolved issue; human review minutes.

Also evaluate on your *own* backlog: 50 real tickets from the last quarter, with the shipped fix as reference.

**中文要点**：SWE-bench 是地板不是天花板——全或无、仓库窄、题目可能被污染、还白送了 test patch，且不看成本与副作用。改测"固定预算下的解决率 + 部分得分 + diff 质量 + 会不会说做不到 + 每个解决任务的成本"。

---

### 8. An autonomous agent has write access to a customer's repository, CI credentials and network access. What is your threat model?

**Anchor answer**

Start with assets and attack paths, not with a checklist of tools.

| Asset | Attack path | Control |
| :--- | :--- | :--- |
| Source code | exfiltrate over the network | **egress allowlist**, no raw internet, proxy with logging |
| CI secrets | read env vars, print them in logs, base64 them into a commit | secrets never mounted in the agent container; OIDC short-lived tokens; log redaction |
| Supply chain | modify `package.json`/workflow to publish or run on merge | require human approval for CI config, lockfile and release paths; signed commits |
| Branch protection | force-push `main` | agent token can only push to `agent/*` branches; no admin scope |
| Customer trust | prompt injection inside repo content (README, issue, dependency docs) | treat all repo text as **untrusted data**, never as instructions |

Three principles that carry most of the weight:

1. **Least privilege per task, not per customer.** A token that can only push one branch of one repo, valid for one hour.
2. **Assume prompt injection succeeds.** Then the question becomes "what can a fooled agent still do?" — if the answer is "nothing but open a PR", you are fine.
3. **Everything auditable.** Intent log before action, result after, full trace retained; alerts on unusual egress or credential use.

Also model the insider case: a legitimate user asks the agent to do something destructive. Scope limits and approval gates must bind on *action*, not on who asked.

**中文要点**：先列资产与攻击路径；三原则——按任务最小权限（只能推 `agent/*` 分支、一小时有效）、假设 prompt injection 一定会成功（只留"开 PR"这一条路）、动作前写意图日志并可审计；CI/锁文件/发布路径必须人工审批。

---

### 9. As a Deployed Engineer, you are rolling Devin into a 2,000-engineer organisation. What do the first ninety days look like?

**Anchor answer**

Ninety days is a **change-management problem with an integration tail**, not a deployment.

| Window | Focus | Exit criteria |
| :--- | :--- | :--- |
| Days 1–14 | discovery: pick 2 teams, map their workflow, find the 3 ticket types that are high-volume and low-ambiguity | a written scorecard: which work is agent-shaped |
| Days 15–40 | pilot on one repo, tightly scoped (test coverage, migration, dependency bumps); integrate GitHub/Jira/SSO; security review signed off | first merged PRs, baseline metrics captured |
| Days 41–65 | expand to 5–8 teams; build internal champions; office hours; a "known failures" doc | ≥40% week-over-week active usage in pilot orgs |
| Days 66–90 | scale via enablement (docs, recorded walkthroughs, Slack bot), define QBR metrics, feedback loop into product | adoption + merge rate trend, exec review |

Metrics that matter: **PRs merged per active engineer per week**, review minutes per PR, revert rate, and adoption decay after week 2 (the real signal).

Risks to name: shadow adoption before security approval; execs expecting headcount maths in month one; teams burning agent runs on tasks the agent cannot do (needs an intake filter). The DE's job is to make the first successful workflow repeatable and then get out of the way.

**中文要点**：90 天是变更管理不是部署——前两周做发现与评分卡，15–40 天单仓试点（测试/迁移/依赖升级类任务）+ 安全评审，41–90 天扩到多团队做赋能；核心指标是"每人每周合并 PR 数、review 时长、回退率、第二周后的留存"。

---

## 附：这 9 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 取舍与产品判断 | 1（8 小时砍需求）、2（多 agent 立场）、6（云 vs 编辑器） |
| 上下文工程 | 3（找代码开销） |
| 训练/RL 经验 | 4（环境与奖励设计） |
| 基础设施硬功 | 5（数千并发 + 40 分钟 CI） |
| 评测成熟度 | 7（SWE-bench 陷阱） |
| 安全与交付 | 8（威胁模型）、9（90 天落地） |
