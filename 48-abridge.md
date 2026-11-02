# 48 · Abridge

> 11 题。出处：`README.en.md` § Abridge。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Turn a conversation into billable diagnosis codes. What is the accuracy bar, and how do you build to it?

**Anchor answer**

The bar is **asymmetric** because the two error directions are not the same crime.

| Error | Consequence | Tolerable rate |
| :--- | :--- | :--- |
| False positive code (not documented) | **upcoding** — payer audit, clawback, fraud exposure | ~0 at scale; must be near-zero |
| False negative (documented, missed) | lost revenue, incomplete risk profile | low, but recoverable later |
| Right code, wrong specificity (E11.9 vs E11.65) | revenue loss or compliance finding | moderate, fixable by coder |
| No supporting evidence span | the real blocker — a code without documentation is indefensible | 0 |

So the design target is **high recall at ~99% precision, with a quotable evidence span for every code**. In risk-adjustment (HCC) work the audit standard is explicit: the condition must be documented *and* MEAT (Monitor, Evaluate, Assess, Treat) in that encounter — "history of" does not count.

How to build to it:

1. **Two-stage: candidate generation → evidence verification.** Generate candidates from the transcript (and chart history as *hints*, never as evidence), then require an entailment check between the code's definition and a verbatim span.
2. **Never let the model pick the final code alone.** Output `{code, span, confidence, rationale}` and route to a certified coder; measure *coder acceptance*, not model accuracy, as the product metric.
3. **Gold set built by CPC/CRC coders, double-coded**, with disagreements adjudicated — inter-coder kappa tells you the ceiling. If two coders disagree 8% of the time, a model above that is fitting noise.
4. **Stratify**: specialty, encounter type, language, and payer. A model at 0.95 F1 overall can be 0.6 on behavioral health.
5. **Calibrate per code family** (E11.x vs I50.x) because prevalence differs by orders of magnitude; a global threshold over-codes rare conditions.

**中文要点**：编码的错误是**不对称**的——假阳性等于 upcoding（审计/欺诈风险），所以精度要 ~99%，且每个码必须带原文证据片段（MEAT 标准）；做法是"候选生成→证据蕴含校验→认证编码员复核"，用双人标注的一致率当上限。

---

### 2. The note should be ready before the clinician leaves the room. Build me the latency budget, and tell me where the money goes.

**Anchor answer**

Assume the visit ends, the clinician wants to review and sign within **60-90 seconds**.

| Stage | Budget | Notes |
| :--- | :--- | :--- |
| Audio upload / streaming ASR finalise | 3-8s | already streaming during the visit; only the tail is on the critical path |
| Speaker attribution + section detection | 1-2s | cheap classifier over the diarized transcript |
| Note generation (per section, parallel) | 8-25s | the dominant term |
| Grounding/verification pass | 2-5s | claim → span check, can overlap generation |
| Render + EHR prefill | 1-3s | |

The insight that changes the budget: **generate during the visit.** A 20-minute encounter produces ~3k words of transcript; by minute 10 you can draft HPI and ROS on partials and only regenerate the assessment/plan at the end. Effective post-visit latency drops to a few seconds.

Where the money goes — it is **not** the ASR:

| Cost | Driver | Control |
| :--- | :--- | :--- |
| LLM input tokens | full transcript (5-10k tokens) × **one call per section** → the same transcript billed 5-8 times | section-scoped retrieval of relevant spans; prefix caching |
| LLM output tokens | ~600-1200 words per note | cap per section; no chain-of-thought in production |
| Regeneration | clinician edits triggering rewrites | cache by (transcript hash, section, instruction) |
| ASR GPU | audio-minutes × RTF | batch the non-urgent tail; smaller model for diarization |
| Storage/egress | audio + transcript retention | retention policy is both a cost and a compliance lever |

Typical shape: input tokens dominate by 3-5×. **Retrieval-scoped prompting** (send only the spans relevant to the section) is the single biggest lever — and it also improves quality by removing distractor dialogue.

**中文要点**：预算 60-90s，关键不是加速而是**在问诊过程中就增量生成**，收尾只重写评估/计划；成本大头是"整段转写 × 每节一次调用"的输入 token，用按节检索相关片段 + 前缀缓存能省最多。

---

### 3. The patient's chart already lists their medications. How would you use that to improve transcription of drug names, and how would you keep it from backfiring?

**Anchor answer**

Mechanism: **contextual biasing** — inject a per-encounter hotword list into the ASR decoder (shallow fusion over a trie, or a biasing logit boost) rather than post-hoc string replacement.

Sources for the list, in priority order:

1. Active medication list from the chart (normalized to generic + brand names).
2. Problem list → likely drug families (diabetes → metformin, glipizide, insulin glargine).
3. Clinic/specialty formulary — the top 200 drugs for that service line.
4. Prior transcripts for this patient (their pronunciations, their nicknames: "the little blue one" does not help ASR, but "my met-for-min" does).

Why it backfires — and these are the real failure modes:

| Backfire | Mechanism | Mitigation |
| :--- | :--- | :--- |
| **Hallucinated drug** | biasing is strong enough to hear "metformin" in "met her for men…" | cap the boost; require acoustic evidence; verify against phoneme distance |
| Stale list | chart says warfarin, patient stopped it months ago | weight by recency; never treat chart as ground truth for what was *said* |
| Near-miss pairs | hydrALAZINE / hydrOXYzine, LAMICTAL / LAMISIL, glipiZIDE / glyBURIDE | bias only as a *pair set* and disambiguate with a second pass on the audio |
| Bias crowding | 50+ hotwords degrade general WER | cap at ~20-50 terms, rank by relevance; re-measure overall WER |
| Silent correction | the note then asserts a drug the patient never mentioned | post-ASR **verification gate**: if a biased term was decoded but the audio likelihood is low, keep the raw alternative and flag |

Non-negotiable: **biasing changes what the ASR hears; it must never change what the note claims.** Keep the raw n-best and audio timestamps so a reviewer can check, and measure "biasing-induced insertions per 1k drug mentions" as a first-class metric.

**中文要点**：用**上下文热词偏置**（chart 用药 + 问题列表 + 科室药典）而不是事后替换；反噬在于"强偏置听出根本没说的药"、过期清单、近似药名对；要给偏置设上限、保留原始 n-best、并单列"偏置导致的药物插入率"指标。

---

### 4. Design a service that turns the conversation into draft orders: labs, imaging, referrals, prescriptions, via tool calls against the EHR.

**Source**: https://outcomeschool.com/blog/how-does-function-calling-work-in-llms

**Anchor answer**

The hard part is not the tool call; it is that **the model must never be the authorizing party**.

| Phase | What happens | Guardrail |
| :--- | :--- | :--- |
| Extract | pull candidate orders from dialogue (`{type, name, code, laterality, rationale, span}`) | every order needs a transcript span |
| Resolve | map to orderable catalog (LOINC/CPT/RxNorm) for *this* facility | unresolved → do not guess, ask the clinician |
| Validate | duplicates, interactions, dose range, age/weight, coverage rules, required fields | deterministic rules engine, not an LLM |
| Present | draft order set in the UI with the supporting quote and any flag | clinician is the signer |
| Commit | write via FHIR `ServiceRequest`/`MedicationRequest`/`Procedure` with idempotency key | read-back of the created id |
| Audit | `{who drafted, who signed, model version, evidence span, edits}` | immutable log |

Tool schema design that matters:

- **One tool per resource type**, with enums for `priority`, `route`, `frequency`, and a required `rationale_span`.
- `preview_order(...)` returning the resolved catalog match and estimated cost/prior-auth need, separate from `create_order(...)`. The model calls preview first; creation is a distinct, confirmed step.
- Never expose a tool that both creates and schedules; a single call that implicitly books an appointment is the classic irreversible surprise.

Failure modes: the model ordering on "we should check an A1c" (a musing, not an order) — require explicit intent language or clinician confirmation; duplicate orders on retry — idempotency; wrong laterality — never infer from ambiguous speech, force a field; and *silent* catalog substitution (ordered "CBC", catalog maps to "CBC with differential") — show the substitution.

**中文要点**：模型只能"起草"，不能"授权"：抽取必须带原文片段 → 映射到本机构 orderable 目录（映射不到就反问）→ 确定性规则校验（重复/相互作用/剂量）→ 预览 → 临床医生签 → 幂等写入并回读 id + 审计。

---

### 5. Walk me through writing a finished note back into Epic. What goes wrong?

**Anchor answer**

Path depends on the integration; assume the modern one.

| Step | Detail |
| :--- | :--- |
| Auth | SMART on FHIR launch (OAuth2) per clinician, or a backend system client with a per-tenant JWT. Token scope must include the write resource. |
| Identify | `Patient` (FHIR id, not MRN), `Encounter` (visit id), `Practitioner` (author), `PractitionerRole` for the note type |
| Write | `DocumentReference` (or `Composition`, or the vendor's note resource) with `type` = note template, `author`, `date`, `content[].attachment` (text/plain or text/html), `context.encounter` |
| Verify | read back by id; assert `status`, `author`, `subject` |
| Link | attach to the encounter; trigger downstream (coding, charging) |

What actually goes wrong:

1. **Wrong patient.** Ambiguous names, twins, shared MRNs across facilities, or a stale `Patient` reference from a cached session. This is a never-event; assert patient identity at write time, not at session start.
2. **Duplicate notes.** Retries without an idempotency key create two notes; Epic-side there may be no unique constraint to save you. Use a client-generated `identifier` and query-before-create.
3. **Note type / author mismatch.** Writing an "Progress Note" where the template requires "SOAP", or authoring as the wrong practitioner — Epic may reject or, worse, accept and route it to the wrong inbox.
4. **Unsigned vs signed.** Pushing a "signed" note bypasses the clinician's attestation; some endpoints require a cosign workflow and will silently drop your content into a draft only the author can see.
5. **Order/timing metadata.** `date` must be the encounter time in the right timezone, not the generation time, or the note sorts into the wrong day and breaks billing.
6. **Auth orphaning.** SMART tokens tied to a clinician who changes roles mid-deployment → 403s at 2am with no alert.

Mitigation: a **non-production Epic environment** (or a sandbox with synthetic patients) for every integration test, a reconciliation job that diffs intended vs written notes, and alerts on any write whose read-back fails.

**中文要点**：写回 Epic 用 DocumentReference/Composition + SMART on FHIR，坑主要在**写错病人**（要用 FHIR id 而非 MRN、写入时再断言）、**重复笔记**（没有幂等会重试出两份）、笔记类型/作者不匹配、以及"已签名"绕过医生确认。

---

### 6. Two good clinicians write different notes for the same visit. So how do you evaluate note quality at all?

**Anchor answer**

Stop trying to score against *the* reference note. Score along orthogonal axes, each with its own method.

| Axis | Method | Metric |
| :--- | :--- | :--- |
| **Faithfulness** | extract atomic claims from the generated note, check each against the transcript (NLI/LLM judge with span requirement) | unsupported-claim rate, per 1k words |
| **Completeness** | required-element checklist per note type (HPI elements, ROS, meds reconciled, A&P per problem) | element coverage %, weighted by billing/safety relevance |
| **Correctness of specifics** | exact match on structured facts: doses, laterality, dates, numbers | error rate (must be ~0) |
| **Structure/format** | section presence, ordering, template conformance | deterministic pass/fail |
| **Style/utility** | pairwise preference between two notes, **tie allowed**, blinded, by 3+ clinicians | win rate with inter-rater agreement |
| **Downstream** | clinician edit distance, time-to-sign, code capture | tracked, not gating |

The methodological point: **multiple acceptable references**. Build 3-5 clinician-written notes per encounter and score "is the candidate acceptable relative to *any* of them" for completeness, while faithfulness stays absolute. Report the human-human agreement (Cohen's κ) as the ceiling — an AI at 0.7 when humans agree at 0.62 is not a 0.7.

Also necessary:

- **Stratify by specialty and visit complexity**; an aggregate score hides that the model is bad at oncology follow-ups.
- **Adversarial set**: negation, family history, "patient denies", ambiguous pronouns, multiple problems discussed out of order.
- **Freeze a gold set** and never train on it; refresh quarterly with new encounters.
- **Measure the reviewer, not just the model**: LLM-as-judge correlates ~0.6-0.8 with clinicians; calibrate against human labels per axis and re-calibrate when the judge model changes.

**中文要点**：不跟"唯一正确笔记"比，而是拆成**忠实性**（每条断言能否在转写里找到依据）、**完备性**（按笔记类型的必需要素）、**具体值精确性**、**结构合规**、**盲评成对偏好**；多份参考 + 报告人-人一致率作为上限。

---

### 7. Edit rate is the obvious measure of clinician trust. What does it hide, and what would you instrument instead?

**Anchor answer**

Edit rate hides at least five distinct things, and several of them are inverted signals.

| What it hides | Mechanism |
| :--- | :--- |
| **Severity** | a deleted comma and a corrected dose count the same. 3% edits that are all dosages is worse than 15% stylistic. |
| **Complacency** | as trust grows, clinicians skim and sign. Edit rate falls while *unreviewed* error rate is unchanged or rising. |
| **Template conformity** | a clinician rewriting the note into their personal style is not a quality signal at all. |
| **Selection bias** | only notes that get reviewed/edited are measured; silently accepted bad notes are invisible. |
| **Suppressed edits** | clinicians who gave up editing (too slow) migrate to dictating from scratch — that shows up as churn, not edit rate. |
| **Non-note work** | fixing a wrong med in the chart, re-ordering a lab — real cost, outside the note diff. |

Instrument instead:

1. **Typed edit taxonomy** on the diff: `fact_added`, `fact_deleted`, `value_corrected` (dose/number/laterality), `negation_flipped`, `style_only`, `section_moved`. Report the share of *clinically material* edits and **weight by severity**.
2. **Sampled audit of accepted notes** (the ones with zero edits) against the transcript — this is the only way to see the complacency tail. 50-100 notes/week, blinded.
3. **Time-to-sign and time-in-review** per note; a drop in review time with a flat audit error rate is the complacency alarm.
4. **Downstream corrections**: chart medication edits within 24h, order cancellations, addenda, patient callbacks.
5. **Failure escalation rate**: how often the clinician abandons the AI note entirely.
6. **Per-clinician and per-specialty breakdowns** — averages hide the 5% of users who fight the product daily.

Rule of thumb: edit rate is a *leading indicator of adoption*, never of quality. Track it, gate on audited error rate.

**中文要点**：编辑率把"删个逗号"和"改错剂量"算成一样，还会因医生开始不细看而**下降**——那是反向信号。要按类型与临床严重度给编辑加权、抽样审计"零编辑"的笔记、并看复核时长、下游改单/撤回率。

---

### 8. A generated note contains a medication the patient never mentioned. Treat that as a safety incident: how do you detect it before a clinician sees it?

**Anchor answer**

Detection must be **pre-render and fail-closed**, not a post-hoc audit.

Pipeline:

1. **Closed-world extraction.** Every medication entity in the draft is extracted with its surface span and sentence. The note's medication set must be a subset of `transcript-mentioned meds ∪ chart meds explicitly marked "patient reports current use"`. Anything outside the closed world is a hard block — this catches hallucination regardless of how plausible it reads.
2. **Span grounding.** For each med, align the note sentence to transcript turns (lexical + embedding + character-level alignment). Require a supporting span; if the best span scores below threshold, flag.
3. **Negation and attribution check.** "Patient denies taking X" must not become "taking X"; "mother takes X" must not become the patient's med. Run a negation/experiencer classifier over the *transcript span*, not the note, and compare polarity.
4. **Dose/route consistency.** If the note says 20mg and the transcript says 10mg, that is the same incident class.
5. **Gate**: unsupported medication → the note renders without it, the clinician sees a review banner with the candidate and the audio timestamp, and the event is logged with a `note_id`.

Why fail-closed matters: a note is a legal document and a billing input. Dropping a med the clinician can add back costs seconds; a hallucinated med can reach the patient's chart, the pharmacy, and a downstream order.

After the gate: incident process — root-cause (model, prompt, retrieval contamination, biasing), add the case to a **regression eval set**, publish the rate per 1k notes, and set a hard release gate (e.g. no ship above a fixed hallucinated-med rate). Also check the inverse: meds *mentioned but dropped*, which is silent under-treatment documentation.

**中文要点**：必须在渲染前拦截且 fail-closed——笔记里的药物集合要落在"转写里提到的 ∪ 明确标注正在服用的 chart 药"这个闭集内，逐条做片段对齐 + 否定/主体校验 + 剂量一致性，不通过就不进笔记并给医生带音频时间戳的复核提示。

---

### 9. Clinicians will not sign what they cannot verify. How would you build span-level provenance from every line of the note back to the conversation?

**Anchor answer**

Provenance is a **data model problem first**, a UI problem second.

| Layer | Design |
| :--- | :--- |
| Capture | transcript stored as immutable segments: `{seg_id, speaker, t_start, t_end, text, asr_conf, audio_offset}` |
| Generation | constrain output to a structured schema: `{section, sentence, citations: [seg_id...], quote}` — cite-or-die |
| Fallback | if free generation is used, align post-hoc: sentence → candidate spans (BM25 + embedding), then character-level alignment to find the minimal supporting phrase |
| Storage | provenance keyed by `(note_id, note_version, section, sentence_id)`; never overwrite a signed version |
| UI | hover/click a sentence → highlight the quote and seek the audio; show "unsupported" for anything without a citation |

Hard cases and how to handle them:

1. **Paraphrase.** The note says "denies chest pain"; the transcript says "no chest pain, no shortness of breath". Citation should point at the *question + answer pair*, not just the answer. Store a turn-window, not a single sentence.
2. **Aggregation.** "Patient has had three falls in six months" may span four turns. Allow multi-citation and show all spans.
3. **Negation and inference.** "Labs pending" implies an order; that is an inference, so it must cite the turn where it was discussed *and* be labelled as derived.
4. **Chart-sourced content.** "Current meds: metformin 500mg" may come from the chart, not the conversation. Provenance must distinguish **three sources**: transcript span, chart record (with resource id), or model inference (flagged, never auto-accepted).
5. **Speaker attribution errors.** If diarization swaps speakers, the quote is right and the attribution is wrong — show the speaker label with confidence so the clinician can spot it.

Verification UX: default the view to **"show unsupported sentences"** rather than making the clinician hunt. The metric: % of sentences with a resolvable citation, and % of signed notes where the clinician opened at least one citation.

**中文要点**：先做数据模型——转写是不可变的分段时间戳片段，生成时强制"引用或拒绝"（cite-or-die），自由生成再事后对齐；来源要区分**转写片段 / chart 记录 / 模型推断**三类，UI 默认高亮"无依据"的句子。

---

### 10. PHI is in every audio file, transcript and note you touch. How does that shape the architecture, and what can you send to a third-party model API?

**Anchor answer**

PHI changes *where* computation can happen, *how long* data lives, and *which* vendors are eligible. It is an architecture constraint, not a policy PDF.

| Data | Sensitivity | Send to third-party? |
| :--- | :--- | :--- |
| Raw audio | highest — **voice is biometric**, cannot be de-identified reliably | only inside a BAA-covered, zero-retention endpoint; ideally not at all |
| Transcript with names/dates/MRNs | high — direct identifiers + clinical facts | only with BAA + no-training + zero retention |
| De-identified transcript (consistent pseudonyms, shifted dates) | moderate | yes, with a documented de-identification method |
| Note draft | high — same as transcript | same as transcript |

Architecture consequences:

1. **BAA before bytes.** Every model vendor, ASR vendor, and vector store touching PHI needs a Business Associate Agreement; no BAA → no data, no exceptions for "just testing". Test with synthetic patients.
2. **Zero-retention and no-training contractually *and* technically.** Disable vendor logging at the account/org level, verify with a canary string that never appears in vendor dashboards, and re-verify after each vendor release.
3. **De-identification in the pipeline, not at the edge.** Run clinical NER (names, locations, dates, MRN, phone, ages >89) + consistent per-patient pseudonym mapping so coreference survives; date-shift per patient, not per document, or the timeline breaks and the note is wrong.
4. **Minimise what leaves.** Send spans, not full transcripts, to the LLM; keep audio inside your VPC; run diarization/ASR in-region.
5. **Retention as a product feature.** Default 30-90 days for audio, configurable per tenant, with hard delete propagation into caches, logs, and eval sets (a stale eval set is a PHI leak that persists).
6. **Audit every access** with `who/what/purpose`; clinicians' accesses and model accesses are logged differently.

The honest caveat: aggressive de-identification *degrades* the product — ages, dates, and rare-disease specifics are exactly what the note needs. So the real answer is often **on-prem/VPC deployment with the vendor's model weights or a private endpoint**, not "de-identify everything and hope".

**中文要点**：PHI 决定架构：无 BAA 不发数据、零留存且不训练（并用 canary 验证）、去标识化在管道内做且**按病人一致地**替换与偏移日期、只发相关片段不发整段、把留存期做成产品能力并让删除穿透缓存与评测集。

---

### 11. Our audio is a clinic room: two or three speakers, background noise, accents, and a vocabulary full of drug names. How would you build and improve the ASR for that?

**Anchor answer**

Build for **clinically significant errors**, not WER.

| Component | Choice | Rationale |
| :--- | :--- | :--- |
| Capture | clinic-room mic array + beamforming, or a phone/tablet with noise suppression | source quality is the cheapest WER you will ever buy |
| Front end | dereverberation + denoise, but **A/B it**: aggressive denoise destroys fricatives and hurts drug names |
| Diarization | speaker embedding + clustering, constrained to 2-4 speakers; anchor the clinician via enrollment | you need "who said it", not just "what" |
| ASR | streaming model with medical-domain adaptation, plus contextual biasing from the chart | latency budget forces streaming |
| Post | medical term correction, per-speaker punctuation, drug-name normalization to RxNorm | turns text into something usable |
| Review | confidence-gated flagging; low-confidence drug/dose spans surfaced to the clinician | last line of defense |

Improvement loop, ranked by value per engineering week:

1. **Error mining from clinician edits.** Diff the AI note against the signed note, back-project the diff to the transcript span, and you get *labelled* errors on real audio — for free. Cluster them: drug names, accents, crosstalk, numbers.
2. **Stratified test sets.** By accent, specialty, room type, speaker count. A single WER number hides that you are at 30% on one accent.
3. **Metrics that matter**: drug-name error rate, dose/number error rate, negation-flip rate, speaker-attribution error rate — plus WER as a diagnostic only.
4. **Domain adaptation**: continue training on in-domain audio with medical LM fusion; add the clinic's top-500 drug list as a biasing lexicon.
5. **Targeted collection**: pay for 20-50 hours of the exact failure stratum (e.g. non-native English + cardiology) rather than 500 hours of generic clinic audio.

Failure modes specific to clinic rooms: two people talking over each other (the note must attribute, not merge), the clinician's back turned, exam-room HVAC hum at the same band as speech, and dictation-style fast drug lists ("metoprolol, atorvastatin, lisinopril") where biasing must not collapse them into one term.

**中文要点**：先买"采集质量"（麦克风阵列/去混响），再做 2-4 人约束的说话人分离；核心指标不是 WER 而是**药名/剂量/否定翻转/说话人归属错误率**；提升最快的办法是把医生编辑回投到音频片段上做免费标注。

---

## 附：这 11 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 临床 ASR 深度 | 11（诊室识别）、5（药名偏置）、2（延迟预算） |
| 医疗正确性底线 | 1（编码精度）、8（幻觉用药）、9（逐句溯源） |
| 系统集成能力 | 5（写回 Epic）、4（草拟医嘱） |
| 评测设计 | 6（多参考评测）、7（编辑率的盲区） |
| 合规架构 | 10（PHI 与第三方模型） |
