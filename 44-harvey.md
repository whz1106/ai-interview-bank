# 44 · Harvey

> 11 题。出处：`README.en.md` § Harvey。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Paired coding: write a chunker for a legal document that never splits a clause and carries enough context that a retrieved chunk is self-contained

**Source**: https://outcomeschool.com/blog/chunking-strategies-for-rag

**Anchor answer**

**Reference implementation**

```python
import re

CLAUSE_RE = re.compile(r'^\s*(?:ARTICLE|Article|SECTION|Section)\s+[\dIVX]+|^\s*\d+(?:\.\d+)*\s+[A-Z]')
CAP = 700  # tokens, typical

def clause_units(text):
    """Split on clause headings only. A clause is atomic: never cut inside it."""
    lines, units, buf, head = text.splitlines(), [], [], ""
    for ln in lines:
        if CLAUSE_RE.match(ln) and buf:
            units.append((head, "\n".join(buf)))
            buf = [ln]; head = ln.strip()[:80]
        else:
            buf.append(ln)
    if buf:
        units.append((head, "\n".join(buf)))
    return units

def chunk(doc_title, units, defs):
    out = []
    for head, body in units:
        parts = [body] if len(body) <= CAP else _split_sentences(body, CAP)
        for i, p in enumerate(parts):
            refs = [t for t in defs if t in p]           # terms this chunk uses
            ctx = "\n".join(f'"{t}": {defs[t]}' for t in refs)
            out.append({
                "text": p,
                "prefix": f"{doc_title} > {head} (part {i+1}/{len(parts)})",
                "context": ctx,                           # embedded, not stored elsewhere
            })
    return out
```

Over-long clauses are the one legal exception: split at sentence boundaries, but prepend `(part k/n)` and the clause heading, so a fragment still says which clause it belongs to.

**Key points**

- Split on **structural markers**, not fixed token windows. A 512-token window will cut a proviso in half and the retrieval hit becomes legally wrong.
- Each chunk carries: document title, section path, the defined terms it references, and its part index.
- Never merge two clauses into one chunk to hit a size target — retrieval precision drops and citation becomes ambiguous.

**中文要点**：按条款结构切，不按 token 窗口切；超长条款在句子边界二次切并保留「第几条 + 第 k/n 段」前缀；把该块用到的定义术语内联进上下文，块才自洽。

---

### 2. A lawyer asks about a 200-page credit agreement where the operative clause on page 140 depends on a defined term on page 8. How do you build retrieval that gets this right?

**Source**: https://outcomeschool.com/blog/chunking-strategies-for-rag

**Anchor answer**

Embedding similarity alone cannot solve this: page 140 and page 8 are near-orthogonal in vector space, and the link is **symbolic, not semantic**. So build the link explicitly at index time.

| Layer | Build step | What it buys |
| :--- | :--- | :--- |
| Definition extraction | Regex for `"Term" means`, `shall mean`, `has the meaning given in` → `term → definition_text, page` | A term table per document |
| Reference graph | For each chunk, find defined terms it uses and internal refs (`Section 7.2`, `Schedule 3`) → edges | Symbolic retrieval path |
| Chunk enrichment | Inline the definition text of used terms into the chunk's context header | The chunk is self-contained even if retrieval never leaves page 140 |
| Query expansion | Detect quoted terms / capitalized defined terms in the query, resolve via the term table | "Change of Control" maps to its definition |

Retrieval then runs three channels and fuses: dense over enriched chunks, lexical (BM25) over the term table, and graph expansion one hop along `uses → defines`. Rerank the union with a cross-encoder.

Failure mode to expect: terms defined by cross-reference to another term (`"Permitted Indebtedness" has the meaning in Section 6.1(b)`) — follow the chain to depth 2 and store the resolved text, or annotate `unresolved` so the answer says so rather than guessing.

**中文要点**：条款和定义之间是符号引用不是语义相似，向量检索必然失效；索引期抽取「术语→定义」表 + 条款引用图，把定义文本内联进 chunk，查询时按术语路由并做一跳图扩展。

---

### 3. When would you put a whole contract in the context window instead of retrieving over it? Defend the answer with numbers

**Source**: https://outcomeschool.com/blog/lost-in-the-middle-problem-in-llms

**Anchor answer**

Do the arithmetic first, then decide. A 200-page agreement is ~100k words ≈ **130k–150k tokens** (typical legal prose runs ~1.35 tokens/word). So:

| Regime | Fits? | Per-query cost | When correct |
| :--- | :--- | :--- | :--- |
| Whole doc in context | Only if window ≥ ~2× doc size (200k+ models) | Whole doc re-billed every turn unless cached | One document, many questions, questions require global reasoning (consistency, defined-term sweep, "is anything unusual") |
| Retrieval | Always | ~2–6k tokens of chunks | Many documents, narrow questions, cost/latency sensitive |

Decision rule I use: **whole-doc if `doc_tokens < 0.3 × usable_window`, and the session asks ≥3 questions of it, and prompt caching is available.** Otherwise retrieve. The 0.3 factor is not timidity — it reserves room for the answer, the instruction block, and the U-shaped attention deficit documented in lost-in-the-middle: recall for facts in the middle third of a long context degrades materially even when the fact is present, so filling 90% of the window is where "it's in the context but it didn't use it" begins.

The killer against whole-doc at scale is not the window, it is billing: 5,000 contracts × 140k tokens = 700M input tokens per full sweep, versus ~15M with retrieval.

**中文要点**：先算 token：200 页合同约 13–15 万 token。文档 < 可用窗口 30%、同一文档要问多轮、且能命中前缀缓存 → 整篇塞进去；否则检索。别塞满，中段召回会掉。

---

### 4. Design an agent that takes a draft NDA and returns a redlined Word document reflecting the firm's playbook, not a chat response

**Source**: https://outcomeschool.com/blog/ai-orchestration

**Anchor answer**

The deliverable is a `.docx` with **real tracked changes** (`w:ins` / `w:del` in OOXML), authored by a named reviewer, so a partner can accept/reject each one. Emitting a text diff or a chat reply is a failed answer.

| Stage | Mechanism |
| :--- | :--- |
| Parse | `python-docx`/OOXML → paragraph list with stable IDs; keep `rsid` and numbering intact |
| Segment | Detect clause type per paragraph (confidentiality, term, governing law, assignment…), tag paragraphs that are headings/definitions and must not be edited |
| Playbook | Each rule = `{clause_type, condition, preferred_language, fallback_languages, severity}` — structured data, versioned, not prose in a prompt |
| Propose | LLM returns **edit ops**: `{paragraph_id, exact_span, action: insert_after\|replace\|delete, text, rule_id, rationale}` — never free-form redlines |
| Apply | Rewrite OOXML with `w:ins`/`w:del` runs inside the paragraph, keeping surrounding runs untouched |
| Validate | Doc reopens; every op anchored to an existing paragraph+span; no overlapping ops; reject-count reported |

Hard constraints: the model may not rewrite whole paragraphs — span-anchored ops only, otherwise you get silent reformatting and lose the ability to diff. Unmatched rules become **comments**, not edits, so a human decides.

Failure modes: run-splitting in OOXML means one logical phrase is spread across five runs (so span matching must be done on concatenated text, then mapped back); and numbering/list XML breaks if you insert new paragraphs instead of editing in place.

**中文要点**：交付物是带真实修订痕迹（w:ins/w:del）的 docx，不是聊天回复；playbook 做成结构化规则库；模型只输出「段落 ID + 精确 span + 操作」的编辑指令，未命中规则走批注交人判断。

---

### 5. Present the architecture for a workflow reviewing 5,000 contracts against an 18-question diligence checklist, returning a review grid

**Source**: https://outcomeschool.com/blog/ai-orchestration

**Anchor answer**

Constraint that shapes everything: 5,000 × 18 = **90,000 cells**. You cannot run 90k LLM calls of a full contract each; and a grid where 40% of cells say "unclear" is worthless to a deal team.

| Layer | Design |
| :--- | :--- |
| Normalize | OCR + clause segmentation + clause-type tagging once per document (5000 calls, not 90k) |
| Route each of the 18 questions | Tier A *extractive* (does it exist? quote it) → regex/classifier, zero LLM. Tier B *judgment* (is the cap acceptable per playbook?) → LLM on the relevant clause only, ~1–2k tokens |
| Cell contract | `{doc_id, q_id, status: yes/no/na/needs_review, evidence: {clause_id, span}, confidence, quote}` — every cell must carry a span or be `needs_review` |
| Grid | Document × question matrix, sortable by exception; export xlsx with hyperlinks back to the clause |
| QA sampling | Human reviews 100% of `needs_review`, plus a random 2% of high-confidence cells to estimate error rate |

The key economic move is **question→clause routing**: most of the 18 questions map to 2–4 clause types, so you retrieve a 1k-token slice instead of a 40k-token document. That is roughly a 20× token reduction and it also improves accuracy, because the model stops being distracted by irrelevant clauses.

Failure modes: absence questions ("no change-of-control clause") are systematically over-reported as "yes" when retrieval simply failed — so absence must be asserted only after a targeted full-document search pass, never inferred from a failed retrieval.

**中文要点**：5000×18=9 万格，先把文档切一次并按条款类型打标，把每个问题路由到 2–4 类条款只喂 1k 上下文；每格必须带原文 span 否则标 needs_review；"不存在"结论只能来自全文定向检索，不能来自检索失败。

---

### 6. A new frontier model is released and scores better on your benchmarks. What happens before it reaches customers?

**Source**: https://outcomeschool.com/blog/llm-evaluation

**Anchor answer**

Benchmark improvement is an **input to the gate, not a pass**. The gate has layers, and a model can fail any of them.

| Gate | What it measures | Fails if |
| :--- | :--- | :--- |
| Regression suite | Golden set of real tasks with graded rubrics; run old vs new on the same prompt version | Any task category drops beyond noise (use paired comparison, not point estimates) |
| Grounding fidelity | % of sentences with a valid citation span; unsupported-claim rate | Citation coverage down, or citations point to the wrong clause |
| Refusal / not-found behaviour | Does it say "not present in the document" instead of inventing? | More confident hallucination on absent facts |
| Adversarial set | Prompt injection inside documents, jurisdiction traps, conflicting terms | Injection success rate rises |
| Ops | p50/p95 latency, cost/query, output-token inflation | A "better" model that writes 2× longer is a cost regression |
| Shadow + canary | Live traffic on a small cohort, ambiguities reviewed | Escalation-to-legal rate up |

Two gotchas worth saying in the room. First, **a new model can be better on average and worse on your tail** — legal work is tail-dominated, so weight per-category and per-document-type results, not the aggregate. Second, prompt versions do not transfer: a prompt tuned with "you must" for one model can degrade another, so the gate runs on re-tuned prompts, and you keep the old model pinned for rollback at the routing layer.

**中文要点**：榜单分只是入场券；要过回归集、引用保真度、"查无此条"的拒答行为、文档内注入对抗集、延迟与成本四道闸。新模型可能均分更高但在尾部更差，法律场景看尾部；提示词不能跨模型直接复用。

---

### 7. Every assertion in a Harvey answer needs to link back to a specific passage. Design the grounding system, and tell me how you would measure the unsupported-claim rate

**Source**: https://outcomeschool.com/blog/rag-evaluation

**Anchor answer**

Design principle: **citations are generated as part of the claim, not appended afterwards**. Post-hoc attribution is where unsupported claims hide.

| Layer | Mechanism |
| :--- | :--- |
| Addressable units | Every retrieved chunk has an immutable `chunk_id` + char span; the model may only cite these IDs |
| Constrained generation | Structured output: `{claim, cite: [chunk_id, span], confidence}` per sentence; a claim with no cite is a schema violation |
| Verifier | Second pass, NLI/cross-encoder: does the cited span **entail** the claim? Three outcomes: supported / partially / not-supported |
| Repair | Not-supported → rewrite the sentence with the source text or drop it; never "fix the citation" to a different chunk silently |
| Render | Quote + pin cite in the UI; unsupported claims are visually marked, not hidden |

Measuring the **unsupported-claim rate**:

1. **Automatic proxy** on 100% of traffic: `1 − (entailed claims / total claims)`. Cheap, monotone, but biased — label 300 samples by hand to calibrate it, and re-calibrate after every model change.
2. **Human audit** on a stratified random sample (e.g. 200 answers/quarter, stratified by document type and answer length) with two annotators and adjudication; report inter-annotator agreement, because this number is the one you put in front of a customer.
3. **Slice it**: per claim type (rule statement vs. factual extraction), per document family. Aggregate rates hide a 15% failure mode on one template.

**中文要点**：引用是生成时的一部分（结构化输出强制每句带 chunk_id+span），不是事后补；再用 NLI 做蕴含校验，"不支持"就改写或删除而不是换引用。无支撑率 = 全量自动代理指标 + 季度抽样人工标注校准，并分切片看。

---

### 8. An agentic research query returns a memo citing a case that was overruled. Where does that get caught?

**Source**: https://outcomeschool.com/blog/rag-evaluation

**Anchor answer**

Not in generation — a model cannot know a case was overruled from the opinion text alone. It gets caught in a **validation pass over the citation set**, before the memo is assembled.

| Check | Input | Catches |
| :--- | :--- | :--- |
| Existence | Citation → authority database lookup (reporter cite, docket) | Fabricated or malformed citations |
| Treatment / citator | Shepard's / KeyCite-style negative-treatment flags, or internal citator table | Overruled, reversed, superseded, questioned |
| Temporal | Authority date vs. the facts' date vs. "as of" date | Citing a case that did not exist yet, or a statute since amended |
| Jurisdictional fit | Court level, circuit/state, binding vs. persuasive | Persuasive authority presented as binding |
| Quote fidelity | Fetch the actual passage, string-match the quoted language | Misquoted holdings |

Design rules that make this actually work:

1. **Citations are a typed structure from the start** (`{case, reporter_cite, court, year, pin_cite}`), so the validator has something to query. Free-text citations are unvalidatable.
2. **Negative treatment is a hard block, not a warning** — the memo either omits the case or states the treatment explicitly ("overruled by X").
3. **Cache citator responses with a short TTL** and record the citator snapshot date in the memo's audit trail; "was good law when we checked" is a defensible position, "was good law" is not.
4. **Regression case in the eval set**: every overruled-case fixture you have becomes a permanent test; this failure recurs whenever you change retrieval or the model.

**中文要点**：模型无法从判决书正文知道案子已被推翻，必须在引用集合上做校验：存在性 → 引证器负面处理（推翻/撤销）→ 时间效力 → 管辖层级 → 引文逐字核对。引用要生成为结构化字段才可校验；负面处理应硬阻断并写明"已被 X 推翻"。

---

### 9. Two partners at the same firm are on opposite sides of a deal. Design the data isolation for that, on top of normal multi-tenancy

**Source**: https://outcomeschool.com/blog/ai-agent-memory

**Anchor answer**

Normal multi-tenancy says "tenant A cannot see tenant B". This case breaks that: **same tenant, mutually exclusive access**. You need an ethical wall (information barrier) as a first-class object, not a tag.

| Layer | Design |
| :--- | :--- |
| Subject | `matter_id` is the unit of need-to-know; every ingested document carries `{tenant, matter, side}` |
| Wall object | `wall = {id, matter, members_a[], members_b[], effective_at}` — created at conflicts check, immutable, audited |
| Principal | Request identity is `(user, matter_context)`. A user can be legitimately on matter X and walled off it in context Y |
| Enforcement point | **Retrieval filter pushed into the query** (metadata pre-filter on the vector index + lexical index), not a post-filter on results. Post-filtering leaks via rank gaps, counts, and "0 results" side channels |
| Cache | Cache key includes `wall_epoch`; a cached answer built before the wall is created must not be served after |
| Artifacts | Memos, agent memory, conversation history inherit the wall of the matter they were produced in — memory is the classic leak |
| Audit | Every retrieval logs `{user, matter, wall_id, doc_ids_returned, denied_count}` |

Two traps. First, **summaries and generated artifacts** are where walls break: a permitted document summarized into a memo becomes readable in the wrong context. Second, an agent with tool access can re-request the blocked document through a different connector — so the filter must live at the data layer shared by all tools, never in the prompt.

**中文要点**：这是同租户内的信息隔离墙，墙体必须是一等对象（matter 级 need-to-know）；过滤要压进检索查询做前置 metadata 过滤，不能对结果后置过滤；缓存 key 含 wall_epoch；摘要/记忆/生成物继承所属 matter 的墙。

---

### 10. Estimate the cost and turnaround of running your diligence workflow over a 5,000-document data room, and tell me which lever you'd pull first

**Source**: https://outcomeschool.com/blog/ai-orchestration

**Anchor answer**

Build the estimate as a chain, and state assumptions as assumptions.

**Sizing (typical experience values, verify per deal):**

| Quantity | Assumption | Total |
| :--- | :--- | :--- |
| Documents | 5,000 | — |
| Pages/doc | ~40 avg | 200k pages |
| Tokens/doc | ~8k (many docs are short) | ~40M tokens to read once |
| Extraction pass | 1 pass, clause-routed | ~25M input |
| Judgment calls | 18 questions × ~60% requiring LLM on 1.5k-token slices | ~90k × 1.5k ≈ 135M… **too much** |

The last line is the whole insight: naive 18-questions-per-document blows up. So the lever order is:

1. **Cut LLM calls before cutting price.** Route questions to clause types, use extractive checks for existence/quote questions, skip `na` cells by document type. This typically removes 60–80% of calls — bigger than any discount.
2. **Prompt-cache the shared rubric** (playbook + 18 question definitions) and put it first in the prompt so the prefix is reused across all 90k calls.
3. **Batch API / off-peak** for the bulk pass — roughly half price, hours of latency, fine for overnight diligence.
4. **Tier the model**: frontier model only on `needs_review` cells and high-value questions.

Turnaround: a 5,000-doc room at ~2–3k docs/hour with concurrency is an overnight first pass plus a day of human review on exceptions. What you sell is not the first pass, it is the exception queue.

**中文要点**：先算清 18 问×5000 文档会在 LLM 调用量上爆炸；第一杠杆是砍调用（按问题路由到条款、存在性问题用抽取式、可跳过的 na 直接跳），其次是共享 rubric 走前缀缓存、批量接口、分级用模型。交付的是异常队列不是首轮结果。

---

### 11. A partner reports that Harvey missed a change-of-control clause in a contract it reviewed. Debug it

**Source**: https://outcomeschool.com/blog/rag-evaluation

**Anchor answer**

Never start by blaming the model. Walk the funnel — the failure is in exactly one stage, and each stage has a distinguishable signature.

| # | Stage | Diagnostic | Signature |
| :--- | :--- | :--- | :--- |
| 1 | Ingest | Is the clause text in the corpus at all? Grep the raw extracted text | Missing → OCR/scanned page/embedded object/redline doc where deletions were dropped |
| 2 | Segmentation | Is the clause inside a chunk, and is the chunk intact? | Split across chunk boundary; or clause is a *definition* on another page |
| 3 | Retrieval | Was that chunk retrieved for this question? Check logged doc_ids | Not retrieved → lexical mismatch (clause says "Change of Control Event", query says "change of control") |
| 4 | Prompt assembly | Was the chunk truncated/token-dropped when the context was built? | Present in retrieval, absent in the actual prompt |
| 5 | Generation | With the chunk in context, does it answer? Replay the exact prompt | Model says "none found" → instruction/format issue, e.g. an over-strict "only answer if explicit" rule |
| 6 | Post-processing | Is the cell status computed correctly? | Answer present but grid cell shows `no` — a status mapping bug |

Then the fix, by stage: add OCR fallback; keep definitions linked to the clauses that use them; add a **synonym/term-table expansion for defined terms** so "Change of Control Event" is retrievable as "change of control"; assert absence only after a full-document targeted search; add this document as a permanent regression fixture.

And the structural lesson: a single reported miss usually indicates a **class** of misses. Query the eval logs for all cells where the same clause type scored `no` and check whether they share the same ingest or definition page pattern.

**中文要点**：按漏斗逐段定位：入库（OCR/修订痕迹丢失）→ 切块（拆断或落在定义页）→ 检索（术语名与查询不同名）→ 上下文拼装（截断）→ 生成（过严指令）→ 状态映射。修法核心是定义术语的同义词表扩展 + 全文定向检索才允许判"无"，并把该文档加入回归集。

---

## 附：这 11 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 是否真懂法律文档结构（条款/定义/交叉引用） | 1、2、8、11 |
| 能否算清成本、把 5000 文档跑通 | 3、5、10 |
| 工程落地能力（OOXML、span 编辑、真实交付物） | 4 |
| 引用可核验与无支撑率度量 | 7、8 |
| 安全与信息隔离（伦理墙） | 9 |
| 上线闸门与评测纪律 | 6 |
