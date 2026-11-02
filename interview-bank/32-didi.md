# 32 · DiDi

> 21 题。出处：`README.en.md` § DiDi。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. What does the FFN layer in a Transformer do, and why does it expand before contracting?

**Source**: https://outcomeschool.com/blog/feed-forward-networks-in-llms

**Anchor answer**

Attention mixes information **across positions**; the FFN transforms information **within each position**, independently and identically. It is a two-layer MLP applied token-wise:

```
h = W2 · act(W1 · x + b1) + b2      # d_model -> d_ff -> d_model
```

In a modern decoder block with a gated variant it's `h = W2 ( SiLU(W1 x) ⊙ (W3 x) )`, with `d_ff ≈ 4·d_model` (or `8/3·d_model` for SwiGLU, chosen to keep parameter count equal by dropping the constant term).

**What it does, concretely**: it is where most parameters live (roughly two-thirds of a dense transformer's weights) and where the model stores and retrieves learned patterns. The interpretability result worth citing: `W1` rows behave like **keys** — pattern detectors that fire on specific input features — and `W2` columns like **value** vectors that write a direction back into the residual stream. So the FFN functions as a key-value memory, which is why most factual knowledge is attributed to it and why MoE (which sparsifies exactly this layer) works at all.

**Why expand then contract**: a higher-dimensional intermediate gives the layer room to make features **linearly separable** that were entangled in `d_model`. If you kept the width, `W2 W1` would collapse to a single rank-`d` linear map — the nonlinearity would be sandwiched inside a low-rank bottleneck and nearly all the capacity would vanish. The rank expansion *is* the capacity.

Practical corollaries: interleaving FFN and attention is not arbitrary (attention routes, FFN stores); shrinking `d_ff` to save memory costs disproportionately more quality than shrinking heads; and in MoE you keep `d_ff` per expert small so that `k` active experts ≈ one dense FFN's FLOPs.

**中文要点**：attention 跨位置混合信息，FFN 逐 token 变换；W1 行像 key、W2 列像 value，所以 FFN 是键值记忆、也是参数和知识的主要载体。先升维是为了让特征线性可分——不升维就等于退化成秩为 d 的单个线性映射，容量全丢。

---

### 2. Compare MHA, MQA, GQA and MLA: pros and cons of each.

**Source**: https://outcomeschool.com/blog/grouped-query-attention and https://outcomeschool.com/blog/kv-cache-compression

**Anchor answer**

They differ in **how many distinct K/V heads exist**, which is exactly what the KV cache size scales with.

| Variant | KV heads | Cache per token (per layer) | Quality | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **MHA** | `h` (one per query head) | `2·h·d_head` | baseline | original transformer; cache explodes at long context |
| **MQA** | `1`, shared by all `h` | `2·d_head` → `/h` | measurable drop, training less stable | PaLM, Falcon; great decode throughput, worse quality |
| **GQA** | `g` groups, `h/g` queries per KV head | `2·g·d_head` → `/ (h/g)` | ≈MHA | Llama 2/3, Qwen, Mistral — the current default |
| **MLA** | low-rank latent `c` + decoupled RoPE key | `d_c + d_rope` (e.g. 512+64) | ≈ or > MHA | DeepSeek; best compression ratio |

**Why the tradeoff exists.** Decode is bandwidth-bound — every token reads the whole cache, so halving it nearly halves decode time. GQA with `g = h/8` typically cuts the cache 8× versus MHA for a ~1% quality difference: the best price/performance point found so far, which is why it's the default.

**MLA** goes further: instead of sharing heads, it compresses K/V into a low-dimensional latent and up-projects at use time (the projection can be absorbed into the weight matrices, so the FLOPs mostly disappear), while a separate small RoPE key preserves positional behaviour that naive low-rank compression destroys. Cost: implementation and cache management are harder (a latent cache isn't a head-shaped tensor), and prefill does extra matmuls.

**Where each is wrong**: MQA is a bad choice for a quality-sensitive reasoning model; MHA is bad for long-context serving; GQA is the safe default; MLA only pays off if you control the whole stack.

**中文要点**：区别只在 KV 头数——MHA 全、MQA 一个、GQA 分组、MLA 低秩潜变量+解耦 RoPE。decode 是带宽瓶颈，GQA 用 8 倍压缩换约 1% 质量，所以成了默认；MLA 压得更狠但要自己掌控训练和推理栈，实现与缓存管理都更复杂。

---

### 3. Compare PPO, DPO and GRPO, and write down the SFT loss.

**Source**: https://outcomeschool.com/blog/reinforcement-learning-from-human-feedback-rlhf and https://outcomeschool.com/blog/group-relative-policy-optimization-grpo

**Anchor answer**

**SFT loss** — token-level cross-entropy on response tokens only:

```
L_SFT = - (1/N) Σ_{i=1..N} Σ_{t=1..|y_i|} log π_θ( y_{i,t} | x_i, y_{i,<t} )
```

where `N` counts supervised tokens (not sequences), and prompt/template/tool tokens are masked out. This detail matters more than it looks: if you average per sequence instead of per token, long answers are under-weighted and short ones dominate.

| | PPO | GRPO | DPO |
| :--- | :--- | :--- | :--- |
| Reward model | required | optional (verifiable reward works) | none |
| Value/critic | **required** | **removed** (group mean baseline) | none |
| Sampling | on-policy | on-policy, `G` per prompt | **offline** |
| Objective | clipped ratio × advantage + KL | clipped ratio × group-normalized advantage + KL | logistic loss on (chosen, rejected) |
| Memory | actor + critic + RM + ref | actor + ref (× reward model if any) | policy + ref |
| Ceiling | highest | high | limited by the data |
| Failure mode | instability, critic mismatch | zero-variance groups, length bias | off-policy drift, `β` tuning |

**How to choose**: DPO when you have good preference pairs and need a cheap, stable alignment pass. GRPO when the task has a checker (math, code, tool use) and you want the model to exceed its data. PPO when you need fine-grained per-token credit assignment and can afford the critic — increasingly rare, since GRPO removed most of PPO's value at a fraction of the complexity.

**中文要点**：SFT 是只在回答 token 上算的逐 token 交叉熵（按 token 数平均，别按序列）。PPO=RM+critic+在线采样，天花板最高最不稳；GRPO 去掉 critic 用组内均值，可验证奖励即可训；DPO 离线成对偏好、最便宜但受数据上限约束。

---

### 4. How do you build the loss mask for multi-turn message training?

**Anchor answer**

The mask decides which tokens the model is *trained to produce*. Get it wrong by one token and the model learns to predict the user's next message — loss curves look great, samples are garbage.

**Procedure:**

1. **Render the full conversation once** through the chat template, producing one string plus the tokenizer's offset mapping.
2. **Tokenize the rendered string once.** Do not concatenate separately-tokenized messages — BPE merges across message boundaries and your per-message token counts silently drift.
3. **Build a boolean mask by locating each message's span in the rendered string** (search for the template delimiters, in order), then map spans → token indices via the offsets.
4. **Supervise only assistant content tokens.** Mask system, user, tool-result content.
5. **Decide about the assistant header.** `<|im_start|>assistant\n` is usually masked (the harness emits it), but the **EOS/end-of-turn token must be supervised** — otherwise the model never learns to stop and you get runaway generation.
6. Labels: `labels = input_ids.clone(); labels[~mask] = -100`.

**Implementation gotcha**: with `apply_chat_template(tokenize=True)` you lose offsets; the reliable path is `render → tokenize(return_offsets_mapping=True)` or tokenizing incrementally per message with `add_special_tokens=False` and recording exact lengths. Then **verify empirically**: decode only the supervised positions and read them. If you see a user sentence, the mask is off.

**Multi-turn specifics**: for tool-calling data, supervise the assistant's tool-call arguments but not the injected tool result; and when truncating long conversations, truncate whole turns and keep at least one assistant turn supervised, or the sample contributes zero loss.

**中文要点**：整段渲染后一次性 tokenize（别分条拼，BPE 会跨边界漂移），用 offset 把消息区间映射回 token 位置，只在 assistant 内容上算 loss：模板头 mask 掉但 **EOS 必须监督**（否则模型学不会停），tool result mask 掉。上线前解码一遍被监督的位置自检。

---

### 5. Why introduce RAG at all? Which LLM problems does it actually solve?

**Anchor answer**

RAG exists because **a frozen model cannot know things that changed after training, cannot know your private data, and cannot cite itself.**

| Problem | Does RAG solve it? |
| :--- | :--- |
| Knowledge cutoff / freshness | ✅ yes — the index updates in minutes |
| Private / tenant data | ✅ yes — and it can enforce per-document ACLs at retrieval |
| Attribution | ✅ yes — you can return the chunk that supports the claim |
| Factual hallucination on lookup questions | ⚠️ partly — it converts "make it up" into "retrieve it", but adds a retrieval-failure mode |
| Long-tail specifics (prices, SKUs, policy clauses) | ✅ yes, and this is the real win |
| Reasoning ability | ❌ no |
| Instruction following / format / tone | ❌ no |
| Global aggregation ("how many contracts mention X") | ❌ no — that's a SQL/analytics job |

The blunt framing for an interview: **RAG is a memory system, not an intelligence system.** It changes *what the model has in front of it*, never *what the model can do*. If the failure is reasoning, retrieval makes it worse — you get a confident wrong synthesis over correct documents.

And the second blunt point: in production, most bad RAG answers are **retrieval** failures, not generation failures. Typical experience is that the majority of "the model is wrong" tickets resolve to a chunking or recall problem. That's why you instrument recall@k separately from answer quality; otherwise you tune prompts forever against a retrieval bug.

**When not to use it**: the corpus is small and static (put it in the prompt), the question needs a join or an aggregate (call a tool/database), or the requirement is deterministic exactness (structured lookup, not similarity search).

**中文要点**：RAG 解决的是"模型不知道/记不住/不能引用"——时效性、私有数据、可归因；不解决推理与格式。关键认知：RAG 是记忆系统不是智能系统，且线上大多数"模型答错"其实是检索没召回，所以 recall@k 必须和答案质量分开度量。

---

### 6. Where did your data come from and how was it constructed? Walk me through chunking, retrieval and evaluation.

**Source**: https://outcomeschool.com/blog/chunking-strategies-for-rag

**Anchor answer**

Answer this as a pipeline with the decisions visible, and lead with the eval set — that's the part that signals real experience.

**Sources & construction**: historical support tickets (question/answer pairs, which double as eval ground truth), product and policy docs, and internal SOPs. Cleaning: dedupe near-duplicates by shingle hash, strip headers/footers and boilerplate, normalise dates/units, detect and route PII, and tag every chunk with source, version, product line, and ACL.

**Chunking**: structure-first (split on headings/sections, then by length), 400–600 tokens with ~15% overlap, and a **heading breadcrumb prepended** to each chunk ("Refund policy > Cross-border orders > …"). Headings are the cheapest disambiguation signal there is; without them, "the limit is 30 days" is meaningless out of context. Tables are kept atomic — a split table is worse than an oversized chunk. If the corpus has a Q&A shape, one chunk per Q is often better than a fixed window.

**Retrieval**: BM25 + dense, fused with RRF, then a cross-encoder reranker over the top 50 down to 5–8, then a token budget for context assembly with dedupe.

**Evaluation, split by stage** — this is the key point:

| Stage | Metric | Ground truth |
| :--- | :--- | :--- |
| Chunking | % of answers contained in a single chunk | manual audit of 50 questions |
| Retriever | recall@k, MRR | gold chunk ids |
| Reranker | NDCG@5 | gold chunk ids |
| Generator | faithfulness, answer correctness, citation validity | rubric judge + human spot check |
| End-to-end | answer rate, abstain rate, escalation | production logs |

Build the eval set **before** tuning, include **negative cases** (questions the corpus cannot answer) so refusal is measurable, and re-run it on every chunking or model change.

**中文要点**：数据来自工单 + 政策文档，清洗去重去页眉并打 ACL/版本文档元信息；分块结构优先 400~600 token + 15% 重叠 + 标题面包屑（表格必须原子）；检索 BM25+dense RRF 融合再 cross-encoder 重排。评测必须分层：分块覆盖率、recall@k、NDCG、忠实度、端到端，且必须含负样本才能测拒答。

---

### 7. Would you put a driver's history and preferences into a knowledge base (RAG) or into fine-tuning? Why?

**Anchor answer**

Neither, for most of it — and saying so is the point of the question.

| Content | Where it belongs | Why |
| :--- | :--- | :--- |
| Completed trips, earnings, ratings, current incentives, penalty records | **structured store behind a tool** (`get_driver_stats(driver_id, window)`) | per-driver, exact, time-varying, needs aggregation. "My last 5 trips" is a query, not a similarity search |
| Policies, dispute rules, how incentives are calculated | **RAG**, shared corpus | same text for everyone, changes on a release cadence, needs citation |
| Stable preferences (preferred area, shift pattern, home direction preference) | **compact profile summary injected into the prompt** | small, high-value, must be exact — a 200-token profile beats retrieving 5 chunks |
| Tone / response style / dialect | **fine-tune**, if ever | this is behaviour, and it's the only thing fine-tuning is actually good at |

**Why not RAG for trip history**: the access path is wrong (embedding similarity can't express "last 5 trips" or "sum of earnings this week"), the data is time-sensitive (an index lag of minutes is a wrong answer to a driver asking about today), and cardinality is per-driver — a separate retrieval problem for every user.

**Why not fine-tune for facts**: it teaches a distribution, not a lookup table. You cannot update it in minutes, cannot delete a record on request, and cannot cite it. The failure mode is a model that confidently states yesterday's incentive rate. The classic mistake: fine-tuning a few thousand "driver asks → answer" pairs and concluding it "worked" because the eval set came from the same distribution.

Fine-tune only for *style* at a latency/cost point a prompt can't reach — and keep the facts outside the weights.

**中文要点**：行程/收入/评分这类按司机、时变、需聚合的数据应放结构化库走工具调用；政策类共享知识用 RAG；稳定偏好用 200 token 的画像摘要直接注入。RAG 的访问路径和时效都不适合个人流水，微调更是教分布不教查表——既不能更新也不能删除。

---

### 8. In Elasticsearch, how do you balance indexing depth against query speed, and do you need hot/cold separation?

**Anchor answer**

Indexing depth is a **write-time cost you pay to lower query-time cost**, and the levers are explicit:

| Lever | Write cost | Query effect |
| :--- | :--- | :--- |
| More analyzed fields / analyzers | ↑ | better recall on text |
| `doc_values` / `norms` kept | ↑ disk | needed for sort/agg vs. scoring |
| `index: false` on unused fields | ↓ | none — pure win |
| Lower `refresh_interval` (1s) | ↑↑ | visibility latency only, not speed |
| More shards | ↑ (per-shard overhead) | more parallelism, then worse |
| `forcemerge` to fewer segments | one-off ↑ | faster search, less heap |

Rules that hold in production: target **10–50 GB per shard**, keep shard count roughly equal to search concurrency, set `refresh_interval: 30s` (or `-1` during bulk load) unless real-time visibility is a product requirement, and — the biggest single win — **don't index what you never query** (`enabled: false` for unneeded objects, `index: false` for payload fields).

**Hot/cold**: yes for **time-series with retention** (logs, trip events, audit trail): hot (SSD, active writes, ≥1 replica) → warm (force-merged, fewer replicas) → cold (searchable snapshots / frozen tier on object storage) → delete via ILM. You get most of the cost saving from the tiering policy, not from query tuning.

**But**: for a product-catalogue or knowledge-base workload, hot/cold is the wrong answer — there's no age dimension, every document is equally likely to be queried, and tiering just adds cold-start latency. The right answer there is replica count, shard sizing, and mapping discipline. Distinguishing these two workloads is what the question is testing.

**中文要点**：索引深度是"用写换读"：真正的大头是不索引用不到的字段、控制每片 10~50GB、按需调 refresh；热冷分层只适用于有时间维度和保留期的时序数据（日志、行程事件），商品/知识库场景没有年龄维度，分层只会带来冷启动延迟。

---

### 9. If you use Elasticsearch as a vector store, how do you combine vector and keyword search?

**Source**: https://outcomeschool.com/blog/how-does-hybrid-search-work

**Anchor answer**

Three ways, in increasing order of correctness:

1. **Two queries, manual fusion.** Run BM25 (`query`) and kNN (`knn`) separately, normalise scores, weighted-sum. Simple, but the normalisation is the trap: BM25 is unbounded, cosine is `[-1,1]` — a fixed weight that works on one corpus silently breaks on another.
2. **Single request with both clauses** (ES 8.x supports `knn` + `query` in one search), then fuse. Better latency, same calibration problem.
3. **RRF (reciprocal rank fusion)** — `score = Σ 1/(k + rank_i)`, `k≈60`. Needs no score calibration at all, which is why it's the default choice. Available as an `rrf` retriever in recent 8.x versions, or implementable in a small pipeline.

**Practical pipeline**: BM25 top-100 ∪ kNN top-100 → RRF → cross-encoder rerank the top-50 to 5–8 → context assembly.

**Gotchas that only show up when you run it:**

- **Filter placement.** Put the ACL/metadata filter *inside* the `knn` clause, not after it. Post-filtering a kNN result returns fewer than `k` documents — often zero.
- **`num_candidates`** (per shard, default ~1.5×k) trades recall for latency; raise it before blaming the embedding model.
- **HNSW build cost is real.** Vector indexing is far more expensive than inverted indexing; reindexing a large corpus with HNSW is a capacity-planning problem, not a config tweak.
- **Updates force graph repair.** A changed vector means re-linking the graph, unlike an inverted index which just drops postings.
- **`dense_vector` dims and similarity must match the model**; `cosine` in ES expects normalised or handles it internally — mixing this up gives plausible-but-wrong rankings that look like a bad model.

Route by query type: identifier-like queries (order numbers, error codes) go lexical-only; natural-language questions go hybrid.

**中文要点**：BM25 与 kNN 各自 top-100，用 RRF 融合（不需要分数归一化，是默认选择）再 cross-encoder 重排。三个坑：ACL 过滤必须放进 knn 子句里（后置过滤会返回不足 k 条）、num_candidates 先调再说模型不行、HNSW 构建和更新成本远高于倒排，要提前做容量规划。

---

### 10. Design a smart driver dispatch assistant: order in, accept-or-skip recommendation with reasons out. What is the architecture, and what factors drive the decision?

**Anchor answer**

The critical design call up front: **the decision is made by a small calibrated model, the LLM writes the explanation and answers follow-ups.** Sub-second latency and calibrated probabilities rule out an LLM in the decision path.

| Layer | Contents |
| :--- | :--- |
| Event | order created / driver idle / driver asks "what's near me" |
| Candidate generation | top-N orders by geo + ETA (H3/geohash buckets, not full scan) |
| Features | order: pickup/dropoff distance, fare, surge, expected duration, destination demand; driver: location, heading, current load, accept/cancel history, hours online, home-direction preference; context: weather, area supply-demand ratio, time of day |
| Decision model | GBDT/DNN scoring **P(accept)** × **expected net value**, with the platform constraint (don't strip a zone) as a hard filter |
| LLM layer | renders reasons from the *feature vector*, handles "why this one", "what if I skip" |
| Action | recommendation with an explicit accept/skip, never silent auto-accept |
| Feedback | logged decision + outcome + driver reaction → retraining label |

**Factors that drive the decision**, ranked by what actually moves accept rate: net earnings per km after deadhead, ETA to pickup (drivers hate a long approach), destination desirability (toward home/known demand), the driver's own historical pattern, surge and incentive progress, and fatigue/hours-online.

**The LLM must never invent the numbers.** Pass the computed factors and let it phrase them: "18 min to pickup, ¥4.2 net gain, deadheads away from your usual area." Grounded generation only.

**Metrics**: accept rate, post-accept cancellation, driver satisfaction, platform-level balance (order allocation Gini), and *intervention rate* — how often drivers override the recommendation. Measure accept rate against a control group, not against yesterday.

**中文要点**：决策用小模型（GBDT/深度模型算 P(accept) × 期望净收益），LLM 只负责把特征向量翻译成理由和答疑——亚秒延迟和校准概率都容不下 LLM 在决策环里。关键因子：每公里净收入、接驾 ETA、目的地是否顺路、司机历史偏好、激励进度。指标必须看对照组和司机改判率。

---

### 11. In a stack-trace-to-code-location pipeline, is the LLM interaction single-turn or multi-turn? Why?

**Anchor answer**

**Multi-turn — but a tool-driven loop, not a chat.** The reason is structural: you cannot fit the repo in the context, so localization is inherently sequential search.

Single-turn fails because the model must decide *what to look at next based on what it just found*: parse trace → resolve frame to file → read the function → follow the call into another module → check whether the fix touched it. That is a search trajectory, and the second step's query depends on the first step's result.

A workable loop:

1. **Turn 0 (deterministic)**: parse the trace, resolve each frame to `file:line` via a symbol index / debug info. No LLM needed — this alone localizes a surprising fraction of traces.
2. **Loop** (cap 5–8 hops): `search_symbol`, `read_function`, `find_callers`, `git_log` on the candidate file. Each turn must produce one new candidate, otherwise it's spinning.
3. **Hypothesis scratchpad**: keep an explicit ranked list of candidate frames with evidence, updated each turn — not "remember what we discussed".
4. **Stop** when the stack is fully explained, or the budget expires.
5. **Output** must cite `file:line` per claim, verified by a read.

Why not let it run unattended: unbounded hop count is where cost and drift come from; the most common failure is the agent circling between two plausible frames. Cap the hops and de-duplicate reads by `(path, range)` hash.

**When single-turn is right**: the trace already names the file and you only need a fix suggestion (one shot, cacheable), or the task is matching against known error signatures — deterministic, no LLM loop at all.

**中文要点**：必须多轮，但形式是工具驱动的搜索循环而非聊天——仓库塞不进上下文，定位本质是"看到结果再决定查什么"。turn 0 用符号索引确定性地把帧解析成 file:line，之后限 5~8 跳并维护候选假设表；单轮只适用于已知文件要修复建议，或错误签名匹配。

---

### 12. What do you do when the agent hallucinates and points at the wrong code, or when a file exceeds the context window?

**Source**: https://outcomeschool.com/blog/how-does-context-compaction-work

**Anchor answer**

**Hallucinated location** — treat every file/line/symbol claim as unverified until the harness checks it:

| Claim | Check |
| :--- | :--- |
| "the bug is in `service/order.py:142`" | does the path exist? does the quoted snippet match at that line? |
| "function `apply_refund` is called here" | resolve the symbol in the index; grep for callers |
| "this was changed recently" | `git log` on that file |

Enforce **"no evidence, no claim" in code, not in the prompt**: drop any finding whose anchor doesn't verify, and force a re-search turn instead of accepting it. The specific failure mode to name: the model pattern-matches a plausible module name from another repo or another service — the name *sounds* right, and the surrounding prose is fully consistent, which is exactly why reviewers miss it.

**Oversized file** — never dump it. Escalate the abstraction level:

1. **Symbol extraction** (tree-sitter/AST) → function/class boundaries, then retrieve only the relevant function plus its callers/callees.
2. **Search-with-context**: grep for the pattern, read ±10 lines around hits.
3. **Hierarchical summary**: file summary → section summary → code, so the model can ask for the part it needs.
4. **Spill to artifact**: write the full file to a scratch path, keep `{path, summary, symbols}` in context, read ranges on demand.
5. **Compaction** when the window fills: drop old tool outputs first, keep file paths verbatim (never paraphrase an identifier).

Two cheap guards that matter: dedupe reads by `(path, range)` hash so the agent can't re-read the same 500 lines five times, and cap the number of distinct files opened per task.

**中文要点**：幻觉定位靠 harness 校验——路径存在、引用片段匹配、符号能在索引里解析，不通过就丢弃并强制重新检索（"无证据不下结论"要写在代码里而不是提示词里）。超大文件绝不整篇塞：先用 AST 取符号、grep 带上下文、分层摘要、全文落 artifact 只留路径，压缩时先丢旧 tool 输出且标识符逐字保留。

---

### 13. How do you measure a diagnostic agent's accuracy, and can that validation loop be automated?

**Source**: https://outcomeschool.com/blog/ai-agent-evaluation

**Anchor answer**

**Ground truth is the hard part, and it comes from history**: take resolved incidents where the fix commit is known, and use the files touched by the fix as the gold location. That gives you hundreds of cases for free — with a caveat: the fix touched three files and only one is the cause, so label a `primary` and report both strict (top-1 exact) and lenient (any of the culprit set) accuracy.

Metrics:

| Metric | What it catches |
| :--- | :--- |
| Top-1 / top-3 file accuracy | does it point at the right file |
| Line accuracy within ±N lines | how precise |
| Time / hops to localization | efficiency |
| **Wrong-but-confident rate** | the metric that matters to users — a refusal is cheap, a confident misdirection is expensive |
| Fix-suggestion validity (compiles / tests) | second stage |

**Automation — yes, and it should be the default.** The loop: replay harness feeds each historical trace to the agent, compares cited locations to the gold set, computes metrics, and stores the full trajectory per case. Then:

- **Regression gate in CI**: any prompt/model/retriever change re-runs the set; block on a drop beyond a threshold.
- **Failure taxonomy** auto-bucketed: retrieval miss / symbol-resolution error / wrong frame chosen / hallucinated file / budget exhausted.
- **LLM-as-judge only for the explanation**, never for location correctness — location is checkable deterministically, and judging it with an LLM adds noise for nothing.
- **Production canary**: "engineer accepted the suggestion" and "time to fix" as the online signal.

Re-run each case ≥3 times (agents are stochastic) and re-sample gold labels quarterly — stale labels are the failure mode of every automated eval.

**中文要点**：ground truth 用历史已修复工单的 fix commit 触碰文件，区分 strict/lenient 并标 primary；核心指标除了 top-k 还要有"错得自信率"。可以并且应该自动化：回放 → 比对 → 存轨迹 → CI 回归门禁 + 失败分类 + 线上"工程师采纳率"；位置正确性用确定性比对，LLM 评判只用于解释质量。

---

### 14. Design a customer-service agent that can look up orders, issue coupons and handle complaints. How do planning, tool calling, memory and execution verification each land?

**Anchor answer**

| Layer | Design |
| :--- | :--- |
| **Planning** | intent classification → slot filling → **a plan template per intent**, not free-form planning. `refund_request` = look up order → check eligibility → check policy → execute → confirm. Free-form planners are reserved for multi-intent conversations, and even there the steps come from registered actions |
| **Tool calling** | typed schemas; read tools and write tools separated; every call carries `user_id` + conversation id for authorisation; **idempotency key** on coupon issue and refund; per-conversation rate limits |
| **Memory** | session state object: verified `order_ids`, intent, confirmed slots, already-issued actions. Long-term: verified preferences and past tickets only — never raw chat, never an unverified inference |
| **Verification** | precondition check before (is this order eligible? already refunded?), **read-back after every write** (fetch the coupon, confirm amount and expiry), and a policy engine that rejects out-of-policy actions even when the model asks |

The four are not independent: verification is what makes tool calling safe, and memory is what makes verification *possible* ("already refunded" is only knowable if the session state recorded the earlier attempt).

**Failure modes to design against:**

1. **Double-issue on retry** — the model retries after a timeout; without an idempotency key the user gets two coupons.
2. **Wrong-order action after a topic switch** — the user changed order mid-conversation; the agent acts on the stale id. Fix: re-confirm the id whenever the order reference changes.
3. **Invented policy** — the model paraphrases a refund rule it half-remembers. Fix: policies come from retrieval with citations, and the policy engine enforces them numerically.
4. **Escalation that loses context** — handoff must carry the state object and the transcript.

**中文要点**：规划用"意图 → 槽位 → 每个意图一个固定计划模板"，自由规划只留给多意图；工具读写分离、带用户身份和幂等键、每会话限流；记忆只存校验过的状态和偏好；验证是前置条件 + 写后回读 + 策略引擎兜底。四个经典坑：重试重复发券、切订单后用旧 id、编造政策、转人工丢上下文。

---

### 15. How do you check the quality and accuracy of LLM answers, and how do you trust a checker agent that hallucinates too?

**Source**: https://outcomeschool.com/blog/llm-evaluation and https://outcomeschool.com/blog/llm-as-a-judge

**Anchor answer**

You don't trust the checker — you **bound what it is allowed to decide**, and you measure the checker itself.

Layered, cheapest first:

| Layer | Catches | Trust |
| :--- | :--- | :--- |
| **Deterministic checks** | schema validity, arithmetic, unit consistency, entity/ID existence, citation resolvable, JSON parses | total — these cannot hallucinate |
| **Grounded claim checking** | per-claim: is there a supporting quote in the provided evidence? | high, if the claim/evidence pair is narrow |
| **Rubric LLM judge** | tone, completeness, whether the answer addresses the question | medium — must be calibrated |
| **Second model family** | disagreement triage only | used as a signal, not a verdict |
| **Human sample audit** | everything the above missed | ground truth, at a fixed rate |

Three rules that make a judge trustworthy:

1. **Narrow the question.** "Is claim X supported by this quoted evidence?" is far more reliable than "is this answer good?". Broad judgments drift with prompt wording and answer length.
2. **Require the evidence.** A verdict without a quoted span is rejected and re-asked. This converts an opinion into a checkable artifact.
3. **Calibrate the judge.** Label 200–300 examples by hand, then report the *judge's* precision and recall and its agreement with humans. A judge with 0.7 agreement is usable as a regression signal and unusable as an auto-approval gate — knowing which is the whole point.

Also: judge against a reference answer when you have one (much more reliable), randomize option order to kill position bias, and never let the judge see the model's own reasoning as evidence.

**中文要点**：不去"相信"判官而是限制它的职权：确定性检查（schema/算术/ID/引用可解析）永远先跑，然后做 claim 级"有据支撑"判定，最后才是宽泛的评分判官。让判官可靠的三条：把问题问窄、判定必须附引用原文、用 200~300 条人工标注标定判官自身的准确率——0.7 一致率只能当回归信号，不能当自动放行门。

---

### 16. In an LLM service, how do you cut hallucination, control cost and hold p95? Which lever do you pull first?

**Anchor answer**

**First lever: grounding plus an abstain path.** Not the model, not the prompt wording. Make the answer derived from retrieved/tool-returned evidence with citations, and give the system an explicit way to say "I don't know" (`score < threshold → clarify or escalate`). Measure unsupported-claim rate; that number moves more than any prompt tweak. It is also the cheapest change.

Then, in order of return per unit of effort:

| Lever | Effect |
| :--- | :--- |
| **Shrink the context** — rerank to top-5, drop boilerplate, cap history | cuts cost *and* latency *and* often improves accuracy |
| **Cache** — prefix/prompt cache for the static preamble, semantic cache for repeated FAQ | large cost win, only latency win if hit rate is high |
| **Route by difficulty** — small model for the easy 70%, escalate on low confidence | biggest cost lever after caching |
| **Constrain output** — structured output, max tokens, stop sequences | decode time is output-bound; this is the latency lever |
| **Stream** — first token in ~300ms | perceived latency, not p95 |

**Holding p95** is a tail problem, and tails come from: long inputs (cap them, or chunk + map-reduce), cold caches, retries after provider 429/5xx (cap retries and have a degraded path), and tool timeouts. Fix with per-request SLO budgeting (a deadline propagated into every hop), a fallback path, and load shedding instead of queueing. `p95 = max over paths` — so the fix is usually eliminating one slow path, not making the average faster.

What I would *not* start with: fine-tuning. It's the most expensive lever, needs fresh data, and doesn't fix grounding — a fine-tuned model hallucinating is still hallucinating.

Track TTFT and total separately — different causes, different owners.

**中文要点**：第一根杠杆是 grounding + 明确的拒答路径，而不是换模型或改提示词；然后按性价比依次做上下文瘦身（同时降本降延迟还可能提准确率）、缓存、按难度路由、输出约束、流式。p95 是长尾问题：长输入、冷缓存、重试、工具超时，靠请求级 deadline + 降级路径 + 主动限流，而不是优化平均值。别一上来就微调。

---

### 17. Implement multi-head self-attention from scratch.

**Source**: https://outcomeschool.com/blog/multi-head-attention-in-transformers

**Anchor answer**

**Reference implementation**

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):        # x: [B, T, D]
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)   # each [B, T, D]

        # [B, T, D] -> [B, h, T, d_head]
        def split(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)   # [B, h, T, T]
        if attn_mask is not None:
            # bool mask: True = allowed; or float mask with -inf on forbidden
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        attn = self.dropout(F.softmax(scores, dim=-1))

        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(y)


def causal_mask(T, device=None):
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
```

**Key points**

- **Scale before softmax, by `sqrt(d_head)`.** Not by `sqrt(d_model)` — the softmax is per head, over `d_head` dimensions.
- **Mask with `-inf` (or bool) before the softmax**, never zero out probabilities afterwards: post-softmax zeroing gives rows that no longer sum to 1.
- `.transpose()` is a stride trick, not a copy; `.contiguous()` is required before `.view()` on the merge step (or use `.reshape`).
- The fused `qkv` projection is one GEMM instead of three — a real ~10–20% kernel-level win in eager mode.
- In production use `F.scaled_dot_product_attention`, which dispatches to FlashAttention/memory-efficient kernels; the above is the reference for understanding and for debugging numerics.
- `dropout` applies to attention weights, and must be disabled at inference (`model.eval()`).
- Weight init: `xavier_uniform_` is the standard for these projections.

**中文要点**：重点是先除 `sqrt(d_head)` 再 softmax；mask 要在 softmax 前用 -inf 或 bool，事后清零会让行和不为 1；transpose 是 stride 视图，合并时需 contiguous/reshape；qkv 合并成一个 GEMM 更快；生产环境直接用 SDPA 走 FlashAttention。

---

### 18. How do you debug a slow MySQL query? What do you read in EXPLAIN, and when does a composite index stop working?

**Anchor answer**

**Find it before fixing it**: slow query log (`long_query_time`), `performance_schema.events_statements_summary_by_digest` (ordered by total time, not mean — the frequent 50ms query can cost more than the rare 3s one), or APM traces.

**Then EXPLAIN / EXPLAIN ANALYZE**, reading in this order:

| Field | What you're looking for |
| :--- | :--- |
| `type` | `ALL` (full scan) or `index` (full index scan) is the alarm |
| `key` / `possible_keys` | is the index you expected even considered? |
| `key_len` | **how many prefix columns are used** — the fastest leftmost-prefix check |
| `rows` | estimate; multiply by `filtered` for the real workload |
| `Extra` | `Using filesort`, `Using temporary`, `Using index` (covering), `Using index condition` |

The core smell is **rows examined ≫ rows returned**. That's a selectivity problem, and no amount of query rewriting fixes it.

**A composite index `(a,b,c)` stops working when:**

1. You skip the leftmost column (`WHERE b=…`).
2. A **range** lands on a middle column (`a=1 AND b>2 AND c=3`) — `c` can no longer seek.
3. A function or implicit cast wraps the column (`DATE(ts)=…`, string column compared to a number).
4. Leading wildcard `LIKE '%x'`, or `OR` across non-indexed columns.
5. Collation/charset mismatch on a join column.
6. The optimizer correctly decides a scan is cheaper (low selectivity, small table).

**Fixes, in order**: add/adjust the index (put equality columns first, the range column last), make it covering so no table lookup is needed, keyset-paginate instead of `LIMIT 100000, 20`, and only then consider caching or denormalisation.

**中文要点**：先用慢日志/performance_schema 按总耗时找目标查询（高频 50ms 往往比偶发 3s 更贵），EXPLAIN 按 type → key → **key_len** → rows×filtered → Extra 顺序读。核心症状是扫描行数远大于返回行数。复合索引用不上：跳过最左列、中间列是范围条件、列上有函数/隐式转换、前导通配符、排序规则不一致。

---

### 19. How does SSE work, and what do you watch out for when streaming with it?

**Source**: https://outcomeschool.com/blog/how-does-token-streaming-work

**Anchor answer**

SSE is a **long-lived HTTP response** with `Content-Type: text/event-stream`, over which the server writes newline-delimited frames:

```
event: token
id: 42
data: {"delta":" world"}

```

(double newline terminates a frame; `retry:` sets the client's reconnect delay; lines starting with `:` are comments and are used as heartbeats.)

Client side: `EventSource` (auto-reconnect, auto `Last-Event-ID` replay) or `fetch` + `ReadableStream` (no auto-reconnect, but you get headers, POST bodies, and abort).

**What actually bites in production:**

| Issue | Fix |
| :--- | :--- |
| **Proxy buffering** — nginx/ALB holds the response, tokens arrive in one lump | `proxy_buffering off;` and `X-Accel-Buffering: no`; verify with curl, not with localhost |
| Compression delaying frames | disable gzip on the stream, or accept it only at high chunk sizes |
| **HTTP/1.1 6-connection-per-origin limit** — six tabs and the site hangs | HTTP/2 (multiplexed) |
| Idle connections killed by LBs | heartbeat comment every 15–30s |
| No reconnect in fetch-based clients | implement `Last-Event-ID` + server-side replay buffer, or re-request |
| Client disconnects, generation keeps running | propagate abort → cancel the upstream LLM call (this is pure GPU waste otherwise) |
| Per-token flush | batch 2–5 tokens; flush-per-token destroys throughput at scale |

SSE is one-way, text/UTF-8 only — that's usually fine for token streaming and is exactly why it beats WebSocket here (no custom protocol, works through most proxies, trivially debuggable with curl).

**For LLM streaming specifically**: emit a terminal `[DONE]` sentinel (clients depend on it), send usage in a final event, and measure **TTFT separately from total time** — they're different SLOs with different fixes.

**中文要点**：SSE 是长连接 HTTP + `data:` 帧，事件空行分隔、`:` 开头是心跳。线上最常见的坑是代理缓冲（nginx `proxy_buffering off` + `X-Accel-Buffering: no`）、HTTP/1.1 每域 6 连接上限、以及客户端断开后生成还在跑（必须 abort 上游）。LLM 场景要发 `[DONE]` 并把 TTFT 与总时长分开定 SLO。

---

### 20. Which channels do you use to keep up with new AI techniques, products and developments?

**Anchor answer**

I keep it to a small number of high-signal sources and a personal test loop, because the field's noise-to-signal ratio is the actual problem.

| Channel | Why |
| :--- | :--- |
| Model/provider changelogs and docs | the only place pricing, context limits and deprecations are stated as fact |
| arXiv `cs.CL` / `cs.LG` + Hugging Face daily papers | method-level detail; skim abstracts, read maybe one in twenty |
| **Reference implementations** (vLLM, SGLang, FlashAttention, HF transformers) | reading the scheduler or the kernel teaches more than any blog |
| Release notes of serving/infra tools | paged attention, speculative decoding, quantization changes — where real cost wins come from |
| HN + r/LocalLLaMA | deployment reality: what breaks, what's overhyped, actual hardware costs |
| Benchmarks with prices: LMArena, SWE-bench, Artificial Analysis | comparable numbers instead of vendor claims |
| Conferences / internal reading group | depth, and it forces me to explain things |

The part I'd emphasise in an interview: **the announcement is rarely the signal — the second-order detail is.** Pricing, rate limits, context limits, and whether a model still behaves after a week of real traffic decide whether it changes my stack; launch-day demos decide nothing. And I keep a **personal gold set** of ~20 tasks from my own domain, so any new model or technique gets tested against my problems in an hour rather than judged by vibes.

Budget: roughly 3–4 hours a week, mostly reading code and changelogs rather than papers. And I write down what changed that affects something I own — an unread bookmark is not "keeping up".

**中文要点**：少而高信噪比：厂商 changelog/文档（唯一说清价格与限制的地方）、arXiv + HF daily、**参考实现源码**（vLLM/SGLang/FlashAttention）、infra 工具 release notes、HN 与 r/LocalLLaMA 看落地现实。真正的信号是二阶细节——价格、限流、上下文上限、跑一周后是否还稳；再配一个自己领域的 20 题 gold set 快速验证。

---

### 21. Tell me about taking an AI feature from zero to production, and what went wrong along the way.

**Anchor answer**

Structure it as: what shipped, what broke, what I changed.

**What shipped.** An LLM assistant inside an existing operational workflow (support/diagnosis style): retrieval over internal docs plus a handful of read-only tools, human review on every output at launch, behind a per-team flag.

**What went wrong, in the order it appeared:**

1. **Offline eval overestimated quality.** My golden set was cleaner than production: proper punctuation, one question per message, no typos or code-switching. Real traffic was short, fragmented, and often contained two intents. Fix: sample real traffic into the eval set weekly, labelled by intent type.
2. **Retrieval looked fine, the model ignored it.** Recall@k was good, but answers drifted to parametric memory for anything numeric. Fix: cite-or-refuse — an assertion layer that rejects any number not traceable to a retrieved span.
3. **p95 blew up on long inputs.** One pasted 40-page document took the tail from 2s to 40s. Fix: input caps, chunked summarisation for oversized inputs, streaming for perceived latency, and a per-request deadline with a degraded short answer.
4. **Cost per request was above the human baseline** for the first two months. Fix: prefix caching on the static preamble, reranking to fewer chunks, and routing easy intents to a small model. Cost dropped without touching quality.
5. **Users didn't trust it** even when right. Fix: show the exact quoted span inline, not a document link.

**The lesson I'd state plainly**: the model was maybe 20% of the work; the other 80% was eval harness, logging, versioning and rollback. What made it shippable was the *ability to detect* a regression, not prompt quality.

**中文要点**：讲"上线了什么 → 依次坏了什么 → 怎么修"。五个真实故障：离线评测比线上乐观（真实流量短、碎、多意图）、检索没问题但模型忽略引文（改成 cite-or-refuse）、长输入把 p95 从 2s 拉到 40s、成本高于人工基线（前缀缓存+重排+小模型路由）、用户不信任（展示引用原文）。结论：模型只占 20% 工作量，能发现回归才是可上线的关键。

---

## 附：这 21 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 模型底子 | 1（FFN 为何升维）、2（四种 attention）、3（PPO/DPO/GRPO + SFT loss）、4（多轮 loss mask） |
| 手写代码能力 | 17（MHSA）、4（mask 实现细节） |
| RAG 工程 | 5（为什么上 RAG）、6（数据与评测）、7（画像该放哪）、9（ES 混合检索） |
| 后端基本功 | 8（ES 索引与冷热）、18（慢查询与复合索引）、19（SSE） |
| 业务场景设计 | 10（派单助手）、14（客服 agent 四层）、13（诊断准确率怎么量） |
| 落地与判断力 | 12（幻觉定位/超长文件）、15（怎么信判官）、16（降幻觉/降本/保 p95 的先后）、21（0→1 的坑） |
| 文化与自驱 | 20（信息渠道）、21（复盘能力） |
