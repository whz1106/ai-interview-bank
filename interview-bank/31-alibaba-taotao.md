# 31 · Alibaba (Taobao/Tmall, Alibaba Cloud)

> 27 题。出处：`README.en.md` § Alibaba (Taobao/Tmall, Alibaba Cloud)。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. How does MoE work, how do you balance load, and is expert activation a batch-level, token-level or other decision? Implement a MoE in Python.

**Source**: https://outcomeschool.com/blog/mixture-of-experts

**Anchor answer**

A MoE layer replaces one dense FFN with `E` expert FFNs plus a router. The router scores each token, `top_k` (typically 2 of 8–256) experts are selected, and their outputs are merged with the normalized gate weights. Parameters scale with `E`; FLOPs scale with `k`.

**Granularity is the first design decision:**

| Granularity | What it means | Reality |
| :--- | :--- | :--- |
| Token-level | each token picks its own top-k | the default in every serious implementation (GShard, Switch, Mixtral, DeepSeek) |
| Batch-level | one expert set per batch | only a throughput trick; kills the adaptivity that made MoE worth it |
| Expert-choice | experts pick their top tokens | perfectly balanced by construction, but a token can be dropped entirely |

**Load balancing**, in order of what actually works: (1) a **bias on router logits** adjusted per step by observed load — no aux loss, used by DeepSeek-V3; (2) the Switch aux loss `α · E · Σ f_e · P_e` with `α≈0.01`; (3) capacity factor + token dropping (drop the overflow tokens' residual instead of padding). Without any of these, **expert collapse** happens within a few thousand steps: two experts absorb everything, the rest get zero gradient and die.

**中文要点**：token-level 路由是默认答案，batch-level 只是吞吐技巧；负载均衡首选 router logit 上的自适应 bias，其次 aux loss + capacity factor；不做就是专家坍缩。

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, d_model, d_ff, n_experts=8, top_k=2):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
            for _ in range(n_experts))
        self.top_k, self.n_experts = top_k, n_experts

    def forward(self, x):                       # x: [B, T, D]
        B, T, D = x.shape
        flat = x.reshape(-1, D)                 # <-- token-level routing
        probs = F.softmax(self.gate(flat), -1)  # [N, E]
        w, idx = probs.topk(self.top_k, -1)     # [N, k]
        w = w / w.sum(-1, keepdim=True)         # renormalise over selected experts

        out = torch.zeros_like(flat)
        for e in range(self.n_experts):
            mask = (idx == e).any(-1)           # tokens routed to expert e
            if mask.any():
                gate_w = (w * (idx == e)).sum(-1)[mask]
                out[mask] += gate_w.unsqueeze(-1) * self.experts[e](flat[mask])

        # Switch-style aux loss: push usage towards uniform
        frac = torch.zeros(self.n_experts, device=x.device).scatter_add_(
            0, idx.flatten(), torch.ones(idx.numel(), device=x.device, dtype=probs.dtype))
        frac = frac / (flat.size(0) * self.top_k)
        aux = self.n_experts * (frac * probs.mean(0)).sum()
        return out.view(B, T, D), aux

moe = MoE(512, 2048)
y, aux = moe(torch.randn(2, 16, 512))
```

**Key points**

- `top_k=2` with renormalized weights is the standard; `k=1` (Switch) is cheaper but noisier.
- The aux loss must be added to the total loss with a small coefficient; too large and routing stops being input-dependent.
- Production reality check: the dense compute is cheap, the **all-to-all communication** between expert-parallel ranks is what hurts. Expert placement (grouped by node, 8 experts per GPU) matters more than the router math.
- Small batches break MoE: with `B·T < E`, experts see 1–2 tokens and the router gradient is pure noise.

---

### 2. NCCL timeouts are common in thousand-GPU training. How do you locate and fix them?

**Anchor answer**

A NCCL timeout almost never means NCCL is broken. It means **one rank did not arrive at a collective** and the others sat in the kernel until the watchdog fired (default 600s in PyTorch, 1800s for some jobs).

Root causes, ordered by how often they bite:

| # | Cause | Signature |
| :-: | :--- | :--- |
| 1 | **Divergent control flow** — one rank skipped an `all_reduce` because of a NaN/overflow branch, an early `break`, or a rank-0-only `print` guarded collective | step counters diverge across ranks |
| 2 | **Straggler** — slow node, throttled GPU (ECC/thermal), uneven data length, a rank stuck in dataloader | one rank's log timestamp lags minutes behind |
| 3 | **Network** — a down IB link, wrong `NCCL_SOCKET_IFNAME`, MTU mismatch, ECN misconfiguration | `NET/IB : Got completion with error` |
| 4 | Dead rank — OOM kill, segfault, container evicted | process count drops |
| 5 | Collective-size mismatch — tensor shape differs, e.g. variable sequence count per rank | `Invalid usage` right after |

Localization procedure: enable `NCCL_DEBUG=INFO`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` (or the FlightRecorder), reproduce, then read the last completed collective per rank — the culprit is the rank whose traceback shows *fewer* collectives, or the one whose last log line is oldest. `py-spy dump --pid` on all ranks instantly shows whether a rank is in `ncclCommWait` (network) or in Python (logic).

Fixes: run a fixed number of optimizer steps on every rank (all-reduce the NaN flag, then skip or don't, uniformly), length-bucket the data, add a per-rank heartbeat to the monitor, and prefer elastic restart over tuning the timeout — raising `NCCL_TIMEOUT` only converts a 10-minute failure into an hour-long one.

**中文要点**：超时=有 rank 没到集合操作，不是 NCCL 坏了。按控制流分叉 > 掉队节点 > 网络 > 挂掉 > shape 不一致排序；用 NCCL_DEBUG + FlightRecorder 看谁少做了一次 allreduce，py-spy 看栈；别只会调大 timeout。

---

### 3. How do you do kernel fusion, and why can fusion make things slower? What is warp specialization on Hopper?

**Source**: https://outcomeschool.com/blog/decoding-flash-attention

**Anchor answer**

Fusion = collapsing several kernels into one so intermediates never round-trip through HBM. Three levels: (1) compiler fusion of elementwise chains (inductor/Triton), (2) hand-written producer-consumer fusion (LayerNorm+GEMM epilogue, RoPE+attention), (3) algorithmic fusion — **FlashAttention** is the archetype: QKᵀ, softmax and PV fused inside SRAM so the `[N,N]` score matrix never exists in global memory.

**Why fusion can be slower** — the failure modes are physical, not theoretical:

- **Occupancy collapse.** Fused kernels hold more registers and shared memory. Past ~128 registers/thread or a shared-memory limit, you lose resident blocks and latency hiding, and lose more than you saved in HBM traffic.
- **Parallelism loss.** One giant kernel over 80 SMs with only 40 blocks is worse than four small saturating kernels.
- **Recomputation.** Fused attention recomputes in the backward pass; for short sequences the recompute cost exceeds the saved traffic.
- **Tail effects / persistent kernels** on irregular shapes.
- **Lost vectorization** when the compiler can no longer see alignment.

**Warp specialization on Hopper**: inside one CTA, split warps by role — producer warps issue `cp.async.bulk.tensor` (TMA) loads into shared memory, consumer warps run `wgmma` tensor-core math, synchronized through `mbarrier` arrive/wait. The point is that TMA copies are asynchronous, so the math warps never stall on global memory and the copy warps never occupy tensor cores. Combined with thread-block clusters and distributed shared memory, a CTA can read a neighbour's SMEM directly instead of going through L2.

Rule of thumb: specialize when the kernel is matmul-bound with a long memory pipeline (attention fwd/bwd, GEMM+softmax). Don't specialize a memory-bound elementwise kernel — barrier overhead dominates.

**中文要点**：融合是省 HBM 往返，不是万能；变慢通常是寄存器/SMEM 压力把 occupancy 打掉、并行度不足、反向重算。Hopper warp specialization 用 TMA 生产者 warp + wgmma 消费者 warp 通过 mbarrier 同步，只在访存流水长、计算密集时才划算。

---

### 4. How does KV cache access differ between prefill and decode?

**Source**: https://outcomeschool.com/blog/kv-cache-in-llms and https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization

**Anchor answer**

Same data structure, two completely different access regimes:

| | Prefill | Decode |
| :--- | :--- | :--- |
| Tokens per step | all prompt tokens `T` | 1 (per sequence) |
| Kernel shape | big GEMM `[T,D]×[D,·]` | GEMV `[1,D]×[D,·]` |
| Bottleneck | **compute** (tensor cores) | **memory bandwidth** |
| KV access | **write** `T` new entries, sequential | **read the entire cache** each step |
| Arithmetic intensity | high | ~`2·B` FLOP per byte read → bound by HBM |
| Scaling | quadratic attention cost in `T` | linear in context length, per token |

The number that makes it concrete: for a 70B-class model with 8 KV heads, 80 layers, head_dim 128, fp16, one token of cache costs `2 × 80 × 8 × 128 × 2 B ≈ 327 KB`. At a 4k-token context you re-read ~1.3 GB **per generated token**. That is why decode throughput is quoted in tokens/s/GPU and improves with more HBM bandwidth, not more FLOPs — and why GQA/MLA (shrink the cache) beats anything else for decode.

Consequences for serving:

1. **Chunked prefill** — split a long prompt into chunks and interleave with decode steps, so a 30k-token prefill doesn't stall every other request's inter-token latency.
2. **Prefix caching** — the prefill KV for a shared system prompt is computed once and reused; this is the single biggest win for chat workloads.
3. **PagedAttention** — allocate KV in fixed-size blocks so you don't waste 30–60% of HBM on over-reserved max-length buffers.
4. **Prefill/decode disaggregation** — they want different hardware ratios (prefill: compute-dense; decode: bandwidth + big batch).

**中文要点**：prefill 是算力受限的大 GEMM，KV 只写一遍；decode 是带宽受限的 GEMV，每生成一个 token 都要把整个 KV 读一遍。所以 decode 优化方向是缩 KV（GQA/MLA）、分页、前缀缓存和 chunked prefill。

---

### 5. What post-training stages come after SFT, and what is each for?

**Source**: https://outcomeschool.com/blog/how-does-fine-tuning-work

**Anchor answer**

SFT teaches *the format of an answer*; everything after it teaches *which answer is better*. The stages are alternatives and add-ons, not a mandatory pipeline.

| Stage | Data | What it actually buys | Typical cost |
| :--- | :--- | :--- | :--- |
| **SFT** | (prompt, gold response) | instruction following, chat template, tool-call syntax, domain tone | 10k–1M samples, 1–3 epochs |
| **Rejection sampling / best-of-N distillation** | sample N, keep the ones a verifier passes | a cheap quality bump without RL infra; self-distillation | needs a verifier |
| **Reward model** | pairwise preferences | converts human judgement into a scalar you can optimize | 100k+ pairs, and it is the thing that gets hacked |
| **RL (PPO / GRPO)** | prompts + verifiable or model reward | reasoning, self-correction, behaviours absent from the data | expensive, unstable, needs on-policy sampling |
| **Preference optimization (DPO/ORPO)** | pairs | style/safety/helpfulness alignment without a critic | cheap, offline, no exploration |
| **Safety / refusal tuning** | adversarial + benign pairs | calibrated refusal, jailbreak resistance | small, but over-refusal is the classic side effect |
| **Distillation** | teacher traces | move the result into a small/fast model | as much compute as you saved |

Ordering that works in practice: SFT → verifiable-reward RL if the task has a checker (math, code, tool use) → preference optimization for tone/safety → distill. Adding DPO before the model can produce a correct answer mostly teaches it to be verbose.

The trap: pouring knowledge into SFT. Fine-tuning teaches a distribution, not a lookup table; facts belong in RAG or tools, and the failure mode is a model that sounds authoritative and cites nothing.

**中文要点**：SFT 教格式，其后各阶段教"哪个答案更好"：拒绝采样最便宜、RM+RL 天花板最高、DPO 最稳但无探索。有可验证奖励就上 GRPO，其余走偏好优化，最后蒸馏；知识别塞进 SFT。

---

### 6. Explain the GRPO loss. Why move to GRPO instead of staying with SFT?

**Source**: https://outcomeschool.com/blog/group-relative-policy-optimization-grpo

**Anchor answer**

GRPO is PPO with the value network deleted. For each prompt you sample `G` responses from the old policy, score them, and normalize **within the group** to get the advantage:

```
Â_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)

L = -E[ 1/G Σ_i 1/|o_i| Σ_t  min( ρ_{i,t} Â_i , clip(ρ_{i,t}, 1-ε, 1+ε) Â_i ) ] + β·KL(π_θ ‖ π_ref)
ρ_{i,t} = π_θ(o_{i,t} | q, o_{i,<t}) / π_old(o_{i,t} | q, o_{i,<t})
```

Three moving parts: the **group mean as baseline** (replaces the critic), the **clipped importance ratio** (keeps the update near the sampling policy), and the **KL to a reference model** (stops drift/degeneration).

**Why not stay with SFT**: SFT is offline imitation on a fixed set — it can only push probability up on the tokens in your data, so it cannot express "answer A beats answer B" and cannot discover behaviour that isn't already in the corpus. A reasoning trace that fails 90% of the time is 90% negative signal in SFT; under GRPO only the *relative* ranking matters, so the model learns to prefer the successful trajectory even when every trajectory is imperfect. That is the whole point: **SFT needs a correct answer, RL needs only a comparison.**

Failure modes you will actually hit: **zero-variance groups** (all `G` correct or all wrong → `Â=0` → no gradient; filter them out and oversample hard prompts), division by zero when `std=0`, reward hacking of the verifier, and length bias — longer answers get more tokens of gradient and, with a length-normalized reward, can be gamed by padding.

**中文要点**：GRPO = 去掉 critic 的 PPO，用组内 mean/std 当 baseline，保留 clip 比率和 KL。相比 SFT 的关键优势是"只需比较、不需标准答案"，能把失败样本里相对更好的轨迹学出来；坑是全对/全错组梯度为零。

---

### 7. How is advantage computed, why not use reward directly, and why is importance sampling needed?

**Anchor answer**

**Advantage = reward minus a baseline**: `A(s,a) = r - b(s)`. In GRPO `b` is the group mean (optionally divided by the group std); in PPO it is `V(s)` plus GAE over time.

**Why not the reward directly.** The policy-gradient estimator is `∇J = E[∇log π(a|s) · r]`. It is unbiased, but the variance is huge: if a hard prompt has mean reward 0.2, then ~80% of your samples carry a *negative* multiplier and every token in them gets pushed down — including the correct tokens in a mostly-right answer. Subtracting any baseline that does not depend on the sampled action leaves the expectation unchanged (`E[∇log π · b] = 0`) while removing the prompt-level offset. That makes the update "better than average *for this prompt*", which is exactly the signal you want. Normalizing by std further equalizes prompts of different difficulty (technically a biased estimator, empirically worth it).

**Why importance sampling.** Samples come from `π_old`; gradients are taken on `π_θ`. After the first inner step the two diverge, so `E_{π_old}[∇log π_θ · A]` no longer estimates the on-policy gradient. The ratio `ρ = π_θ/π_old` corrects the distribution mismatch. With exactly one gradient step per batch, `ρ ≡ 1` and IS is a no-op — the moment you do multiple epochs per rollout batch (which is where the sample efficiency comes from), it is mandatory. Clipping `ρ` bounds the trust region; without it a single high-advantage batch can blow up the policy.

Subtlety worth stating: the ratio is computed **per token**, but the advantage is per sequence. That asymmetry is what makes token-level clipping meaningful.

**中文要点**：advantage = reward − baseline，减 baseline 不改变梯度期望但大幅降方差，否则难样本里正确的 token 也被压。重要性采样是因为样本来自旧策略、多轮内层更新后分布已偏，比率 ρ 做纠正，clip 限制信任域。

---

### 8. Credit assignment: how does a sequence-level reward get distributed across tokens in GRPO?

**Anchor answer**

Bluntly: **in GRPO it doesn't get distributed at all.** `Â_i` is computed once per response and broadcast to every token in it:

```
loss_i = -1/|o_i| · Σ_t  min(ρ_{i,t} Â_i, clip(ρ_{i,t}) Â_i)
```

Every token of a winning trajectory is reinforced equally, including the wrong intermediate steps; every token of a losing one is suppressed, including the correct reasoning that led to a single arithmetic slip. That is the sparse-reward credit-assignment problem, and GRPO's answer is "use enough samples that the noise averages out" — which works surprisingly well for verifiable tasks, and fails when trajectories are long and the error is localized.

The ways to actually assign credit:

| Method | Mechanism | Cost |
| :--- | :--- | :--- |
| **Token-level discounting** | multiply the advantage by `γ^t` or weight later tokens more | free, weak |
| **Process reward model (PRM)** | score each step, advantage per segment | needs step labels, and PRMs are hackable |
| **Turn-level advantage (multi-turn)** | reward per tool call / turn | needs per-turn rewards |
| **Verifier-guided filtering** | drop samples whose reasoning is provably invalid before computing the group | cheap, high value |
| **GAE with a critic (PPO)** | per-token advantage from `V(s)` | a value model, i.e. what GRPO removed |

Two biases to watch: **length normalization** divides by `|o_i|`, so a short correct answer and a long correct answer contribute differently per token — models learn to game length in whichever direction the reward favors. And **masking**: template tokens, tool-result tokens and padding must not be in the loss, or you reinforce the harness rather than the policy.

**中文要点**：GRPO 里序列级 advantage 直接广播到每个 token，中间步骤的对错不区分；修法是 PRM、按 turn 给奖励、验证器过滤坏轨迹，或退回 PPO+GAE。注意按长度归一化会引入长度偏置，模板/tool 返回的 token 必须 mask 掉。

---

### 9. What does DPO improve over PPO, and what should you watch out for when training it?

**Source**: https://outcomeschool.com/blog/direct-preference-optimization-dpo

**Anchor answer**

DPO observes that the RLHF objective has a closed-form optimal policy, so the reward can be expressed in terms of the policy itself and substituted into the Bradley–Terry preference likelihood. What remains is a plain classification loss on `(chosen, rejected)` pairs:

```
L = -log σ( β·[ log π_θ(y_w|x) - log π_ref(y_w|x) - log π_θ(y_l|x) + log π_ref(y_l|x) ] )
```

**What it improves**: no reward model, no value model, no on-policy sampling loop. Two models in memory instead of four, one supervised-style training job, far more stable. It also dodges reward hacking of an explicit RM, because there is no separate RM to hack.

**What to watch out for:**

1. **Distribution shift.** DPO assumes the preference pairs come from the policy's own distribution. Pairs generated by a different model (or by humans writing ideal answers) produce an implicit reward that is wrong off-policy — the model learns to imitate the pair source, not to prefer.
2. **`β` is not a free knob.** Too low → the implicit reward dominates and you get degenerate, repetitive, high-margin outputs; too high → almost no learning.
3. **No exploration.** It's offline: the model cannot discover an answer better than the best one in your pairs. Reasoning tasks usually need RLVR, not DPO.
4. **Length bias.** Longer chosen responses inflate the log-prob gap; a length-controlled variant or length-matched pairs is often necessary.
5. **Label noise.** Human preference agreement is typically only 70–80%; a few percent of flipped labels visibly degrades the result.
6. **Reference drift.** When you iterate (DPO → new model → DPO), the reference must track the base you actually started from, or the KL term silently disappears.

**中文要点**：DPO 把 RLHF 的闭式解代回偏好似然，省掉 RM 和 critic，训练稳。但它是离线的：偏好对必须来自当前策略分布，β 过小会退化，长度偏置和标注噪声是两个最常见杀手，推理任务它替代不了 RLVR。

---

### 10. When do you tune the prompt and when do you fine-tune? How do you decide between RAG and fine-tuning for a customer?

**Anchor answer**

Escalate in cost order, and only when the cheaper lever is demonstrably exhausted:

**prompt → tools/RAG → fine-tune → RL.**

| Symptom | Lever |
| :--- | :--- |
| Wrong format, wrong tone, ignores an instruction | prompt / few-shot (hours) |
| Missing or stale facts, needs citations, per-user data | RAG + tools (days) |
| Correct knowledge but wrong *behaviour* — never follows the schema, drifts off-persona, too slow/expensive at scale | fine-tune (weeks) |
| Task has a verifiable reward and the model is near the ceiling | RL |

Decision rules that survive contact with a customer:

1. **Ask what changes.** Knowledge that changes weekly rules out fine-tuning — you'd retrain every week and still be stale.
2. **Ask whether they need attribution.** If "which document says this" matters, it's RAG by construction; a fine-tuned model cannot cite.
3. **Count the labels.** Under ~100 examples you cannot fine-tune meaningfully; you can still do few-shot.
4. **Check the latency/cost target.** If the requirement is 8B-model latency at 10k QPS, fine-tune a small model on the big model's traces — that's a real reason to fine-tune even when RAG would answer correctly.
5. **They're not exclusive.** The common production answer is a fine-tuned model for format/tool-calling discipline *plus* RAG for facts.

The failure to name explicitly: fine-tuning a knowledge base into weights. It produces confident, uncitable, un-updatable, undeletable answers — the worst of all four properties.

**中文要点**：按 提示词 → RAG/工具 → 微调 → RL 的成本顺序升级。判断口诀：知识常变或要引用就走 RAG，行为/格式/延迟不达标才微调；两者常常组合。最忌讳把知识库微调进权重——既不能引用也不能更新删除。

---

### 11. How do chunking, BM25 plus dense hybrid retrieval, multi-hop retrieval and query rewriting fit into one pipeline?

**Source**: https://outcomeschool.com/blog/how-does-hybrid-search-work and https://outcomeschool.com/blog/how-does-a-reranker-work

**Anchor answer**

Six stages, and each one exists because the previous one fails in a specific way:

| Stage | Does | Fails when |
| :--- | :--- | :--- |
| **Parse + chunk** | structure-aware split (headings, tables), 300–800 tokens, 10–20% overlap, heading breadcrumb prepended | boundaries cut a table or a procedure in half |
| **Enrich** | per-chunk summary, keywords, hypothetical questions, metadata | enrichment drifts from the source text |
| **Index** | BM25 (sparse) + dense vectors, same doc ids, same ACL field | the two indexes go out of sync |
| **Query rewrite** | decompose multi-part questions, expand acronyms, HyDE, resolve pronouns from history | it destroys exact-match queries (error codes, SKUs) |
| **Hybrid retrieve** | top-100 BM25 ∪ top-100 dense, fused with RRF | score scales differ (BM25 unbounded, cosine 0–1) and naive weighted sums fail |
| **Rerank** | cross-encoder over top-50 → top-5..8 | latency: a cross-encoder is 10–100× a bi-encoder |

**Multi-hop** is not a stage but a loop around retrieval: if the retrieved context contains a bridge entity the question depends on but doesn't resolve ("the policy for *that* tier"), issue a second query derived from the first result. Cap it at 2–3 hops — unbounded hops are the main cause of runaway RAG cost and drift.

Two hard-won details. First, **route by query type** rather than always rewriting: identifier-like queries should go straight to BM25 (lexical) with rewriting disabled. Second, **fusion with RRF, not weighted sums** — `1/(k+rank)` needs no score calibration and is robust across corpora, whereas a weight you tuned on one tenant silently breaks on the next.

Measure each stage separately — recall@k, then NDCG, then faithfulness — or you cannot tell a retrieval miss from a generation failure.

**中文要点**：分块→增强→双索引→查询改写→BM25+向量 RRF 融合→cross-encoder 重排，多跳是外层的循环（限 2~3 跳）。两个关键取舍：融合用 RRF 不用加权求和，标识符类查询要绕过改写直接走 BM25；每段单独度量才能定位问题。

---

### 12. Design an enterprise knowledge base with ACLs and PII. Where does permission filtering live?

**Anchor answer**

The rule: **permission filtering happens inside retrieval, before ranking — never after generation, and never only at the UI.**

Why not post-filter: retrieve top-10, 8 are forbidden, you answer from 2 and the answer is empty or wrong. Why not prompt-level filtering ("don't use documents you shouldn't"): that's a suggestion, not a control, and the text is already in context — it has been leaked to the model provider.

Layers, defense in depth:

| Layer | Mechanism |
| :--- | :--- |
| Ingest | extract document ACL (owner, group, inherited folder ACLs), store as a filterable field on **every chunk** |
| Index | tenant-partitioned indexes, or a filtered HNSW (ES `knn.filter`, pgvector `WHERE`, Milvus partition keys) so ANN traverses only permitted nodes |
| Query | resolve principal → group set at request time from the IdP; build the filter expression server-side, never from client input |
| Post-retrieval | re-verify each chunk's ACL against the principal (catches stale cached filters) |
| Generation | citations returned only for chunks that passed; log the principal and doc ids |

PII: detect at ingest (regex for structured identifiers + NER for names/addresses), then **tokenize into a vault** (`PERSON_7f3a`) rather than deleting — deletion destroys the semantics, and a reversible mapping lets you rehydrate only for authorised viewers. Keep PII in a separate store from the embedding index, because embeddings leak: a raw name embedded in a vector is recoverable by inversion.

Failure to design for: **ACL staleness.** Permissions are revoked while chunks sit in a semantic cache or KV prefix cache. Everything cached must carry an ACL version, and a permission change must bump it.

**中文要点**：权限过滤必须在检索内部、排序之前——事后过滤会让 top-k 塌掉，提示词里的"别用无权限文档"根本不是控制。ACL 落到 chunk 级 + 索引级过滤 + 检索后复核；PII 用可逆 token 化放独立库，因为嵌入向量可被反演。最易漏的是缓存里的 ACL 过期。

---

### 13. How do long-conversation summarisation and the harness data flow fit together?

**Source**: https://outcomeschool.com/blog/how-does-context-compaction-work

**Anchor answer**

Compaction is a **harness function, not a model behaviour**. The model cannot see its own token count reliably and will not volunteer to forget; the harness owns the transcript, the budget, and the rewrite.

The data flow:

```
turn appended → token accounting (per message + tool output)
              → threshold (~70–80% of window) → compaction decision
              → choose victims (oldest turns; tool outputs first, they're the biggest)
              → summarise into a structured template
              → replace victims with summary + pointers to spilled artifacts
              → re-inject a "state card" at the END of context
              → next model call
```

The summary template matters more than the summarizer model. It should be slot-filled, not prose: decisions taken, constraints stated by the user, open questions, entities and IDs, artifact paths, what has been tried and failed. **Anything that must be exact — order ids, amounts, file paths, code — is preserved verbatim or referenced by path, never paraphrased.** That single rule prevents most compaction-induced failures, because the classic failure is a summarizer that turns "refund up to ¥398" into "process the refund".

Where it goes wrong:

1. **Recursive drift** — summary of a summary of a summary. Keep the original user constraints in a protected region that is never compacted.
2. **Silent capability loss** — the compaction drops tool schemas or an approval decision, and the agent re-asks for something already approved.
3. **Lost provenance** — the summary says "we confirmed the price" and nobody can tell where from.
4. **Latency spikes** — compaction itself is an LLM call inside the loop; it must be budgeted and preferably run off the critical path.

**中文要点**：压缩是 harness 的职责不是模型行为。流程：token 记账 → 阈值触发 → 挑最老的 turns（先砍 tool 输出）→ 结构化模板摘要 → 保留 artifact 路径 → 状态卡贴到上下文末尾。铁律：ID/金额/路径逐字保留，用户原始约束放进"永不压缩"区，否则递归摘要必然漂移。

---

### 14. Where is the line between the agent loop and the harness, and how do you early-stop a long-running task?

**Source**: https://outcomeschool.com/blog/ai-agent-loop and https://outcomeschool.com/blog/harness-engineering-in-ai

**Anchor answer**

**The loop** is the model-turn cycle: assemble messages → call the model → parse tool calls → execute → append observations → repeat. **The harness** is everything that decides whether that cycle is allowed to continue, and what it is allowed to touch: context assembly and compaction, tool registry and schemas, sandboxing and permissions, budgets, retries and circuit breakers, checkpointing and resume, approval gates, telemetry.

The dividing test: *if a component needs to know what the model said, it's the loop; if it needs to know what the system is doing — cost, time, permissions, state — it's the harness.* A useful corollary: harness logic must be **deterministic and testable without an LLM**. If your stop conditions live in the prompt ("stop when you're done"), you don't have a harness, you have a wish.

**Early stopping, layered** — any one alone is insufficient:

| Layer | Trigger |
| :--- | :--- |
| Hard budget | max steps, cumulative tokens, wall-clock deadline, money |
| No-progress | hash of `(tool, args)` repeats; state digest unchanged for K steps |
| Semantic stagnation | the plan hasn't changed across N steps, or the model re-reads the same file |
| Quality gate | a critic scores progress every N steps; below threshold → abort with partial |
| External | user cancel, upstream deadline, kill switch |

Two design requirements. **Stopping must return a structured reason and a partial deliverable**, never an exception and never `""` — "finished" and "gave up at step 34 with these 3 findings" are different products. And stopping must be **resumable**: the harness checkpoints state so a deadline stop can be continued rather than restarted.

**中文要点**：分工判据是"需要知道模型说了什么=loop，需要知道系统状态（成本/时间/权限/进度）=harness"；停止逻辑必须是确定性代码而非提示词。停止要分层（预算、无进展、语义停滞、质量门、外部取消），并返回结构化原因 + 半成品，且可断点续跑。

---

### 15. What is the relationship between MCP and A2A, and how should a multi-agent communication protocol be designed?

**Source**: https://outcomeschool.com/blog/what-is-mcp-model-context-protocol

**Anchor answer**

They are orthogonal axes, not competitors:

| | MCP | A2A |
| :--- | :--- | :--- |
| Connects | model/agent ↔ **tools & data** | agent ↔ **agent** |
| Direction | vertical (capability) | horizontal (delegation) |
| Discovery | `tools/list`, `resources/list` | agent card at a well-known URL |
| Unit of work | a call (request/response) | a **task** with a lifecycle |
| Transparency | the schema is the contract | the peer is opaque — you get results, not internals |
| Transport | stdio / Streamable HTTP, JSON-RPC | HTTP + SSE / push notifications |

An agent uses MCP to *do things* and A2A to *ask another agent to do things*. Neither replaces the other.

**Designing an agent-to-agent protocol**, the non-negotiables:

1. **Identity and delegated authority.** The message must say who is acting and *on whose behalf*. Without delegation semantics you either over-grant (peer acts as superuser) or can't enforce anything.
2. **A real task state machine**: `submitted → working → input-required → completed | failed | canceled`. `input-required` is the state everyone forgets, and it's what makes long-running tasks possible — the peer must be able to say "I need the order id".
3. **Long-running by default**: async submission, streaming updates, resumable via task id, and idempotency keys on submission.
4. **Structured payload parts** (text / file / structured JSON), not free-form chat.
5. **Error taxonomy** distinct from transport errors — a failed task is a successful message.
6. **Observability**: propagate a trace id across the hop, or you cannot debug a chain of three agents.

**中文要点**：MCP 是纵向的"agent→工具/数据"，A2A 是横向的"agent→agent"，两者互补。协议设计要点是委托身份（代表谁）、带 input-required 的任务状态机、异步可恢复 + 幂等、结构化消息体、独立于传输层的错误分类、以及跨跳 trace id。

---

### 16. The model returns a function call with malformed arguments. What do you do?

**Source**: https://outcomeschool.com/blog/how-does-function-calling-work-in-llms

**Anchor answer**

Never execute it, and never silently "fix" it. Order of operations:

1. **Validate against the schema.** Pydantic/JSON-schema validation returns a precise error path (`items[2].qty: expected int, got "two"`). This is the only step that is always correct.
2. **Deterministic repair of *syntax*, not semantics**: strip markdown fences, remove trailing commas, fix single quotes, unescape double-encoded JSON, coerce obvious types (`"5"` → `5` when the schema says integer), fill defaults, and snap enums to the nearest valid member *only if the match is unambiguous*. Log every coercion — a silent coercion is how you ship a refund to the wrong account.
3. **Feed the error back as an observation.** The tool result is `{error: "invalid_arguments", path, expected, got, schema_excerpt}`. The model usually fixes it on the second attempt. Cap at 2 attempts.
4. **Constrain generation** so it stops happening: JSON-schema-constrained decoding or a grammar, fewer required fields, flatter argument structures, enums instead of free text.
5. **Circuit-break.** After N failures on the same tool, stop retrying and either ask the user a clarifying question or hand off. Retrying the same malformed call is pure burn.

The distinction that matters in an interview: **"malformed" (unparseable / wrong type) and "valid but wrong" (a well-formed call with the wrong order id) need different defenses.** Schema validation catches the first; only preconditions and a dry-run that shows the computed impact catch the second. Confusing them is why teams ship agents that "never fail validation" and still cause incidents.

**中文要点**：先 schema 校验，只做语法级修复（去围栏、引号、类型转换）且必须记日志；把精确错误回灌给模型重试，最多两次；根治靠约束解码 + 枚举化 + 更少必填字段。关键区分：格式错误靠校验，"格式正确但语义错"只能靠前置条件 + dry-run。

---

### 17. What does multi-agent coordination buy over a single agent, and how does the lead agent pick a sub-agent dynamically?

**Source**: https://outcomeschool.com/blog/multi-agent-systems

**Anchor answer**

**What it buys** — four things, and only the first is usually true:

| Benefit | Real? |
| :--- | :--- |
| **Context isolation** — each worker gets a clean window instead of one 200k-token transcript | yes, and it's the biggest real win |
| Parallelism — independent subtasks run concurrently | yes, if they truly are independent |
| Specialization — different tools/prompts/models per role | yes, but a prompt switch often suffices |
| Fault isolation — one worker failing doesn't kill the run | marginal |

**What it costs**: coordination overhead, duplicated retrieval, hard-to-reproduce non-determinism, a 2–5× token multiplier. Be honest: for most enterprise tasks **one agent with good tools beats a fleet**.

**Dynamic selection.** The lead should not hardcode routing. Practical recipe:

1. **Expose sub-agents as a tool.** One `delegate(agent_name, task_brief)` call, with descriptions that state *when not to use it* — the model selects exactly like it selects any other tool.
2. **Cap the registry.** 5–10 visible sub-agents; past that, selection accuracy degrades and you should route to a *category* first.
3. **Filter by capability before the model sees them.** Required tools, permissions, and cost tier prune the candidate list deterministically — never offer a sub-agent that cannot legally do the job.
4. **Pass a task brief, not the transcript.** `{goal, constraints, inputs, expected output shape, what NOT to do}`. Dumping the full history is the #1 cause of context bloat and drift.
5. **Require a structured return**: result, evidence, confidence, and *what I could not do*. That last field is what lets the lead re-plan instead of silently accepting a partial answer.
6. **Set hard depth and call budgets** — A→B→C→A recursion is the classic production incident.

**中文要点**：多 agent 真正买到的是**上下文隔离**（其次并行），代价是 2~5 倍 token 和不可复现性，很多场景单 agent 更划算。动态选人：把子 agent 暴露成带"何时别用"描述的工具、按能力/权限先确定性过滤、只传任务简报不传全部历史、强制结构化回报含"我做不到什么"、加深度和调用预算。

---

### 18. What dimensions does an agent evaluation need, and how do you measure planning ability versus hallucination rate?

**Source**: https://outcomeschool.com/blog/ai-agent-evaluation

**Anchor answer**

An agent eval is a matrix, and end-to-end success alone tells you nothing actionable:

| Dimension | Measured by |
| :--- | :--- |
| **Task success** | end-state assertion (did the DB row change correctly), not "did the answer look right" |
| **Tool correctness** | tool selection accuracy, argument accuracy, call count vs. minimum needed, duplicate calls |
| **Trajectory efficiency** | steps, tokens, wall clock, cost per success |
| **Robustness** | injected tool errors, timeouts, 429s, ambiguous requests, distractor tools |
| **Grounding** | unsupported-claim rate, citation resolvability |
| **Safety** | forbidden-action attempts, approval bypass, PII leakage |
| **Human intervention** | escalation rate, and *why* |

Run each case ≥3 times with a fixed seed — agents are stochastic and a single run's success rate is noise.

**Planning**, two complementary measures: (1) a held-out set with *checkable end states* where the plan must be right for the end state to be reachable — this is objective; (2) rubric grading of the plan itself by a judge with a reference plan, scored on ordering, missing steps, and unnecessary steps. Add a **counterfactual probe**: swap in a distractor tool or a slightly different constraint and see whether the plan adapts. A plan that doesn't change when the world does wasn't a plan.

**Hallucination rate**, measure at the claim level, deterministically where possible: extract atomic factual claims from the trajectory, then classify each as `supported by a tool result`, `contradicted by a tool result`, or `no evidence`. Report the last two separately — "made something up" and "said something true it couldn't justify" have different fixes (grounding vs. citation discipline). Numbers, dates and entity ids should be checked by exact match against the trace, not by an LLM judge.

**中文要点**：评估矩阵七维：任务终态、工具正确性、轨迹效率、鲁棒性、grounding、安全、人工介入率；每个 case 跑 ≥3 次。规划用"可校验终态 + 计划评分 + 反事实探针"，幻觉按 claim 级做"有据/矛盾/无据"三分类，数字和 ID 用精确匹配而非 LLM 判官。

---

### 19. Design a code review agent that only comments and never edits.

**Anchor answer**

The requirement "never edits" is a **capability** constraint, not a prompt instruction. Enforce it structurally.

**Target constraints**: must run on every PR within minutes; must not modify the repo; false-positive rate is the binding constraint (a noisy reviewer gets muted within a week); comments must be actionable and line-anchored.

| Layer | Design |
| :--- | :--- |
| Trigger | PR opened/synchronised webhook; skip drafts and `[skip-review]` |
| Sandbox | container with repo mounted **read-only**, no write token, egress allowlist; the agent's toolset is `read_file`, `grep`, `list_dir`, `git_diff` only |
| Context builder | the diff + ±30 lines of surrounding context + resolved call sites of changed symbols; hard token cap per file |
| Passes | (1) deterministic linters/type-checker first, (2) LLM for design/logic/security, (3) dedupe + rank |
| Output | structured JSON: `{file, line, severity, category, message, evidence, suggestion}` |
| Delivery | posted by a separate service account via a bot token that can only create review comments |

Three details that decide whether it's useful:

1. **Review the diff, not the file.** Models love commenting on pre-existing code. Filter to changed lines and require the anchor to be an added line.
2. **Require evidence.** Every comment cites the lines it is about; a comment that can't quote its anchor is dropped. This alone removes a large class of hallucinated findings.
3. **Be idempotent and quiet.** Hash `(file, line, message)` and skip comments already posted; suppress comments the author resolved. Precision over recall, always: five real findings beat thirty.

Also: never let it claim "tests pass" — it didn't run them, and it has no write access to find out.

**中文要点**：只读是**权限**约束不是提示词：只读挂载 + 只读 token + 工具集里没有写操作。关键三招：只审 diff 的新增行（模型天生爱评论存量代码）、每条评论必须引用锚点否则丢弃、按 hash 去重且尊重已解决。精度优先于召回。

---

### 20. Design an e-commerce agent system with HITL, risk tiers, circuit breakers, sandboxing and audit.

**Anchor answer**

The premise: an e-commerce agent touches money, inventory and customers. Design it as a **transaction system with an LLM planner**, not as a chatbot with tools.

| Component | Responsibility |
| :--- | :--- |
| Gateway | authn, per-user/per-tenant quota, session binding |
| Planner | intent → plan; **constrained to registered actions**, cannot invent one |
| **Policy engine** | maps each action to a risk tier and decides auto / approve / forbid — deterministic, outside the model |
| Executor | typed RPC only (no raw SQL, no shell), per-action timeout, idempotency key |
| Verifier | read-back after every write, precondition checks before |
| Audit sink | append-only log: intent, params, evidence, tier, approver, result, trace id |

**Risk tiers**: T0 read-only (auto) · T1 reversible low-value, e.g. apply a coupon ≤ ¥10 (auto with per-day cap) · T2 money/shipment/address changes (approval or second-check) · T3 irreversible or high-value: refunds above threshold, account deletion, price changes at scale (**forbidden to the agent**).

**HITL**: interrupt + persisted checkpoint (not a blocking HTTP call); the approval UI shows the *computed impact* ("this refunds ¥1,280 across 3 orders"), the evidence, and lets the approver edit parameters; timeout → **auto-reject**.

**Circuit breakers**: per-tool error-rate breaker, per-session action budget, per-user spend cap, and a global kill switch that freezes writes while keeping reads alive.

**Sandboxing**: writes go to a staging layer reconciled against the system of record; every T2+ action must support a dry-run that returns the computed diff.

**Audit** must answer "why", not just "what": which policy, which evidence, which approval.

**中文要点**：当成"带 LLM 规划器的事务系统"设计：策略引擎在模型之外做风险分级（T0 读/T1 低额可逆/T2 需审批/T3 禁止），执行只走类型化 RPC + 幂等键，写后回读校验，熔断按工具/会话/金额分层加全局急停，审计必须记录"为什么"。

---

### 21. Design an e-commerce support agent: how does it handle vague user requests, what does it say when a product cannot be retrieved, and what is the core success metric?

**Anchor answer**

**Vague requests.** Never guess; **resolve ambiguity with the cheapest available signal first**, then ask exactly one question.

1. Use ambient context before asking: logged-in user → recent orders, open tickets, the page they came from. "You mean the headphones from order 2024…?" beats "please provide your order number".
2. If still ambiguous, ask **one** question with 3 concrete options, not an open "please clarify".
3. Bound it: after 2 clarifying turns with no progress, escalate to a human with the transcript attached. An agent that loops on clarifications is worse than no agent.

**Product not retrieved.** State what was searched, never fabricate:

> "I searched our catalogue for 'noise-cancelling earbuds under ¥300' in Audio and found nothing in stock. Closest matches are [A ¥329] and [B ¥279]. Want me to check either, or notify you when [X] is back?"

Three rules behind that: (a) disclose the query that failed — it lets the user correct you; (b) offer *nearest* alternatives from a relaxed query; (c) if the answer would require parametric memory for price/stock/specs, **refuse and hand off** — that is the guardrail that prevents the confident-hallucination failure.

**Core success metric.** Not deflection. Deflection with "I'll connect you to an agent" is a fake win.

The north star is **resolution rate at acceptable CSAT**, with hard guardrails:

| Metric | Role |
| :--- | :--- |
| Resolved without human, per intent | primary |
| Wrong-answer / post-answer escalation rate in 7 days | guardrail — blocks shipping |
| Repeat contact rate | guardrail |
| CSAT on bot-handled vs human-handled | parity check |
| Cost per contact, AHT | efficiency |

A single number is gameable; the pair is not.

**中文要点**：模糊请求先用环境上下文（登录用户的订单）再只问一个带 3 个选项的问题，2 轮不进展就转人工；查不到就明说查了什么、给最近替代、涉及价格库存一律不猜并转人工。核心指标是"可接受 CSAT 下的自助解决率"，配错答率/7 日重复联系率做硬护栏，别只看 deflect rate。

---

### 22. Design the full pipeline from document upload to retrievable, and say where incremental update time goes.

**Anchor answer**

| # | Stage | Notes |
| :-: | :--- | :--- |
| 1 | Upload (presigned URL → object store) | never stream through the app server |
| 2 | Scan + format detect | virus scan, MIME sniff, reject encrypted PDFs |
| 3 | Parse | layout-aware extractor; OCR fallback for scanned pages; tables kept as tables |
| 4 | Normalise | strip boilerplate/headers, fix hyphenation, unify units |
| 5 | Chunk | structure-aware 300–800 tokens, overlap, heading breadcrumb, **content hash per chunk** |
| 6 | Enrich | summary, keywords, ACL from source system |
| 7 | Embed | batched, with a model version stamped on every vector |
| 8 | Index | sparse + dense, same ids; write ACL field |
| 9 | Publish | atomic: flip a `doc_version` pointer so retrieval never sees a half-built doc |
| 10 | Notify | emit `doc.ready` with version |

**Where incremental update time actually goes** (ranked, from experience):

1. **Parse/OCR** — a scanned 200-page PDF is 10–100× a born-digital one. This dominates, and it's why incremental updates must be async with progress, not a blocking API.
2. **Embedding** — throughput caps and provider rate limits; batching matters more than the model choice.
3. **Index refresh and segment merge** — a 1s refresh interval will destroy bulk indexing; set `refresh_interval: -1` during the load and refresh once.
4. **Re-embedding work you didn't need to do** — with per-chunk content hashes you re-embed only changed chunks; without them, a one-line edit re-embeds a 50-page document.
5. **Cache invalidation across retrieval replicas** — often the sneakiest: the index is fresh and answers still cite the old text.

Design rules: idempotent per `(doc_id, version)`, a dead-letter queue for parse failures, and a visible per-document status so the UI can say "indexing" instead of lying.

**中文要点**：上传→扫描→解析(OCR 兜底)→规范化→结构分块(带 chunk 内容哈希)→增强→嵌入(标模型版本)→双索引→按版本指针原子发布。增量耗时排序：OCR 解析 > 嵌入吞吐 > 索引 refresh/merge > 不必要的重嵌入 > 各副本缓存失效。

---

### 23. You build a feature with AI coding and then review it by hand. How do you prove the change is correct and has not broken existing behaviour?

**Anchor answer**

"You read it carefully" is not evidence. Produce artifacts:

| Evidence | How |
| :--- | :--- |
| **A test that fails before and passes after** | write the test first, from the requirement, not from the implementation |
| **Mutation check** | delete a branch / flip a condition in the new code and confirm the test fails. A test that passes on both is worthless |
| **Full existing suite**, not a subset | plus type check and lint; AI edits routinely break call sites it never saw |
| **API existence** | every imported symbol, function signature and config key must actually exist — AI hallucinates APIs confidently (`hasattr`, run it) |
| **Behavioural diff for refactors** | golden input/output snapshots, property tests, or a side-by-side run of old and new |
| **Line-by-line diff review** | read the whole diff, never the summary; check specifically for *deletions* of validation, error handling, and permission checks |

The review checklist that catches the most AI-generated bugs, in my experience:

1. Silent fallbacks — `except Exception: pass`, defaulting to `[]` on an error, `if not x: return True`.
2. Over-broad exception handling replacing a specific one.
3. Changed defaults (a function quietly gains `timeout=None`).
4. Off-by-one in a rewritten loop, or a boundary condition that was handled before and isn't now.
5. Removed guard clauses because they "looked redundant".

Then verify end-to-end on a realistic input and diff the output: run the CLI/service on a golden fixture and compare. Where the change is unobservable that way, add the integration test — which also means the next AI edit has a safety net.

**中文要点**：证据不是"我看过了"：先写会失败的测试 → 变异检验（删分支测试必须挂）→ 跑全量测试和类型检查 → 核对被引用的 API 真实存在 → 重构用快照/属性测试对比 → 逐行读 diff 并专门找被删掉的校验与异常处理。

---

### 24. What are the time complexities of common Python list and dict operations, and why is insert at the head O(n)?

**Anchor answer**

| Operation | list | dict |
| :--- | :--- | :--- |
| `a[i]` / `d[k]` | O(1) | O(1) avg, O(n) worst |
| `append` / `d[k]=v` | O(1) amortized | O(1) amortized |
| `insert(0, x)` / `pop(0)` | **O(n)** | — |
| `insert(i, x)` | O(n − i) | — |
| `x in a` / `k in d` | **O(n)** | O(1) avg |
| `remove` / `del d[k]` | O(n) | O(1) avg |
| slice `a[i:j]` | O(j − i) | — |
| `sort()` | O(n log n) | — |
| iteration | O(n) | O(n) |

**Why insert at the head is O(n).** A CPython `list` is not a linked list — it is a `PyObject**` array (`ob_item`) of pointers sitting *contiguous* in memory, plus `ob_size` and `ob_allocated`. Inserting at index 0 means every existing element must shift one slot to the right: `memmove(n · sizeof(ptr))`. There is no way around it without changing the data structure. Appending is O(1) amortized because CPython over-allocates — `list_resize` grows by roughly `size + size>>3 + 6` (about 12.5% headroom), so reallocation is rare.

Two consequences worth knowing:

- Use `collections.deque` for head/tail work — it's a block-linked structure with O(1) on both ends.
- `list` slicing, `a[::-1]`, and `a.copy()` are O(n) memory operations that hide in hot loops. `a.insert(0, x)` in a loop is the classic accidental O(n²).

**dict**: open addressing with a compact layout (an `indices` array of 32-bit slots + a dense `entries` array), resized when the load factor reaches ~2/3. Lookups hash the key, probe, compare — O(1) average. The O(n) worst case needs deliberately colliding hashes; in practice the real hazard is *unhashable-key exceptions* and memory, not speed.

**中文要点**：list 是连续指针数组，头部插入要 memmove 全部元素所以 O(n)；append 是摊还 O(1)（CPython 超额分配约 12.5%）；`x in list` 是 O(n)，循环里 `insert(0)` 就是意外的 O(n²)，该用 deque。dict 是紧凑开放寻址，平均 O(1)，最坏 O(n) 需要刻意构造哈希冲突。

---

### 25. When does a MySQL composite index stop being used? What is the leftmost-prefix rule, and what do you look at in EXPLAIN?

**Anchor answer**

`INDEX(a, b, c)` is physically sorted by `a`, then `b`, then `c`. Only a **prefix** of that ordering can be used for a seek, which is the leftmost-prefix rule:

- ✅ `a` · `a, b` · `a, b, c` · `a` + range on `b`
- ❌ `b` · `c` · `b, c`

**When it stops being used:**

| Cause | Example |
| :--- | :--- |
| Skipping the leftmost column | `WHERE b = 1` |
| Range on a middle column | `a = 1 AND b > 2 AND c = 3` — `c` no longer seeks (8.0's index skip scan may partially rescue low-cardinality `a`) |
| Function or cast on the column | `WHERE DATE(created_at) = '2024-01-01'`, `WHERE varchar_col = 123` (implicit cast) |
| Leading wildcard | `LIKE '%foo'` |
| `OR` across differently-indexed columns | forces a union or a scan |
| Collation / charset mismatch in a join | index unusable on the joined column |
| Low selectivity | optimizer decides a full scan is cheaper — correctly |
| `ORDER BY` a column not covered | filesort instead |

**EXPLAIN columns that matter**: `type` (`const` > `eq_ref` > `ref` > `range` > `index` > `ALL`), `key` (which index was chosen), **`key_len`** (how many prefix columns are actually used — the fastest way to confirm the leftmost rule), `rows` (estimate), `filtered`, and `Extra`: `Using index` = covering (no table lookup), `Using index condition` = index-condition pushdown, `Using filesort` / `Using temporary` = the usual suspects.

Read `rows × filtered` against the rows actually returned; a ratio in the hundreds means the index isn't selective enough. On 8.0, `EXPLAIN ANALYZE` gives real timings instead of estimates — use it before adding an index.

**中文要点**：复合索引按列顺序排序，只能用最左前缀；中间列一旦是范围条件，后面的列就失效。别踩函数/隐式转换/前导通配符。EXPLAIN 先看 type、key、**key_len**（判断用了几列）、rows×filtered，再找 filesort/temporary；8.0 直接 EXPLAIN ANALYZE 看真实耗时。

---

### 26. Alibaba open-sources Qwen while running a commercial cloud business. What do you make of that strategy?

**Anchor answer**

It is a **distribution play with a defensible monetisation layer underneath**, not philanthropy. The chain: open weights → downloads and fine-tunes → developers default to Qwen-shaped tooling → they need GPUs, serving, and a managed endpoint → Alibaba Cloud bills for it.

| Mechanism | What it captures |
| :--- | :--- |
| Free weights commoditise the model layer | value shifts from "the model" to cloud + inference + integration, where Alibaba already sells |
| Mindshare → default SDK/tokenizer/chat template | switching cost moves from weights to the ecosystem (PEFT configs, quantised variants, Chinese-language tooling) |
| Ecosystem labour | community derivatives, benchmarks, bug reports, and fine-tunes at zero marginal cost |
| Recruiting / brand | "we ship the reference open model" is a hiring argument in a tight market |
| Hardware neutrality | weights are useless without GPUs; open weights increase, not decrease, demand for serving capacity |

**What it costs**: it gives away API margin on exactly the customers most price-sensitive, and it lets third-party hosts (Together, Fireworks, SiliconFlow) arbitrage your own release. It also doesn't win enterprise deals by itself — those need indemnity, SLAs and compliance.

**Why it's coherent anyway**: the licence is chosen per size (Apache-2.0 for the widely-used small/medium models, more restrictive where it matters), and the cadence keeps the community anchored while the commercial tier stays ahead. Open-weight release is a **marketing and standards budget whose return shows up in cloud revenue, not model revenue**.

Where I'd push back: if the open model ever matched the commercial one the strategy would cannibalise itself — so the real question is whether the gap is held by capability or by *serving economics* (context length, rate limits, tooling).

**中文要点**：开源权重把模型层商品化，价值转移到云、推理和集成——而这正是阿里云在卖的；同时锁定 tokenizer/工具链生态、白拿社区微调和算力需求。代价是让出 API 毛利、被第三方托管套利，且开源赢不了要 SLA 和合规的企业单。

---

### 27. Walk me through a project you took from a vague requirement to delivery, and the reasoning behind your decisions.

**Anchor answer**

The method I actually use: **ambiguity is resolved by artifacts, not meetings.**

1. **Restate.** One page: the problem in the user's words, the measurable success metric, and an explicit **non-goals** list. Getting the non-goals signed off is what prevents scope creep later.
2. **Define the metric before the design.** For an internal agent: "resolves X% of this ticket type without human, with wrong-answer rate under Y%". Without a number you cannot choose between a 70%-accurate demo and a 92%-accurate product.
3. **Buy evidence cheaply.** A 200-case offline eval set built from *real historical* inputs, run against a prompt-only baseline, tells you in two days whether the task is feasible. A demo tells you nothing.
4. **Cut to one workflow, end to end.** Ship the narrowest slice that a real user can use: one intent, real tools, real permissions, behind a flag with human review of every output.
5. **Iterate on a failure taxonomy**, not on vibes: classify every bad output as retrieval miss / tool error / reasoning error / format error / policy refusal, and fix the biggest bucket.
6. **Widen the slice only when the error budget allows it.**

Decisions I'd highlight: what I said no to (the chat UI before the eval existed), the tradeoff I accepted (lower automation, higher precision early — a wrong answer costs more trust than a slow one), and what went wrong: **offline eval overestimated quality** because the set was cleaner than production traffic; the fix was sampling real traffic into the eval set weekly.

Close with ops: prompt/model versioning, a rollback path, drift monitoring on the input distribution.

**中文要点**：用产出物消除模糊：一页纸问题陈述 + 非目标 + 上线前定好可测指标；先用 200 条真实历史 case 做离线基线判断可行性；切一个最窄端到端流程带 flag 上线；按失败分类（检索/工具/推理/格式/拒答）修最大那一桶。最常翻车的是离线评测高估效果——要持续把真实流量采样进评测集。

---

## 附：这 27 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 训练/算法底子 | 1（MoE 手写）、3（融合与 warp specialization）、4（prefill/decode）、6（GRPO 公式）、8（credit assignment） |
| 集群与稳定性 | 2（NCCL 定位）、3（融合的性能反直觉） |
| 云平台/运行时 | 12（ACL 与 PII）、14（loop vs harness）、15（MCP/A2A）、22（文档管线增量耗时） |
| 淘系业务 Agent | 20（风险分级 + HITL + 审计）、21（模糊请求与成功指标）、19（只评论不改代码） |
| RAG 工程 | 11（全链路）、12（权限放哪）、13（压缩与 harness） |
| 工程基本功 | 23（AI 代码怎么证明对）、24（list/dict 复杂度）、25（复合索引与 EXPLAIN） |
| 价值观/判断力 | 26（Qwen 开源策略）、27（模糊需求到交付） |
