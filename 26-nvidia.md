# 26 · NVIDIA

> 13 题。出处：`README.en.md` § NVIDIA。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Here's a CUDA kernel that's 10x slower than expected. Without running it, what are the usual suspects, and how do you confirm each?

**Source**: https://outcomeschool.com/blog/how-does-a-gpu-work-for-deep-learning

**Anchor answer**

A 10x gap is almost never "the GPU is slow" — it is one of five structural mistakes. Read the kernel top-down and check in this order:

| Suspect | Why 10x | How to confirm without running |
| :--- | :--- | :--- |
| **Uncoalesced global loads** | one 32B sector per lane instead of per warp → up to 8–32x wasted bandwidth | look at the index expression: is the innermost loop over `threadIdx.x` contiguous? A thread's stride must be 1 along the fastest-varying dim |
| **Occupancy capped by registers/shared mem** | too few warps to hide DRAM latency | count registers per thread by hand (`__launch_bounds__` absent, big unrolled loops, many live temporaries); shared mem per block vs 48/100 KB limit |
| **Low arithmetic intensity** | re-reading global instead of staging to shared/registers | count bytes moved per FLOP; if < ~10, you are bandwidth-bound by construction |
| **Divergence + serialized branches** | 32 lanes execute both sides of `if` | inspect branches that depend on data (e.g. `if (val > 0)` on random input) |
| **Launch configuration** | grid smaller than SM count, block size not a warp multiple, tail waves | multiply `gridDim * blockDim`; compare with SM count × max blocks/SM |

Then the two silent killers: **`double` or `expf` accidentally promoted**, and **timing the H2D copy / `cudaDeviceSynchronize`** inside the measurement.

Confirmation without a profiler: compile with `-Xptxas -v` (registers, spill), and diff against a reference kernel that only changes one variable.

**中文要点**：10x 吞吐差几乎只有五类原因——非合并访存、寄存器/共享内存压死 occupancy、算术强度太低、warp 内分支发散、launch 配置不满 SM；先看索引表达式和 `-Xptxas -v`，别急着开 profiler。

---

### 2. Implement the block manager for a paged KV cache: allocate, append, free, and copy-on-write prefix sharing.

**Source**: https://outcomeschool.com/blog/paged-attention-in-llms

**Anchor answer**

**Reference implementation**

```python
class BlockManager:
    """Paged KV: block_size tokens per physical block; table = logical -> physical."""

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.free = list(range(num_blocks))[::-1]   # LIFO -> O(1) alloc, hot blocks stay warm
        self.ref = [0] * num_blocks                 # ref == 1  => privately writable

    def _alloc(self) -> int:
        if not self.free:
            raise MemoryError("KV pool exhausted")   # caller preempts / swaps
        b = self.free.pop()
        self.ref[b] = 1
        return b

    def _release(self, b: int) -> None:
        self.ref[b] -= 1
        if self.ref[b] == 0:
            self.free.append(b)

    def append(self, table: list, filled: int, n: int) -> int:
        """Append n tokens after `filled` existing ones; returns new token count."""
        end = filled + n
        while filled < end:
            idx = filled // self.block_size
            if idx == len(table):                   # need a new block
                table.append(self._alloc())
            elif self.ref[table[idx]] > 1:          # shared & partially full -> COW
                old = table[idx]
                table[idx] = self._alloc()
                self._copy_block(old, table[idx])   # device D2D copy of B*2*H*D elems
                self._release(old)
            filled = min((idx + 1) * self.block_size, end)
        return end

    def fork(self, table: list) -> list:
        """Prefix sharing: child gets its own table pointing at the same blocks."""
        for b in table:
            self.ref[b] += 1
        return list(table)

    def free_seq(self, table: list) -> None:
        for b in table:
            self._release(b)
```

**Key points**

- **Copy-on-write**: `ref == 1` means "privately writable". A block with `ref > 1` must be cloned before you write into it — and only the **partially-filled tail block** of a shared sequence can ever be a COW target, because the full ones are behind the write cursor.
- Fork must bump every block's ref; free must decrement and only recycle at zero. Getting this wrong either leaks the pool or hands a live block to a second sequence (silent KV corruption, not a crash).
- Free list as a stack → O(1) alloc/release, and it keeps hot blocks cache-warm.
- Failure mode to name: **fragmentation**. Fixed block size trades internal waste (avg half a block per sequence) against external waste (`cannot allocate` while total free is large).

**中文要点**：块管理器三件事——空闲栈 O(1) 分配、引用计数支持前缀共享、写前复制只发生在"部分填充的最后一块"；真正难的是碎片，不是链表。

---

### 3. A model runs fine in FP32 but produces garbage after conversion to FP16. Debug it.

**Anchor answer**

Split the causes into **range** vs **precision** — they look different in the logs and have different fixes.

| Class | Symptom | Typical location |
| :--- | :--- | :--- |
| **Overflow** | `inf`/`nan` appearing early, loss → NaN | attention logits before softmax, `exp`, LayerNorm variance, large embedding values, loss sum |
| **Underflow** | loss decreases then stalls; many exact zeros | gradients < 6e-8, small attention weights, `1/sqrt(d)` scalings |
| **Accumulation in FP16** | slow drift, dead layers, wrong results only on long reductions | GEMM accumulation, softmax denominator, `mean` over 1024+ elements |
| **Stale scale / wrong cast point** | garbage from token 1 | checkpoint converted with wrong scale, `.half()` applied to buffers that must stay FP32 |

Diagnosis order that actually works:

1. Run the FP32 model and the FP16 model on the **same batch**, compare per-layer max-abs-diff. The first diverging layer localizes it in one pass.
2. Test **bfloat16**. If bf16 is clean and fp16 is not, it is range (bf16 has fp32 exponent) — not precision.
3. Check that **reductions accumulate in FP32**: `layer_norm`, `softmax`, `cross_entropy`, and GEMM with `accumulate=fp32`.
4. Confirm **loss scaling** / autocast is actually on for training, and that `eps` in norms is not 1e-5 (underflows relative to fp16 mantissa).
5. Keep embeddings, first conv, and the final logits head in FP32 — cheap, removes most of the pain.

**中文要点**：先把问题分成 range 与 precision：出现 inf/nan 是溢出，loss 停滞是下溢，长归约误差是累加精度；用 bf16 对照一步就能区分，再用逐层 max-abs-diff 定位首个发散层。

---

### 4. Explain strategies to combat overfitting in tree-based classification models.

**Anchor answer**

Trees overfit differently from neural nets: they do not diverge, they **memorize the tail**. So the fixes are structural first, regularization second, data last.

| Lever | Mechanism | Practical note |
| :--- | :--- | :--- |
| `max_depth` / `max_leaves` | caps interaction order | depth 6–8 covers most tabular problems; leaf-wise growth (LightGBM) needs `num_leaves` more than depth |
| `min_child_weight` / `min_samples_leaf` | stops splits that isolate 1–3 noisy rows | the single most effective knob on noisy labels |
| `subsample` + `colsample_bytree` | decorrelates trees (bagging) | 0.7–0.9 |
| L1/L2 (`reg_lambda`) | shrinks leaf values | helps calibration as much as accuracy |
| **Early stopping on a time-based validation set** | stops at the point of validation degradation | must be *time* split, not random — random split leaks the future |
| Feature pruning | removes leaky IDs / high-cardinality keys | a user_id feature is a memorization license |

Data-side: deduplicate near-identical rows across folds, fix label noise (a tree happily splits on a mislabeled cluster), and add monotonic constraints where domain knowledge exists — they cut variance with no accuracy loss.

Things that do **not** help: adding more trees with a small learning rate and no early stopping is usually neutral, not harmful; `min_split_gain=0` is harmful because it keeps zero-gain splits that only add noise.

**中文要点**：树模型的过拟合是"记住长尾"，先控结构（叶子数/min_child_weight），再正则，最后修数据；验证集必须按时间切，随机切会漏未来信息——很多"过拟合"其实是泄漏。

---

### 5. Summarize the differences and benefits of the Adam optimizer compared with other methods for neural-network image classification.

**Anchor answer**

Adam keeps two per-parameter states and divides by their square roots:

- `m = β1·m + (1-β1)·g` — momentum, first moment
- `v = β2·v + (1-β2)·g²` — second moment, per-parameter step size

So the update is `θ -= lr · m̂ / (√v̂ + ε)`: **per-parameter adaptive scale + momentum**, with bias correction for the first steps.

| Optimizer | Update | Where it wins | Where it loses |
| :--- | :--- | :--- | :--- |
| SGD + momentum | `lr·m` | CNNs with long schedules; often +0.5–1 pt test accuracy at equal tuning budget | extremely lr-sensitive; needs per-layer tuning |
| Adagrad | accumulate all `g²` | sparse features (NLP embeddings) | lr decays to zero, dies on long training |
| RMSProp | EMA of `g²` | RNNs, non-stationary objectives | no momentum |
| **Adam / AdamW** | as above; AdamW decouples weight decay | fast early convergence, robust to lr and gradient scale, works with heterogeneous layers (ViT, transformers, mixed precision) | 2 extra states = 3x parameter memory; can generalize slightly worse than well-tuned SGD |

The image-classification nuance interviewers want: **Adam converges faster but SGD often generalizes better** on ResNet-style CNNs — attributed to flatter minima. The modern resolution is not "pick one": use AdamW with `lr=1e-4..3e-4`, cosine schedule + warmup, decoupled weight decay ~0.05, and only fall back to SGD if you have the tuning budget. Also note Adam's `ε=1e-8` is meaningless in FP16 — use FP32 master weights.

**中文要点**：Adam = 动量 + 逐参数二阶缩放 + 偏差修正；好处是对 lr 和梯度尺度鲁棒、前期收敛快，代价是 3 倍参数内存且泛化常略逊于调好的 SGD；现代做法是 AdamW + cosine + warmup。

---

### 6. A network confuses pugs and pit bulls and some training labels are wrong. How do you modify the model and the data?

**Anchor answer**

Two bugs wearing one costume: **label noise** and **shortcut learning**. Fix both, and measure them separately.

**Data side**

1. Find the wrong labels instead of eyeballing: train k-fold, take out-of-fold predicted probability for the given label, rank ascending — the bottom tail is enriched in errors (confident learning / `cleanlab`). Human-review only that tail.
2. Build a **clean, human-verified test set** before anything else. Without it you cannot tell whether a change helped or just fit the noise.
3. Counter the shortcut: pugs and pit bulls differ in **snout, ear shape, body proportion**, but datasets are full of correlated context (leash, couch, owner). Test for it by cropping the animal, or by training on background-only and checking accuracy stays near chance.

**Model side**

| Change | Why |
| :--- | :--- |
| Higher input resolution / crop-based training | these classes are separable by fine geometry; 224px destroys the snout |
| Strong geometric augmentation, mild color jitter | the cue is shape, not color |
| Label smoothing or a **noise-robust loss** (symmetric CE / GCE) | stops fitting the residual mislabels |
| Class-balanced sampling + focal loss | if one breed dominates |
| Add an "ambiguous/other" class | the model should be allowed to abstain |
| Ensemble + calibration | disagreement is your label-audit queue |

**中文要点**：既有错标也有捷径学习（靠牵引绳、沙发认狗）；先用 out-of-fold 概率排序定位错标、建干净测试集，再从分辨率/裁剪/几何增强入手逼模型看口鼻和体型，别只调 loss。

---

### 7. How do you evaluate a clustering model's effectiveness without pre-labelled groups?

**Anchor answer**

No labels means you evaluate **three different things**, and none of them is "accuracy":

| Question | Metric | Caveat |
| :--- | :--- | :--- |
| Are clusters compact and separated? | silhouette, Davies–Bouldin, Calinski–Harabasz | all assume convex/globular clusters — k-means wins by construction, density-based methods lose unfairly |
| Is the structure real or an artifact of the seed? | bootstrap resampling → Adjusted Rand Index between runs | low ARI means the answer is noise, regardless of silhouette |
| Is the partition *useful*? | downstream task on top of cluster id (classification of something you do have, or an A/B on business action) | the only metric that survives a product review |
| How many clusters? | elbow, gap statistic, silhouette-vs-k | never pick k from one curve; report the plateau |

Two habits that separate practitioners from textbook answers:

1. **Always inspect samples per cluster.** Aggregate stats hide the failure mode where "cluster 3" is just "rows with a missing field".
2. **Check stability under perturbation** (feature subset, 1% row drop, scaling change). A clustering that changes identity when you drop 1% of rows cannot drive a decision.

And say the honest thing: if any labelled data exists later, report ARI/NMI against it as a sanity check — internal metrics can all look great on a partition that means nothing.

**中文要点**：无标签只能评三件事——紧致分离度（silhouette 等，但偏爱凸簇）、稳定性（bootstrap 的 ARI）、下游有用性；再加上逐簇抽样人看，否则"簇 3"可能只是缺字段的行。

---

### 8. You want to serve a 70B-parameter model on a single 80 GB GPU. Walk me through whether it fits and what single-stream tokens/sec you'd expect.

**Source**: https://outcomeschool.com/blog/how-does-model-quantization-work · https://outcomeschool.com/blog/kv-cache-in-llms

**Anchor answer**

**Step 1 — weights.** `70B × bytes/param`:

| Precision | Weights | Verdict on 80 GB |
| :--- | :--- | :--- |
| FP32 | 280 GB | no |
| BF16/FP16 | 140 GB | no |
| FP8/INT8 | 70 GB | fits, ~8 GB left → almost no KV |
| INT4 (AWQ/GPTQ) | ~35–38 GB | fits comfortably, ~40 GB for KV + workspace |

**Step 2 — KV cache.** `2 × layers × kv_heads × head_dim × bytes` per token. For a Llama-3-70B shape (80 layers, 8 KV heads, 128 head_dim, FP16) that is ~327 KB/token → 40 GB ≈ **120k cached tokens**. Note GQA is what makes that number survivable; with MHA it would be 10x.

**Step 3 — decode speed.** Single stream is memory-bandwidth-bound: `tok/s ≈ BW / bytes_read_per_step`.

- H100 SXM, ~3.35 TB/s, INT4 weights → 35 GB read → **~95 tok/s theoretical**; realistic **40–60 tok/s** at 50–60% achieved bandwidth.
- INT8 → ~70 GB per step → ~25–35 tok/s realistic.
- FP16 does not fit; if you shard it you are now at ~45 tok/s minus inter-GPU sync.

**Step 4 — what breaks it.** Context length (KV grows linearly), batch >1 (weights amortize but KV and activation memory bite), and long prefill (compute-bound, different math — prefill is where the GPU actually gets busy).

**中文要点**：70B 单卡 80G 只有 INT4 舒服（35G 权重 + 40G KV，约 12 万 token）；decode 是带宽瓶颈，tok/s ≈ 带宽/每步读字节，H100 INT4 实测 40~60 tok/s，别拿理论峰值。

---

### 9. What does TensorRT / TensorRT-LLM actually do to a model to make it faster, and when will it not help?

**Source**: https://outcomeschool.com/blog/how-does-tensorrt-llm-work

**Anchor answer**

It is three layers of work, and knowing which layer gives you the speedup is the actual interview signal.

| Layer | What happens | Typical win |
| :--- | :--- | :--- |
| **Graph** | constant folding, dead-code elimination, cast elimination, **kernel fusion** (LayerNorm+residual, QKV GEMM, activation epilogues) | 1.2–1.6x, mostly from removing memory round-trips |
| **Weights** | layout transforms (interleaved QKV, pre-packed for tensor cores), **quantization** (FP8/INT8/INT4 AWQ/GPTQ), per-channel scales | 1.5–2.5x, mostly bandwidth |
| **Runtime** | CUDA Graphs (kills per-kernel launch overhead), **in-flight / continuous batching**, paged KV cache, custom FlashAttention-style kernels, plugin all-reduce | 2–10x on throughput; matters most at small batch |

**When it does not help** (be specific — this is what separates "I read the blog" from "I shipped it"):

1. **Shape coverage misses.** The engine is built against min/opt/max profiles. A request shape outside the profile falls back to a slower kernel — real regressions come from someone sending 4097 tokens against a 4096 profile.
2. **Unsupported ops** break fusion and dump you into a fallback subgraph, sometimes slower than PyTorch.
3. **You are launch-bound with a tiny model** — CUDA Graphs fix that, but if the whole step is <50 µs you are fighting Python/host overhead, and the fix is the serving loop, not the engine.
4. **Compute-bound training**, or when the wall clock is dominated by tokenization, network, or a tool call.
5. **Every rebuild is per-GPU-arch.** A T4 engine will not run on H100; portability is a real ops cost.

**中文要点**：分三层——图优化（融合/常量折叠）、权重（布局 + 量化）、运行时（CUDA Graph、continuous batching、paged KV、定制 attention）；它不能救 shape 超出 profile、不支持的算子、以及瓶颈根本不在 GPU 的场景。

---

### 10. Design the parallelism strategy for serving a 405B-parameter dense model. TP, PP, EP: what goes where and why?

**Anchor answer**

First correct the premise: **EP does not apply — the model is dense.** Expert parallelism is an MoE concept; if someone asks "where does EP go" for a dense 405B, they are testing whether you will invent an answer.

Sizing: 405B BF16 = **810 GB of weights**. On 80 GB H100s that is ≥11 GPUs for weights alone, plus KV cache, activations, and CUDA graphs.

| Dimension | Placement | Reason |
| :--- | :--- | :--- |
| **TP** | within a node, degree 8 (NVLink) | TP does two all-reduces per layer; over PCIe/IB it becomes the bottleneck. Never cross nodes with TP if you can avoid it |
| **PP** | across nodes, with interleaved/virtual stages | PP only sends activations at stage boundaries — cheap on slow links. Keep stages balanced by measured layer time, not layer count |
| **DP** | replica sets, for throughput | multiplies capacity, no model splitting |
| **CP / sequence parallel** | for long context | splits the sequence, not the weights; needed when KV per request exceeds per-GPU budget |

For **inference** on one 8×H100 node (640 GB total), BF16 does not fit (810 GB of weights). FP8 gets weights to ~405 GB, leaving ~200 GB for KV — feasible but tight, which is why most deployments run PP=2 × TP=8 across two nodes. For **training**, add ZeRO/FSDP sharding of optimizer states (Adam FP32 ≈ 16 bytes/param ≈ 6.5 TB) — that dominates the cluster plan more than the model does.

Failure modes to name: PP bubble idle, TP all-reduce latency at small per-GPU batch, stage imbalance causing a straggler, and KV cache being the thing that actually caps concurrency.

**中文要点**：先纠正前提——dense 模型没有 EP；TP 只在 NVLink 节点内（8 卡），PP 跨节点（只传激活，便宜），DP 扩吞吐，长上下文加 CP；405B BF16 权重 810GB，真正的约束是 KV 和优化器状态。

---

### 11. Design a podcast search engine with transcript indexing. / Design a recommendation algorithm for type-ahead search.

**Anchor answer**

**Part A — podcast search over transcripts**

| Stage | Design | Failure it prevents |
| :--- | :--- | :--- |
| Ingest | ASR → **word-level timestamps**, speaker turns, chapter detection | you must be able to deep-link to `episode?t=1234` |
| Segment | 20–40 s windows with 5 s overlap, keep the sentence boundary | too long → diluted embedding; too short → no context |
| Index | hybrid: BM25 over segments **+** dense ANN; metadata (show, host, date, topic) | pure embeddings miss exact names/numbers, pure BM25 misses paraphrase |
| Query | ASR-noise-robust normalization, entity extraction, synonym expansion | spoken queries are messy |
| Rank | fuse (RRF or weighted), then a cross-encoder rerank of top 50 | cheap first stage, expensive second |
| Serve | group hits by episode, return one timestamped snippet per episode | ten snippets from one episode is a bad result page |

**Part B — type-ahead**

Two-trie design: a **prefix trie of historical queries** (with per-prefix top-k precomputed, not scanned at query time) plus a **semantic fallback** for cold/rare prefixes. Score = `log(freq) + recency decay + personalization bonus − unsafe-term penalty`, and return in <50 ms. Keys: precompute top-k per prefix offline, truncate at depth ~4, keep the trie updated on a sliding window so trending queries surface, and diversify the last slot with an exploratory suggestion.

**中文要点**：播客搜索必须带时间戳深链，切 20~40 秒重叠片段、BM25+向量混合召回再 cross-encoder 精排、按 episode 聚合；type-ahead 用前缀 trie + 预计算 top-k + 语义兜底，50ms 内返回。

---

### 12. A customer's LLM chatbot on 8 GPUs is “too slow and too expensive.” You have one week with them. What do you do?

**Source**: https://www.youtube.com/watch?v=jV2sCj4lHYk · https://outcomeschool.com/blog/llm-inference-optimization

**Anchor answer**

Week plan: **measure, then fix in ROI order**. Never start with a kernel.

| Day | Action | Output |
| :--- | :--- | :--- |
| 1 | Instrument: TTFT, TPOT, tokens/s, batch size over time, GPU util + memory, **cost per 1k tokens**, prompt/completion token split, cache hit rate | one dashboard, one number to move |
| 2 | Find the binding constraint: GPU idle → scheduling; memory full → KV; compute pegged at batch 1 → bandwidth; huge TTFT → prefill/prompt length | the diagnosis |
| 3–4 | Apply the top 3 levers | measure each |

The levers, in order of typical payoff on an 8-GPU deployment:

1. **Continuous batching.** A batch-1 server is the most common root cause; saturation usually gives 5–20x throughput at the same hardware.
2. **Prefix caching / KV reuse.** Chatbots resend a 2–4k-token system prompt every turn; caching it cuts prefill cost and TTFT dramatically.
3. **Quantization** (FP8 or INT4 AWQ) — 1.5–2x throughput, 2x KV capacity.
4. **Cap output length + smaller model routing.** 30% of replies do not need the 70B model; route by intent.
5. **TP sizing** — make sure TP=8 is actually faster than TP=4×2 replicas for their traffic (it often is not at low load).

The week's real deliverable is not a config diff — it is a **benchmark harness and a cost dashboard** they can run after you leave.

**中文要点**：一周流程是"先量化再优化"：TTFT/TPOT/每千 token 成本 + GPU 利用率定瓶颈，然后按 ROI 顺序上 continuous batching、前缀缓存、量化、输出限长与小模型路由；真正的交付是能复跑的压测与成本看板。

---

### 13. Describe a time you dealt with conflicting priorities or stakeholder feedback. What would your current manager say about you?

**Anchor answer**

This is a two-part question; answer both, and the second part is where candidates lose points by getting vague.

Structure that reads as senior:

1. **Name the conflict precisely** — not "people disagreed", but "research wanted accuracy, infra wanted p99 latency under 200 ms, and we had one GPU budget".
2. **Make the tradeoff explicit and numeric** — put the options on one axis: `option → cost, latency, accuracy, ship date`. A conflict that becomes a table usually resolves itself.
3. **Say what you decided and what you gave up.** "We shipped the smaller model with a 1.8 pt recall drop to hit the date, with a plan to re-evaluate in 6 weeks." Decisions without a stated sacrifice sound like you avoided the decision.
4. **Show the dissent handled, not suppressed** — the losing stakeholder got their concern recorded as a metric with a threshold that would reopen the decision.

For "what would your manager say": pick **two** traits and give evidence, not adjectives. E.g. "they'd say I over-communicate in writing — every tradeoff doc I send has the option table and the decision log, and two engineers have told me they used it to unblock themselves." Avoid "I'm a team player" with no artifact behind it.

Trap to avoid: a story where you were simply right and everyone else was wrong. Interviewers read that as low self-awareness.

**中文要点**：先说清冲突是什么，再把选项摆成"成本/延迟/精度/日期"一张表，明确说出放弃了什么；第二部分必须给两个特质 + 具体证据（文档、同事引用），别讲"我很配合"这种形容词。

---

## 附：这 13 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| CUDA / kernel 真实功力 | 1（10x 慢的嫌疑清单）、2（paged KV 块管理器）、3（FP16 崩掉怎么定位） |
| 显存与并行推理 | 8（70B 单卡算账）、10（405B TP/PP/EP 放哪） |
| 客户现场能力 | 12（一周救火 8 卡）、11（播客搜索 + type-ahead） |
| ML 基础是否扎实 | 4（树模型过拟合）、5（Adam vs SGD）、6（错标 + 捷径）、7（无标签聚类评估） |
| 软素质 | 13（冲突优先级） |
