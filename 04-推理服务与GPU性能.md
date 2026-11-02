# 推理服务与 GPU 性能

共 **73 题**，覆盖 **26 家**，另有 **跨公司通用** 一节。

> 考显存账、批处理、KV 管理、并行策略与延迟预算。数字题最能问出真踩过坑的人。

> 题目为**英文原文**（源仓库 Apache-2.0），**答案为中文整理**。每条附 `Source` 时给出原始出处。授权与归属见 [NOTICE.md](NOTICE.md)。

## 目录

| 公司 / 来源 | 题数 |
| :--- | :-: |
| 跨公司通用 | 15 |
| Anthropic | 2 |
| Meta | 1 |
| xAI | 2 |
| Mistral AI | 2 |
| DeepSeek | 1 |
| Moonshot AI（Kimi） | 2 |
| Sarvam AI | 1 |
| Microsoft | 2 |
| Amazon / AWS | 1 |
| Apple | 4 |
| NVIDIA | 3 |
| Tesla | 1 |
| 字节跳动 | 5 |
| 腾讯 | 5 |
| 京东 | 3 |
| 美团 | 4 |
| Groq | 5 |
| Together AI | 3 |
| Perplexity | 1 |
| Cursor | 1 |
| Glean | 1 |
| Character.AI | 4 |
| ElevenLabs | 1 |
| Abridge | 1 |
| Figure AI | 1 |
| Waymo | 1 |


---

## 跨公司通用

### 1. 解释 prefill 与 decode 阶段。为什么 prefill 是计算受限，而 decode 是内存带宽受限？

<!-- EN: Explain the prefill and decode phases. Why is prefill compute-bound and decode memory-bandwidth-bound? -->

**Asked at**: Moonshot AI · NVIDIA · Together AI · 字节跳动 · 阿里巴巴（淘天 / 阿里云）
**Source**: https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization

**参考答案**

prefill 一次前向吃下整个提示：一个 `[T, d] × [d, d]` 的 GEMM，每个从 HBM 读来的权重字节被复用 `T` 次。算术强度随提示长度增长——8k token 时是数千 FLOP/byte，远高于 H100 的拐点（BF16 约 295 FLOP/byte），所以瓶颈在张量核。

decode 每步**每条序列只出一个 token**：同样的权重被完整重读一遍却只产出一个 token，约 2 FLOP/byte。张量核空转，tok/s ≈ `HBM 带宽 / 权重字节`。这就是为什么 70B BF16 单卡无论 ALU 多快都只有 20–40 tok/s，也是为什么批处理是重回算力密度的唯一办法——批为 `B` 时，一次权重读取喂 `B` 个 token。

| 阶段 | 瓶颈 | 手段 |
| :--- | :--- | :--- |
| Prefill | 算力（注意力对 T 二次） | FP8 matmul、分块、前缀缓存 |
| Decode | HBM 带宽 + KV 容量 | 大 batch、权重量化、投机解码 |

两个实践后果：把长 prefill 混进解码批会卡住所有流式用户（队头阻塞 → chunked prefill）；10 万以上上下文时 prefill 不再是 MLP 受限而是注意力受限，FlashAttention 类 kernel 比 matmul 吞吐更重要。

**速记**：prefill 是 GEMM、算力瓶颈；decode 每步只出一个 token、权重读一遍，2 FLOP/byte，纯显存带宽瓶颈；批处理是让 decode 重回算力密集的唯一手段。

### 2. 什么是连续批处理（continuous / in-flight batching），为什么它取代了静态批处理？

<!-- EN: What is continuous (in-flight) batching and why did it replace static batching? -->

**Asked at**: Anthropic · xAI · Mistral AI · NVIDIA · Together AI
**Source**: https://outcomeschool.com/blog/continuous-batching-in-llms

**参考答案**

静态批处理凑齐 N 个请求，pad 到最长序列，然后跑到**最慢**的那个结束。而对话输出长度分布很散（中位数几百 token，p99 上千），提前结束的序列会让自己的槽位在 batch 余生里空转——典型槽位利用率只有 30–50%，padding 还在为没人读的 token 烧算力。

连续批处理**在每一次前向迭代**上做调度决策：

| 静态 | 连续 |
| :--- | :--- |
| batch 只凑一次 | 每步重组 batch |
| 槽位到 batch 结束才释放 | 序列一吐 EOS 立刻释放 |
| pad 到最大长度 | 不 pad，按序列各自的位置 |
| 吞吐上限由最差请求决定 | 上限由 GPU 决定，不由长尾决定 |

它成立的原因是：decode 跨步除了 KV cache 之外无状态，一个槽位就是若干 KV block 加一个位置计数器，所以新请求能中途加入。

随之而来的责任：KV block 分配器、KV 满时的**抢占**（换出或重算）、准入控制（长生成不能饿死短请求）、按请求的 SLO 而非单一队列。连续批处理提升的是**吞吐**，**不**降低单请求延迟——batch 越大会让 TPOT 更差，这正是延迟敏感档位需要独立资源池的原因。

**速记**：静态批处理被最慢请求拖死、填充浪费，槽位利用率常只有三到五成；连续批处理逐步重组 batch，序列一结束立刻换新请求，换来数倍吞吐，但单请求延迟不变、甚至更差。

### 3. PagedAttention 是如何工作的，它解决了 KV cache 碎片化的什么问题？

<!-- EN: How does PagedAttention work, and what problem of KV-cache fragmentation does it solve? -->

**Asked at**: NVIDIA · Together AI
**Source**: https://outcomeschool.com/blog/paged-attention-in-llms · https://outcomeschool.com/blog/how-does-vllm-work

**参考答案**

KV cache 只随实际生成的 token 增长，但服务端必须按 `max_model_len` 预留，因为最终长度未知。按序列预留连续显存会带来三种浪费：内部碎片（一个 300 token 的对话预留了 8k）、外部碎片、以及无法共享公共前缀。

PagedAttention 借操作系统虚拟内存的思路：KV 存成**固定大小的 block**（默认 16 token），每条序列持有自己的 *block table* 把逻辑块映射到物理块，物理块不必连续。

| 特性 | 效果 |
| :--- | :--- |
| 按需分配 | 浪费 ≈ 一个未填满的块（<4%） |
| block table 间接寻址 | 不需要连续预留 |
| 引用计数 + copy-on-write | 共享 system prompt / beam 分支共享 block |

代价是注意力 kernel 里要按 block table 做 gather：读取不如连续 FlashAttention buffer 那样合并，块太小时表遍历开销更明显。块大小是旋钮——短对话 8 就够，长上下文用 32–64 降开销。

标志性数字：真实对话长度分布下，朴素预留浪费 60–80% 的 KV 显存；分页把它压到 5% 以下，几乎 1:1 换成并发序列数。vLLM 的吞吐来自这个并发，而不是 kernel 本身。

**速记**：KV 必须按最大长度预留，导致六到八成显存被浪费；PagedAttention 用固定大小 block + block table 做虚拟内存式管理，浪费降到 4% 以下，外加前缀共享，直接换成并发数。

### 4. 什么是投机解码（speculative decoding）？为什么输出质量不受损，以及在什么情况下它没有帮助？

<!-- EN: What is speculative decoding? Why is output quality preserved, and when does it not help? -->

**Asked at**: NVIDIA · Together AI · 字节跳动
**Source**: https://outcomeschool.com/blog/speculative-decoding

**参考答案**

一个便宜的草稿模型（小模型、n-gram 查表，或目标模型上的额外头如 Medusa/EAGLE）先提 `k` 个 token；目标模型在**一次**前向里为全部 `k+1` 个位置打分——因为 decode 是带宽受限，这次前向的成本约等于一个 decode 步。保留最长被接受的前缀。

质量是**精确**保持的，不是近似，因为接受用的是拒绝采样：以 `min(1, p_target(x)/p_draft(x))` 接受草稿 token `x`，被拒时从残差分布 `norm(max(0, p_target − p_draft))` 采样。合成分布与目标分布完全一致——贪心还是采样，输出都是目标模型自己会产出的。

接受率为 `α` 时，每个目标步的期望接受 token 数是 `(1 − α^(k+1))/(1 − α)`；加速比 ≈ 再除以草稿开销。

什么时候没收益：

- **大 batch**：目标步变成算力受限，校验 `k` 个 token 的成本 ≈`k` 倍 FLOPs——交叉点常在 batch 32–64。
- **高温/开放式对话**：`α` 崩掉；代码和摘要对草稿的容忍度高得多。
- 草稿与目标**领域不匹配**，或草稿本身太大、它自己的延迟成了瓶颈。
- **KV 预算紧**：草稿权重和额外 KV 会吃掉并发所需的内存。

所以它是低到中等并发下的**延迟**工具，不是吞吐工具。

**速记**：小模型起草、大模型一次前向并行校验，接受率用 `min(1, p/p)` 的拒绝采样保证**分布完全一致**；大 batch 下校验成本随 k 线性增长，收益消失，所以它是低并发降延迟工具。

### 5. 解释前缀缓存（prefix caching）/ prompt caching。什么时候该用，以及什么会使已缓存的前缀失效？

<!-- EN: Explain prefix caching / prompt caching. When should you use it, and what invalidates a cached prefix? -->

**Asked at**: Moonshot AI · Character.AI · 字节跳动
**Source**: https://outcomeschool.com/blog/how-does-prompt-caching-work

**参考答案**

前缀缓存把某个 token 前缀的 KV block 存下来，键是哈希链（block hash = hash(父哈希, 本块 token)）。命中前缀就整段跳过 prefill：TTFT 大致按复用比例下降，省下的 prefill 算力还能给别的请求。

只要在可变内容前有一段长而**稳定**的内容就值得用：2k 的 system prompt、工具 schema 块、一批召回的文档、多轮对话的长历史。生产对话常能命中 60–90%；工具列表不断变化的 agent trace 近乎 0。

什么会让它失效——缓存对象是**精确 token 前缀**，不是语义：

| 失效源 | 原因 |
| :--- | :--- |
| 提示中任何靠前的编辑 | 下游所有哈希都变 |
| 增删/重排/改写工具列表 | 工具 JSON 在前部 |
| system prompt 里插了时间戳、请求 id、用户名 | 每个请求都是新前缀 |
| JSON 键序不确定 | 一个键换位就毁掉 5 万 token 的命中 |
| 图片/音频不同、分词器或对话模板不同 | token id 变了 |
| 换了 LoRA 适配器 | 适配器改变 KV |
| KV 压力下 LRU 驱逐或 TTL 到期 | 块没了 |

设计铁律：**稳定内容在前，易变内容在后**，并正式序列化（键排序）。另外注意：除非路由按前缀哈希固定请求（SGLang 的 radix tree）或有共享 KV 存储，缓存是**按节点**的。

**速记**：键是"精确 token 前缀"的哈希链，不是语义；任何前置改动（系统提示、工具顺序、时间戳、JSON 键序、LoRA、分词器）都会让整段失效——稳定内容放前面是唯一有效设计。

### 6. 比较 FP16、BF16、FP8、INT8、INT4 与 FP4 在推理服务中的表现。每往下走一档会坏在哪里？

<!-- EN: Compare FP16, BF16, FP8, INT8, INT4 and FP4 for serving. What breaks at each step down? -->

**Asked at**: Mistral AI · Apple · NVIDIA · Together AI · Character.AI
**Source**: https://outcomeschool.com/blog/how-does-model-quantization-work

**参考答案**

| 格式 | 字节/参数 | 收益 | 什么会坏 |
| :--- | :--- | :--- | :--- |
| FP16 | 2 | 基线 | 大模型训练时范围溢出（推理无此问题） |
| BF16 | 2 | FP32 的量程、8 位尾数 | 推理上无可测损失，默认选择 |
| FP8 (E4M3) | 1 | Hopper/Blackwell 上约 1.8× 吞吐 | 激活离群值 >6σ；需 per-tensor 缩放，稳妥要 per-token/per-block |
| INT8 (W8A8) | 1 | 老 GPU 上接近 FP8 | 不做 SmoothQuant 时激活离群值把精度打崩，小模型最先受害 |
| INT4 权重-only (AWQ/GPTQ) | 0.5 | 小约 4×，对带宽受限的 decode 收益大 | 约 7B 以下退化，数学/代码/抄写先崩；反量化开销可能让大 batch 更慢 |
| FP4 (NVFP4, block scale) | 0.5 | Blackwell 上比 FP8 再快 2× | 需要 per-block scale + 常要 QAT；朴素 FP4 明显损伤推理能力 |

两条来自真实上线的规矩：

1. **损失并不均匀**。首尾层、`down_proj`、以及任何需要精确抄写的任务（JSON、ID、数字）最先坏。要在自己的任务上测，别看困惑度。
2. **KV cache 要单独量化**。FP8 KV 大致让并发翻倍；INT4 KV 则会让长上下文针检索悄悄失效，而短提示评测看起来完全正常。

另外：权重-only 量化加速 decode（带宽）但**不加速** prefill（算力），长提示负载可能一点收益都没有。

**速记**：BF16 是无脑默认；FP8 要处理激活离群值，INT8 需 SmoothQuant，INT4 权重-only 在 7B 以下崩、且只降 decode 不降 prefill；KV cache 量化要单独验证长上下文。

### 7. 比较张量并行、流水线并行、数据并行、序列并行与专家并行。什么时候需要组合使用？

<!-- EN: Compare tensor, pipeline, data, sequence and expert parallelism. When do you combine them? -->

**Asked at**: Google DeepMind · Meta · Amazon（AWS） · NVIDIA

**参考答案**

| 策略 | 切什么 | 通信 | 最佳场景 |
| :--- | :--- | :--- | :--- |
| 数据并行（DP） | 副本 | 前向无通信 | 扩吞吐 |
| 张量并行（TP） | 每个权重矩阵 | 每层 2 次 all-reduce，延迟关键 | 机内 NVLink，≤8 卡 |
| 流水并行（PP） | 按层分阶段 | 边界处少量 P2P，但有气泡 | 跨机、超大模型 |
| 序列/上下文并行（SP/CP） | token 轴 | K/V ring 或 all-gather | 10 万以上上下文 |
| 专家并行（EP） | MoE 专家 | 每层 all-to-all，对负载不均敏感 | MoE 模型、大集群 |

支配性规则：**带宽最高的维度放最内层**。TP 走机内 NVLink，PP 跨机走 InfiniBand，DP 最外。把 TP 推到 8 卡之外（出了 NVLink 域）是经典错误——每层 all-reduce 走 400G IB，每个 decode 步都多几毫秒，ITL 直接毁掉。

两节点上跑 405B 这类偏稠密模型的组合示例：TP=8 × PP=2 × DP=N，微批数 ≥2×PP 阶段数，流水气泡才接近 `(PP−1)/(微批数+PP−1)`。

什么时候别组合：batch 1 时 PP 几乎全是气泡——小批 decode 要 TP+DP，不是 PP。EP 只有在路由均衡时才有意义，一个热点专家会串行化整个 all-to-all。

**速记**：带宽最高的维度放最内层——TP 关在 NVLink 单机 8 卡内，PP 跨机，DP 最外；TP 跨机做 all-reduce 是 decode 延迟的头号杀手，PP 在 batch 1 全是气泡。

### 8. 估算服务一个 70B 模型所需的 GPU 显存：权重、KV cache、激活值、碎片。

<!-- EN: Estimate the GPU memory needed to serve a 70B model: weights, KV cache, activations, fragmentation. -->

**Asked at**: NVIDIA · 字节跳动
**Source**: https://outcomeschool.com/blog/kv-cache-in-llms · https://outcomeschool.com/blog/paged-attention-in-llms

**参考答案**

按 Llama-2-70B 的形状算（80 层、64 头、8 个 KV 头、head_dim 128）：

| 组成 | 算式 | 结果 |
| :--- | :--- | :--- |
| 权重 BF16 | 70B × 2 B | 140 GB |
| 权重 FP8 | 70B × 1 B | 70 GB |
| 每 token KV | 2 × 80 × 8 × 128 × 2 B | 320 KB |
| 8k 上下文的 KV | 320 KB × 8192 | 每条序列 2.6 GB |
| 激活 | 每阶段 分块 × hidden × 2 B，加 logits 峰值 | 瞬时约 1–3 GB |
| 碎片 | 分页（<4%）vs 预留（60–80%） | 0.05–5 GB |
| 运行时开销 | CUDA graph、workspace、通信 buffer | 每进程约 2 GB |

所以规划是：先放权重，再用 `KV 预算 /（每 token KV × 平均序列长）` 得并发数。8×H100（640 GB）+ BF16 权重：约 490 GB 给 KV → 大约 **180 条并发的 8k 序列**；权重和 KV 都用 FP8 则翻倍还不止。

真正的坑是词表 logits 峰值：8k token 的 prefill 要物化 `[8192, 128k]` 的分数，单次分配约 2 GB，实际 OOM 的往往是它而不是稳态那笔账。第二个坑：每 token KV 随 **KV 头数**线性，所以 GQA（8 vs 64）直接换来 8× 并发——模型选型比任何服务端调优都更有决定权。

**速记**：先算权重（BF16 140GB），再按 `每 token 320KB × 长度` 算 KV；真正 OOM 的往往是 prefill 的 vocab logits 瞬时分配，而 GQA 的 KV 头数直接决定并发上限。

### 9. 什么是 TTFT、TPOT、ITL 与吞吐，它们之间如何相互权衡？

<!-- EN: What are TTFT, TPOT, ITL and throughput, and how do they trade against each other? -->

**Asked at**: Microsoft · Apple · Perplexity
**Source**: https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization · https://www.youtube.com/watch?v=XD8DD4cEHu0

**参考答案**

| 指标 | 定义 | 主要由谁决定 |
| :--- | :--- | :--- |
| TTFT | 请求 → 首 token | 排队等待 + prefill |
| TPOT | (总时长 − TTFT) / (n_out − 1) | 单步 decode 成本 |
| ITL | token 之间一个个的间隔 | decode 步 + 任何抢占槽位的 prefill |
| 吞吐 | 全集群的 token/s 或 请求/s | batch、硬件、量化 |

TPOT 是均值，**用户感知的是 ITL p99**。均值 20 ms 但偶发一次 600 ms 停顿，体感就是坏的；而那个停顿几乎总是某个长 prefill 占着 batch，不是 kernel 慢。

权衡：

- **batch ↑** → 吞吐 ↑、TPOT ↑、TTFT ↑（排队）。decode 带宽受限，所以在变为算力受限之前，加 batch 几乎免费。
- **chunked prefill** → 保住 ITL，代价是一些 TTFT 和几个百分点的吞吐。
- **投机解码** → 小 batch 降 TPOT/ITL，大 batch 中性到负收益。
- **更长的输出上限 / 流式** → 吞吐不变，感知延迟大变。

Little's law 把一切串起来：`并发 = 吞吐 × 延迟`。固定 SLO 下的容量规划是**并发预算**，不是一个 QPS 数字。

交互式目标（典型体感）：TTFT 300–800 ms，TPOT 20–50 ms，ITL p99 约 100 ms 以内。超过约 30–40 tok/s 人已经分辨不出差别——把余量花在 TTFT 上。

**速记**：TPOT 是均值，用户感知的是 ITL p99，而 p99 尖刺基本都来自长 prefill 抢占 batch；批越大吞吐越高、TPOT 越差，扩容要按 Little's law 算并发而不是 QPS。

### 10. 做一下 roofline 计算：在 batch size 为 1 时，一张 H100 服务 70B 模型每秒能产出多少 token？

<!-- EN: Do the roofline maths: how many tokens/sec can one H100 produce for a 70B model at batch size 1? -->

**Asked at**: NVIDIA · Together AI
**Source**: https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization

**参考答案**

batch=1 的 decode 每产一个 token 要把所有权重读一遍，因此：

```
tok/s ≈ HBM bandwidth × efficiency / weight bytes
```

H100 SXM：峰值 3.35 TB/s，实际可达约 80–85%。70B BF16 权重 = 140 GB。

`3.35e12 / 140e9 ≈ 24 tok/s` 理论值，按真实效率约 **19–20 tok/s**。算术强度 2 FLOP/byte，而拐点是 `989e12 / 3.35e12 ≈ 295` FLOP/byte——只用到约 0.7% 的峰值算力，张量核在这里无关。

然后是物理问题：140 GB 装不进 80 GB，所以必须 TP=2，每卡持 70 GB、每 token 读 70 GB → roofline `3.35e12 / 70e9 ≈ 48 tok/s`，再减去两次每层 all-reduce。实测 Llama-2-70B 在 2×H100、batch 1 下大约 30–40 tok/s。

几条线性、值得背下来的推论：

| 变化 | 对 batch-1 decode 的影响 |
| :--- | :--- |
| FP8 权重 | 约 2× tok/s |
| INT4 权重 | 约 4×（直到反量化开销咬回来） |
| batch 8 | 聚合约 7×，TPOT 略差 |
| batch 147 | decode 到达拐点，算力成为限制 |

**速记**：batch=1 时 `tok/s ≈ 带宽 / 权重字节`，70B BF16 单卡理论 24、实际 20 左右，算力利用率不到 1%；70B 装不进 80GB，实际要 TP=2，roofline 约 48、实测 30~40。

### 11. 什么时候你会选 vLLM、SGLang、TensorRT-LLM 还是自研栈？

<!-- EN: When would you choose vLLM vs SGLang vs TensorRT-LLM vs a custom stack? -->

**Asked at**: NVIDIA · Together AI · 腾讯 · 字节跳动
**Source**: https://outcomeschool.com/blog/how-does-vllm-work · https://outcomeschool.com/blog/how-does-sglang-work · https://outcomeschool.com/blog/how-does-tensorrt-llm-work

**参考答案**

| | vLLM | SGLang | TensorRT-LLM | 自研 |
| :--- | :--- | :--- | :--- | :--- |
| 模型覆盖 | 最广（HF 当天支持） | 广，最新模型略滞后 | 窄，精挑细选 | 你自己造的 |
| 招牌能力 | PagedAttention、LoRA 多适配器 | RadixAttention 前缀树、压缩 FSM 结构化输出 | 单卡 kernel 最优、FP8/FP4、in-flight batching | 特殊 KV 传输、确定性 |
| 迭代成本 | 分钟级 | 分钟级 | 每个（模型, GPU, TP, 量化）组合都要编译，几十分钟 | 一个团队 |
| 运维复杂度 | 低 | 中 | 高 | 极高 |

能扛住生产检验的选型规则：**按模型迭代频率选，不按 benchmark tok/s 选**。每周更新模型或同时托管几十个微调版本，vLLM/SGLang 在运维上完胜，即使编译引擎快 20%。只服务一两个冻结模型、硬件固定、要榨出最后 20%，TensorRT-LLM 才回本。流量前缀重（agent、多轮、长公共 system prompt）或大量输出受限 JSON 时，SGLang 是默认——那里 RadixAttention 与 FSM 解码器值 10–30%。

自研只有硬约束才成立（确定性延迟、非标准互连、自建 KV 传输的 PD 分离）。否则你是在养一个推理团队去重新推导 vLLM。

**速记**：按**模型迭代频率**选型而非跑分——模型常换选 vLLM/SGLang，模型冻结且要榨干硬件选 TRT-LLM；前缀重的 agent/多轮场景 SGLang 的 RadixAttention 值得 10~30%。

### 12. 你会如何把 LLM 推理服务成本降低 10 倍？列举所有可用手段并排序。

<!-- EN: How would you cut LLM serving cost by 10x? Enumerate every lever and rank them. -->

**Asked at**: Microsoft · Amazon（AWS） · NVIDIA · Cursor
**Source**: https://www.youtube.com/watch?v=jV2sCj4lHYk · https://outcomeschool.com/blog/llm-inference-optimization

**参考答案**

按"每个工程周的回报"排序，需求侧优先——因为它动的是分子不是分母：

| # | 杠杆 | 典型倍数 | 备注 |
| :-: | :--- | :--- | :--- |
| 1 | 砍掉浪费的 token（裁剪提示、限制输出、去掉啰嗦 CoT、删掉没用的 few-shot） | 1.5–3× | 最便宜，不动基础设施 |
| 2 | 缓存：前缀/KV 缓存、精确缓存、FAQ 型流量的语义缓存 | 1.5–3× | 固定 system prompt 命中 60–90% |
| 3 | 路由：简单的那 60–80% 走小模型/蒸馏模型，长尾走前沿模型 | 3–10× | 单品最大杠杆 |
| 4 | 利用率：连续批处理 + 合适的 batch 与并发 | 1.5–2× | 集群常只有 15–30% GPU 利用率 |
| 5 | 量化：FP8 权重 + FP8 KV | 约 1.8× | 低 batch decode 用 INT4 |
| 6 | 低 batch 下投机解码 / MTP | 2–3× tok/s | 带宽受限时既降延迟也降 $/token |
| 7 | 硬件与合同：新一代卡、批处理走 spot、稳定负载走预留 | 1.5–3× | 波动负载用按 token 计费的供应商 |
| 8 | 集群规模下的 PD 分离 | 1.2–1.4× | 只在 KV 传输够快时成立 |
| 9 | 蒸馏到更小的模型 | 3–20× | 天花板最高，最慢也最险 |

10 倍从来不是单一杠杆，而是 `1.6 × 1.5 × 1.4 × 1.3 × 1.2`。先做度量：把每千次请求的花费拆成**输入**与**输出** token、按路由、按租户。输出 token 单价是输入的 3–5 倍，一段啰嗦 JSON 或话痨 system prompt 是成本 bug，不是模型选择。

**速记**：先砍需求侧——删无用 token、前缀/语义缓存、小模型路由，再谈供给侧的批处理、FP8、投机解码、硬件；十倍永远是五个 1.3~1.6 相乘，且先按输入/输出 token 拆分账单。

### 13. 一次没有改动模型的发布之后，你的 p99 延迟翻倍了。讲讲你的排查过程。

<!-- EN: Your p99 latency doubled after a deploy with no model change. Walk through the diagnosis. -->

**Asked at**: OpenAI · Amazon（AWS） · Databricks · Perplexity

**参考答案**

先按分位切开看：**p50 不动、p99 翻倍 = 排队或长尾队列问题，不是 kernel 变慢**。kernel 退化会把 p50 一起拉高。如果 p50 也动了，去 diff 引擎/驱动/GPU 池。

然后按顺序查：

1. **流量形状**。输入长度 p99、`max_tokens` 默认值、并发会话数。利用率 ρ→0.9 时排队延迟会爆，流量涨 20% 就能让 p99 翻倍，而算力完全没变。
2. **batch 组成**。长 prefill 与 decode 交错 → 队头阻塞。看每请求 ITL 与提示长度的关系。
3. **缓存命中率**。改一行 system prompt，或重排一次工具列表，前缀缓存就归零，那笔 prefill 成本全进延迟路径。这是最常见的"什么都没改"的发布。
4. **重试**。网关重试策略把 2% 的错误率放大成 20% 的额外负载；查 429/5xx 和重复请求 id。
5. **显存压力**。KV block 驱逐/抢占（换出或重算），表现为恰好是最长对话出现数秒级停顿。
6. **逐字 diff 发布内容**。分词器版本（同样文本多 15% token）、对话模板、调度器参数、CUDA/驱动、池子里新增的机型、MTU 变更。
7. **邻居干扰**。一条退化的 NVLink/IB 链路会让节点上所有人的 TP 集合通信一起卡。

方法：让每个看板都能按租户、路由、输入长度分桶、节点切片；动了的那个切片就是答案。

**速记**：先看 p50——p50 不动、p99 翻倍是排队或长尾队列问题，不是 kernel 变慢；排查顺序是流量形状、batch 组成、前缀缓存命中率、重试风暴、KV 抢占、分词器/模板版本。

### 14. 什么是 chunked prefill，为什么它能在混合流量下改善尾延迟？

<!-- EN: What is chunked prefill, and why does it improve tail latency under mixed traffic? -->

**参考答案**

一段 `T` token 的 prefill 会在整个时长内占着 batch——大模型上 16k–32k 提示就是几百毫秒。调度器若把它当整体准入，同批所有解码请求都要等它：ITL 瞬间尖刺到 prefill 的时长，流式输出肉眼可见地卡住。混合负载下（大量短对话 + 少量长 RAG/代码库提示），就是这一个效应主导了 p99 ITL。

chunked prefill 把提示切成 `C` 个 token 的块（通常 512–2048），**在固定 token 预算下逐步调度**：每次迭代准入若干块 token 加上所有运行中序列的 decode token。decode 永远不丢槽位。

| | 整块 prefill | 分块 |
| :--- | :--- | :--- |
| ITL p99 | 尖刺到 prefill 时长 | 平滑，受预算约束 |
| 长请求的 TTFT | 最好 | 变差（摊到更多步） |
| 吞吐 | 最好 | 低几个百分点 |

吞吐代价是真实的：每个块都要为注意力从 HBM 重读前缀的 KV，块太小会浪费带宽、让 GPU 喂不饱。

调参规则：每步 token 预算 ≈ decode batch 的 2–4 倍。太小 → 步数太多、GEMM 效率差；太大 → ITL 尖刺回归。另外要给单请求 `max_num_batched_tokens` 设上限，免得一个 10 万 token 的用户独占队列。

**速记**：长 prefill 整块入批会让所有解码请求的 ITL 尖刺到几百毫秒；按固定 token 预算切块混批后 ITL 变平滑，代价是长请求 TTFT 变差、吞吐降几个点，块大小要按解码批量的 2~4 倍调。

### 15. 解释 prefill/decode 分离式（disaggregated）推理服务，以及它在什么时候才划算。

<!-- EN: Explain disaggregated prefill/decode serving and when it pays for itself. -->

**Asked at**: Moonshot AI · Groq · 字节跳动
**Source**: https://outcomeschool.com/blog/prefill-decode-disaggregation

**参考答案**

prefill 和 decode 想要的是相反的机器：prefill 是大 GEMM，要算力和短命内存；decode 是 GEMV，要带宽和巨大的 KV 容量。放在同一个池子里，每个调度决策都是妥协，而且长提示不断抢走 decode 槽位。分离方案把 **P worker** 和 **D worker** 拆开跑，KV 通过 NVLink/RDMA 从 P 传到 D。

它划算的条件：

- 提示相对输出很长（RAG、代码库问答、文档对话）——prefill 占成本的大头；
- 你要在能接长提示的同时守住 ITL SLO；
- 集群足够大，拆池不会损失批处理效率；
- 硬件异构——把互联好的贵卡放在 prefill 需要的位置。

KV 传输超过节省时它就不划算。算一下：70B、8k 提示 → 2.6 GB KV。NVLink 约 900 GB/s，约 3 ms（可以）；400G IB 约 50 ms；100G 以太约 210 ms（TTFT 直接死）。经验法则：传输时间超过 TTFT 预算的 10–15%，就别拆。

另外：小集群因批处理变差而亏的比赚的多；大多数单区域部署，chunked prefill 已经把尾部延迟的抱怨解决了。

**速记**：prefill 要算力、decode 要带宽与 KV 容量，拆池后靠 KV 传输衔接；算清 `KV 字节 × 长度 / 互连带宽`——8k 提示 2.6GB，NVLink 3ms 可行、100G 以太 210ms 直接毁掉 TTFT，小集群拆分反而更亏。


---

## Anthropic

### 1. 设计一个批处理推理系统，让 100 个请求的耗时与 1 个请求相同。

<!-- EN: Design a batched inference system where 100 requests take the same time as 1. -->

**Asked at**: Anthropic
**Source**: https://outcomeschool.com/blog/continuous-batching-in-llms

**参考答案**

这个前提只在负载**受显存带宽约束**（即小 batch 的 decode）时成立。先把这点说清楚，再按它设计。

| 阶段 | 特性 | 批处理收益 |
| :--- | :--- | :--- |
| Prefill | 算力受限，`O(T²)` | 收益小，padding 浪费还伤性能 |
| Decode | 带宽受限（读全部权重、算量极小） | 近似线性，直到转为算力受限 |
| KV cache | 每序列持续增长 | 分页分配，避免碎片 |

核心是**连续批处理（continuous / in-flight batching）**：每个 decode step 从「当前正在生成的序列」组批，在迭代边界注入新请求、淘汰已完成请求；而不是静态批处理里一条短请求等最长的那条。

```text
loop:
  step = scheduler.pick(max_batch, token_budget)
  k, v = paged_kv.gather(step.seqs)
  logits = model.decode(step.tokens, k, v)   # 一次融合 kernel
  scheduler.update(finished, new_arrivals)
```

「100 个 == 1 个」在批次进入算力受限区时就失效，延迟重新线性增长——要指出这个拐点并实测。延迟 SLO 与批处理天然冲突：按 **token 预算**（每步总 token 上限）而不是纯条数准入，否则一次 8k token 的 prefill 就能拖垮所有人的 p99。分块 prefill 把大 prefill 与 decode 交错，保护 token 间延迟。分页 KV（vLLM 式）才让「随时进出批次」变便宜，连续缓冲区会强制预分配。

**速记**：continuous batching + paged KV 才能让 100 个请求≈1 个；只在带宽瓶颈区成立，用 token 预算而非纯条数控批。

### 2. 为 Claude 规模的 LLM API 设计服务栈。在不破坏 p99 延迟的前提下最大化 GPU 利用率。

<!-- EN: Design the serving stack for a Claude-scale LLM API. Maximise GPU utilisation without wrecking p99 latency. -->

**Asked at**: Anthropic
**Source**: https://outcomeschool.com/blog/llm-inference-optimization

**参考答案**

张力是结构性的：利用率想要大批次，p99 想要短队列。解法是**把两个延迟特性不同的阶段拆开、把租户隔离**，而不是拧某一个旋钮。

| 层 | 设计 | p99 保护 |
| :--- | :--- | :--- |
| 边缘 / 网关 | 鉴权、配额、请求整形、流式 | 尽早丢弃，绝不在边缘排队 |
| 路由 | 模型档位、缓存查找、前缀亲和 | 前缀命中的请求粘性路由 |
| 调度 | 连续批处理、分块 prefill、优先级类 | 给交互流量预留槽位 |
| Prefill 池 | 算力密集、可解耦 | 与 decode 的 SLO 分离 |
| Decode 池 | 带宽密集、分页 KV | 按 token 预算准入 |
| KV 存储 | 分页块、前缀共享、offload | 内存有界 + 淘汰策略 |

关键取舍：**prefill 与 decode 必须解耦**，长 prompt 在合置 worker 上会卡住 decode step，拆开后两个池各自按自己的瓶颈（FLOPs vs 带宽）扩容。准入按**每步 token 预算**，队列深度按「时间」而非请求数设限——「等待 > 300ms 就降级或丢弃」。p99 由队列尾部决定，所以要略微超配，并按**排队延迟**而不是平均利用率做自动扩缩；95% 利用率通常是 p99 灾难。聊天场景杠杆最大的是前缀缓存（共享 system prompt 变成免费 prefill）。权重与 KV 在质量允许处量化，但要按自家评测而不是困惑度验收。

**速记**：靠「prefill/decode 分离 + 按 token 预算准入 + 按排队延迟扩容」同时拿利用率和 p99；前缀缓存是聊天场景最大杠杆。


---

## Meta

### 1. 你需要向数亿助手用户服务一个 Llama 级别的 70B+ 模型。服务栈长什么样，钱花在哪里？

<!-- EN: You need to serve a Llama-class 70B+ model to hundreds of millions of assistant users. What does the serving stack look like and where does the money go? -->

**Source**: https://outcomeschool.com/blog/llm-inference-optimization

**参考答案**

钱花在 **decode 的 GPU·秒**上，而 decode 是**显存带宽瓶颈**——所以整栈的目标只有一个：在 KV cache 撑得住的前提下把 batch 拉大。

| 层 | 干什么 | 杠杆 |
| :--- | :--- | :--- |
| 路由 | 便宜模型先答，难题才升级 | 40–70% 流量根本不需要大模型 |
| 前缀缓存 | 缓存 system prompt 与工具 schema 的 KV | agent 流量 prefill 降 50%+ |
| Prefill/decode 分离 | 两段硬件形态不同 | 各自 MFU 都上升 |
| 批处理 | 连续/在途 batching | 吞吐近线性增长直到 KV 上限 |
| KV 显存 | PagedAttention、fp8/int8 KV | 约 2× batch，质量几乎无损 |
| 权重 | fp8/int8，必要时 4bit | HBM 减半，batch 更大 |
| 投机解码 | 小 draft 模型 | 接受率高时延迟降 1.5–2× |

容量自检：70B fp8 约 70 GB 权重，80 GB H100 只剩约 10 GB——不够。所以**每个副本必须跨 ≥2 卡做 TP**（或换更小/更狠量化的模型），而每副本 batch 由 KV 字节数封顶：`2 × layers × kv_heads × head_dim × seq × bytes`。8k 上下文下这个数直接决定你的 $/1k tokens。

失败模式：共享 prompt 引发**缓存击穿**（相同请求要合并）；一条超长生成造成**队头阻塞**（分块 prefill + max_tokens 上限）；**路由抖动**（同一会话里来回切模型，用户看到质量断崖）。

哪里不该优化：长输出时 prefill 只占小头，团队却常年在编码器上过度投入。

**速记**：钱在 decode 的 GPU·秒，瓶颈是带宽——目标是把 batch 拉大。路由/前缀缓存/PD 分离/PagedAttention+fp8 KV/投机解码；70B fp8 ≈70GB，单卡放不下，TP≥2。


---

## xAI

### 1. 估算在 128k 上下文下服务 70B 级模型所需的 KV cache 显存。当它放不下时你怎么办？

<!-- EN: Estimate the KV-cache memory to serve a 70B-class model at 128k context. What do you do when it doesn't fit? -->

**Source**: https://outcomeschool.com/blog/kv-cache-in-llms · https://outcomeschool.com/blog/kv-cache-compression

**参考答案**

公式：`每 token 字节 = 2（K 与 V）× layers × kv_heads × head_dim × dtype 字节`。**起作用的是 KV 头数，不是 Q 头数**。

取 70B GQA（8 个 KV 头、head_dim 128、80 层、bf16）：
`2 × 80 × 8 × 128 × 2 = 327,680 B ≈ 320 KiB/token`，`× 131,072 ≈ 40 GiB / 一条 128k 序列`。

也就是说，权重（bf16 约 140 GB，本身已要跨 ≥2 卡做 TP）之后，单条满长会话就能吃光一张 80 GB 卡。8 条并发 128k 就是约 320 GiB 的缓存——**并发塌掉的原因永远是显存，不是 FLOPs**。

| 手段 | 效果 | 代价 |
| :--- | :--- | :--- |
| 更多 KV 头共享（MQA/GQA/MLA） | 小 4–8 倍 | 质量，且需重训 |
| Paged block + 前缀共享 | 消除碎片、共享 system prompt | 分配器复杂度 |
| FP8 / INT4 KV | 小 2–4 倍 | 上下文越长精度损失越明显 |
| 滑窗 / 混合注意力 | 缓存恒定 | 无精确远距离召回 |
| 限并发 × 上下文（准入控制） | 线性 | 产品限制 |
| 缓存卸载到 host/CPU | 放得下 | 每步 PCIe 延迟 |

**速记**：按 KV 头算：320 KiB/token × 128k ≈ 40 GiB/条；装不下时依次上 GQA/MLA → paged+前缀共享 → FP8 KV → 滑窗 → 限流。

### 2. 为 LLM API 设计一个限流器，其中成本随 token 数而非请求数增长。

<!-- EN: Design a rate limiter for an LLM API where cost scales with tokens, not requests. -->

**参考答案**

请求数不是正确单位：一次 20 万 token 的 prefill 约等于 1000 次 200 token 的闲聊。单位必须是 **token**；而准入时输出长度未知，所以只能**预扣 + 结算**。

1. **成本模型**：`cost = 输入 token + 期望输出 token`（用该租户历史的分位数，或用其 `max_tokens`）。若 prefill/decode 分池，两者权重不同。
2. **准入**：每租户一个 token bucket（`rate`、`burst`）**加**一个全局桶，两个都过才放行——只有租户桶时，N 个各自守规的租户照样打爆集群。
3. **预扣**：先扣 `input + max_tokens`，完成时退还 `max_tokens − 实际`。
4. **在途记账**：预扣额度**不在**桶里，而是放在带 TTL 的 held 计数中，否则 worker 崩溃会永久泄漏配额。

```python
if b.tokens < n or global_.tokens < n:
    return False, (n - b.tokens) / b.rate      # retry_after
b.tokens -= n; global_.tokens -= n
```

**哪里会坏**：每 worker 本地桶会向上漂移，必须把增量推给共享计数器（Redis Lua 原子完成 refill+check+decrement），或给每个节点发一份**配额租约**——省掉往返，代价是超发上界等于租约大小，租约取租户速率的 1–5% 可接受。流式场景结算变成强制项：按 chunk 增量扣，到顶直接截断流，而不是事后发现超支。

设计规则：**花额度 fail-closed，读额度 fail-open**——计数器存储挂了就退化成保守的本地桶，而不是全站 500。

**速记**：按 token 就"预扣+结算"：预付 input+max_tokens、完成退差；租户桶 + 全局桶双闸门，分布式用 Redis 原子脚本或配额租约，崩溃靠 TTL 防泄漏。


---

## Mistral AI

### 1. 估算服务 Mistral 7B 的 KV cache 显存，并设计滑动窗口注意力所能支持的滚动缓冲区（rolling buffer）缓存。

<!-- EN: Estimate the KV-cache memory for serving Mistral 7B, and design the rolling-buffer cache that sliding-window attention enables. -->

**Source**: https://outcomeschool.com/blog/kv-cache-in-llms · https://outcomeschool.com/blog/how-does-sliding-window-attention-work

**参考答案**

Mistral 7B：32 层、32 Q 头、8 KV 头、head_dim 128、bf16。

`每 token 字节 = 2 × 32 × 8 × 128 × 2 = 131,072 B = 128 KiB/token`

| 上下文 | 朴素 cache | 环形 cache（W=4096） |
| :--- | :--- | :--- |
| 4k | 512 MiB | 512 MiB |
| 32k | 4 GiB | 512 MiB（恒定） |
| 128k | 16 GiB | 512 MiB（恒定） |

L4/A10 级卡扣掉约 14 GB int4 权重后可用 24 GiB：朴素 cache 只能跑 1 条 128k 会话；环形 cache 能跑约 40 条，因为每条永远只驻留 4k 的缓存。

**设计**：每层每条序列一个固定环。

```python
slot = pos % W                      # 覆盖最旧
k_cache[layer][slot] = k
allowed = (q_pos >= k_pos) & (q_pos - k_pos < W)   # 对角带状 mask
# RoPE 的位置用真实计数，不用 slot
```

实践里两处会坏：RoPE 必须用**绝对位置**而不是 `slot`，否则长上下文续写退化；mask 必须是**对角带**而不是前缀，环绕之后用朴素 `slot < n` 会静默地注意到过期 slot。另外，对一个全新的 128k prompt 做 prefill，那一趟仍是完整 128k 注意力——窗口只约束显存，不约束 prefill 的计算。

**速记**：128 KiB/token；环形 cache 按 W=4096 定额（512 MiB/序列）而非随长度增长；坑是 RoPE 用绝对位置、mask 写成对角带。

### 2. 你需要针对客户的硬件对模型做量化。如何选择量化方案，以及如何证明质量没有退化？

<!-- EN: You need to quantize a model for a customer's hardware. How do you choose a scheme, and how do you prove quality hasn't regressed? -->

**Source**: https://outcomeschool.com/blog/how-does-model-quantization-work

**参考答案**

从硬件**往上**选，不要从论文往下套。四个问题定方案：

| 约束 | 若是 | 方案 |
| :--- | :--- | :--- |
| 卡有 fp8 张量核（H100/L40S）？ | decode 受带宽限制 | **FP8（W8A8）**，近乎无损、工程量最小 |
| 只有 int8/int4 kernel（老 GPU 或 CPU）？ | 受显存限制 | **INT8 weight-only**，或 **INT4（GPTQ/AWQ）** |
| batch 1 延迟最关键？ | 带宽受限 | weight-only INT4 胜出，量化激活只增加 kernel 开销 |
| 大 batch / 吞吐优先？ | 算力受限 | W8A8/W4A8 有用；W4A16 几乎没用 |

对客户的默认建议：**AWQ 或 GPTQ int4、group size 128、KV 用 fp8**，30B 模型能塞进一张 24 GB 卡。

**证明没退化**——困惑度不算证据。跑分级闸门：

1. **任务套件**：用客户自己的标注集，算 exact-match/F1，并**分切片**报。
2. **格式合法率**：满足 JSON schema 的解析比例——量化模型最先在这里塌。
3. **长上下文针测试**：在客户真实上下文长度上做；int4 KV 也是最先在这里退化。
4. **分布偏移**：与 fp16 在 1–2k prompt 上比下一 token 分布的 KL，超阈值的切片标红。
5. **拒答/幻觉率**：固定 prompt 集，量化会移动校准。

然后拿**相同 prompt** 在 5% 流量上 A/B，比任务成功率与人工点赞。闸门：没有任何切片劣化超过事先约定的 delta，且最差 5% 的 prompt 必须人工逐条看。对外给**逐切片**的表，不是一个大数字。

**速记**：从硬件能力倒推（fp8 卡用 W8A8，老卡用 int4 weight-only，batch1 优先就别量化激活）；验证看 schema 合法率、长上下文、逐切片指标，ppl 不算证据。


---

## DeepSeek

### 1. 描述你会如何在 GPU 显存受限的情况下，以低延迟服务一个 671B 参数的 MoE 模型。

<!-- EN: Sketch how you would serve a 671B-parameter MoE model with low latency under GPU-memory constraints. -->

**Source**: https://outcomeschool.com/blog/llm-inference-optimization

**参考答案**

一句话：先算清最少卡数，再用 EP + MLA 压显存、分离 prefill/decode、把通信藏起来。

**先算账**：671B 参数 FP8 ≈ 671 GB 权重。80 GB 卡上，连一个 KV 字节都还没算就要 ≥ 9 张卡——所以问题不是"一张卡装不装得下"，而是"最少几台设备、怎么让它们别闲着"。

| 层次 | 决策 | 理由 |
| :--- | :--- | :--- |
| 权重精度 | FP8 分块量化，norm/gate/head 留 BF16 | HBM 减半，Hopper 上解码约 2× |
| 并行 | 跨节点 EP + MLA 压缩 KV；节点内尽量不做 TP | MoE 按专家切分天然干净，TP 每层加 all-reduce 收益小 |
| 拓扑 | prefill 池（算力密集）与 decode 池（带宽密集）分离 | 两阶段想要的 batch 形状相反，混放两边都浪费 |
| KV | MLA latent 缓存（V3 级模型约 70 KB/token）+ 前缀缓存 | 用 MHA 的话 10 万上下文 KV 会远超权重 |
| 批处理 | 连续批处理、chunked prefill、按专家桶的 decode 微批 | 让 all-to-all 离开关键路径 |
| 热专家 | 顶层几个专家在多 rank 上复制 | 路由呈 Zipf 分布，一个热专家就是整步延迟 |

**必须点名的失败模式**：小 batch 下 all-to-all 主导（于是单用户延迟与整机吞吐直接对立）；单个慢 rank 拖住所有 rank；专家复制会撑爆你本来就不够的显存；长 prompt 让 TTFT 大到必须做前缀缓存，否则 p99 无法挽回。

若 SLO 是交互式对话，宁可多给 2–4 倍 GPU、把单卡占用压小，也别硬往最少卡数上挤。

**速记**：671B FP8 权重就 671GB，先算最少卡数；再 EP + MLA 压 KV、prefill/decode 分离、连续批处理、热专家复制。核心矛盾是小 batch 下 all-to-all 主导延迟，宁多给卡别硬挤。


---

## Moonshot AI（Kimi）

### 1. 讲讲你为什么像 Mooncake 那样把 prefill 与 decode 拆分到不同机器上。这带来什么，代价又是什么？

<!-- EN: Walk me through why you would disaggregate prefill and decode onto separate machines, as Mooncake does. What does that buy you and what does it cost? -->

**Source**: https://outcomeschool.com/blog/prefill-decode-disaggregation

**参考答案**

一句话：prefill 算力密集、decode 带宽密集，混批互相拖累；分离换来独立扩缩与缓存复用，代价是 KV 传输。

**为什么冲突**。prefill 是计算受限（大 GEMM、高算术强度）；decode 是显存带宽受限（每序列一个 token，读整个 KV cache）。放进同一批，调度器只能取折中的 chunk size，且每个 decode step 都继承 prefill 的延迟抖动。

**分离买到什么**：

| 收益 | 机制 |
| :--- | :--- |
| 独立扩缩 | prefill 池按 TTFT SLO 定，decode 池按 TPOT SLO 定——卡数不同、并行策略不同 |
| 无队头阻塞 | 一个 10 万 token 的 prefill 不再卡住所有人的 decode |
| 缓存复用 | decode 节点维护**全局 KV 池**（DRAM/SSD 分层），按前缀哈希命中共享的系统提示词 |
| 利用率更高 | 每个池跑自己天然的 batch 形状 |

**代价**：

- **KV 传输**。10 万 token 上下文每请求数百 MB，走 RDMA 要几十毫秒，可能超过它换来的 decode 时间。长上下文把传输变成新瓶颈——所以要按层流式传输、与计算重叠。
- **多一跳**关键路径，以及一个必须跟踪缓存局部性的全局调度器，否则 token 被送到需要重新拉取的节点。
- **故障面**：两个池、两个故障域，权重更新时还有缓存一致性。

**什么时候别做**：短 prompt、低并发、单机部署——传输与调度开销占主导，你会比同居一机的引擎更慢。

**速记**：prefill 算力密集、decode 带宽密集，混批互相拖累；分离后可独立扩缩、消除队头阻塞、做全局 KV 池前缀复用；代价是 KV 传输（长上下文几百 MB）、多一跳与全局调度复杂度。短 prompt 低并发纯亏。

### 2. 一个聊天助手每一轮都会重发很长的对话历史。你如何避免重复计算全部内容，其中有哪些坑？

<!-- EN: A chat assistant re-sends a long conversation history on every turn. How do you avoid recomputing all of it, and what are the pitfalls? -->

**Source**: https://outcomeschool.com/blog/how-does-prompt-caching-work

**参考答案**

一句话：按内容哈希分块建前缀树，命中前缀直接复用 KV，只 prefill 增量；坑几乎全在缓存键与布局上。

**机制：前缀 KV 缓存，以内容哈希为键**

1. 分词后切成固定大小的块（16–128 token）。
2. 对每块哈希，**哈希包含位置与所有前序块的哈希**——形成块哈希上的基数树/前缀 trie。
3. 查最长匹配前缀，复用其 KV 块，只 prefill 后缀。
4. 该步结束后把新块追加进缓存。

多轮对话是最理想的情形：每轮历史都是上一轮的严格扩展，因此第 $n$ 轮只需 prefill 新用户消息 + 助手自己的回复。

**坑，按代价排序**：

| 坑 | 后果 |
| :--- | :--- |
| 缓存键漏了模型版本 / LoRA 适配器 / 分词器 | 静默返回陈旧或错误的 KV——输出错，而不是报错 |
| 块粒度 | 第 5000 位改一个 token 会让其后全部失效；把易变部分（系统提示、时间戳）放在**最后** |
| 淘汰策略 | 块级 LRU 会和运行中的 batch 抢 HBM；应该分层下沉到 DRAM/SSD 而不是丢弃 |
| 跨租户前缀共享 | 时序侧信道可泄漏 prompt 内容；敏感部署用 per-tenant salt 做哈希 |
| 前缀布局 | 在 prompt 顶部注入任何动态内容都会毁掉命中率——同一产品实测 0% vs 80% |

经验法则：稳定的放前面、易变的放最后，并把**前缀命中率**当成与 TTFT 并列的一等指标来埋点。

**速记**：按块哈希（含位置与前缀哈希）建 radix 树，命中前缀直接复用 KV 只 prefill 增量；坑在缓存键漏模型版本/LoRA/分词器、块粒度导致一处改动全失效、淘汰抢显存、跨租户侧信道、动态内容放开头把命中率打到 0。


---

## Sarvam AI

### 1. 在没有数据中心 GPU 的成本敏感或端侧硬件上，你如何部署一个能力不错的助手？讲讲整套效率工具箱。

<!-- EN: How do you deploy a capable assistant on cost-sensitive or on-device hardware without a datacentre GPU? Walk through the efficiency toolkit. -->

**Source**: https://outcomeschool.com/blog/how-does-model-quantization-work

**参考答案**

一句话：沿效率阶梯往下走，每一级更便宜但更贵在精度；成败取决于"在目标设备上量"和"用目标语言校准"。

| 阶梯 | 技术 | 典型收益 | 代价 |
| :--- | :--- | :--- | :--- |
| 1 | **选更小的基座**（1–8B，从旗舰蒸馏） | 10–50× FLOPs | 能力上限 |
| 2 | **INT4 权重量化**（AWQ / GPTQ / GGUF Q4_K_M） | 权重小 4×，解码约 2–3× | 小幅 perplexity 损失；需 per-channel scale |
| 3 | **KV cache 量化 / 更短上下文** | 并发用户多 2–4× | 长上下文退化 |
| 4 | **投机解码**（极小 draft 或 MTP 头） | 1.5–2× tokens/s | draft 的显存 |
| 5 | **运行时与 kernel 选择**（llama.cpp/ONNX Runtime/CoreML、NEON、int8 GEMM） | 相对朴素 PyTorch 约 2× | 工程投入 |
| 6 | **剪枝 / 少激活参数的 MoE** | 降低算力 | 需重训，质量不均 |
| 7 | **prompt/前缀缓存 + 短系统提示** | 多轮场景收益大 | 缓存显存 |

**比量化格式更决定成败的两条规则**：

1. **在目标设备上量，不要在代理机上量**。手机上的 7B INT4 是**显存带宽受限**的——解码速度正比于每 token 读取的 MB，所以一个保留 FP16 激活的 4-bit 模型，收益远小于压缩比暗示的水平。
2. **按你真正会遇到的数据做量化**。印度文字与代码的激活离群值分布和英文聊天截然不同；用英文维基做校准集，产出的模型会在天城文上崩。用领域匹配的校准集，并且上线前一定在 te/ta/bn/hi 上跑评测。

**什么时候别折腾**：如果设备可用内存 < 4 GB 且任务需要多步推理，端侧就是错误答案——把 ASR/TTS 放端上，LLM 路由到就近的小区域服务器。

**速记**：按阶梯做——先选小基座/蒸馏，再 INT4 权重（AWQ/GPTQ/GGUF）、KV 量化、投机解码、换运行时与前缀缓存。两条规则：在真机上量（解码是带宽瓶颈，看 MB/token），以及用目标语言做校准与验收——英文维基校准会在印度语上崩。


---

## Microsoft

### 1. 一个 Copilot 聊天功能从请求到出现首个有用内容的 p95 预算是 3 秒。时间花在哪里，你如何压缩？

<!-- EN: A Copilot chat feature has a p95 budget of 3 seconds to first useful content. Where does the time go, and how do you cut it? -->

**Source**: https://www.youtube.com/watch?v=XD8DD4cEHu0 · https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization

**参考答案**

先拆预算再谈优化——"首个**有用的**内容"是产品指标，不是 TTFT：

| 环节 | p95 典型占比 | 抓手 |
| :--- | :--- | :--- |
| 鉴权 + 会话 + 地理 RTT | 100–250 ms | 连接复用、边缘终结、同区域推理 |
| 检索（按 ACL 裁剪的邮件/文件/会议搜索） | 300–800 ms | 并行扇出、预热索引、限制候选数、先按 ACL 预过滤 |
| Prompt 组装 + 长上下文分词 | 100–300 ms | 系统提示词已缓存、限制片段数、去重 |
| **Prefill**（算力受限，∝ prompt token 数） | 300–900 ms | 前缀缓存、缩短检索上下文、分块 prefill |
| p95 排队 / 准入 | 0–1000 ms，双峰 | 预留容量、交互流量专用优先道、与批处理分池 |
| 首 token 解码 | 50–150 ms | 投机解码、更小的草稿模型 |
| 流式 + 客户端渲染 | 100–300 ms | 流式输出、增量渲染、不要等引用就绪 |

真正能压 p95（而非均值）的两件事：

1. **尾部是排队，不是算力。** p95 请求的 prompt 是中位数的 3–5 倍，而更长的 prompt 在批调度器里也等得更久；限制检索上下文能一次修好 prefill 与调度两件事。
2. **前缀缓存命中率是聊天场景最大的杠杆**：系统提示词 + 工具 schema 每轮完全相同。一旦被破坏（注入时间戳、每轮重排工具顺序），就是你自找的 300–600 ms 税。

没测量之前不要碰 kernel 级优化：多数 Copilot 形态的产品里模型只占 p95 的 40% 以下，剩下 60% 是检索加调度。

**速记**：先拆 3s 预算——p95 瓶颈通常是"长上下文 prefill + 排队"，不是算力；最大杠杆是前缀缓存命中率与截断检索上下文，其次才是投机解码。

### 2. 估算为 1 亿周活跃用户新增一个 LLM 摘要功能的年度服务成本，以及你会如何把它降低 10 倍。

<!-- EN: Estimate the annual serving cost of adding an LLM summary feature for 100 million weekly active users, and how you'd cut it by 10x. -->

**Source**: https://www.youtube.com/watch?v=jV2sCj4lHYk · https://outcomeschool.com/blog/llm-inference-optimization

**参考答案**

自顶向下估，假设必须显式说出来（下面的数字是**典型量级占位符**，不是实测值——先说清，再让面试官纠正）：

| 驱动项 | 假设 | 取值 |
| :--- | :--- | :--- |
| WAU | 题面给定 | 1 亿 |
| 每人每天摘要次数 | 参与度猜测 | 2 |
| 每天调用数 | 1e8 × 2 | 2 亿 |
| 输入 token | 20 封邮件/文档的线程块 | ~6,000 |
| 输出 token | 8 条要点 | ~250 |
| 单价 | 中档模型、混合计价 | 输入 ~$2.5 / 1M，输出 ~$10 / 1M |
| 日成本 | 2e8 × (6k×2.5 + 250×10)/1e6 | ≈ $3.5 M |
| 年成本 | × 365 | **≈ $1.2 B** |

这个数字就是题目存在的理由：1e8 规模下，**token 数就是乘积**。

10 倍杠杆，按"单位风险收益"排序：

1. **输入压缩（3–6×）**：离线分层摘要线程，摘要的摘要重生成很便宜；输入上限设为最近 N 条消息加滚动摘要。
2. **级联（2–4×）**：路由把 70–85% 的线程交给小而便宜的模型，只在长度、歧义、判定置信度低时升级。质量风险集中在升级切片，必须单独监控。
3. **缓存（1.5–3×）**：线程相同 + prompt 版本相同 → 输出相同；"再总结一次这个线程"的命中率高得惊人。
4. **自建 + 量化（2–5×）**：仅在持续高利用率时成立；GPU 利用率低于 ~30% 时 API 更便宜，交叉点就是全部论证。
5. **离线批处理**：非延迟敏感的一律夜间批跑。
6. **限制输出**：250 → 120 token 直接把解码成本砍半；注意只有在压缩改变了输入输出比之后，解码才会成为主导项。

**速记**：1e8 规模下成本 = 调用数 × token 数 × 单价，先给可被挑战的假设表再算；10 倍靠输入压缩 + 小模型级联 + 缓存，自建量化只在利用率够高时划算。


---

## Amazon / AWS

### 1. 某客户托管在 Bedrock 上的工作负载成本过高。在不造成不可接受的质量损失的前提下，大幅削减推理成本。

<!-- EN: A customer's Bedrock-hosted workload costs too much. Cut inference cost dramatically without unacceptable quality loss. -->

**Source**: https://www.youtube.com/watch?v=jV2sCj4lHYk · https://outcomeschool.com/blog/llm-inference-optimization

**参考答案**

绝不要开口就说"换更便宜的模型"，你会立刻被反问"然后质量下降呢"。先讲**测量**，再讲阶梯。

**第 0 步——归因**：按 `模型 × 功能 × token 类型（输入/输出）× 租户` 拆开销。几乎总是：一两个功能、由**输入 token** 主导（30–60k token 的 RAG 上下文）、一个贵模型在服务一类根本不需要它的请求。

**第 1 步——零质量风险的便宜赢法：**

| 抓手 | 典型效果 |
| :--- | :--- |
| Prompt/前缀缓存（稳定系统提示词 + 工具 schema + 长文档前缀） | 缓存输入 token 省 40–90% |
| 裁剪检索（top-k 20 → 5、去重、去样板） | 输入常降 3–5× |
| 限制输出 + "用不超过 N 条回答" | 按 token 算解码才是贵的一侧 |
| 离线/异步任务走批处理 API | 非延迟敏感工作有大折扣 |
| 掐掉 agent 空转：减少步数、不重复调工具、无进展即停 | agent 工作负载 2–10× |

**第 2 步——真正的钱，但质量风险必须测量：**

1. **级联/路由**：便宜模型处理简单请求（分类器或 logprob/短回答启发式判定），约 15–25% 升级到前沿模型。要**按切片**评估，别看均值——伤害集中在升级切片。
2. **蒸馏**：把前沿模型的行为蒸馏进小模型，用你自己的流量微调；窄任务上通常保住大部分质量、价格只是零头。
3. **预留吞吐 vs 按需**：交叉点取决于利用率（大致是持续负载超过预留容量的 30–40%）；低于它，按需 + 缓存更赢。按他们真实的昼夜曲线算账。
4. **量化自建**（经 SageMaker/Bedrock 自定义模型）只在利用率高时；闲置 GPU 是账单上最贵的东西。

**第 3 步——治理**：按租户预算、token 配额、"每个成功任务的成本"作为一等看板指标，以及一条把成本变更绑到同一套质量评测集上的回归闸门。

**速记**：先按"模型×功能×输入/输出 token"归因，再走阶梯：前缀缓存 + 砍检索上下文 + 限制输出 + 离线批处理零风险；级联路由与小模型蒸馏才是数量级省钱，但必须按切片评估。


---

## Apple

### 1. 你需要在内存和功耗预算都很紧张的手机上运行一个约 3B 参数的语言模型。相比在数据中心服务同一个模型，需要做哪些改动？

<!-- EN: You need to run a ~3B-parameter language model on a phone with tight memory and power budgets. What changes versus serving the same model in a datacenter? -->

**Source**: https://outcomeschool.com/blog/cloud-vs-on-device-model-deployment · https://outcomeschool.com/blog/how-does-model-quantization-work

**参考答案**

这是完全不同的运行区间，不是"把同一个服务问题缩小一号"。

| 维度 | 数据中心 | 手机 |
| :--- | :--- | :--- |
| batch | 32–256（吞吐是目标） | **1**，没有任何可摊薄的东西 |
| 绑定约束 | GPU 算力 / HBM 容量 | **内存带宽 + 峰值功耗 + 散热包络** |
| 权重 | bf16/fp8，几百 GB 显存可用 | int4/int8，必须塞进与系统共享的几 GB **统一内存** |
| 延迟指标 | TTFT p95、聚合 tokens/s | 单个用户的 TTFT 与稳态 tokens/s，且预算按"App 启动级"算 |
| 失败模式 | 任务 OOM | 被系统 **jetsam 杀进程**；ANE/GPU 显存上限独立且更小 |
| 调度 | 单一后端、成熟 kernel | CPU / GPU / ANE 算子支持各不相同，必须按 NPU 支持的形状构图，否则回退 |
| 并发 | 多用户 | 前台 App 与后台守护进程争抢；模型可能在会话中途被驱逐 |

**设计上真正改变的几件事：**

1. **激进量化**：int4 仅权重量化（分组、逐通道 scale）是默认；3B 在 4 bit ≈ 1.5–2 GB，这是"能跑"与"被杀"的分界线。质量掉了就把激活与 KV cache 留高精度。
2. **batch=1 时词表与 embedding 尺寸很贵**：LM head 是每 token 一次 `[vocab × d]` 的 GEMM，128k 词表对 3B 模型在手机上占解码时间的可观比例。裁词表或用因式分解的头。
3. **显存风险在上下文，不在参数**：KV cache 随对话长度增长，要截断、驱逐或用滑窗注意力。
4. **prefill 在手机上是算力受限且痛的**：1–2k token 的 prompt 可能就要 200 ms 以上的 ANE 时间；保持 prompt 短，并缓存系统前缀。
5. **散热**：持续解码会降频。设计成突发式、可取消的生成，绝不要拿前 10 秒的成绩当数值。

**速记**：手机是 batch=1、受带宽/功耗/热限制，还要防 jetsam；权重必须 int4（3B≈1.5–2GB）、词表在 decode 上很贵、真正风险是 KV cache，持续生成会降频。

### 2. 解释训练后量化（post-training quantization）与量化感知训练（QAT）的区别。把权重压到 2–4 bit 时会出现什么问题，你如何恢复质量？

<!-- EN: Explain post-training quantization versus quantization-aware training. What breaks when you push weights to 2–4 bits, and how do you recover quality? -->

**Source**: https://outcomeschool.com/blog/how-does-model-quantization-work

**参考答案**

| | PTQ | QAT |
| :--- | :--- | :--- |
| 时机 | 训练之后，无反传 | 微调中/后，前向插入**假量化**节点，反向用直通估计器（STE） |
| 成本 | 几分钟到几小时，校准集几百条样本 | 全量微调成本，但常常只需一段短恢复训练 |
| 上限 | 仅权重量化到 ~4 bit 还不错，再往下崩得快 | 可达可用的 2–3 bit，int4 精度损失很小 |
| 依赖 | 与部署分布匹配的校准数据 | 训练数据 + 教师（蒸馏帮助很大） |

**2–4 bit 时崩在哪里：**

1. **离群通道**：少数激活/权重通道量级是其余的 10–100 倍，单一全局 scale 要么裁掉它们、要么把所有人的分辨率毁掉。修法：分组/逐通道 scale、**AWQ** 式激活感知缩放、SmoothQuant 式把激活离群迁移进权重。
2. **低位舍入误差跨层累积**，困惑度超线性上升——8 bit 几乎免费，4 bit 小心点没问题，3 bit 需要 QAT + 蒸馏，2 bit 对通用模型仍是研究问题。
3. **KV cache 与注意力比 MLP 权重更敏感**：朴素量化会毁掉长上下文行为，即使短 prompt 的评测看起来正常。
4. **kernel 不匹配**：NPU 没有原生加速的格式（例如非标准分组大小的 3 bit）会走"反量化再 GEMM"，可能比 8 bit **更慢**。

**恢复手段**：embedding + LM head + 首尾 block 保留高精度（混合精度）；PTQ 用 GPTQ/AWQ；QAT 从 fp16 教师蒸馏；并用**任务指标而非困惑度**评估——困惑度 0.3 的损失，在结构化抽取上可能是 5 个点。

**速记**：PTQ 靠校准、便宜，8→4 bit 基本无损；QAT 用假量化 + STE 才能到 2–3 bit。崩的原因是离群通道、跨层误差累积、KV cache 更敏感、非标准位宽没有原生 kernel。

### 3. 估算一个 3B 端侧模型在 4k 上下文下的 KV cache 显存占用，并说出能压缩它的手段。

<!-- EN: Estimate the KV-cache memory for a 3B on-device model at 4k context, and name the levers that shrink it. -->

**Source**: https://outcomeschool.com/blog/kv-cache-in-llms · https://outcomeschool.com/blog/kv-cache-compression

**参考答案**

先给公式，再代入一组**典型**的 3B 配置（假设要说出口——答案是这套算术，不是那个常数）：

```
bytes = 2 (K and V) × layers × kv_heads × head_dim × seq_len × batch × bytes_per_element
```

| 假设 | 取值 |
| :--- | :--- |
| 层数 | 28 |
| kv_heads × head_dim（GQA） | 8 × 128 = 1024 |
| 序列长 | 4096 |
| dtype | fp16（2 B） |

`2 × 28 × 1024 × 4096 × 2 = 469,762,048 B ≈ 448 MiB`。

能加分的健全性检查：若同一模型用**完整 MHA**（32 个 kv 头而不是 8），是 4 倍 → 约 1.75 GiB，这在手机上不叫缓存，叫整个 App 的预算。**是 GQA 让端侧 4k 上下文成为可能。**

**杠杆，按效力排序：**

1. **更少的 KV 头**（MQA/GQA）——架构级，对主导项有 4–8 倍效果；事后很难改，属训练期决策。
2. **更低位宽**——int8 KV 约 2 倍；int4 配分组 scale 约 3–4 倍，但要在长上下文上测精度，逐 token 离群很难受。
3. **更短的有效上下文**——滑窗注意力，或用滚动摘要驱逐旧轮次。最便宜，也是产品团队最常忘、直到 jetsam 杀进程才想起来的一条。
4. **跨并发请求/功能共享前缀**——系统提示词只留一份，而不是 N 份。
5. **分页/非连续分配**——避免碎片导致的超额预留，在内存受限设备上与绝对大小同样重要。
6. **压缩/驱逐启发式**（H2O、保留 attention sink）——留前几个 token + 近期窗口；质量风险在"用户引用了 30 轮之前说过的内容"。

**速记**：`2×层数×kv_heads×head_dim×seq×batch×字节`，典型 3B/28 层/GQA 8×128/fp16/4k ≈ 448 MiB；换 MHA 就是 1.75 GiB——GQA 是端侧 4k 的前提。

### 4. 你的端侧功能首 token 延迟（TTFT）是 1.8 s。讲讲你会如何诊断并修复它。

<!-- EN: Time-to-first-token for your on-device feature is 1.8 s. Walk me through diagnosing and fixing it. -->

**Source**: https://www.youtube.com/watch?v=XD8DD4cEHu0

**参考答案**

不要猜——先用打点把 1.8 s **分段**，因为端侧的分布和数据中心完全不同：

| 环节 | 端侧典型占比 | 表现 |
| :--- | :--- | :--- |
| App/UI + 首次冷启动路径 | 100–600 ms | 会话建立、守护进程启动 |
| **模型加载 / 权重换页** | 0–900 ms（双峰！） | 从 flash mmap 换入、ANE 编译，仅首次 |
| 分词 + prompt 组装 | 20–80 ms | 长系统提示词、聊天历史重新序列化 |
| **Prefill** | 300–900 ms | 算力受限，∝ prompt 长度，在手机 NPU 上每 token 成本是解码的 10–50 倍 |
| 采样 + 首个 detokenize | 10–40 ms | |
| 渲染 / 流式界面挂载 | 50–200 ms | 首 token 已存在但屏幕上是空的 |

**诊断顺序**

1. 是不是双峰？比较冷启动与热启动。只有第一次请求 1.8 s，那是**权重换页/ANE 编译**，不是 prefill——用 mmap + `madvise`、保持模型常驻、App 启动时做一次预热。
2. 打印真实 prompt token 数。900 token 的系统提示词在手机 NPU 上常是 600 ms 以上的 prefill。裁掉它，并**跨轮缓存前缀 KV**。
3. 看热状态与后台 CPU 争抢——在插着电、刚装完的机器上测出来的数字，你永远复现不了。
4. 确认图有没有回退到 CPU。一个不支持的算子会把整个子图静默搬离 NPU，你看到 4 倍 prefill 却没有任何报错。

**修法，按性价比排序**：缓存系统提示词的前缀 KV（最大单项收益，常 30–50%）、模型常驻 + 预热、缩短/截断 prompt、分块 prefill 并先流式渲染骨架、把"有用内容"与"完整生成"解耦——第一句一出现就渲染。

**速记**：1.8s 必须先按阶段打点；端侧常见是双峰——只有第一次慢就是权重换页与 ANE 编译；最大单项收益是缓存系统提示词前缀 KV + 模型常驻。


---

## NVIDIA

### 1. 你想在单张 80 GB GPU 上服务一个 70B 参数模型。讲讲它是否放得下，以及你预期的单流 tokens/sec 是多少。

<!-- EN: You want to serve a 70B-parameter model on a single 80 GB GPU. Walk me through whether it fits and what single-stream tokens/sec you'd expect. -->

**Source**: https://outcomeschool.com/blog/how-does-model-quantization-work · https://outcomeschool.com/blog/kv-cache-in-llms

**参考答案**

**第 1 步——权重**：`70B × 每参数字节`：

| 精度 | 权重 | 80 GB 上的结论 |
| :--- | :--- | :--- |
| FP32 | 280 GB | 不行 |
| BF16/FP16 | 140 GB | 不行 |
| FP8/INT8 | 70 GB | 能装，剩 ~8 GB，几乎没 KV 空间 |
| INT4（AWQ/GPTQ） | ~35–38 GB | 舒服地装下，~40 GB 留给 KV 与 workspace |

**第 2 步——KV cache**：每 token `2 × 层数 × kv_heads × head_dim × 字节`。按 Llama-3-70B 形状（80 层、8 个 KV 头、head_dim 128、FP16）约 327 KB/token → 40 GB ≈ **12 万缓存 token**。注意是 GQA 让这个数字能活得下来；用 MHA 会再大 10 倍。

**第 3 步——解码速度**：单流是带宽受限：`tok/s ≈ BW / 每步读取字节数`。

- H100 SXM、约 3.35 TB/s、INT4 权重 → 每步读 35 GB → **理论 ~95 tok/s**；实际带宽利用率 50–60% 时是 **40–60 tok/s**。
- INT8 → 每步 ~70 GB → 实际 ~25–35 tok/s。
- FP16 根本装不下；切分后大约 ~45 tok/s，还要减掉跨 GPU 同步。

**第 4 步——什么会打翻它**：上下文长度（KV 线性增长）、batch > 1（权重被摊薄，但 KV 与激活内存开始咬人）、以及长 prefill（算力受限，是另一套数学——prefill 才是 GPU 真正忙起来的地方）。

**速记**：70B 单卡 80G 只有 INT4 舒服（35G 权重 + 40G KV，约 12 万 token）；decode 是带宽瓶颈，tok/s ≈ 带宽/每步读字节，H100 INT4 实测 40~60 tok/s。

### 2. TensorRT / TensorRT-LLM 究竟对模型做了什么让它变快，以及什么情况下它帮不上忙？

<!-- EN: What does TensorRT / TensorRT-LLM actually do to a model to make it faster, and when will it not help? -->

**Source**: https://outcomeschool.com/blog/how-does-tensorrt-llm-work

**参考答案**

它做三层工作，而知道加速来自哪一层才是真正的面试信号。

| 层 | 做了什么 | 典型收益 |
| :--- | :--- | :--- |
| **图** | 常量折叠、死代码消除、cast 消除、**kernel 融合**（LayerNorm+残差、QKV GEMM、激活 epilogue） | 1.2–1.6 倍，主要来自消除内存往返 |
| **权重** | 布局变换（交错 QKV、为 tensor core 预打包）、**量化**（FP8/INT8/INT4 AWQ/GPTQ）、逐通道 scale | 1.5–2.5 倍，主要是带宽 |
| **运行时** | CUDA Graph（消掉逐 kernel 启动开销）、**在飞/continuous batching**、paged KV cache、定制 FlashAttention 类 kernel、插件化 all-reduce | 吞吐 2–10 倍；小 batch 时最关键 |

**什么时候不帮忙**（要具体，这是"读过博客"与"真出过货"的分界）：

1. **形状覆盖不到**。引擎按 min/opt/max profile 构建，超出 profile 的请求形状会回退到慢 kernel——真实回归常常是因为有人对着 4096 的 profile 发了 4097 个 token。
2. **不支持的算子**打断融合，把你丢进回退子图，有时比 PyTorch 还慢。
3. **小模型下瓶颈是 launch 开销**——CUDA Graph 能治，但整个 step <50 µs 时你在和 Python/主机开销搏斗，该修的是服务循环而不是引擎。
4. **算力受限的训练**，或墙钟时间被分词、网络、工具调用主导的场景。
5. **每次重建都绑定 GPU 架构**。T4 的引擎跑不了 H100，可移植性是实打实的运维成本。

**速记**：分三层——图优化（融合/常量折叠）、权重（布局 + 量化）、运行时（CUDA Graph、continuous batching、paged KV、定制 attention）；它救不了 shape 超 profile、不支持的算子、瓶颈不在 GPU 的场景。

### 3. 为服务一个 405B 参数的稠密模型设计并行策略。TP、PP、EP 各自放在哪里、为什么？

<!-- EN: Design the parallelism strategy for serving a 405B-parameter dense model. TP, PP, EP: what goes where and why? -->

**参考答案**

先纠正前提：**EP 不适用——这是稠密模型**。专家并行是 MoE 概念；如果有人对稠密 405B 问"EP 放哪"，他在测你会不会硬编一个答案。

算规模：405B BF16 = **810 GB 权重**。在 80 GB H100 上，光权重就要 ≥11 张卡，还要加 KV cache、激活和 CUDA Graph。

| 维度 | 放置 | 原因 |
| :--- | :--- | :--- |
| **TP** | 节点内，度 8（NVLink） | TP 每层两次 all-reduce，走 PCIe/IB 就成了瓶颈。能避免就绝不让 TP 跨节点 |
| **PP** | 跨节点，配合交错/虚拟 stage | PP 只在 stage 边界传激活——慢链路上也便宜。按实测层时间而非层数来平衡 |
| **DP** | 副本集，扩吞吐 | 倍增容量，不切模型 |
| **CP / 序列并行** | 长上下文 | 切序列不切权重；当单请求 KV 超过单卡预算时需要 |

对**推理**：单台 8×H100（共 640 GB）BF16 装不下（810 GB 权重）。FP8 把权重压到 ~405 GB，剩 ~200 GB 给 KV——可行但紧，所以多数部署用两节点 PP=2 × TP=8。对**训练**：还要加 ZeRO/FSDP 分片优化器状态（Adam FP32 ≈ 16 字节/参数 ≈ 6.5 TB）——它比模型本身更主导集群规划。

要点名的失败模式：PP 气泡空转、小 per-GPU batch 下 TP all-reduce 延迟、stage 不均衡造成掉队者，以及真正限制并发的是 KV cache。

**速记**：先纠正前提——dense 模型没有 EP；TP 只在 NVLink 节点内（8 卡），PP 跨节点（只传激活），DP 扩吞吐，长上下文加 CP；405B BF16 权重 810GB，真正约束是 KV 与优化器状态。


---

## Tesla

### 1. 车载算力预算固定。讲讲你如何量化并剪枝一个视觉模型，同时不丢失小目标的召回率。

<!-- EN: The onboard compute budget is fixed. Walk me through quantizing and pruning a vision model without losing recall on small objects. -->

**Source**: https://outcomeschool.com/blog/how-does-model-quantization-work

**参考答案**

顺序很重要：**先量化能容忍的部分，再做结构化剪枝，最后用蒸馏恢复。**

| 步骤 | 做法 | 为什么保护小目标 |
| :--- | :--- | :--- |
| 1 | 逐层敏感度分析：INT8 PTQ，在**小目标切片**（<32 px）上测召回，而不是聚合 mAP | 小目标特征能量低；量化误差相对于特征幅值，它们最先死 |
| 2 | **首尾层、检测头、所有 norm/softmax 保持 FP16** | 这里的混合精度只花约 5% 的加速，却挡住大部分损伤 |
| 3 | 优先**逐通道**权重 scale；PTQ 掉超过 0.5 个点就上 QAT | 逐 tensor 缩放正是掉 3 个点的地方 |
| 4 | 剪**通道/滤波器（结构化）**，绝不剪非结构化稀疏 | 没有 2:4 感知 kernel 的非结构化稀疏在真实硬件上零加速；2:4 也要求形状匹配 |
| 5 | **不要降输入分辨率。** 剪最深的 stage / neck | 分辨率是小目标召回的单一最大杠杆 |
| 6 | 从 FP32 教师蒸馏到剪枝后的学生，含 neck 上的特征级损失 | 一天训练通常能恢复 50–80% 的损失 |

同时要量化**后处理**（NMS/decode），否则它的 FP32 转换会把收益吃回去；并且要重测端到端延迟——一个 2 倍 MAC 减少但变成内存受限的模型，可能什么都没改善。

**速记**：顺序是先量化（后端敏感层保 FP16、逐通道 scale）、再结构化剪枝（只剪通道，别指望非结构化稀疏真加速）、最后用 FP32 teacher 蒸馏恢复；评估必须看 <32px 切片召回。


---

## 字节跳动

### 1. 以 8x7B 模型为例：KV cache 和权重各占 GPU 显存的大致比例是多少？还有哪些东西占用显存？

<!-- EN: Take an 8x7B model: roughly what share of GPU memory goes to KV cache versus weights? What else occupies memory? -->

**Asked at**: 字节跳动
**Source**: https://outcomeschool.com/blog/kv-cache-in-llms

**参考答案**

权重是固定的，KV cache 随 batch × 上下文增长，所以占比完全取决于你对流量的假设。

权重：8 × 7B ≈ 56B 参数，fp16 下约 **112 GB**，再加 1~2 GB 的缓冲区。8×80GB = 640 GB 的机器上，剩约 520 GB 给其他所有东西。

每 token 的 KV：`2 · L · n_kv · d_head · bytes`。以 Mixtral 风格的 7B 专家（32 层、32 头、`d_head=128`、fp16、无 GQA）计算约 **512 KB/token**；8k 上下文就是每条序列约 4 GB，于是约 120 条并发的 8k 序列就能把剩余显存吃满。

其他占用，通常被忽略：

| 占用者 | 量级 | 备注 |
| :--- | :--- | :--- |
| 激活 / workspace | 1~5 GB | 取决于 prefill batch 与 TP |
| CUDA graph + kernel | 1~3 GB | 按 batch 桶静态捕获 |
| 显存碎片 | 空闲量的 10~20% | PagedAttention 存在的理由 |
| 专家权重副本 | 取决于模型 | EP/TP 决定复制几份 |
| 前缀缓存池 | 可配置 | 共享、可驱逐，与 KV 争空间 |

结论性的一句话：MoE 模型上**权重主导**，KV 只是小项；长上下文的稠密服务里比例反过来，KV 成为硬约束。

**速记**：56B 权重 fp16 约 112GB，KV 每 token 约 512KB（8k 上下文≈4GB/条）；还有激活、CUDA graph、碎片和前缀缓存池。MoE 是权重主导，长上下文稠密服务则反过来。

### 2. 前缀缓存命中了，但 KV block 已被下放到 SSD。取回路径是怎样的，回传到 GPU 的过程中是否必须经过主机内存？

<!-- EN: A prefix cache hits, but the KV blocks have been offloaded to SSD. What is the fetch path, and does it have to go through host memory on the way back to the GPU? -->

**Asked at**: 字节跳动
**Source**: https://outcomeschool.com/blog/how-does-prompt-caching-work

**参考答案**

路径是：**SSD →（页缓存 / 主机 DRAM 暂存）→ pinned 主机缓冲 → H2D 拷贝 → GPU HBM**，然后重映射块表，prefill 跳过已缓存的前缀。

必须过主机内存吗？对普通 NVMe 来说是的——多数服务器拓扑下设备无法直接 DMA 进显存，中转缓冲不可避免。两种绕法：

| 路径 | 机制 | 适用 |
| :--- | :--- | :--- |
| GPUDirect Storage | NVMe 直接写入 GPU HBM，跳过中转缓冲 | 需要支持的 NIC/PCIe 拓扑与驱动 |
| 无 GDS 的优化 | 大块 pinned 暂存 + `cudaMemcpyAsync` 重叠拷贝 | 默认做法 |

决定"这次命中值不值"的成本模型：单块 NVMe 约 3~7 GB/s，10 万 token 前缀的 KV 约 50 MB，I/O 约 10 ms；但 H2D 拷贝和块表重映射还要叠加延迟。同一前缀的第二次命中便宜得多，因为页缓存已经持有数据——所以**第二次命中才是好命中**。

要指出的失败模式：若请求在等待取回时发生驱逐，两个请求会把同一批块来回抖动。必须给离载块加引用计数，并定死"绝不驱逐正在传输的块"，否则会出现看起来像随机延迟毛刺的重算风暴。

**速记**：SSD→页缓存/主机暂存→pinned 缓冲→H2D→HBM，普通 NVMe 必须过主机内存（GPUDirect Storage 才能绕开）；第二次命中才便宜，离载块要被引用计数保护否则抖动。

### 3. 在 vLLM 中，PagedAttention 和连续批处理（continuous batching）各自解决了什么问题？

<!-- EN: What do PagedAttention and continuous batching each solve in vLLM? -->

**Asked at**: 字节跳动
**Source**: https://outcomeschool.com/blog/how-does-vllm-work

**参考答案**

两者在不同层次解决不同问题，缺一不可。

| | PagedAttention | 连续批处理 |
| :--- | :--- | :--- |
| 层次 | **显存分配器** | **调度器** |
| 问题 | KV 必须是按最大长度预留的连续块 | 一个 batch 要等最长序列结束才收新请求 |
| 机制 | 固定大小块 + 每序列块表，类似 OS 虚拟内存 | iteration 级准入：结束的序列立刻退出，新序列每步可加入 |
| 省下 | 内部与外部碎片，并支持共享与换出 | 长短不齐的生成过程中的 GPU 利用率 |

值得记住的数字：碎片化过去会浪费 60~80% 的 KV 显存，块分配在多数负载下能压到个位数百分比。

关键交互：分页之所以有效，是因为调度器能以块为粒度换出或抢占；连续批处理之所以有效，是因为分配器能按块增长序列。两者合起来，把 decode 从"整批等最坏情况"变成"吞吐 ≈ 各序列独立 token 速率之和"。

代价也要说：块表给注意力 kernel 加了一层间接寻址（必须定制 kernel）；激进的连续批处理会抬高短请求的 TTFT 抖动，除非配合 chunked prefill 或优先级调度。

**速记**：PagedAttention 是**显存分配器**（块表 + 共享，消除 60~80% 碎片），连续批处理是**调度器**（逐 iteration 进出）；两者互为前提，代价是注意力需专用 kernel、短请求 TTFT 抖动。

### 4. 解释投机解码（speculative decoding）。它对哪些模型没有帮助？

<!-- EN: Explain speculative decoding. Which models does it not help? -->

**Asked at**: 字节跳动
**Source**: https://outcomeschool.com/blog/speculative-decoding

**参考答案**

用小模型起草 `k` 个 token，目标模型在**一次**前向里全部验证，接受最长正确前缀并对首个被拒位置重采样。因为验证是一次批量前向，最多 `k` 个 token 只付一次目标模型成本。保证是：配合标准的拒绝采样修正，输出分布与目标模型**完全一致**——它是纯延迟技巧，不是近似。

加速比 ≈ `接受 token 数 / (1 + 起草成本/验证成本)`，所以只有接受率高且起草便宜时才有效。

| 不起作用的情形 | 原因 |
| :--- | :--- |
| 目标模型本来就小 / batch 已经很大 | decode 是带宽受限且接近饱和，起草只增加延迟 |
| 高熵或创意型输出 | 接受率跌到 50~60%，比值 ≈1 |
| 起草与目标分布差异大 | 被拒 token 也要付验证成本 |
| 严格低延迟单流 + 慢起草模型 | 起草成本是串行的，占主导 |
| 极高 QPS 下的大 batch | 你已是吞吐受限，投机解码几乎无收益还占 KV 显存 |

反直觉的一点：大 batch 下投机解码常常**降低**吞吐，因为验证把算力浪费在被拒 token 上，而你本来就受算力限制。它是单流延迟工具。音视频、量化过的目标模型、以及域偏移的起草模型（比如用代码草稿去服务聊天目标）是典型的无效场景。

**速记**：小模型起草、大模型一次前向并行验证，输出分布严格不变；收益取决于接受率/起草成本比；大 batch 已算力饱和时它反而降吞吐，属单流延迟优化。

### 5. 为什么 prefill/decode 分离会提升整体吞吐，以及在什么情况下对你来说值得这么做？

<!-- EN: Why does prefill/decode disaggregation raise overall throughput, and when is it worth it for you? -->

**Asked at**: 字节跳动
**Source**: https://outcomeschool.com/blog/prefill-decode-disaggregation

**参考答案**

因为 prefill 与 decode 的**瓶颈正好相反**，混布时会互相干扰。

| 阶段 | 受限于 | 理想的硬件姿态 |
| :--- | :--- | :--- |
| Prefill | 算力（大 GEMM、高算术强度） | 高 SM 占用、大 batch |
| Decode | 显存带宽（逐 token、权重流式读） | 低算力占用、延迟敏感 |

混布时，一次长 prefill 会卡住同批所有正在 decode 的序列——就是那种 TPOT 正常但 TTFT p99 突刺的经典现象。拆分后各池独立调优（chunk 大小、batch、甚至不同 GPU 型号），KV 通过 NVLink/RDMA 传输。

值得做的条件：

- **TTFT 与 TPOT 的 SLO 互相冲突**，且你在服务长输入短输出（大 system prompt 的对话、RAG）。
- 利用率偏低，因为 prefill 突发不频繁但很猛。
- 能承担 KV 传输：1 万 token 前缀是数十 MB，走 RDMA 是个位数毫秒。

不值得的情况：集群太小（失去灵活性、传输成本吃掉收益），或 prompt 短到 prefill 根本不是可见停顿。厂商宣称 2 倍吞吐的基准通常假设长 prompt + 严格 TPOT，那是最佳情形。

**速记**：prefill 算力受限、decode 带宽受限，混部时长 prefill 会卡住整批 decode 造成 TTFT 尖刺；拆池可各自调优，代价是 KV 传输；小集群或短 prompt 不值得。


---

## 腾讯

### 1. vLLM 的核心思想是什么？PagedAttention 和连续批处理各自解决了什么问题？

<!-- EN: What is the core idea behind vLLM? What do PagedAttention and continuous batching each solve? -->

**Asked at**: 腾讯
**Source**: https://outcomeschool.com/blog/how-does-vllm-work

**参考答案**

核心思想：**把 KV cache 当操作系统内存问题来处理**，再在这个分配器之上按 token 粒度调度。

| 组件 | 解决的问题 | 机制 |
| :--- | :--- | :--- |
| PagedAttention | KV 必须按最大长度预留成一块连续内存 → 浪费 60~80% | 固定大小 KV 块 + 每序列块表；物理页可不连续 |
| 连续批处理 | batch 要等最长序列，短序列让 GPU 空转 | iteration 级调度：结束的序列退出，新序列每步加入 |
| 前缀缓存 | 相同的 system prompt 每个请求都重算 | 基于哈希的块共享；相同前缀指向同一批物理块 |
| Chunked prefill | 一次长 prefill 卡住所有 decode | 把 prefill 切块，与 decode 步骤交错 |

值得点明的依赖关系：分页让**共享**（前缀缓存）和**抢占**（按块换出/重算）成为可能；连续批处理让长短不齐的 batch 变得可负担。单独任何一个都拿不到那个标志性吞吐。

代价：带块表间接寻址的定制注意力 kernel（不能原样套用 FlashAttention 路径）、激进批处理下短请求 TTFT 方差变大、以及碎片从序列级转移到块表级。还有一点：`gpu_memory_utilization` 太高会在突发时 OOM，太低则浪费——要按真实流量调，不要按 benchmark 调。

**速记**：核心是把 KV cache 当操作系统内存问题——分页块表消除 60~80% 碎片并支持共享/抢占，连续批处理做逐 iteration 调度；两者互为前提，代价是定制 kernel 与短请求 TTFT 抖动。

### 2. FlashAttention v2 相比 v1 的主要改进有哪些？

<!-- EN: What were the main improvements in FlashAttention v2 over v1? -->

**Asked at**: 腾讯
**Source**: https://outcomeschool.com/blog/decoding-flash-attention

**参考答案**

两者都是精确算法（无近似）；v2 是 **IO 与并行度**的重写，数学没变。

| 改进 | v1 | v2 |
| :--- | :--- | :--- |
| 工作划分 | 按 batch × head 并行，每个线程块负责一块行 | 还按**序列长度**并行——块数多得多，短序列上占用率更好 |
| 循环顺序 | K/V 在外层、Q 在内层 → 每步都要重缩放输出 | Q 在外层、K/V 在内层 → 内层循环**不需要重缩放输出**，共享内存流量更少 |
| 非矩阵乘 FLOPs | 相对偏高 | 降低；softmax 重缩放折进 epilogue |
| warp 级布局 | 跨 warp 切分 K 维需要共享内存做归约 | 每个 warp 独占一块行 → softmax **无需跨 warp 通信** |
| 结果 | 相对标准注意力约 2× | 报告配置下相对 v1 约 2~4×，长序列更多 |

要讲清的核心直觉：注意力是显存带宽受限的，因为 `n×n` 分数矩阵根本不需要写回 HBM。分块把它留在 SRAM 里；在线 softmax（维护 running max + 重缩放累加器）让分块保持精确。

v3（Hopper）加了异步 `wgmma` 与软/矩阵乘重叠——那是另一套机制（生产者/消费者 warp 特化），不只是重新调参。实用提醒：现代栈里你很少直接调这些；`scaled_dot_product_attention` 会挑后端，而 varlen 与因果掩码正是实测加速比与论文数字不同的地方。

**速记**：v2 是 IO 与并行度重写（按序列维并行、Q 外层 K/V 内层免去输出重缩放、warp 各持行切片免跨 warp 通信），数学精确；v3 靠 Hopper 异步 wgmma 与 warp 特化，是另一套机制。

### 3. 在训练数据进入训练任务之前，你如何做质量检查？

<!-- EN: How do you quality-check training data before it goes into a run? -->

**Asked at**: 腾讯
**Source**: https://outcomeschool.com/blog/llm-evaluation

**参考答案**

跑一条**分级闸门**，每一级都便宜且杀掉一类不同的失败；绝不要依赖一个"质量分"。

| 闸门 | 检查 | 杀掉什么 |
| :--- | :--- | :--- |
| 去重 | 精确哈希 → MinHash/LSH 近似去重 → 后缀数组找长重复 | 记忆化、抓取样板被放大 10 倍权重 |
| 启发式 | 长度、符号比例、语种识别、重复 n-gram 比例、模板/样板正则 | 导航垃圾、截断的 HTML、"点这里"页面 |
| 模型打分 | 质量/教育价值分类器或小裁判；对参考模型算困惑度做过滤 | 通顺但无用的文本 |
| 污染检查 | 对**所有**评测集做 n-gram 重叠（包括以后才用的） | 虚高的 benchmark 数字 |
| 安全/PII | 毒性、PII 正则 + NER、授权元数据 | 法律与合规风险 |
| 人工 | 每个来源抽 200~500 篇人读 | 以上全部漏掉的东西 |

值得作为经验引用的数字：去重通常能删掉原始抓取的 30~60%；质量分类器一般只保留 5~15%。反直觉的一点：**裁判模型打的分与长度正相关**，所以一定要做长度归一，否则你训出一个话痨模型。

要指名的失败模式：污染检查只跑一次就忘了——公开评测集会持续泄漏进后续每次抓取。维护一个覆盖你可能用到的一切 benchmark 的滚动 n-gram 索引，并把过滤后的数据集连同哈希一起版本化，让每次训练可复现。

**速记**：分级闸门——精确/近似去重、启发式过滤、模型打分、与所有评测集的污染检测、PII 与授权、人工抽检；去重通常砍掉三到六成原始网页；注意质量分与长度相关，必须长度归一，否则训出"话多"模型。

### 4. 为什么 LLM 是一个 token 一个 token 地流式输出，底层用的是什么协议？

<!-- EN: Why do LLMs stream token by token, and what protocol sits underneath? -->

**Asked at**: 腾讯
**Source**: https://outcomeschool.com/blog/how-does-token-streaming-work

**参考答案**

因为模型本来就是每步解出一个 token，**把它们缓存起来只会白增延迟、没有任何收益**。用户感知的延迟主要由首 token 时间决定，不是总时长。

按栈从下往上：

| 层 | 机制 |
| :--- | :--- |
| 模型 | 自回归解码，每步产出一个 token（投机解码时是几个） |
| 服务端 | 浏览器场景默认用 SSE（`text/event-stream`）——单向、纯 HTTP、对代理友好 |
| 服务端（备选） | 需要双向时（打断、插话）用 WebSocket，或 HTTP chunked transfer |
| 载荷 | OpenAI 风格 `data: {"choices":[{"delta":{"content":"…"}}]}`，以 `[DONE]` 收尾 |
| 边缘 | 代理/CDN 必须关掉缓冲，否则流会静默失效 |

生产中最常咬人的四种失败：

1. **代理缓冲**——Nginx `proxy_buffering on` 或开了压缩的 CDN 会把 chunk 合并；用户先是什么都看不到，然后一次性全到。
2. **token 边界切断**——直接按 token 字符串推流会切断多字节字符和 markdown；要在 UTF-8 安全边界切（记住一个 token 不等于一个字符）。
3. **流中错误**——响应头已发出后无法返回 5xx；只能在流内发一个错误事件然后关闭。
4. **取消**——客户端断开必须中止生成，否则你在为没人要的答案烧 GPU。token 计费出错也常常出在这里。

**速记**：模型本来就逐 token 解码，缓冲只会白增延迟，而用户感知看 TTFT；底层多用 SSE（双向/打断才用 WebSocket）；坑在代理缓冲把流合并、按 token 切会切断多字节字符、流内错误无法回 5xx、断连不取消会白烧 GPU。

### 5. 在一个 API 服务中，你会从哪些方面入手降低响应延迟？

<!-- EN: Where do you look to cut response latency in an API service? -->

**Asked at**: 腾讯
**Source**: https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization

**参考答案**

先把延迟拆开（`TTFT + token 数 × TPOT`），再打最大的那一项——多数人优化错了地方。

| 项 | 手段 | 典型效果 |
| :--- | :--- | :--- |
| 排队 | 准入控制、优先级队列、按队列深度而非 CPU 自动扩容 | 常常是最大的单项收益，且通常在看板上不可见 |
| TTFT | 对共享 system prompt 做前缀缓存、缩短 prompt、chunked prefill | prompt 大部分是样板时，能砍掉 TTFT 的 30~70% |
| TPOT | 加大 batch、GQA/MLA、fp8 KV、投机解码 | decode 是带宽受限，批处理才是杠杆 |
| 网络/出口 | 同区域部署、连接复用、流式返回 | 几十毫秒，免费 |
| 后处理 | 别等完整答案再渲染；直接流式 | 感知而非真实 |

真正有效的顺序：按接口分别测 p50/p95/p99，然后看**排队等待 vs 计算时间**。以我的经验，在一个看起来健康的服务里，排队等待占 p99 的 40~70%，任何 kernel 优化都救不了。

要指名的反模式：一次性把所有优化全开（无法归因，而且高 batch 下投机解码可能反而有害）；在 decode 已经带宽受限时为了省显存去量化权重、白丢精度；以及 SLO 是 p99 却去优化平均延迟。

何时收手：如果你是高 batch、低排队等待的算力受限状态，再往下就是换模型，而不是调服务。

**速记**：先把延迟拆成 `TTFT + token×TPOT` 并按 p50/p99 分别看**排队等待 vs 计算**——排队常占 p99 的 40~70%，改 kernel 无用；前缀缓存砍 TTFT，批处理砍 TPOT，别一次全上否则无法归因。


---

## 京东

### 1. 你用过哪些推理优化工具（TensorRT、vLLM）？它们如何工作，你的使用体验如何？

<!-- EN: Which inference optimisation tools have you used (TensorRT, vLLM)? How do they work and what was the experience? -->

**Source**: https://outcomeschool.com/blog/how-does-vllm-work and https://outcomeschool.com/blog/how-does-tensorrt-llm-work

**参考答案**

| | vLLM | TensorRT-LLM |
| :--- | :--- | :--- |
| 核心思路 | PagedAttention（分页 KV 块）+ 连续批处理 | 预编译引擎、融合 kernel、in-flight batching |
| 优势 | 上手快、兼容 HF、采样灵活 | 固定形状+固定 GPU 上峰值吞吐/延迟最好 |
| 劣势 | 峰值性能稍差、Python 调度开销 | 每个模型/GPU/精度都要 build，一改就重编 |
| 运维现实 | 一个镜像跑多个模型 | 引擎缓存当制品，锁死 CUDA/驱动 |

vLLM 真正赢在三处，按贡献排序：**连续批处理**（跑完的序列立刻让出槽位，长短混跑时吞吐不塌）、**分页 KV**（不再按最大长度预留 2× 冗余，并发会话装得更多），其次才是前缀缓存。

线上学到的：

- **前缀缓存只在前缀真共享时才划算**（system prompt + 工具 schema）。每个请求都带独特开场白的话，命中率接近零，白付哈希成本。
- **KV cache 量化是最便宜的两倍并发**——质量预算允许就上，用自己的评测集测，别看 perplexity。
- **TensorRT-LLM 赢在固定形状的批量离线作业**；vLLM 赢在长尾流量与模型频繁迭代。多数团队最后是线上 vLLM、批量管线 TensorRT-LLM。

**速记**：vLLM 赢在 PagedAttention + 连续批处理、落地快；TensorRT-LLM 赢在编译后峰值性能，但每个模型/GPU/精度都要重建；前缀缓存只在 system prompt 真共享时有效。

### 2. 你如何处理数亿条记录：清洗、预处理、高效加载，以及你踩过的坑？

<!-- EN: How do you handle hundreds of millions of records: cleaning, preprocessing, efficient loading, and the traps you hit? -->

**参考答案**

当成数据管线问题，不是建模问题。

| 阶段 | 做法 | 坑 |
| :--- | :--- | :--- |
| 摄入 | 列式（Parquet）+ 按日期/租户分区，别用裸 JSONL | 小文件比大文件更杀吞吐 |
| 清洗 | 规则过滤优先（长度、语种、模板哈希、PII 正则），模型过滤其次 | 模型过滤贵 100 倍且不确定——过滤器要版本化 |
| 去重 | 近似重复用 MinHash/LSH，完全重复用哈希 | 去重顺序会改变数据集，必须记录 |
| 分词 | 离线 tokenize 并缓存，pack 到定长 | tokenizer 是 CPU 密集型——远早于 IO 成为瓶颈 |
| 加载 | 流式分片、预取、按 seed 确定性 shuffle | 打乱 1 亿索引数组要几 GB；先打乱分片顺序再片内打乱 |
| 配比 | 按领域/长度桶采样 | 随机切分会把同一商品泄漏进训练和评测 |

零售规模的真实数字：1 亿短文档大约 10¹¹ token，tokenize 后的产物以**TB** 计，所以「处理一次、永久缓存」是硬要求；每个实验都重 tokenize 会浪费好几天。并且永远保留一层不可变的原始层：清洗规则错了要能重算，而不用重爬。

**速记**：当成数据工程——Parquet 分区、规则过滤先于模型过滤、MinHash 去重并记录顺序、离线 tokenize 缓存（CPU 先成瓶颈）、分片内 shuffle；原始层不可变，规则改了能重算。

### 3. 一个模型在生产环境中表现退化，质量变差或延迟升高。讲讲完整的诊断过程。

<!-- EN: A model degrades in production, worse quality or higher latency. Walk me through the full diagnosis. -->

**参考答案**

先拆开两个症状——它们通常不同因。

**延迟**：分解，不要看平均：

| 分段 | 查什么 |
| :--- | :--- |
| 排队 | 并发 vs 容量、KV 压力、batch 是否打满 |
| Prefill | prompt 变长了？检索多塞了块？ |
| Decode | 输出长度漂移、采样参数变了 |
| 后处理 | reranker、工具调用、网络 |

**质量**：按可能性排序：

1. **输入分布漂移**（新品、新活动、新人群）——看输入长度/embedding 漂移，不只看日志。
2. **依赖变了**——检索索引用新 embedding 模型重建、prompt 模板被改、供应商静默升级了模型版本。要 diff **实际生效的** prompt，不是仓库里那份。
3. **反馈闭环**——模型开始推它以前推过的东西；看商品覆盖度，不能只看准确率。
4. **模型本身**——排最后；基座模型没有部署很少自己变差。

方法：每周用固定 golden set 跑生产配置回放，保持一条平基线；再用影子流量与旧配置做二分。没有回放装置，每次事故都是考古。

**速记**：先拆两症状——延迟按队列/prefill/decode/后处理分解，质量按"输入漂移 → 依赖变更（索引、模板、模型版本）→ 反馈闭环 → 模型本身"排序；关键是每周用固定 golden set 回放生产配置。


---

## 美团

### 1. bf16 和 fp16 有什么区别，各自是多少位，你在训练中如何选择精度？

<!-- EN: What is the difference between bf16 and fp16, how many bits is each, and how do you choose precision in training? -->

**Source**: https://outcomeschool.com/blog/how-does-model-quantization-work

**参考答案**

两者都是 16 位，切分方式不同：

| | 符号 | 指数 | 尾数 | 动态范围 | 精度 |
| :--- | :-: | :-: | :-: | :--- | :--- |
| fp16 | 1 | 5 | 10 | 约 6e-5 … 65504 | 约 3 位十进制 |
| bf16 | 1 | 8 | 7 | 与 fp32 相同（约 1e-38…3e38） | 约 2 位十进制 |

后果：fp16 会**上溢和下溢**——小于 6e-5 的梯度被冲成 0，所以要做 loss scaling（动态 `GradScaler`），而且在大 logits 的注意力里仍然难逃 NaN。bf16 有 fp32 的范围，不需要缩放；代价是尾数更粗，表现为小模型或精度敏感任务上收敛更慢（早期 RL 与 layernorm 密集网络的结论）。

怎么选：

1. **硬件支持就上 bf16**（A100/H100/TPU/Apple/较新消费卡），做 LLM 训练与推理的默认。
2. **fp16 + loss scaling** 只用于 Ampere 之前的卡（V100/T4），且优化器里保留 fp32 主权重。
3. **fp32** 用于小模型、数值调试，以及任何出现 NaN 的层（softmax logits、layernorm 统计量）。
4. 永远混合精度：fp32 主权重、bf16 计算、fp32 优化器状态。

实践提醒：推理时 fp16 与 bf16 差别很小；训练稳定性上差别巨大，「切到 bf16」能修掉大多数神秘 NaN。

**速记**：都是 16 位——fp16 是 5 位指数/10 位尾数（精度高但范围窄、必须 loss scaling），bf16 是 8 位指数/7 位尾数（范围同 fp32、免缩放但尾数粗）；有 Ampere 以上就选 bf16，优化器状态永远留 fp32。

### 2. attention 计算有哪些显存优化策略？

<!-- EN: What memory optimisation strategies exist for attention computation? -->

**Source**: https://outcomeschool.com/blog/kv-cache-in-llms and https://outcomeschool.com/blog/paged-attention-in-llms

**参考答案**

两个不同的显存问题，经常被混为一谈。

**A. 训练 / prefill——$n^2$ 的分数矩阵**

| 技术 | 效果 |
| :--- | :--- |
| FlashAttention（分块 + 在线 softmax） | 从不在 HBM 物化 $n\times n$ 矩阵；O(n) 显存，且 HBM 往返更少所以更快 |
| 激活重算 | 只存块输入，反向时重算；多约 30% 计算换大量显存 |
| 序列 packing + varlen kernel | 样本间不留 padding 浪费 |
| Ring / 上下文并行注意力 | 超长上下文时把序列切到多卡 |

**B. 解码——KV cache**

| 技术 | 效果 |
| :--- | :--- |
| GQA / MQA | KV 头变少 → cache 按分组倍数缩小 |
| MLA | 缓存低秩潜变量而非每头 K/V |
| PagedAttention | 块级分配，不按最大长度预留；顺带支持前缀共享 |
| KV 量化（int8/fp8） | cache 约小一半；用自己的评测测质量，别看 perplexity |
| 滑窗 / 稀疏注意力 | 把 cache 限制在最近 w 个 token |
| 前缀缓存复用 | 共享 system prompt 只算一次 |

容量规划口诀：每 token KV 字节 = 2 × 层数 × kv 头数 × head_dim × 字节数。32 层、8 个 KV 头、128 维、fp16 时约 128 KB/token——8k 上下文 × 100 并发会话 ≈ 100 GB。决定 batch size 的是这个数，不是 FLOPs。

**速记**：训练/prefill 的瓶颈是 n² 分数矩阵（FlashAttention 分块+在线 softmax、激活重算）；解码的瓶颈是 KV cache（GQA/MQA、MLA、PagedAttention、KV 量化、滑窗）；先用 KV 字节公式算容量再决定并发。

### 3. 分布式训练中 ZeRO-2 与 ZeRO-3 的核心区别是什么？

<!-- EN: What is the core difference between ZeRO-2 and ZeRO-3 in distributed training? -->

**参考答案**

ZeRO 依次把优化器状态、梯度、参数切分到各数据并行 rank 上。

| 阶段 | 切了什么 | 单卡显存（Ψ 个参数） | 通信 |
| :--- | :--- | :--- | :--- |
| DDP | 不切 | 16Ψ 字节（fp16 权重 + fp32 主权重 + Adam m,v） | 梯度 all-reduce |
| ZeRO-1 | 优化器状态 | ~4Ψ + 12Ψ/N | 梯度 all-reduce |
| ZeRO-2 | + 梯度 | ~2Ψ + 14Ψ/N | 梯度 reduce-scatter（更便宜） |
| ZeRO-3 | + 参数 | ~16Ψ/N | reduce-scatter + **每层 all-gather 参数** |

核心差别：**ZeRO-2 每张卡仍持有完整参数集**，ZeRO-3 不持有，因此能训单卡放不下的模型。代价是前向逐层 all-gather、反向再来一遍，通信量约为 ZeRO-2 的 1.5 倍，对延迟敏感的小层受害最大。

实践含义：

- 装得下时 **ZeRO-2 通常更快**——通信少、重叠简单。
- 权重+梯度+状态超出单卡显存就必须 **ZeRO-3**，且它与小 micro-batch 不搭（all-gather 开销要靠 batch 摊薄）。
- 跨节点时叠加张量/流水并行；纯 ZeRO-3 跨节点会受限于互联带宽。

**速记**：ZeRO-2 只切优化器状态和梯度，每卡仍持有完整参数；ZeRO-3 连参数一起切，能训单卡放不下的模型，代价是每层前反向各一次 all-gather、通信约 1.5 倍；装得下就选 ZeRO-2。

### 4. LLM 为什么要逐 token 流式输出？它底下跑的是什么协议？

<!-- EN: Why do LLMs stream token by token, and what protocol is underneath? -->

**Source**: https://outcomeschool.com/blog/how-does-token-streaming-work

**参考答案**

因为模型本来就是逐 token 生成的（自回归解码），不流式就等于浪费：总延迟一样，但**首 token 时间**被拉长成整个生成时长。流式把「等 6 秒」变成「300 ms 出首字 + 边读边等」，这是产品收益，不是算力收益。

底下三层：

| 层 | 机制 |
| :--- | :--- |
| 传输 | HTTP/1.1 `Transfer-Encoding: chunked`，或 HTTP/2/3 流式；只有需要双向才用 WebSocket |
| 协议 | SSE：`text/event-stream`，反复的 `data: {...}\n\n` 帧，以 `data: [DONE]` 结束；每帧是一个 JSON delta |
| 载荷 | OpenAI 风格的 `choices[0].delta.content`，最后一帧带 `finish_reason` |

只有生产上才会遇到的坑：

1. **解码默认不是增量的**——一个 token 可能是半个 UTF-8 序列，或 byte-fallback token 的一半。要缓冲到解码字符串合法为止，否则吐出 `�`。
2. **反向代理会缓冲。** Nginx 要 `proxy_buffering off`；CDN 可能把整条流扣住，收益归零。gzip 若不按块 flush 也会缓冲。
3. **用量与错误最后才到**——token 数要从最后一个事件对账，并处理中途断掉、没有 `finish_reason` 的流。
4. 长工具调用期间要靠注释帧（`: ping`）保活，否则撞上空闲超时。

**速记**：模型本来就逐 token 生成，流式把"等 6 秒"变成"300ms 出首字"；底层是 chunked HTTP + SSE 的 `data:` 帧（末尾 `[DONE]`）；坑在增量解码（半个 UTF-8 字符）、代理缓冲、中途断流无 finish_reason。


---

## Groq

### 1. LPU 完全没有 HBM，只有片上 SRAM。针对这种机器重新做一遍 decode 的 roofline 分析，并告诉我哪些结论会改变。

<!-- EN: An LPU has no HBM at all, just on-die SRAM. Redo the decode roofline argument for that machine and tell me what changes. -->

**Source**: https://outcomeschool.com/blog/how-does-an-lpu-work

**参考答案**

结论：GPU 上 batch=1 的 decode 算术强度约 2 FLOP/byte，远低于拐点，所以每 token 时间 ≈ `权重字节 / HBM 带宽`；把 HBM 拿掉，整个论证的形状都变了——**约束从带宽换成容量**。

| 项 | GPU | 纯 SRAM |
| :--- | :--- | :--- |
| 权重来源 | HBM，3 TB/s 量级 | 片上 SRAM，每芯片约 80 TB/s（Groq 公开数字） |
| 每芯片每 token | `W/BW_hbm` | `W_slice/BW_sram` ≈ 230 MB / 80 TB/s ≈ **3 µs** |
| 容量 | 每卡 80 GB，宽裕 | **230 MB——真正的硬约束，不是带宽** |
| 拐点 | 需要 batch ~100+ | batch ≪ 10 就到达 |

三个后果：

1. **roofline 变成容量而非带宽。** 70B、8-bit 约 70 GB，必须切到 ≥300 块芯片。设计问题从"搬多少字节"变成"整个东西放不放得下、怎么路由"。
2. **互连取代 HBM 成为上界。** 每 token 延迟 ≈ 每芯片计算（µs 级）+ 流水穿越 + 跨 fabric 的激活交换，网络的延迟与跳数开始主导。
3. **batching 很早就失去收益。** 权重常驻、按 SRAM 速度反复读，batch=1 已接近算力屋顶；加 batch 主要用于填满阵列利用率，且受"能从权重里匀出多少 SRAM 给 KV"限制。

**速记**：去掉 HBM 后 roofline 的约束从"带宽"换成"**容量**"：70B/8bit 必须切到 300+ 卡，瓶颈变成片间互连延迟；单 chip 每 token 约 3 µs 量级（230MB / 80TB/s），batch=1 就接近算力屋顶，所以 batching 早就不划算了。

### 2. 一个 70B 稠密模型、8-bit 权重，每颗芯片约 230 MB SRAM。讲讲部署方案和单位经济性。

<!-- EN: A 70B dense model at 8-bit weights, chips with ~230 MB of SRAM each. Walk me through the deployment and the unit economics. -->

**参考答案**

结论：先算容量，再算吞吐，最后一定要用"卡数 × 卡时"而不是 $/token 来算账。

**容量。** 70B × 1 byte ≈ 70 GB 权重，每芯片 230 MB → 光放权重就要 **≥ 305 块芯片**，此时给 KV cache、激活和暂存留的 SRAM 是零。真实部署要高于这个下限（张量并行切分必须合法，比如头数可整除），所以正确的思考粒度是整机架。

**吞吐。** 每生成一个 token，每块芯片要扫一遍自己的权重切片；约 80 TB/s 下 230 MB 约 3 µs，所以计算侧下限是个位微秒，而**实际**每 token 时间由流水深度与 300 块芯片间的链路穿越主导。公开口径下 70B 级模型在这里是每流每秒数百 token 的量级——这是数量级，不是规格。

**单元经济性**：`每 token 成本 ≈ (卡数 × 每卡时价) / (3600 × 每秒 token 数)`。320 块卡、几百 tok/s 时，单流性价比极差：token 之间大部分硅片在空转。靠并发放大——多条序列共享同一份常驻权重，聚合吞吐随 batch 上升直到算力饱和或 KV 用光 SRAM，而**KV 才是这里的硬上限**（容量已被权重吃掉）。

所以诚实的卖法是**固定成本下的延迟**，卖给那些原本为深队列 GPU 付费的负载：实时语音、agent、交互式代码生成。

不要承诺 GPU 式的批量折扣：这台机器的 batch 曲线很早就平了，应按"流/延迟档 + 你能真正维持的利用率"定价。

**速记**：70GB 权重 / 230MB ≈ 300+ 卡只是**容量下限**，KV 几乎没地方放；单 token 约几微秒但实际被流水与互连主导；经济性要按 `卡数 × 卡时` 算，单流很亏，靠共享常驻权重的并发把摊薄做回来——所以卖的是"固定成本下的确定性延迟"，不是批量折扣。

### 3. 在 GPU 上你通过批处理来摊薄权重读取。在纯 SRAM 机器上，批处理的账该怎么算，这又应如何改变我们的定价方式？

<!-- EN: On a GPU you batch to amortise weight reads. What is the batching calculus on an SRAM-only machine, and how should that change how we price? -->

**参考答案**

结论：GPU 上 batch=1 的 decode 是访存受限的 GEMM，每 token 无论如何都要付 `W` 字节的 HBM 流量，所以加序列近乎免费；在纯 SRAM 机器上权重不离开芯片，这个逻辑反过来——**batching 是利用率手段，不是降本杠杆**。

| 问题 | GPU | 纯 SRAM |
| :--- | :--- | :--- |
| batching 是否摊薄权重读 | 是，主要收益 | 边际收益；读本身便宜约两个数量级 |
| 什么限制 batch | HBM 里的 KV（很大） | **KV 与权重共享 230 MB——极小** |
| batch 是否降低每 token 成本 | 强烈 | 微弱，且只在阵列未打满时 |
| batch 是否增加延迟 | 是（排队） | 是，而且会破坏你卖出去的 p99 |

所以真正的旋钮是**固定步长预算下的并发**：能塞多少序列就塞多少，跑一个确定性步，一次性返回所有 token——得到接近 GPU 的聚合吞吐，而尾部延迟不随负载恶化。

定价含义：成本是 `卡数 × 时间`，单流已贴近算力下限，收入跟着可持续的并发占用来。按三点定价：

1. **延迟档 / 预留流**——确定性 p99 本身就是产品；
2. **按利用率的承诺合同**——占着 320 条流里 3 条的客户是被其他人补贴的，折扣应给到预留容量；
3. **短输出的廉价交互流量**——这类负载在 GPU 上白等队列时间；不要给长批量离线任务打折，那是最差的适配。

**速记**：SRAM 机器上权重常驻，batch 摊薄权重读的收益很小，真正的上限是**和权重抢 230MB 的 KV**；所以 batching 是"填满阵列"而非降本杠杆；定价应卖"确定性 p99 的保留流/延迟档"，而不是学 GPU 按 token 批量打折。

### 4. 确定性是它最主要的卖点。它在 p99 上究竟带来了什么，以及为什么这对 agentic 工作负载尤其重要？

<!-- EN: Determinism is the headline claim. What does it actually buy at p99, and why does it matter especially for agentic workloads? -->

**参考答案**

结论：确定性把 p99 从"随负载涌现的统计量"变成"编译期算出来的常数"。

机制上：没有动态 batching、没有缓存未命中方差、没有调度器、没有 SM 争抢，每步耗时就是编译方案隐含的一个常量。于是延迟分布塌缩——p50 与 p99 只差几个百分点，p99 成为**可以写进合同的数字**。

这比峰值速度值钱，原因有三：

1. **容量规划变成算术。** 步长固定，就能精确算出整机架支持多少并发准入，且在峰值仍然成立；GPU 的 p99 则要为不可预测的尾部留余量。
2. **输出跨运行逐位一致。** 同一提示、同一采样序列、同一 logits → 可复现的 bug 和可复现的 A/B。GPU 上非确定 kernel + 依赖负载的 batching，让"它挂过一次"根本无法排查。
3. **尾延迟不被邻居污染**——别人凌晨三点的批量任务挪不动我的语音通话。

为什么 agent 负载尤其在意：agent 是**链式**调用，一个任务常 10~40 次，中间还夹工具 I/O。延迟复利、方差也复利——重尾下整链 p99 由 40 步里最差的一步决定，而第 37 步超时会丢掉前面全部工作。确定性把链路预算变成 `N × step`，也让**取消与 deadline 传播**变得可做。

要主动说的免责：确定性只在给定 batch 组成与路由的前提下成立，准入决策与工具延迟仍在外面；而且如果主机侧的分词器或采样器不确定，逐位一致也白搭——团队通常在这里漏掉保证。

**速记**：确定性把 p99 从"随负载涌现"变成"编译期算出来的常数"，p50/p99 只差几个百分点；agent 场景是 10~40 步的链式调用，方差会复利放大、超时就丢掉前面全部工作，所以"固定 step 时间"比峰值吞吐更值钱。

### 5. 当专家选择依赖数据时，你如何在静态调度的互联结构上服务一个大型 MoE 模型？

<!-- EN: How would you serve a large mixture-of-experts model on a statically scheduled fabric when expert selection is data-dependent? -->

**参考答案**

结论：先说清矛盾——编译器要求整图的驻留与时间已知，而 MoE 让两者都变成输入的函数。两个约束相互独立，所以拆开处理。

| 约束 | 性质 | 缓解手段 |
| :--- | :--- | :--- |
| **容量**：所有 E 个专家常驻 | 静态、可解 | 专家并行；复制热专家；专家量化比注意力更狠 |
| **路由**：哪些 token 去哪些专家 | 动态、不可调度 | capacity factor、丢 token、批量 all-to-all |

固定 fabric 上可行的做法：

1. **按最坏情况预留，而非平均。** 编译一个每专家最多接受 `C` 个 token 的步（capacity factor ~1.25），溢出时丢掉门控权重最低的 token（它们保留残差）。这样把数据相关图变成**步长有保证的静态图**，丢包率成为可度量的质量旋钮。
2. **每步两个阶段**：经固定置换表做 gather/scatter，再对预留槽位做稠密矩阵乘。fabric 的 all-to-all 跑在编译期确定的槽位映射上。
3. **复制热专家**而不是动态负载均衡——领域内热度是稳定的，静态复制就消除了大部分倾斜。这是"先测量、再编译"胜过"运行时自适应"的少数场景。
4. **把 top-k 决策留在主机边界**，让编译器只看到"一个待应用的置换"。

代价：丢 token 在长尾路由上是真实的质量损失，粗粒度 capacity factor 会浪费硅片。最差的是细粒度 MoE——交换变成延迟主导，而每个专家的矩阵乘又小到填不满阵列。

**速记**：MoE 把"容量"（静态可解）和"路由"（数据相关）拆开处理：容量用专家并行+热专家复制，路由用固定 permutation 表 + capacity factor + 最低门控权重丢 token，把动态图变成**步长有界的静态图**；细粒度 MoE 最不适合固定时隙架构。


---

## Together AI

### 1. 为一个支持连续批处理的推理引擎设计调度器。

<!-- EN: Design the scheduler for a continuous-batching inference engine. -->

**Source**: https://outcomeschool.com/blog/continuous-batching-in-llms · https://outcomeschool.com/blog/how-does-vllm-work

**参考答案**

结论：连续批处理意味着每步都重新组批——完成的序列离开、等待的加入。所以调度器不是队列，而是**每步在显存预算上做的准入控制器**。

| 阶段 | 决策 | 约束 |
| :--- | :--- | :--- |
| 1. 退役 | 丢掉完成/中止的序列，释放块 | 立即释放 KV，不要等下一步 |
| 2. 准入 | 从等待队列提升到运行队列 | **prefill 的 token 预算**，不是请求数 |
| 3. 驱逐 | 抢占最低优先级，重算或换出 | KV 块池耗尽 |
| 4. 组批 | 运行序列 + chunked prefill | `max_num_seqs`、`max_num_batched_tokens` |
| 5. 执行 | 一次融合前向 | 每序列的注意力元数据 |

区分好调度器与朴素调度器的选择：

1. **prefill 与 decode 预算解耦。** 长提示不能卡住 200 个 decoder；chunked prefill（每步固定 token 预算，如 512~2048）让 ITL 保持平坦，同时仍能准入新工作。
2. **分页 KV 是前提。** 固定大小的块（16 token）+ 每序列块表 → 没有外部碎片，前缀共享退化为引用计数。
3. **抢占方式要选对。** 短上下文重算便宜、长上下文换出到主机便宜，按序列选。
4. **公平性**：纯 FCFS 会让长生成饿死在一堆短对话后面；两条 lane，或按已服务 token 做赤字轮转。
5. **按 (模型, GPU, TP 度) 自动调 batch 上限**——这是一条膝点随上下文长度移动的曲线。

必须点名的失败模式：按**请求数**准入。两个 32k 提示的请求和三十三个 200-token 请求在计数器上一样，行为却完全不同。

**速记**：调度器是**每步重算的准入控制器**而非队列：先退役释放 KV、按 token 预算预填、满了就抢占（短的重算/长的换出），并解耦 prefill 与 decode 预算；按请求数准入是最典型的错。

### 2. 解释投机解码（speculative decoding）。它在什么时候有帮助、什么时候有害，以及为什么要让投机模型适配线上真实流量？

<!-- EN: Explain speculative decoding. When does it help, when does it hurt, and why adapt the speculator to live traffic? -->

**Source**: https://outcomeschool.com/blog/speculative-decoding · https://outcomeschool.com/blog/n-gram-speculation-in-llms

**参考答案**

结论：投机解码是一场赌注——*验证比生成便宜*。便宜的草稿模型提出 γ 个 token，目标模型**一次前向**全部验证，接受最长正确前缀（拒绝采样保证分布与目标模型完全一致）。加速比 ≈ `接受数 / 验证代价`。

| 场景 | 效果 | 原因 |
| :--- | :--- | :--- |
| **低并发/单流** | 大幅收益（常见 2~3 倍） | 验证是访存受限，多验几个 token 近乎免费 |
| **高并发、算力饱和** | **变差** | 草稿的 FLOPs 变成纯开销 |
| 可预测文本（代码编辑、从上下文抽取、JSON） | 接受率高 | 草稿可以直接抄提示（n-gram 查表） |
| 创意、高熵文本 | 接受率低 | 为约 1 个 token 付了 γ 次提议 |
| 超长上下文 | 草稿成本上升 | 草稿要在同样 KV 上做注意力 |

为什么必须按线上流量自适应：接受率是**工作负载**的属性，且随租户和时段漂移；静态 γ 或草稿模型是在一个几周后就不再像生产的评测集上调出来的。

1. **按请求类别**（端点、提示长度分桶、温度）测接受率，不要只看全局。
2. **按 α 调 γ**：每次验证的期望 token 数 ≈ `(1-α^(γ+1))/(1-α)`；α=0.5 时 γ=8 是纯浪费。
3. **按类别换草稿**：拷贝密集的代码编辑与 RAG 抽取用 n-gram 查表；开放对话用蒸馏的 1B 草稿；batch 一旦越过饱和点就关掉投机。
4. **用线上指标做闸门**（每步接受 token 数、p99 ITL）并带自动熔断。经典生产错误：它赢了单流基准，但在 40 并发时是净损失。

**速记**：投机解码是"验证比生成便宜"的赌注：低并发/可预测文本大赚，**高并发算力饱和时会亏**；accept rate 是流量属性，所以要按请求类调 γ、按场景换 drafter（n-gram 抄上下文最适合代码编辑），并用接受率做自动熔断。

### 3. 为一个专属 endpoint 定价：估算 70B 模型每百万输出 token 的成本，并解释吞吐与延迟之间的权衡。

<!-- EN: Price a dedicated endpoint: estimate cost per million output tokens for a 70B model, and explain the throughput-latency trade. -->

**参考答案**

结论：把它当一条链来算，逐步亮出假设，并标明哪些是典型区间而不是实测。

**第一步，显存。** 70B 在 FP8 下约 70 GB 权重。8×H100-80GB（640 GB）正好一个副本还有空间放 KV；BF16 是 140 GB，仍是一个节点但 KV 余量减半。

**第二步，吞吐。** 聊天型配比（约 1k 提示 / 400 输出 token）下，一个 8×H100 节点开连续批处理，聚合输出通常落在每秒数千 token 量级，且随上下文长度变化很大（[典型经验值]——实测，别照抄）。取 3,000 tok/s（batch 舒适、算力接近饱和）。

**第三步，成本。**

```
node cost      ≈ 8 GPU × ~$2.5/GPU-hr ≈ $20/hr   [典型云价]
tokens/hour    = 3,000 tok/s × 3,600    = 10.8 M
cost/M tokens  = $20 / 10.8             ≈ $1.85 at 100% utilisation
at 60% realised utilisation             ≈ $3.1 / M output tokens
```

再加上：为峰值必须持有的空闲容量（最大头）、失败与重试请求、prefill token（计不计费）、冷启动。真实的专享报价是**每百万输出 token 约 $3–6**，下限由预留容量而不是已消费 token 决定。

把权衡说清：batch size 是沿曲线移动的旋钮——加大提高吞吐（更便宜的 $/token），也加大排队延迟（ITL/p99 变差）。所以专享端点应按**预留吞吐 + 延迟 SLA** 定价，超出预留的突发流量要更贵，因为突发是靠压低别人的延迟换来的。两个 token 量相同但 ITL 目标 50 ms 与 500 ms 的客户，服务成本完全不同——**给 SLO 定价，不要给 token 定价**。

**速记**：先算显存（70B FP8≈70GB，8×H100 一节点），再按典型聚合吞吐 ~3k output tok/s、$20/卡时推得满负荷 ~$1.85/M、实际利用率 60% 后 ~$3/M；报价要卖**预留吞吐 + 延迟 SLA**，因为 batch 加大是拿 p99 换成本。


---

## Perplexity

### 1. 某次发布后，p95 首 token 延迟（TTFT）从 1.2 s 退化到 3 s。讲讲你会如何定位并修复。

<!-- EN: p95 time-to-first-token regressed from 1.2 s to 3 s after a release. Walk me through finding and fixing it. -->

**Source**: https://www.youtube.com/watch?v=XD8DD4cEHu0

**参考答案**

结论：先确认它是真的。p50 持平而 p95 涨到 3 倍，是**排队**的特征，不是算力的特征。先查客户端指标是否包含重试，以及负载构成有没有变（更长的提示、更多并发）。

**第一步：拆 TTFT。** 永远不要 debug "TTFT"，要 debug 它的组成部分：

`TTFT = 排队等待 + 路由/鉴权 + 检索（若有） + prefill + 首个 decode 步`

每一段都打 span。回归只可能落在其中一段，这一下就把搜索空间砍半。

| 跳起来的段 | 原因 |
| :--- | :--- |
| 排队等待 | 到达率突增、自动扩缩滞后、为吞吐牺牲延迟的 batch 调大 |
| Prefill | 提示变长了——新 chat 模板、检索 top-k 调大、系统提示变长、加了图片 token |
| Prefill | chunked-prefill 大小改了；调度器开始优先 decode |
| Prefill | **前缀缓存未命中**——缓存键变了（模板、工具列表、顺序），共享系统提示不再命中 |
| 首个 decode 步 | 丢了 CUDA graph 捕获、投机解码被关、量化 kernel 换了 |
| 路由 | 某个开关或新模型版本把流量导到了更冷的池子 |

**第二步：把这次发布与上一个好版本做 diff。** 依赖、模型权重版本锁定、分词器、模板、检索配置、`max_tokens`、批处理/调度参数。实践中最常见的真因是**提示变长了**（检索多塞了上下文），而 prefill 随长度近似二次增长，token +40% ≈ prefill +90%。

**第三步：用回放做二分。** 抓 1000 条周二的真实请求，用开关开/关跑新构建。确认到具体段之后修复，并把**按段的 SLO** 加到看板上，让下一次回归自己指认自己。

**速记**：先看 p50/p95 形状——p50 平而 p95 涨三倍是排队问题不是算力问题。把 TTFT 拆成"排队 + 路由 + 检索 + prefill + 首 decode"逐段打点；最常见真因是 prompt 变长（检索 top-k 或模板改动）和前缀缓存 key 变化导致 miss。


---

## Cursor

### 1. 为数百万 DAU 服务一个自研补全模型：讲讲你的推理成本模型，以及你最有效的三个杠杆。

<!-- EN: Serving a custom completion model to millions of DAU: walk me through the inference-cost model and your top three levers. -->

**Source**: https://www.youtube.com/watch?v=jV2sCj4lHYk · https://outcomeschool.com/blog/how-does-cursor-work

**参考答案**

先写成本模型，杠杆自然浮出来：

`每日成本 = 每日请求数 × (prefill_tokens × p_price + output_tokens × d_price) / 缓存命中调整系数`

编辑器的形状很特殊：**巨大前缀 + 极小补全**。一次 tab 请求可能带 2000–8000 token 的文件上下文，只吐 20–100 token。所以 prefill 决定账单，decode 决定**延迟**。

| 杠杆 | 机制 | 效果 |
| :--- | :--- | :--- |
| **让请求根本不发生** | 防抖 + 用户继续打字时取消在途请求 | tab 场景下多数已发请求从未被渲染 |
| **前缀缓存（prefix caching）** | 稳定系统提示 + 文件头排前面，易变的尾部（光标附近）排最后 | prefill 变成缓存命中；一个 token 顺序错就让下游全失效 |
| **专用小模型 + 量化** | 面向 next-edit 训练的补全模型、FP8、投机解码（speculative decode）| 同等接受率下每 token 便宜 5–20× |

其余但真实的手段：按档位路由（免费/付费用不同模型）、按排序而非原始大小截上下文、连续批处理（continuous batching）服务器上打包、把推理放在用户就近区域。

必须点名的失败模式：产品已经在丢掉大部分请求，却还在优化 kernel——取消路径通常比推理栈更值钱。

**速记**：成本 ≈ 请求数 ×(prefill 贵 + decode 长)；编辑器是「大前缀 + 小补全」。三杠杆：取消/防抖让请求不发、前缀缓存让 prefill 命中、专用小模型 + 量化降单 token 成本。


---

## Glean

### 1. 讲讲一次查询的延迟预算：查询理解 → 检索 → 重排 → LLM 回答。你在哪里花时间，又在哪里削减？

<!-- EN: Walk me through the latency budget of a query: query understanding → retrieval → rerank → LLM answer. Where do you spend and where do you cut? -->

**Source**: https://outcomeschool.com/blog/rag-evaluation

**参考答案**

"感觉快"的目标大致是 **p95 < 1.2s 出首 token**、4s 内给出完整短答。预算要显式分配：

| 阶段 | p50 / p95（典型） | 花还是砍 |
| :--- | :--- | :--- |
| 查询理解 | 分类器 15 / 60 ms；LLM 改写扩写则 **150–350 ms** | 砍：只对标记为模糊/过长的查询做 LLM 改写，短导航类查询完全跳过 |
| 检索（分片扇出） | 40 / 120 ms | 花：并行扇出 + 硬性 150 ms 截止；对慢分片做尾延迟对冲 |
| 重排（cross-encoder，50–100 候选） | 80 / 250 ms | 砍得最狠：剪到 40 个候选、GPU 批处理，或导航类查询直接绕过重排 |
| LLM 回答（首 token） | **300 / 800 ms** | 花：这是用户真正看到的数字；缓存 system + 检索前缀 |
| 流式生成完成 | 1–3 s | 砍：按意图限制答案长度 |

收益真正来自：

1. **检索与查询理解并行**——改写投机执行，冲突则丢弃。
2. **重排候选剪枝**：100 → 40 通常只掉个位数 nDCG，重排时间减半；要实测，不要拍脑袋。
3. **前缀缓存 + prompt 顺序纪律**（稳定指令在前、易变用户上下文在后），避免每轮重付 prefill。
4. **流式输出，且先展示检索来源再出答案**——渐进披露对感知延迟的改善大于任何单阶段优化。

**速记**：先给每段定预算（重排与 LLM 首 token 是两大块）；能砍的是 LLM 改写（仅模糊查询启用）和重排候选数（100→40），能花的是并行扇片与流式；稳定指令放最前吃前缀缓存。


---

## Character.AI

### 1. 我们的服务成本主要由 KV cache 而非权重决定。把它降低一个数量级，并告诉我你要放弃什么。

<!-- EN: Our serving cost is dominated by KV cache, not weights. Get it down by an order of magnitude and tell me what you give up. -->

**Source**: https://outcomeschool.com/blog/kv-cache-compression

**参考答案**

10 倍从来不是靠一个技巧。要叠加，并诚实交代代价：

| 手段 | KV 字节的典型倍数 | 放弃什么 |
| :--- | :--- | :--- |
| GQA / MQA（更少 KV 头） | 相对 MHA 4–8× | 已默认——需重训，细粒度召回略有损失 |
| KV 量化（int8 / fp8，逐通道） | 2× | 离群通道有质量风险；需逐层校准 scale |
| **跨用户前缀共享**（同角色的同一人设前缀） | 对 *prefill* 有效 2–5× | 仅当前缀逐字节相同；每用户记忆块会破坏共享 |
| 淘汰低注意力 KV（注意力分数或近期窗口策略） | 长上下文下 2–4× | **远期内容的精确召回没了**——聊天可接受，法律/代码致命 |
| 卸载到主机内存 / NVLink 池 | 释放 HBM，不省算力 | 增加 PCIe 延迟；生成中途换出会掉吞吐 |
| Paged attention | 因碎片省约 20–40% HBM | 不算压缩，是消除浪费，先做它 |

不显然的一点：这个产品里最便宜的大收益通常是**前缀共享**，因为一条消息的 KV 大头是共享角色卡加公共 system prompt，不是用户自己的历史。缓存 key 必须包含人设版本。

必须说清的硬事实：10 倍目标要求接受**远距离召回退化**。缓解办法是把紧凑记忆块放进 prompt 而不依赖原始 KV，并在"跨会话长上下文探针"集上测召回，而不只看 perplexity。

**速记**：10 倍不是单点技巧：GQA + KV int8 + 跨用户前缀共享 + 低注意力 KV 淘汰 + 分层卸载叠加；最便宜的大头常是"同角色共享人设前缀 KV"；代价是远期精确召回，用显式记忆块补偿并在会话级探针上衡量。

### 2. 这里的对话平均约 180 条消息。设计位于轮次之间的缓存。

<!-- EN: Dialogues here average around 180 messages. Design the cache that sits between turns. -->

**Source**: https://outcomeschool.com/blog/how-does-prompt-caching-work

**参考答案**

三层，而决定一切的约束是：**KV 只存在于一块 GPU 上**。会话必须钉在持有其缓存的那张卡上，因此**路由本身就是缓存设计的一部分**，不是其上的一层。

| 层 | 存什么 | 淘汰 | 命中画像 |
| :--- | :--- | :--- | :--- |
| L1 每 GPU 前缀缓存 | 只追加会话的块哈希 KV，加共享人设前缀块 | 块级 LRU，短 TTL（分钟级） | 热路径：下一轮只需 prefill 新用户 token |
| L2 主机/池内存 | 冷会话 KV，块粒度 | 容量 + 会话 TTL（小时） | 暂停后回来：超过几千 token 时重载优于重新 prefill |
| L3 重算 | 人设 + 摘要 + 最近轮次 | — | 会话过期或换了 GPU |

180 条消息下能撑住的机制：

1. **块哈希**（radix / paged 前缀缓存）：对 token 块做哈希，共享人设前缀每张 GPU 只存一份，被该角色所有会话引用。
2. **会话亲和路由**：对 `conversation_id` 做一致性哈希；迁移活会话的代价是一次完整重 prefill，所以轻微倾斜也要优先粘住。
3. **失效才是难点**——记忆重写或人设更新会改变前缀，静默作废该会话的 L1。给前缀打版本（`persona_v` + `summary_v`）并作为缓存 key 的一部分，绝不用不同前缀的 KV 服务请求。
4. **延迟账**：180 条 × 约 40 token ≈ 7k token 历史。每轮重 prefill 是"约 150 ms"与"40 ms 以内"首 token 的差别；这里缓存的价值高于砍模型尺寸。

**速记**：KV 只在单卡上，会话亲和路由是缓存设计的一部分。三层：GPU 内块哈希前缀缓存（人设前缀跨会话共享）、主机内存冷会话、过期重算；关键是失效——摘要或人设更新会悄悄改前缀，必须版本化并进缓存 key。

### 3. 你选择在 int8 下原生训练，而不是做训练后量化。为这个选择辩护。

<!-- EN: You train natively in int8 rather than doing post-training quantization. Defend that. -->

**Source**: https://outcomeschool.com/blog/how-does-model-quantization-work

**参考答案**

诚实的辩护不是"int8 更好"，而是"**PTQ 校准不了这个数据分布**"。

| 论点 | 细节 |
| :--- | :--- |
| 分布由用户生成且带对抗性 | 角色扮演、表情、自造人名、中英混写、30 种语言。PTQ 只用几百条样本校准，观测不到的尾部恰恰是用户所在区域，离群激活会被裁掉 |
| 离群值必须被"学着绕开"，而不是裁掉 | 原生 int8 让模型自己调整权重分布以留在可表示区间；QAT 在训练中收窄动态范围，而不是事后压扁 |
| 内核 | int8 matmul + int8 KV 路径可与模型协同设计（逐通道 scale 内建）；对 bf16 训好的模型做 PTQ 只能事后凑 scale |
| 重量化的成本 | 内容/人设更新与微调很频繁；重训贵，但生产环境里一次搞砸的 PTQ 回归更贵 |

坦白说清放弃的东西：

1. **训练稳定性**——需要高精度 master 权重、随机舍入或梯度裁剪，以及量化器退火的 warmup；loss 尖峰很常见。
2. **scaling law 迁移更弱**——int8 checkpoint 不能从 bf16 轻松续训，跨精度比较也不干净，某种程度被锁定。
3. **小模型质量天花板**——1B 以下的模型从 int8 掉的点比大模型多；要实测，不能假设。
4. **生态摩擦**——kernel、工具链、评测框架都以 bf16 为先。

再加一句能证明真做过的补充：注意力往往仍跑更高精度，所以"原生 int8"是逐层的，不是全局的。要维护 int8 专用评测集——PTQ 式评测抓不到激活范围回归。

**速记**：真理由不是"int8 更好"，而是 PTQ 校准集覆盖不了用户生成分布（角色扮演、表情、混写、自造名词），尾部激活被裁；原生 int8 让模型训练中适应可表示范围并与 int8 kernel/KV 协同。代价是训练不稳、跨精度迁移受阻、小模型掉点更多、生态仍以 bf16 为主。

### 4. 估算我们服务一条消息的成本，并告诉我哪个杠杆对它的影响最大。

<!-- EN: Estimate what one message costs us to serve, and tell me which lever moves it most. -->

**Source**: https://outcomeschool.com/blog/kv-cache-compression

**参考答案**

自底向上搭，数字标注为"典型"而非实测：

**单条消息（典型长会话）：**

| 项 | 值 |
| :--- | :--- |
| 用户消息 | 约 40 token |
| prompt 总量（人设 + 摘要 + 180 轮） | 约 7,000 token |
| 回复 | 约 120 token |
| prefill 工作量 | 7,000 token/轮（除非命中缓存） |
| decode 工作量 | 120 token/轮 |

自建服务下，单条成本 ≈ `(prefill_tokens × t_prefill + output_tokens × t_decode) / tokens_per_GPU_second × GPU_hour_price`。有意义的比例是：**7,000 token 的 prefill 约为 120 token 回复的 58 倍 token 量**，且高 batch 下 prefill 是算力受限、decode 是显存带宽受限。所以支出的形状大致是：prefill 在总量上占主导（除非缓存），decode 主导延迟。

最能撬动它的手段，按序：

1. **KV / 前缀缓存**——把每轮 7,000 的 prefill 变成约 40 个新 token，prefill 项降约 100 倍。没有别的能接近。
2. **缩短有效 prompt**——人设 + 摘要的预算：7k 的 prompt 是产品决策（保留多少历史/人设），不是模型属性。
3. **batch size / 连续批处理**——每 token 的 GPU 小时亚线性但明显改善；这是规模化下单条成本的主要运维杠杆。
4. **KV 量化 + 淘汰**——降低 HBM 占用、提高可行 batch；收益体现为吞吐，而不是单条算术。
5. **短或低风险回复走更小更便宜的模型**——路由，配质量护栏。

点睛一句：180 条消息的会话里，**缓存就是成本模型**。

**速记**：单条 = 约 7k prompt + 120 输出 token，prefill token 量约为回复的 58 倍；最大杠杆是 KV/前缀缓存（每轮只剩 ~40 新 token），其次缩短有效 prompt（人设+摘要属产品决策）、连续批处理、KV 量化淘汰提升 batch、按场景路由小模型。


---

## ElevenLabs

### 1. 服务实时 TTS 与服务文本 LLM 是不同的容量问题。为什么，以及你如何做容量规划？

<!-- EN: Serving real-time TTS is a different capacity problem from serving a text LLM. Why, and how do you plan capacity? -->

**参考答案**

文本 LLM 的容量按**每 GPU 的 token/s**（连续批处理）衡量；TTS 的容量按**每 GPU 的并发实时流**衡量。四个结构性差异：

| 维度 | LLM | 实时 TTS |
| :--- | :--- | :--- |
| 工作单位 | token（可变成本） | 每秒音频（每秒固定成本） |
| 延迟 SLO | TTFT（可 500ms 以上） | **TTFA 约 150–300ms，且每一块都算** |
| 批处理 | 连续批处理、大 batch | batch 约 1–4；延迟不允许排队 |
| 输出 | 文本，小 | 音频，每流 10–100 KB/s |

核心指标是 **RTF**（实时因子）= 每音频秒消耗的计算秒。RTF 0.1 的模型若能批处理，每 GPU 等价可撑约 10 路流；batch size 1 时只能撑 1/0.1，且被声码器/解码器抖动吃掉大量尾部。

容量规划步骤：

1. 在**目标并发**下测 **p99 RTF**，不是均值——自回归编解码解码器随 batch 非线性退化。
2. 换算：`GPU 数 = 峰值并发流 × p99_RTF ÷ 0.7`（30% 余量给突发、抢占、滚动发布）。
3. 加按租户流级上限的准入控制器；排队比快速 429 更糟，因为已排队的流已经错过了 TTFA。
4. 建模日内峰值：语音 agent 集中在工作时段，所以按峰值采购，让批量任务（非交互配音、有声书）吸收低谷。

**陷阱**：按均值 RTF 做的容量计划看起来没问题，然后在下午三点崩掉——尾部由长时间占用流的长文本请求主导，而不是请求率。

**速记**：TTS 容量单位是"并发实时流"而非 token/s，SLO 是每块音频的 TTFA；按 p99 RTF 算 GPU（留 30% 余量），做按租户的流级准入控制——排队比快速拒绝更糟。


---

## Abridge

### 1. 病历记录应该在临床医生离开诊室之前就准备好。帮我做延迟预算，并告诉我钱花在哪里。

<!-- EN: The note should be ready before the clinician leaves the room. Build me the latency budget, and tell me where the money goes. -->

**参考答案**

假设就诊结束，医生要在 **60–90 秒**内复核并签字。

| 阶段 | 预算 | 备注 |
| :--- | :--- | :--- |
| 音频上传 / 流式 ASR 定稿 | 3–8s | 就诊中已流式；只有尾部在关键路径上 |
| 说话人归属 + 章节检测 | 1–2s | 在分离后的转写上跑廉价分类器 |
| 笔记生成（按章节并行） | 8–25s | 主导项 |
| 落地/校验通过 | 2–5s | 断言 → 片段检查，可与生成重叠 |
| 渲染 + EHR 预填 | 1–3s | |

改变预算的关键洞察：**在就诊过程中就生成。** 20 分钟的就诊产出约 3 千词转写；到第 10 分钟就能用 partial 起草 HPI 与 ROS，结束时只重生成评估/计划。有效的事后延迟降到几秒。

钱花在哪——**不是** ASR：

| 成本 | 驱动 | 控制 |
| :--- | :--- | :--- |
| LLM 输入 token | 整段转写（5–10k token）× **每章节一次调用** → 同一份转写被计费 5–8 次 | 按章节检索相关片段；前缀缓存 |
| LLM 输出 token | 每份笔记约 600–1200 词 | 按章节设上限；生产环境不上思维链 |
| 重生成 | 医生编辑触发重写 | 按 (转写哈希, 章节, 指令) 缓存 |
| ASR GPU | 音频分钟 × RTF | 非紧急尾部批处理；分离用小模型 |
| 存储/出网 | 音频 + 转写留存 | 留存策略既是成本也是合规杠杆 |

典型形状：输入 token 以 3–5 倍占主导。**按章节检索范围化 prompt**（只发与该章节相关的片段）是最大杠杆——它同时通过去掉干扰对白提升质量。

**速记**：预算 60–90s，关键不是加速而是**在问诊过程中增量生成**，收尾只重写评估/计划；成本大头是"整段转写 × 每节一次调用"的输入 token，用按节检索相关片段 + 前缀缓存最省。


---

## Figure AI

### 1. 一位同事想把语义层搬到云上，以便使用大得多的模型。讲讲这个方案的延迟预算。

<!-- EN: A colleague wants to move the semantic layer to the cloud so you can use a much bigger model. Walk me through the latency budget. -->

**参考答案**

先按频率分层——这个论证只对慢回路成立。

| 回路 | 频率 | 截止时间 | 能上云？ |
| :--- | :--- | :--- | :--- |
| 关节伺服 / 平衡 | 500–1000Hz | 1–2ms | 绝不 |
| 全身控制（WBC） | 100–500Hz | 2–10ms | 绝不 |
| 视觉运动策略（VLA） | 5–50Hz | 20–200ms | 临界 |
| 语义层（物体身份、任务规划、对话） | 0.2–2Hz | 500ms–2s | 可以，需降级 |

语义层的云上预算：

| 项 | 典型值 |
| :--- | :--- |
| 上行（压缩 RGB-D + 状态） | 20–60ms |
| 推理服务排队 | 10–300ms（**杀手**） |
| 大模型推理 | 150–800ms |
| 下行（计划/语义结果） | 20–60ms |
| **合计 p50 / p99** | **约 0.3s / 1.5–3s** |

物理含义：以 1 m/s 行进时，p99 的 2s 意味着语义决策描述的是机器人 2 米前的位置。对"杯子在哪个抽屉"没问题；对"有人正踏进通道吗"不可接受。

说"可以"之前的要求：

1. **异步、非阻塞**：语义层提议，本地栈决断。绝不把云调用放进控制路径。
2. **时效记账**：每个结果带时间戳，本地规划器拒绝早于 X 的结果并降级。
3. **真正可用的本地降级**——机上小模型覆盖安全关键子集，而不是一个只会抛异常的桩。
4. **断连策略**：保持、减速、还是靠边停。按行为分别决定，并通过在任务中拔网线来测试。
5. **带宽/功耗**：电池供电的人形以 10–30fps 推视频要真金白银的能量与热预算——这常常才是真正的阻塞点，而不是延迟。

结论：搬*语义与规划*，绝不搬*反应式控制*。安全层完全留在机上。

**速记**：先按回路分频——伺服/WBC 绝不能上云，VLA 临界，只有 0.2–2Hz 的语义层可以；云的杀手是排队（p99 1.5–3s），所以必须异步、带时间戳与过期拒绝、有真正可用的本地降级，并按行为分别定义断连降级。


---

## Waymo

### 1. 为车载技术栈做算力和延迟预算。当模型变大时，哪里会先撑不住？

<!-- EN: Budget the compute and latency for the onboard stack. What breaks when a model gets bigger? -->

**参考答案**

从执行器截止时间倒推。20 m/s 下，100ms 端到端延迟 = 2 米行程——这定了基调。

| 阶段 | 频率 | 延迟预算 | 运行位置 |
| :--- | :--- | :--- | :--- |
| 传感器采集 + 时间同步 | 10–20Hz | 10–20ms | 硬件打时间戳 |
| 感知（检测/分割/跟踪） | 10Hz | 30–50ms | GPU/加速器 |
| 传感器融合（激光+相机+毫米波 → BEV） | 10Hz | 10–20ms | GPU |
| 预测（agent，8s 时域） | 10Hz | 20–40ms | GPU，可流水线 |
| 规划 | 10Hz | 10–30ms | CPU/GPU |
| 控制 | 100Hz | 5–10ms | CPU，硬实时 |
| **传感到执行合计** | — | **约 100–150ms** | 含 2 倍安全余量 |

然后是被忘掉的约束：

1. **功耗与散热**才是硬约束，不是 FLOPs。密封后备箱里 500W 加速器会降频；持续吞吐可能是峰值的 60–70%，并在 30 分钟行程中变化。
2. **显存带宽**主导 BEV/大注意力模型；模型可以算力很轻却被带宽卡住。
3. **尾延迟（p99.9）**而非均值——40ms 的模型若偶发 400ms，不能拖停流水线。各阶段要有界队列与**丢弃过期**语义。
4. **确定性与 WCET** 供安全论证：最坏执行时间必须有界；动态批处理在这里是负债。

模型变大先崩的地方：

| 崩点 | 症状 |
| :--- | :--- |
| p99 延迟 | 振荡、晚刹、"犹豫"行为 |
| 热预算 | 性能在行驶中途下降，而不是启动时 |
| 缺少兜底 | 没有更小的模型可降级 |
| 重新验证成本 | 每次尺寸变化都作废安全证据 |
| 流水线耦合 | 一个慢阶段迫使所有消费者缓冲 → 内存增长、抖动 |

缓解：把知识蒸馏到固定预算的学生模型、输入 ROI 裁剪与级联（便宜检测器 → 昂贵校验器）、逐层校准的量化（INT8/FP8），以及**延迟感知调度器**按工况选模型尺寸（停车时多算，高速时少算）。

**速记**：从执行器截止时间倒推——全程约 100–150ms，控制环硬实时 5–10ms；真正约束是**功耗/散热**与 p99 尾延迟，不是 FLOPs。模型变大先崩在尾延迟与热降频，必须配蒸馏、级联、量化与按工况选模型的调度器。
