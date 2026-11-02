# LLM 内部机制与架构

共 **73 题**，覆盖 **18 家**，另有 **跨公司通用** 一节。

> 考底层原理：attention 变体、位置编码、MoE 路由、并行与显存。

> 题目为**英文原文**（源仓库 Apache-2.0），**答案为中文整理**。每条附 `Source` 时给出原始出处。授权与归属见 [NOTICE.md](NOTICE.md)。

## 目录

| 公司 / 来源 | 题数 |
| :--- | :-: |
| 跨公司通用 | 16 |
| Anthropic | 3 |
| OpenAI | 3 |
| Meta | 2 |
| Mistral AI | 2 |
| Cohere | 1 |
| DeepSeek | 4 |
| Moonshot AI（Kimi） | 4 |
| 智谱 AI（GLM） | 5 |
| 阿里巴巴（Qwen） | 4 |
| Sarvam AI | 1 |
| Amazon / AWS | 1 |
| 字节跳动 | 5 |
| 腾讯 | 5 |
| 阿里巴巴（淘天 / 阿里云） | 4 |
| 滴滴 | 4 |
| 美团 | 5 |
| Hugging Face | 3 |
| Character.AI | 1 |


---

## 跨公司通用

### 1. 解释缩放点积注意力（scaled dot-product attention），以及 1/sqrt(d_k) 这个缩放因子为什么重要。

<!-- EN: Explain scaled dot-product attention and why the 1/sqrt(d_k) scaling factor matters. -->

**Source**: https://outcomeschool.com/blog/scaling-dot-product-attention · https://outcomeschool.com/blog/math-behind-attention-qkv

**参考答案**

注意力是一次软的键值查询：给每个 key 相对 query 打分，softmax 成权重，返回 value 的加权和。

$$\text{Attn}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

对独立、近似单位方差的 $q,k\in\mathbb{R}^{d_k}$，点积的方差是 $d_k$，标准差随 $\sqrt{d_k}$ 增长。$d_k=128$ 时 logit 的 σ≈11。这么宽的分布上做 softmax 实际等于 one-hot：最大的 logit 以巨大优势胜出。

| | 不缩放 | 除以 $1/\sqrt{d_k}$ |
| :--- | :--- | :--- |
| logit 的 σ | $\sqrt{d_k}$ | 1 |
| softmax 熵 | 接近 0 | 可用区间 |
| 穿过 softmax 的梯度 | 非赢家≈0 | 健康 |
| 行为 | 硬 argmax，训练停滞 | 软混合 |

所以缩放因子不是装饰：它把注意力留在 softmax 有梯度、真能混合多个 value 的区间。这个推导假设的是初始化时的统计量；训练后模型会漂移，所以现代架构常加 **QK-norm**（按头归一化 q、k）或 **logit soft-capping**（Gemma 2 用 tanh 把 logit 限在 ±50），而不是死守这个常数。

什么时候可以不要它：如果你把 q 和 k 归一化成单位长度，方差论证就不适用了。

**速记**：q·k 的方差是 d_k，不缩放时 logit 标准差 √d_k 让 softmax 退化成 one-hot、梯度消失；1/√d_k 把方差拉回 1。实践中靠 QK-norm 或 logit soft-cap 补，而不是死守这个常数。

### 2. 什么是 KV cache？在大规模场景下它带来哪些显存影响？请推导公式。

<!-- EN: What is the KV cache, and what are its memory implications at scale? Derive the formula. -->

**Asked at**: OpenAI · xAI · Mistral AI · Amazon · Apple · NVIDIA · Together AI · Character.AI · ByteDance · Tencent · Meituan · Alibaba (Taobao/Tmall, Alibaba Cloud)
**Source**: https://outcomeschool.com/blog/kv-cache-in-llms · https://outcomeschool.com/blog/kv-cache-compression

**参考答案**

decode 是自回归的，所以历史 token 的 K 和 V 永远不变。缓存把它们存一次；之后每步只为新 token 投影 q，并在缓存的 K/V 上做注意力。

$$\text{bytes}=2\cdot L\cdot H_{kv}\cdot d_{head}\cdot T\cdot B\cdot b$$

其中 2 = {K,V}，$L$ 层数，$H_{kv}$ KV 头数，$T$ 上下文长度，$B$ 批大小，$b$ 每元素字节。

具体：Llama-3-8B 有 $L=32$、$H_{kv}=8$、$d_{head}=128$、fp16 → $2\cdot32\cdot8\cdot128\cdot2 = 131{,}072$ B = **每 token 128 KiB**。32k 上下文对**单条**序列就是 4 GiB。70B（$L=80$）每 token 320 KiB → 32k 时 10 GiB。

| 影响 | 后果 |
| :--- | :--- |
| 与 $T\times B$ 线性 | 上下文一长，并发就塌 |
| 与权重分开算 | 权重装得下、KV 装不下——这才是常见 OOM 原因 |
| 每个 decode 步都得读 | decode 变成 HBM 带宽受限，而非 FLOP 受限 |
| 分配粒度 | 按请求预留连续 buffer 会因碎片浪费 20–40% → PagedAttention |

可用的杠杆：GQA/MLA 砍 $H_{kv}$ 或每 token 宽度；KV 量化（fp8 大约减半，约 4-bit 以下开始有明显质量损失）；共享 system prompt 用前缀缓存；长对话用驱逐/压缩。

**速记**：公式 `2·L·H_kv·d_head·T·B·b`；Llama-3-8B 每 token 128 KiB，32k 上下文单条就 4 GiB。KV 与权重分开算，它是并发上不去的真凶，也是分页/量化/GQA 存在的理由。

### 3. 什么是 Multi-Query Attention（MQA）和 Grouped-Query Attention（GQA）？它们各自牺牲了什么？

<!-- EN: What are Multi-Query Attention (MQA) and Grouped-Query Attention (GQA), and what do they trade away? -->

**Asked at**: Meta · Mistral AI · ByteDance · Meituan · DiDi
**Source**: https://outcomeschool.com/blog/grouped-query-attention

**参考答案**

它们回答同一个问题：**多少个 query 头共享一个 K/V 头？**

| 变体 | KV 头数 | KV 缓存 | 质量 | 解码速度 |
| :--- | :--- | :--- | :--- | :--- |
| MHA | $H$（每个 q 头一个） | 基线 | 最好的参照 | 最慢 |
| GQA | $G$，$1<G<H$ | $\times G/H$ | ≈MHA | 接近 MQA |
| MQA | 1 | $\times 1/H$ | 退化 | 最快 |

交换是显式的：**你在删掉 key/value 子空间的多样性**。MQA 让所有头共用一份 K/V，头之间只能在 query 上有差异，训练也更不稳——常需要更长训练或更谨慎的初始化来补回质量。GQA 是工业界最终选的插值：通常 8 个 KV 头对 32–64 个 query 头，即 8 倍缓存削减而分数保持 MHA 水平。

两个实践要点：

1. **GQA 可以迁移**。已训好的 MHA 模型，把每组内的 KV 头做均值池化、再用原算力几个百分点继续预训练，就能变成 GQA——"我们改用 GQA"的迁移大多这么干，而不是从头训。
2. **头数是服务端参数，不只是建模参数**。KV 头越少，每步工作集越小，所以吞吐收益主要出现在高并发和长上下文；batch 1 时差别很小。

**速记**：本质是"多少个 query head 共享一份 K/V"。MQA 共享 1 份、GQA 共享 G 份：缓存按 G/H 缩，质量接近 MHA。MQA 掉点且训练不稳；GQA 可用均值池化 KV head 再短训迁移，是工业界默认。

### 4. 什么是 Multi-head Latent Attention（MLA）？DeepSeek 为什么要引入它？

<!-- EN: What is Multi-head Latent Attention (MLA) and why did DeepSeek introduce it? -->

**Asked at**: DeepSeek · Moonshot AI · ByteDance · Meituan
**Source**: https://outcomeschool.com/blog/kv-cache-compression

**参考答案**

MLA 把**缓存大小**与**头数**解耦。它不缓存 K 和 V，而是每个 token 只缓存一个低秩 latent。

每层每 token：

1. 把隐状态下投影成 latent $c_t\in\mathbb{R}^{512}$，缓存它。
2. 使用时把 $c_t$ 上投影成各头的 K 和 V。
3. 对**另一个小 key** $k^R_t\in\mathbb{R}^{64}$ 施加 RoPE 并同样缓存——因为旋转不与低秩吸收交换，位置信息必须走在 latent 之外。

缓存宽度变成 $512+64=576$ 个 float，而 MHA 是 $2\cdot H\cdot d_{head}$（上万）。DeepSeek-V2 报告约 93% 的 KV 削减，以及相对其稠密前代 5.7× 的生成吞吐。

最巧的是**权重吸收**：$W^{Q}W^{UK}$ 和 $W^{UV}W^{O}$ 可以预先乘好，于是 decode 直接用 q 乘 latent，K/V 根本不用物化出来。

代价：prefill 多了投影、要自定义 kernel，且 latent 是要学出来的——不像 GQA，你没法把现成 MHA 检查点便宜地转过来。长上下文 × 高并发主导成本时选 MLA；想要开箱即用的 kernel 就用 GQA。

**速记**：MLA 缓存一个 512 维 latent + 64 维 decoupled RoPE key，而不是 K/V；靠权重吸收把 up-projection 折进 Q/O，解码时直接和 latent 相乘。换来 MHA 级质量 + MQA 级缓存，代价是 kernel 复杂、无法从已有 MHA 便宜迁移。

### 5. 解释 FlashAttention。它并没有减少 FLOPs，为什么反而更快？

<!-- EN: Explain FlashAttention. It does not reduce FLOPs, so why is it faster? -->

**Asked at**: Together AI · Tencent
**Source**: https://outcomeschool.com/blog/decoding-flash-attention

**参考答案**

因为这里的注意力是**访存受限，不是算力受限**，而 FlashAttention 是一个 IO 感知的精确算法。

朴素 kernel 会把 $S=QK^\top$（$N\times N$）和 $P=\text{softmax}(S)$ 物化到 HBM：写 $N^2$、为 softmax 读 $N^2$、再写、再为 $PV$ 读。这就是在约 3 TB/s 带宽上做好几趟 $N^2$ 往返，而张量核在一边空转。

FlashAttention 把 Q 切成能进 SRAM（约 100–200 KB）的块，流式读入 K/V 块，并维护**运行最大值 $m$ 与运行和 $\ell$**——online softmax——边走边对累加器重标定。那个 $N^2$ 矩阵从头到尾不存在。反向传播时用缓存的统计量重算分块，而不是把它存下来。

| | 朴素 | FlashAttention |
| :--- | :--- | :--- |
| FLOPs | 相同 | 相同 |
| HBM 流量 | $O(N^2)$ 写+读 | $O(N)$ |
| 激活显存 | $O(N^2)$ | $O(N)$ |
| 结果 | 精确 | 精确 |

经验法则：fp16 注意力需要 ≳300 FLOP/byte 才算力受限；物化版本大致是 $O(1)$。

收益消失的地方：**decode**，此时 $q$ 只有一个 token，没有 $N^2$ 的块可融合——要改用 split-KV（flash-decoding）；以及 head dim 超过 256，FA2 不支持。

**速记**：FlashAttention 不省 FLOPs，省的是 HBM 往返：分块进 SRAM + online softmax，永不落地 N×N 矩阵，反向重算。所以它是 IO-aware 而非近似。解码阶段 q 只有一行，收益消失，要换 flash-decoding。

### 6. Byte Pair Encoding 是如何工作的？它有哪些失效场景（数字、代码、非拉丁文字）？

<!-- EN: How does Byte Pair Encoding work, and what are its failure modes (numbers, code, non-Latin scripts)? -->

**Asked at**: Alibaba · Sarvam AI · Hugging Face
**Source**: https://outcomeschool.com/blog/bpe-in-llms · https://www.youtube.com/watch?v=sK2s9I84EVI

**参考答案**

BPE 是把贪心压缩算法变成分词器：

1. 从一个基础字母表开始（字节，或字符 + `<unk>`）。
2. 在整个语料上统计相邻对。
3. 把最高频的一对合并成新符号。
4. 重复到词表预算用完。
5. 编码时按学到的排名顺序、从左到右重放合并。

合并由**频次驱动**，所以任何稀有形态都会被切碎。整个失败面就在这：

| 输入 | 失败 | 缓解 |
| :--- | :--- | :--- |
| 数字 `1234` / `1235` | 变成不同 token，没有数位对齐；算术与按列抄写会坏 | 数位单独切（Llama-2 的正则），或强制 `[0-9]` |
| 印地语 / 泰米尔语 | UTF-8 每字符 3 字节 → 2–3 token/字；相对英语的 fertility 高 3–4 倍 | 更大词表、语言专属分词器、字节回退 |
| 中文 | 约 1.5–2 token/字 → 有效上下文缩水 | 10 万+ 词表 |
| 代码 | 长标识符切分不一致；缩进单独成 token；空白敏感语言吃亏 | 代码感知的预分词（GPT-4 式正则） |
| 提示 | `"hello"` ≠ `" hello"`——前导空格属于 token | few-shot 格式必须逐字节一致 |

另外：训练不足的合并会产生**glitch token**，它们表现为垃圾 embedding；而词表是冻结的——换分词器会废掉下游所有 embedding 矩阵。

**速记**：BPE = 按频次反复合并最常见相邻对，编码时按 merge 排名重放。频率驱动 → 稀有形态必被切碎：数字不对齐、印地语 2~3 token/字、代码标识符抖动、`" hello"` 与 `"hello"` 不同 token。词表一旦定死，换 tokenizer 等于废掉 embedding。

### 7. transformer 中的位置编码是什么？它经历了怎样的演进（正弦 → 可学习 → RoPE → ALiBi）？

<!-- EN: What is positional encoding in transformers, and how has it evolved (sinusoidal → learned → RoPE → ALiBi)? -->

**Source**: https://outcomeschool.substack.com/p/positional-embeddings-in-llms · https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding

**参考答案**

自注意力对排列不变，所以顺序必须被注入。各种设计的差别只在**从哪里进**和**怎么外推**。

| 方案 | 从哪进 | 参数 | 外推 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| 正弦 | 加到输入 embedding | 无 | 差 | 固定的几何频率，base 10000 |
| 可学习绝对位置 | 加到输入 embedding | $L_{max}\times d$ | 无 | GPT-2；到 $L_{max}$ 是硬墙 |
| 相对偏置（T5） | 按头加到注意力 logit，分桶 | 小 | 中等 | 学出来的桶，不是连续距离 |
| RoPE | 按头旋转 q 和 k | 无 | 需缩放 | 结构上就是相对位置，随距离衰减 |
| ALiBi | 对 logit 加线性惩罚，每头一个斜率 | 无 | 好 | 最便宜；固定先验"越远越不相关" |

RoPE 胜出的原因是：它不需要偏置张量就能拿到相对位置——两个旋转向量的点积只依赖 $m-n$。它还自带一个归纳偏置：分数随距离衰减，并且能与分页/流式 kernel 组合。

有意思的反向潮流是 **NoPE**：因果解码器仅凭因果掩码就能推出位置，一些消融显示长度泛化并不更差。现代做法是 RoPE 加一套扩展配方，而不是固定的绝对位置表。

**速记**：位置编码的差别在"注入位置"和"外推能力"：正弦/可学习绝对位置加到输入上，到 max_len 就硬墙；RoPE 旋转 q/k 使点积只依赖相对距离且无参数；ALiBi 直接给 logit 加线性惩罚，最便宜也最敢外推。

### 8. 解释 RoPE，以及位置插值（position interpolation）和 YaRN 如何把上下文扩展到训练长度之外。

<!-- EN: Explain RoPE and how position interpolation / YaRN extend context beyond the trained length. -->

**Asked at**: Meta · Moonshot AI · Alibaba · Tencent · Meituan
**Source**: https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding

**参考答案**

RoPE 把每个头的 $d$ 维向量拆成 $d/2$ 对，在位置 $m$ 处把第 $i$ 对旋转 $m\theta_i$，其中 $\theta_i=\text{base}^{-2i/d}$。因为旋转是正交的，$\langle R_mq, R_nk\rangle$ 只依赖 $m-n$：零参数拿到相对位置。频率几何衰减，所以低 $i$ 的维度转得快（管局部顺序），高 $i$ 的维度转得慢（管全局位置）。

超出训练长度后，朴素修法是**位置插值**：把 $m\mapsto m\cdot L_{train}/L_{new}$ 全量压缩。它有效，但破坏了高频维：相邻 token 变得几乎不可区分，局部语法退化，而且必须微调。

| 方法 | 缩放什么 | 代价 |
| :--- | :--- | :--- |
| PI | 位置，统一压缩 | 破坏高频维，必须全参微调 |
| NTK-aware | base（频率） | 保住高频，但粗糙 |
| **YaRN** | 逐维斜坡 + 温度 | 约 400 步，约预训练量的 0.1% |

YaRN 有两点值得记住：**NTK-by-parts**（波长小于训练长度的维不动；比新上下文还长的维做插值；中间用斜坡混合）以及**注意力温度**因子 $\sqrt{1/t}$，用来补偿在更多 token 上做注意力带来的熵塌缩。温度那部分是大家忘掉的——没有它，扩展后的模型会过度自信、困惑度变差。

**速记**：RoPE 旋转 q/k，使内积只依赖相对位置；高频维管局部、低频维管全局。PI 统一压缩会毁掉高频；YaRN = NTK-by-parts 按维分段插值 + 注意力温度补偿熵塌缩，只需几百步微调。

### 9. Chinchilla 缩放定律说了什么？它与更早的缩放直觉有何不同？

<!-- EN: What do the Chinchilla scaling laws say, and how do they differ from earlier scaling intuitions? -->

**Asked at**: Anthropic

**参考答案**

Chinchilla 拟合出 $L(N,D)=E+A/N^{\alpha}+B/D^{\beta}$，结论是：在**固定算力预算**下，参数量与 token 数应当同比例增长——大致 **每参数 20 个 token**。标志性结果：70B 模型训 1.4T token，在同算力下打败 280B 训 300B。

更早的 Kaplan 式直觉正好相反：算力都花在参数上，数据保持相对少。这个分歧很大程度是方法学问题——最初的拟合在改变 token 数时没有固定学习率调度，使得额外数据的价值被低估了。

| | Kaplan 时代 | Chinchilla |
| :--- | :--- | :--- |
| 分配 | 快堆 $N$，慢堆 $D$ | $N$ 与 $D$ 同步 |
| token/参数 | 约 2–5 | 约 20 |
| 对数据的假设 | 事实上稀缺 | 是紧约束 |

现代的转折：**算力最优不等于部署最优**。参数只付一次钱，但推理成本按生成 token 永远付下去。所以生产模型故意**超训**——8B 喂 15T token，约 1900 token/参数，是 Chinchilla 比例的约 100 倍——用训练 FLOPs 换一个永久更便宜的模型。重复数据超过约 4 个 epoch 就开始掉质量，这也是为什么约束最终落在数据质量而不是原始体量上。

**速记**：Chinchilla 说固定算力下参数量与 token 数同比例涨，约 20 token/参数；Kaplan 时代偏向堆参数，部分源于学习率调度没对齐。但部署算的是终身推理成本，所以生产模型故意"超训"（8B 喂 15T），代价是重复数据超过约 4 轮会掉点。

### 10. 什么是混合专家（MoE）架构？它如何在不让 FLOPs 随之增长的情况下扩展容量？

<!-- EN: What is a mixture-of-experts architecture and how does it scale capacity without scaling FLOPs? -->

**Asked at**: Mistral AI · Cohere · DeepSeek · Moonshot AI · Zhipu AI · Alibaba · Tencent · Meituan · Alibaba (Taobao/Tmall, Alibaba Cloud)
**Source**: https://outcomeschool.com/blog/mixture-of-experts

**参考答案**

MoE 保持注意力块稠密，把每个 FFN 换成 $E$ 个并行专家加一个 router：

$$y=\sum_{i\in\text{top-}k} g_i\,E_i(x),\qquad g=\text{softmax}(\text{top-}k(W_r x))$$

每个 token 只激活 $k$ 个专家（通常 2 个），于是 **FLOPs 跟活跃参数走，显存跟总参数走**。容量——这层能表示的不同函数数量——随 $E$ 增长；单 token 的计算量不变。

| 收益 | 代价 |
| :--- | :--- |
| 同等训练 FLOPs 下更多参数 | 所有专家都必须常驻显存 |
| 每 FLOP 的质量更高 | 专家并行里的 all-to-all 通信 |
| 条件计算 | 路由不稳定、负载塌缩 |

真正的工程在均衡上。朴素 top-2 router 会塌缩到少数专家上；修法是辅助负载均衡损失、带 token 丢弃的容量因子，或 DeepSeek-V3 的**无辅助损失**路由 logit 偏置调整。DeepSeekMoE 还把专家切得更细，并保留一个常开的**共享专家**承载通用知识。

必须说出口的失败模式：小 batch 下，MoE 可能比同等活跃 FLOPs 的稠密模型**每 token 更慢**，因为每个 token 都要散到不同 GPU 上。MoE 只在快速互连下的大 batch 才赢。

**速记**：MoE 用 router 做 top-k 选择，只激活少数专家 → 参数量（容量）涨、FLOPs 不涨。真正难的是负载均衡（辅助损失 / 无辅助损失的 bias 调整）、all-to-all 带宽，以及小 batch 下反而不如稠密模型快。

### 11. 解释预训练（pre-training）、监督微调（SFT）和偏好优化（preference optimisation）之间的区别。

<!-- EN: Explain the difference between pre-training, supervised fine-tuning and preference optimisation. -->

**Asked at**: Meta · Scale AI
**Source**: https://outcomeschool.com/blog/decoding-instructgpt · https://outcomeschool.com/blog/reinforcement-learning-from-human-feedback-rlhf

**参考答案**

三个阶段、三种信号、三种它们真正能改变的东西。

| 阶段 | 数据 | 目标 | 规模 | 装进去什么 |
| :--- | :--- | :--- | :--- | :--- |
| 预训练 | 万亿级原始 token | 下一 token 交叉熵 | 10²²–10²⁵ FLOPs | 知识、语法、推理底座 |
| SFT | 1 万–100 万条示范 | 只在回答 token 上算交叉熵 | 小时–天 | 格式、人格、工具协议 |
| 偏好 | chosen/rejected 对 | RM + PPO 带 KL，或 DPO/GRPO | 天 | 两个都合法的答案里选哪个 |

关键的不对称：**SFT 加不进预训练没学到的知识** —— 它只是把潜在能力显性化、格式化。偏好优化也加不了能力；它重加权分布，所以这个阶段既修语气、拒答校准、啰嗦度和安全，也是**弄坏**东西的阶段（奖励黑客、谄媚、熵塌缩）。

顺序很重要。SFT 在偏好之前，因为偏好方法假设模型已经能产出格式良好的候选。而 1000 条精挑的 SFT 常常打败 10 万条噪声数据——大规模 SFT 的典型失败是风格塌缩：模型在所有领域都模仿标注员的腔调。

**速记**：预训练给知识和能力，SFT 只改"怎么答"（暴露而非新增能力），偏好优化只改"两个都对时选哪个"（语气、拒答、安全），也最容易引入谄媚和奖励作弊。顺序不能反，SFT 数据质量远胜数量。

### 12. 比较 greedy、beam search、top-k、top-p 和 temperature 采样。它们各自在什么情况下会失败？

<!-- EN: Compare greedy, beam search, top-k, top-p and temperature sampling. When does each fail? -->

**Asked at**: Google DeepMind · Apple · Perplexity
**Source**: https://outcomeschool.com/blog/how-does-temperature-control-llm-output · https://outcomeschool.com/blog/how-do-top-k-and-top-p-sampling-work

**参考答案**

温度重塑分布；各种截断方法决定尾巴留多少。

| 方法 | 机制 | 何时失效 |
| :--- | :--- | :--- |
| 贪心 | argmax | 复读循环、无多样性；错过更好的全局续写 |
| Beam search | 保留 $b$ 条联合概率最高的前缀 | 开放对话里寡淡、重复；机器翻译与受限解码才是它的主场 |
| 温度 | logits 除以 $T$ | $T\!\to\!0$ 退化为贪心；$T>1.2$ 开始语无伦次，而它重标的是**整条**尾巴 |
| top-k | 保留最可能的 $k$ 个 | 固定 $k$ 无视分布形状：分布平坦时截掉有效质量，分布尖时仍放进垃圾 |
| top-p | 保留累积概率达 $p$ 的最小集合 | 高温下核会膨胀；$p$ 与模型强相关 |
| min-p | 保留 $p_i>\text{min\_p}\cdot p_{max}$ 的 token | 较新，实战检验不足 |

多数技术栈的处理顺序：**温度 → top-k → top-p**，从左到右作用在 logits 上。

两条实践提醒。第一，采样参数不可跨模型迁移——在某个分词器的 logit 尺度上有效的 $p=0.9,T=0.7$，换一个模型行为就变了。第二，可复现需要种子**以及**相同的 batch 组合和 kernel；批量矩阵乘里的浮点非确定性意味着同一个种子也可能漂。要输出 JSON 或代码，用语法约束，不要靠调温度。

**速记**：temperature 改分布形状，top-k/top-p/min-p 截尾巴；顺序是 temp → top-k → top-p。greedy 会复读，beam 在开放对话里寡淡，top-k 的固定 k 对分布形状不敏感，高温下 top-p 的核会膨胀。结构化输出应上语法约束而不是调温度。

### 13. 长上下文中的“迷失在中间”（lost-in-the-middle）问题是什么？你如何应对？

<!-- EN: What is the lost-in-the-middle problem in long contexts and how do you address it? -->

**Asked at**: Moonshot AI
**Source**: https://outcomeschool.com/blog/lost-in-the-middle-problem-in-llms

**参考答案**

准确率对"金标证据在上下文中的位置"呈 **U 形**：放开头或结尾最好，放中间最差。即使检索完美命中它依然存在，所以这是注意力/利用失败，不是召回失败。

成因，按你能动手的程度排序：

1. **训练分布** —— 答案通常出现在文档靠前位置，模型因此学出了位置先验。
2. **注意力汇与近因** —— 最前面的 token 和最近的 token 获得不成比例的注意力。
3. **干扰** —— 20 个干扰项在场时，中间恰好是"最像但错"的段落聚集处。

确实能改变数字的手段：

| 手段 | 为什么有效 |
| :--- | :--- |
| 重排后把最佳证据放最前**并且**在结尾再说一遍 | 命中 U 形的两端 |
| 要求先引用片段再作答 | 逼模型真去读，而不是扫一眼 |
| 压缩分块 / 逐块摘要 | 把"中间"变短 |
| 更少但更好的块 | 减少干扰 |
| 显式 id 与标题 | 给模型可回指的抓手 |

**量化它**：拿一个多文档 QA 集，置换金标段落的位置，画准确率曲线。曲线平坦说明你没这问题；下陷 10–20 个点，那"1M 上下文"的营销也救不了你。

**速记**：金标证据放开头或结尾准确率最高、放中间明显掉点，且检索全对时依然存在——是"用不上"不是"找不到"。对策：重排后首尾各放一次、先引用再回答、压缩分块、减少干扰项；用位置置换实验量化。

### 14. 为什么现代 transformer 把 LayerNorm 放在 block 之前（pre-block）？RMSNorm 又是什么？

<!-- EN: Why is LayerNorm placed pre-block in modern transformers, and what is RMSNorm? -->

**Source**: https://outcomeschool.com/blog/rmsnorm-root-mean-square-layer-normalization

**参考答案**

原始 Transformer 在残差相加**之后**归一化（post-LN）：$x=\text{LN}(x+F(x))$。这等于在残差高速路上放了一个归一化，每层梯度都必须穿过 LN；深栈需要学习率 warmup 和谨慎初始化，否则直接发散。

pre-LN 把归一化挪进分支里：$x=x+F(\text{LN}(x))$。残差路径变成纯恒等，梯度直通 embedding，不用 warmup 也稳定。代价是残差流的范数随深度增长（约 $\sqrt{L}$），所以 **LM head 之前必须有一个 final norm** —— 漏掉它，logits 会随深度爆炸。

RMSNorm 是现在几乎到处都在用的更便宜的归一化：

$$y=\frac{x}{\sqrt{\frac{1}{d}\sum x_i^2+\epsilon}}\odot g$$

不减均值、不加偏置，只有一个学出来的增益向量。经验发现是：归一化真正起作用的是**重新缩放**，不是重新中心化。它每元素省掉一次归约和一次减法（归一化本身快一位数到低两位数的百分比），质量持平。

相关：一些栈会加 **QK-norm**（对每头的 q、k 做 RMSNorm），以便在高学习率和长上下文下把注意力 logit 控住——同一个技巧，用在 softmax 饱和的地方。

**速记**：post-LN 把归一化放在残差主路上，梯度必须穿过 LN，深网必须 warmup；pre-LN 把 LN 塞进分支，残差是恒等路径，稳定但不加最后的 norm 会导致 logits 随深度爆炸。RMSNorm 去掉减均值，只做缩放，更快且效果持平。

### 15. 解释 SwiGLU，以及为什么现代 LLM 的 MLP 块用门控激活取代了 ReLU/GELU。

<!-- EN: Explain SwiGLU and why gated activations replaced ReLU/GELU in modern LLM MLP blocks. -->

**Asked at**: Meta
**Source**: https://outcomeschool.com/blog/feed-forward-networks-in-llms

**参考答案**

SwiGLU 是门控线性单元：一条投影过 Swish 后，作为**乘法门**作用在另一条投影上。

$$\text{FFN}(x)=W_2\big(\text{Swish}(xW_1)\odot xW_3\big),\qquad \text{Swish}(z)=z\,\sigma(z)$$

为什么胜过 ReLU/GELU：固定的逐元素非线性，形状与输入无关；而门控让网络用同一个输入的另一个学习函数去调制每个特征——几乎不增加深度就多了一次乘性交互。Shazeer 2020 的消融发现，在同等参数和算力下门控变体困惑度更好，且在浅层区间一个门控层大致抵得上两个稠密层。

| | ReLU/GELU MLP | SwiGLU |
| :--- | :--- | :--- |
| 矩阵数 | 2（$d\!\to\!d_{ff}$、$d_{ff}\!\to\!d$） | 3（两个上投影） |
| $d_{ff}=4d$ 时参数 | $8d^2$ | $12d^2$ |
| 等参数取法 | $d_{ff}=4d$ | $d_{ff}=\tfrac{8}{3}d$ |

那个 $\tfrac{8}{3}$ 是最常被漏掉的细节：实现会把它取整到对硬件友好的倍数，所以 Llama 在 $d=4096$ 时用 $d_{ff}=14336$（3.5×，不是 2.67×）。

代价：张量并行时要多切一个矩阵乘，激活显存更多。极小的边缘模型上，朴素 GELU 依然更简单。

**速记**：SwiGLU 用一条 Swish 支路做乘法门控另一条支路，多了一次乘性交互；同参数同算力下困惑度更好。代价是多一个权重矩阵，所以 d_ff 从 4d 降到 8/3·d 保持参数不变（Llama 取 14336 是对齐硬件的近似）。

### 16. 请逐个张量地讲一遍 decoder-only transformer 的一次前向传播发生了什么。

<!-- EN: Walk me through what happens, tensor by tensor, in one forward pass of a decoder-only transformer. -->

**Asked at**: Anthropic
**Source**: https://outcomeschool.com/blog/decoding-transformer-architecture

**参考答案**

用 Llama-3-8B 的维度：$d=4096$、$L=32$、$H=32$ 个 q 头、$H_{kv}=8$、$d_{head}=128$、$d_{ff}=14336$、$V=128256$。

| # | 步骤 | 形状 | 备注 |
| :-: | :--- | :--- | :--- |
| 1 | 分词 → ids | $[B,T]$ | int64 |
| 2 | embedding 查表 | $[B,T,4096]$ | 纯 gather，无矩阵乘 |
| 3 | RMSNorm | $[B,T,4096]$ | pre-norm |
| 4 | QKV 投影 | $[B,T,4096]$、$[B,T,1024]$、$[B,T,1024]$ | $H_{kv}d_{head}=1024$ |
| 5 | RoPE | 同上 | 按位置旋转 q、k 的成对维度 |
| 6 | 因果注意力 | 隐式 $[B,32,T,T]$ | 每个 KV 头广播给 4 个 q 头；融合计算，从不物化 |
| 7 | 输出投影 + 残差 | $[B,T,4096]$ | |
| 8 | RMSNorm → gate/up → SiLU ⊙ → down | $[B,T,14336]\times2 \to [B,T,4096]$ | FLOPs 的大头在这里 |
| 9 | 残差相加，×32 层 | $[B,T,4096]$ | |
| 10 | final norm → LM head | $[B,T,128256]$ | logits |
| 11 | 采样 | $[B,T]$ | |

量级自检：参数 $\approx 12Ld^2+Vd\approx 8$B；每 token FLOPs $\approx 2N\approx16$ GFLOP。$T{=}4096$ 的 prefill 是一大批矩阵乘，约 64 TFLOP → 在几百 TFLOP/s 有效算力下是零点几秒。decode 是 $T{=}1$：同样 16 GFLOP，但它必须**读完整 16 GB 权重**，3 TB/s 下约 200 tok/s——带宽受限，这正是批处理存在的理由。

**速记**：按张量走一遍：ids → embedding → 每层（RMSNorm → QKV 投影 → RoPE → 因果注意力 → O 投影 → 残差 → RMSNorm → SwiGLU 上/门/下 → 残差）→ 末尾 norm → LM head → 采样。参数量 ≈12Ld²+Vd；prefill 是大矩阵乘，decode 是 T=1 的读权重，所以卡在带宽。


---

## Anthropic

### 1. Transformer 模型的关键组件有哪些？每个组件为什么重要？

<!-- EN: What are the key components of a Transformer model and why does each matter? -->

**Asked at**: Anthropic
**Source**: https://outcomeschool.com/blog/decoding-transformer-architecture

**参考答案**

| 组件 | 形状 / 作用 | 为什么重要 |
| :--- | :--- | :--- |
| Token embedding | `[V, d]` | 词表 → 向量；把模型绑死在 tokenizer 上 |
| 位置编码（今天多用 RoPE） | 逐头作用于 Q、K | 没有它模型对排列不敏感 |
| 多头自注意力 | 每头 `Q,K,V`，`d_head = d/h` | 唯一让 token 之间交换信息的地方 |
| MLP / FFN | `d → 4d → d` | 约 2/3 参数量；事实性召回主要在这里 |
| 残差 + LayerNorm | 现代栈用 pre-norm | 让深层梯度不消失 |
| LM head | `d → V`，常与 embedding 共享权重 | 隐状态变概率分布 |

回答「各自为什么重要」的分解方式：注意力是**路由**（谁读谁），MLP 是**计算/记忆**（类似键值存储），归一化负责**稳定**，残差流是两者共同写入的共享总线。

参数量在 `d_model` 量级上大致 MLP : 注意力 = 2:1，所以「每层大半是查找表」字面成立。KV cache 开销由 `2 · layers · heads · d_head` 决定，这正是 GQA/MLA 存在的理由。pre-norm 与 post-norm 的差别，就是「训得出 70B」和「看着它发散」的差别。换 tokenizer 会让 embedding 矩阵失效——那是一次迁移，不是改配置。

**速记**：注意力负责路由、MLP 负责记忆与计算、残差流是共享总线、LayerNorm 保稳定；KV cache 才是推理成本主因。

### 2. 解释无注意力（attention-free）的 transformer 架构及其取舍。

<!-- EN: Explain attention-free transformer architectures and their trade-offs. -->

**Asked at**: Anthropic

**参考答案**

「无注意力」= 保留 block 结构（norm → mixer → residual → MLP），只把 `O(n²)` 的 token 混合算子换成更便宜的。

| 家族 | 机制 | 优势 | 劣势 |
| :--- | :--- | :--- | :--- |
| 状态空间（S4、Mamba、Mamba-2） | 带输入依赖衰减的线性递推 | `O(n)`，每步常数内存 | 精确召回弱，难做逐字复制 |
| 线性注意力 | 核化 `softmax` → `φ(Q)(φ(K)ᵀV)` | `O(n)`，可结合扫描 | 小规模有质量差距 |
| RWKV | 时间混合 + 通道混合递推 | 推理高效、无 KV cache | 训练并行麻烦 |
| 卷积 / Hyena | 长隐式卷积核 | 局部 + 全局模式好 | 仍需大感受野 |
| 混合（注意力 + SSM） | 少量注意力层交错 | 便宜地找回召回能力 | 两套 kernel 的运维复杂度 |

真正的收益在**推理内存**：没有增长的 KV cache，长上下文服务变成平坦成本。真正的损失在**关联召回**：「把 4 万 token 前的那串字符原样复制过来」是注意力天生会做、SSM 只能近似的事。所以生产里的「无注意力」模型多是混合架构，纯 SSM 达到前沿质量仍是开放问题。别只拿 FLOPs 下结论：短上下文下 kernel 启动与矩阵乘效率常常盖过渐进优势。

**速记**：无注意力架构赢在推理常数内存与 O(n)，输在精确召回；生产多是混合架构，短上下文下渐进优势未必兑现。

### 3. 讲一讲与 LLM 架构相关的矩阵运算。

<!-- EN: Walk me through matrix manipulations relevant to LLM architectures. -->

**Asked at**: Anthropic
**Source**: https://outcomeschool.com/blog/math-behind-attention-qkv

**参考答案**

沿一个 token 的路径走，每一步都报形状（`B` batch、`T` token、`d` 模型维、`h` 头数、`d_h = d/h`）。

1. **Embed**：`X [B,T,d]`。
2. **投影**：`Q = X W_q`，`K`、`V` 同理，`W_q [d, h·d_h]`；代价 `3·B·T·d²`。
3. **拆分 + 注意力**：reshape 成 `[B,h,T,d_h]`；`S = QKᵀ/√d_h` → `[B,h,T,T]`；`A = softmax(S)`；`O = AV` → `[B,h,T,d_h]`。
4. **合并**：拼头成 `[B,T,d]`，用 `W_o [d,d]` 投影。
5. **MLP**：`X W_1` → `[B,T,4d]`，激活，`W_2` → `[B,T,d]`。

为什么除以 `√d_h`：`d_h` 个独立单位方差分量点积的方差是 `d_h`，除掉后 logits 保持单位尺度，softmax 不饱和、梯度不消失。

| 运算 | FLOPs（主项） | 内存 |
| :--- | :--- | :--- |
| QKV 投影 | `3·2·B·T·d²` | 权重 `3d²` |
| `QKᵀ` | `2·B·h·T²·d_h` = `2·B·T²·d` | 分数矩阵 `B·h·T²` |
| MLP | `2·2·B·T·d·4d` | 权重 `8d²` |

`T²` 项就是长上下文先撞内存、后撞算力的原因，也正是 FlashAttention 分块要绕开的东西。`W_q`、`W_k`、`W_v` 通常融合成一个 `[d, 3d]` 矩阵乘，少一次 kernel 启动。按 `[B,T,h,d_h]` 布局时，拆头是 **reshape 而不是 copy**。KV cache 存投影后的 `K,V`：每 token `2·layers·h·d_h`。

**速记**：按形状走一遍 QKV → softmax(QKᵀ/√d) → AV → 合并 → MLP；`T²` 先撞内存，1/√d_h 是为了 logits 保持单位方差。


---

## OpenAI

### 1. 解释自注意力（self-attention）。它的计算复杂度是多少？上下文变长时你有哪些选择？

<!-- EN: Explain self-attention. What is its computational complexity, and what are your options when contexts get long? -->

**Asked at**: OpenAI
**Source**: https://outcomeschool.com/blog/self-attention-in-transformers and https://outcomeschool.com/blog/how-does-sliding-window-attention-work

**参考答案**

自注意力让每个 token 用「所有其他 token 的加权平均」构建表示，权重来自 `softmax(QKᵀ/√d_k)`。

`n` 个 token、维度 `d` 时：

| 步骤 | 形状 | 代价 |
| :--- | :--- | :--- |
| `QKᵀ` | `n × n` | `O(n²d)` 时间，`O(n²)` 内存 |
| `softmax` | `n × n` | `O(n²)` |
| `AV` | `n × d` | `O(n²d)` |
| decode 时 KV cache | `2·L·n·d` | 每序列随 `n` 线性 |

所以时间是 `O(n²d)`，实践中的墙是**分数矩阵的 `n²`**，不是权重。

上下文变长时的选项：1）**FlashAttention**——在 SRAM 里分块，不物化 `n²`，FLOPs 不变但 HBM 流量大减；2）**滑窗/局部注意力**——每 token 只看 `w` 个邻居，`O(n·w)`，堆叠后感受野约 `layers × w`，但除非交错全局层，精确长程召回会丢；3）**稀疏/块注意力**——固定模式（Longformer、BigBird）或学习得到；4）**压缩**——摘要旧 KV（StreamingLLM 的 sink token）、量化、GQA/MLA 缩小 cache；5）**用检索替代上下文**——RAG 绕开二次项；6）**SSM / 线性注意力混合**——`O(n)`，精确复制能力弱。

复杂度要报**两个数**：`O(n²d)` 计算与 `O(n²)` 激活内存，后者先 OOM。滑窗只有在足够局部时才没有 `n²` 项；4k 窗口配 32 层理论覆盖约 128k，实际可靠性差得多。推理时 decode 因 KV cache 是每 token `O(n·d)`，长上下文伤的是带宽而非 FLOPs。要说清你会测什么：32k/128k 的 haystack 召回，而不只是困惑度。

**速记**：`O(n²d)` 计算 + `O(n²)` 中间矩阵内存，后者先 OOM；选项是 FlashAttention、滑窗/稀疏、KV 压缩、RAG、混合 SSM。

### 2. 交叉熵、KL 散度和困惑度（perplexity）之间是什么关系？为什么语言模型用交叉熵作为训练损失？

<!-- EN: What is the relationship between cross-entropy, KL divergence and perplexity, and why is cross-entropy the training loss for language models? -->

**Asked at**: OpenAI
**Source**: https://outcomeschool.com/blog/math-behind-cross-entropy-loss

**参考答案**

链条是精确的：

```text
H(P, Q) = H(P) + D_KL(P‖Q)
PPL     = exp(H(P, Q))          # 单位 nat
```

| 量 | 定义 | 角色 |
| :--- | :--- | :--- |
| 熵 `H(P)` | `−Σ p log p` | 数据不可约的不确定性；对模型是常数 |
| 交叉熵 `H(P,Q)` | `−Σ p log q` | 你真正最小化的量 |
| KL `D(P‖Q)` | `Σ p log(p/q)` | 使用 `Q` 的**超额**代价；实际被降低的量 |
| 困惑度 | `exp(H(P,Q))` | 「模型在多少個等概率 token 里选」 |

为什么语言模型用交叉熵：下一 token 预测是对词表的分类，目标是 **one-hot**，所以 `H(P)=0`，于是 `H(P,Q) = D_KL(P‖Q)` 严格成立。最小化 CE = 最小化 KL = 极大似然。梯度也很干净：`∂L/∂logits = p̂ − y`，条件数好且便宜。

因为 `H(P)` 是常数，「最小化 CE」与「最小化 KL」是同一个优化——只有比较不同数据分布上的模型时，这个区分才有意义。困惑度是**每 token 平均** CE 的指数；PPL=10 意味着模型的不确定性相当于在 10 个 token 里均匀选。它在不同 tokenizer 之间不可比。标签平滑会引入一个小的 `H(P)` 项，使 CE 与 KL 不再相等——这也正是它起正则作用的原因。比较不同词表的模型要报 bits-per-byte。

**速记**：`CE = H(P) + KL`，one-hot 目标下 `H(P)=0` 所以 CE 即 KL（等价极大似然）；困惑度是 CE 的指数，跨 tokenizer 不可比。

### 3. 调整 LLM 的上下文窗口大小会带来什么影响？

<!-- EN: What is the effect of adjusting an LLM's context window size? -->

**Asked at**: OpenAI
**Source**: https://www.linkedin.com/posts/amit-shekhar-iitbhu_the-context-window-is-the-llms-working-memory-activity-7437754426175672320-MH9c and https://www.youtube.com/watch?v=CGIhxIaOg3M&lc

**参考答案**

上下文窗口就是**工作记忆**，每个方向都有代价。

| 方向 | 收益 | 代价 |
| :--- | :--- | :--- |
| 变大（32k → 200k+） | 少一次检索往返、能整体推理整库/整文档、少「忘了前面」的 bug | `O(n²)` prefill、KV cache 随 `n` 线性、TTFT 上升、注意力稀释 |
| 变小 | 更便宜更快、省内存、少干扰 | 截断、needle 丢失、更多工程（RAG、压缩） |

不显然的效应：**中段丢失（lost in the middle）**——即使「能装下」，位于长上下文中部的事实召回明显弱于两端；**上下文腐化**——无关 token 累积后指令跟随变差，更多上下文不是单调更好；**KV cache 才是真预算**——每序列 `2 · layers · heads · d_head · n · bytes`，128k 时不做分页/量化/offload 就是每请求几十 GB；**成本按发送的 token 计**——每轮 10 万 token 的 prompt 不管有没有用都收一样的钱。

超出训练长度（RoPE 缩放 / YaRN）可行，但它先降的是**精度**而不是流畅度，上线前必须用 needle 测试评估。实践规则：放**最相关的 10–20%**，并保持 prompt 稳定以命中前缀缓存；把所有东西塞进去通常既慢又差。长上下文与 RAG 是互补的：长上下文用于「在这份文档里推理」，检索用于「在千万文档里找到」。评测要用必须依赖上下文中段的任务，而不只是摘要。

**速记**：上下文变大赢在少一次检索、能整体推理，输在 `O(n²)` prefill、KV 显存、TTFT 与中段信息丢失。


---

## Meta

### 1. 解释 Llama 类模型中的架构选择：为什么用 grouped-query attention、RoPE 和 SwiGLU，而不是 2017 年的原始 Transformer？

<!-- EN: Explain the architectural choices in a Llama-class model: why grouped-query attention, RoPE and SwiGLU instead of the vanilla 2017 Transformer? -->

**Source**: https://outcomeschool.com/blog/grouped-query-attention · https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding · https://outcomeschool.com/blog/feed-forward-networks-in-llms

**参考答案**

一句话：**GQA 买的是服务成本，RoPE 买的是位置泛化，SwiGLU 买的是每 FLOP 的质量**。每项改动针对的都是原版 Transformer 的**推理**代价，不是训练困惑度。

| 改动 | 修的病 | 机制 | 代价/收益 |
| :--- | :--- | :--- | :--- |
| **GQA** | decode 阶段 KV cache 吃掉显存带宽 | $G$ 个 Q 头共享 1 个 KV 头，介于 MHA($G{=}H$) 与 MQA($G{=}1$) | KV 缩小 $H/G$ 倍（70B、8 个 KV 头约 8×），质量略降 |
| **RoPE** | 学习式绝对位置既不能外推，又焊死在权重里 | 按位置角旋转 Q/K，点积只依赖**相对**偏移 | 无位置表、天然相对、可用频率缩放/YaRN 拉长上下文 |
| **SwiGLU** | ReLU FFN 参数利用率低 | $\text{SiLU}(xW_1)\odot xW_3$ 再乘 $W_2$，门控乘性交互 | 每参数质量更好；3 个矩阵故隐层缩到约 ⅔ 保持总参数不变 |
| **Pre-RMSNorm** | post-norm 需要 warmup，深层易发散 | 各子层前归一化，不减均值 | 便宜、稳，深栈标配 |
| **去 bias** | 收益极小 | 去掉全部 bias 项 | 影响可忽略，kernel 更简单 |

必须点出的失败模式：**朴素 RoPE 外推在超过训练长度后明显退化**，需要 base 频率缩放或位置插值，光调大 context window 没用。

**速记**：GQA 省 KV（70B/8 KV 头约 8×）、RoPE 给相对位置且可外推（需频率缩放）、SwiGLU 门控提每参数质量、Pre-RMSNorm 稳；核心是推理便宜。

### 2. 把 LLM 训练从 8 张 GPU 扩展到数千张 GPU 时，哪些东西会崩？现代技术栈如何应对？

<!-- EN: What breaks when you scale LLM training from 8 GPUs to thousands, and how do modern stacks deal with it? -->

**参考答案**

8 卡是单机、无故障；上千卡是**四件事同时坏**，各有标准对策。

| 坏在哪 | 原因 | 对策 |
| :--- | :--- | :--- |
| **通信** | all-reduce/all-gather 随规模增长；跨节点带宽比 NVLink/ICI 低 10–50 倍 | 3D 并行：TP 在节点内、PP/FSDP 跨节点，层级集合通信，通信与计算重叠 |
| **显存** | 激活随 batch×seq 涨；优化器状态 12 B/参数 | ZeRO-3/FSDP 分片、激活重算、序列并行 |
| **数值** | 长时间训练尖峰更多；bf16 累加漂移 | bf16 + fp32 master 权重、梯度裁剪、z-loss、QK-norm |
| **可靠性** | 单卡 MTBF 数小时到数天；4k 卡几小时一次故障 | **异步分片 checkpoint**（约 30 分钟一次）、弹性重启、健康检查、确定性重分片 |
| **数据管线** | 一个输入瓶颈就能饿死上千张卡 | 预分词分片、确定性 shuffle、足够 loader worker 与预取深度 |
| **调试** | 坏 step 是百万分之一 | 逐步指标、loss 尖峰检测、能 dump 出问题 batch |

结构性认知：**千卡下训练作业就是一个分布式系统，ML 只是它的客户端**。MFU 由通信、数据、故障三者中最慢的一环决定，所以成熟团队在 ckpt/重启与通信重叠上的投入远多于模型代码。

另外两条容易被忽略：全局 batch 变了等于换了优化地形（LR 要缩放、warmup 要拉长），8 卡调好的超参不迁移，要预留重调期并盯 LR/梯度范数曲线。

**速记**：通信、显存、数值、可靠性同时坏：TP 进节点 + PP/FSDP 跨节点并重叠通信、重算+分片、bf16+fp32 master 防尖峰、**异步 ckpt + 弹性重启**。


---

## Mistral AI

### 1. Mistral 7B 采用了分组查询注意力（GQA）和滑动窗口注意力（SWA）。这两种机制各自带来什么收益，又各自付出什么代价？

<!-- EN: Mistral 7B shipped with grouped-query attention and sliding-window attention. What does each buy you, and what does each cost? -->

**Source**: https://outcomeschool.com/blog/grouped-query-attention · https://outcomeschool.com/blog/how-does-sliding-window-attention-work

**参考答案**

它们撞的是两堵不同的墙：GQA 打 **decode 期的显存带宽**，SWA 打 **cache 的显存占用量**。两者都不让模型更聪明。

| | GQA | SWA |
| :--- | :--- | :--- |
| 机制 | Q 头分组，每组共享一个 K/V 头（Mistral 7B：32 Q / 8 KV） | 每层每个 token 只看前 *W* 个 token（W=4096） |
| 买到 | KV cache 小 4 倍；每步 decode 读取的显存更少 | cache 由 W 封顶而非序列长度，单 token 成本恒定 |
| 代价 | 相对完整 MHA 略掉质量；对 prefill 影响很小 | 每层信息最多回传 W，32 层只能**间接**触达约 128k |
| 训练 | 可从 MHA uptrain | 必须重训，且与精确长程召回冲突 |

微妙之处：SWA 不等于"看不见第 10 万号 token"。感受野会叠加，第 *L* 层理论上可影响 `W × L` 之前的内容；它真正丢的是**远距离单跳精确检索**。这就是为什么滑窗模型做摘要很自然，做"找出第 3 页的发票号"很弱。

运营上最常被漏掉的代价：GQA 让你只能按 **KV 头数**切 TP。8 个 KV 头意味着没法 16 路张量并行而不复制或填充 KV——选服务拓扑时这是硬约束。

**速记**：GQA 省 decode 带宽（32Q/8KV → cache 小 4 倍），SWA 让 cache 与序列长度解耦；代价是质量微降、长程精确检索变弱，且 TP 只能按 KV 头数切。

### 2. 解释 Mixtral 风格的稀疏专家混合（MoE）模型是如何工作的。为什么一个约 47B 参数的模型能以接近 13B 模型的成本运行？

<!-- EN: Explain how a Mixtral-style sparse mixture-of-experts model works. Why does a ~47B-parameter model run at roughly the cost of a ~13B one? -->

**Source**: https://outcomeschool.com/blog/mixture-of-experts

**参考答案**

每个 transformer block 的 FFN 换成 *E* 个专家（Mixtral 是 8 个）加一个小 router。每个 token 由 router 打分，取 **top-2**，输出是两个专家输出的加权和。注意力、embedding、norm 都是稠密且共享的。

于是：`47B 总量 ≈ 每 token 激活 13B`。成本跟着**激活**参数走，因为 FLOPs 和每 token 的权重**读取量**都取决于你真正乘了哪些矩阵。

| 量 | 随什么增长 | 后果 |
| :--- | :--- | :--- |
| 每 token FLOPs / 延迟 | 激活（~13B） | 比稠密 47B 快约 3.5 倍 |
| 权重显存/HBM | 总量（~47B） | bf16 约 94 GB，要 2×80 GB 或单卡 int4 |
| 质量 | 总量 | 更接近稠密 47B 而非 13B |

**痛点**：(1) **约束是显存不是算力**——你为 47B 权重付钱去买 13B 的速度，延迟敏感的单用户部署是笔坏交易。(2) **路由负载不均**：少数专家变热，尾延迟就是热专家的排队时间。(3) 专家并行是跨节点 **all-to-all**，NVLink 上还行，以太网上很惨。(4) 微调更不稳：router 塌缩与专家欠训练，aux balance loss 不是可选项。

什么时候别用：单卡低 batch 单流服务，或任何互联是瓶颈的环境。

**速记**：只激活 top-2 专家，算力跟 active ~13B、显存跟 total ~47B；真痛点是显存、专家负载不均与跨节点 all-to-all。


---

## Cohere

### 1. 我们的旗舰模型是稀疏 MoE，总参数量约为激活参数量的 10 倍。为什么这种架构很适合私有化企业部署，它的代价又在哪里？

<!-- EN: Our flagship is a sparse MoE with ~10x more total than active parameters. Why is that architecture a good fit for private enterprise deployment, and where does it hurt? -->

**Source**: https://outcomeschool.com/blog/mixture-of-experts

**参考答案**

因为私有化本质是**固定显存预算下的"每 GPU 车队吞吐"**问题，而 MoE 恰好解耦了两条平时互相打架的曲线：质量跟*总*参数走，服务成本跟*激活*参数走。10 倍比例意味着你买到的是大模型的质量、付的是中模型的 FLOPs；而在私有化里 GPU 早已买断，多数时间还闲着。

| 契合点 | 为什么在本地说得通 |
| :--- | :--- |
| 高并发、偏 batch | 单 token 算力小，同样车队能服务多得多 token |
| 硬件预算固定、质量敏感 | 你花的是**已经付过钱**的显存，而不是延迟 |
| 多租户、领域各异 | 专家会特化，法务租户与客服租户压的是不同专家 |
| 本地经济账 | 没有按 token 的云账单，省下的 FLOPs 就是能内部售卖的容量 |

**哪里会痛：**

1. **权重必须全部驻留**。总参数决定显存；10 倍比例可能根本装不下客户的节点，量化是唯一杠杆，而专家权重在没有逐专家校准的情况下比稠密层更难量化。
2. **对互联敏感**。专家并行是 all-to-all；在银行没有 NVLink 域的以太网上，dispatch 延迟直接吃掉 MoE 的优势。
3. **低 batch 的尾延迟**。batch=1 时没有专家复用，热专家就是队列——而企业试点永远是 batch=1。
4. **气隙运维**。专家放置、再平衡、profiling 工具在没有厂商支持时更难跑。

所以要把取舍说明白：两张卡、5 个用户的试点，答案应该是稠密模型。

**速记**：MoE 解耦"质量看总参数、成本看激活参数"，适合已买断显存 + 高并发的私有化；代价是全量权重驻留、专家并行怕以太网、低 batch 被热专家拖尾。


---

## DeepSeek

### 1. 请讲解 DeepSeekMoE。它和 Mixtral 这类标准 top-2 MoE 有什么不同？

<!-- EN: Walk me through DeepSeekMoE. How is it different from a standard top-2 MoE like Mixtral? -->

**Source**: https://outcomeschool.com/blog/mixture-of-experts

**参考答案**

一句话：DeepSeekMoE 在经典 top-2 MoE 上做了两处结构性改动——**专家细粒度化**与**共享专家**。

| 维度 | Mixtral 式 | DeepSeekMoE |
| :--- | :--- | :--- |
| 专家粒度 | 8 个粗粒度 FFN，top-2 | 大量细粒度专家（如 64–256），top-k=6–8 |
| 共享专家 | 无 | 1–2 个**始终**激活 |
| 路由信号 | top-2 gate softmax | top-k gate + 每专家 bias（V3 中无辅助损失） |

**为什么要细粒度**：粗粒度专家必须覆盖一大片知识混合，参数只能保持"通用"。把 FFN 拆成许多小专家，每个可以专精更窄的模式，而 top-k 组合若干小专家能以更少 FLOP 更精确地逼近目标函数。

**为什么要共享专家**：通用知识（句法、高频虚词、一般推理）否则会在每个被路由专家里被重复学习。把它隔离在一个常开专家里，被路由专家才能真正特化——论文消融显示，共享专家吸收通用部分后，路由专家明显更偏领域。

**代价**：路由逻辑更复杂、要处理容量因子（capacity factor）与丢 token 策略，以及 all-to-all 分发——小 batch 时这部分延迟可能超过专家 GEMM 本身。单卡上它严格劣于稠密模型；只有在能把专家摊到多设备时收益才出现。

**速记**：DeepSeekMoE = 细粒度专家（多而小）+ 共享专家（常开吸收通用知识）；换来更强特化，代价是路由开销与 all-to-all 通信，单卡反而不如稠密。

### 2. DeepSeek-V3 使用了无辅助损失的负载均衡。辅助损失原本有什么问题？偏置技巧又是如何工作的？

<!-- EN: DeepSeek-V3 uses auxiliary-loss-free load balancing. What was wrong with the auxiliary loss, and how does the bias trick work? -->

**参考答案**

一句话：辅助损失是第二个目标，天然与语言建模梯度打架；bias 把负载均衡移出梯度路径。

**辅助损失的问题**：系数调高，路由均衡但模型明显变差（损失会把 token 推向未被充分使用的专家，不管该专家是否合适）；系数调低，路由坍塌——少数专家赢下每个 batch，拿到更多梯度、变得更强、赢得更多，形成富者愈富，最后大部分参数是死的。

**bias 技巧**：为每个专家保留一个标量 $b_i$，它**只在选 top-k 时**加到亲和度分数上，不参与 gate 权重计算：

$$s_{i,t} = \text{softmax}(u_t^\top e_i),\quad \text{topk over } (s_{i,t} + b_i),\quad g_{i,t}=s_{i,t}\ \text{(unbiased)}$$

每步之后按负载误差的**符号**更新：过载专家 $b_i \mathrel{-}= \gamma$，欠载专家 $b_i \mathrel{+}= \gamma$。因为更新是符号式且完全在梯度路径之外，它相当于作用在 argmax 上的慢速积分控制器，不可能扭曲语言建模目标。

**坑**：bias 只解决**全局**负载，不解决**单序列 / 单设备**负载。V3 仍保留一个极小的 sequence-wise 辅助损失，专门压制 batch 内不均衡——那才是专家并行里真正产生慢 rank 的原因。

**速记**：辅助损失与语言建模梯度冲突——权重大掉点、权重小坍塌。bias 只在选 top-k 时加、符号式慢速积分控制；但它只管全局负载，仍需极小 sequence-wise 损失防设备间不均。

### 3. 什么是多 token 预测（MTP），为什么要用它来训练？

<!-- EN: What is multi-token prediction (MTP) and why train with it? -->

**Source**: https://outcomeschool.com/blog/decoding-deepseek-v4

**参考答案**

一句话：MTP 挂 $k$ 个轻量头，每个多预测更远一个 token，既加监督密度又自带 draft 模型。

在第 $k$ 层深度，头不只看到 $h_t$，而是把上一深度的隐状态与**当时真正被预测出的 token 的 embedding** 拼接，经投影和一个 transformer block：

$$h_t^{(k)} = \text{Block}_k\big(\text{proj}([\,h_t^{(k-1)};\ \text{Emb}(x_{t+k})\,])\big),\qquad P(x_{t+k+1}\mid \cdot)$$

训练损失是各深度交叉熵之和，每项权重 $\lambda/k$。

**带来什么**：

1. **每 token 信号更密**——一次前向产生 $k{+}1$ 个监督目标，数据效率上升而不用更多 token。
2. **前瞻规划**——要预测后两个 token，表征必须编码句子**意图**而非下一个表面形式；这就是大家含糊称作"模型在做规划"的机制。
3. **免费投机解码**——额外头就是内置 draft。V3 报告第二 token 接受率 85–90%、解码 TPS 约 1.8×。

**代价**：额外参数与激活（V3 正是因此采用串行而非并行 MTP）；推理时若不跑自投机，这些头是纯负担，反之则必须常驻显存。

**速记**：MTP 用"上一深度隐状态 + 真实下一 token 的 embedding"串行预测更远 token，得到更密监督与前瞻表征，还能当自投机 draft；代价是额外参数显存，不投机就是负担。

### 4. 用 PyTorch 实现带共享专家的 top-k MoE 路由，并指出其中的效率与正确性陷阱。

<!-- EN: Implement top-k MoE routing with a shared expert in PyTorch, and point out the efficiency and correctness traps. -->

**Source**: https://outcomeschool.com/blog/mixture-of-experts

**参考答案**

一句话：先 top-k 再归一化、按专家分桶做 grouped GEMM、共享专家完全绕开路由。

**参考实现**

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w1, self.w2, self.w3 = nn.Linear(d, h, False), nn.Linear(h, d, False), nn.Linear(d, h, False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoELayer(nn.Module):
    def __init__(self, d, h, n_routed=64, top_k=6, n_shared=2):
        super().__init__()
        self.top_k, self.n = top_k, n_routed
        self.gate = nn.Linear(d, n_routed, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_routed), requires_grad=False)  # load-balance only
        self.routed = nn.ModuleList([Expert(d, h) for _ in range(n_routed)])
        self.shared = nn.ModuleList([Expert(d, h) for _ in range(n_shared)])

    def forward(self, x):                       # [T, d]
        T, d = x.shape
        logits = self.gate(x)                   # [T, n]
        scores = logits.sigmoid()
        idx = (scores + self.bias).topk(self.top_k, dim=-1).indices
        w = scores.gather(1, idx)
        w = w / w.sum(-1, keepdim=True)         # normalise AFTER top-k

        out = torch.zeros_like(x)
        flat = idx.reshape(-1)
        for e in range(self.n):                 # group by expert: one GEMM per expert
            tok, slot = (flat == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            src = tok // self.top_k
            y = self.routed[e](x[src]) * w[src, slot].unsqueeze(-1)
            out.index_add_(0, src, y)
        for e in self.shared:
            out = out + e(x)
        return out
```

**要点**

- gate 权重在 **top-k 之后**归一化，之前归一化会改变专家间的相对尺度。
- `index_add_` 累加顺序不确定，输出每次运行可能不同；用 `float32` 累加，或按专家做 segment-sum 换取逐位可复现。
- 生产环境别把各专家输出 `torch.stack` 成 `[T, k, d]`：它物化 $k$ 倍激活，还破坏 grouped GEMM 的批量。
- 最疼的坑是在 host 上按专家排序 token。把置换留在设备上，并把每个专家的桶 pad 到整数倍，否则 ragged GEMM 会浪费 20–40% 的 tile。
- 共享专家必须完全绕过路由——"偶尔路由一下"会悄悄把它本应消除的冗余又加回来。
- `topk` 前把 scores 转 `float32`：BF16 下截断处的并列是任意打破的，表现为训练不稳而不是报错。

**速记**：先 top-k 再归一化；`index_add_` 非确定性；别 stack 出 $[T,k,d]$，要按专家分桶 grouped GEMM 并 pad 对齐；共享专家必须完全绕过路由。


---

## Moonshot AI（Kimi）

### 1. Kimi 最大的卖点是超长上下文。当你把上下文从 8K 扩展到几十万 token 时，最先崩掉的是什么，为什么？

<!-- EN: Kimi's headline feature is very long context. When you push from 8K to hundreds of thousands of tokens, what actually breaks first, and why? -->

**Source**: https://outcomeschool.com/blog/lost-in-the-middle-problem-in-llms

**参考答案**

一句话：先崩的不是 $O(n^2)$，而是**可用性**，顺序大致如下。

256K 下靠 FlashAttention 和 chunked prefill，算力账单是撑得住的。真正先坏的是：

1. **位置分布外（OOD）**。训练在 8K，模型从没见过问题与证据相距 20 万 token。RoPE 相位落在训练区间之外，注意力分数变得任意。退化从这里开始，远早于任何 OOM。
2. **注意力稀释 / lost in the middle**。对 $n$ 个 key 做 softmax，每个 key 分到的质量更少。20 万窗口里位于 40% 深度的事实，明显差于 8K 窗口里同一个事实——U 形准确率曲线就是它的签名。
3. **解码带宽**。每 token 的 KV 恒定而序列变长，解码退化为纯 HBM 搬运。20 万 × batch 时 KV 远超权重，FLOP 没变但吞吐崩了。
4. **prefill 延迟**。TTFT 到秒级，用户在察觉准确率变化之前就先感知到了卡顿。
5. **任务数据稀缺**。真正需要整窗口的 20 万 token 文档加任务很少；needle-in-a-haystack 只训练了一种技能。

所以修复顺序是：位置扩展 → 长上下文继续预训练 → 显存/带宽工程 → 任务设计。只做对 (1) 而跳过 (2)，得到的是一个**接受** 256K、却无法在 256K 上推理的模型——这是长上下文发布最常见的翻车方式。

**速记**：先崩的不是 O(n²) 而是可用性——RoPE 位置 OOD → 注意力稀释 → decode 变纯带宽瓶颈 → TTFT 爆炸 → 长文任务数据稀缺。顺序是位置扩展、长文续训、带宽工程、任务设计。

### 2. Kimi K2 使用了多头潜在注意力（MLA）。解释它做了什么，以及与 GQA 在缩减 KV cache 方面相比如何。

<!-- EN: Kimi K2 uses Multi-head Latent Attention (MLA). Explain what it does and how it compares to GQA for KV-cache reduction. -->

**Source**: https://outcomeschool.com/blog/kv-cache-compression

**参考答案**

一句话：MLA 不共享头，而是把 KV 压成低秩 latent，推理时把上投影吸收进 $W^Q/W^O$。

每个 token 只存一个小 latent 加一个解耦的 RoPE key，各头的 $K,V$ 由上投影重建：

$$c_t \in \mathbb{R}^{512},\quad k^{R}_t \in \mathbb{R}^{64}\;\Rightarrow\; K_t = W^{UK}c_t,\; V_t = W^{UV}c_t$$

推理时上投影被**吸收进** $W^{Q}$ 与 $W^{O}$，完整 K/V 从不物化——缓存里只有 $c_t + k^R_t$，每 token 每层 576 个数，而不是 $2 n_h d_h$。

| | MHA（128 头） | GQA（8 组） | MLA |
| :--- | :--- | :--- | :--- |
| 每 token 每层 KV | 约 64 KB（BF16） | 约 4 KB | 约 1.1 KB |
| 同等缓存下质量 | 基线 | 小幅下降 | ≈ MHA（论文称持平或更好） |
| 计算 | 基线 | 基线 | 额外上投影，latent 空间矩阵乘 |
| 实现 | 简单 | 简单 | 权重吸收、两条 RoPE 路径，对朴素 kernel 融合不友好 |

**一句话取舍**：GQA 丢的是 KV 的**头**（你希望影响不大的质量牺牲）；MLA 丢的是 KV 的**秩**（你希望无损的压缩），代价付在 kernel 复杂度上——吸收后的权重让分片、chunked prefill 与前缀缓存记账都明显更难写。

**什么时候不划算**：小模型、短上下文、batch=1。4K 上下文时缓存从来不是瓶颈，MLA 的工程成本买不到任何东西。

**速记**：MLA 把 KV 压成低秩 latent（512）+ 解耦 RoPE key（64），上投影吸收进 $W^Q/W^O$，约 1.1KB/token/层，比 GQA-8 再小 3–4 倍且质量接近 MHA；代价是 kernel 复杂度，短上下文小模型纯亏。

### 3. Kimi K2 是一个 1T 参数的 MoE，每个 token 约 32B 激活参数，包含数百个专家。解释其路由机制以及训练它的系统成本。

<!-- EN: Kimi K2 is a 1T-parameter MoE with ~32B active per token and hundreds of experts. Explain the routing and the systems cost of training it. -->

**Source**: https://outcomeschool.com/blog/mixture-of-experts

**参考答案**

一句话：路由便宜（≈32B 稠密），真正贵的是每层两次 all-to-all 与最慢的那个专家。

**路由**：每专家 sigmoid 亲和度，top-k 选择（K2 风格：384 个路由专家中取 8 + 1 个共享专家），权重在选中集合内重新归一化；bias 只参与选择，从而在没有辅助损失污染语言建模目标的前提下维持均衡。因为 384 个专家分散在多个 rank 上，路由输出同时就是一份**通信调度表**：哪个 token 必须去哪里。

**系统成本，按"多久毁你一次周"排序**：

| 成本 | 量级 | 为什么疼 |
| :--- | :--- | :--- |
| all-to-all 分发 + 合并 | 每 MoE 层两次交换，约 60 层 | 带宽受限；小 batch 时主导整步 |
| 专家负载倾斜 | Zipf 分布；一个热专家就把整步串行化 | 慢 rank 拖住其他所有 rank |
| 优化器状态 | Adam 需 2 个动量；1T 参数即便 BF16 也是 TB 级 | 逼你上 ZeRO/offload 与巨大 checkpoint |
| 激活内存 | 32B 激活不大，但 1T 总量必须常驻 | 限制 PP/EP 度，从而限制每卡 batch |
| 路由不稳定 | logits 漂移 → 训练中专家坍塌 | 需要 QK-clip / bias 控制回路 |

**反直觉之处**：FLOP 很便宜（32B 激活 ≈ 每 token 一个 32B 稠密模型），朴素估算会说"1T 模型、32B 成本"。实际**单步时间**由 all-to-all 与最慢专家决定，不由 GEMM 决定。稀疏模型买来算力效率，又把省下的钱花在通信上。

**速记**：路由是 sigmoid 亲和 + top-8/384 + 1 个共享专家，bias 只参与选路；真正的成本不在 FLOP（≈32B 稠密）而在每层两次 all-to-all、负载倾斜的慢 rank、TB 级优化器状态与权重常驻。

### 4. 如何把一个在 8K–32K 上训练的模型扩展到 128K 甚至更长？

<!-- EN: How do you take a model trained at 8K–32K and make it work at 128K or more? -->

**Source**: https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding

**参考答案**

一句话：三层缺一不可——位置方案、真实长文续训、长文 SFT 与评测。

**1. 位置方案（不需训练，但单独不够）**

| 方法 | 作用 | 代价 |
| :--- | :--- | :--- |
| 位置插值（PI） | 把位置压进训练区间 | 模糊高频维，损害局部顺序 |
| NTK / YaRN（分段） | 低频维插值、高频维不动；注意力温度按 $\sqrt{1/t}$ 缩放 | 需要 `rope_scaling` 配置与逐模型调因子 |
| 抬升基频（$10^4 \to 10^6$） | 旋转更慢 ⇒ 波长更长 | 便宜，但不混合会伤短上下文 |

**2. 继续预训练**。几十亿 token 的真实长文档（书、仓库、长对话）在目标长度上训练。模型在这里学到第 15 万个 token 是**相关的**，而不只是可寻址的。数据必须按长度分层——把互不相关的 2K 页面拼成 128K 的语料，教不会任何长程依赖。

**3. 长上下文 SFT + 评测**。需要跨窗口的任务（多文档问答、仓库级代码、长文摘要）。用 RULER 式多跳、变深度检索来评，而不是单针——单针测试远早于真实推理能力饱和。

**坑**：扩展时务必混入 10–30% 短上下文数据，否则你拿 128K 能力换来了 2–8K 上的明显退步，而用户撞上后者频繁得多。

**速记**：三层缺一不可——位置方案（PI/YaRN/提 base + 温度校正）、真实长文续训（长度分层、别拼接无关联短文）、长文 SFT 与 RULER 式多跳评测；必须混 10–30% 短数据否则短上下文掉点。


---

## 智谱 AI（GLM）

### 1. GLM 最初的预训练目标是自回归空白填充。它和 BERT、GPT 有何不同？团队为何认为它统一了理解与生成？

<!-- EN: GLM's original pre-training objective is autoregressive blank infilling. How does it differ from BERT and GPT, and why did the team argue it unifies understanding and generation? -->

**Source**: https://outcomeschool.com/blog/encoder-vs-decoder-in-transformers

**参考答案**

一句话：自回归空白填充把"理解"与"生成"统一成同一目标在不同跨度长度下的特例。

从文档采样若干片段，各替换为 `[MASK]`，然后**按你选择的顺序逐 token 自回归地生成这些片段**。被破坏的文本双向可见；生成片段对自身因果、对被破坏上下文双向。

| | BERT | GPT | GLM（空白填充） |
| :--- | :--- | :--- | :--- |
| 上下文可见性 | 双向，固定 | 因果，固定 | 双向上下文 + 因果生成 |
| 掩码独立性 | **独立**预测各掩码（条件独立假设） | 不适用 | 片段**联合**预测——片段 2 看得到片段 1 |
| 生成能力 | 不能 | 有 | 有，同一套头、同一目标 |
| 长跨度结构 | 无片段概念 | 无片段概念 | 片段长度是建模选择 |

**为什么统一起了**：理解 = 从完整上下文恢复一个**短**空（本质是 cloze，即 BERT 做的事）；生成 = 恢复一个**长**空——极限情况下整篇文档就是一个被掩码的片段，空白填充恰好退化为标准从左到右的语言模型。于是 NLU 与 NLG 是同一目标的不同跨度长度，一个模型不需要任务专用头。

**常被忘掉的实践细节**：片段顺序是随机的（二维位置编码）。这教会模型以答案的任意子集为条件——也让 GLM 天然适合做 infilling，后来成为代码 fill-in-the-middle 模型的合理基座。

**速记**：自回归空白填充=掩码片段逐 token 自回归生成，被破坏文本双向可见、生成片段因果自洽；相比 BERT 消除片段间独立假设，相比 GPT 获得双向上下文；生成本质是"整篇一个大空"。

### 2. GLM-4.5 是一个总参数 355B、激活参数 32B 的 MoE。解释其经济性：这种拆分带来什么收益，又付出什么代价？

<!-- EN: GLM-4.5 is an MoE with 355B total but 32B active parameters. Explain the economics: what does that split buy you and what does it cost? -->

**Source**: https://outcomeschool.com/blog/mixture-of-experts

**参考答案**

一句话：355B 买容量、32B 买算力——但显存地板仍是 355B。

**买到什么**：每 token 算力约为 32B 稠密模型（$\approx 2 \times 32\text{B}$ FLOPs），因此每 token 的训练成本与服务成本都比 355B 稠密模型降约 10×，而存储知识的**容量**保持 355B。在算力受限的区间（大 batch、prefill、训练）这几乎是白送质量。

**代价**：

| 代价 | 细节 |
| :--- | :--- |
| 显存地板 | 即便 batch=1，355B 全部要常驻——FP8 约 355 GB，所以光放权重就要 ≥ 5 张 H100 |
| 低 batch 带宽 | decode 只读激活专家，但访问模式零散；既没有稠密模型的局部性，也不是整模型流式读取 |
| all-to-all 延迟 | 每层两次交换；batch=1 时它们主导整步 |
| 微调 | 全参微调要为**整个** 355B 存优化器状态，不是激活部分 |
| 路由脆弱性 | 负载不均就把某个专家变成关键路径 |
| 长尾知识质量 | 总容量大，但每个稀有事实只住在少数专家里，路由错就丢 |

**决策规则**：MoE 赢在吞吐受限且显存充裕（云端服务、训练）；稠密赢在显存受限、batch=1 延迟受限，或需要在少量 GPU 上便宜微调。"32B 激活"是**算力**声明，永远不是**显存**声明——面试官问这题，正是因为候选人常把两者混为一谈。

**速记**：355B 保容量、32B 激活降每 token 算力（训练服务各省约 10×）；代价是显存地板仍 355B（FP8 约 355GB）、低 batch 下带宽与 all-to-all 主导、全参微调要存全量优化器状态。"32B 激活"是算力不是显存。

### 3. 用 PyTorch 实现一个 top-k MoE 路由器，然后对比辅助损失负载均衡与无损失方法。

<!-- EN: Implement a top-k MoE router in PyTorch. Then contrast auxiliary-loss load balancing with a loss-free approach. -->

**Source**: https://outcomeschool.com/blog/mixture-of-experts

**参考答案**

一句话：辅助损失只有重要性项带梯度、负载项必须 detach；无损失方案用符号式 bias 更新做积分控制。

**参考实现**

```python
import torch, torch.nn as nn

class TopKRouter(nn.Module):
    def __init__(self, d, n_experts, top_k, mode="bias", gamma=1e-3):
        super().__init__()
        self.top_k, self.mode, self.gamma = top_k, mode, gamma
        self.gate = nn.Linear(d, n_experts, bias=False)
        self.register_buffer("bias", torch.zeros(n_experts))

    def forward(self, x):                      # [T, d]
        logits = self.gate(x)                  # [T, E]
        probs = logits.sigmoid()
        sel_scores = probs + self.bias if self.mode == "bias" else probs
        idx = sel_scores.topk(self.top_k, dim=-1).indices      # [T, k]
        w = probs.gather(1, idx)
        w = w / w.sum(-1, keepdim=True)

        # fractional load per expert, and the aux loss (only in "aux" mode)
        onehot = torch.zeros_like(probs).scatter_(1, idx, 1.0)
        load = onehot.sum(0) / (x.shape[0] * self.top_k)        # [E]
        imp = probs.mean(0)                                    # [E], differentiable
        aux = (load.detach() * imp).sum() * self.n_experts if self.mode == "aux" else x.new_zeros(())
        return idx, w, aux, load

    @torch.no_grad()
    def update_bias(self, load):
        if self.mode != "bias":
            return
        target = 1.0 / self.bias.numel()
        self.bias += self.gamma * torch.sign(target - load)    # sign-only integral control
```

**要点**

- 辅助损失里的 `load` 必须 **detach**——只有重要性项带梯度；否则你在对 argmax 求导，没有意义。
- `bias` 用 **sign** 更新，而不是原始误差：比例更新会振荡，路由器在专家之间来回抖动。
- `bias` 是 buffer：**不能**吃梯度，也不能被 weight decay 碰到。
- 无损失方案仍需一个**极小的** sequence-wise 辅助项来压制每卡不均衡（全局均衡 ≠ 每 rank 均衡）。
- 辅助系数是真超参，有真实的质量代价——这正是 bias 方案的动机。
- `bias` 模式下把辅助项返回为**零张量**而不是 `None`：`None` 混进 loss 求和是共享 trainer 里最常见的静默 bug。

**速记**：aux 里 load 必须 detach、只有重要性项带梯度；bias 用 sign 更新做积分控制（比例更新会震荡），且不能吃梯度/权重衰减；loss-free 仍要极小 sequence-wise 项管住每卡不均衡。

### 4. 什么是多 Token 预测（MTP）？为什么要增加一个 MTP 层，它在推理时有什么帮助？

<!-- EN: What is Multi-Token Prediction (MTP), why add an MTP layer, and how does it help at inference time? -->

**参考答案**

一句话：MTP 预测 $t{+}2, t{+}3, \dots$，额外的小 block 是建模"承诺了 $t{+}1$ 之后"的最小容量，推理时直接当 draft。

串行形式把**实际被选中的中间 token** 的 embedding 与上一深度隐状态一起喂入，经投影和一个 transformer block：

$$h_t^{(k)} = \text{Block}_k\big(\text{proj}([\,h_t^{(k-1)};\ \text{Emb}(x_{t+k})\,])\big)$$

**为什么加一层而不是一个裸线性头**：从 $h_t$ 直接线性探测无法表示"我承诺了 token $t{+}1$ 之后会怎样"——它看不到这个承诺。额外 block 是建模该条件分布的最小容量。串行（而非并行）MTP 还避免了训练/服务不一致，因此头可以复用为 draft。

**推理收益——内置投机解码**：

| | 标准解码 | 带 MTP 头 |
| :--- | :--- | :--- |
| 每次前向 token 数 | 1 | 1 个已验证输入 + 1–2 个草稿 |
| 额外模型 | 需单独训练并托管的 draft 模型 | 已在 checkpoint 里 |
| 典型接受率 | 好的 draft 70–90% | 第二 token 约 85–90%（论文数据） |

因为 draft 很便宜（一个 block，不是整个模型），净效果是解码吞吐约 1.5–2×，显存只多出几个头。

**注意**：收益受接受率约束，而高熵输出（创意写作、非常规代码）上接受率会塌。它是可预测文本的吞吐技巧；另外损失权重 $\lambda/k$ 要调，否则浅层任务会退化。

**速记**：MTP 加深度头预测 $t{+}2/t{+}3$，串行喂回真实中间 token 的 embedding 过一个小 block，否则无法建模"承诺了 $t{+}1$ 之后"；推理时这些头就是内置 draft，白赚 1.5–2× 吞吐，但高熵文本上会塌。

### 5. GLM 从 GLM-130B 起就是中英双语模型。当一个模型要同时服务好两种语言时，分词、数据和评测各会发生什么变化？

<!-- EN: GLM has been bilingual Chinese/English since GLM-130B. What changes in tokenization, data and evaluation when a model must serve both languages well? -->

**参考答案**

一句话：分词要按目标配比学、数据要原生而非翻译、评测必须分语言报。

**分词**。BPE 的 merge 按频率分配，英文为主的语料会让每个汉字占 2–3 个 token，而英文单词只占 1 个。修法：(a) 按**目标比例**而不是原始网页比例采样 merge 训练语料；(b) 强制 **byte fallback**，任何字符都可表示；(c) 训练后逐语言量 fertility——2.0 与 1.3 tokens/word 的差距，对较差语言就是 50% 的成本与上下文惩罚。

**数据**。只用翻译会造出翻译腔——过分平行、不自然干净的句子。需要原生中文网页文本，而且要**平衡**而不是最大化：90:10 时少数语言在句法与文化常识上会明显退化。代码与数学基本语言中立，可以免费加权。

**评测**。中文分与英文分求平均会掩盖一切。

| 陷阱 | 修法 |
| :--- | :--- |
| 基准由英文翻译而来 | 母语者原生出题；检查翻译痕迹 |
| 同一 prompt 两语言拒答行为不同 | 分语言跑安全/拒答测试 |
| 中文基准多为知识回忆 | 每语言都补推理与指令遵循 |
| 某语言每 token 慢 3 倍 | 报告每语言 tokens/word 与每千次请求成本 |

**最关键的一点**：真实世界里最常见的失败不是词汇，而是**语码混用**——双语用户会句内切换，而从没见过句内混用的模型，即便两种单语模式都很强，切换处仍会输出乱码。

**速记**：分词按目标配比训 merge 且保留 byte fallback，逐语言量 fertility；数据要原生文本而非翻译腔，比例失衡让少数语言掉点；评测必须分语言报（原生题目、分语言拒答、每语言 token/成本），最易翻车的是句内中英混说。


---

## 阿里巴巴（Qwen）

### 1. Qwen 使用字节级 BPE，词表约 151K，为多语言覆盖做了增强，并把数字拆成单个字符。为什么做这些选择，权衡是什么？

<!-- EN: Qwen uses byte-level BPE with a ~151K vocabulary, augmented for multilingual coverage and with digits split into single characters. Why those choices, and what are the trade-offs? -->

**Source**: https://outcomeschool.com/blog/bpe-in-llms

**参考答案**

一句话：byte-level 保证零 OOV，大词表压 fertility，数字拆单字符让算术与复制在 token 级可见——三者各有代价。

**byte-level**：没有不可表示的东西——任何 Unicode 字符串、emoji、乱码都退化为字节而不是 `<unk>`。当你服务 29 种以上语言和用户生成文本时，这一点很关键。

**约 151K 词表**：更多槽位意味着每个词需要的 merge 更少，所有语言的 fertility 都下降。32K 时中文约 1.5–2 token/字，印地语更糟；151K 时各文字系统都更接近每个常用词 1 个 token。

| 选择 | 收益 | 代价 |
| :--- | :--- | :--- |
| 大词表 | 序列更短 ⇒ 注意力更便宜、有效上下文更长、few-shot 更好 | embedding + LM head 涨到 $151\text{K} \times d$；softmax 计算；稀有 token 拿到的梯度更新少 |
| byte fallback | 对任意输入鲁棒 | 对稀有文字是最差的切分方式；质量悬崖是静默的 |
| **数字拆成单字符** | `1234` 不可能是不可拆的 token；算术、日期、ID、复制数字变成 token 级操作 | 数字比合并编码多耗 2–4× token |

**为什么拆数字**：若 `2024` 是一个 token，模型无法从 `2024` 泛化到 `2025`——它们是两个无关向量。用单字符时，位值结构可见，算术与复制可以通过位置机制完成。代价是数字密集文本（日志、表格、金融）的序列变长，以及数字级幻觉更容易"看起来合理"。

**很少有人提的取舍**：大词表让蒸馏与量化更难（在小模型上 embedding 矩阵占参数量的主导），而且它会把你锁死——换分词器会让下游所有微调失效。

**速记**：byte-level 保证不 OOV；151K 词表压低各语种 fertility、序列更短；代价是 embedding/LM head 变大、softmax 更贵、稀有 token 欠训练。数字拆单字符让进位/复制/算术在 token 级可见，代价是数字密集文本变长。

### 2. Qwen3 在同一个模型中统一了思考模式与非思考模式，并支持调用方设置思考预算。你会如何训练这样的模型，又如何部署服务？

<!-- EN: Qwen3 unifies a thinking mode and a non-thinking mode in one model with a caller-settable thinking budget. How would you train that, and how would you serve it? -->

**Source**: https://outcomeschool.com/blog/large-reasoning-models

**参考答案**

一句话：四阶段训练（冷启动 → 带长度惩罚的推理 RL → 双答案融合 → 通用 RL），推理侧用预算契约强制收尾。

**训练——四阶段，顺序不能乱**

1. **长 CoT 冷启动**。在过滤后的推理轨迹上做 SFT（只用验证器判定正确的答案），让思考模式有可读格式。便宜，且它决定后续 RL 的上限。
2. **推理 RL**。对可验证数学/代码跑 GRPO，奖励带**长度感知**：对错 + 超出目标预算的软惩罚。没有长度项，RL 会稳定地发现"想得更久"是免费的胜利，于是 3+3 也要 2 万 token。
3. **思考模式融合**。在**同一 prompt 同时出现长 CoT 与直接答案**的混合上做 SFT，并带显式模式 token。这一步才真正教会开关；跳过它，两种模式会互相渗漏。
4. **通用 RL**。对两种模式施加偏好/规则奖励，管有用性、格式与工具使用——让非思考模式不是"被截断的思考"。

**推理时的预算控制**是一份解码期契约：调用方传 `max_thinking_tokens`；服务端跟踪推理段长度，到顶就注入结束思考 token 并强制进入作答阶段。训练样本里必须包含"撞到上限"的例子，模型才知道如何突然收尾——没被训练过强制终止的模型，被切断时会产出不连贯的最终答案。

**服务影响**

| 关注点 | 处理 |
| :--- | :--- |
| 延迟双峰 | 分队列或按请求预算；容量按思考模式规划 |
| KV 压力 | 每并发用户一条 3 万 token 推理轨迹是真实 HBM；限制每卡并发 |
| 流式体验 | 推理流到可折叠面板，答案流到主面板 |
| 前缀缓存 | 共享系统提示词是唯一可靠的命中点；思考轨迹是每请求独有 |

**速记**：四阶段——长 CoT 冷启动、带长度惩罚的推理 RL（否则奖励退化成想得越久越好）、同 prompt 双答案融合 SFT（真正教会开关）、双模式通用 RL；推理侧 `max_thinking_tokens` 到点注入结束标记，训练时就要见过被截断的样本。

### 3. Qwen 同时提供稠密和 MoE 模型（30B 约 3B 激活；235B 约 22B 激活）。什么时候你会选 30B-A3B MoE 而不是 32B 稠密模型？

<!-- EN: Qwen ships both dense and MoE models (30B with ~3B active; 235B with ~22B active). When would you pick the 30B-A3B MoE over a 32B dense? -->

**Source**: https://outcomeschool.com/blog/mixture-of-experts

**参考答案**

一句话：参数**总量**相近，成本曲线完全不同——MoE 是批处理技术，不是低延迟技术。

| 维度 | 30B-A3B MoE | 32B 稠密 |
| :--- | :--- | :--- |
| 显存中的权重 | 约 30B（所有专家都要常驻） | 约 32B |
| 每 token FLOPs | 约 3B 等效 | 约 32B 等效 |
| 大 batch 吞吐 | **好得多**（每 token FLOP 少 10×） | FLOP 受限 |
| batch=1 解码延迟 | FLOP 上占优，但零散的专家权重读取与路由开销吃掉大半 | 可预测、稠密、对带宽友好 |
| 1–2 张卡上微调 | 难（专家状态、负载不均） | 直接 |
| 4-bit 量化 | 不均匀（专家重要性不同） | 均匀 |
| 同等训练 token 下质量 | 每 FLOP 效率通常略低，每 token 容量更高 | 缩放可预测 |

**选 MoE 的场景**：并发用户多、吞吐受限；每副本有 ≥ 2–4 张卡让路由开销摊薄；负载以 prefill 为主（RAG、长 prompt）；想要 30B 级质量但只付 3B 级算力。

**选稠密的场景**：单卡或端侧部署；需要为多租户做廉价 LoRA 微调；流量是 batch=1 的突发型、可预测延迟比每 token 成本更重要；或必须激进量化且无法容忍专家不均匀退化。

**实测中的反直觉**：单卡 batch=1 时 MoE 的优势基本蒸发——路由开销与非合并的专家权重读取抵消了 FLOP 节省。

**速记**：两者权重都是 30B 级，区别在每 token 算力——MoE 约 3B 稠密成本、吞吐高 10 倍，但 batch=1 时路由开销与零散专家读取把优势吃光；并发高、prefill 重、每副本 2–4 卡选 MoE，单卡/边端、多租户 LoRA、激进量化选稠密。

### 4. Qwen2.5 通过 YaRN 加双块注意力（Dual Chunk Attention）把上下文扩展到 128K（Turbo 版约 1M），基本无需额外训练。解释其原理，以及为什么事后扩展很有吸引力。

<!-- EN: Qwen2.5 extends context to 128K (and ~1M for Turbo) using YaRN plus Dual Chunk Attention, mostly training-free. Explain how, and why post-hoc extension is attractive. -->

**Source**: https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding

**参考答案**

一句话：YaRN 按波长分段处理 RoPE 维度并做温度校正，DCA 把长序列切块把 $O(n^2)$ 压到近线性。

**YaRN（NTK-by-parts）**——按波长区别对待 RoPE 维度：

| 维段 | 波长 vs 训练长度 | 操作 |
| :--- | :--- | :--- |
| 高频（短 λ） | 任何长度下都正常 | **不缩放**——缩放会模糊局部顺序 |
| 低频（长 λ） | 远超范围 | **插值**位置（位置 $t \to t/s$） |
| 中间 | 模糊地带 | 用因子 $\gamma$ 在两者间渐变 |

外加**注意力温度**校正 $\sqrt{1/t}$，补偿更长上下文带来的熵变化——不做这一步，长上下文注意力过于平坦，perplexity 看着正常而质量已下降。

**双块注意力（Dual Chunk Attention）**——服务侧技巧。把序列切成块（如 32K）。块内：普通因果注意力，用相对位置。跨块：每个前序块用 keys 的压缩表示。结果：注意力成本随块数近似线性增长而不是随 token 平方增长，且大多数配对的相对位置编码仍落在**训练过的**范围内。

**后验扩展为什么有吸引力**：你只需付几十亿 token 的继续预训练加一次配置改动，而不是重训一个预训练已花掉 10–15T token 的模型。你保住短上下文质量，可以从同一基座产出 128K 与 1M 两个变体，还能按客户调缩放因子。

**诚实的局限**："基本免训练" ≠ 免费。若连一小段长上下文继续预训练都不做，YaRN 扩展出来的模型在长输入**推理**上会退化，哪怕检索测试通过——它能找到一根针，但没法比较两根针。

**速记**：YaRN 按波长分段——高频维不动、低频维插值、中间渐变，并做 $\sqrt{1/t}$ 温度校正；DCA 切块，块内普通因果注意力、块间用压缩表示，把 $O(n^2)$ 压成近线性且让相对位置留在训练范围。后验扩展便宜，但纯免训练版常"能检索不会推理"。


---

## Sarvam AI

### 1. 为什么分词是印度语种 LLM 的首要瓶颈？低 fertility 的分词器如何改变经济性？

<!-- EN: Why is tokenization the first bottleneck for Indian-language LLMs, and how does a low-fertility tokenizer change the economics? -->

**Source**: https://outcomeschool.com/blog/bpe-in-llms

**参考答案**

一句话：英文语料训出的 BPE 把一个印地语词切成 4–8 个 token，这一个比例会放大成四重税。

天城文、泰米尔文、泰卢固文、孟加拉文、马拉雅拉姆文在 UTF-8 里每字符 3 字节，而在英文网页上学的词表把 32K–50K 次 merge 都花在英语语素上。结果：一个印地语词变成 4–8 个 token，英文约 1.3 个。

| 效应 | 后果 |
| :--- | :--- |
| 上下文档 | 128K 窗口只装得下约 2–3 万印地语词，英文能装约 9 万 |
| 成本档 | 同等单价下每请求多 3–6× token |
| 训练档 | 每文档 token 更多 ⇒ 算力更多、有效 epoch 更少 |
| 能力档 | 长程多跳推理退化，因为注意力预算花在拼写上而不是语义上 |
| 形态档 | 印地语/泰米尔语黏着且屈折；过度碎片化的分词器让词干与词尾对模型不可见，破坏"同词根不同格"的模式 |

**修它的经济学**：把一门语言从约 4.5 降到约 1.5 tokens/word，推理成本降约 3×、有效上下文翻三倍。对一个月 1000 万通的政府热线，这是可行与不可行的分界——它比大多数模型质量工作杠杆更大，而且便宜：重训分词器加几十亿 token 修复预训练，而不是重训基座。

**反直觉之处**：低 fertility 的分词器还改善**推理**，不只是成本。同样的内容用更少 token，模型有限的注意力预算能覆盖更多命题；形态也在 token 级可见（词尾是独立但一致的片段，而不是任意的字节碎片）。

**速记**：印度文字 UTF-8 每字符 3 字节，英文语料训出的 BPE 把一个词切成 4–8 token，放大成上下文、成本（3–6×）、训练算力与能力四重税；形态被切碎还伤"同词根不同格"。修法只需重训分词器 + 少量修复预训练。


---

## Amazon / AWS

### 1. 为什么 Transformer 取代了 RNN 用于语言建模？KV cache 在推理时究竟带来什么？

<!-- EN: Why did transformers displace RNNs for language modelling, and what exactly does the KV cache buy you at inference time? -->

**Source**: https://outcomeschool.com/blog/how-do-rnns-and-transformers-differ · https://outcomeschool.com/blog/kv-cache-in-llms

**参考答案**

三个原因，按重要性排序：

1. **路径长度**：RNN 连接 token `i` 与 `j` 要经过 `|i−j|` 个串行步；transformer 只需**一次**注意力跳。短梯度路径意味着长程依赖可学，不需要门控的杂技。
2. **并行性**：RNN 在时间上本质串行，序列维度无法用于硬件并行，训练吞吐被步延迟锁死。Transformer 在所有位置并行（teacher forcing + causal mask），这才让万亿 token 的训练在经济上可行。
3. **归纳偏置**：RNN 把一切压进一个固定向量的瓶颈；注意力让所有位置保持可寻址，取回某个早期 token 是查表而不是重建。

代价：注意力是 `O(n²)` 计算、`O(n)` 状态，而 RNN 是 `O(n)` 计算、`O(1)` 状态。这是诚实的取舍——超长序列上 transformer 每 token 更差，所以才有线性注意力/SSM 混合。

**KV cache**：因果注意力下，token `t` 的 key/value 一旦算出就不再变化。缓存所有历史位置的 `K,V`，把每 token 生成从 `O(n²)` 重算变成对缓存的 `O(n)` 注意力——一个解码步就剩几次 GEMM 加内存读。**代价**：显存 `= 2 · 层数 · kv_heads · head_dim · 序列长 · batch · 字节数`，这就是长上下文 × 大 batch 会显存受限、以及解码是带宽受限（算术强度 ≈ 1）而非算力受限的原因。抓手：GQA/MQA（更少 kv 头）、分页分配、量化缓存、驱逐/压缩、前缀共享。

**速记**：Transformer 赢在"任意两 token 距离为 1 + 训练可完全并行"，代价是 `O(n²)`；KV cache 把解码从重算全序列降为只读缓存，本质是把计算换成显存带宽。


---

## 字节跳动

### 1. 完整讲一遍自注意力，并解释为什么点积要除以 sqrt(d_k)。

<!-- EN: Walk through self-attention end to end, and explain why the dot product is divided by sqrt(d_k). -->

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/math-behind-attention-qkv

**参考答案**

一句话：自注意力是"投影—算相似度—按相似度加权求和"，除以 `√d_k` 是为了把点积的方差从随维度线性增长拉回 1，让 softmax 停在可训练的温度区间。

计算流程：每个 token 的 `x ∈ R^d` 分别投影成 `Q = xW_q`、`K = xW_k`、`V = xW_v`，维度都是 `R^{d_k}`。序列长度 `n` 时算 `S = QKᵀ/√d_k ∈ R^{n×n}`，按行做 softmax 得 `A = softmax(S)`，再 `O = AV`，最后经 `W_o` 投回 `d`。多头就是把 `d_k = d/h` 的切片并行做一遍再拼接。

为什么要除：

| 步骤 | 效果 |
| :--- | :--- |
| `q·k = Σ q_i k_i` 共 `d_k` 项 | 若各项 i.i.d. 均值 0 方差 1，方差随 `d_k` **线性**增长 |
| 分数标准差 | 变成 `√d_k` 而不是 1 |
| `d_k = 128` 时不做缩放 | logits 散布 ±10 以上 → 某一位置概率 ≈1 |
| 后果 | 过 softmax 的梯度消失，注意力饱和到单个 token |
| 缩放后 | 分数方差 ≈1，softmax 处于可用温度区间 |

最关键的直觉：去掉缩放**不会发散，而是停滞**——loss 卡在平台，因为每个头都塌到同一位置、不再回传梯度。这也是部分长上下文模型改用 QK-norm（直接归一化 q、k）而不只依赖 `1/√d_k` 的原因。

**速记**：除以 `√d_k` 是把点积方差从随维度线性增长拉回 1；不除不会爆炸而是**softmax 饱和、梯度消失**，训练停滞而非发散。

### 2. 梳理从 MHA 到 MQA、GQA 和 MLA 的演进。每一步实际节省了什么？

<!-- EN: Trace the evolution from MHA to MQA, GQA and MLA. What does each step actually save? -->

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/grouped-query-attention

**参考答案**

四者数学完全一致，唯一变化的是 **KV cache 的大小**，而 KV cache 正是 decode 并发度的实际上限。

| 变体 | K/V 头数 | 相对 MHA（`h` 头）的 KV 缓存 | 换来什么 / 付出什么 |
| :--- | :--- | :--- | :--- |
| MHA | `h` | 1×（基准） | 质量最好，显存最差 |
| MQA | 1 | `1/h` | 显存收益巨大，但质量下降、大规模训练不稳 |
| GQA | `g`（如 8） | `g/h` | 接近 MHA 的质量，缓存只剩零头；今天的标准做法 |
| MLA | 低秩隐向量 `c` | 压缩到约 `d_c/h·d_h` | 压缩率最高，但要做权重吸收 + 自研 kernel |

要记住的公式：每 token 缓存 = `2 · n_layers · n_kv_heads · d_head · bytes`。以 80 层、`d_head=128`、fp16 的 70B 级模型为例：MHA（64 个 KV 头）≈ 2.6 MB/token，GQA-8 ≈ 0.33 MB/token——同一张卡上就是 8 倍的并发差。

MLA 把 KV 压成共享隐向量，再按头即时重建；DeepSeek 额外留出一段解耦的 RoPE 维度，让位置信息不被压缩掉。代价是：现成注意力 kernel 用不了，缓存要么解压、要么吸收进 `W_q`/`W_o`。

什么时候不必折腾：如果负载是 prefill 主导（长输入、短输出），KV 缓存根本不是瓶颈，应该去优化注意力的算力开销。

**速记**：MHA→MQA→GQA→MLA 只在压缩 **KV cache**（decode 并发上限）；GQA-8 相对 MHA 是 8 倍显存差，MLA 压得更狠但要吸收权重和自研 kernel。

### 3. Qwen3 如何在思考模式与非思考模式之间切换？如何训练并部署一个同时具备两种模式的模型？

<!-- EN: How does Qwen3 switch between thinking and non-thinking mode? How do you train and serve one model that does both? -->

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/large-reasoning-models

**参考答案**

它是**一套权重、两种行为**，靠提示模板加预算切换，不是两个模型。

四件事让它成立：

1. **特殊 token。** 思考模板插入 think 块的起始符，非思考模板不给，模型就不会进入推理循环；再由停止 token 或模板本身强制进入答案段。
2. **预算控制。** `thinking_budget` 限制推理 token 数，触顶时运行时注入"思考结束"token，模型转去作答。"最多想 N 个 token"这种指令遵循必须专门训，不是免费的。
3. **分阶段训练。** 先用长 CoT 的 SFT/RL 练出推理能力，再插入一个短促的"思考模式融合"阶段，混合思考与非思考数据，让两个分布都保持锐利。跳过融合阶段，非思考模式会明显退化。
4. **服务侧。** KV cache 与权重共用，思考模式只是多花输出 token；路由器按任务难度选模式，token 预算才是真正的延迟杠杆。

必须点出的失败模式：预算在推导中途被截断，会产出"自信但错误"的短答案——所以模型要被训练成会**放弃**整条链去作答，并在无能为力时给出"预算不足"式的兜底表述。

**速记**：同一套权重靠**模板 + 思考预算**切换；关键是训练时做 thinking/non-thinking 融合，否则关掉思考能力会明显退化；预算截断会逼出错误短答案，需专门训练放弃与兜底。

### 4. 仅解码器（decoder-only）模型能在分类任务上打败仅编码器（encoder-only）模型吗？

<!-- EN: Can a decoder-only model beat an encoder-only one on a classification task? -->

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/encoder-vs-decoder-in-transformers

**参考答案**

能——通常在标签少、标签空间开放、或输入很长且任务需要推理而非表层模式匹配的时候。

| 场景 | 谁赢 | 为什么 |
| :--- | :--- | :--- |
| ≥10 万标注样本、固定 5 类意图 | 编码器（BERT 类） | 双向上下文、便宜、线性头、推理快 |
| 只有几百条标注，或要求零样本 | 解码器 | 预训练指令遵循能力，无需训头 |
| 标签集每周都变 | 解码器 | 不用重训，把标签写进 prompt |
| 长文档 + "为什么"式推理 | 解码器 | 生成理由比一个 CLS 向量强 |

机制上：编码器双向可见，每个 token 表示已经聚合了完整上下文，天生适合单向量决策；解码器的因果掩码让最后一个位置只看左文，实践中取末位或均值池化，单 token 更弱，但预训练带来的指令语义强得多。

工程上的诚实答案：**解码器每 1k 次请求的成本高 10~100 倍**，只有当精度收益大于成本差距、或因此完全省掉标注工程时，它才算"赢"。可靠的生产模式是分层：编码器吃掉分布主干的 90%，解码器接长尾和低置信度升级上来的难例。

**速记**：少样本、标签空间常变、长文档要推理时 decoder 更优；但单次推理成本高 10~100 倍，工程上常用"小模型兜大头 + 大模型升级尾部"。

### 5. 为什么 FFN 先扩张再收缩？SwiGLU 相比 ReLU 或 GELU 带来什么好处？

<!-- EN: Why does the FFN expand and then contract? What does SwiGLU buy you over ReLU or GELU? -->

**Asked at**: ByteDance
**Source**: https://outcomeschool.com/blog/feed-forward-networks-in-llms

**参考答案**

先扩后缩（`d → 4d → d`）是为了给模型一块**足够宽、逐 token 独立的非线性特征空间**，这是注意力给不了的。注意力跨位置混合信息，但本质是线性加权求和；FFN 才是每个 token 各自做非线性变换的地方。隐藏层越宽，可容纳的"键值记忆"越多——常见解读是第一个矩阵做模式检测，第二个矩阵做查表并把结果写回残差流。

SwiGLU 改的是激活函数和参数量：

| 变体 | 形式 | 隐藏维 | 备注 |
| :--- | :--- | :--- | :--- |
| ReLU/GELU MLP | `W₂·act(W₁x)` | `4d` | 2 个矩阵，ReLU 有死神经元风险 |
| SwiGLU | `W₂(swish(W₁x) ⊙ W₃x)` | `≈ 8d/3` | 3 个矩阵，乘性门控 |

两个必须说清的事实：`8/3` 是为了让 SwiGLU 的参数量与 `4d` 的 MLP 持平；门控提供了一条**依赖数据的乘性通路**，在同等参数下效果更好（PaLM 的结论）。代价是多一次矩阵乘，kernel 与张量并行切分都要重新考虑——`w1·w3` 融合也正是某些推理栈实测 FLOPs 与 roofline 模型对不上的原因。

**速记**：FFN 是逐 token 的非线性特征变换（注意力的补充），先扩后缩是为更宽的键值记忆；SwiGLU 用门控乘性交互在同等参数量（隐藏维取 8/3·d）下效果更好，代价是多一个矩阵。


---

## 腾讯

### 1. 为什么生成式 LLM 采用仅解码器架构，而不是编码器-解码器架构？

<!-- EN: Why do generative LLMs use decoder-only rather than encoder-decoder? -->

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/encoder-vs-decoder-in-transformers

**参考答案**

因为**生成任务只需要因果前缀**，而只有解码器的结构用同一个训练目标就能服务所有任务。

原因按权重排序：

| 原因 | 机制 |
| :--- | :--- |
| 一个目标覆盖所有任务 | 在原始文本上做下一 token 预测；翻译/问答/摘要都变成"续写 prompt"——不需要任务专用头，也不需要成对数据 |
| 没有注意力瓶颈 | 编码器-解码器要把整段源文本挤过一个固定宽度的交叉注意力接口；只有解码器的模型在每一层都能直接关注原始 token |
| 更容易扩展 | 单一栈，不用平衡编码器/解码器，没有交叉注意力；KV cache 管理统一（这正是分页与前缀缓存能做得简单的原因） |
| 上下文学习可用 | 示例天然可以塞进前缀；编码器-解码器因为源文本被单独编码，利用得没那么好 |
| 训练效率 | 每份文档的每个 token 都是训练信号；编码器-解码器要把容量花在编码器的重建任务上 |

要接受的代价：没有双向上下文（对分类/片段抽取很关键，那里 BERT 类模型仍然更好且便宜 10 倍）；同一前缀复用时 prefill 是重复劳动（这靠前缀缓存而非架构解决）。

诚实的一点：语音翻译和部分多模态模型仍以编码器-解码器为主，因为输入输出模态差异足够大，独立编码器是值得的。只有解码器是在**文本**上赢了，不是普适真理。

**速记**：生成任务只需要因果前缀，一个 next-token 目标就能覆盖所有任务、且避免固定宽度交叉注意力瓶颈、KV 管理统一；代价是无双向上下文（分类任务仍应选 BERT 类），语音/多模态仍常用编码器-解码器。

### 2. pre-norm 和 post-norm 有什么区别，它如何影响训练稳定性？

<!-- EN: What is the difference between pre-norm and post-norm, and how does it affect training stability? -->

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/encoder-vs-decoder-in-transformers

**参考答案**

差别在于 LayerNorm 相对残差相加的位置，而它改变的是**梯度通路**，不只是数值。

| | Post-norm（原始） | Pre-norm（现代默认） |
| :--- | :--- | :--- |
| 顺序 | `LN(x + Sublayer(x))` | `x + Sublayer(LN(x))` |
| 残差通路 | 每个 block 都要**穿过** LN | **恒等**——从输出到输入是一条干净的高速路 |
| 初始化时的梯度 | 被 `~1/√L` 衰减，需要 warmup | O(1)，无需 warmup |
| 深度扩展 | 超过约 12~24 层就退化，除非加技巧 | 可扩到 100+ 层 |
| 最后一层 | 不需要额外 LN | 必须在输出头前加**最终 LN** |
| 浅层时的峰值质量 | 略好 | 略差 |

机制：pre-norm 下残差流不做归一化，方差随深度增长——所以需要最终 LN，也是一些模型再加缩放残差（`x + α·Sublayer(...)`）控制增长的原因。post-norm 的 LN 在残差通路内部，梯度每层都要过 LN 的雅可比，这就是早期训练脆弱、原论文需要长 warmup 的根源。

常见的误解：用了 pre-norm 还"为了保险"保留 warmup——没坏处，但掩盖了 pre-norm 真正的收益是**取消 warmup 这个必需项**。另外，训练好的 post-norm 模型不能中途切成 pre-norm：权重不通用。

**速记**：post-norm 把 LN 放在残差相加**之后**，梯度每层都要穿过 LN 的雅可比，需长 warmup；pre-norm 残差是恒等通路，可无 warmup 训到上百层，但要加**最后一层 LN**，且残差方差随深度增长。

### 3. RoPE 如何用绝对位置编码实现相对位置感知？

<!-- EN: How does RoPE achieve relative position awareness using an absolute position encoding? -->

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding

**参考答案**

RoPE 在若干二维子空间里，把 query 和 key 按**与位置成正比的角度**旋转。因为旋转是正交变换，`R_m q` 与 `R_n k` 的内积只依赖 `m − n`。

机制，对 `d_k/2` 组配对：

1. 把 `q` 拆成 `(q_{2i}, q_{2i+1})` 各组。
2. 第 `i` 组在位置 `m` 处旋转 `θ_i = m · ω_i`，其中 `ω_i = 10000^{-2i/d}`。
3. `q̃ = R(θ) q`；`k` 同样按自己的位置旋转。
4. 于是 `⟨R_m q, R_n k⟩ = ⟨q, R_{m−n} k⟩`——只与相对偏移有关。

比推导更重要的工程后果：

| 性质 | 含义 |
| :--- | :--- |
| 无学习参数 | 没有位置表要扩展；这才是上下文扩展在原理上可行的原因 |
| 施加在 `W_q`/`W_k` **之后**、注意力之前 | 所以它必须在注意力 kernel 内部，也是融合 kernel 要接收 `cos/sin` 表的原因 |
| 只表达相对 | 没有绝对位置信号 → 模型无法仅凭内容判断"第一个 token"；起始身份要靠别的线索 |
| 基频 `θ` 控制长度泛化 | 提高基频（如 10k → 500k）压缩高频旋转，延长可用长度 |

各种扩展技巧都作用在频率轴上：位置插值（除以 `m`）、NTK-aware 缩放（非均匀缩放 `ω`）、YaRN（分带缩放 + 注意力温度）。要指出的失败：朴素外推到训练长度之外会急剧且**静默**退化——短 prompt 的困惑度看着正常，长程检索已经崩了。

**速记**：RoPE 按位置旋转 Q/K，正交性使内积只依赖 `m−n`，因此"绝对编码"产生相对感知；无参数可外推，但需调基频/插值扩展长度；朴素外推会在长文检索上**静默**崩坏而短 prompt 看不出。

### 4. 为什么大家都在探索 MoE？它相对稠密模型的核心优势是什么？

<!-- EN: Why is everyone exploring MoE, and what is the core advantage over dense? -->

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/mixture-of-experts

**参考答案**

MoE 把**总参数**与**每 token 算力**解耦：容量（知识）可以变大，而每次前向的成本不必跟着涨。

| | 稠密 | MoE（`n` 选 `k`） |
| :--- | :--- | :--- |
| 参数 | `P` | `~n·P_expert`（大得多） |
| 每 token FLOPs | `~2P` | `~2·k·P_expert` + 路由 |
| 显存 | 权重 = 参数 | **所有专家必须常驻**，除非做卸载 |
| 算力/显存平衡 | 耦合 | 解耦——算力便宜、显存昂贵 |
| 训练动力学 | 稳定 | 路由塌缩、负载不均、专家训练不足 |

核心收益是**每单位质量所需算力**：用约 1 倍稠密模型的 FLOPs 得到约 8 倍大的模型。DeepSeek 与 Mixtral 都报告在相同激活算力下质量更好。

代价构成了全部工程故事：

1. **显存没省。** 所有专家都在 HBM 里，省的是 FLOPs 不是字节。这让 roofline 反转——MoE 的 decode 每个有用 FLOP 更加带宽受限。
2. **负载均衡。** 没有辅助损失（或 DeepSeek 的偏置式均衡），路由会塌到少数专家，其余永远训不到。
3. **通信。** 专家并行意味着每层一次 all-to-all；token 分布差就会把算力收益变成网络瓶颈。
4. **batch 敏感。** 小 batch 下每专家分到的样本极少 → GEMM 效率差。MoE 是大 batch、高吞吐技术；单流延迟往往**更差**。

什么时候别用：延迟敏感的少量用户服务，或者专家常驻不起的小集群。

**速记**：MoE 把总参数与每 token 算力解耦——容量变大而 FLOPs 几乎不变；但它**不省显存**（专家全驻留）、需负载均衡与 all-to-all 通信，小 batch 下单流延迟反而更差。

### 5. 以 Qwen 和 DeepSeek 为例：介绍一个你熟悉的模型，并对两者做技术对比。

<!-- EN: Take Qwen and DeepSeek as examples: introduce a model you know well and compare the two technically. -->

**Asked at**: Tencent
**Source**: https://outcomeschool.com/blog/mixture-of-experts

**参考答案**

把回答组织成一张"按真正不同的轴"列的对比表，然后明确表态。

| 维度 | Qwen3（稠密 + MoE 多尺寸） | DeepSeek（V3/R1） |
| :--- | :--- | :--- |
| 注意力 | GQA + 标准 RoPE | **MLA** + 解耦 RoPE：KV 压成隐向量 |
| FFN | 各尺寸下稠密与 MoE 并存 | 细粒度 MoE，共享专家 + 路由专家 |
| 路由 | 标准 top-k + 辅助损失 | 无辅助损失的偏置均衡，另加序列级损失 |
| 精度 | bf16 训练，部分部署用 fp8 | 大规模 **fp8 混合精度训练** |
| 推理 | 单模型内混合思考/非思考模式 | 独立推理模型（R1）+ 长 CoT |
| MTP | — | **多 token 预测**头作为辅助目标 |

每个选择的代价：MLA 在同等宽度下把 KV 缓存压到比 MHA 少约一个数量级，这才让长上下文与高并发可负担——但需要权重吸收和自研 kernel。细粒度专家加共享专家提升了专业化，同时保留一条稳定的"常开"通路，代价是更复杂的 all-to-all。

选边并给出理由：产品若需要一个模型同时具备快速与推理两种模式，混合模式设计在运维上更简单；若追求长上下文吞吐，MLA 的缓存压缩是更大的杠杆。重点不是排名，而是展示你知道**每个旋钮转了哪一格、付了什么账**。

**速记**：对比要落在具体旋钮：MLA 把 KV 压到隐向量（省一个数量级缓存，代价是吸收与自研 kernel）、细粒度 MoE + 共享专家、fp8 训练、MTP；Qwen 的混合思考模式在运维上更简单。


---

## 阿里巴巴（淘天 / 阿里云）

### 1. MoE 如何工作，如何做负载均衡，专家激活是 batch 级、token 级还是其他粒度的决策？用 Python 实现一个 MoE。

<!-- EN: How does MoE work, how do you balance load, and is expert activation a batch-level, token-level or other decision? Implement a MoE in Python. -->

**Source**: https://outcomeschool.com/blog/mixture-of-experts

**参考答案**

MoE 层用一个路由（router）把单层稠密 FFN 换成 `E` 个专家 FFN：路由给每个 token 打分，选出 `top_k`（典型 2 个，候选 8–256 个），再按归一化后的门控权重合并输出。参数量随 `E` 增长，FLOPs 只随 `k` 增长。

**粒度是第一个设计决策：**

| 粒度 | 含义 | 现实 |
| :--- | :--- | :--- |
| token 级 | 每个 token 自己选 top-k | 所有严肃实现（GShard、Switch、Mixtral、DeepSeek）的默认 |
| batch 级 | 整个 batch 共用一组专家 | 只是吞吐技巧，丢掉了 MoE 值得做的自适应性 |
| 专家选择（expert-choice） | 专家挑自己的 top token | 天然完全均衡，但某个 token 可能被整体丢弃 |

**负载均衡**按"实际有效程度"排序：(1) 路由 logits 上的**自适应 bias**，按每步观测到的负载调整，不需要辅助损失，DeepSeek-V3 就是这招；(2) Switch 辅助损失 `α · E · Σ f_e · P_e`，`α≈0.01`；(3) capacity factor + 丢 token（丢弃溢出 token 的残差，而不是 padding）。三者都不做，**专家坍缩**会在几千步内发生：两个专家吃掉全部 token，其余梯度为零直接死掉。

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
        flat = x.reshape(-1, D)                 # <-- token 级路由
        probs = F.softmax(self.gate(flat), -1)  # [N, E]
        w, idx = probs.topk(self.top_k, -1)     # [N, k]
        w = w / w.sum(-1, keepdim=True)         # 在选中的专家上重新归一化

        out = torch.zeros_like(flat)
        for e in range(self.n_experts):
            mask = (idx == e).any(-1)           # 路由到专家 e 的 token
            if mask.any():
                gate_w = (w * (idx == e)).sum(-1)[mask]
                out[mask] += gate_w.unsqueeze(-1) * self.experts[e](flat[mask])

        # Switch 风格辅助损失：把使用率推向均匀
        frac = torch.zeros(self.n_experts, device=x.device).scatter_add_(
            0, idx.flatten(), torch.ones(idx.numel(), device=x.device, dtype=probs.dtype))
        frac = frac / (flat.size(0) * self.top_k)
        aux = self.n_experts * (frac * probs.mean(0)).sum()
        return out.view(B, T, D), aux

moe = MoE(512, 2048)
y, aux = moe(torch.randn(2, 16, 512))
```

三条工程要点：`top_k=2` + 重新归一化是标准配置，`k=1`（Switch）更省但更抖；辅助损失要以小系数加到总损失上，系数过大会让路由不再依赖输入；真实的瓶颈不是稠密计算，而是专家并行各 rank 之间的 **all-to-all 通信**——专家放置（按节点分组、每卡 8 个专家）比路由数学更影响吞吐。小 batch 会毁掉 MoE：`B·T < E` 时专家只见 1–2 个 token，路由梯度全是噪声。

**速记**：token 级路由是默认答案，batch 级只是吞吐技巧；负载均衡首选 router logit 上的自适应 bias，其次 aux loss + capacity factor；不做就是专家坍缩。

### 2. 千卡训练中 NCCL 超时很常见。你如何定位并修复？

<!-- EN: NCCL timeouts are common in thousand-GPU training. How do you locate and fix them? -->

**参考答案**

NCCL 超时几乎从不意味着 NCCL 坏了。它意味着**某个 rank 没走到集合操作**，其余 rank 在 kernel 里等到看门狗触发（PyTorch 默认 600s，部分作业 1800s）。

按"咬人频率"排序的根因：

| # | 原因 | 特征 |
| :-: | :--- | :--- |
| 1 | **控制流分叉** —— 某个 rank 因 NaN/溢出分支跳过了一次 `all_reduce`、提前 `break`，或用了只有 rank-0 才进的 `print` 保护集合操作 | 各 rank 的 step 计数对不上 |
| 2 | **掉队者** —— 慢节点、被限频的 GPU（ECC/过热）、数据长度不均、卡在 dataloader 里 | 某个 rank 的日志时间戳落后几分钟 |
| 3 | **网络** —— IB 链路 down、`NCCL_SOCKET_IFNAME` 配错、MTU 不一致、ECN 配置错 | `NET/IB : Got completion with error` |
| 4 | 进程死掉 —— OOM kill、段错误、容器被驱逐 | 进程数变少 |
| 5 | 集合操作尺寸不一致 —— 张量形状不同，例如各 rank 的序列数不同 | 紧接着 `Invalid usage` |

定位流程：打开 `NCCL_DEBUG=INFO`、`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`（或 FlightRecorder），复现后读每个 rank "最后完成的集合操作"——凶手是 traceback 里集合操作次数**更少**、或最后一条日志最老的那个 rank。对所有 rank 跑 `py-spy dump --pid`，能立刻看出是卡在 `ncclCommWait`（网络）还是卡在 Python（逻辑）。

修复：每个 rank 跑固定步数的 optimizer step（先 all-reduce NaN 标志，再统一跳过或不跳过）；数据按长度分桶；给监控加每个 rank 的心跳；优先弹性重启而不是调大超时——把 `NCCL_TIMEOUT` 调大只是把一个 10 分钟的故障变成一小时的故障。

**速记**：超时=有 rank 没到集合操作，不是 NCCL 坏了。按控制流分叉 > 掉队节点 > 网络 > 挂掉 > shape 不一致排序；用 NCCL_DEBUG + FlightRecorder 看谁少做了一次 allreduce，py-spy 看栈；别只会调大 timeout。

### 3. 如何做算子融合？为什么融合有时反而更慢？Hopper 上的 warp specialization 是什么？

<!-- EN: How do you do kernel fusion, and why can fusion make things slower? What is warp specialization on Hopper? -->

**Source**: https://outcomeschool.com/blog/decoding-flash-attention

**参考答案**

融合（fusion）就是把多个 kernel 合成一个，让中间结果不必往返 HBM。三个层次：(1) 编译器对 elementwise 链的融合（inductor/Triton）；(2) 手写生产者-消费者融合（LayerNorm+GEMM 的 epilogue、RoPE+attention）；(3) 算法级融合——**FlashAttention** 是原型：QKᵀ、softmax、PV 在 SRAM 内融合，`[N,N]` 的分数矩阵从不落在全局内存里。

**为什么融合会更慢**，失败模式是物理的，不是理论上的：

- **占用率塌陷**：融合后的 kernel 占用更多寄存器和共享内存。超过约 128 寄存器/线程或撞到 SMEM 上限，常驻 block 数下降、延迟隐藏能力丢失，损失超过省下的 HBM 流量。
- **并行度不足**：一个巨型 kernel 铺在 80 个 SM 上却只有 40 个 block，比四个小而饱满的 kernel 更差。
- **重算**：融合 attention 在反向要重算；序列短时重算开销大于省下的带宽。
- **尾效应 / 常驻 kernel** 在不规则形状上。
- **丢失向量化**：编译器再也看不出对齐。

**Hopper 上的 warp specialization**：在一个 CTA 内按角色切分 warp——生产者 warp 发 `cp.async.bulk.tensor`（TMA）把数据搬进共享内存，消费者 warp 跑 `wgmma` 张量核计算，两者通过 `mbarrier` 的 arrive/wait 同步。要点是 TMA 拷贝是异步的，所以计算 warp 不会等全局内存，拷贝 warp 也不占张量核。再叠加 thread-block cluster 与分布式共享内存，一个 CTA 可以直接读邻居的 SMEM，而不必绕 L2。

经验法则：kernel 是 matmul 受限且访存流水很长时（attention 前后向、GEMM+softmax）才值得特化；纯访存受限的 elementwise 不要特化，barrier 开销会吃掉一切。

**速记**：融合是省 HBM 往返，不是万能；变慢通常是寄存器/SMEM 压力把 occupancy 打掉、并行度不足、反向重算。Hopper warp specialization 用 TMA 生产者 warp + wgmma 消费者 warp 通过 mbarrier 同步，只在访存流水长、计算密集时才划算。

### 4. prefill 和 decode 阶段对 KV cache 的访问有何不同？

<!-- EN: How does KV cache access differ between prefill and decode? -->

**Source**: https://outcomeschool.com/blog/kv-cache-in-llms and https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization

**参考答案**

同一个数据结构，两种完全不同的访问模式：

| | Prefill | Decode |
| :--- | :--- | :--- |
| 每步 token 数 | 全部 prompt token `T` | 1（每条序列） |
| Kernel 形状 | 大 GEMM `[T,D]×[D,·]` | GEMV `[1,D]×[D,·]` |
| 瓶颈 | **算力**（张量核） | **显存带宽** |
| KV 访问 | **写入** `T` 条新记录，顺序 | 每步**读整个缓存** |
| 算术强度 | 高 | 每字节读约 `2·B` FLOP → 受 HBM 限制 |
| 扩展性 | attention 随 `T` 二次增长 | 每 token 随上下文长度线性增长 |

把它算实的那个数字：70B 级模型、8 个 KV 头、80 层、head_dim 128、fp16，一个 token 的缓存是 `2 × 80 × 8 × 128 × 2 B ≈ 327 KB`。4k 上下文下，**每生成一个 token** 就要重读约 1.3 GB。所以 decode 吞吐是按 tokens/s/GPU 报的，随 HBM 带宽提升而不是随 FLOPs 提升——也是为什么 GQA/MLA（把缓存变小）对 decode 的收益高于任何别的优化。

对服务栈的四个后果：

1. **chunked prefill** —— 长 prompt 切片，与 decode 步交错，避免一个 30k token 的 prefill 卡住其他请求的 token 间延迟。
2. **前缀缓存（prefix caching）** —— 共享 system prompt 的 prefill KV 只算一次并复用，是 chat 负载里最大的一笔收益。
3. **PagedAttention** —— KV 按固定块分配，不再为 max_length 预留而浪费 30–60% 的 HBM。
4. **prefill/decode 分离部署** —— 两者需要的硬件配比不同（prefill 吃算力，decode 吃带宽 + 大 batch）。

**速记**：prefill 是算力受限的大 GEMM，KV 只写一遍；decode 是带宽受限的 GEMV，每生成一个 token 都要把整个 KV 读一遍。所以 decode 优化方向是缩 KV（GQA/MLA）、分页、前缀缓存和 chunked prefill。


---

## 滴滴

### 1. Transformer 中的 FFN 层是做什么的？为什么它先扩张再收缩？

<!-- EN: What does the FFN layer in a Transformer do, and why does it expand before contracting? -->

**Source**: https://outcomeschool.com/blog/feed-forward-networks-in-llms

**参考答案**

attention 在**位置之间**混合信息；FFN 在**每个位置内部**变换信息，各位置独立且同构。它是一个逐 token 作用的两层 MLP：

```
h = W2 · act(W1 · x + b1) + b2      # d_model -> d_ff -> d_model
```

在现代 decoder block 的门控变体里是 `h = W2 ( SiLU(W1 x) ⊙ (W3 x) )`，其中 `d_ff ≈ 4·d_model`（SwiGLU 用 `8/3·d_model`，是为了去掉常数项后参数量持平）。

**它具体做什么**：绝大多数参数住在这里（稠密 transformer 约三分之二的权重），模型学到和取用的模式也在这里。值得引用的可解释性结论：`W1` 的行像**键**——在特定输入特征上触发的模式检测器；`W2` 的列像**值**向量——把一个方向写回残差流。所以 FFN 充当键值记忆，这也解释了为什么大部分事实知识被归因于它，以及为什么 MoE（恰好把这层稀疏化）能成立。

**为什么先升维再降维**：更高的中间维度给这一层留出空间，把在 `d_model` 里纠缠的特征变成**线性可分**的。如果保持同样宽度，`W2 W1` 会塌缩成一个秩为 `d` 的线性映射——非线性被夹在低秩瓶颈里，容量几乎全丢。秩的扩张**就是**容量本身。

实用推论：FFN 与 attention 交替不是随意的（attention 负责路由，FFN 负责存储）；为省显存而缩 `d_ff`，质量损失不成比例地大于减少头数；在 MoE 里每个专家的 `d_ff` 要小，使得激活 `k` 个专家的 FLOPs ≈ 一个稠密 FFN。

**速记**：attention 跨位置混合信息，FFN 逐 token 变换；W1 行像 key、W2 列像 value，所以 FFN 是键值记忆、也是参数和知识的主要载体。先升维是为了让特征线性可分——不升维就等于退化成秩为 d 的单个线性映射，容量全丢。

### 2. 对比 MHA、MQA、GQA 和 MLA：各自的优缺点。

<!-- EN: Compare MHA, MQA, GQA and MLA: pros and cons of each. -->

**Source**: https://outcomeschool.com/blog/grouped-query-attention and https://outcomeschool.com/blog/kv-cache-compression

**参考答案**

它们的区别在于**有多少个独立的 K/V 头**，而 KV 缓存大小正是按这个缩放的。

| 变体 | KV 头数 | 每 token 每层缓存 | 质量 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **MHA** | `h`（每个 query 头一个） | `2·h·d_head` | 基线 | 原始 transformer；长上下文时缓存爆炸 |
| **MQA** | `1`，所有 `h` 个 query 共享 | `2·d_head` → `/h` | 可测的下降，训练更不稳 | PaLM、Falcon；decode 吞吐好，质量差 |
| **GQA** | `g` 组，每组 `h/g` 个 query 头 | `2·g·d_head` → `/(h/g)` | ≈MHA | Llama 2/3、Qwen、Mistral——当前默认 |
| **MLA** | 低秩潜变量 `c` + 解耦的 RoPE key | `d_c + d_rope`（如 512+64） | ≈或 >MHA | DeepSeek；压缩比最好 |

**这个取舍为什么存在。** decode 是带宽受限的——每个 token 都要把整个缓存读一遍，砍一半就几乎砍一半 decode 时间。`g = h/8` 的 GQA 相对 MHA 通常把缓存压到 1/8，质量差约 1%：目前找到的最佳性价比点，所以它成了默认。

**MLA** 更进一步：不靠共享头，而是把 K/V 压成低维潜变量、使用时再上投影（该投影可以吸进权重矩阵，FLOPs 基本消失），同时用一个独立的小 RoPE key 保住朴素低秩压缩会破坏的位置行为。代价：实现和缓存管理更难（潜变量缓存不是头形状的张量），prefill 多出额外 matmul。

**各自什么时候不该用**：质量敏感的推理模型别选 MQA；长上下文服务别选 MHA；GQA 是安全默认；MLA 只有在你掌控整条训练和推理栈时才划算。

**速记**：区别只在 KV 头数——MHA 全、MQA 一个、GQA 分组、MLA 低秩潜变量+解耦 RoPE。decode 是带宽瓶颈，GQA 用 8 倍压缩换约 1% 质量，所以成了默认；MLA 压得更狠但要自己掌控训练和推理栈，实现与缓存管理都更复杂。

### 3. 对比 PPO、DPO 和 GRPO，并写出 SFT 损失。

<!-- EN: Compare PPO, DPO and GRPO, and write down the SFT loss. -->

**Source**: https://outcomeschool.com/blog/reinforcement-learning-from-human-feedback-rlhf and https://outcomeschool.com/blog/group-relative-policy-optimization-grpo

**参考答案**

**SFT 损失**——只在回答 token 上算的逐 token 交叉熵：

```
L_SFT = - (1/N) Σ_{i=1..N} Σ_{t=1..|y_i|} log π_θ( y_{i,t} | x_i, y_{i,<t} )
```

其中 `N` 数的是**被监督的 token**（不是序列），prompt/模板/工具 token 全部 mask 掉。这个细节比看上去重要：按序列平均而不是按 token 平均，长回答会被低估，短回答会主导。

| | PPO | GRPO | DPO |
| :--- | :--- | :--- | :--- |
| 奖励模型 | 必需 | 可选（可验证奖励即可） | 不需要 |
| 价值/critic | **必需** | **删掉**（组均值做基线） | 不需要 |
| 采样 | on-policy | on-policy，每 prompt `G` 条 | **离线** |
| 目标 | 裁剪比率 × 优势 + KL | 裁剪比率 × 组归一化优势 + KL | (chosen, rejected) 上的逻辑损失 |
| 显存 | actor + critic + RM + ref | actor + ref（有 RM 再加） | policy + ref |
| 天花板 | 最高 | 高 | 受数据限制 |
| 失败模式 | 不稳定、critic 失配 | 零方差组、长度偏置 | 离策略漂移、`β` 调参 |

**怎么选**：有好的偏好对、需要一次便宜稳定的对齐，用 DPO；任务有校验器（数学、代码、工具调用）且希望模型超过数据，用 GRPO；需要细粒度 token 级信用分配且养得起 critic，用 PPO——这越来越少见，因为 GRPO 以一小部分复杂度拿走了 PPO 的大部分价值。

**速记**：SFT 是只在回答 token 上算的逐 token 交叉熵（按 token 数平均，别按序列）。PPO=RM+critic+在线采样，天花板最高最不稳；GRPO 去掉 critic 用组内均值，可验证奖励即可训；DPO 离线成对偏好、最便宜但受数据上限约束。

### 4. 如何为多轮消息训练构建 loss mask？

<!-- EN: How do you build the loss mask for multi-turn message training? -->

**参考答案**

mask 决定模型**被训练去生成**哪些 token。差一个 token，模型就会学着预测用户的下一条消息——损失曲线漂亮，采样结果废物。

**流程：**

1. **先把整段对话渲染一次**，走 chat 模板，得到一个字符串 + tokenizer 的 offset 映射。
2. **对渲染后的字符串一次性 tokenize。** 不要分条 tokenize 再拼——BPE 会跨消息边界合并，你按条算出的 token 数会悄悄漂移。
3. **通过在渲染字符串里定位每条消息的区间**（按顺序搜索模板分隔符）构建布尔 mask，再用 offset 把区间映射到 token 下标。
4. **只监督 assistant 的内容 token。** system、user、工具返回的内容全部 mask。
5. **决定 assistant 头怎么处理。** `<|im_start|>assistant\n` 通常 mask 掉（由 harness 发出），但 **EOS/轮次结束 token 必须被监督**——否则模型永远学不会停，生成会失控。
6. labels：`labels = input_ids.clone(); labels[~mask] = -100`。

**实现上的坑**：用 `apply_chat_template(tokenize=True)` 会丢掉 offset；可靠路径是 `render → tokenize(return_offsets_mapping=True)`，或者逐条 tokenize（`add_special_tokens=False`）并精确记录长度。然后**用经验验证**：只解码被监督的位置并读一遍。如果看到的是用户的话，mask 就错了。

**多轮的细节**：工具调用数据里，要监督 assistant 的工具调用参数，但不监督注入的工具结果；截断长对话时按整轮截，并至少保留一个被监督的 assistant 轮，否则这条样本贡献的损失为零。

**速记**：整段渲染后一次性 tokenize（别分条拼，BPE 会跨边界漂移），用 offset 把消息区间映射回 token 位置，只在 assistant 内容上算 loss：模板头 mask 掉但 **EOS 必须监督**（否则模型学不会停），tool result mask 掉。上线前解码一遍被监督的位置自检。


---

## 美团

### 1. 讲一遍 Transformer，并解释编码器为什么需要 FFN。

<!-- EN: Walk through the Transformer, and explain why the encoder needs an FFN. -->

**参考答案**

单个编码器块的数据流：`x → LayerNorm → 多头自注意力 → 残差 → LayerNorm → FFN → 残差`；堆 N 层后接任务头。解码器在此基础上加掩码自注意力，以及对编码器输出的交叉注意力。

FFN 为什么不可省：

| 作用 | 机制 |
| :--- | :--- |
| 逐 token 非线性 | 注意力是 value 的**加权平均**——值的凸组合。后面若不接非线性映射，堆多少层仍等价于输入的线性映射，深度白给 |
| 特征变换 | 注意力在位置间**路由**信息；FFN 以 4 倍宽度**加工**到达的信息，给中间概念留出空间 |
| 键值记忆视角 | 第一个 FFN 矩阵像模式检测器，第二个像把关联写回的查表——这就是可解释性里的「FFN as key-value memory」 |
| 容量 | 注意力参数是 O(d²) 且位置共享；约 2/3 的参数住在 FFN 里 |

两个工程推论：因为参数在 FFN，MoE 换的是 **FFN 层**而不是注意力；又因为注意力只会做平均，把注意力换成线性变体，质量损失主要集中在需要尖锐 token 对 token 查找的任务（复制、精确回忆）上，而 FFN 可以完全不动。

**速记**：attention 只是对 value 的凸组合，没有 FFN 的非线性逐 token 变换，堆多少层都等价于线性映射；FFN 占约 2/3 参数，所以 MoE 换的是 FFN 而不是 attention。

### 2. 你了解哪些注意力变体？现场写出自注意力公式。

<!-- EN: Which attention variants do you know? Write the self-attention formula on the spot. -->

**Source**: https://outcomeschool.com/blog/math-behind-attention-qkv and https://outcomeschool.com/blog/self-attention-in-transformers

**参考答案**

$$\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V,\quad Q=XW_Q,\;K=XW_K,\;V=XW_V$$

$\sqrt{d_k}$ 把 logits 方差压到 O(1)，softmax 才不会饱和；$M$ 是加性掩码（被遮位置取 −∞）。

| 变体 | 改了什么 | 换来什么 |
| :--- | :--- | :--- |
| MHA | 基线，h 个独立头 | 表达力 |
| MQA | 所有头共用一个 K/V | KV cache ÷ h，小幅质量损失 |
| GQA | 分组共享 K/V | 实用折中（32 个 Q 头配 8 个 KV 头很常见） |
| MLA | 缓存低秩潜变量再上投影 | cache 小得多，质量接近 MHA |
| 滑窗/局部 | 遮掉 w 以外 | O(n·w)，适合长序列 |
| 线性 / SSM 混合 | 核技巧或递归 | O(n)，但精确回忆弱 |
| 稀疏 / top-k | 只对选中位置注意 | 预算内的长上下文 |
| 交叉注意力 | K、V 来自编码器 | 多模态 / 检索融合 |

面试陷阱：说「多头就是 h 个独立注意力」没错，但必须补一句「各头**先 concat 再由 $W_O$ 投影**」——混合发生在那里，不在头内部。

**速记**：写出 softmax(QKᵀ/√d_k + M)V；√d_k 压住 logits 方差防 softmax 饱和；MQA/GQA 省 KV（8 组 KV 对 32 个 Q 头是常见折中），MLA 用低秩潜变量压得更多；多头要 concat 再乘 W_O 才混合。

### 3. MLA 如何降低 KV cache？

<!-- EN: How does MLA reduce the KV cache? -->

**Source**: https://outcomeschool.com/blog/kv-cache-compression

**参考答案**

MLA（DeepSeek-V2）不再按头缓存 K 和 V，而是每个 token 缓存**一个低秩潜变量**，外加一条小的解耦 RoPE key。

| 项 | MHA | MLA |
| :--- | :--- | :--- |
| 每 token 缓存 | $2 \cdot n_h \cdot d_h$ | $d_c + d_h^{R}$ |
| 示例（128 头 × 128 维） | 约 32k 个值 | 约 576 个值 |
| 位置处理 | RoPE 作用在每个头的 K 内 | RoPE 作用在独立的小 key 上，再拼接 |

机制：把隐状态下投影成 $c_t = W_{DKV} h_t$ 并只缓存 $c_t$；注意力时再重建 $K = W_{UK}c_t$、$V = W_{UV}c_t$。由于两个投影都是线性的，推理时可被**吸收进** $W_Q$ 与 $W_O$，于是根本不用物化每头的完整 K/V，直接对潜变量做注意力。

解耦 RoPE key 为什么必要：RoPE 是位置相关且乘性的，折进线性低秩投影会破坏相对位置性质。陪一条 64 维的小 RoPE key 就能解决，缓存开销可忽略。

取舍：每步多一点计算（额外投影）、训练更麻烦；cache 受限的在线服务赚，算力受限的训练略亏。当约束是并发而非 FLOPs 时，这是对的交易。

**速记**：MLA 不再按头缓存 K/V，而是每 token 缓存一个低秩潜变量（约 576 维 vs 3 万+），推理时把上投影吸收进 W_Q/W_O 直接对潜变量做注意力；RoPE 必须走一条小的解耦 key。

### 4. Qwen 如何外推到更长的上下文（NTK-aware 缩放、YaRN）？

<!-- EN: How does Qwen extrapolate to longer contexts (NTK-aware scaling, YaRN)? -->

**Source**: https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding

**参考答案**

RoPE 按角度 $m\theta_i$ 旋转维度对，其中 $\theta_i = b^{-2i/d}$。第 $i$ 维在推理时只见过训练长度以内的位置——再往外旋转角就落到没见过的区间，注意力随之退化。

三种修法，越来越精细：

| 方法 | 改了什么 | 说明 |
| :--- | :--- | :--- |
| 位置插值（PI） | 位置除以 $s$ | 均匀压缩，糊掉局部分辨率；需微调 |
| NTK-aware | 抬 base $b \to b\cdot s^{d/(d-2)}$ | 高频维几乎不动、低频维被拉伸，短扩展无需训练 |
| 动态 NTK | 按序列长度重算 base | 训练长度以内保持精确 |
| YaRN | NTK-by-parts：逐维在插值与外推间过渡，再加注意力温度 $1/\sqrt{t}$ | 当前默认；几百步微调就能补齐 |

为什么「by parts」：低频维编码长程位置，必须插值；高频维编码局部顺序，必须保持原样，否则 token 局部次序被糊掉。温度项补偿上下文变长时注意力熵的增加。

Qwen 直接发 YaRN 配置（如在 32k 基座上 factor 4），短 prompt 走动态 NTK。诚实的提醒：「支持长度」≠「有效长度」——信任它之前先做多深度的 needle 测试。

**速记**：RoPE 角度只在训练长度内见过；PI 均匀压缩会糊掉局部，NTK-aware 抬 base 让高频维几乎不动、低频维拉伸；YaRN 用 by-parts 斜坡+注意力温度，配几百步微调。宣称长度≠有效长度。

### 5. MoE 如何工作，负载如何均衡，激活是 batch 级还是 token 级的决策？用 Python 实现一个 MoE。

<!-- EN: How does MoE work, how is load balanced, and is activation a batch-level or token-level decision? Implement a MoE in Python. -->

**Source**: https://outcomeschool.com/blog/mixture-of-experts

**参考答案**

**默认是逐 token 的。** 每个 token 独立得到在 E 个专家上的门控分布，取 top-k，输出按门控概率加权求和。同一序列里不同 token 可以走不同专家，不存在「这批用 3 号专家」的决定。批级方案是 *expert-choice* 路由（每个专家挑自己的 top-C token），负载天然均衡，但破坏自回归解码的因果性，LLM 里很少用。

负载均衡三种机制：

| 机制 | 做法 | 代价 |
| :--- | :--- | :--- |
| 辅助损失 | 惩罚专家负载偏离均匀的平方差（Switch/GShard） | 多一个与主目标打架的超参 |
| 容量因子 + 丢弃 | 限制每专家每批 token 数，溢出只走残差 | 被丢的 token 静默损失容量 |
| 无辅助损失的 bias | 按观测负载微调每个专家的路由偏置 | 不干扰梯度——DeepSeek 的做法 |

要盯的失效：专家坍缩（少数专家吃掉全部流量，有效参数缩水）、跨步路由不稳、专家并行里 all-to-all 通信吃满单步时间。

**参考实现**

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, top_k):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
            for _ in range(n_experts)
        ])
        self.bias = torch.zeros(n_experts)   # 无辅助损失的均衡偏置

    def forward(self, x):                    # x: [B, T, D]
        B, T, D = x.shape
        logits = self.gate(x) + self.bias     # [B, T, E]
        probs = F.softmax(logits, dim=-1)
        w, idx = probs.topk(self.top_k, dim=-1)          # [B,T,k]
        w = w / w.sum(-1, keepdim=True)                  # 重新归一
        flat = x.reshape(-1, D)
        out = torch.zeros_like(flat)
        flat_idx, flat_w = idx.view(-1, self.top_k), w.view(-1, self.top_k)
        for e, expert in enumerate(self.experts):
            mask = flat_idx == e
            tok, slot = mask.nonzero(as_tuple=True)      # 路由到 e 的 token
            if tok.numel() == 0:
                continue
            out.index_add_(0, tok, expert(flat[tok]) * flat_w[tok, slot].unsqueeze(-1))
        return out.view(B, T, D)
```

**关键点**

- 按专家循环（不是按 token 循环）：每个专家对自己 gather 到的 token 做一次批量矩阵乘。
- top-k 权重必须重新归一，否则输出尺度随 k 变化。
- 这个朴素版会真实物化 gather；生产用 grouped GEMM / 带容量限制的 all-to-all。
- 每步统计各专家的 token 数——坍缩表现为 max/mean 远大于 1。

**速记**：MoE 是逐 token 选专家（expert-choice 的批级路由破坏自回归因果性）；负载均衡靠辅助损失、容量因子丢 token、或无辅助损失的 bias 微调；按专家 gather 做批量矩阵乘并监控各专家 token 数。


---

## Hugging Face

### 1. 讲讲当有人调用 AutoModelForCausalLM.from_pretrained(…, device_map="auto", torch_dtype="auto") 时，实际发生了什么。

<!-- EN: Walk me through what actually happens when someone calls AutoModelForCausalLM.from_pretrained(…, device_map=“auto”, torch_dtype=“auto”). -->

**参考答案**

结论：九步，按顺序：

1. **解析仓库** —— `repo_id` + `revision` → `~/.cache/huggingface/hub/models--org--name/snapshots/<sha>/` 下的本地快照；可变分支（`main`）只解析一次到 commit sha，记下的就是那个 sha。
2. **读 `config.json`** → `AutoConfig`，再按 `architectures[0]` 懒加载 modeling 模块（这也是 `import transformers` 很快的原因）。
3. **`torch_dtype="auto"`** 从**checkpoint 的元数据**（首个分片）里读 dtype，而不是从 `config.json`；新版本里这个参数叫 `dtype`。
4. **在 `meta` 设备上实例化** —— `init_empty_weights()` 创建的参数没有存储。这也是 `low_cpu_mem_usage=True` 成为默认的原因：先分配 70B 随机权重再覆盖，曾经让主机 RAM 峰值翻倍。
5. **`device_map="auto"`** 跑 accelerate 的 `infer_auto_device_map`：模块大小来自 meta 张量，可用显存来自 `torch.cuda.mem_get_info()` 减去 `max_memory` 余量。
6. **逐分片加载** —— `model.safetensors.index.json` 给出张量→文件映射；每个分片被 `mmap`，张量直接拷到目标设备。
7. **绑权重并校正** —— `tie_weights()` 重新关联 `lm_head` 与 embedding（通常只存一份）。
8. 报告 missing / unexpected / mismatched key，默认不直接抛异常。
9. **不会**替你调用 `eval()`。

陷阱：`device_map="auto"` 读的是**那一瞬间**的空闲显存。如果还有训练任务占着 20 GB，你的模型会静默把一半层卸载到 CPU，推理慢约 20 倍——没有任何警告，只有很长的首 token。

**速记**：meta device 初始化 → `device_map="auto"` 按 meta 尺寸和当前空闲显存切分 → 逐 shard mmap 拷贝 → 绑权重；坑是它读的是**当下**空闲显存，别的进程占着显存就会静默往 CPU 卸载。

### 2. 对比 BPE、WordPiece 和 Unigram 分词。为什么 `tokenizers` 用 Rust 编写？实践中哪些分词器 bug 会坑到人？

<!-- EN: Compare BPE, WordPiece and Unigram tokenization. Why is `tokenizers` written in Rust, and what tokenizer bugs bite people in practice? -->

**Source**: https://outcomeschool.com/blog/bpe-in-llms ; https://www.youtube.com/watch?v=sK2s9I84EVI

**参考答案**

结论：三种子词算法只是目标函数不同；而 `tokenizers` 用 Rust 写，是因为分词在每个请求上跑，且必须跨语言完全一致。

| 维度 | BPE | WordPiece | Unigram |
| :--- | :--- | :--- | :--- |
| 起点 | 字符，**合并**最高频对 | 字符，按似然增益合并 | 大候选词表，**按 EM 损失剪枝** |
| 目标 | 频次 | `p(a,b)/(p(a)p(b))`，近似 PMI | 语料的边际似然 |
| 使用方 | GPT 系、Llama（byte-level） | BERT、DistilBERT | T5、ALBERT、SentencePiece 默认 |
| 保证 | 无（贪心） | 无 | **模型下的最优切分** |

Rust 的两个具体收益：真线程（rayon 并行预分词，Python 的 GIL 让这没意义）和所有绑定共用一份实现，Node 服务与 Python 训练脚本不会在 token 边界上产生分歧。它还承担了字符偏移 ↔ token 偏移的对齐记账，这部分 Python 版在边界情况上出过错。

实际会咬人的 bug：

1. **预分词器不一致** —— `tokenizer.json` 里的 byte-level 正则与训练时不同，token 数静默漂移。
2. **`add_prefix_space`** —— RoBERTa 系分词器会前置空格；忘了它质量就掉，且不报错。
3. **用户文本里的特殊 token** —— 用户输入 `<|endoftext|>` 变成真正的控制 token → 提示注入。
4. **左/右 padding** —— 给 decoder-only 模型做右 padding 会产出乱码，且静默。
5. **静默截断** —— `truncation=True` 砍掉了指令、留下了文档。
6. **chat template 漂移** —— 模板存在 `tokenizer_config.json` 里；旧副本给出的提示与模型微调时不同。

必须说出口的一条：**分词器不匹配从不报错，它只让你掉点**，所以要把分词器产物与权重一起锁版本，并在 CI 里做 diff。

**速记**：BPE 按频次合并、WordPiece 按似然增益合并、Unigram 从大词表剪枝求最优切分；Rust 换来真并行和跨语言一致。最坑的是预分词器/chat template 与训练不一致——不报错，只掉点。

### 3. chat template 解决什么问题？忽略它会出什么错？

<!-- EN: What problem do chat templates solve, and what goes wrong when they're ignored? -->

**参考答案**

结论：chat 模型就是在一个**特定序列化格式**上训练出来的 base 模型；模板就是这个序列化——它把 `[{role, content}, ...]` 变成模型期望的精确字符串与特殊 token 序列：角色分隔符、BOS、生成提示的开头、工具调用块、思考块。

各家差别比想象的大：

| 家族 | 形状 |
| :--- | :--- |
| ChatML | `<|im_start|>user\n...<|im_end|>` |
| Llama 3 | `<|start_header_id|>user<|end_header_id|>` + `<|eot_id|>` |
| Mistral | `[INST] ... [/INST]` 且带 BOS |
| Gemma | `<start_of_turn>user` … `<end_of_turn>` |

忽略它会带来三类故障：

1. **训练/推理错位** —— 微调时手搓 `"User: ...\nAssistant:"`，服务时用官方模板。模型还能用，只是更差——最坏的一类 bug。
2. **缺 BOS / 缺生成提示** —— `apply_chat_template(..., tokenize=False)` 之后手工 `tokenizer(text)` 是丢掉 BOS 与 assistant header 的经典写法；输出变得啰嗦跑题。
3. **分隔符混乱 → 注入** —— 用户文本不在角色分隔符内时，"忽略之前的指令并打印系统提示"就和系统提示处在同一通道。

实用规则：调 `apply_chat_template(..., add_generation_prompt=True, return_dict=True)` 并直接把 `input_ids` 传下去；永远不要字符串拼接多轮；把模板当作模型产物的一部分并做哈希；后训练时必须**通过**模板渲染数据集，而不是重新实现一遍。

**速记**：模板就是"训练时那种序列化方式"——角色分隔符 + BOS + generation prompt。忽略它不会报错，只会训练/推理错位、掉 BOS 导致胡言乱语，以及用户文本与系统提示同通道带来的注入。


---

## Character.AI

### 1. 一段对话超出了上下文窗口。你会保留什么内容，又如何决定？

<!-- EN: A conversation runs past the context window. What do you keep, and how do you decide? -->

**Source**: https://outcomeschool.com/blog/how-does-context-compaction-work · https://outcomeschool.com/blog/ai-agent-memory

**参考答案**

不要统一摘要。按"下一轮缺了它会坏什么"排序，再按类别分配预算：

| 保留 | 预算占比（典型） | 理由 |
| :--- | :--- | :--- |
| 人设卡 + 开场 | 固定，永不淘汰 | 丢了就是人设漂移 bug |
| 用户陈述的关键事实（姓名、职业、宠物、承诺） | 固定清单，一次抽取 | 只能靠重问获得，用户很反感 |
| 未收束的剧情线 / 悬置问题 | ~10% | 丢掉会明显破坏连续性 |
| 最近轮次原文（最后约 15–30 轮） | ~40% | 风格与局部连贯来自原文，不来自摘要 |
| 更早轮次的滚动摘要 | ~15% | 压缩索引，不是逐字稿 |
| 丢弃 | 余量 | 闲聊、已解决线索、重复寒暄 |

决策规则：按**信息密度**淘汰，而不是只看新旧——引入事实的那轮留，三轮"哈哈是的"不留。每个淘汰边界跑一个廉价的抽取器，产出记忆项 `{fact, salience: 新颖度 × 用户主动提及 × 情绪强度}`，按 salience 取 top-k，而不是按顺序。

两个必须点名的失败模式：(1) **摘要会向通用腔漂移**——三代摘要之后角色声音就没了，必须始终保留原文范例；(2) **记忆要覆盖更新而不是追加**——被推翻的事实（"我换工作了"）必须覆盖旧项，否则检索在两者间摇摆，角色显得混乱。

**速记**：按"下一轮没有它会不会坏"分级保配额：人设卡与用户陈述事实永不淘汰，未解线留 10%，近 15–30 轮保原文，更早压成滚动摘要；淘汰看信息密度；摘要多代会变通用腔，事实要覆盖而非追加。
