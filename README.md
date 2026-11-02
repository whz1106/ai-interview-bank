# Interview Bank

面试官自用题库。题目来自公开报道的真实面试经历（源仓库 Apache-2.0），**答案为本仓库自行整理**，每条附 `Source` 链接供延伸阅读。

> 为什么不直接搬原文：Outcome School 等博客正文有版权，逐字搬运不合规。本仓库的答案是**重写的技术要点**（技术事实不受版权保护，表达受保护），需要细节时回原文。授权与归属见 [`NOTICE.md`](./NOTICE.md)。

## 使用方式

- 每题结构：`Question` → `Asked at` → `Source` → `Anchor answer` → `中文要点`
- 面试时**不要全问**，按候选人简历命中的技术点挑 5~8 条
- 想记自己的观察，在该题下方追加 `> 实测：...`
- 写作规格见 [`_SPEC.md`](./_SPEC.md)（新增题目前先读）

## 跨公司通用（119 题）

| 文件 | 题数 | 主题 |
| :--- | :-: | :--- |
| [01-agents-and-tools.md](01-agents-and-tools.md) | 12 | Agent 与工具调用 ✅ |
| [02-rag-and-retrieval.md](02-rag-and-retrieval.md) | 12 | RAG 与检索 ✅ |
| [03-evaluation-and-observability.md](03-evaluation-and-observability.md) | 10 | 评估与可观测 ✅ |
| [04-inference-and-gpu.md](04-inference-and-gpu.md) | 15 | 推理服务与 GPU 性能 |
| [05-fine-tuning-and-alignment.md](05-fine-tuning-and-alignment.md) | 12 | 微调 / 后训练 / 对齐 ✅ |
| [06-system-design.md](06-system-design.md) | 10 | AI 系统设计 |
| [07-coding.md](07-coding.md) | 12 | 编码与数据结构（含可运行代码） |
| [08-safety-and-security.md](08-safety-and-security.md) | 10 | 安全与负责任 AI |
| [09-llm-internals.md](09-llm-internals.md) | 16 | LLM 内部机制与架构 |
| [10-multimodal-and-voice.md](10-multimodal-and-voice.md) | 10 | 多模态 / 语音 / 音频 |

## 前沿 AI 实验室（197 题）

| 文件 | 题数 | 公司 |
| :--- | :-: | :--- |
| [11-anthropic.md](11-anthropic.md) | 37 | Anthropic |
| [12-openai.md](12-openai.md) | 31 | OpenAI |
| [13-google-deepmind.md](13-google-deepmind.md) | 22 | Google DeepMind & Google AI |
| [14-meta.md](14-meta.md) | 26 | Meta (Superintelligence Labs, FAIR, Llama) |
| [15-xai.md](15-xai.md) | 10 | xAI |
| [16-mistral.md](16-mistral.md) | 8 | Mistral AI |
| [17-cohere.md](17-cohere.md) | 10 | Cohere |
| [18-deepseek.md](18-deepseek.md) | 10 | DeepSeek |
| [19-moonshot.md](19-moonshot.md) | 10 | Moonshot AI (Kimi) |
| [20-zhipu.md](20-zhipu.md) | 11 | Zhipu AI (GLM) |
| [21-alibaba-qwen.md](21-alibaba-qwen.md) | 10 | Alibaba (Qwen) |
| [22-sarvam.md](22-sarvam.md) | 12 | Sarvam AI |

## 大型科技公司（82 题）

| 文件 | 题数 | 公司 |
| :--- | :-: | :--- |
| [23-microsoft.md](23-microsoft.md) | 10 | Microsoft |
| [24-amazon.md](24-amazon.md) | 23 | Amazon (AWS) |
| [25-apple.md](25-apple.md) | 11 | Apple |
| [26-nvidia.md](26-nvidia.md) | 13 | NVIDIA |
| [27-tesla.md](27-tesla.md) | 11 | Tesla |
| [28-consumer-ml.md](28-consumer-ml.md) | 14 | Uber / Netflix / LinkedIn / Airbnb / Pinterest / Spotify |

## 中国互联网公司（184 题）

| 文件 | 题数 | 公司 |
| :--- | :-: | :--- |
| [29-bytedance.md](29-bytedance.md) | 38 | 字节跳动 |
| [30-tencent.md](30-tencent.md) | 37 | 腾讯 |
| [31-alibaba-taotao.md](31-alibaba-taotao.md) | 27 | 阿里巴巴（淘天 / 阿里云） |
| [32-didi.md](32-didi.md) | 21 | 滴滴 |
| [33-jd.md](33-jd.md) | 27 | 京东 |
| [34-meituan.md](34-meituan.md) | 34 | 美团 |

## AI 基础设施与平台公司（66 题）

| 文件 | 题数 | 公司 |
| :--- | :-: | :--- |
| [35-databricks.md](35-databricks.md) | 9 | Databricks |
| [36-groq.md](36-groq.md) | 12 | Groq |
| [37-together.md](37-together.md) | 7 | Together AI |
| [38-huggingface.md](38-huggingface.md) | 11 | Hugging Face |
| [39-scale-ai.md](39-scale-ai.md) | 12 | Scale AI |
| [40-perplexity.md](40-perplexity.md) | 15 | Perplexity |

## AI 原生应用公司（114 题）

| 文件 | 题数 | 公司 |
| :--- | :-: | :--- |
| [41-cursor.md](41-cursor.md) | 17 | Cursor (Anysphere) |
| [42-cognition.md](42-cognition.md) | 9 | Cognition (Devin, Windsurf) |
| [43-sierra.md](43-sierra.md) | 12 | Sierra |
| [44-harvey.md](44-harvey.md) | 11 | Harvey |
| [45-glean.md](45-glean.md) | 10 | Glean |
| [46-character-ai.md](46-character-ai.md) | 12 | Character.AI |
| [47-elevenlabs.md](47-elevenlabs.md) | 8 | ElevenLabs |
| [48-abridge.md](48-abridge.md) | 11 | Abridge |
| [49-figure-waymo.md](49-figure-waymo.md) | 24 | Figure AI (12) + Waymo (12) |

## 前向部署与企业级 AI（20 题）

| 文件 | 题数 | 公司 |
| :--- | :-: | :--- |
| [51-palantir.md](51-palantir.md) | 20 | Palantir |

## 合计

**782 题 / 47 个文件**（跨公司 119 + 公司专属 663）

## 来源

- 题目清单与公司归属：`ninetyhe-90/ai-engineering-interview-questions-company-wise-zh-cn`（Apache-2.0）
- 英文原版：`pallavi-shekhar/ai-engineering-interview-questions-company-wise`
- 延伸阅读：各题 `Source` 链接

## 说明

本仓库**仅存放面试题库**，不包含候选人题面。候选人用的实操任务已移出（备份在原工作目录 `归档-候选人任务/`），需要时另建独立仓库，避免题库与题面混在一处。
