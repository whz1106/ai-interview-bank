# Interview Bank

面试官自用题库。题目来自公开报道的真实面试经历，**答案为本仓库自行整理**，每条附原文出处链接供延伸阅读。

> **为什么不是直接搬原文**：LinkedIn / Outcome School 的文章有版权，逐字搬运不合规；本仓库的答案是**重写的技术要点**（技术事实不受版权保护，表达受保护），并保留 `Source` 链接，需要细节时回原文看。

## 使用方式

- 每条题目结构：`Question` → `Asked at`（出现过的公司）→ `Anchor answer`（可直接照着问的答案骨架）→ `中文要点`（速记一行）
- 面试时**不需要全问**，按候选人简历命中的技术点挑 5~8 条
- 想加自己的观察记录，直接在该题下方追加 `> 实测：...`

## 目录

| 文件 | 题数 | 主题 | 状态 |
| :--- | :-: | :--- | :--- |
| [01-agents-and-tools.md](./01-agents-and-tools.md) | 12 | Agent 与工具调用 | ✅ |
| [02-rag-and-retrieval.md](./02-rag-and-retrieval.md) | 12 | RAG 与检索 | ✅ |
| [03-evaluation-and-observability.md](./03-evaluation-and-observability.md) | 10 | 评估与可观测 | ✅ |
| 04-inference-and-gpu.md | 15 | 推理服务与 GPU 性能 | ⬜ |
| 05-fine-tuning-and-alignment.md | 12 | 微调 / 后训练 / 对齐 | ⬜ |
| 06-system-design.md | 10 | AI 系统设计 | ⬜ |
| 07-coding.md | 12 | 编码与数据结构 | ⬜ |
| 08-safety-and-security.md | 10 | 安全与负责任 AI | ⬜ |
| 09-llm-internals.md | 16 | LLM 内部机制与架构 | ⬜ |
| 10-multimodal-and-voice.md | 10 | 多模态 / 语音 / 音频 | ⬜ |

## 来源

- 题目清单与公司归属：`ninetyhe-90/ai-engineering-interview-questions-company-wise-zh-cn`（英文原版 `pallavi-shekhar/ai-engineering-interview-questions-company-wise`）
- 延伸阅读：各题 `Source` 链接指向 Outcome School 等公开博客

## 合规提示

本仓库若为 **public**，候选人可直接搜到这里。若题库要长期存放，建议：
1. 把 `interview-bank/` 移到**私有仓库**；或
2. 本仓库转 **private**，公开的候选人题面另建一个仓库。

`README.md`（候选人任务）与 `interview-bank/`（面试官题库）应分仓存放。
