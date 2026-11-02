# 写作规格（子代理必读）

## 输入
唯一题目来源（英文逐字）：
`/Users/whz/Downloads/招聘简历/ai-engineering-interview-questions-company-wise-zh-cn/README.en.md`

用 `### <Section Name>` 定位章节，取该章节内**所有以 `- ` 开头且不以 `  -` 开头**的行作为题目（逐字保留英文）。

每题的元信息来自紧随其后的缩进行：
- `  - Asked at: [A](#a), [B](#b)` → 提取公司名，去 markdown 链接
- `  - Answer: ...` → 提取其中的 URL 作为 `Source`

## 输出
目录：`/Users/whz/Downloads/招聘简历/ai-chatroom-challenge/interview-bank/`
文件名与章节对应关系由任务单独指定。

## 文件格式（严格照此，参考 01-agents-and-tools.md）

```md
# <两位编号> · <主题英文名>

> <N> 题。出处：`README.en.md` § <Section Name>。

---

### 1. <题目，英文逐字，去掉行尾句点保持原样>

**Asked at**: 公司A · 公司B · 公司C
**Source**: https://...

**Anchor answer**

<答案正文>

**中文要点**：<一行中文速记，30~60 字>

---

### 2. ...
```

- 无 `Asked at` 时**整行省略**；无 `Source` 时**整行省略**
- 最后一题后加：

```md
---

## 附：这 N 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| ... | ... |
```

## 答案写法（关键）

1. **必须自己写，禁止逐字复制任何博客正文**（有版权）。技术事实可自由使用。
2. 长度 **120~260 词**，英文。用表格、有序列表、对照结构，避免大段散文。
3. 每个答案要有一个**「只有真做过才知道」的钩子**：一个数字、一个失败模式、一个具体取舍、一条反直觉结论。
4. 结构建议：一句话定义/结论 → 分类表或机制分解 → 关键取舍与失败模式 → 一句"什么时候别用它"。
5. 如果题目是纯 coding 题（"Implement X"），写：`**Reference implementation**` 代码块（Python）+ `**Key points**` 要点列表，不要写虚构的性能数字。
6. 如果题目是设计题，写：目标约束 → 架构分段（表）→ 关键失败模式 → 验证方法。

## 硬性禁止
- 禁止编造公司名或"Asked at"归属（只照抄题目源里的）
- 禁止编造具体数字当作已被验证的事实；确需举例用「典型经验值」并标注
- 禁止写 `TODO`、`待补充`、占位符
- 禁止修改 `README.md`、`NOTICE.md`、`LICENSE-questions-source.txt` 及 01/02/03/05 已有文件
