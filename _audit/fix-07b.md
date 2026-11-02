# 修复记录：07b.md（`07-编码与数据结构.md` 第 59–116 题）

覆盖 `_audit/07b.md` 的 20 处问题（严重 4 / 中等 9 / 轻微 7）。全部处理。

| 题号 | 报告问题 | 严重度 | 改法 | 处理 |
| --- | --- | --- | --- | --- |
| 93 | `claim` 只接受 `UNASSIGNED`，k 人共识永远达不成（死锁） | 严重 | 改为租约式领取：`leases[aid]`，`_open()` 判定「已提交 + 在飞租约 < k」，交卷即归还租约，同一题可被 k 人同时领 | 已修 |
| 93 | `agree() >= 0.67` 把 2/3 判为不通过 | 严重 | `agree` 返回整数票数，改判 `3*票数 >= 2*人数`，要点补浮点坑说明 | 已修 |
| 114 | 「bug」是编造的，被标注为错误的代码其实是对的 | 严重 | 换成真实缺陷：`for node in frontier` 时 `frontier.append(nb)`，Python 按索引迭代会走到当天新增元素，一天走完多跳；正文/要点/速记同步改写 | 已修 |
| 116 | 单字段（姓名）命中就自动合并，两个 John Smith 变一个 | 严重 | 强标识（含 `@` 的邮箱、10 位电话）才 union；姓名桶只产出 `review` 候选对（含 `city_same` 标记，桶内 pair 数上限 20） | 已修 |
| 59 | `latest(k)` 未对有效元素数 clamp，返回未写槽位 | 中等 | `valid = min(k, head-tail)`，返回 `(ts, vals, ok)` 掩码标出真实数据；补要点 | 已修 |
| 62 | 历史缺 assistant 的 `tool_calls`，`role:"tool"` 无主语；重复分支吞掉响应；`seen` 死变量 | 中等 | 每轮先 append assistant（含 `tool_calls` id），tool 消息带 `tool_call_id`；重复检测改为回灌 ERROR 观察；删 `seen`；补 `SYSTEM`/`schemas` 使块自洽 | 已修 |
| 95 | 错误分类取 `exc.status` 而非 `status_code`，故障转移整体失效 | 中等 | `status_code` 优先、回落 `status`，判不出则 FATAL；只有 RETRYABLE 才计入熔断 | 已修 |
| 96 | LSH 阈值说成「≈0.8」，与自己给的公式矛盾 | 中等 | 按 `P(s)=1-(1-s^4)^32` 复算：P(0.3)=0.23、P(0.5)=0.87、P(0.6)=0.99，50% 检出点 s≈0.38，陡峭区 0.35–0.55；速记同步 | 已修 |
| 98 | 核心重建步骤用 `...` 省略、`scores[:beam]` 只对 batch 0 生效、`done` 无上限 | 中等 | 补全 beam 重建（batch-major `view(B, beam)`、EOS 出束、占位补齐形状、`done` 按分数截断到 beam、无 EOS 时回退最优活假设） | 已修 |
| 105 | 调用未定义的 `_split_sentences`；`CAP` 注释说 token 实际比字符 | 中等 | 定义 `_split_sentences`（句末标点打包、单句超长整句放出）；`CAP_CHARS = 2800 ≈ 700 token`，`chunk(..., cap=CAP_CHARS)` | 已修 |
| 106 | docstring 签名 `(q,user,n)` 与调用 `s(user,n)` 不符；「parallel」实为串行 | 中等 | 签名补 `q` 并调用 `s(q, user, n)`；改 `ThreadPoolExecutor` 真并行（`map` 保序），空分片/k<=0 早返回 | 已修 |
| 109 | 声称的 `gt_len==0` 防护不存在（ADE 除以 0 → nan） | 中等 | 除数 `np.maximum(gt_len,1)`，`mask` 叠加 `valid`，`gt_len==0` 整条置 NaN 交调用方剔除；docstring/要点同步 | 已修 |
| 116 | 「按字段定权威源」的存活规则未实现 | 中等 | 逐字段按 `TRUST` 取首个非空值，电话按 `updated_at` 取最新；`id` 取最可信源，保留 `member_ids`/`sources` | 已修 |
| 63 | `out.pop(1 if out[1]["role"]=="system" else 0)` 会删 pinned | 轻微 | 固定从 `len(pinned)` 位置裁（先摘要、再最早近期轮次），并补 `one_line` 定义使块可运行 | 已修 |
| 80 | `close()` 后 `log()` 静默 append，消息永久丢失 | 轻微 | `_closed` 时 `log()` 抛 `RuntimeError`（含阻塞被唤醒的情况）；要点补说明 | 已修 |
| 87 | 用到 `collections.deque` 但未 import | 轻微 | 代码块补 `import collections` | 已修 |
| 89 | 「vLLM/TPI」缩写错误 | 轻微 | 改为 `vLLM/TGI` | 已修 |
| 97 | `pack()` 被称作 first-fit-decreasing，实为 next-fit decreasing | 轻微 | 结论/docstring/速记统一改为 next-fit decreasing，并注明 FFD 需维护多个未满批 | 已修 |
| 99 | 声称带「类型位」，payload 里没有 | 轻微 | payload 改为 `name:类型位:child_hash` 且目录哈希加 `tree:` 前缀分域；表格同步 | 已修 |
| 91 | 「截断下载会接近段错误」表述绝对化（存疑项） | 轻微 | 改为「结构校验只在打开时做一次；映射建立后文件被截断才会以 SIGBUS 出现在读页时」 | 已修 |

## 未处理项

| 题号 | 原因 |
| --- | --- |
| 无 | 20 处全部处理（Q91 为报告标注的存疑项，按「去掉绝对化断言」处理） |

## 自检

- 文件题数：116（应为 116）
- `### ` / `<!-- EN: ` / `**参考答案**` / `**速记**：` 四处数量一致：是（116 / 116 题内各 1 条；全文 `<!-- EN: ` 另在文件说明行出现 1 次，属原有文案）
- 实际跑过的代码题：
  - Q93：k=3 时 alice→bob→carol 依次领到同一题、第 4 人领不到；3/3 通过、2/3 通过、1/3 进 `IN_REVIEW`；重复提交/无租约提交被拒；过期租约可重领 → 通过
  - Q114：新 buggy 版对 `A→B→C`、`days=1` 返回 `{A,B,C}`（真 bug 复现）；修复版 `days=1→{A,B}`、`days=2→{A,B,C}`、`per_day=[{A},{B},{C}]`，自环/环/无向图用例通过 → 通过
  - Q116：两个 John Smith（不同邮箱/电话/城市）→ 2 个实体 + 1 条候选；同邮箱的三源记录合并为 1 条、`member_ids` 保序、CRM 缺字段由 ERP 回填、电话取 `updated_at` 最新 → 通过
  - Q59：push 1 条后 `latest(3)` 返回 `[0,0,1]` 且 `ok=[F,F,T]`；写满后 `ok` 全真、`dropped==2` → 通过
  - Q62：假 LLM 三轮工具调用，历史中每条 tool 消息都有前驱 assistant 且 `tool_call_id` 命中；重复调用与未知工具都产生 ERROR 观察；预算耗尽走强制总结 → 通过
  - Q98：`torch` 实跑（临时安装 2.15.0）：无 EOS 回退、EOS 出束、`beam=1` 退化贪心、每 batch 凑满 beam 条；并与暴力枚举在 beam 覆盖全空间时 Top-1 分数/序列完全一致 → 通过
  - Q105：3419 字符的超长条款切成 2 片（2774/644，均 ≤ cap），拼接回原文一致；句子不被截断；`part 1/2` 前缀与术语上下文正确 → 通过
  - Q106：k 路归并结果与暴力归并一致、跨分片去重、空分片/k=0/分片不足；6 个 0.25 s 分片总耗时 0.26 s（真并行） → 通过
  - Q109：`gt_len=[0,1,2,3,4]` → 第 0 条 ade/fde 为 NaN，其余与逐场景循环参考实现逐项相等；全 0 与超 T 用例通过 → 通过
  - Q63：pinned 消息在极端预算下保留；裁剪只发生在 pinned 之后；未到 80% 原样返回 → 通过
  - Q80：4 生产者 × 25 条各 1 条不丢（100 条全部落盘），`close()` 排空，关闭后 `log()` 抛 `RuntimeError` → 通过
  - Q99：相同快照同哈希、内容变化改哈希、同内容文件共享 blob 哈希、空文件与空目录不再撞哈希 → 通过
