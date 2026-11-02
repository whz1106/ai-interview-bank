# 51 · Palantir

> 20 题。出处：`README.en.md` § Palantir。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Implement a set of shape classes that compute area, then extend them to handle a new shape.

**Reference implementation**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math


class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...

    def describe(self) -> str:
        return f"{type(self).__name__}(area={self.area():.3f})"


@dataclass(frozen=True)
class Circle(Shape):
    r: float
    def __post_init__(self):
        if self.r <= 0: raise ValueError("radius must be > 0")
    def area(self): return math.pi * self.r ** 2


@dataclass(frozen=True)
class Rectangle(Shape):
    w: float
    h: float
    def __post_init__(self):
        if self.w <= 0 or self.h <= 0: raise ValueError("sides must be > 0")
    def area(self): return self.w * self.h


@dataclass(frozen=True)
class Triangle(Shape):
    a: float
    b: float
    c: float
    def __post_init__(self):
        s = sorted((self.a, self.b, self.c))
        if s[0] <= 0 or s[0] + s[1] <= s[2]:
            raise ValueError("violates triangle inequality")
    def area(self):
        s = (self.a + self.b + self.c) / 2
        return math.sqrt(s * (s - self.a) * (s - self.b) * (s - self.c))


class Polygon(Shape):          # extension: no existing class changes
    def __init__(self, points):
        self.pts = tuple(points)
        if len(self.pts) < 3: raise ValueError("need >= 3 points")
    def area(self):
        x, y = zip(*self.pts)
        return 0.5 * abs(sum(x[i] * y[i - 1] - x[i - 1] * y[i]
                             for i in range(len(x))))
```

**Key points**

1. **Abstract base + one method per behaviour** = the answer to "extend it": a new shape is a new class, zero edits elsewhere (OCP). The anti-pattern is `if shape.type == "circle"` in a `total_area()` function — every new shape edits that function.
2. **Validate in the constructor**, not in `area()`. A `Triangle(1,1,5)` that constructs successfully and returns `nan` is worse than an exception.
3. **`frozen=True`** for value semantics: shapes are hashable, comparable, and cannot be mutated into an invalid state after validation.
4. If asked to add *composition* (a `Composite`/`Group` of shapes), note that `area()` then recurses — and that a visitor/registry is the alternative when you cannot modify classes (e.g. a third-party shape).

**中文要点**：抽象基类 + 构造时校验 + `frozen` 值对象；新增形状 = 新增子类，不改任何已有代码，而不是在 `total_area()` 里堆 `isinstance` 分支。

---

### 2. Write a SQL query that joins and aggregates across tables to answer a business question.

**Answer**

Question: *monthly completed revenue and order count per region for 2025, including regions with zero revenue that month, plus each region's share of the month.*

```sql
WITH completed AS (
    SELECT c.region,
           date_trunc('month', o.created_at)::date AS month,
           SUM(oi.qty * oi.unit_price)             AS revenue,
           COUNT(DISTINCT o.id)                    AS orders
    FROM orders o
    JOIN customers   c  ON c.id = o.customer_id
    JOIN order_items oi ON oi.order_id = o.id
    WHERE o.status = 'completed'
      AND o.created_at >= DATE '2025-01-01'
      AND o.created_at <  DATE '2026-01-01'
    GROUP BY 1, 2
),
months AS (SELECT DISTINCT month FROM completed)
SELECT r.region,
       m.month,
       COALESCE(c.revenue, 0) AS revenue,
       COALESCE(c.orders, 0)  AS orders,
       ROUND(100.0 * COALESCE(c.revenue, 0)
             / NULLIF(SUM(c.revenue) OVER (PARTITION BY m.month), 0), 2)
           AS pct_of_month
FROM (SELECT DISTINCT region FROM customers) r
CROSS JOIN months m
LEFT JOIN completed c ON c.region = r.region AND c.month = m.month
ORDER BY m.month, r.region;
```

**Key points**

1. **Join order matters for the aggregate**: joining `order_items` multiplies `orders` rows, so `COUNT(DISTINCT o.id)` — not `COUNT(*)` — is the only correct denominator.
2. **`LEFT JOIN` off a `CROSS JOIN` skeleton** is how you keep zero-revenue regions; a plain inner join silently deletes them and the chart lies.
3. **Date filtering on the fact table, not the dimension**, and a half-open range (`>= Jan 1`, `< Jan 1 next year`) so the last day is included regardless of time component.
4. **`NULLIF(...,0)`** guards divide-by-zero; `SUM() OVER (PARTITION BY)` computes share without a second pass.
5. Say aloud what you assumed: grain of `order_items.qty`, whether `unit_price` is pre-discount, and whether refunds are a separate status. In a Palantir interview the *questions you ask about the data model* are graded as much as the SQL.

**中文要点**：`order_items` 的 join 会放大行数，所以要用 `COUNT(DISTINCT o.id)`；保住零收入区域要靠 `CROSS JOIN` 骨架 + `LEFT JOIN`；时间用半开区间，占比用窗口函数，并主动确认金额口径与退款状态。

---

### 3. Build a function that fetches paginated data from a REST API, handling page size and total-page logic.

**Reference implementation**

```python
import json, time
import requests

def fetch_all(url, params=None, page_size=100, max_pages=None, session=None):
    s = session or requests.Session()
    base = dict(params or {})
    out, seen = [], set()
    page = 1
    while True:
        for attempt in range(5):
            r = s.get(url, params={**base, "page": page, "page_size": page_size},
                      timeout=(5, 30))
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
                continue
            r.raise_for_status()
            break
        else:
            raise RuntimeError(f"giving up on page {page}")

        body = r.json()
        items = body.get("data") or body.get("items") or []
        if not items:
            break

        for it in items:
            key = it.get("id")
            if key is None:                       # no id -> content hash
                key = json.dumps(it, sort_keys=True)
            if key in seen:
                continue                          # API overlap / retry duplicate
            seen.add(key)
            out.append(it)

        total_pages = body.get("total_pages") or body.get("pages")
        if total_pages is not None:
            if page >= int(total_pages):
                break
        elif len(items) < page_size:              # last page by short read
            break
        if max_pages and page >= max_pages:
            break
        page += 1

    return out
```

**Key points**

1. **Two termination signals, not one.** Prefer the server's `total_pages`; fall back to `len(items) < page_size`. Never rely only on "empty page" — many APIs return the last page again instead.
2. **Dedupe by id**, because pagination over a mutating table shifts rows between pages and you will see duplicates (and miss rows — note that to the interviewer).
3. **Retry only idempotent GETs**, with `Retry-After` support and exponential backoff; 4xx other than 429 is a bug in your request, not a transient failure.
4. **Bound the loop** (`max_pages`) so a broken `total_pages` cannot spin forever.
5. **Cursor pagination is strictly better** when offered (`next_cursor`) — no offset drift, no deep-page cost. Say this and offer the variant.
6. For 10k+ pages: stream to disk, not memory, and add a checkpoint so a crash resumes rather than restarts.

**中文要点**：终止条件要两条（`total_pages` 优先，短页兜底），按 id 去重（翻页期间数据变动会重复/漏行），只对 GET 做带 `Retry-After` 的重试，并给循环封顶；有 cursor 就用 cursor。

---

### 4. Find and fix a double-counting bug in a function that tallies values in a HashMap.

**Answer**

```python
# BUGGY
def tally(events):
    counts = {}
    for e in events:
        sku = e["sku"]
        counts[sku] = counts.get(sku, 0) + 1
        if sku in counts:                 # always true -> second increment
            counts[sku] += 1
    return counts
```

Two increments per event. Symptom: every count is even, and `sum(counts.values()) == 2 * len(events)` — which is the invariant that catches it.

```python
from collections import Counter

def tally(events):
    # de-duplicate by event id first: retries make the same event appear twice
    seen, unique = set(), []
    for e in events:
        eid = e.get("event_id")
        if eid is not None:
            if eid in seen:
                continue
            seen.add(eid)
        unique.append(e)
    return Counter(e["sku"] for e in unique)
```

**Key points**

1. **Idempotency at the source, not in the counter.** The visible bug is the double `+= 1`; the *latent* bug is that the input contains duplicated events (retries, at-least-once delivery, join fan-out). Fixing only the visible one leaves a system that double-counts whenever upstream retries.
2. **Assert invariants**: `sum(counts.values()) == len(unique_events)` and (if quantities exist) `sum(counts.values()) == sum(qty)`. Unit tests on the total catch this class instantly; tests on individual keys may not.
3. **`Counter` over `dict.get`** removes the hand-rolled branch where the bug lived.
4. **Check the join too**: if `events` came from a SQL join, the duplicate rows are the real cause — dedupe by event id, don't dedupe by `sku` (that would hide genuine repeats).
5. When interviewing: state the reproduction (`len == 2×`), the mechanism, the fix, then the *class* of bug (missing idempotency key) and the test that would have caught it.

**中文要点**：表面是 `+= 1` 写了两遍，真因是上游重试/join 扇出导致事件重复——所以要按 `event_id` 去重再计数，并用"计数总和 == 去重后事件数"这一不变量做测试。

---

### 5. Debug a program that models infection spread across a social graph.

**Answer**

Typical buggy implementation:

```python
def spread_buggy(graph, origin, days):
    infected, frontier = {origin}, [origin]
    for _ in range(days):
        new = []
        for node in frontier:
            for nb in graph.get(node, ()):
                if nb not in infected:      # `infected` mutated below!
                    infected.add(nb)        # infects mid-iteration
                    new.append(nb)
        frontier = new
    return infected
```

The bug: `infected.add(nb)` happens **inside** the same-day loop, so a node reached at day *d* through one neighbour can immediately infect its own neighbours at day *d* as well — multi-hop within a single day. With a path `A→B→C`, C is infected on day 1 instead of day 2. Fix: collect the whole day's new infections first, then commit.

```python
def spread(graph, origin, days):
    infected = {origin}
    frontier = [origin]
    per_day = [{origin}]
    for _ in range(days):
        nxt = {nb for node in frontier for nb in graph.get(node, ())} - infected
        infected |= nxt                  # commit after the full sweep
        per_day.append(nxt)
        frontier = list(nxt)
        if not nxt:
            break                        # epidemic died out
    return infected, per_day
```

**Key points**

1. **Two-phase sweep** (compute then commit) is the whole fix; the classic phrasing is "mutating the visited set during BFS gives you DFS-like timing".
2. **BFS, not DFS** — DFS gives the right reachable set and the wrong day counts. If the question asks "on which day", the algorithm is BFS layers.
3. **Boundary bugs to check**: off-by-one on `days` (day 0 = patient zero?), self-loops, directed vs undirected edges, and parallel edges.
4. **Invariants**: `per_day[i]` are disjoint, their union is `infected`, and `len(infected)` is non-decreasing.
5. **Test the shape, not just the count**: `A-B-C` with one day must yield `{A,B}`, not `{A,B,C}`. That single test fails on the buggy version.

**中文要点**：bug 是"同一天内就沿新感染节点继续扩散"（在遍历中改 `infected`），要改成先收集当天新感染、再统一提交；并且必须是 BFS 分层，DFS 会给出正确的可达集但错误的天数。

---

### 6. You inherit an 800-line pipeline script from a previous deployment. It's slow and occasionally produces wrong numbers. The original author is gone. Go.

**Answer**

Correctness first, then speed — but *characterise before you touch anything*.

| Step | Action | Output |
| :--- | :--- | :--- |
| 1. Freeze | copy the script, pin dependencies, commit. Snapshot its current output for the last N runs | a baseline you can diff against |
| 2. Establish truth | find one period where a human already validated the numbers (finance close, prior report). If none exists, rebuild a 100-row slice by hand | a golden reference |
| 3. Map the script | list every input, output, side effect, and the order of stages — not line by line, stage by stage | a one-page data-flow diagram |
| 4. Find the wrongness | diff against the golden reference per stage, not at the end: run the pipeline with intermediate outputs dumped, bisect where the numbers diverge | the exact stage and row class |
| 5. Fix correctness | write a regression test for the failing case **first**, then fix; make failures loud (row-count assertions, null checks, dup checks) | test suite |
| 6. Then profile | `cProfile` / query plans / `EXPLAIN ANALYZE`. Usually: N+1 queries, row-by-row Python over a dataframe, recomputing a join per date | the hot 3 lines |
| 7. Refactor in seams | extract stages behind interfaces, add idempotency and a run manifest (inputs hash → output hash) | maintainable pipeline |
| 8. Reconcile | run old and new in parallel for 2 cycles, reconcile daily totals to the cent | sign-off |

**Key points**

1. **"Occasionally wrong" almost always means a data-shape assumption**: duplicate keys, nulls in a join, timezone boundaries, or a silently truncated string. Instrument null/dup/row-count checks at every stage boundary; they localise the failure in minutes instead of weeks.
2. **Never optimise before you have a correctness oracle** — a fast wrong number is worse than a slow right one.
3. **Diff at stage boundaries**, not at the output: the output diff tells you *that* it is wrong, the stage diff tells you *where*.
4. **Parallel-run reconciliation** is the only acceptable cutover for a billing/ops pipeline. Ship with a kill switch.
5. Write down what you learned in the repo — the next person inherits your map, not the original maze.

**中文要点**：先冻结并抓基线，用一个"人工已核对的期间"做金标准，**按阶段 diff** 定位（不是只看最终输出），先补回归测试再修；性能最后做，且新旧并行跑两个周期对账后才切换。

---

### 7. Given exports from three customer systems, each with its own customer records, write code to produce one deduplicated set of entities and explain your design.

**Reference implementation**

```python
import re, unicodedata
from collections import defaultdict

SUFFIXES = {"inc", "llc", "ltd", "corp", "co", "gmbh", "sa", "plc"}

def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]+", " ", s.casefold())
    toks = [t for t in s.split() if t not in SUFFIXES]
    return " ".join(toks)

def norm_email(e: str) -> str:
    return (e or "").strip().casefold()

class DSU:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[rb] = ra

def dedupe(records):
    # records: list of dicts with source, name, email, phone, address, id
    keys = [{"name": norm_name(r.get("name", "")),
             "email": norm_email(r.get("email", "")),
             "phone": re.sub(r"\D", "", r.get("phone") or "")[-10:]}
            for r in records]

    dsu = DSU(len(records))
    exact = defaultdict(list)
    for i, k in enumerate(keys):
        for field, val in k.items():
            if len(val) >= 4:
                exact[(field, val)].append(i)
    for idxs in exact.values():
        for j in idxs[1:]:
            dsu.union(idxs[0], j)

    clusters = defaultdict(list)
    for i in range(len(records)):
        clusters[dsu.find(i)].append(i)

    # survivorship: per field, trust order by source + completeness
    TRUST = {"crm": 0, "erp": 1, "legacy": 2}
    out = []
    for members in clusters.values():
        members.sort(key=lambda i: (TRUST.get(records[i]["source"], 9),
                                     -sum(1 for v in records[i].values() if v)))
        base = dict(records[members[0]])
        base["member_ids"] = [records[i]["id"] for i in members]
        base["sources"] = sorted({records[i]["source"] for i in members})
        out.append(base)
    return out
```

**Key points**

1. **Blocking before comparison** (`(field, normalized_value)`) — 10⁵ records means 10¹⁰ pairs without it.
2. **Normalise aggressively, compare conservatively**: unicode NFKD, casefold, legal suffixes, phone digits-only last-10, email domain lowercased. `"Acme Inc."` and `"ACME, Inc"` must collide.
3. **Union-find** so identity is transitive; a pairwise "best match" merge is order-dependent and produces duplicates.
4. **Never auto-merge on name alone.** Name+address or email or phone; a single-field match is a *review candidate*, not a merge. Two different "John Smith" in one city is the standard disaster.
5. **Survivorship rules must be explicit and per-field** (CRM owns email, ERP owns billing address, most-recent wins for phone), plus `member_ids`/`sources` for audit — Palantir's whole thesis is that the lineage is part of the answer.
6. **Human review queue** for scores in the ambiguous band, and make merges reversible (store the cluster history, never hard-delete the loser).

**中文要点**：先做**分块**再比较，用 NFKD/大小写/法律后缀/电话后 10 位做规范化，用并查集做传递合并；绝不能仅凭姓名自动合并，按字段定权威源，并保留 member_ids/来源以便审计与回滚。

---

### 8. Users ask “how many open orders are blocked on a supplier issue?” Plain RAG gets this wrong. Why, and what's the right architecture?

**Anchor answer**

Because the question is an **aggregation over a live relational fact base**, and vector RAG retrieves *documents*, not *rows*.

| Why plain RAG fails | Mechanism |
| :--- | :--- |
| No counting | top-k chunks contain some orders, never the set; "12 chunks" ≠ "the count" |
| No set semantics | "open" ∧ "blocked" ∧ "supplier issue" must intersect exactly; similarity has no AND |
| Staleness | embeddings are a snapshot; an order released this morning is still "open" in the index |
| No joins | supplier issue lives in a different table than order status |
| Silent partial answers | the LLM writes a number from three example rows — plausible, wrong, unmarked |

Right architecture: **route by question type, then answer over structured data.**

1. **Intent classification** — `count/aggregate` vs `lookup` vs `policy/definition`.
2. **Text-to-query over the semantic layer** — generate SQL against a curated ontology/view (`orders`, `supplier_issues`, `order_blockers`) with documented joins and enum values, not raw tables.
3. **Validate before executing**: parse the SQL, check tables/columns against the catalog, check that filters use enum members, add mandatory freshness predicates, cap scanned rows.
4. **Execute read-only** with a statement timeout and row limit; for counts, return the number *and* the query.
5. **Compose the answer from the result set** — the number comes from the database; the prose only explains it.
6. **Use RAG for the glossary**, not the count: what "blocked" means in policy, which supplier codes are in scope.

Guardrails: never let the model narrate a figure that isn't in the result payload; on ambiguity ("blocked" = credit hold? quality hold?) ask a clarifying question instead of guessing; log the generated query so analysts can audit and so failures become few-shot examples.

**中文要点**：这类问题是**对活数据的聚合**，向量检索拿不到集合语义与计数。正确架构是"意图分类 → 在语义层上生成并校验 SQL → 只读执行 → 用结果集组织答案"，RAG 只用来解释术语。

---

### 9. What is an ontology in the Palantir sense, and why put LLM agents on top of one instead of on raw tables and documents?

**Anchor answer**

In the Palantir sense an ontology is not a taxonomy document — it is an **operational model of the enterprise**: typed objects (Customer, Order, Asset, Aircraft), their properties, their links (an Order *belongs to* a Customer, *fulfilled by* a Shipment), and the **actions** that may be applied to them (reassign, flag, approve), all bound to the underlying data sources with permissions and lineage.

Why agents belong on it:

| Raw tables + docs | Ontology |
| :--- | :--- |
| Schema is technical (`ord_hdr`, `cust_mstr`) | Vocabulary is the business's own ("open order", "supplier issue") |
| Joins must be rediscovered per query | Links are declared once, correct by construction |
| Permissions are per-table, hard to compose | Object/row/column-level policy travels with the object |
| "Do something" is a SQL write nobody can audit | Actions are typed, validated, permissioned, auditable operations |
| The LLM guesses meaning from column names | The model reads a curated, documented model |

The deeper argument is **the ontology is the contract for agency**: if an agent can only act through declared actions with preconditions and side effects, you get validation, HITL gating, audit, and simulation ("dry-run this action") for free. On raw tables, an agent that can write is an agent that can corrupt.

Costs and failure modes to acknowledge: an ontology is only as good as its curation — a stale link definition silently poisons every answer; modelling is slow and requires domain experts; and an ontology can *constrain* novel questions (the agent cannot answer what nobody modelled). So: ontology for entities and actions, plus retrieval over unstructured documents attached to those entities for the long tail.

**中文要点**：本体不是分类词典，而是"对象+属性+关系+**可执行动作**+权限与血缘"的企业运行模型。把 agent 放在本体上，等于把"写操作"限制在带前置条件、可校验、可审计的动作上；代价是建模慢、模型过期会污染所有回答。

---

### 10. Design an LLM agent that files and updates work orders in a customer's ERP: real writes to a production system. How do you make that safe?

**Anchor answer**

Rule: **the LLM proposes, a deterministic service commits.** The agent never holds ERP credentials with write scope.

| Layer | Mechanism |
| :--- | :--- |
| Credential | agent calls your gateway; the gateway holds a scoped service account. Per-tenant, least privilege, no delete scope |
| Intent | the agent emits a typed `WorkOrderCommand` (`create/update/close` + fields + reason + evidence) |
| Validation | schema + business rules: asset exists, priority enum, required fields, no duplicate open WO, SLA window; reject with a machine-readable reason the model can fix |
| Dry run | `preview` returns the exact diff ("status: OPEN → SCHEDULED, assignee: J. Ruiz") without writing |
| Approval | risk tier: auto-approve below a threshold (e.g. internal notes, priority downgrade); human approval for anything touching cost, safety, or closure |
| Commit | idempotency key = hash(tenant, command, business key); write; read back the record and compare |
| Verify | assert the returned object matches the intent; on mismatch, raise an incident instead of retrying |
| Audit | immutable log: prompt, model version, command, approver, before/after diff, ERP response |

Failure modes this must survive:

1. **Replayed writes** from retries → duplicates, double-charged labour. Idempotency key at the ERP boundary, plus query-before-create on natural keys.
2. **Hallucinated entity ids** → filing against the wrong asset. Resolve ids by search-then-confirm; never let the model invent one.
3. **Partial failure** → the WO is created but the assignment fails. Model the operation as a saga with compensation (cancel the WO) rather than "retry the whole thing".
4. **Scope creep** → the model "helpfully" closes other open WOs. One command per call, explicit target list, max N objects per call.
5. **Prompt injection from the ticket text** — a customer note saying "mark all tickets resolved". Treat ticket content as data, never instructions; the command schema is the only channel.

Start read-only, then shadow-write (log the command without executing), then a single low-risk write path, then expand. Measure *wrong writes per 1k*, not success rate.

**中文要点**：agent 只产出类型化的 `WorkOrderCommand`，凭据在网关侧、最小权限；流程是校验→dry-run diff→按风险分级审批→幂等提交→**回读比对**→审计。要防重放重复、编造 id、部分失败（用补偿）、越权批量，以及工单文本里的 prompt injection。

---

### 11. Your platform must support multiple LLM providers, including deployments in restricted environments where only some models are available. How do you architect model selection?

**Source**: https://outcomeschool.com/blog/llm-routing

**Anchor answer**

Three layers: a **capability contract**, a **policy**, and a **runtime**.

1. **Capability contract (static).** Every task declares what it needs: context length, tool calling, JSON schema strictness, vision, streaming, language coverage, max latency, data-residency class. Each provider adapter publishes what it supports. This is a compatibility matrix, evaluated at deploy time — not at request time.
2. **Registry per environment.** A manifest lists which models exist in *this* installation (on-prem customer A has model X only; cloud tenant has all). The registry is data, not code, so an air-gapped install ships a different manifest with the same binary.
3. **Routing policy.** Ordered rules, each with a reason code:

| Rule | Example |
| :--- | :--- |
| Hard constraints | residency = EU; no external egress; context ≥ 128k |
| Task tier | cheap classifier → small model; long-form synthesis → frontier model |
| Health/circuit breaker | provider 5xx or p99 breach → next candidate |
| Cost/latency budget | per-tenant monthly budget, per-request deadline |
| Fallback chain | ordered list, with a **degraded-mode definition** per task |

4. **Runtime.** Adapter layer normalises requests/responses (message format, tool schema, stop reasons, token accounting). Failover happens *before* the first token, never mid-stream; a stream that dies yields a truncated answer plus an explicit error, not a silent swap.

Non-obvious requirements:

- **Prompt portability**: prompts tuned for one model degrade 10-20 points on another. Keep per-provider prompt variants and evaluate each on the same golden set; do not assume a generic prompt works.
- **Determinism of routing**: log the routing decision with every request (constraints evaluated, model chosen, fallback used) — otherwise you cannot reproduce an incident.
- **Per-environment capability tests** at startup: a smoke test per model, so a missing deployment fails at boot, not at 3am.

**中文要点**：分三层——能力契约（上下文/工具/JSON/流式/驻留地）、按环境分发的模型清单（数据而非代码，适配离线部署）、规则化路由（硬约束→任务档位→熔断→预算→回退链）；适配层做协议归一，且每个 provider 需要各自的 prompt 变体与同一套评测。

---

### 12. How do you evaluate an LLM workflow before and after giving it access to production operations?

**Anchor answer**

The access change is the intervention, so the evaluation must be **paired**: same inputs, two configurations, with production access off and on.

Before access (offline):

| Axis | Method |
| :--- | :--- |
| Task correctness | golden set of real historical cases with known-good outcomes (200-1000) |
| Tool-choice correctness | did it pick the right action, with the right arguments, on the right object? argued separately from outcome |
| Refusal/abstain | cases where it *should* ask or refuse; false-confidence rate |
| Grounding | every claim traceable to a retrieved record; unsupported-claim rate |
| Robustness | adversarial inputs: ambiguous requests, injection in ticket text, missing fields |
| Cost/latency | tokens, tool calls, p95 end-to-end |

After access (online) — the interesting part, because offline scores do not predict production:

1. **Shadow mode first**: the workflow runs against the real API in read-only/dry-run, proposing actions; humans execute. Compare proposed vs human action → an *action-level* agreement rate on live traffic.
2. **Canary with bounded blast radius**: 1-5% of traffic, one tenant, actions capped (max N per day, max value per action), full audit, instant kill switch.
3. **Counterfactual review**: sample accepted actions and ask "would a competent human have done this?" — measures silent failure that no error metric catches.
4. **Outcome metrics**: actions reverted, escalations, time-to-resolution, cost per case, incident count. Adoption rate is not quality.
5. **Distribution monitoring**: input drift, tool-error rates, refusal rate, latency — a sudden refusal spike usually means an upstream schema change.

The thing that bites: **the eval set was built from pre-access traffic**, where the workflow could not act. Granting write access creates a feedback loop — the system's own actions change the state that future inputs describe. So keep a frozen replay set of environments (snapshots) for regression, and evaluate *decisions* on identical world states.

**中文要点**：把"开放写权限"当成实验处理——同一批历史用例做配对评测；上线先 shadow（只提议不执行）比"提议 vs 人工动作"一致率，再小流量灰度+动作上限+一键关停；还要监控动作被撤回率与输入漂移，并用快照世界状态做回归。

---

### 13. A freight rail operator loses tens of millions a year to unplanned locomotive downtime. Decompose this into an engineering plan.

**Anchor answer**

Decompose along the money, not the technology.

| Layer | Question | Artifact |
| :--- | :--- | :--- |
| Value model | where does the loss come from? | per-failure-class cost = lost freight revenue + recovery + penalty + knock-on delay |
| Pareto | which 3 failure classes are 60%+? | ranked table from maintenance + dispatch data |
| Data audit | do we have sensor telemetry, work orders, parts, location, weather? | joinable keys per asset per time |
| Feasibility | is the failure detectable *early enough to act*? | lead-time analysis vs maintenance scheduling window |
| Intervention | what does the operator do with a 72h warning? | a work order, an inspection, a speed restriction, a swap |
| Economics | cost of false alarms vs saved events | threshold chosen by expected value |

The engineering plan:

1. **Ingest and unify**: telemetry historian (per-loco sensors, 1-60s), maintenance records (free-text!), parts/inventory, dispatch schedule, location. Deterministic asset identity across systems is the first real work.
2. **Label the past**: every unplanned failure becomes an event with a window; match to telemetry. Expect 10-30% of work orders to be unlinkable — quantify that before modelling.
3. **Baseline first**: a rule-based threshold system (oil pressure trend, coolant delta, vibration RMS) that the operator already trusts. A model that cannot beat the rule is not deployable.
4. **Model**: survival/remaining-useful-life per failure class; start with gradient boosting on engineered features, not a transformer on raw waveforms — the label volume rarely justifies it.
5. **Decision layer**: the output is a ranked inspection list per depot per day, constrained by crew and parts availability. Optimisation, not prediction, is where the savings land.
6. **Deploy**: shadow → recommend → auto-create work orders for the top class only.

Metrics: **downtime hours avoided per month** (not AUC), false-alarm rate per loco-week (the operator's real tolerance), lead time distribution, and alerts that were never actioned (a signal your recommendation ignores logistics).

**中文要点**：按钱拆——先算每类故障的综合损失、找 60% 的前三类、确认数据可 join、并用"提前量够不够排检修"筛可行性；建模用生存/RUL + 梯度提升，但价值落在**排班与备件约束下的检查清单排序**，指标是每月避免的停运小时与误报率。

---

### 14. Design a system to improve traffic in NYC.

**Answer**

"Improve traffic" is not a requirement. Decompose into decisions someone actually owns:

| Owner | Decision | Data | Lever |
| :--- | :--- | :--- | :--- |
| DOT signal ops | timing plans per corridor per time-of-day | loop radar, cameras, probe speeds | green splits, offsets, transit signal priority |
| Curb management | loading zones, bus lanes, double-park enforcement | curb occupancy, citations, deliveries | zone redesign, dynamic pricing |
| Incident response | dispatch to a blockage | 911/511 feeds, probe speed anomalies | faster clearance |
| Traveller | route choice | live conditions | information, not control |

Pick one: **corridor signal optimisation with incident-aware re-timing.** Architecture:

1. **Ingest**: probe vehicle speeds (taxi/FHV, ~1-5 min lag), loop detectors, bus AVL, incident feeds, construction permits, weather.
2. **State estimation**: per-link speed + queue length, fused; explicitly handle the sparse-sensor problem (many intersections have no detector).
3. **Prediction**: short-horizon (15-60 min) demand/flow, plus incident impact.
4. **Control**: optimisation over signal plans with transit priority and pedestrian constraints; simulated before dispatch, human-approved.
5. **Evaluation**: A/B by corridor-week with matched control corridors (traffic is not stationary; naive before/after is confounded by weather, school terms, incidents).

Reality checks: NYCDOT already runs adaptive signals (Midtown in Motion) — come with awareness of that, not a naive pitch. Pedestrian safety is a hard constraint, not a term in the objective; a plan that speeds cars by shortening walk time will be (rightly) rejected. And induced demand means corridor-level gains can be zero system-wide — measure at the network level or do not claim it.

**中文要点**：先问"谁做哪个决定"（信号配时/路侧管理/事故处置/出行信息），选一条廊道的信号优化做：探针+线圈+公交 AVL 融合估状态→短时预测→带公交优先与行人约束的优化→走廊-周 A/B 对照；注意 NYC 已有自适应信号，且缩短行人过街时间来提速不可接受。

---

### 15. Design a sync system between two employee record systems.

**Answer**

Assume HRIS (system of record for identity) ⇄ directory/badge system. The hard parts are **ownership, deletions, and conflicts**.

| Decision | Design |
| :--- | :--- |
| Authority | field-level, not system-level: HRIS owns name/title/department/manager/status; the directory owns badge id, physical access groups |
| Identity | a stable internal `person_id` mapped to each system's id; never join on email (it changes on marriage/renaming) |
| Trigger | change-data-capture or webhook from the authoritative side; polling as a safety net with a cursor |
| Transport | idempotent upsert with a per-record version/`updated_at`; batch + per-record ack |
| Conflict | last-writer-wins only for fields with one owner; otherwise reject and route to a review queue |
| Deletion | **soft delete / deactivate**, never hard delete; termination is a workflow, not a row removal |
| Ordering | per-person sequential (keyed queue), global order not required |
| Failure | retry with backoff, dead-letter queue, and a reconciliation job that diffs both sides nightly |

The failure modes that actually happen:

1. **Rehire with the same employee id** — the person returns; a naive "deleted means gone forever" tombstone blocks them. Keep a generation/revision counter.
2. **Two systems both "fixing" each other** — an update loop. Mark writes with a source tag and ignore echoes.
3. **Termination lag** — access must be revoked *before* HR finishes paperwork; drive deprovisioning from the HR event, not a nightly batch.
4. **Manager chains** — a manager change cascades to approvals; sync order matters (update the person before the reports).
5. **Clock skew** — `updated_at` comparisons across systems with unsynced clocks cause lost writes. Use a server-assigned monotonic version.

Testing: a synthetic conflict suite (concurrent edit, rehire, rename, missing field), plus a nightly reconciliation report that must be empty. Measure sync lag p95 and the number of manual interventions per week.

**中文要点**：字段级定权威源、用内部 `person_id` 而不是邮箱关联、CDC/webhook 为主+轮询兜底、幂等 upsert 带版本号、软删除而非硬删、失败走死信+每日对账；坑是再入职、互相回写死循环、离职权限回收滞后与时钟偏移。

---

### 16. Design a system that lets multiple teams query a shared dataset without exposing raw data.

**Anchor answer**

"Without exposing raw data" is ambiguous — clarify *to whom* and *what threat*. Usually: teams may compute on the data but must not see row-level PII, must not exfiltrate the corpus, and the owner must be able to prove who did what.

| Layer | Mechanism |
| :--- | :--- |
| Access | SSO + group-based roles; purpose-bound access requests with expiry |
| Policy | row filters (tenant/region), column masks (PII hashing/tokenisation), dynamic views per role |
| Query surface | SQL over **views/row-access policies**, not the base table; no `SELECT *` export path |
| Aggregation guard | minimum cell size (k-anonymity-ish), query budget, block small-group filters that reconstruct individuals |
| Egress | result-size caps, no bulk download, watermarked exports, audit of every result set |
| Audit | query text, user, purpose, rows scanned, result hash — immutable |
| Governance | data contracts per dataset, owner-approved schema changes, retention |

Two implementation routes, pick by threat model:

1. **In-warehouse policy** (row-level security + masking + views + query logging). Cheapest, best performance, but relies on the warehouse's enforcement and full trust in admins.
2. **Clean-room / enclave** (approved queries run inside a controlled compute, only aggregates leave). Stronger, slower, needs a query review or a differential-privacy layer.

Non-obvious failure modes:

- **Differencing attacks**: two allowed queries whose difference isolates one person. Need query history analysis and overlapping-subset restrictions, not just per-query checks.
- **Join re-identification**: the shared dataset is "anonymous", but joining it with a public dataset (voter rolls, LinkedIn) re-identifies people. Column-level k-anonymity does not prevent this; generalisation/DP does.
- **Masking that breaks analytics**: hashing an id destroys joins teams legitimately need — tokenise deterministically per consumer instead.
- **The export that nobody watches**: the audit log exists but has no alerting; a 500k-row export at 2am is the actual breach.

**中文要点**：先澄清"不暴露"是防谁——按威胁模型选"仓内行级权限+列掩码+视图+查询审计"或"数据洁净室只出聚合"；要防差分攻击与外部数据 join 重识别，标识符要用确定性 tokenization 而非哈希，并对大结果集导出做告警。

---

### 17. Design an application to catalog and log species while exploring an unfamiliar environment.

**Answer**

Constraints first: **offline, low-power, uncertain labels, and the point is the log, not the model.**

| Layer | Design |
| :--- | :--- |
| Capture | camera + GPS + timestamp + audio; every observation gets an immutable `obs_id` and geohash |
| On-device inference | small classifier for coarse taxon (bird/mammal/plant/insect) + "unknown"; runs at 1-5 fps on a sample of frames, not all |
| Candidate generation | top-k species with calibrated confidence; **k is capped** and low-confidence collapses to "unknown" |
| Logging | append-only local store (SQLite/JSONL), synced opportunistically; never blocks on network |
| Human loop | one-tap accept/correct; corrections become labels with provenance |
| Sync | conflict-free: observations are immutable, corrections are new records referencing `obs_id` |
| Review | taxonomist review queue for uncertain/high-value sightings; range maps and seasonality as priors |

Design choices that matter:

1. **"Unknown" must be a first-class output.** In an unfamiliar environment, most observations are out-of-distribution; a model forced to pick from 10k classes will confidently label a novel beetle as a common one. Report `unknown` and rank candidates.
2. **Priors are cheap accuracy**: geographic range + season + time of day prune the candidate set by orders of magnitude before the model runs. This is the highest-leverage component and it needs no training.
3. **Log evidence, not just labels**: store the crop, the full frame, GPS accuracy, and the model version. A species record without evidence is unverifiable, and a wrong record can contaminate a biodiversity dataset for years.
4. **Battery/thermal**: continuous camera + inference kills a phone in an hour. Duty-cycle (motion/time triggers), quantise the model, and make capture survivable without inference.
5. **Invasive-species alerts**: if a high-confidence detection of a listed invasive species fires, that changes the product from cataloguing to alerting — decide up front whether that is in scope, because it implies a verification SLA.

Evaluation: top-1/top-3 accuracy **split by "seen species" vs "novel species"**, calibration (does 0.8 mean 80%?), and the human-correction rate per 100 observations. Offline field test with no network is the only real acceptance test.

**中文要点**：离线优先、观测不可变+可追溯证据（裁剪图/原帧/GPS 精度/模型版本）；"unknown" 必须是一等输出；先靠地理分布+季节先验剪枝再跑模型；用功耗占空比保续航；评测要分"已知种/新种"并看校准与人工纠正率。

---

### 18. A customer executive says “the AI keeps getting things wrong” and wants to cancel the pilot. Walk me through your next 48 hours.

**Anchor answer**

Do not defend the model. Convert a vague complaint into a **ranked defect list with names attached**.

**Hour 0-4 — contain and listen.**
Get the executive to name three concrete instances. Ask: wrong *what* — the answer, the source, the action, the format? Who noticed? What did they do instead? Meanwhile pull the full traces for those three (prompt, retrieved context, tool calls, output, user feedback). Never argue from aggregate metrics in this conversation; argue from their examples. Offer a written 48-hour plan before you leave the room, and a named single point of contact.

**Hour 4-24 — classify, don't fix yet.**
Bucket every complaint from the last 30 days:

| Bucket | Typical share | Fix owner |
| :--- | :--- | :--- |
| Expectation mismatch (asked for something out of scope) | often 30-40% | you + product: scope the pilot in writing |
| Data/retrieval gap (the answer exists but was not retrieved) | 20-30% | engineering: chunking, index, freshness |
| Genuine model error | 15-25% | prompt/model/eval |
| Integration/UX (right answer, wrong place, wrong format) | 15% | UX + integration |
| User error / training gap | 10% | enablement |

**Hour 24-40 — fix the top two that are fixable in a day**, and *prove* it on their three examples plus 20 regression cases. Ship behind a flag; do not ship a broad change under time pressure.

**Hour 40-48 — reset the contract.** Present: what was wrong, what is fixed, what is out of scope, what the accuracy bar is and how it will be measured weekly, and what you need from them (feedback channel, a named super-user). Offer a 30-day checkpoint with pre-agreed metrics rather than an open-ended pilot.

The honest move that saves deals: if the complaint is scope mismatch, say so plainly and propose a smaller pilot that can succeed. Over-promising to save the quarter is how pilots die in month three.

**中文要点**：48 小时内不要辩模型——先拿到三个具体反例并拉全链路 trace，把 30 天投诉按"预期错位/检索缺口/模型错误/集成 UX/使用问题"分桶，优先修当天能修的两项并用回归证明；最后重写试点范围与周度验收指标，必要时主动缩小范围。

---

### 19. Why Palantir, and why this team? Tell me about a time you pushed back on a customer request.

**Anchor answer**

The first half needs a specific, checkable reason — not "I like hard problems".

A structure that lands: **the kind of problem → what you have done that proves it → what is different here → what you want to learn.**

- "I keep ending up on problems where the model is the easy part and the integration, permissions and trust are the hard part. At [X] I shipped [concrete thing] and the thing that made it work was [data model / deployment constraint / audit trail], not the model. Palantir is the place where that ratio is the job rather than an obstacle. This team specifically because [their public work / the deployment type / the domain] is where I want to go deeper."
- Name a **team-specific fact**: a product you have read about, a deployment pattern, an engineering blog post. Generic enthusiasm is the most common rejection reason at this stage.

The pushback story must have: the request, your objection, the evidence you brought, how you handled the relationship, and the outcome — including if you were overruled.

Example shape: "A customer asked us to auto-approve refunds under $500 to hit a cycle-time target. I modeled it: at their volume, expected fraud plus reversal cost exceeded the labour saved, and it removed the human check on the only category that had produced incidents. I brought the numbers and a compromise — auto-approve only for repeat customers with no dispute history, 90-day review. They took the compromise. I was overruled on the threshold, which turned out fine because we caught a bad actor in week two and tightened it."

What they are testing: do you push back **with evidence and without contempt**, do you distinguish "I disagree" from "I will block", and can you commit to a decision you lost.

**中文要点**：答"为什么 Palantir"要给可核查的具体理由（问题类型+你的证据+该团队的具体事实），别说"喜欢有挑战的问题"；反驳客户的故事要有请求、你的反对、你带的数据、关系处理与结果，包括被否决后如何执行。

---

### 20. Palantir works with defence and intelligence agencies. How do you think about that, and what would you do if asked to build something you're uncomfortable with?

**Anchor answer**

Answer directly and with a stated line — hedging reads worse than disagreement.

The reasoning to lay out:

1. **The work is real and consequential.** Palantir's customers make decisions about deployment, targeting, benefits, immigration, policing. The software shapes those decisions, so "I just write code" is not available as a position.
2. **There is a legitimate case**: democratic states need competent, auditable software; the alternative is worse tools with less oversight, and the audit trail Palantir sells is itself a governance mechanism. Many engineers find that persuasive.
3. **There is a legitimate limit**: some uses cross a line for an individual — autonomous targeting, mass surveillance of a domestic population, systems whose errors are unreviewable. Those lines differ per person, and pretending they do not exist is dishonest.

Then state **your** line concretely, at the level of *use*, not company: "I will work on logistics, fraud, maintenance, humanitarian response. I would not work on a system whose output directly selects a person for lethal action without human review, or on domestic mass collection. I would raise it, in writing, with my lead and with the ethics/compliance channel."

Process, if asked to do something uncomfortable:

1. **Ask questions first** — most "uncomfortable" asks are ambiguous; clarify the actual use, the customer, and the review chain before refusing.
2. **Escalate to a named person**, not a Slack thread: your manager, then the programme's compliance owner.
3. **Put the concern in writing with the specific risk**, and ask for the documented approval path.
4. **Accept being overruled on taste, not on legality or safety.** If it is the latter and the answer is "proceed anyway", decline and be prepared for the consequence.
5. Say what you would do if the answer is no: transfer, or leave. Interviewers respect a stated boundary far more than a claim that no boundary exists.

Also worth saying: security clearance and compartmented work impose real constraints (you cannot discuss the work, sometimes not even with your team) — ask about that honestly, because it affects daily life more than the ethics debate does.

**中文要点**：直说立场，别打太极——承认工作有后果、"只是写代码"不成立，也承认民主国家需要可审计的工具；把红线定在**用途**层面（可做后勤/反欺诈/人道，不做无人工复核的致命决策或境内大规模采集），并说明升级路径与"答案是否"时的选择。

---

## 附：这 20 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 编码基本功 | 1（形状类）、3（分页）、4（重复计数）、5（感染扩散） |
| 数据工程 | 2（SQL）、7（三源去重）、15（双系统同步） |
| 接手烂摊子 | 6（800 行管线） |
| 本体与 agent 架构 | 9（本体）、10（ERP 真写）、8（聚合类问题） |
| 平台与评测 | 11（多 provider 路由）、12（开权限前后评测）、16（共享数据不暴露原文） |
| FDE 场景拆解 | 13（机车停机）、14（NYC 交通）、17（物种记录）、18（客户要取消试点） |
| 文化与价值观 | 19（为什么 Palantir/反驳客户）、20（国防与红线） |
