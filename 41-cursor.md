# 41 · Cursor (Anysphere)

> 17 题。出处：`README.en.md` § Cursor (Anysphere)。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Build a hash tree to organise data in a repository.

**Anchor answer**

A repository is a recursive name→node mapping, so the natural structure is the one Git already uses: **content-addressed blobs, hashed directory nodes, one root hash**.

| Layer | Node | Identity |
| :--- | :--- | :--- |
| Blob | file bytes | `sha256(content)` |
| Tree | directory | `sha256(sorted(name + child_hash))` |
| Root | repo snapshot | the tree hash of `/` |

Three properties fall out for free, and they are the actual reason to build it:

1. **Dedupe** — identical files share a blob hash, so storage is per-content, not per-path.
2. **Cheap equality** — two snapshots are identical iff their root hashes match; `O(1)` for "did anything change".
3. **Prunable diff** — equal subtree hash means *nothing underneath changed*, so directory-level pruning turns a full-tree walk into a walk over changed paths only (next question).

The invariants you must hold: hash the **sorted** child list (filesystem order is not stable), include the **type bit** (file vs directory, else a file and an empty dir collide), and separate path from content (never hash an absolute path into the blob — it kills dedupe and breaks on rename).

**Reference implementation**

```python
import hashlib


def _h(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Node:
    __slots__ = ("hash", "children", "is_file")

    def __init__(self):
        self.children = {}
        self.hash = None
        self.is_file = False


def _seal(node):
    if node.is_file:
        return node.hash
    for name in sorted(node.children):
        _seal(node.children[name])
    payload = "".join(f"{n}:{node.children[n].hash};" for n in sorted(node.children))
    node.hash = _h(payload.encode())
    return node.hash


def build_tree(snapshot: dict[str, str]) -> Node:
    """snapshot: {path: content} -> sealed Merkle root."""
    root = Node()
    for path, content in snapshot.items():
        parts = [p for p in path.strip("/").split("/") if p]
        node = root
        for d in parts[:-1]:
            node = node.children.setdefault(d, Node())
        leaf = node.children.setdefault(parts[-1], Node())
        leaf.is_file, leaf.hash, leaf.children = True, _h(content.encode()), {}
    _seal(root)
    return root
```

**Key points**

- `build_tree` is `O(total bytes + n log n)`; the log factor is only the per-directory sort.
- Sealing bottom-up means one pass — no re-hashing of parent after every insert.
- A snapshot is the whole map; you never mutate a sealed node, you rebuild and compare roots.

**中文要点**：仓库就是"名字→节点"的递归映射，Git 的做法最省事——blob 只哈希内容、目录哈希排序后的子项、根哈希代表整个快照；换来秒级相等判断和可剪枝的 diff（下一题）。

---

### 2. Given a repository snapshot (path → content), build a Merkle tree and write the function returning which files changed between two snapshots without comparing every file's content.

**Anchor answer**

The trick is not the tree — it is **not descending into equal subtrees**. At each level you compare hashes; only when a directory hash differs do you open its children. A repo where 3 files changed in `src/` never touches `vendor/`, `docs/` or `tests/`.

The diff has to handle four cases per child, and interviewers almost always catch the two people forget:

| Case | Action |
| :--- | :--- |
| both files, equal hash | skip |
| both files, different hash | report (**modified**) |
| only one side | report every file under it (**added/deleted**) |
| file ↔ directory at same name | report old files + new files |

In the reference code below, `stats["pruned"]` counts equal-hash subtrees that were never entered — that counter is the cheapest way to prove the optimisation is real, and a brute-force `O(files)` comparison in the test asserts the pruned walk returns exactly the same set.

**Reference implementation**

```python
import hashlib


def _h(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Node:
    __slots__ = ("hash", "children", "is_file")

    def __init__(self):
        self.children = {}
        self.hash = None
        self.is_file = False


def _seal(node):
    if node.is_file:
        return node.hash
    for name in sorted(node.children):
        _seal(node.children[name])
    node.hash = _h("".join(f"{n}:{node.children[n].hash};"
                           for n in sorted(node.children)).encode())
    return node.hash


def build_tree(snapshot):
    root = Node()
    for path, content in snapshot.items():
        parts = [p for p in path.strip("/").split("/") if p]
        node = root
        for d in parts[:-1]:
            node = node.children.setdefault(d, Node())
        leaf = node.children.setdefault(parts[-1], Node())
        leaf.is_file, leaf.hash, leaf.children = True, _h(content.encode()), {}
    _seal(root)
    return root


def _collect(prefix, node, out):
    if node is None:
        return
    if node.is_file:
        out.append(prefix)
        return
    for name in sorted(node.children):
        _collect(f"{prefix}/{name}" if prefix else name, node.children[name], out)


def changed_files(old, new):
    out, stats = [], {"pruned": 0}

    def walk(prefix, a, b):
        if a is not None and b is not None and a.hash == b.hash:
            stats["pruned"] += 1          # identical subtree: never enter it
            return
        names = set(a.children) if a else set()
        if b:
            names |= set(b.children)
        for name in sorted(names):
            path = f"{prefix}/{name}" if prefix else name
            ca = a.children.get(name) if a else None
            cb = b.children.get(name) if b else None
            if ca is None:
                _collect(path, cb, out)
            elif cb is None:
                _collect(path, ca, out)
            elif ca.is_file and cb.is_file:
                if ca.hash != cb.hash:
                    out.append(path)
            elif ca.is_file != cb.is_file:
                _collect(path, ca, out)
                _collect(path, cb, out)
            else:
                walk(path, ca, cb)

    walk("", old, new)
    return out, stats


def brute(old, new):
    o = {p: _h(c.encode()) for p, c in old.items()}
    n = {p: _h(c.encode()) for p, c in new.items()}
    return sorted(p for p in set(o) | set(n) if o.get(p) != n.get(p))


if __name__ == "__main__":
    a = {"src/app.py": "print(1)", "src/lib/util.py": "x=1",
         "README.md": "hi", "docs/a.md": "a"}
    b = dict(a, **{"src/app.py": "print(2)", "docs/b.md": "b"})
    ta, tb = build_tree(a), build_tree(b)
    got, stats = changed_files(ta, tb)
    assert sorted(got) == brute(a, b) == ["docs/b.md", "src/app.py"], got
    assert ta.hash == build_tree(a).hash          # deterministic
    print(got, stats)                             # 1 pruned subtree
```

**Key points**

- Cost is `O(changed paths × depth)`, not `O(repository size)`.
- Same-content files at different paths share a blob hash — dedupe and rename detection come from the same mechanism.
- Rename looks like delete + add. Real Git adds similarity search on top; mention that as a separate step, don't pretend the tree gives it.
- Failure mode: hashing unsorted children or omitting the file/dir tag makes structurally different trees collide.

**中文要点**：核心是**哈希相等就不下沉**——只沿变化的目录往下走，复杂度从 O(仓库大小) 降到 O(变化路径×深度)；别忘了"文件↔目录同名"这一支，并用暴力对比做自检。

---

### 3. Print the top view of nodes in a binary tree.

**Anchor answer**

Top view = for each horizontal distance from the root, the **topmost** node. Two implementations, both `O(n)`:

| Approach | How | Note |
| :--- | :--- | :--- |
| BFS (recommended) | queue of `(node, column)`, record the **first** node seen per column | BFS visits in depth order, so first-write-wins is exactly "topmost" |
| DFS | recurse with `depth`, keep the node with the **smallest depth** per column | needs an explicit depth comparison, easier to get wrong |

The BFS version is preferred because the invariant is structural: level order *is* the definition of "topmost", so no comparison logic is needed.

Three details that decide whether it passes:

1. Record **before** pushing children — otherwise a sibling at the same column overwrites.
2. Columns can be negative; store in a dict keyed by column, then sort keys ascending. Do not assume `0..n`.
3. Tie-break: two nodes can share a column at different depths — deepest loses, and the same-depth case is decided by BFS order (left-to-right on that level), which is the conventional answer.

Output for the tree in the test is `[4, 2, 1, 3, 7]` — note that node `5` and node `6` are invisible: they sit behind ancestors at the same column.

**Reference implementation**

```python
from collections import deque
from dataclasses import dataclass


@dataclass
class Node:
    val: int
    left: "Node | None" = None
    right: "Node | None" = None


def top_view(root):
    if root is None:
        return []
    first = {}
    q = deque([(root, 0)])
    while q:
        node, col = q.popleft()
        if col not in first:          # first visit per column == topmost
            first[col] = node.val
        if node.left:
            q.append((node.left, col - 1))
        if node.right:
            q.append((node.right, col + 1))
    return [first[c] for c in sorted(first)]


if __name__ == "__main__":
    r = Node(1, Node(2, Node(4), Node(5)), Node(3, Node(6), Node(7)))
    assert top_view(r) == [4, 2, 1, 3, 7]
    assert top_view(Node(1)) == [1]
    assert top_view(None) == []
    print(top_view(r))
```

**Key points**

- `O(n)` time, `O(width)` space — the queue holds one level, the dict holds one entry per column.
- Sorting columns is `O(w log w)`, negligible next to the traversal.
- Edge cases the tests cover: empty tree, single node, all-left chain (answer is the chain itself), duplicate columns across levels.

**中文要点**：按列取"最高"节点；BFS + 每列首次写入即正确，DFS 则要显式比深度。列可为负，用 dict 存再排序输出；同列更深者丢弃。

---

### 4. Find duplicate files in a file system.

**Anchor answer**

Never hash everything: that turns one `read` of the whole tree into several. Use the classic **size → head-hash → full-hash** funnel, which eliminates almost all files by cheap metadata alone.

| Stage | Cost | Kills |
| :--- | :--- | :--- |
| Group by size (`os.stat`) | no read | every unique-size file |
| Hash first 4 KB | one small read | most same-size-but-different files |
| Full streaming `sha256` | read everything, once | the rest |

Then report groups with ≥2 members. Practical notes:

- **Skip symlinks and hardlinks.** A hardlink is the *same inode*, so it is a "duplicate" that costs nothing and is not what the user means. Compare `(st_dev, st_ino)` or use `os.path.islink`.
- **Stream in 1 MiB chunks** — a full-file `read()` on a 40 GB VM image is how you OOM the tool.
- **Size 0 is a trap**: all empty files are trivially "identical". Usually you either exclude them or report them as a separate class.
- Content hash, not path hash: after a rename the file is still the same blob.
- If you want to keep one copy, dedupe by content hash and prefer the shortest / most recently modified path — but say the choice out loud, it is a product decision.

Expected behaviour on the sample tree: three files with the same bytes in two different directories collapse into one group of three.

**Reference implementation**

```python
import hashlib
import os


def find_duplicates(root):
    by_size = {}
    for dirpath, _, names in os.walk(root):
        for name in names:
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                continue
            st = os.stat(path)
            if st.st_size == 0:
                continue
            by_size.setdefault(st.st_size, []).append(path)

    groups = []
    for size, paths in by_size.items():
        if len(paths) < 2:
            continue
        by_head = {}
        for path in paths:
            with open(path, "rb") as f:
                head = hashlib.sha256(f.read(4096)).hexdigest()
            by_head.setdefault((size, head), []).append(path)

        for candidates in by_head.values():
            if len(candidates) < 2:
                continue
            by_full = {}
            for path in candidates:
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                by_full.setdefault(h.hexdigest(), []).append(path)
            groups.extend(sorted(g) for g in by_full.values() if len(g) > 1)
    return groups


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        for rel, body in [("a/x", "same"), ("a/y", "same"),
                          ("b/z", "diff"), ("b/w", "same"), ("empty1", "")]:
            p = os.path.join(d, rel)
            os.makedirs(os.path.dirname(p) or d, exist_ok=True)
            with open(p, "w") as f:
                f.write(body)
        print(find_duplicates(d))   # one group of 3 files
```

**Key points**

- Read each file at most once, and only after size + head hash agree.
- `O(n log n)` dominated by directory traversal, not hashing.
- Follow-up they usually ask: *find duplicates across two directories* — same code, union the paths and keep the origin directory as metadata.

**中文要点**：先按大小分组、再比前 4KB 哈希、最后全文流式哈希——用元数据砍掉绝大多数文件；记得跳过 symlink/硬链接（同 inode 不算重复）和 0 字节文件。

---

### 5. Implement the core of an editor text buffer: efficient insert/delete at arbitrary positions and fast line lookup. What structure do you pick?

**Anchor answer**

I pick a **piece table**, with a lazily rebuilt newline index for line lookup. It is what VS Code actually uses, and it matches the access pattern of an AI editor: a fixed original document plus an append-only stream of new text.

| Structure | Insert/delete | Memory | Why not |
| :--- | :--- | :--- | :--- |
| `str` / `list[char]` | `O(n)` shift per edit | 1× | dies at 10 MB with many edits |
| Gap buffer | `O(1)` at cursor, `O(n)` elsewhere | 1× | great for typing, terrible for scattered AI edits |
| Rope / B-tree | `O(log n)` | high constant | best asymptotics, most code |
| **Piece table** | `O(p)` piece ops, no text copy | 1× + appended bytes only | many pieces after heavy editing |

Two buffers make the design work: **`orig`** (immutable, never rewritten) and **`add`** (append-only). An edit never copies text — it *splits a piece* and inserts a new one pointing into `add`. Undo is then trivial: restore the previous piece list.

Line lookup: keep the newline offsets of the current text. Rebuild is `O(n)` but only after an edit; every query afterwards is a `bisect` — `O(log n)`. For a real editor you would store newline counts **inside the tree nodes** so the index costs `O(log n)` to maintain too; I say that explicitly rather than pretending the lazy version scales.

**Reference implementation**

```python
import bisect


class TextBuffer:
    """Piece table: immutable original + append-only add buffer."""

    def __init__(self, text: str = ""):
        self._orig = text
        self._add: list[str] = []
        self._pieces: list[list[int]] = []      # [src, start, length]; src 0 == orig
        self._length = 0
        self._nl: list[int] | None = None       # newline offsets, lazily rebuilt
        if text:
            self._pieces.append([0, 0, len(text)])
            self._length = len(text)

    # -- buffers ---------------------------------------------------------
    def _buf(self, src: int) -> str:
        return self._orig if src == 0 else self._add[src - 1]

    def _locate(self, pos: int) -> tuple[int, int]:
        if not 0 <= pos <= self._length:
            raise IndexError(pos)
        acc = 0
        for i, (_, _, length) in enumerate(self._pieces):
            if pos <= acc + length:
                return i, pos - acc
            acc += length
        return len(self._pieces) - 1, self._pieces[-1][2]

    # -- editing ---------------------------------------------------------
    def insert(self, pos: int, text: str) -> None:
        if not text:
            return
        self._add.append(text)
        new_id = len(self._add)
        if not self._pieces:
            self._pieces = [[new_id, 0, len(text)]]
        else:
            i, off = self._locate(pos)
            src, start, length = self._pieces[i]
            repl = []
            if off:                                  # keep the left part
                repl.append([src, start, off])
            repl.append([new_id, 0, len(text)])
            if off < length:                         # keep the right part
                repl.append([src, start + off, length - off])
            self._pieces[i:i + 1] = repl
        self._length += len(text)
        self._nl = None

    def delete(self, pos: int, n: int) -> None:
        n = min(n, self._length - pos)
        if n <= 0:
            return
        end, kept, acc = pos + n, [], 0
        for src, start, length in self._pieces:
            lo, hi = acc, acc + length
            acc = hi
            if hi <= pos or lo >= end:               # untouched
                kept.append([src, start, length])
                continue
            left = max(pos, lo) - lo
            right = min(end, hi) - lo
            if left:
                kept.append([src, start, left])
            if right < length:
                kept.append([src, start + right, length - right])
        self._pieces, self._length = kept, self._length - n
        self._nl = None

    # -- reading ---------------------------------------------------------
    def __len__(self) -> int:
        return self._length

    def __str__(self) -> str:
        return "".join(self._buf(s)[st:st + ln] for s, st, ln in self._pieces)

    def _newlines(self) -> list[int]:
        if self._nl is None:
            self._nl = [i for i, ch in enumerate(str(self)) if ch == "\n"]
        return self._nl

    def line_count(self) -> int:
        return len(self._newlines()) + 1

    def line(self, idx: int) -> str:
        nl = self._newlines()
        if not 0 <= idx <= len(nl):
            raise IndexError(idx)
        text = str(self)
        start = nl[idx - 1] + 1 if idx else 0
        end = nl[idx] if idx < len(nl) else len(text)
        return text[start:end]

    def position_of_line(self, idx: int) -> int:
        nl = self._newlines()
        return 0 if idx == 0 else nl[idx - 1] + 1

    def line_of_position(self, pos: int) -> int:
        return bisect.bisect_right(self._newlines(), pos - 1)


if __name__ == "__main__":
    b = TextBuffer("hello\nworld\n")
    assert str(b) == "hello\nworld\n" and b.line_count() == 3
    b.insert(0, ">> ")
    b.insert(len(b), "last line")
    b.delete(3, 5)                                   # drop "hello"
    assert str(b) == ">> \nworld\nlast line", repr(str(b))
    assert b.line(0) == ">> " and b.line(1) == "world" and b.line(2) == "last line"
    assert b.position_of_line(2) == str(b).index("last")
    assert b.line_of_position(b.position_of_line(2)) == 2
    print("ok")
```

**Key points**

- No edit ever copies the document; only the piece list changes, so an edit is `O(#pieces)` with tiny constants.
- Adjacent pieces from the same buffer should be coalesced on every edit, otherwise a burst of keystrokes creates thousands of one-character pieces and every `str()` becomes slow.
- `delete` never mutates `orig` — that is what makes multi-level undo and concurrent readers cheap.
- Known weakness: `_locate` is a linear scan. Production uses a balanced tree keyed by cumulative length; that is the next thing I would change.

**中文要点**：选 piece table——原文不可变 + 追加缓冲，编辑只拆分 piece 不拷贝文本；行查找用延迟重建的换行偏移索引 + bisect。真正生产化还要把长度/换行计数放进平衡树节点里。

---

### 6. Serving a custom completion model to millions of DAU: walk me through the inference-cost model and your top three levers.

**Source**: https://www.youtube.com/watch?v=jV2sCj4lHYk · https://outcomeschool.com/blog/how-does-cursor-work

**Anchor answer**

Cost model first, because the levers fall out of it:

`cost/day = requests/day × (prefill_tokens × p_price + output_tokens × d_price) / cache_hit_rate_adjustment`

For an editor the shape is unusual: **huge prefix, tiny completion**. A tab request might carry 2000–8000 tokens of file context and emit 20–100 tokens. So prefill dominates the bill, and decode dominates the *latency*.

| Lever | Mechanism | Effect |
| :--- | :--- | :--- |
| **Kill requests before they cost anything** | debounce + cancel-in-flight when the user keeps typing | on tab, the majority of issued requests are never rendered |
| **Prefix caching** | stable system prompt + file head ordered first, volatile cursor tail last | prefill becomes a cache hit; one misordered token invalidates everything downstream |
| **Small, specialised model + quantization** | a completion model trained for next-edit, FP8, speculative decode | 5–20× cheaper per token than a frontier chat model at equal acceptance |

Secondary but real: route by tier (free vs paid gets different models), cap context by rank rather than raw size, batch on continuous-batching servers, and put inference in-region near users.

Failure mode to name: optimising kernels while the product already throws away most requests — the cancellation path is usually worth more than the serving stack.

**中文要点**：成本 ≈ 请求数 ×(prefill 贵 + decode 长)；编辑器是"大前缀 + 小补全"。三杠杆：取消/防抖让请求根本不发、前缀缓存让 prefill 命中、专用小模型 + 量化/speculative 降低单 token 成本。

---

### 7. Long context windows keep getting cheaper. Why not drop retrieval and stuff the whole repo into context for every request?

**Source**: https://outcomeschool.com/blog/lost-in-the-middle-problem-in-llms · https://outcomeschool.com/blog/how-does-cursor-work

**Anchor answer**

Three independent reasons, and only one of them is price.

1. **Attention quality degrades before the window fills.** Recall is U-shaped — material in the middle of a long prompt is used far less reliably than material at the edges. A 200k window does not mean 200k tokens of usable evidence; a repo dumped in the middle is mostly unattended noise.
2. **Prefill is quadratic-ish and, worse, cache-hostile.** Every keystroke changes the file you are editing. If the edited file sits early in the context, everything after it is a cache miss — you pay full prefill per keystroke. Retrieval lets you keep a **stable prefix** and append only a small volatile tail.
3. **A monorepo is bigger than any window and changes constantly.** 100k files at ~2k tokens each is orders of magnitude past the window; and the "whole repo" you built 30 seconds ago is stale.

Retrieval also gives something stuffing cannot: **ranking and provenance**. You can say *why* this file was included, and evaluate retrieval separately from generation.

The honest counterpoint: retrieval has its own failure — it can miss the file that matters, and top-k rewards similarity, not necessity. Best current practice is hybrid: retrieve to *select*, then include generously (whole file, not snippet) once selected, ordered stably for caching.

**中文要点**：不是钱的问题——中等位置的信息本就attention 衰减、编辑后前缀缓存全失效、仓库远大于窗口且时时变化；检索还给你可解释的排序与可单独评估的链路。正解是"检索选文件 + 整文件喂入 + 稳定前缀"。

---

### 8. Design the harness for an agent that makes multi-file changes from a natural-language task. How do you keep it from wrecking a codebase?

**Source**: https://outcomeschool.com/blog/how-does-cursor-work · https://outcomeschool.com/blog/harness-engineering-in-ai

**Anchor answer**

The harness is a **plan → edit → verify → present** loop where the agent can never touch the working tree directly.

| Stage | Mechanism | Guardrail |
| :--- | :--- | :--- |
| Context | repo map + search tools + open files; read-before-write enforced | agent cannot edit a file it has not read in this session |
| Plan | explicit file list + intended change per file, shown to user | user can reject before any token is spent on edits |
| Edit | **anchored patches** (`search` block must match exactly once) rather than whole-file rewrites | ambiguous or stale anchor → reject, re-read, retry |
| Verify | run the project's own lint/typecheck/tests in a sandbox | red gate blocks the "done" claim |
| Apply | all changes as one multi-file transaction with per-file generation stamps | conflicting user edit → that file is dropped and re-proposed |
| Undo | single checkpoint per agent run | one keystroke reverts the whole run |

Failure modes worth naming out loud: whole-file rewrite of a 2000-line file silently drops code the model did not read; parallel tool calls editing the same file race; "search/replace" applied to the *wrong* occurrence when the anchor is not unique; agents that delete a test to make it pass.

The verification gate is the one that actually prevents wreckage — an agent that must pass the repo's own tests cannot ship a plausible-but-broken diff.

**中文要点**：harness = 计划→带锚点编辑→跑仓库自带 lint/test→整体事务提交；铁律是"没读过不许改""锚点不唯一就拒绝""失败不许宣称完成"，再加一个可一键撤销的 checkpoint。

---

### 9. Design an agentic AI system that can autonomously adapt to new tasks.

**Anchor answer**

"Adapt" is three separable capabilities, and conflating them is why these systems disappoint:

| Capability | Mechanism | Ceiling |
| :--- | :--- | :--- |
| **Adapt within a task** | observe → act → read result → revise plan; failure becomes an observation, not an exception | solid, this is just a good loop |
| **Adapt across tasks in a session** | explicit task state (done / pending / blocked) + notes file the agent rewrites | good, if state is structured not "chat history" |
| **Adapt to a new domain** | tool discovery + a retrieval layer over docs/schemas | weak — needs a human to bound the tool set |

Architecture in four parts: a **planner** that emits a typed plan; an **executor** with a small, well-described tool set (search, read, write, run, ask); a **critic** with a *clean context* that checks the result against the original goal (self-review in the same context is worthless); and a **skill store** — successful traces distilled into reusable procedures, retrieved by task embedding.

The hard constraint: autonomy must be **bounded by verification**. Adaptation without a way to check the outcome is just improvisation; the system should be able to say "I cannot verify this" and stop. That single behaviour is what separates a demo from a system you can leave running.

**中文要点**：把"自适应"拆成三层——任务内闭环、会话内显式状态、跨领域工具/文档发现；关键是用**干净上下文的 critic** 校验结果，并允许系统说"我验证不了"然后停下。

---

### 10. Design Cursor's tab (next-edit prediction) system: it must feel instant (sub-100 ms perceived latency) for millions of daily users.

**Source**: https://outcomeschool.com/blog/how-does-cursor-work

**Anchor answer**

Budget the 100 ms before designing anything:

| Slice | Budget |
| :--- | :--- |
| Client context assembly (local index, diff of recent edits) | 5–10 ms, in the editor process |
| Network to nearest edge | 10–25 ms |
| Queue + prefill (cache hit) | 20–40 ms |
| First token → ghost text rendered | 10–20 ms |

That budget forces four architectural decisions:

1. **The model emits one short prediction, not a completion.** Next-edit = "the next change you are about to make", so output is a cursor position + a short insertion. Constrain decoding (grammar / region limit) so decode is bounded.
2. **Prefix cache is the product.** Order the prompt as `[system][file head][stable context][cursor tail]`. Anything volatile goes last, or every keystroke pays full prefill.
3. **Client-side debounce + cancellation.** A request whose context is already stale must be killed, not awaited. Most "latency" complaints are actually stale suggestions arriving late; suppress rather than race.
4. **Perceived ≠ measured.** Render the first token as an inline ghost immediately; the rest streams. p99 is the metric that matters — a 40 ms p50 with a 900 ms p99 reads as "laggy".

Evaluation must include accept rate, acceptance latency and **revert rate**, sliced per language.

**中文要点**：100ms 得先切预算（组装 10 / 网络 20 / 缓存命中的 prefill 30 / 首 token 20）；关键是把 prompt 排成"稳定前缀 + 易变尾巴"吃满缓存，客户端做防抖与取消，先渲染首 token，且盯 p99 而非 p50。

---

### 11. How would you index a 100k-file monorepo so an AI editor can retrieve relevant context, and keep the index fresh as the user edits?

**Source**: https://outcomeschool.com/blog/how-does-cursor-work

**Anchor answer**

Two indexes, because they answer different questions:

| Index | Built from | Answers |
| :--- | :--- | :--- |
| **Symbol / structure** | tree-sitter parse → file → symbol → (signature, byte range) | "where is `Foo` defined", "what calls this" — exact, cheap, no model |
| **Semantic** | embeddings per *symbol-sized chunk* (200–1500 tokens), never whole files | "where do we handle retries" — fuzzy intent |

Plus **lexical search** (trigram/BM25) which, in code, frequently beats embeddings: identifiers are rare tokens.

Practical shape:

1. Chunk on AST boundaries (function/class), with path + language + enclosing symbol as metadata — not fixed 512-token windows.
2. Store in a vector index keyed by `(repo, commit-bucket, path, symbol_hash)` so a changed function is a single-key update, not a reindex.
3. Freshness: on save, hash the file; if the hash changed, reparse + re-embed **that symbol set only**, debounced ~1 s. Deletes tombstone; renames re-key.
4. At query time: merge semantic + lexical + "currently open files" + import-graph neighbours, dedupe by symbol, rank, then return whole symbols.
5. Degrade gracefully: the editor must work with a stale index — fall back to live `ripgrep` over the user's working tree.

Failure to call out: embedding the *whole* repo on every commit. Cost scales with churn, not with repo size.

**中文要点**：双索引——AST 符号索引（精确、免模型）+ 符号粒度向量索引，再叠词法检索；增量更新按"文件哈希变化→只重算该文件的符号"，查询时合并 open files 与 import 图，且索引陈旧时要能退化到实时 grep。

---

### 12. The model is streaming a multi-file edit while the user keeps typing in one of those files. How do you apply the edits without corrupting the buffer?

**Anchor answer**

The invariant: **an edit is only valid against the exact document version it was computed from**, and the editor's document model — not the server — is the only writer.

Mechanism:

1. **Version stamp.** Every request carries `(path, content_hash, revision)` per file. Every streamed hunk names the file + the stamp it was generated against.
2. **Apply as anchored, minimal hunks.** A hunk is `{file, anchor_before, replace, anchor_after}` — not "lines 40–120 replaced". Anchors survive insertions elsewhere in the file.
3. **Per-hunk rebase at apply time.** Right before applying hunk *k*, re-read the buffer. Hash still matches → apply atomically inside one editor transaction. Doesn't match → try the anchor search; if the anchor region itself was touched by the user, **drop this hunk, keep the rest**, and tell the user "this part conflicted, here is the proposal".
4. **Never hold a lock across the stream.** Buffer stays editable; queued hunks are validated at apply time, not at receive time.
5. **Undo grouping**: one transaction per agent run, so undo removes the whole edit and not 40 fragments.

The corruption cases this prevents: applying line-range patches after the user inserted 5 lines at the top (the classic runaway shift), and two hunks in the same file being applied against each other's output.

Honest limit: if the user rewrites the exact function the agent is editing, do not merge — surface the diff and let the human choose.

**中文要点**：编辑只对"生成时的那一版"有效——请求带 `(path, hash, revision)`，流式下发的每个 hunk 用上下锚点而非行号，落盘前再校验哈希/锚点，冲突就**只丢这个 hunk**；全程不锁缓冲区，整轮合成一个 undo 组。

---

### 13. Your agent model outputs an edited version of a 500-line file. Applying it verbatim is slow and error-prone. How do you make “apply” fast and reliable?

**Source**: https://outcomeschool.com/blog/how-does-cursor-work

**Anchor answer**

Applying verbatim means "diff 500 lines against 500 lines" — an `O(n·m)` LCS that also reproduces the model's incidental whitespace churn. Replace it with a **layered apply pipeline**, each layer cheap and verifiable:

| Layer | Technique | Handles |
| :--- | :--- | :--- |
| 1 | **Structured hunk format**: model emits `{anchor, old_text, new_text}`, not the whole file | normal case, no diffing at all |
| 2 | Exact `str.find` of `old_text`, require exactly one match | `O(file)` per hunk |
| 3 | Fuzzy match: whitespace-normalised, then token-level similarity with a threshold; pick best-scoring window | indentation drift, small paraphrase |
| 4 | **Fast-apply model**: a small model merges "intent diff" into the *current* file content | heavy drift, multiple overlapping hunks |
| 5 | Full-file rewrite, last resort, shown as a diff for approval | structural rewrites |

Then **verify structurally**: parse the result (tree-sitter) — if the file no longer parses, refuse. Cheap post-checks catch the classic failure where a fuzzy match lands in the wrong function.

Two more rules: apply all hunks in **one pass bottom-up by offset** so earlier hunks don't shift later ones; and stream hunk-by-hunk so the user sees progress after 200 ms instead of waiting for the 500-line file.

**中文要点**：别做整文件 diff——让模型输出带锚点的 hunk，按"精确匹配 → 模糊匹配 → fast-apply 小模型 → 整文件重写"逐层降级；落盘后必须重新 parse 校验，且按偏移从后往前一次性应用。

---

### 14. An agent needs to iterate on code (run builds, tests, lints) without disturbing what the user sees in their editor. Architect that.

**Anchor answer**

Separate the **workspace** from the **view**. The agent never edits the user's buffers; it edits a disposable copy and reports back diffs.

| Layer | Choice | Reason |
| :--- | :--- | :--- |
| Workspace | `git worktree` or a container clone at the same commit | cheap isolation, exact same code, no copy of the user's uncommitted work unless opted in |
| Execution | sandboxed container: CPU/mem/time limits, no prod credentials, egress allowlist | a runaway `npm install` must not become an incident |
| Feedback | tail-truncated logs + structured exit codes returned to the model | raw 10k-line logs destroy the context window |
| Presentation | unified diff streamed into the editor | user reviews and applies; nothing appears "already changed" |
| Cleanup | TTL per sandbox, snapshot on completion | orphaned containers are the #1 cost leak |

Key details that show you have done this:

1. **Deterministic bootstrap** — the agent must not `npm install` on every step; use a warm image/overlay and cache `node_modules` by lockfile hash.
2. **Same-commit guarantee** — record the base SHA in every result, so a diff can never be applied to a moved branch silently.
3. **Streaming with truncation policy** — show the last N lines of failure, plus "grep-able" full log stored at a path the agent can search.
4. **Cancellation** — user closes the tab, the container dies.

Failure mode: running the agent against the user's live filesystem and trusting it to be tidy. It will leave `.cache/`, modify lockfiles, and occasionally `rm -rf` something.

**中文要点**：用 `git worktree`/容器克隆出副本 + 沙箱限权限时，agent 只在副本里跑 build/test；结果以 diff 回推到编辑器；日志要截断存储，注意孤儿沙箱和 lockfile 级缓存。

---

### 15. How do you evaluate a code-editing model before shipping it? Design the offline and online eval story for tab or agent edits.

**Anchor answer**

Offline and online answer different questions; shipping on acceptance rate alone is the standard mistake.

**Offline**

| Suite | What it measures | Trap |
| :--- | :--- | :--- |
| Executable tasks (repo + hidden tests) | did the patch actually work | contamination if repos are public |
| Edit-level match on held-out real suggestions | format fidelity, minimal-diff discipline | overfits to the eval set |
| Replay of accepted/rejected production pairs | ranking quality vs. the current model | distribution drift |
| Adversarial: stale context, partial file, conflict | does it propose nonsense when unsure | usually omitted |

Use the *same* harness for both models and report per-language slices; a global number hides Python/TS divergence.

**Online**

Acceptance rate is necessary but gameable — a model that suggests tiny safe edits scores well and helps little. So pair it with:

- **persistence**: is the suggestion still there 30 s / 10 min later? Reverted suggestions are worse than no suggestion.
- **follow-on churn**: how many edits immediately correct the accepted one.
- **latency p50/p99** and cancellation rate.
- **net effect**: keystrokes saved per 100 edits, and test pass rate on agent runs.

Always run an A/B with a guardrail metric (revert rate, undo rate, time-to-commit) and randomise a small holdback so you keep measuring the counterfactual.

**中文要点**：离线用可执行任务 + 编辑级匹配 + 生产接受/拒绝对回放；在线别只看接受率（会奖励"小而安全"的废话），要配 30s/10min 留存率、回退率、后续修补率、p99 延迟，并留 1% 对照组。

---

### 16. You have two days in our codebase and no assigned task. What do you build, and how do you spend the time?

**Anchor answer**

Structure the answer as a **decision procedure**, then walk one concrete path through it. What is being graded is scoping and judgement, not the artefact.

**Selection criteria** (in order): something the team already feels pain about → small blast radius → demoable in 4 minutes → touches the real product surface, not a side repo.

A workable schedule:

| Time | Activity |
| :--- | :--- |
| 0–2 h | read the repo: build, test, the module the round's interviewer owns; run it locally; write down 5 friction points |
| 2–3 h | ask one engineer "which of these would you actually merge" — this is the highest-value hour |
| 3–5 h | pick one, write a 1-page scope: what ships, what is explicitly out |
| Day 1 rest | implement the smallest end-to-end slice behind a flag |
| Day 2 AM | tests, failure paths, telemetry |
| Day 2 PM | demo + a short written note: what I learned, what I'd do with two weeks |

Anti-patterns to name explicitly: refactoring an architecture you do not yet understand; building a greenfield toy that never touches their code; and trying to fix everything. Also mention you would use AI tools aggressively for boilerplate but keep the design decisions yours and be able to defend every line.

**中文要点**：两天的价值在"选题与收敛"——先跑通仓库、列 5 个痛点、问工程师哪个真会 merge，再砍到 4 分钟能演示的一条竖切；明确说出你**不**做什么（大重构、独立玩具）。

---

### 17. Tell me about a time you made short-term sacrifices for long-term gains.

**Anchor answer**

Answer with a **shape**, not a vibe. The interviewer is scoring four things: was the cost real, was the payoff measured, was the alternative genuinely cheaper, and did you keep a way out.

Structure:

1. **The cheap path** — one sentence on what was faster and why it was tempting (ship the feature, patch the symptom, skip the migration).
2. **The cost you accepted, quantified** — a sprint of no visible progress, a two-week slip, a temporary 30% latency regression during dual-write. Say the number; "it was hard" scores zero.
3. **The payoff, measured** — what changed afterwards: incident count, onboarding time, deploy frequency, deleted lines of code.
4. **The hedge** — how you kept the bet reversible: feature flag, dual-write with a shadow read, a decision deadline with an explicit rollback criterion.
5. **What you'd do differently** — one honest miss (usually: you under-communicated the cost upward, and had to defend it mid-flight).

Good raw material: deleting an abstraction everyone built on; writing the migration before the feature; spending a sprint on observability so the next three incidents are one-hour instead of one-week; saying no to a customer commitment to keep the test suite green.

Avoid: stories where the "long-term gain" is unmeasured, and stories where the sacrifice was other people's time without their consent.

**中文要点**：按"便宜路径是什么→你付出的可量化代价→事后可测量的收益→如何保留回退余地→你会改进什么"五段讲；关键是代价要有数字、收益要有指标，否则听起来像自我感动。

---

## 附：这 17 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 数据结构基本功 | 1、2（哈希树）、3（top view）、5（文本缓冲区） |
| 真实工程手感 | 12（流式编辑冲突）、13（apply 管线）、14（沙箱迭代） |
| 延迟/成本直觉 | 6（推理成本）、10（tab 100ms 预算） |
| 检索与上下文 | 7（长上下文 vs 检索）、11（monorepo 索引） |
| 评测成熟度 | 15（接受率陷阱）、8（验证闸门） |
| 文化与自驱 | 16（两天无人指派）、17（长期取舍） |
