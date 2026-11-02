# 27 · Tesla

> 11 题。出处：`README.en.md` § Tesla。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Implement non-maximum suppression. Then vectorise it.

**Anchor answer**

**Reference implementation**

```python
import numpy as np

def iou_1_to_n(box, boxes):
    """box: (4,) xyxy. boxes: (N,4). Returns (N,) IoU."""
    ix1 = np.maximum(box[0], boxes[:, 0]); iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2]); iy2 = np.minimum(box[3], boxes[:, 3])
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)

def nms(boxes, scores, iou_thr=0.5, max_out=100):
    order = np.argsort(-scores)
    keep = []
    while order.size and len(keep) < max_out:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        # one vectorised IoU against the current winner, then mask
        mask = iou_1_to_n(boxes[i], boxes[rest]) <= iou_thr
        order = rest[mask]
    return np.array(keep, dtype=np.int64)
```

**Key points**

- The loop is over **kept boxes**, not over pairs — that is why it is already vectorised in the inner dimension. Fully branch-free NMS needs a sort + IoU matrix + iterative masking (torchvision does this on GPU).
- **Sort once, before the loop.** Re-sorting inside is the classic 10x mistake.
- IoU variant matters in production: `IoU` vs `GIoU` vs center-distance (for rotated boxes, axis-aligned IoU is wrong — use BEV rotated IoU or a distance threshold on centers).
- **Class-aware**: run per class, or offset boxes per class into disjoint coordinate space so a single batched NMS call works.
- Thresholds: 0.5 is too aggressive for crowded scenes (pedestrians overlap legitimately) — use soft-NMS or a lower threshold plus a top-k cap.
- **Pre-filter by score**, not just top-k: on 100k anchors, worth doing before sorting.

**中文要点**：NMS 外层按分数排序只做一次，内层用向量化 IoU 一次算"当前框 vs 剩余全部"再掩码；旋转框不能用轴对齐 IoU；拥挤场景 0.5 阈值会吃掉真目标，用 soft-NMS 或降阈值配 top-k。

---

### 2. Write an efficient ring buffer for high-rate sensor data with a fixed memory budget.

**Anchor answer**

**Reference implementation**

```python
import numpy as np

class RingBuffer:
    """Fixed-capacity, single-writer/single-reader, overwrite-oldest on full."""
    def __init__(self, capacity, item_shape, dtype=np.float32):
        assert capacity & (capacity - 1) == 0, "capacity must be a power of two"
        self.mask = capacity - 1
        self.buf = np.zeros((capacity, *item_shape), dtype=dtype)
        self.ts = np.zeros(capacity, dtype=np.float64)
        self.head = 0          # next write slot
        self.tail = 0          # oldest valid slot
        self.dropped = 0

    def push(self, item, t):
        if (self.head - self.tail) == self.mask + 1:   # full
            self.tail = (self.tail + 1) & self.mask    # overwrite oldest
            self.dropped += 1
        self.buf[self.head & self.mask] = item
        self.ts[self.head & self.mask] = t
        self.head += 1

    def pop(self):
        if self.tail == self.head:
            return None
        i = self.tail & self.mask
        self.tail += 1
        return self.ts[i], self.buf[i]

    def latest(self, k=1):
        hi = min(self.head, self.tail + self.mask + 1)
        idx = [(hi - j) & self.mask for j in range(k, 0, -1)]
        return self.ts[idx], self.buf[idx]
```

**Key points**

- **Power-of-two capacity → `& mask` instead of `%`**, and magic-number indexing (`head`/`tail` are monotonic 64-bit counters, never wrapped) so "full" vs "empty" is unambiguous.
- **Choose the overflow policy explicitly**: overwrite-oldest (perception wants the freshest data) vs drop-newest (logging wants every sample). Silent drops must be counted — a `dropped` counter is your only evidence of overrun.
- Fixed memory means **preallocate everything**; no `append`, no per-message allocation. On the sensor path that allocation is the jitter.
- In C++: atomics with acquire/release on head/tail, pad the counters to separate cache lines, and keep the buffer aligned so DMA/camera writes are not split across sectors.
- Backpressure beats buffering: if consumer lag exceeds one buffer, the pipeline is wrong — drop and alarm rather than grow.

**中文要点**：定长环形的关键是指针单调递增 + 容量取 2 的幂用 `& mask` 取模，从而干净区分空/满；溢出策略（覆盖最旧 vs 丢新）必须显式选并计数，内层绝不能有 per-message 分配。

---

### 3. How would you design the neural network architecture for multi-camera 3D object detection without lidar?

**Anchor answer**

The whole problem is **depth ambiguity**, so the architecture is mostly a bet on how you resolve it.

| Stage | Design choice | Why |
| :--- | :--- | :--- |
| Backbone | shared image backbone + FPN over 6–8 cameras | shared weights keep latency linear in cameras |
| View transform | **BEV**: LSS-style predicted depth distribution, or BEVFormer-style cross-attention to 3D queries | one metric space to fuse cameras; attention version needs no explicit depth GT |
| Temporal | ego-motion-aligned BEV feature queue (4–8 frames) | a single frame cannot resolve velocity or occluded objects; motion gives you both |
| Heads | center-based heatmap + regress `(x, y, z, w, l, h, yaw, vx, vy)` + attribute heads | anchor-free matches BEV grid; velocity is needed downstream by the planner |
| Training signal | **lidar as privileged teacher**: depth supervision, and distillation of lidar-trained teacher into the image student | lidar is available offline; using it only at train time is the standard trick |

Concrete numbers to reason with: BEV grid 0.5 m/cell over 100 m × 100 m; a car at 50 m is ~15 px wide at 1280p — so small-object recall is a **resolution and feature-stride** problem, not a head-design problem.

Failure modes: calibration drift (projection silently wrong → the model learns to ignore cameras), rolling shutter and unsynchronized cameras, and far-field BEV cells with almost no pixel support. Convert 3D→2D projection consistency into a training loss/assert to catch calibration drift.

**中文要点**：无激光雷达的 3D 检测本质是"如何解深度"——BEV 视图变换 + 时序对齐（给速度与遮挡）+ 中心点式 3D 头；训练时用激光雷达当 teacher 做深度监督/蒸馏；远距离目标受分辨率与 stride 限制，标定漂移是隐形杀手。

---

### 4. How do you handle extreme class imbalance in rare-event detection (e.g. a child running into the road)?

**Anchor answer**

Rates here are ~1e-6 to 1e-7 per frame, so **accuracy is meaningless** and sampling alone cannot fix it.

| Layer | Technique | Note |
| :--- | :--- | :--- |
| Loss | focal loss, class-balanced weighting, cost-sensitive CE | weight by **cost of miss**, not by inverse frequency alone |
| Sampling | positive oversample + **hard-negative mining** | random negatives are useless; mine "stroller", "cyclist", "parked car with motion" |
| Architecture | two-stage: cheap high-recall trigger (motion toward ego path) → precise classifier on crops | recall in stage 1, precision in stage 2 |
| Features | temporal: tracklet velocity/heading over 1 s, ego-motion-relative trajectory | a child entering the path is a *motion* event, not a single-frame appearance |
| Decision | threshold chosen on the **PR curve at a fixed false-positive-per-mile budget** | the operating point is a product decision |

Evaluation: PR-AUC and recall at fixed FPR (e.g. <1 FP per 10k frames), never ROC-AUC or accuracy. Report per-slice (night, rain, occluded) and with confidence intervals over miles — rare events are Poisson-ish, so a delta on 40 events is noise.

The trap: oversampling positives until the model fires on every pedestrian. Fix with hard negatives and a **calibration check** — expected FP per mile computed on a held-out continuous drive, not a shuffled frame set (shuffling destroys the base rate).

**中文要点**：稀有事件必须按"每英里误报预算"选阈值、用 PR-AUC/固定 FPR 下的召回评估；做法是 focal + 难负样本挖掘 + 两阶段（高召回触发 → 精分类）+ 时序运动特征；别用 accuracy，别打乱帧做验证。

---

### 5. Explain how you would auto-label a fleet dataset and what quality controls you would put on it.

**Anchor answer**

Auto-labelling is **offline perception running slower and with sensors the car does not have at inference time** (lidar, multi-frame, higher resolution).

Pipeline:

1. **Trigger → clip** (10–20 s) around the event, with ego pose and calibration version.
2. **Offline teacher ensemble**: 3D detector on lidar + multi-frame tracking + SLAM-consistent aggregation. Track-level labels (not per-frame) are far more consistent.
3. **Propagate**: fit a track through the clip, project to every camera → boxes, lanes, depth, semantics.
4. **Hard-case routing**: low confidence, ensemble disagreement, track breaks, geometry implausible → human queue.

Quality controls, in the order they actually catch bugs:

| Control | Catches |
| :--- | :--- |
| **Ensemble disagreement rate** per class | teacher blind spots |
| **Geometric plausibility** (size, ground contact, speed continuity) | projection/calibration errors |
| **Track continuity** (no teleporting, no flicker) | frame-level noise |
| **Stratified human audit** (N per ODD slice, per class) | systematic bias you did not think to look for |
| **Frozen human-labelled golden set**, re-scored every labeler release | silent regression in the labeler |
| Dedup near-identical clips | 10k copies of one intersection dominate the loss |

Failure modes: **confirmation bias** (student learns teacher errors and the metric on auto-labels looks great), long-tail erosion (rare classes get no labels at all), and label-version drift (mixing labels from two labeler versions in one training run).

**中文要点**：自动标注 = 用离线更强感知（激光雷达、多帧、更高分辨率）当 teacher，按 track 级而非帧级出标签；质检靠集成分歧、几何合理性、轨迹连续性、分层人工抽检和冻结的黄金集，最怕确认偏误与长尾消失。

---

### 6. How would you detect and handle distribution shift between fleet data and your training set?

**Anchor answer**

Detect at three levels, because each catches what the others miss:

| Level | Signal | Method |
| :--- | :--- | :--- |
| **Input** | image statistics, scene/weather/ODD classifier mix, embedding distribution | per-slice share vs training prior; PSI / KL on binned features; MMD on backbone embeddings |
| **Output** | detection count, confidence histogram, track length, class mix | confidence-distribution shift is the cheapest early warning |
| **Performance** | disengagement rate, shadow-mode disagreement with the deployed stack, human audit of interventions | ground truth, but lagging and biased |

Handling, in order: (1) **confirm it is shift and not a logging bug** — a changed camera ISP or a new firmware will look exactly like covariate shift; (2) **slice the metric** to find which ODD degraded, since an aggregate number hides a rain-only regression; (3) mine and label the new slice (active learning on high-uncertainty samples); (4) reweight or add the slice to training, or fine-tune; (5) if the ODD is genuinely new and unvalidated, **restrict the ODD** rather than ship a guess.

The counterintuitive part: a distribution shift can leave accuracy unchanged and still be dangerous, because the *decision boundary region* moved. Monitor per-slice recall on rare classes, not just aggregate mAP.

**中文要点**：三层监控——输入分布（PSI/MMD、场景分类占比）、输出分布（置信度直方图是最便宜的预警）、真实表现（影子模式/接管）；先排除日志与固件变更造成的假漂移，再按 ODD 切片定位，最后挖数据补标或直接收缩 ODD。

---

### 7. The onboard compute budget is fixed. Walk me through quantizing and pruning a vision model without losing recall on small objects.

**Source**: https://outcomeschool.com/blog/how-does-model-quantization-work

**Anchor answer**

Order matters: **quantize what tolerates it, prune structurally, then recover with distillation.**

| Step | Do | Why it protects small objects |
| :--- | :--- | :--- |
| 1 | Profile per-layer sensitivity: INT8 PTQ, measure recall on the **small-object slice** (<32 px), not aggregate mAP | small-object features are low-energy; quantization error is relative to feature magnitude, so they die first |
| 2 | Keep **first/last layers, the detection head, and any norm/softmax in FP16** | mixed precision here costs ~5% of the speedup and most of the damage |
| 3 | Prefer **per-channel** weight scales; QAT if PTQ loses >0.5 pt | per-tensor scaling is where the 3 pt drops come from |
| 4 | Prune **channels/filters (structured)**, never unstructured sparsity | unstructured sparsity gives zero latency win on real hardware without 2:4-aware kernels — and 2:4 still needs the right shape |
| 5 | **Do not reduce input resolution.** Prune the deepest stages / neck instead | resolution is the single biggest lever on small-object recall |
| 6 | Distill from the FP32 teacher on the pruned student, including feature-level loss on the neck | recovers 50–80% of the loss in a day of training |

Also quantize the **post-processing** (NMS/decode) or its FP32 casts will eat the gains, and re-measure end-to-end latency — a 2x MAC reduction that becomes memory-bound can move nothing.

**中文要点**：顺序是先量化（后端敏感层保 FP16、逐通道 scale）、再结构化剪枝（只剪通道，别指望非结构化稀疏真加速）、最后用 FP32 teacher 蒸馏恢复；小目标最先被量化误差吃掉，评估必须看 <32px 切片召回。

---

### 8. Design the data engine: fleet triggers → upload → labelling → retraining → shadow-mode validation → release.

**Anchor answer**

Six stages, and the design question for each is **what makes it fail**.

| Stage | Mechanism | Failure mode |
| :--- | :--- | :--- |
| **Trigger** | onboard detectors for: disengagement, hard brake/steer, low-confidence perception, new ODD (geofence), shadow disagreement, user report | trigger precision too low → fleet floods you with 50k clips/day; too high → you miss the tail |
| **Upload** | prioritised queue, compression, dedup by geo/time hash, privacy redaction before it leaves the car | bandwidth is scarce; unprioritised upload starves the important clip |
| **Label** | auto-label (offline teacher) → human on disagreement/uncertainty → track-level propagation | confirmation bias; labeler version drift |
| **Dataset** | versioned, sliced by ODD, **stratified sampling** so one intersection cannot dominate, point-in-time correctness | silent leakage between train and eval clips (same drive, adjacent frames) |
| **Train** | fixed recipe, same eval suite every run, regression gate on rare-class slices | metric chasing on aggregate mAP while a slice regresses |
| **Shadow → release** | new stack runs silently, disagreements triaged; staged rollout by region/vehicle cohort, with rollback | shadow agreement can look great while both stacks share a blind spot |

The two numbers that define whether the engine works: **trigger→training latency** (weeks to days) and **label yield per uploaded GB**. If either is bad, more triggers just add cost.

**中文要点**：数据引擎每一环都有自己的失败模式——触发太松淹没标注、上传不优先、自动标注确认偏误、划分泄漏（同一段行驶相邻帧）、影子模式共享盲区；核心 KPI 是"触发到训练"的延迟与每 GB 有效标签产出。

---

### 9. Disengagement rate is a weak proxy. How would you actually measure whether an autonomy release is safer than the last one?

**Anchor answer**

Disengagement is weak for three specific reasons: the **ODD mix changes** between releases, driver attention varies, and the event rate is so low that you cannot get significance in a reasonable number of miles.

What to measure instead, in layers:

| Layer | Metric | Notes |
| :--- | :--- | :--- |
| **Counterfactual / simulation** | collision and near-miss rate on a fixed, versioned scenario suite; replay-based "what would the new stack have done" | highest power per compute; the suite must be adversarial and *frozen* |
| **Shadow mode** | disagreement rate, and **who was right** on the triaged subset | leading indicator, no safety cost |
| **On-road safety-critical events** | hard braking, evasive steer, at-fault near-miss per 100k miles, with Poisson/negative-binomial CIs | needs scale; report intervals, not point estimates |
| **Leading indicators** | perception recall on rare-class slices, time-to-collision distribution, comfort (jerk) | catches regressions weeks earlier |
| **Human audit** | stratified review of interventions with a rubric, double-annotated for agreement | fixes the "driver disengaged for a non-safety reason" noise |

State the statistical reality out loud: distinguishing a 20% reduction in a 1-per-100k-mile event needs order 10^6+ miles per arm. That is why simulation plus leading indicators, not on-road counts, are the decision instrument.

**中文要点**：接管率弱在 ODD 构成变化、驾驶员注意力差异、事件太稀疏；应该用冻结场景集的仿真反事实 + 影子模式分歧 + 带置信区间的安全关键事件率 + 稀有类召回等先行指标，并明说路上实测需要百万英里量级才显著。

---

### 10. How would you fuse camera, radar and IMU inputs into a single perception stack, and where would you fuse them?

**Anchor answer**

Fuse at the point where each sensor's **failure mode is still recoverable** — and be explicit that IMU is not a perception sensor.

| Modality | What it uniquely gives | Where to fuse | Rate |
| :--- | :--- | :--- | :--- |
| Camera | semantics, class, lane, signs | backbone / BEV features | 20–36 Hz |
| Radar | **radial velocity** and range, works in rain/dark | mid-fusion: BEV features, plus velocity association at track level | 20–70 Hz |
| IMU + wheel speed | ego motion, yaw rate, accel | **state estimation / ego-motion compensation**, not object features | 100–1000 Hz |

Design: per-modality encoder → align to a common BEV frame using calibrated extrinsics and **ego-motion propagated to a common timestamp** → cross-attention (or gated concat) → shared detection/tracking heads → tracker with radar Doppler as a velocity measurement.

Three things that decide whether this works:

1. **Time sync.** Hardware-triggered capture with per-frame timestamps; fusing a 30 ms-old radar sweep with a camera frame at 20 m/s is a 0.6 m error before you even start.
2. **Calibration as a monitored quantity**, not a one-time file — a 0.2° yaw error drifts a 50 m point by 17 cm.
3. **Graceful degradation**: radar-only and camera-only paths must each be validated; a modality that dies must not silently zero-out a BEV cell.

Failure modes: radar multipath/ghosts, water and dirt on the lens, IMU bias drift, and fusion weights that over-trust camera at night.

**中文要点**：融合点选在"各传感器失效仍可恢复"的位置——相机/雷达在 BEV 特征层做中融合，IMU 只用于自车运动补偿与状态估计，不进目标特征；真正的成败在时间同步、标定漂移监控、以及单模态失效时能优雅降级。

---

### 11. Tell me about the most technically demanding thing you have shipped, and what you would do differently.

**Anchor answer**

Pick the project on **technical difficulty × your ownership**, not on business glamour. Then answer in four beats, ~90 seconds:

1. **The hard part, in one sentence with a number.** "The model had to run in 30 ms on a 60 W budget with a 4-camera input" — the constraint is what makes it technically demanding. "It was a big project with many stakeholders" is not a technical answer.
2. **The one decision you made that others would not have.** Name the rejected alternative and why: "we kept global attention in the neck instead of pure conv, because the ablation on cross-camera consistency was worth 1.4 pt mAP at 8% latency".
3. **What actually went wrong** — mid-project, in your own code, with the debugging story. Interviewers weight this the highest; a story with no failure is a story with no detail.
4. **What you would do differently, precisely.** "I would have built the latency harness in week one instead of week six; we lost two weeks optimizing the wrong layer." Specific, cheap, and about process — not "I would communicate more".

Avoid: a project you only contributed one component to (cannot answer "what would you do differently"), and a story where the resolution is someone else's decision. Have a second, smaller story ready as a follow-up — the natural next question is "what did you learn", and the best answer is another example of applying that lesson.

**中文要点**：选"技术难度 × 你自己拥有"的项目，四拍：一句话带数字的约束、别人不会做的那个决策与被否掉的替代方案、自己代码里真实翻车的调试过程、以及具体到"第一周就该搭延迟 harness"的改进；别讲只有你参与一个模块的项目。

---

## 附：这 11 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 视觉感知架构 | 3（无雷达多相机 3D 检测）、10（相机/雷达/IMU 在哪一层融合） |
| 数据引擎与闭环 | 5（自动标注与质检）、6（分布漂移）、8（触发→上传→标注→影子发布） |
| 端侧部署 | 7（量化 + 剪枝保住小目标）、2（定长环形缓冲） |
| 安全与评估思维 | 9（接管率为什么弱）、4（极端不均衡稀有事件） |
| 手写代码 | 1（NMS 与其向量化） |
| 软素质 | 11（最难的项目 + 会怎么改） |
