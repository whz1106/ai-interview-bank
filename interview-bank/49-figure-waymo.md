# 49 · Figure AI & Waymo

> 24 题（Figure AI 12 + Waymo 12）。出处：`README.en.md` § Figure AI、§ Waymo。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

## Figure AI

### 1. Behaviour cloning on teleoperation data has a well-known failure mode. What is it, and what do you do about it on a real humanoid?

**Anchor answer**

**Covariate shift (compounding error)**: the policy is trained only on expert states, but at deployment it visits states the expert never visited — and in those states it has no supervised label, so it does something arbitrary, which pushes it further off-distribution.

Second, subtler failure: **multi-modality averaging**. Humans solve a task several ways; an MSE-trained regressor averages two valid trajectories and produces an invalid one (hand goes through the table, or hovers). The symptom is a policy that is "almost right" and freezes.

Third, on humanoids specifically: **idle/slow collapse**. If demonstrations contain pauses and the policy is scored on imitation loss, standing still is locally optimal and cheap — the robot learns to hesitate.

Fixes, roughly in order of payoff:

| Fix | What it addresses |
| :--- | :--- |
| **Action chunking** (predict H steps) + temporal ensembling | compounding error, hesitation |
| **Diffusion / flow-matching policy** | multi-modality averaging |
| **DAgger-style on-policy correction**: run the policy, have the operator take over and label recovery states | covariate shift directly |
| **Recovery demonstrations**: deliberately start near-failure and demonstrate the recovery | the states that matter most are the rarest |
| **Noise/perturbation augmentation** to observations + demo starting-state diversity | coverage |
| **Speed-weighted loss / drop idle frames** | hesitation |

What you learn on hardware: the demo data is 90% "successful", but the policy spends most of its life in the 10% "recovering". Budget collection for recovery, or you will be fine-tuning forever.

**中文要点**：行为克隆的病根是**协变量偏移**（只见过专家状态，出错后进入无标注区域）和多模态平均（两种解法平均成非法动作）；对策是动作分块+扩散策略+DAgger 式在策略纠正，并专门采集"恢复"演示。

---

### 2. A whole-body controller trained entirely in simulation has to run on real hardware. What transfers, what does not, and how do you close the gap?

**Anchor answer**

| Transfers well | Transfers badly |
| :--- | :--- |
| Kinematics, joint limits, link geometry | actuator dynamics: backlash, torque saturation, thermal derating |
| Contact *geometry* (roughly) | contact *compliance*: rubber feet, sole deformation, soft ground |
| Gravity/inertia if masses are identified | friction (stick-slip), stiction at low speed |
| Balance strategies, gait timing | sensor latency, filter delay, IMU bias/drift |
| Policy architecture & training recipe | battery voltage sag → torque available drops through the run |

The gap is usually **latency + actuator model**, not physics fidelity. A policy trained with zero observation delay is unstable at 20-40ms of real delay; a policy that assumes ideal torque commands oscillates when the real joint controller has a first-order lag.

Closing it, in practice:

1. **System identification first**: measure mass/inertia, joint friction, motor constants, and — most importantly — the *closed-loop* step response of the actuator as commanded by software. Fit a small learned actuator network (`commanded torque → achieved torque`) and put it in the sim.
2. **Randomise what you cannot identify**, but randomise the *right* things: latency (0-50ms), friction, payload, CoM offset, floor height/friction. Over-randomising everything produces a conservative, sluggish policy that walks like it is drunk.
3. **Train with the deployment stack in the loop**: same control rate, same filters, same quantization, same observation history buffer. Simulation of the *software* is cheaper and more valuable than more photorealistic rendering.
4. **Residual/adaptation on hardware**: a learned residual on top of the sim policy, or online adaptation of a few parameters (payload, friction) from recent state.
5. **Progressive transfer**: sim → hardware-in-the-loop bench (legs in a rig) → tethered → free. Each stage has its own pass criteria.

The failure you should expect: it works in sim at 0.1% failure, then on hardware the first 200 steps are fine and it fails at minute 3 — thermal derating and battery sag are *slow* dynamics that no episode-length randomisation covers.

**中文要点**：能迁移的是运动学、接触几何、平衡策略；迁不动的是**执行器动力学、接触柔顺性、延迟、电池压降**。先做系统辨识（尤其是"指令力矩→实际力矩"的闭环响应），把延迟和摩擦作为随机化重点，并在仿真里跑真实的控制栈。

---

### 3. Where does reinforcement learning fit on top of imitation learning for manipulation, and what makes the reward the hard part?

**Anchor answer**

RL's job is **not** to learn the task from scratch — it is to fix the specific things BC cannot.

| Placement | What RL optimises | Why BC alone fails there |
| :--- | :--- | :--- |
| Residual RL | a small corrective action on top of the BC policy | BC cannot recover from its own drift |
| Recovery RL | getting back to a good state after perturbation | recovery states are rare in demos |
| Fine-tune / RLHF-style | success rate under a task reward, starting from a BC prior | demos encode "what the human did", not "what works" |
| Skill selection / sequencing | which learned skill to invoke | long-horizon credit assignment |

Reward is the hard part because manipulation success is **sparse, contact-rich, and easy to game**:

1. **Sparse reward** → exploration is hopeless in 20-DoF action space; you need the BC prior, reward shaping, or a curriculum.
2. **Reward hacking is the default**, not the exception: reward "gripper near object" → the robot hovers; reward "object lifted" → it throws the object; reward "button pressed" → it presses with the wrist camera.
3. **Dense shaping biases the solution**: a shaping term for "upright" makes the policy refuse to tilt the object even when the task demands it.
4. **Sim reward ≠ real reward**: in sim you can read object pose; on hardware you estimate it, and the estimator's error becomes a reward the policy exploits.
5. **Human preference rewards** (video-based) are cheap but noisy and reward *appearance* of success — a policy that looks confident while failing.

Practical recipe: **sparse task-success reward + BC regularisation (KL to the demo policy) + a learned success classifier for dense signal, validated by human spot-checks**. And always measure the reward on held-out real rollouts, because a reward that only goes up in sim is a reward that has been hacked.

**中文要点**：RL 在 IL 之上主要做**残差纠正、恢复、按成功率的微调**；奖励难在稀疏、易被 hack（靠近即悬停、抬起即乱抛）、稠密项会带偏解法，实际做法是稀疏成功奖励 + 对 BC 策略的 KL 正则 + 学到的成功分类器，且必须在真机留出集上验奖励。

---

### 4. A colleague wants to move the semantic layer to the cloud so you can use a much bigger model. Walk me through the latency budget.

**Anchor answer**

First separate the loops by rate — the argument only works for the slow one.

| Loop | Rate | Deadline | Can it go to cloud? |
| :--- | :--- | :--- | :--- |
| Joint servo / balance | 500-1000Hz | 1-2ms | never |
| Whole-body control / WBC | 100-500Hz | 2-10ms | never |
| Visuomotor policy (VLA) | 5-50Hz | 20-200ms | borderline |
| Semantic layer (object identity, task planning, dialogue) | 0.2-2Hz | 500ms-2s | yes, with fallback |

Cloud budget for the semantic layer:

| Term | Typical |
| :--- | :--- |
| Uplink (compressed RGB-D + state) | 20-60ms |
| Queueing at the inference service | 10-300ms (**the killer**) |
| Large-model inference | 150-800ms |
| Downlink (plan/semantic result) | 20-60ms |
| **Total p50 / p99** | **~0.3s / 1.5-3s** |

What that means physically: at 1 m/s a p99 of 2s means the semantic decision describes where the robot *was* 2 metres ago. For "which drawer is the mug in" that is fine; for "is that a person stepping into the aisle" it is unacceptable.

Requirements before saying yes:

1. **Asynchronous, non-blocking**: the semantic layer proposes, the local stack disposes. Never put a cloud call in the control path.
2. **Staleness accounting**: every result carries a timestamp; the local planner rejects results older than X and falls back.
3. **Local fallback that is genuinely usable** — a small on-robot model that handles the safety-critical subset, not a stub that throws.
4. **Connection loss policy**: hold, slow, or park. Decide per behaviour, and test it by pulling the network mid-task.
5. **Bandwidth/power**: streaming video at 10-30 fps from a battery-powered humanoid costs real energy and thermal budget — often the actual blocker, not latency.

Verdict: move *semantics and planning*, never *reactive control*. And keep the safety layer entirely onboard.

**中文要点**：先按回路分频——伺服/WBC 绝不能上云，VLA 临界，只有 0.2-2Hz 的语义层可以；云的杀手是排队（p99 1.5-3s），所以必须异步、带时间戳与过期拒绝、有真正可用的本地降级，并在断网时按行为分别降级。

---

### 5. Design the teleoperation data pipeline. Why is data collection the bottleneck in robotics rather than compute?

**Anchor answer**

Pipeline stages, each with its own failure:

| Stage | Content | Failure mode |
| :--- | :--- | :--- |
| Task spec | instruction text, success criteria, object set, start-state distribution | vague criteria → unusable labels |
| Rig | VR controller / leader-follower arms / exoskeleton / glove | kinematic mismatch → retargeting artifacts |
| Calibration | camera extrinsics, joint zero offsets, workspace frame | drift between sessions corrupts the dataset silently |
| Capture | multi-view RGB(-D), proprio, commanded + measured actions, force/torque, audio | **clock sync** is the classic bug |
| QA | dropouts, occluded cameras, operator competence, episode length | 20-40% of raw teleop is unusable |
| Annotation | language instruction, subtask segmentation, success/failure | post-hoc labelling does not scale |
| Storage | 30-60Hz state × N cameras × thousands of hours | egress and index cost |
| Conversion | episode → training shards, train/val split by *task and scene*, not by frame | leakage inflates eval |
| Lineage | rig version, operator id, calibration hash, code commit | irreproducible datasets |

Why data, not compute:

1. **The unit is robot-seconds with a human in the loop.** One hour of good demonstrations costs operator time + hardware time + reset time; a reset-heavy task yields maybe 30-60 usable episodes per hour.
2. **Coverage is combinatorial.** Object poses × lighting × backgrounds × grasps; you cannot buy your way out with more GPUs, only with more *diverse* hours.
3. **The marginal demo is worth less than the last.** The first 50 demos teach the task; the next 500 teach the long tail, and each is more expensive to obtain (harder scenes, more resets, more curation).
4. **Label quality is human.** Success/failure, language, subtask boundaries — all human judgement, and it caps dataset quality regardless of compute.

Levers that actually work: shared autonomous collection (robot executes, human only intervenes), simulation for the safe/easy 70%, replay-and-relabel instead of re-collect, and rigorous QA so you do not train on 30% garbage.

**中文要点**：管线九段，最易出事的是**时钟同步、标定漂移、QA 通过率（原始遥操作 20-40% 不可用）**；瓶颈在人因——机器人小时×操作员×复位时间，且覆盖度是组合爆炸，GPU 换不来多样性。

---

### 6. You have 10 hours of demonstrations for a new task and budget for 50 more. How do you decide what to collect, and what return do you expect?

**Anchor answer**

Do not collect uniformly. Run a **coverage/failure analysis** first, then spend on the gaps.

Step 1 — baseline. Train on the 10h, roll out 50-100 episodes on hardware, and classify failures:

| Failure class | Signal | What to collect |
| :--- | :--- | :--- |
| Never attempted (task ambiguity) | policy freezes / wrong subtask | demonstrations with clearer language + subtask segmentation |
| Off-distribution start states | fails only for objects at workspace edges | demos from those start states |
| Recovery failure | knocks object, then cannot recover | **recovery demos starting from the failure state** |
| Precision failure | grasp near-miss, insertion jam | slow, deliberate demos + more pose variety |
| Perception failure | wrong object, occlusion, glare | same task, varied lighting/occlusion, plus labelled negatives |

Step 2 — cluster & prioritise. Embed observed states (proprio + vision features), cluster, compare to demo coverage. The most valuable data is in the **thin clusters where the policy is confident but wrong** — high-visitation, high-error regions. Rare-but-catastrophic states get a fixed quota regardless of frequency.

Step 3 — budget the 50 hours roughly as: 40% targeted failure-state demos, 25% scene/object diversity, 15% recovery, 10% speed/style variation, 10% held-out validation scenarios you never train on.

Expected return, honestly stated: imitation learning has a steep then flat curve. The first 10h typically gets you most of the achievable success rate; 50 more buys maybe 10-20 points *if* they are targeted, and near-zero if they are more of the same. The measurable return is not average success — it is **tail failure reduction** and **fewer human interventions per hour**, which is what deployment economics care about.

If you can, spend part of the budget on **on-policy correction** (DAgger) instead of more clean demos: one hour of corrective data on the policy's own failure states beats five hours of fresh successful demos.

**中文要点**：先跑基线并把失败分类（未尝试/分布外起始/恢复失败/精度/感知），再按"高访问+高错误"的稀薄聚类定向采集；50 小时里约 40% 给失败态、15% 给恢复；收益主要是尾部失败下降与干预次数减少，不是平均成功率。

---

### 7. How do you evaluate a manipulation policy when every trial costs robot time and every failure has physical consequences?

**Anchor answer**

Treat evaluation as an **experiment design** problem, because the budget is the binding constraint.

| Decision | Choice |
| :--- | :--- |
| Scenario suite | fixed, versioned, seeded initial states; stratified easy/medium/hard |
| Success criterion | machine-checkable (object in target region + no drop + within time), plus a human spot-check for the ambiguous cases |
| Sample size | binomial: to distinguish 80% from 90% at 95% confidence you need ~200 trials *per condition*; most teams run 20 and over-read noise |
| Partial credit | subtask progress score, so a 10-trial run still tells you *where* it fails |
| Safety | physical rig (tether, padded cell), torque/velocity limits, e-stop operator, no live-human proximity during eval |
| Cost control | sim pre-screen (reject policies that fail obvious cases), automated reset, parallel cells, short episodes |

The statistical point interviewers want: **a policy at 18/20 and one at 16/20 are indistinguishable.** Report Wilson intervals, not percentages.

Design the suite to separate concerns:

1. **Capability suite** — does it do the task at all (loose tolerance, canonical objects).
2. **Robustness suite** — pose/lighting/background/payload perturbations from the training distribution's edges.
3. **Regression suite** — the 50 known-hard cases from past failures; never shrink it, only add.
4. **Stress/abuse suite** — adversarial object placement, partial occlusion, human interference. Run rarely, log everything.
5. **Efficiency** — time-to-complete, path length, number of regrasps; a slower policy is a worse policy in production even at equal success.

And record video + full state for every trial: the marginal cost of a trial is mostly the reset, so never waste one by not capturing why it failed.

**中文要点**：把评测当实验设计——固定分级场景集、机器可判的成功准则、按二项分布算样本量（20 次分不出 80% 与 90%）、报告 Wilson 区间；分能力/鲁棒/回归/压力四套，并记录每次的视频与状态。

---

### 8. You ship a policy to 300 robots. It works in the lab and degrades in the field. Debug it.

**Anchor answer**

Field degradation is almost never "the model is worse"; it is **the input distribution moved**. Debug systematically, not by retraining.

| Hypothesis | Signal to pull | Test |
| :--- | :--- | :--- |
| Sensor/calibration drift | per-robot extrinsics hash, camera exposure, IMU bias | compare calibration timestamp vs failure onset |
| Lighting/environment | failure rate by site, time of day, window orientation | stratify logs by site and hour |
| Payload/tooling | end-effector config, gripper wear, object supplier batch | cohort split by hardware revision |
| Thermal/compute | GPU clock, inference latency p99, fan/thermals | correlate failures with latency spikes |
| Network/telemetry | command dropouts, clock skew | timeline overlay |
| Operator/usage shift | task mix changed, new object SKUs, faster takt time | compare task distribution vs lab suite |
| Model/version skew | which policy binary is on each robot | fleet manifest diff |

Method:

1. **Fleet manifest + telemetry first.** If you cannot say which binary, calibration, and hardware revision each robot runs, stop and fix that — everything else is guesswork.
2. **Build the field replay set.** Ship full observation/action logs (downsampled if bandwidth-bound) and replay failing episodes in sim with the same policy. If it fails in replay, you can debug locally; if it succeeds, the difference is unlogged input (timing, calibration, sensor).
3. **Cohort comparison**: works-here/fails-there pairs with matched tasks. The diff between cohorts is the cause, and it is usually one of: camera mounting angle, floor material, ambient light, or inference latency.
4. **Canary rollback**: version the policy, keep the previous one deployable, and A/B by site — never a fleet-wide push.
5. **Distillation of the fix**: whatever it is (more augmentation, latency randomisation, calibration check at boot), add a **pre-flight self-test** on the robot so it fails loudly at boot rather than mid-task.

The lesson: lab success is measured on your lab's distribution. Deploy a *monitor* of input statistics (embedding drift, latency, calibration age) alongside the policy, not just success rate.

**中文要点**：现场退化几乎都是输入分布漂移，不是模型变差。先建"机队清单+全量遥测"，再做现场回放（回放能复现就能本地调），用"同任务的成功/失败机队对照"找差异——通常是相机安装角、地面材质、光照或推理延迟；并加开机自检+灰度回滚。

---

### 9. Design the safety architecture for a learned whole-body policy operating near people.

**Anchor answer**

Rule zero: **the learned policy is never the safety layer.** Safety is a separate, verifiable stack that constrains the policy.

| Layer | Mechanism | Guarantee |
| :--- | :--- | :--- |
| L0 Hardware | torque/current limits, joint stops, compliant covering, physical e-stop | bounded energy regardless of software |
| L1 Certified low-level controller | hard limits on velocity/acceleration/workspace, watchdog on command freshness | stops on stale or invalid commands |
| L2 Safety-rated perception | separate sensor channel (safety lidar / depth) with its own processor, not the learned perception stack | human detection independent of the model |
| L3 Speed-and-separation | ISO 10218 / TS 15066 style protective separation distance: `S = v·(t_reaction + t_brake) + margin` | distance scales with speed |
| L4 Action shield | safety filter / control barrier function: project the policy's action onto the safe set, or **clamp to zero** | policy can never command a collision |
| L5 Behavioural | degraded modes: slow, hold, retreat, park; audible/visual state indication | predictable failure behaviour |
| L6 Process | hazard analysis (STPA), fault injection, staged deployment, incident review | evidence, not vibes |

Design details that decide whether it works:

1. **The shield must be cheap and deterministic.** A 5ms safety filter can run at control rate; a second neural net cannot be trusted to certify the first.
2. **Clamping ≠ veto.** If the shield frequently zeroes the action, the policy is in a state it was never trained for — log it as a safety-relevant event and use it as training data.
3. **Human detection needs a fail-safe direction.** A false negative (missing a person) is catastrophic; a false positive (stopping for a shadow) is only annoying. Tune asymmetrically and monitor the false-negative rate with a separate channel.
4. **Whole-body adds fall risk.** A humanoid that stops abruptly can fall onto the person it was avoiding — the safe action may be "brace and step back", not "freeze". This must be in the shield's action set, not learned.
5. **Test the safety stack, not the policy**: inject faults (drop frames, corrupt commands, kill a camera, place a person at 0.5m) and require the specified response every time.

**中文要点**：学到的策略永远不是安全层——L0 硬件限矩/急停、L1 认证底层控制器、L2 **独立安全感知通道**、L3 速度-分离距离、L4 动作护盾（CBF/夹紧）、L5 降级模式；护盾要便宜确定，且"冻结"未必安全（人形可能跌倒），要用故障注入验证。

---

### 10. What is a vision-language-action model, and how is it different from an LLM with tools?

**Anchor answer**

A VLA is a single network mapping `(images, proprioception, instruction) → action tokens/continuous actions`, trained on robot trajectories. An LLM with tools maps `(text + maybe images) → a discrete API call`.

| | VLA | LLM + tools |
| :--- | :--- | :--- |
| Output | continuous action chunk, 5-50Hz | one of N tool schemas, once per turn |
| Training data | robot teleoperation (thousands of hours) | text/code, plus tool-use traces |
| Grounding | pixels → 3D geometry → contact | symbols → API semantics |
| Closed loop | 20-50Hz re-planning on its own visual input | seconds-to-minutes, external state read via tools |
| Failure mode | silent motor-level error, no introspection | wrong tool/args, visible in the transcript |
| Safety | needs a shield; unverifiable | permissions + schema validation |
| Strength | contact-rich manipulation, generalisation across objects | planning, code, long-horizon logic |

The crucial difference is **the action space**. An LLM's tool call is a symbolic commitment someone else executes and can reject. A VLA's output *is* the motion — there is no intermediate validator unless you build one, and errors are physical, immediate, and sometimes irreversible.

Why the boundary is blurring: VLAs are increasingly trained with web-scale VLM backbones, and "VLA + tool calls" hybrids exist — the model emits either `action` or `call_skill(...)`. In a real system the honest split is: **LLM/VLM plans and reasons; VLA executes contact-rich skills; a classical controller guarantees the dynamics; a shield guarantees safety.**

What a VLA still cannot do well: long-horizon tasks (minutes), counting/arithmetic, reading fine text, explaining its failure, and recovering from a state outside its data. If the task needs those, it needs a planner above it.

**中文要点**：VLA 是"图像+本体感知+指令 → 连续动作块"的单一网络，以 5-50Hz 闭环，训练数据是遥操作轨迹；LLM 工具调用输出的是**符号化的 API 承诺**，有校验与拒绝的机会。动作空间不同决定了 VLA 的错误是物理且不可逆的，所以必须有护盾和上层规划器。

---

### 11. Helix splits into a large slow model and a small fast one. Why not run a single end-to-end network?

**Anchor answer**

Because the two halves have **different frequency, data, and failure requirements**, and one network cannot be optimal for both.

| | System 2 (slow, large VLM) | System 1 (fast, small visuomotor) |
| :--- | :--- | :--- |
| Rate | 5-10Hz | 100-200Hz |
| Input | scene + instruction (semantic) | latest images + proprio |
| Role | "what to do": object, goal, subtask | "how": joint-level trajectory |
| Data | web-scale VLM pretraining + robot data | robot trajectories only |
| Failure cost | wrong subtask (recoverable) | jerky/unsafe motion (immediate) |

Arguments for splitting:

1. **Latency budget.** A 7B+ VLM at 100Hz is impossible on-robot within power/thermal limits. The fast loop must run every 5-10ms; the slow loop can afford 100-200ms.
2. **Pretraining leverage.** The semantic half benefits from internet-scale VLM data; the motor half would be swamped by it (text/image tokens are irrelevant to torque).
3. **Failure isolation.** If the slow model outputs a bad subtask, the fast controller still produces smooth, safe motion; a monolithic net failing means the robot twitches.
4. **Independent iteration.** You can swap the planner or retrain the controller without re-validating the whole stack — a huge safety-case advantage.
5. **Asynchrony.** The slow model can think while the fast one acts, which is exactly how biological motor control is organised.

Costs of splitting — be honest about them:

- **Interface mismatch**: the slow model's latent goal must be expressed in something the fast model was trained on (language subtask, target pose, latent token). A bad interface loses information and caps performance.
- **Coordination latency**: a fast policy may execute a stale goal for 200ms; needs a freshness check and an interrupt path.
- **Compounding**: slow model right, fast model wrong (or vice versa) — hard to attribute. You need per-level evaluation.
- **Joint training is harder** but often necessary: train the interface end-to-end even if you deploy it split.

**中文要点**：慢/快两层是**频率、数据、失败代价**都不同才拆的：7B VLM 不可能 100Hz 在机上跑，且语义预训练对力矩毫无用处；拆开的代价是接口信息损失与协调延迟，所以接口要能端到端训练，并给快层加目标新鲜度检查与中断路径。

---

### 12. Explain action chunking. Why predict a sequence of future actions instead of the next one?

**Anchor answer**

Action chunking: the policy outputs `a_{t..t+H-1}` (H ≈ 10-100) and executes a prefix of length `k` before re-planning (receding horizon). ACT, Diffusion Policy, and most current VLAs do this.

Why it works:

| Problem | How chunking helps |
| :--- | :--- |
| **Compounding error** | fewer decision points; a small error is corrected over H steps instead of being amplified at every step |
| **Multi-modality** | a chunk is a *consistent* trajectory segment; averaging two modes within a chunk is much less destructive than averaging single actions at a branch point |
| **Inference cost** | one forward pass covers H control steps, so a big model can run at 10Hz while the controller runs at 100Hz |
| **Temporal consistency** | smooth, non-jittery motion; no per-step noise |
| **Contact-rich tasks** | a grasp is a *sequence*; predicting it step-by-step fragments the dynamics |

Costs and the fixes:

1. **Reactivity.** Executing a full chunk of 50 steps means up to 0.5s of blindness to perturbation. Fix: execute only `k < H` (e.g. k=8 of H=50) and re-plan.
2. **Chunk-boundary discontinuities.** Naive re-planning jumps at each boundary. Fix: **temporal ensembling** — average overlapping predictions with weights (ACT), or blend the first steps.
3. **Delayed failure response.** If the object slips at step 2, you still have 48 steps of a now-wrong plan. Fix: a fast reflex/detector that preempts the chunk (slip detection, force threshold), not a bigger model.
4. **Latency hiding is not free**: if inference takes longer than `k` steps, the robot stalls. Chunk length must be chosen with the measured inference latency, not by tuning success rate alone.

Design rule of thumb: `H` ≈ one to two "subtask primitives" (a reach, a grasp), `k` ≈ `ceil(inference_latency / control_period) + 1`. Then verify by injecting perturbations mid-chunk and measuring recovery time — a policy that scores 95% on static trials can be unusable if it takes 0.6s to notice a slip.

**中文要点**：动作分块=一次输出 H 步并只执行前 k 步（滚动时域），好处是减少复合误差、缓解多模态平均、摊薄推理成本；代价是反应变慢、块边界跳变——用时间集成+快速反射抢占，且 H/k 要按实测推理延迟定。

---

## Waymo

### 1. In NumPy, compute minADE and minFDE for multi-modal trajectory predictions with variable-length ground truth. No Python loops.

**Reference implementation**

```python
import numpy as np

def min_ade_fde(pred, gt, gt_len):
    """
    pred:   (M, K, T, 2)  float  -- M scenarios, K modes, T steps, xy
    gt:     (M, T, 2)     float  -- padded ground truth
    gt_len: (M,)          int    -- number of valid steps per scenario (>=1)
    returns: ade (M,), fde (M,)  -- per-scenario, best mode
    """
    M, K, T, _ = pred.shape
    steps = np.arange(T)
    mask = steps[None, :] < gt_len[:, None]              # (M, T)

    err = np.linalg.norm(pred - gt[:, None, :, :], axis=-1)   # (M, K, T)

    # ADE over valid steps only
    ade = (err * mask[:, None, :]).sum(-1) / gt_len[:, None]  # (M, K)

    # FDE at the last *valid* step, per mode
    last = np.clip(gt_len - 1, 0, T - 1)[:, None, None]       # (M, 1, 1)
    last = np.broadcast_to(last, (M, K, 1))
    fde = np.take_along_axis(err, last, axis=-1)[..., 0]      # (M, K)

    return ade.min(axis=1), fde.min(axis=1)
```

**Key points**

1. **Mask, don't slice.** Variable lengths kill any `pred[:, :, :n]` approach; multiply by a `(M, T)` boolean mask broadcast over modes.
2. **`min` over modes happens per scenario, then `mean` over scenarios.** `np.min(err)` globally is a different (wrong) metric — it lets every scenario pick a different mode *and* conflates averaging order.
3. **FDE is at the last valid step, not index `-1`.** Padded timesteps contain garbage or zeros; taking `-1` silently rewards predicting "stationary at origin".
4. **Guard `gt_len == 0`** with `clip`/`maximum(1)`; otherwise you emit `nan` and it propagates into your dashboard.
5. **Broadcasting shape check**: `gt[:, None, :, :]` is the step that makes `(M,K,T,2)`; getting it wrong broadcasts over modes silently if `K == M`.
6. **Definition check in the room**: minADE can mean "best mode per scenario" (standard) or "best single mode across the dataset". State which one before coding.

**中文要点**：用 `(M,T)` 布尔 mask 处理变长、`min` 先按场景再按模态取、FDE 要取最后一个**有效**步而非 `-1`（padding 会让"原地不动"刷分），并防护 `gt_len==0`；答题前先确认 minADE 是"逐场景最优模态"。

---

### 2. Modular perception, prediction and planning, or end-to-end learned driving? Make the case, then tell me what you would actually build.

**Anchor answer**

The case for modular:

| Argument | Detail |
| :--- | :--- |
| Verifiability | each module has metrics and invariants ("no predicted trajectory enters a known static obstacle") |
| Safety case | you can argue about interfaces; you cannot argue about a 500M-parameter net's latent state |
| Attribution | a bad outcome traces to perception, prediction, or planning |
| Team scaling | separate owners, separate test suites |
| Debuggability | replay a scenario and inspect the intermediate objects |

The case for end-to-end:

| Argument | Detail |
| :--- | :--- |
| No interface bottleneck | the prediction output (K trajectories per agent) throws away information the planner needs |
| Optimises the real objective | modular training optimises proxies; a perception metric can improve while driving gets worse |
| Rare interactions | joint reasoning over agents emerges from data instead of being hand-specified |
| Less hand-engineering | no cost functions, no rule trees for every edge case |

What I would actually build: **modular in structure, learned within and across modules.**

1. Keep explicit, inspectable interfaces (objects, occupancy, predicted futures) because they are what the safety case is made of — but let modules share a **learned BEV representation** rather than hand-crafted features.
2. **Train the planner through a differentiable/approximate prediction model** so gradients reach the parts that matter, while keeping the deployed interfaces explicit.
3. Keep a **classical fallback + rule layer** (hard constraints: never cross a red light, never enter an occupied cell) outside the learned stack — this is the part regulators and safety engineers can audit.
4. Run a **shadow end-to-end model** on fleet data; promote it where it demonstrably beats modular on scenario suites, module by module, not as a big-bang replacement.

The honest position: "end-to-end" as a research direction is right; "end-to-end and nothing else" is incompatible with a certifiable safety case today. The winning architecture is a learned monolithic-ish core wrapped in verifiable constraints.

**中文要点**：模块化的价值是可验证、可归因、可分工，端到端的价值是没有接口信息瓶颈、直接优化最终目标；实际该建的是"结构模块化 + 模块内/跨模块学习"——共享 BEV 表示、用可微预测反传梯度，但保留显式接口与规则约束层供安全论证。

---

### 3. Design the output representation for a behaviour prediction model. What metrics would you gate it on?

**Anchor answer**

Representation: **per-agent multi-modal trajectories with calibrated probabilities**, plus an occupancy head.

```
for each agent a (up to N):
  modes k = 1..K (K ≈ 6)
    prob_k        : softmax over modes
    traj_k[0..H]  : (x, y) at 10Hz for H = 8s (80 steps)
  optional: per-mode (speed profile, heading, yaw-rate)
global: joint/marginal choice, occupancy grid (T × H × W), scene-level score
```

Design decisions that matter:

1. **Marginal vs joint.** Marginal per-agent predictions permit physically impossible combinations (two cars in the same cell). Add a **consistency/scene-level score** or a joint sampler for the planner.
2. **Fixed K with learned mode collapse control.** Without a winner-takes-all or anchor-based assignment, all K modes converge to the mean — the classic failure. Use anchors (clusters of training trajectories) or a diversity loss.
3. **Probability calibration** matters more than raw accuracy: the planner multiplies likelihood by cost. Use temperature scaling / isotonic calibration on a held-out set and gate on ECE.
4. **Coordinate frame**: agent-centric for learning, world-frame for the planner; document the transform or you will chase a 2m offset bug.
5. **Include a "no-op / stationary" mode explicitly** — otherwise stopped traffic is predicted as creeping.

Gating metrics:

| Metric | Why gate on it |
| :--- | :--- |
| minADE / minFDE (K=6) | standard, comparable |
| Miss rate @ 2m | tail behaviour, not averages |
| mAP / mode-selection accuracy | are probabilities meaningful |
| NLL / ECE | calibration |
| Off-road rate, collision rate vs GT | physical validity |
| Overlap rate | mode collapse detector |
| Downstream planner metrics (hard braking, comfort) | **the real gate** — prediction metrics can improve while driving worsens |

Gate per scenario cluster (intersections, cut-ins, pedestrians at crosswalks) with confidence intervals, not one global number.

**中文要点**：输出是"每 agent K 条轨迹 + 校准过的模态概率 + 占用栅格"，要显式建模静止模态、防模态塌缩（锚点/WTA）、区分 marginal 与 joint；门禁看 miss rate/ECE/off-road/overlap，最终以**下游规划指标**为准。

---

### 4. Where do vision-language models and foundation models genuinely help in an autonomy stack, and where are they a liability?

**Anchor answer**

Split by **offline vs onboard**; almost all near-term value is offline.

| Use | Where | Verdict |
| :--- | :--- | :--- |
| Auto-labelling (boxes, lanes, semantics) with VLM pre-labels + human verification | offline | big win — labelling is the long pole |
| Scenario mining by natural language ("construction with a flagger") | offline | big win — turns a taxonomy problem into a query |
| Data curation / dedup / captioning | offline | win |
| Sim agent behaviour ("an aggressive cyclist") | offline/sim | win, but validate realism |
| Map QA, doc/code assistance | offline | win |
| Passenger interaction, "why did you stop?", trip preferences | onboard, non-safety | win (with guardrails) |
| Unusual-scene semantic reasoning ("is that a person in a costume?") | onboard, advisory | maybe — as a *hint* to a classical stack |
| Directly producing control | onboard | liability |
| Certification evidence | anywhere | liability — nondeterministic, no guarantees |

Why onboard is a liability:

1. **Latency/power** — a 7B VLM at 10Hz does not fit a vehicle's thermal budget; running it slower means stale semantics.
2. **No guarantees** — no bounded error, no monotonicity, hallucination on rare inputs (exactly the inputs that matter).
3. **Non-determinism** complicates replay, root-cause, and the safety case.
4. **Distribution shift** — web-pretrained concepts do not cover "flooded underpass at dusk with a mattress in lane 2".
5. **Adversarial fragility** — text in the scene, stickers, projections; a language-conditioned model can be *instructed* by the environment.

The defensible architecture: VLMs as **offline force multipliers** (labels, mining, sim, tooling) and as **onboard advisors** whose output is fused as one low-weight sensor into a stack that still has hard constraints. The test to apply: *if the VLM is wrong, does the vehicle still behave safely?* If not, it is in the wrong place.

**中文要点**：VLM 的真价值几乎都在离线——自动标注、自然语言场景挖掘、数据策管、仿真行为；上车只能做"非安全的建议通道"。它上车的负债是延迟功耗、无保证、不可复现、分布外与对抗脆弱（环境里的文字能"指令"它）。判据：它错了车还安全吗？

---

### 5. Budget the compute and latency for the onboard stack. What breaks when a model gets bigger?

**Anchor answer**

Budget from the actuation deadline backwards. At 20 m/s, 100ms of end-to-end latency = 2m of travel — that sets the tone.

| Stage | Rate | Latency budget | Where it runs |
| :--- | :--- | :--- | :--- |
| Sensor capture + time sync | 10-20Hz | 10-20ms | HW timestamping |
| Perception (detect/segment/track) | 10Hz | 30-50ms | GPU/accelerator |
| Sensor fusion (lidar+camera+radar → BEV) | 10Hz | 10-20ms | GPU |
| Prediction (agents, 8s horizon) | 10Hz | 20-40ms | GPU, can be pipelined |
| Planning | 10Hz | 10-30ms | CPU/GPU |
| Control | 100Hz | 5-10ms | CPU, hard real-time |
| **Sensor-to-actuation total** | — | **~100-150ms** | with 2× safety margin |

Then the constraints people forget:

1. **Power and thermal**, not FLOPs, is the binding constraint. A 500W accelerator in a sealed trunk derates; sustained throughput can be 60-70% of peak, and it changes over a 30-minute drive.
2. **Memory bandwidth** dominates for BEV/large attention models; a model can be compute-light and bandwidth-bound.
3. **Tail latency (p99.9)**, not mean — a 40ms model with a 400ms outlier must not stall the pipeline. Pipeline stages with bounded queues and **drop-stale** semantics.
4. **Determinism and WCET** for the safety case: worst-case execution time must be bounded; dynamic batching is a liability here.

What breaks when a model gets bigger:

| Break | Symptom |
| :--- | :--- |
| p99 latency | oscillation, late braking, "hesitant" behaviour |
| Thermal budget | performance drops mid-drive, not at boot |
| Fallback absence | no smaller model to degrade to |
| Re-validation cost | every size change invalidates the safety evidence |
| Pipeline coupling | one slow stage forces every consumer to buffer → memory growth, jitter |

Mitigations: distillation into a fixed-budget student, input ROI cropping and cascades (cheap detector → expensive verifier), quantisation (INT8/FP8) with per-layer calibration, and a **latency-aware scheduler** that chooses model size per condition (more compute when stopped, less at speed).

**中文要点**：从执行器截止时间倒推——全程 ~100-150ms，控制环硬实时 5-10ms；真正约束是**功耗/散热**与 p99 尾延迟，不是 FLOPs。模型变大先崩在尾延迟与热降频，必须配蒸馏、级联、量化与按工况选模型的调度器。

---

### 6. You have hundreds of millions of fleet miles. How do you find and use the rare scenarios that matter?

**Anchor answer**

You cannot search what you cannot index, so the work is: **detect → tag → index → retrieve → prioritise.**

Detection (cheap triggers first):

| Trigger | Signal |
| :--- | :--- |
| Kinematic | hard braking, high jerk, evasive steer, near-miss (min TTC, PET, PRT) |
| System | disengagement, takeover, planner/model disagreement, MPC infeasibility |
| Uncertainty | perception OOD score, prediction entropy, tracker churn |
| Rule-based | unprotected left with oncoming traffic, construction cones, school zone, emergency vehicle |
| Semantic | VLM captions/clusters over sampled frames, natural-language queries |

Then:

1. **Auto-tag into an ontology** (scene type × agent type × weather × road geometry × interaction). Tags must be multi-label and versioned; without an ontology, "rare scenario" is just "weird video".
2. **Embed and index** every segment (BEV features + trajectory + caption). ANN search makes "find more like this" a query instead of a mining project.
3. **Deduplicate aggressively** — 100M miles contains millions of near-identical "lead vehicle braking". Cluster first, then sample per cluster; otherwise the rare class is drowned.
4. **Prioritise by risk, not rarity**: `priority = severity_if_failed × exposure × uncertainty`. A rare-but-harmless oddity is a distraction; a common-but-mishandled merge is the release blocker.
5. **Close the loop**: each mined scenario becomes (a) a sim regression test, (b) a labelled training example, (c) a metric in the release gate. If it does not land in one of those three, mining is a hobby.

What surprises people: the long tail is not only "weird objects" — most of it is **ordinary geometry under unusual timing** (a pedestrian arriving 0.4s earlier, a truck parked 3m further). That is why scenario mining must index *interaction timing*, not just object categories.

**中文要点**：先"能检测"再"能检索"——用运动学/系统/不确定性/规则触发器筛，用 VLM 打标签建可版本化的本体，向量化入 ANN 索引并按聚类去重；按"失效严重度×暴露×不确定性"排序，且每条挖出的场景必须落进仿真回归/训练集/发布门禁之一。

---

### 7. Design a system that finds driving segments similar to a given one across the entire fleet archive.

**Anchor answer**

A retrieval system, with three non-obvious design choices.

| Component | Design | Why |
| :--- | :--- | :--- |
| Segmentation | fixed 10-20s windows with 50% overlap, aligned on events | too short loses context, too long dilutes the signal |
| Representation | multi-view: (a) BEV occupancy/agent features, (b) ego+agent trajectories, (c) scene-graph tokens, (d) VLM caption embedding | different queries need different notions of "similar" |
| Index | ANN (HNSW/IVF-PQ) per view, plus metadata index (city, weather, time, vehicle, map id) | exact search over 10^8 segments is not viable |
| Query | seed segment → per-view kNN → **score fusion** (weighted sum / reciprocal rank fusion) → rerank top-100 with a heavier model or human | fusion fixes single-view false positives |
| Serving | precompute at ingest; query in seconds; export as a scenario bundle (logs + video + map) | analyst workflow matters as much as the model |

Choices that decide quality:

1. **Precompute at ingest, never at query time.** Encoding 100M segments on demand is impossible; do it once when the segment lands, store the vector next to the log pointer, and accept that model upgrades require a backfill job (plan for it: versioned embeddings, dual-write during migration).
2. **Similarity must be conditioned on what you care about.** "Similar geometry" (map + agent layout), "similar behaviour" (interaction pattern), and "similar appearance" (construction, weather) are different spaces. Offer them as separate query modes rather than one blended embedding — blended spaces are mediocre at everything.
3. **Temporal invariance.** The same interaction shifted 2s is the same scenario; align by event anchor before embedding, or you retrieve only identical phase.
4. **Hard negatives** matter: a parking lot and a road with the same occupancy pattern are not similar. Train/validate the encoder with contrastive pairs from human judgements, and measure retrieval precision@k on a labelled query set.
5. **Metadata filters first** to shrink the candidate set (city, time-of-day, road class) — ANN recall in a filtered subspace degrades, so use filtered-ANN indexes.

Cost control: store embeddings (~1-4KB/segment) separate from logs; keep the heavy reranker for the top-100 only; cache popular queries. And ship the export path, because the system's output is a *scenario someone can replay*, not a similarity score.

**中文要点**：按事件对齐切段、多视图表示（BEV/轨迹/场景图/ caption）分别建 ANN 索引再做分数融合重排，元数据先过滤；关键工程是**入库时预计算向量**（版本化+回填方案）与"相似性要分模式"（几何/行为/外观），输出必须可导出为可回放场景。

---

### 8. You are opening in a new city. Structure the safety case.

**Anchor answer**

A safety case is a **structured argument with evidence**, not a test report. Structure it as: claim → sub-claims → evidence → assumptions/imitations.

| Claim | Sub-claims | Evidence |
| :--- | :--- | :--- |
| ODD is defined and bounded | city-specific ODD: roads, weather, time, speed, construction | ODD document; geo-fence implementation; runtime ODD monitor |
| Hazards are identified | STPA/hazard analysis for local traffic patterns | hazard log with mitigations traced to tests |
| The stack handles local specifics | unprotected turns, roundabouts, turn-on-red rules, tram/bus lanes, signage language, cyclist density | scenario suite built from map + law + local drives; per-scenario pass rates with CIs |
| Map is adequate | lane geometry, signals, stop lines, school zones, HOV rules | map QA audit vs ground truth drives; freshness SLA |
| Behaviour is acceptable to locals | comfort, assertiveness, horn/hazard etiquette | shadow-mode + supervised drives; human ratings |
| Risk is bounded | collision/near-miss rate per mile vs benchmark with CI | miles driven in each phase, surrogate safety metrics |
| Failures degrade safely | MRC (minimal risk condition) reachable, remote assistance | fault injection, MRC drill results |
| Operations are ready | remote ops staffing, incident response, first responders training | runbooks, drills, SLA |

Phasing (this is the part interviewers probe):

1. **Map + data first**: drive the city with human drivers, build the map, mine scenarios, build the local scenario suite.
2. **Simulation**: replay + synthetic variants; gate on scenario pass rates.
3. **Closed course** with local geometry (roundabouts, tram tracks).
4. **Shadow mode**: system runs but does not actuate; compare to human driver, measure interventions needed.
5. **Supervised with safety driver**, expanding geography ring by ring, then driverless in a bounded geofence with weather/time restrictions.

Honest limitations to state: statistical confidence needs miles — you cannot claim "safer than a human" on 100k miles; the claim is "no observed unacceptable-severity events, with these metrics at these CIs, under this ODD, with these mitigations". And local *law* is a first-class requirement: right-turn-on-red, cyclist passing distance, and school-bus rules differ and must be encoded, not learned from a handful of demos.

**中文要点**：安全论证是"主张→子主张→证据→假设与局限"的结构，覆盖 ODD 定义、STPA 危害、本地交规与几何、地图质量、降级/MRC、运营响应；落地按"地图与数据→仿真→封闭场地→影子→安全员监督→限定区域无人"分级，并如实声明统计置信度的不足。

---

### 9. Disengagement rate is a weak safety proxy. How would you actually measure whether the Driver is safe enough to ship?

**Anchor answer**

Disengagement rate fails on five counts:

1. **Subjective**: the safety driver's threshold varies by person, mood, and route difficulty.
2. **Denominator games**: easy geofenced miles inflate the rate; harder ODD lowers it without any change in safety.
3. **Reporting inconsistency**: some teams log "proactive takeover", others only "imminent risk".
4. **It counts interruptions, not harm**: 1000 unnecessary disengagements look worse than 5 near-misses that were never logged.
5. **It says nothing about the counterfactual**: would the system have crashed, or just driven slowly?

Build a **layered evidence model**:

| Layer | Metric | Role |
| :--- | :--- | :--- |
| Outcome | collisions per million miles, severity-weighted; injury rate | the ground truth, but too rare to gate on alone |
| Surrogate | near-miss rate: min TTC < 1.5s, PET < 2s, hard braking, evasive steer; **severity-weighted** | frequent enough to be statistically useful |
| Counterfactual | replay each disengagement/intervention in sim with the same initial state; did the system actually fail? | separates necessary from unnecessary |
| Scenario | pass rate on a fixed safety-relevant suite, per cluster, with CIs | leading indicator, regression-proof |
| Fault tolerance | fault-injection pass rate (sensor loss, compute slowdown, actuator fault) | tail risk |
| Comfort/legality | hard-braking rate per 1k miles, traffic violations, remote-assist load | operational quality |

How to gate a release:

1. **Absolute risk thresholds per ODD segment**, not fleet averages: e.g. no cluster may exceed X near-misses per 10k miles with a 95% upper bound below the human benchmark.
2. **Confidence intervals everywhere** — and require enough exposure per cluster to make the interval informative.
3. **Counterfactual simulation of every intervention** as a required artifact; unexplained interventions block release.
4. **Pre-registered metrics** before the test campaign, so nobody picks the flattering one afterwards.

The one-liner: *disengagement rate measures the driver's patience; near-miss rate with counterfactual replay measures the system's competence.*

**中文要点**：脱离率主观、可被里程数操纵、只数打断不算伤害，且没有反事实。要做分层证据：碰撞/严重度加权近失事率（TTC/PET）、**每次接管的仿真反事实回放**（是否真失败）、固定场景集的通过率与置信区间、故障注入；门禁用按 ODD 分段的绝对风险上界。

---

### 10. How do you build a simulator you would trust to gate a release?

**Anchor answer**

| Axis | Requirement | How to earn trust |
| :--- | :--- | :--- |
| **Sensor fidelity** | lidar/camera/radar renderings realistic enough that perception behaves the same | replay real sensor logs; compare detection/tracking metrics sim vs real on the same scene |
| **Dynamics fidelity** | vehicle responds like the real one (tyres, brake lag, actuator delay) | system identification against instrumented test-track manoeuvres |
| **Agent behaviour** | other actors behave like real traffic, including rare aggression/errors | behaviour model trained on fleet data; validate distributions (gaps accepted, speeds, TTC) against real |
| **Scenario coverage** | matches the ODD × hazard taxonomy | coverage matrix over scenario ontology, reported as gaps not totals |
| **Reproducibility** | same seed → same result, forever | versioned sim, seeds, and assets; determinism tests in CI |
| **Correlation with reality** | sim failures predict real failures | replay known real incidents; measure detection rate (must catch them) and false-alarm rate |

The two tests that actually decide trust:

1. **Incident replay**: take every real safety-relevant event of the last year, replay it in sim with the same stack. If sim does not reproduce the failure, the simulator cannot gate anything — it is a demo.
2. **Blind A/B**: run a stack version in sim and on a closed course with matched scenarios, and check that the *ranking* of versions agrees. Ranking agreement matters more than absolute numbers.

Governance, which is where most sim programmes actually fail:

- **Scenario suite is versioned and frozen per release**; adding a scenario mid-campaign invalidates comparison.
- **No training on the gate set.** Leakage is the default because mining and gating use the same archive — keep a held-out, access-controlled suite.
- **The metric must be pre-registered** (which cluster, which threshold, which CI).
- **Human review of a sample of sim passes**, because simulators are optimistic in known ways (agents are polite, sensors are clean, maps are perfect).

**中文要点**：可信仿真要四条同时成立——传感器/动力学/交通体保真、覆盖矩阵、确定性可复现、**与现实的排序相关性**；两个硬测试是"重放过去一年的真实安全事件必须复现失败"和"仿真与封闭场地的版本排序一致"，再加门禁集与训练集隔离。

---

### 11. Two days before a release decision, simulation shows a 15% increase in hard-braking events in one scenario cluster. Walk me through what you do.

**Anchor answer**

Do not average it away, and do not reflexively delay. Run a **triage ladder**, with a decision at each rung.

Rung 1 — is the signal real?

| Check | Disqualifier |
| :--- | :--- |
| Sample size / CI on the 15% | if the cluster has 40 events, 15% is noise — report the interval |
| Metric definition changed? | a new hard-brake threshold or a sim version bump explains many "regressions" |
| Scenario suite changed? | newly added scenarios are harder by construction |
| Exposure shift | more traffic density in the sim's agent model |

Rung 2 — cause. Bisect the release's changes (model, planner cost weights, perception thresholds, map update) against the frozen suite. Reproduce one failing case, pull the trace, look at the planner's cost breakdown and the perception inputs. Typical findings: a prediction model that now outputs a slightly different mode ordering; a comfort weight that got retuned; a perception recall gain that introduces phantom obstacles.

Rung 3 — severity. A 15% rise in hard braking in a cluster of *parking-lot creep* is a comfort regression; in *highway merge at 100 km/h* it is a rear-end risk. Map the cluster to real-world consequence and to real-fleet exposure (how many of these does the fleet encounter per 1k miles?).

Rung 4 — decide, explicitly:

1. **Root-caused and benign** → ship, document, add a monitoring alert.
2. **Root-caused and fixable in hours** → fix, re-run the suite, ship.
3. **Not understood** → do **not** ship into that cluster. Options: geofence restriction, disable the feature that triggers it, ship the previous planner for that ODD slice, or delay.
4. **Understood but inherent trade-off** (e.g. more conservative merging to fix a worse failure) → make it an explicit written trade-off with the safety owner's sign-off, not a silent pass.

Non-negotiables: pre-registered thresholds and an owner who can say no; the decision recorded with evidence; and the failing cluster added to the permanent regression suite.

**中文要点**：先问"信号真不真"（样本量与置信区间、指标/场景集/仿真版本是否变），再二分定位改动、看 planner cost 与感知输入，然后**按场景映射真实后果**；处理只有四种：根因良性则发、能几小时修则修、不懂就**限制 ODD 或延期**、本质权衡则书面签字。绝不允许"平均掉"。

---

### 12. Why carry lidar, radar and cameras rather than cameras alone? Where would you fuse them?

**Anchor answer**

Because the three have **different, largely independent failure modes** — and a safety case needs independence, not just average accuracy.

| Sensor | Strength | Fails at |
| :--- | :--- | :--- |
| Camera | dense semantics: signs, lights, lanes, gestures, colours | low sun/glare, night, rain on lens, fog; no direct depth (monocular depth is learned and biased) |
| Lidar | accurate geometry, works in darkness, direct range | rain/fog/spray/snow (backscatter), retroreflective/black targets, cost, moving-parts reliability |
| Radar | direct Doppler velocity, penetrates rain/fog/dust, long range, cheap | low angular resolution, clutter/ghosts, poor static discrimination, height ambiguity |

The decisive argument is **common-mode failure**: cameras alone can be blinded by a low sun or a dirty windscreen — both cameras at once. Radar keeps seeing through that. Conversely, radar alone cannot read a stop sign or a hand signal. Redundancy only helps when the physics differ.

Fusion placement:

| Level | Where | Trade-off |
| :--- | :--- | :--- |
| Early / raw | point-painting lidar into camera images, or unified voxel/BEV | max information, needs precise time sync + extrinsics, brittle to mis-calibration |
| Mid / feature | BEV feature fusion (camera features lifted to BEV, concatenated with lidar/radar BEV) | current default: robust, learnable, tolerant of partial sensor loss |
| Late / object | track-level fusion with covariance-weighted association | interpretable, easy to degrade gracefully, loses raw signal |
| Post-object verification | radar/lidar confirming or vetoing a camera-only detection | cheap safety net; keep it separate from the learned fusion |

Practical notes: fuse in **BEV** with explicit per-sensor masks so the model trains with dropout and degrades when a sensor dies mid-drive (test it: kill the lidar at 60 km/h and require a defined behaviour). Time sync at the hardware level (PTP/GPS), extrinsics re-validated online — a 0.2° extrinsic drift silently biases every fused range estimate, and it is the most common real-world fusion bug.

**中文要点**：三种传感器失效物理不同，价值在于**共模失效独立性**（低太阳/脏风挡会同时致盲所有相机，雷达仍能测速距）；融合主流在 BEV 特征层，需硬件级时间同步、在线外参校验，并用随机丢传感器训练以保证降级行为。

---

## 附：这 24 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 机器人学习深度 | Figure 1（BC 失效）、2（sim2real）、3（RL 与奖励）、12（动作分块） |
| 机器人系统与数据 | Figure 5（遥操作管线）、6（采集预算）、7（策略评测）、8（机队退化） |
| 机器人架构与安全 | Figure 4（上云延迟）、9（安全架构）、10（VLA vs 工具）、11（双系统） |
| 自动驾驶基础 | Waymo 2（模块化 vs 端到端）、3（预测表示）、12（多传感器融合） |
| 自动驾驶工程 | Waymo 1（NumPy 指标）、5（车载算力预算）、7（相似片段检索） |
| 安全论证与发布 | Waymo 8（新城市安全案）、9（脱离率之外）、10（可信仿真）、11（发布前回归） |
