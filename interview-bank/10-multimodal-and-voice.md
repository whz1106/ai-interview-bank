# 10 · Multimodal, Speech and Voice AI

> 10 题。出处：`README.en.md` § Multimodal, Speech and Voice AI。

---

### 1. How do vision-language models get images into an LLM: projector, cross-attention, or native tokens?

**Asked at**: Meta · Alibaba · Tencent
**Source**: https://outcomeschool.com/blog/multimodal-ai · https://outcomeschool.com/blog/decoding-vision-transformer-vit

**Anchor answer**

Three families, distinguished by **where the modalities meet**.

| Family | Mechanism | Example shape | Strength | Weakness |
| :--- | :--- | :--- | :--- | :--- |
| Projector / adapter | ViT patches → MLP → prepended into the LLM token stream | 576 visual tokens at 336px (24×24 patches) | simple, reuses a frozen LLM, trains in days | token count explodes with resolution |
| Resampler / cross-attention | a small set of learned queries (Q-Former) or gated cross-attn layers read the vision features | 32–64 tokens | constant budget regardless of resolution | the bottleneck loses OCR/fine detail |
| Native / early fusion | image patches and text tokens share one transformer and one objective, sometimes discretized | any | best deep fusion, one model | needs massive multimodal pretraining; discrete-token variants lose detail |

The practical decision is a **token-budget decision**. A single 336px image at 576 tokens is fine; a scanned invoice at 1344px with AnyRes tiling is ~2,880–5,760 tokens, which is what actually drives cost and latency.

Two hard-won details: the projector should be trained *after* the vision encoder is aligned, and OCR quality tracks **pixel budget**, not model size — downsampling patches to save tokens reliably destroys small text. Cross-attention architectures (Flamingo-style) preserve the LLM's text behaviour better under interleaved data, but are awkward to serve because every layer gains a branch.

**中文要点**：三条路线——projector 把 patch 投成 token 拼进序列（简单、token 随分辨率爆）、resampler/cross-attn 用固定数量的 query 压缩（省 token 但丢 OCR 细节）、native 早期融合（融合最深但预训练代价大）。真正的取舍是 token 预算。

---

### 2. What changes when you move from images to video?

**Asked at**: Meta · ByteDance

**Anchor answer**

Everything that worked per-frame still works; what breaks is **time**.

| Dimension | Image | Video |
| :--- | :--- | :--- |
| Token count | 576 per image | ×frames — 1 min at 1 fps × 576 ≈ 34k tokens |
| Input tensor | $[B,C,H,W]$ | $[B,C,T,H,W]$ → 3D patches / tubelets |
| New modelling need | — | temporal order, motion, causality |
| New failure | — | counting, event ordering, "what happened before X" |
| Extra stream | — | audio track, subtitles/ASR |

The engineering responses:

1. **Sample, don't stream.** Fixed frame counts (8/16/32) or adaptive keyframe selection on shot boundaries; frame-rate is a hyperparameter you tune against the question type.
2. **Pool along time.** Spatiotemporal attention with temporal pooling, or tubelet embeddings that merge $2\times2\times2$ patches.
3. **Make time explicit.** Interleave timestamps into the text prompt (`[00:12]`), which measurably fixes ordering questions.
4. **Hierarchy for long video.** Shot-level summaries → clip embeddings → retrieve relevant windows instead of stuffing an hour into context.

The counter-intuitive finding: most video-VLM errors are not perception errors but **temporal aggregation** errors — the model sees each frame correctly and still answers "the person entered after the door closed". Evaluate temporal ordering and counting separately from per-frame VQA, or you will not see this.

**中文要点**：从图像到视频，变的是 token 数（×帧数）和"时间"这一维：要采样/关键帧、做时空池化、把时间戳写进 prompt、长视频走分层摘要+检索。多数错误不是看不清单帧，而是时序聚合错——评估要单独测顺序和计数。

---

### 3. Budget the latency for a real-time voice agent: VAD, ASR, LLM, TTS, network. Where does the time go?

**Asked at**: Sarvam AI · ElevenLabs
**Source**: https://outcomeschool.com/blog/design-a-real-time-voice-ai-agent

**Anchor answer**

Target: **<800 ms** from end of user speech to first audio out. Above ~1.5 s the conversation feels broken and people start talking over the agent.

| Stage | Typical | What dominates |
| :--- | :--- | :--- |
| VAD / end-of-turn | 200–600 ms | **silence timeout** — the fixed cost nobody budgets |
| ASR finalisation | 100–300 ms | wait-for-final vs partial |
| LLM TTFT | 300–800 ms | prompt length + prefill, not decode |
| TTS first chunk | 100–300 ms | text length needed before first phoneme |
| Network / jitter buffer | 50–200 ms | RTT + playout buffer |

The insight: **end-of-turn detection is the largest controllable chunk of latency**, and it is pure waiting. Cutting it needs a semantic turn model (predict "they're done" from prosody + partial text) instead of a 500 ms silence timer — this alone can buy 300–400 ms.

Standard mitigations: stream ASR partials and start the LLM on a stable partial; generate the reply sentence-by-sentence so TTS can start on sentence one; pre-warm TTS and keep the websocket open; co-locate stages in one region; cache the system prompt (prefix cache) so TTFT does not include the persona.

Never optimise the average: budget against **p95**, because a single 3 s outlier ruins perceived quality more than 200 ms on the median.

**中文要点**：目标首音 <800ms。最大可控开销是"判停"（等静音 200~600ms），用语义判停替代固定静音阈值能省 300ms+；其余靠流式 ASR、按句切分喂 TTS、预热连接、前缀缓存。按 p95 预算，不看均值。

---

### 4. Design barge-in / interruption handling for a voice agent

**Asked at**: ElevenLabs
**Source**: https://outcomeschool.com/blog/design-a-real-time-voice-ai-agent

**Anchor answer**

Barge-in = the user starts talking while the agent is talking, and the agent must **stop fast, stop cleanly, and know what it said**.

Requirements: stop audible output in <150 ms; do not self-trigger; keep a coherent history afterwards.

| Layer | Design |
| :--- | :--- |
| Acoustic | AEC (echo cancellation) so the agent's own TTS doesn't register as speech; VAD on the post-AEC signal |
| Trigger rule | require N ms of *speech-like* energy **plus** an ASR partial that is not a backchannel ("uh-huh", "yeah") — backchannels should not interrupt |
| Action | flush the TTS playout buffer, cancel the in-flight LLM request, drop queued TTS chunks |
| State | mark the assistant turn `interrupted` with the text actually spoken (from the TTS chunk log, not the generated plan) |
| Recovery | treat the interruption as a new user turn; optionally re-offer the cut sentence at the end |

The subtle failure: two independent cancellation paths (audio out and token generation) that get out of sync. You must **truncate history to what was actually played**, otherwise the model believes it already said the sentence it never emitted, and the next answer references a fact the user never heard.

Measure false-interrupt rate (agent stops when it shouldn't) and stop latency separately; they trade off directly through the trigger threshold.

**中文要点**：barge-in 三件事——150ms 内真停、别被自己的 TTS 触发（必须 AEC + 排除"嗯/对"这类 backchannel）、历史按"实际播出的文字"截断。最容易错的是只取消生成不取消播放，或反之，导致模型以为自己说过没说出的话。

---

### 5. Cascaded ASR+LLM+TTS versus native speech-to-speech: argue both sides

**Asked at**: ElevenLabs
**Source**: https://outcomeschool.com/blog/design-a-real-time-voice-ai-agent

**Anchor answer**

| | Cascade (ASR→LLM→TTS) | Native S2S |
| :--- | :--- | :--- |
| Latency | 3 sequential hops, ~600–900 ms floor | one model, ~200–400 ms achievable |
| Prosody / emotion | flattened — the LLM never hears it | preserved end-to-end |
| Turn-taking | rigid, VAD-driven | can do full-duplex, backchannels, overlap |
| Debuggability | full text transcript at every hop | opaque; needs an ASR side-channel anyway |
| Controllability | swap any component, inject a system prompt, log for compliance | monolithic; prompt/adherence weaker |
| Failure mode | ASR error is unrecoverable — the LLM answers the wrong question confidently | fluent hallucinated speech, weak tool calling |

Argument for cascade: it is the only shape that satisfies regulated industries (full transcript, deterministic redaction, auditable), and component iteration is cheap — upgrading the LLM is a config change, not a retrain.

Argument for native: half the latency budget disappears, and paralinguistics — hesitation, sarcasm, "mm-hm" — survive. Those are exactly what make a voice agent feel human, and no amount of TTS prosody modelling recovers them once ASR has thrown them away.

The pragmatic 2026 answer is **hybrid**: a native audio front-end for turn-taking and expressiveness, with a text LLM (or text tokens) as the reasoning backbone, plus ASR running in parallel for logs and safety filters.

**中文要点**：级联胜在可控、可审计、组件可换、延迟是三级串行；端到端胜在保留韵律/情绪、能做全双工、延迟低一半，但不可控且爱编造。生产多用混合：原生音频前端管听感与判停，文本模型做推理主干，ASR 旁路留日志与安全过滤。

---

### 6. How do you evaluate ASR quality beyond WER, and TTS quality when there is no single correct output?

**Asked at**: ElevenLabs

**Anchor answer**

WER is a blunt average — text normalization choices can swing it more than model quality.

ASR, layered:

| Metric | Catches |
| :--- | :--- |
| Entity WER (names, drugs, dosages, IDs) | the errors that actually cost money; often 3–5× overall WER |
| Semantic WER / embedding distance | paraphrase-equivalent recognition |
| Inverse text normalization accuracy | "two fifteen pm" → `14:15`, currency, decimals |
| Hallucination rate on silence/noise | Whisper-family fluent invention on non-speech |
| Partial stability | how often streaming hypotheses rewrite (UX) |
| Stratified WER | accents, code-switch ratio, far-field, child speech |

TTS has no reference, so evaluate along three axes: **intelligibility** (round-trip ASR CER on hard text — entity lists, numbers, OOV proper nouns), **naturalness** (MOS/CMOS, ABX preference, prosody on questions and lists), and **speaker fidelity** (embedding cosine to the target voice). Add production metrics: time-to-first-audio, RTF, and the **catastrophic-failure rate** — the fraction of utterances with a mispronunciation or truncation, which users remember far more than average MOS.

Always report confidence intervals and keep a frozen regression set per language; MOS without a paired test tells you nothing.

**中文要点**：WER 会被文本归一化带偏。ASR 要分层看实体 WER、语义 WER、ITN、静音段的幻觉率、流式稳定性、分口音分层。TTS 没有唯一答案，就测可懂度（回环 CER）、自然度（MOS/CMOS/ABX）、音色相似度，外加首音延迟与"灾难性错误率"。

---

### 7. How do you handle code-switching and accents in a production ASR system?

**Asked at**: Sarvam AI · Abridge

**Anchor answer**

Treat them as two different problems that get conflated.

**Code-switching** (Hinglish, Singlish, Spanglish) breaks the assumption of one language per utterance:

1. Train on **real** CS audio. Concatenating monolingual clips does not produce natural switch points and the model learns the wrong prosodic cues.
2. Do not force a single language ID per utterance — use frame/token-level LID or a unified multilingual tokenizer with language-agnostic graphemes (Devanagari + Latin in one vocab).
3. Handle transliteration: the same name appears as `Rahul` and `राहुल`; normalize both ways at the LM level.
4. Evaluate WER **per language segment**, plus a switch-boundary metric; aggregate WER hides that one language is fine and the other is broken.

**Accents** are a distribution problem:

- Augment with tempo/speed perturbation, SpecAugment, room impulse responses, and noise — cheap and effective.
- Lightweight accent adapters or per-cluster LoRA beats one giant model when you have labelled accent clusters.
- **Contextual biasing** (hotword/shallow-fusion list of names, drug names, product SKUs) is usually the single highest-ROI fix in production — it converts open-vocabulary guessing into constrained decoding.

Report a stratified dashboard by accent × CS ratio; a model that improves the average while regressing the worst slice is a regression.

**中文要点**：语码转换要靠真实混合语料 + 帧级 LID/统一多语词表，别按整句定语言；转写变体（Rahul/राहुल）在 LM 层归一。口音是分布问题：加噪/变速增广、按簇 LoRA、再加热词偏置——后者通常 ROI 最高。评估必须分口音×混说比例分层。

---

### 8. Explain streaming TTS chunking and jitter-buffer sizing

**Asked at**: ElevenLabs

**Anchor answer**

Streaming TTS trades **prosody** against **time-to-first-audio**, and the chunker is where that trade is made.

Chunking rules:

1. **Split on prosodic boundaries** — sentence, then clause (comma/colon), never at a fixed character count. Fixed-size chunks chop words and flatten intonation; listeners rate it clearly worse.
2. **First chunk small, later chunks larger.** Emit the first clause (~10–20 words) to hit a low TTFB, then accumulate longer chunks where prosody matters more.
3. **Carry lookahead.** Each chunk needs trailing context (punctuation + a few words) for the model to place the final fall; discard the audio generated for the lookahead.
4. **Overlap and cross-fade** ~20 ms at boundaries to avoid clicks and phase discontinuity; keep speaker state (embedding, prosody) across chunks or the voice drifts.

Jitter buffer:

| Size | Symptom |
| :--- | :--- |
| too small | underruns → glitches, stutter |
| too large | added latency, worse barge-in |
| adaptive | target ≈ p95 network jitter + decode variance, shrink slowly, grow fast |

Start at 2–3 frames of 20 ms, adapt from observed arrival jitter, and log underrun rate as a first-class SLI — it is invisible in average latency but deafening to the user.

**中文要点**：按句/从句切，不按固定字数；首块小以抢 TTFB，后续块放大以保韵律；带 lookahead 并做 20ms 交叉淡化。抖动缓冲目标 ≈ p95 抖动 + 解码方差，只慢缩快扩；underrun 率要当 SLI 盯。

---

### 9. Design a diarisation system and explain how you attribute roles, not just clusters

**Asked at**: Abridge

**Anchor answer**

Diarisation answers "who spoke when"; role attribution answers "which one is the doctor". They are different systems.

Pipeline:

1. **VAD + segmentation** (energy/neural VAD, 1–2 s windows with overlap).
2. **Speaker embeddings** per window (ECAPA-TDNN / x-vector), L2-normalized.
3. **Clustering** — AHC or spectral for unknown count, then VB-HMM resegmentation to smooth boundaries.
4. **Overlap handling** — separate overlapped regions (target-speaker extraction) or at least flag them; overlap is the dominant DER error source.

Role attribution needs signals outside the clustering:

| Signal | Use |
| :--- | :--- |
| Channel metadata | two-channel telehealth recording → role by channel, done |
| Self-introduction / address terms | "I'm Dr. Lee", "thanks for seeing me today" |
| Turn-taking asymmetry | the asker of assessment questions, the one who summarises |
| Domain lexicon | clinical phrasing, prescription talk |
| LLM over labelled transcript | cheap, surprisingly strong; ask for role per speaker with evidence spans |
| Enrollment voiceprint | known clinician → match by embedding |

Two production notes: DER of ~10% can be perfectly acceptable for content extraction but fatal for role labels, so measure **role accuracy** as its own metric; and never let a role assignment overwrite speaker identity — keep clusters stable and map roles on top, or a single bad LLM call corrupts the transcript.

**中文要点**：diarisation 只回答"谁在什么时候说"，角色归属是另一套系统：靠声道元数据、自我介绍、提问/总结的会话不对称、领域词、以及带说话人标签的 LLM 推断。DER 10% 对内容够用但对角色标签致命，所以要单独测角色准确率，且角色映射不能覆盖说话人聚类。

---

### 10. How would you build multimodal retrieval over images, video and text in one index?

**Source**: https://outcomeschool.com/blog/how-do-image-embeddings-work

**Anchor answer**

"One index" is a query-routing decision before it is an embedding decision.

Architecture:

| Layer | Choice |
| :--- | :--- |
| Encoders | text encoder; CLIP/SigLIP-style joint image-text encoder; video = keyframe embeddings + ASR transcript + shot boundaries |
| Derived text | caption + OCR + ASR per asset — often more retrievable than the raw visual vector |
| Store | one collection with **named vectors** per modality (Qdrant/pgvector/Weaviate), plus metadata: modality, timestamp, source, permission |
| Query | text query → text vector + image vector in the joint space; hybrid BM25 over OCR/captions; fuse by RRF |
| Rerank | VLM/LLM rerank on the top 50, with the frame or crop in context |

Key design points:

1. **Do not rely on CLIP alone for documents.** Visual embeddings are weak on text-in-image (charts, slides, screenshots); OCR + caption text usually wins there, and the two are complementary — fuse, don't choose.
2. **Time is metadata, not a vector.** For video, index shot-level units and store `[start,end]`; retrieval returns a window, and the VLM reads that window.
3. **Chunk asymmetry.** Text chunks are ~300 tokens; images are one unit; video shots are one unit. Store the unit size so fusion does not systematically favour one modality.
4. **Evaluate per modality** with recall@k on hard negatives (same scene, different frame), and check cross-modal queries separately — average recall hides a broken image path.

**中文要点**：先定路由再定向量：文本/图像用联合空间编码器，视频按镜头切并保留时间戳，同时为每个资产建 caption+OCR+ASR 的派生文本索引，用 RRF 融合再上 VLM 重排。文档类场景别只靠 CLIP——图内文字要靠 OCR；评估要分模态算 recall@k。

---

## 附：这 10 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 真做过 VLM | 1（图像注入）、2（视频时序） |
| 真做过语音产品 | 3（延迟预算）、4（barge-in）、8（流式切分） |
| 架构判断力 | 5（级联 vs 端到端）、10（统一索引） |
| 数据/评估功底 | 6（WER 之外）、7（口音与混说）、9（角色归属） |
| 只看过 demo | 从 2 和 4 问起，最容易被问穿 |
