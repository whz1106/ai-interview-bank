# 47 · ElevenLabs

> 8 题。出处：`README.en.md` § ElevenLabs。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Write a service that proxies streaming TTS to a browser and cancels cleanly when the user navigates away.

**Reference implementation**

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx

app = FastAPI()
UPSTREAM = "http://tts.internal/v1/text-to-speech/stream"

@app.get("/tts")
async def tts(request: Request, text: str, voice: str = "aria"):
    # One client per request: cancellation must close the upstream socket.
    client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))

    async def body():
        try:
            async with client.stream(
                "POST", UPSTREAM,
                json={"text": text, "voice": voice, "format": "mp3_44100_128"},
                headers={"accept": "audio/mpeg"},
            ) as upstream:
                upstream.raise_for_status()
                async for chunk in upstream.aiter_raw():   # no buffering
                    if await request.is_disconnected():
                        break                              # cancel upstream
                    yield chunk
        finally:
            await client.aclose()                          # releases the stream

    return StreamingResponse(
        body(),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",   # nginx: do not buffer the stream
            "Content-Encoding": "identity",
        },
    )
```

**Key points**

1. **Never accumulate the body.** `aiter_raw` + `yield` keeps memory flat; `await resp.read()` would defeat streaming and add the whole clip's latency.
2. **Cancellation is the hard part.** The browser closing the socket cancels the ASGI task; the `finally` block closes the upstream connection, which is what actually stops billing for the remaining audio.
3. **Do not retry mid-stream.** Once bytes are sent you cannot replay them; retry only before the first chunk, and pass a client-supplied request id so the TTS backend de-duplicates.
4. **First-byte latency**: start upstream generation as soon as the first sentence arrives from the LLM — buffer to the first clause boundary, not the full answer. Chunking on punctuation costs prosody, so keep a small lookahead.
5. **Intermediaries**: any proxy that buffers (nginx, some LBs, gzip middleware) silently converts this into a batch endpoint. Disable buffering and compression explicitly.
6. **Backpressure**: if the client is slow, `yield` blocks and the upstream buffer grows — set a bounded read timeout and kill the stream rather than OOM.

**中文要点**：流式代理的成败在于"不缓冲 + 取消要传到上游"：`aiter_raw` 逐块转发、`finally` 关上游连接、中途绝不重试，并把 nginx/中间件的 buffering 关掉。

---

### 2. Serving real-time TTS is a different capacity problem from serving a text LLM. Why, and how do you plan capacity?

**Anchor answer**

Text LLM capacity is measured in **tokens/sec per GPU** with continuous batching; TTS capacity is measured in **concurrent real-time streams per GPU**. Four structural differences:

| Dimension | LLM | Real-time TTS |
| :--- | :--- | :--- |
| Unit of work | token (variable cost) | second of audio (fixed cost per second) |
| Latency SLO | TTFT (can be 500ms+) | **TTFA ~150-300ms, every chunk** |
| Batching | continuous batching, large batch | batch ≈ 1-4; latency forbids queueing |
| Output | text, small | audio, 10-100 KB/s per stream |

The core metric is **RTF** (real-time factor) = compute seconds per audio second. A model at RTF 0.1 sustains ~10 streams per GPU-equivalent *if* you can batch; at batch size 1 you sustain ~1/0.1 with heavy tail loss from vocoder/decoder jitter.

Planning procedure:

1. Measure RTF at **p99**, not mean, and at the intended concurrency — autoregressive codec decoders degrade non-linearly with batch.
2. Convert: `GPUs = peak_concurrent_streams × p99_RTF ÷ 0.7` (30% headroom for burst, preemption, and rolling deploys).
3. Add an admission controller with a per-tenant stream cap; queueing is worse than a fast 429 because a queued stream already missed its TTFA.
4. Model the diurnal burst: voice agents concentrate in business hours, so buy for peak and let batch (non-interactive dubbing, audiobook) absorb the trough.

**The trap**: capacity plans built on mean RTF look fine and then fail at 3pm, because the tail is dominated by long-form requests holding streams open, not by request rate.

**中文要点**：TTS 的容量单位是"并发实时流"而非 token/s，SLO 是每块音频的 TTFA；按 p99 RTF 而非均值算 GPU（留 30% 余量），并做按租户的流级准入控制——排队比快速拒绝更糟。

---

### 3. Design the safety stack for voice cloning: consent, watermarking and abuse response.

**Anchor answer**

Voice cloning safety is a **pipeline of gates**, and each gate has a different failure mode.

| Stage | Mechanism | What it stops | Residual risk |
| :--- | :--- | :--- | :--- |
| Enrollment | liveness challenge: read a randomly generated phrase, on camera, with prompt variation | replay of scraped audio | synthetic enrollment (deepfake-of-a-deepfake) |
| Consent | signed authorization + scope (which use, which expiry); revocation registry | impersonation of a public figure | disputed consent, estates |
| Identity | KYC for professional/production voices; blocklist of high-risk speakers checked at synthesis | political/celebrity cloning | look-alike voices |
| Generation | **inaudible watermark** embedded in the waveform (robust to MP3/AAC re-encode and mild resampling) + C2PA provenance manifest on the file | detection after the fact | cropping/re-recording removes it |
| Detection | classifier scoring "is this our model's output for this speaker" + speaker-similarity check on uploads | laundering through our own API | open-source models |
| Response | abuse intake → triage SLA → revoke voice, ban tenant, preserve evidence, notify | repeat abuse | already-published audio |

Three things practitioners learn the hard way:

1. **Consent must be re-checked at synthesis, not only at enrollment.** A voice enrolled for "internal training videos" then used in a robocall is the actual incident shape.
2. **Watermarks fail under re-recording** (speaker → mic → phone). Keep a perceptual-hash side channel and treat watermarking as detection aid, not proof.
3. **Abuse response needs a name and an SLA.** A queue nobody owns is equivalent to no policy.

**中文要点**：声纹克隆安全是分级闸门——活体挑战注册、带范围与有效期的授权、生成端不可听水印+C2PA、上传端检测、以及有主责人和 SLA 的滥用响应；授权要在**合成时**再校验。

---

### 4. Budget the end-to-end latency for a real-time voice agent. Why is time-to-first-audio a different problem from an LLM's time-to-first-token?

**Source**: https://outcomeschool.com/blog/design-a-real-time-voice-ai-agent

**Anchor answer**

Target: a human-feeling turn gap of **~300-500ms** from end of user speech to first agent audio. Budget it top-down:

| Stage | Typical p50 | Notes |
| :--- | :--- | :--- |
| Endpointing (VAD + silence) | 200-500ms | the largest and most-ignored cost |
| ASR finalisation | 50-150ms | streaming partials already running |
| LLM TTFT | 150-400ms | first *sentence*, not first token |
| TTS TTFA | 60-200ms | first audio chunk |
| Network / jitter buffer | 40-100ms | WebRTC/RTP |
| **Total** | **~0.5-1.3s** | needs parallelisation to hit target |

Why TTFA ≠ TTFT:

1. **TTFT is one token.** TTFA needs enough phonological context to produce *prosodic* speech: "I" alone gives you a wrong intonation and a click. You must wait for a phrase boundary, then synthesize — so TTS adds a **text-accumulation delay** that a text LLM never pays.
2. **TTFT is a server metric; TTFA is a media metric.** It includes codec framing (20ms frames), playout buffer, and the client's audio pipeline. A 90ms TTFA that lands in a 400ms jitter buffer is a 500ms TTFA.
3. **Backpressure is different.** Tokens arrive over SSE and can be rendered progressively; audio cannot be "partially played" — a truncated MP3 frame is a click.

Levers: streaming ASR partials that start the LLM before endpointing, speculative first-token generation, sentence-level incremental TTS (flush on `,`/`.`), pre-synthesized acknowledgements ("mm-hm", "right") to mask the gap, and barge-in support so the user can interrupt rather than wait.

**中文要点**：实时语音的预算是"说完→首音"300-500ms，最大头是端点检测；TTFA ≠ TTFT，因为 TTS 要等短语边界才能出对语调，且它算的是媒体时延（含抖动缓冲），不是服务器时延。

---

### 5. Text normalisation is where TTS quality actually dies in production. Walk me through it.

**Anchor answer**

TN turns raw text into what a speaker would say. It fails on **ambiguity**, and ambiguity is unbounded, so the design is: deterministic rules for the 95% that is unambiguous, a contextual model for the rest, never the other way round.

| Class | Input → spoken | Failure mode |
| :--- | :--- | :--- |
| Numbers/units | `5'10"` → "five foot ten" | `5-10` is a range, `5/10` is a date or a score |
| Dates | `03/04/2025` → locale-dependent | US vs EU swap; silently wrong, not crashed |
| Currency | `$1.2M` → "one point two million dollars" | `$` in CAD/AUD context |
| Ranges | `2019-2021` → "twenty nineteen to twenty twenty-one" | read as subtraction |
| Homographs | `read`, `live`, `lead`, `wind` | needs POS/tense, not a lexicon |
| Acronyms | `IV`, `US`, `HIV` | initialism vs word; `IV` is also Roman four |
| Identifiers | order `#A1234`, phone, VIN | must be **spelled**, never normalized into words |
| Code-switching | mixed zh/en product names | language ID per span |

Architecture:

1. **Segment + language ID** first; TN is per-locale.
2. **Protect spans** (URLs, emails, SKUs, names) with regex/entity tagging so rules cannot corrupt them. This is the single highest-value engineering step.
3. **Rule engine with confidence**: each rule emits `(spoken, confidence, reason)`.
4. **Model fallback** only below threshold, and only with a constrained decoder over the input span (rewrite, not generate) — an unconstrained model invents digits.
5. **Verify by round-trip**: synthesize → ASR → compare normalized forms. This catches the 3am bugs that unit tests miss.

Watch: over-normalisation of model names ("GPT-4o" → "gee pee tee four oh" is wrong), and SSML that a downstream engine silently ignores.

**中文要点**：TN 死在歧义上——日期、区间、同形异读、以及"该拼读的 ID 被念成词"。做法是先分段定语言、**先把 URL/SKU/人名保护起来**、规则出置信度、低置信才走受约束的模型改写，并用"合成→ASR 回转"验证。

---

### 6. Design the dubbing pipeline: an English video becomes Spanish, same speakers, same timing.

**Anchor answer**

Five stages, each with a measurable failure:

| Stage | Technology | Constraint that dominates |
| :--- | :--- | :--- |
| 1. Ingest & separate | source separation (dialogue / music / SFX), scene detection | bleed on overlapping dialogue |
| 2. Transcribe | ASR with word timestamps + diarization + punctuation | crosstalk, on-screen text, laughter |
| 3. Translate | MT with **length/register constraints**, glossary for names & brands | Spanish runs ~15-25% longer than English |
| 4. Re-voice | cross-lingual TTS conditioned on the original speaker embedding (or voice conversion on a synthesized base) | timbre drift, lost emotion, wrong accent |
| 5. Align & mix | time-stretch within ±8-10%, then re-time at cut points; remix with preserved music/SFX | stretch artifacts, lip-sync drift |

Key engineering decisions:

- **Chunk at breath groups, not sentences.** A 40-word sentence cannot be re-timed; splitting at clause boundaries gives the aligner slack.
- **Translation must know the timing budget.** Pass per-segment duration and a target syllable budget to the MT decoder (length-controlled decoding), otherwise stage 5 fights stage 3 forever.
- **Global tempo drift beats local stretching.** Distribute ±3% across the whole clip and keep only a few local stretches; audiences notice local warble much more than a 3% overall speed change.
- **Speaker identity is a per-clip constant.** Lock the speaker embedding across segments; re-estimating per segment makes the voice wobble.

What breaks in production: overlapping speakers (needs per-speaker separation, not per-turn), songs (do not dub; keep original), and on-screen titles (must be re-rendered, not translated by TTS at all).

**中文要点**：配音五段——分离音轨、带词级时间戳+说话人分离的 ASR、**带时长约束**的翻译、跨语种音色保持的 TTS、±10% 内的时间拉伸与混音；要在意群切分，并让整体微调速度而非局部乱拉。

---

### 7. A hospital group schedules and confirms outpatient appointments by phone, manually, with three staff on a rota. Design what we would build for them.

**Anchor answer**

Start by refusing to build a chatbot. The unit of value is a **completed appointment**, so the system is a scheduling engine with a voice front end.

| Layer | Build | Why |
| :--- | :--- | :--- |
| Telephony | SIP trunk + media server, barge-in, DTMF fallback | patients press keys when speech fails |
| Dialogue | streaming ASR → intent + slot filling → policy-driven dialogue manager (deterministic state machine for booking) | booking must not be free-form generation |
| Identity | DOB + last name + (optionally) a callback number match; **no clinical detail before verification** | HIPAA-grade minimum necessary |
| Scheduling | write through the EHR/scheduling API with idempotency keys, slot holds with TTL | double-booking is the cardinal sin |
| Confirmation | SMS/IVR callback, reminder cadence, reschedule link | the actual ROI is no-shows |
| Escalation | warm transfer to the 3 staff with a transcript summary | humans handle exceptions, not the happy path |

Metrics that decide go/no-go: containment (calls resolved without transfer), **booked-and-kept rate**, average handle time, and misbooking rate (must be ~0, measured by audit).

Failure modes to design against explicitly:

1. **Name/DOB collision** → wrong patient's appointment. Require a second factor; never read back clinical info to an unverified caller.
2. **Slot race** with the front desk → atomic hold + expiry, and a reconciliation job.
3. **Accents and elderly callers** → ASR confidence gating, slow-repair prompts ("I heard *Tuesday the 4th*, is that right?"), DTMF always available.
4. **After-hours emergencies** → hard-coded triage script that routes to emergency services, never to the booking flow.

Roll out on one clinic, one call type (confirmations first, they are outbound and low-risk), then booking.

**中文要点**：别做"聊天机器人"，做带语音前端的排班引擎：SIP+打断、确定性预约状态机、身份二次校验后才谈临床信息、带幂等键与短 TTL 占位的写库、以及转人工兜底；先上外呼确认（低风险）再上预约。

---

### 8. A contact centre wants to replace its IVR with voice agents. Run the engagement.

**Anchor answer**

Forward-deployed discipline: **the model is 20% of the work; integration, evaluation and change management are the other 80%.**

Week 1 — measure before promising. Pull 2-4 weeks of call recordings and CDRs. Segment by intent and build the value model:

| Segment | Share | Automatable now? |
| :--- | :--- | :--- |
| Status/ETA, balance, hours, password reset | often 30-50% | yes, deterministic + API |
| Reschedule/cancel, address change | 10-20% | yes, with write access |
| Billing disputes, complaints | 10-15% | assist only, transfer |
| Sales/retention | 10% | do not automate — it is the revenue |

Week 2-3 — pick **one intent with a system of record**. Nothing happens without an API to act on; a voice agent that can only talk is a demo.

Design constraints to state up front: latency target (see TTFA budget), containment vs transfer policy, barge-in, DTMF fallback, disclosure that the caller is talking to an AI (legally required in several jurisdictions), and a hard "no PHI/payment data to a third-party model without a BAA/DPA" rule.

Week 3-5 — build the eval set from **real calls**, not synthetic scripts: 200-500 utterances stratified by accent, background noise, and intent. Metrics: task completion, containment, transfers, and **wrong-action rate** (the only one that can cost money).

Pilot on 5-10% of traffic behind a flag with instant rollback. Instrument every turn (transcript, intent, tool calls, latency) and review 50 calls/week by hand for the first month.

What kills these projects: automating the top-volume intent before confirming API write access, and letting containment be the headline KPI — agents that hang up on frustrated callers score beautifully.

**中文要点**：FDE 节奏是"先量后承诺"：按意图切分流并算价值，只挑**有系统 API 可写**的意图先做；用真实通话建评测集，盯任务完成率与误操作率而非"拦截率"，灰度+可回滚+每周人工听 50 通。

---

## 附：这 8 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 实时音频工程功底 | 1（流式代理与取消）、4（TTFA 预算）、2（容量规划） |
| 语音模型深度 | 5（TN）、6（配音流水线） |
| 安全与合规 | 3（克隆安全栈） |
| FDE / 交付能力 | 7（医院预约）、8（IVR 替换） |
