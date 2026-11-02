# 22 · Sarvam AI

> 12 题。出处：`README.en.md` § Sarvam AI。答案为本仓库自行整理，`Source` 为原文延伸阅读。

---

### 1. Write code to measure a tokenizer's fertility across languages, and explain what you would do with the result.

**Source**: https://outcomeschool.com/blog/bpe-in-llms

**Anchor answer**

**Reference implementation**

```python
from collections import defaultdict
import statistics

def measure_fertility(tok, corpus: dict[str, list[str]]) -> dict[str, dict]:
    """corpus: {lang: [sentences]}. Returns per-language tokenizer statistics."""
    rows = {}
    for lang, sents in corpus.items():
        toks_per_sent, chars, words = [], 0, 0
        unk, bytes_total = 0, 0
        for s in sents:
            ids = tok.encode(s, add_special_tokens=False)
            toks_per_sent.append(len(ids))
            chars += len(s)
            words += len(s.split()) or 1
            for t in ids:
                piece = tok.decode([t])
                if piece in ("<unk>", "\ufffd"):
                    unk += 1
                bytes_total += max(1, len(piece.encode("utf-8")))   # bytes the token covers
        n_tok = sum(toks_per_sent)
        rows[lang] = {
            "tokens_per_word": n_tok / words,          # classic fertility
            "tokens_per_char": n_tok / chars,
            "chars_per_token": chars / n_tok,          # inverse: compression ratio
            "bytes_per_token": bytes_total / n_tok,
            "unk_rate": unk / n_tok,
            "median_len": statistics.median(toks_per_sent),
            "p95_len": sorted(toks_per_sent)[int(0.95 * len(toks_per_sent)) - 1],
        }
    base = rows[next(iter(rows))]["tokens_per_word"]
    for lang, r in rows.items():
        r["relative_cost_vs_base"] = r["tokens_per_word"] / base
    return rows
```

**Key points**

- Report **`tokens_per_word` (fertility)** and **`bytes_per_token`** together. Fertility alone lies for scripts with different word-segmentation conventions — Hindi, Tamil and Telugu don't delimit words like English, so also report tokens per *character* and per *UTF-8 byte*.
- Always use a **parallel corpus** (same sentences, translated) or you are measuring corpus differences, not tokenizer differences.
- Segment corpus by *dialect and script* too — Devanagari Marathi and Tamil behave completely differently.
- Watch `unk_rate`: with byte fallback it's 0, but bytes-per-token collapsing to ~1 is the silent red flag — that language is being spelled out byte by byte.
- Track p95 length, not just the mean: the tail is what triggers truncation and context overflow.

**What to do with the result:** a >1.5× relative cost versus the base language means the language is paying a 50% tax on context, latency and price. Actions, cheapest first: (1) add a script-specific normaliser (Unicode NFC, schwa handling); (2) re-train *only the tokenizer* on a rebalanced corpus and re-initialise embeddings (cheap but needs repair training); (3) add the missing script to the vocab and continue pretraining; (4) if you own the whole model, learn the vocab on a target-ratio multilingual corpus from the start.

**中文要点**：同时报 tokens/word（fertility）与 tokens/byte——印地语、泰米尔语等没有英文式分词，单看 fertility 会骗人；必须用平行语料，分脚本分方言统计，看 p95 而非均值，并留意 byte fallback 导致 bytes/token 掉到 1 这个隐性情信号。>1.5× 相对成本就要动手：先加脚本归一化，再考虑重训分词器或扩词表续训。

---

### 2. Why is tokenization the first bottleneck for Indian-language LLMs, and how does a low-fertility tokenizer change the economics?

**Source**: https://outcomeschool.com/blog/bpe-in-llms

**Anchor answer**

Because Indian scripts are brutally expensive under an English-trained BPE. Devanagari, Tamil, Telugu, Bengali and Malayalam are 3 bytes per character in UTF-8, and a vocabulary learned on English web text spends its 32K–50K merges on English morphemes. Result: one Hindi word becomes 4–8 tokens, against ~1.3 for English.

That single ratio propagates everywhere:

| Effect | Consequence |
| :--- | :--- |
| Context tax | a 128K window holds ~20–30K words of Hindi instead of ~90K of English |
| Cost tax | 3–6× more tokens per request at the same price per token |
| Training tax | more tokens per document ⇒ more compute, fewer effective epochs |
| Capability tax | long multi-hop reasoning degrades because the model spends its attention budget on spelling instead of meaning |
| Morphology tax | Hindi/Tamil agglutinate and inflect; an over-fragmented tokenizer hides stems and suffixes from the model, hurting the pattern "same lemma, different case" |

**The economics of fixing it:** moving a language from ~4.5 to ~1.5 tokens per word cuts inference cost by ~3× and triples effective context. For a government helpline at 10M calls/month, that is the difference between viable and not — it's a bigger lever than most model-quality work, and it's cheap: tokenizer retraining plus a few billion tokens of repair pretraining, versus retraining the base.

**The counterintuitive bit:** a low-fertility tokenizer also improves *reasoning*, not just cost. With fewer tokens for the same content, the model's limited attention budget covers more propositions, and morphology becomes visible at the token level (suffixes as separate but consistent pieces rather than arbitrary byte fragments).

**中文要点**：印度文字 UTF-8 每字符 3 字节，英文语料训出的 BPE 把一个印地语词切成 4–8 个 token，直接放大成上下文税、成本税（3–6×）、训练算力税与能力税（注意力花在拼写上）；形态变化被切碎还伤及"同词根不同格"的模式。修法只需重训分词器 + 少量修复预训练，却同时降成本、扩有效上下文、提升推理。

---

### 3. How do you deploy a capable assistant on cost-sensitive or on-device hardware without a datacentre GPU? Walk through the efficiency toolkit.

**Source**: https://outcomeschool.com/blog/how-does-model-quantization-work

**Anchor answer**

Work down this ladder; each rung is cheaper than the one above but costs more accuracy.

| Rung | Technique | Typical gain | Cost |
| :--- | :--- | :--- | :--- |
| 1 | **Pick a smaller base** (1–8B, distilled from the flagship) | 10–50× FLOPs | capability ceiling |
| 2 | **INT4 weight-only quantisation** (AWQ / GPTQ / GGUF Q4_K_M) | 4× smaller weights, ~2–3× decode speed | small perplexity hit; per-channel scales needed |
| 3 | **KV-cache quantisation / smaller context** | 2–4× more concurrent users | long-context degradation |
| 4 | **Speculative decoding** with a tiny draft (or MTP heads) | 1.5–2× tokens/s | memory for the draft |
| 5 | **Runtime + kernel choice** (llama.cpp/ONNX Runtime/CoreML, NEON, int8 GEMM) | 2× vs naive PyTorch | engineering effort |
| 6 | **Pruning / MoE with few active params** | compute reduction | retraining, uneven quality |
| 7 | **Prompt/prefix caching + short system prompt** | large on multi-turn | cache memory |

**Two rules that decide success more than the quantisation format:**

1. **Measure on the target device, not on a proxy.** A 7B INT4 model on a phone is **memory-bandwidth-bound** — decode speed tracks MB/token read, so a 4-bit model that keeps FP16 activations gains less than the compression ratio suggests.
2. **Quantise for the data you'll actually see.** Indic scripts and code have very different activation outlier profiles from English chat; a calibration set of English Wikipedia will produce a model that degrades on Devanagari. Use a domain-matched calibration set, and always run an eval on te/ta/bn/hi before shipping.

**When not to bother:** if the device has < 4 GB usable RAM and the task needs multi-step reasoning, on-device is the wrong answer — do ASR/TTS on device and route the LLM to a small regional server.

**中文要点**：按阶梯做——先选小基座/蒸馏，再 INT4 权重（AWQ/GPTQ/GGUF）、KV 量化、投机解码、换运行时（llama.cpp/CoreML）与前缀缓存。两条成败规则：在真机上量（解码是带宽瓶颈，看 MB/token），以及用目标语言（天城文/泰米尔文）做校准集与验收——用英文维基校准会在印度语上崩。

---

### 4. Design cross-lingual RAG: the knowledge base is in English and Hindi, but users ask in Tamil, Telugu or transliterated Hinglish.

**Anchor answer**

**The core choice: do not translate the query and hope.** Three architectures, and the honest trade-off:

| Approach | How | Strength | Weakness |
| :--- | :--- | :--- | :--- |
| Translate → monolingual retrieve | MT query to EN/HI, then retrieve | reuses a strong English index | MT errors on named entities and code-mixed text; two hops of loss |
| **Multilingual embedding space** | one encoder aligned across scripts; retrieve directly | no MT dependency, handles code-mixing | alignment is imperfect for low-resource languages; embeddings drift on transliteration |
| Translate the *documents* | index everything in one pivot language | simple, one index | corpus-sized MT cost; translationese loses Indic-specific facts |

**Recommended: multilingual dense retrieval + lexical hybrid, with query normalisation.**

1. **Normalise the query first.** Transliterated Hinglish ("mera ration card status check karo") must be script-normalised or romanisation-aware — otherwise the lexical arm returns nothing. Keep a transliteration index alongside the native-script index.
2. **Hybrid retrieval**: multilingual dense (e.g. an Indic-capable embedding) + BM25 on both scripts, fused by RRF. Lexical catches scheme names and IDs that dense embeddings blur.
3. **Rerank cross-lingually** with a multilingual cross-encoder on the top 50 — this is where most of the quality comes from.
4. **Answer in the user's language**, citing the source passage in its original language; the generator handles the pivot.
5. **Evaluate per language**, with query sets authored natively, including code-mixed and transliterated variants. A Tamil query set made by translating English questions measures MT, not retrieval.

**The failure that bites:** entity mismatch — "Pradhan Mantri Awas Yojana" appears in the corpus as "PMAY", "प्रधानमंत्री आवास योजना" and "PM आवास योजना". Build an alias/abbreviation dictionary per scheme; no embedding model will bridge all of those.

**中文要点**：三种架构里推荐"多语言向量 + 词法混合"而非"翻译查询再检索"；先做查询归一化（罗马化 Hinglish 要能匹配），dense+BM25 双脚本 RRF 融合，再用多语言 cross-encoder 重排 top-50，最后用用户语言作答并附原文引用；真正的坑是实体别名（PMAY / प्रधानमंत्री आवास योजना），必须建同义词表；评测必须用母语者写的查询集，含混说与音译变体。

---

### 5. Sarvam-M ships hybrid think/non-think modes and was post-trained with SFT then RLVR. How would you build that, and why RLVR over vanilla RLHF?

**Source**: https://outcomeschool.com/blog/large-reasoning-models

**Anchor answer**

**Build.**

1. **Base + continued pretraining** on a balanced Indic/English mix (tokenizer fixed *first* — see the fertility question).
2. **SFT with two answer styles** for the same prompt: a compact direct answer and a reasoning trace where the domain has a checkable answer (math word problems, date/eligibility logic, tabular extraction). Include a mode token so the switch is explicit.
3. **RLVR.** Sample $G$ rollouts per prompt, score with a **verifier**: exact numeric match for arithmetic, unit tests for code, exact span match for extraction, and — importantly for Indic — a **transliteration-tolerant matcher** so a correct Tamil answer isn't scored wrong for spelling variance.
4. **Preference pass on the direct mode only**, for tone and brevity. Keeping RLHF off the thinking mode prevents style pressure from shortening reasoning.

**Why RLVR over vanilla RLHF here:**

| | RLHF | RLVR |
| :--- | :--- | :--- |
| Data | thousands of human preference pairs, per language | no preference labels; just a verifier |
| Indic coverage | annotators scarce for ta/te/bn/ml; cost scales with languages | verifier works identically in every language |
| Failure mode | reward hacking on length/tone, annotator bias | sparse reward; needs difficulty curriculum |
| Signal | subjective, noisy | exact on the covered domains |

For a company whose differentiator is *breadth across many languages*, RLHF's per-language annotation cost is the binding constraint; RLVR is the only approach that scales to 10+ languages at once. The price is domain coverage: no verifiable reward for "explain this scheme kindly", so the non-think mode still needs preferences.

**中文要点**：先固定分词器 → 双风格 SFT（直接答 + 可验证域的思维链，带模式 token）→ RLVR（组内采样 + 验证器，印地语系要容忍音译写法差异）→ 仅对直接模式做偏好优化以免风格压力压短推理。选 RLVR 是因为 RLHF 要按语言做人工偏好标注，成本随语言数线性增长；RLVR 的验证器对所有语言一视同仁，代价是只覆盖可验证域。

---

### 6. A regional government wants an assistant in a low-resource language with only a few thousand sentences of clean text. How do you adapt a model to it?

**Source**: https://outcomeschool.com/blog/how-does-fine-tuning-work

**Anchor answer**

A few thousand sentences is **far too little** to teach a language, but enough to teach a *task and a voice*. Split the problem.

| Phase | Data | Method | Purpose |
| :--- | :--- | :--- | :--- |
| 0. Script/normalisation | rules | Unicode NFC, script mapping, transliteration tables | removes gratuitous variance before any learning |
| 1. Lexical bootstrap | 5–50K sentences of *unlabelled* monolingual text (newspapers, Wikipedia, gov PDFs) | continue pretraining (LoRA or full, low LR) | vocabulary and morphology, not behaviour |
| 2. Instruction tuning | the few thousand clean sentences → converted to tasks | QLoRA, rank 16–64, all linear layers | format, tone, domain terms |
| 3. Augmentation | translate the 3K into English and back; paraphrase via a strong multilingual model; template expansion for forms/FAQs | filter by round-trip consistency | 5–10× effective size |
| 4. Evaluation | native speakers, 100–200 items | human rating + task accuracy | the only number that matters |

**Non-negotiables:**

- **Replay English/Indic general data (20–30%)** inside the LoRA mix, or you get catastrophic forgetting — the model becomes good at the target language and bad at instruction following in general.
- **Never trust translated benchmarks** for this language; have speakers write 100 items.
- **Data quality > quantity, brutally.** 3K native sentences beat 50K machine-translated ones; MT output teaches translationese syntax that native speakers immediately notice.
- **Check the tokenizer first.** If the language is 5 tokens/word, you're spending your tiny data budget on spelling — retrain the tokenizer before anything else.

**Expectation setting:** with this budget you can reach "usable for the narrow task, requires human review" — not "generally fluent". Say that out loud, and design the product so a human is in the loop.

**中文要点**：几千句不足以教语言，但足以教任务与语气——先用 5–50K 无标注单语文本继续预训练打词汇/形态基础，再用这几千句做 QLoRA 指令微调，配合回译/模板扩增 5–10×，并混入 20–30% 通用数据防灾难遗忘；务必先查分词器 fertility，别把数据预算花在拼写上。预期是"窄任务可用 + 人工复核"，不是通用流利。

---

### 7. Design a real-time voice agent for a citizen helpline in Hindi and three regional languages, targeting sub-250 ms perceived latency over a phone line.

**Source**: https://outcomeschool.com/blog/design-a-real-time-voice-ai-agent

**Anchor answer**

**Latency budget first, because it constrains everything.**

| Stage | Budget | Notes |
| :--- | :--- | :--- |
| Telephony + jitter buffer | 20–40 ms | 8 kHz, G.711; resample to 16 kHz for ASR |
| VAD endpointing | 100–200 ms of silence | too short ⇒ cuts the user off mid-word (fatal in Hindi) |
| Streaming ASR (partial → final) | 150–300 ms after speech end | run partial hypotheses; don't wait for a full sentence |
| LLM first token | 200–400 ms | short system prompt, prefix-cached; small model routing |
| TTS first audio chunk | 100–200 ms | streaming synthesis, not sentence-level |

**Perceived** latency is what matters: if the agent emits an acknowledgment ("जी, बताइए") within ~200 ms, users tolerate a 600 ms answer. **Backchannel + filler is the cheapest latency trick there is.**

**Architecture**

1. **VAD + barge-in.** User can interrupt mid-TTS; kill synthesis immediately and re-open ASR. Without barge-in the line feels like an IVR and callers hang up.
2. **Streaming ASR** with language auto-detect per utterance, not per call — callers switch between Hindi and their language mid-sentence.
3. **Dialogue manager** as a state machine for structured flows (scheme eligibility, application status), LLM only for free-form. Deterministic flows are cheaper, auditable, and never hallucinate a scheme.
4. **Streaming TTS** in the detected language, with a voice per language, plus a fallback to the majority language.
5. **Fallback path**: two failed ASR turns ⇒ offer DTMF ("press 1") — phone lines are noisy and this saves the call.

**Failure modes:** code-mixed input (Hinglish) breaking the ASR; numbers and scheme names mangled (read back critical values for confirmation); network loss mid-call (resume from state, don't restart); and accent variance across districts — evaluate per district, not per state.

**中文要点**：先定延迟预算（电话/抖动 20–40ms、端点 100–200ms、ASR 150–300ms、LLM 首 token 200–400ms、TTS 首包 100–200ms），并用 200ms 内的确认语换用户容忍；架构上必须支持 barge-in 打断、按句而非按通话做语种检测（混说常态）、结构化流程用状态机而非 LLM、两轮识别失败降级 DTMF；数字与计划名要回读确认。

---

### 8. How would you evaluate an Indic LLM properly? Why is running translated English benchmarks not enough?

**Source**: https://outcomeschool.com/blog/llm-evaluation

**Anchor answer**

Translated benchmarks measure the **translator**, not the model, and they fail in four specific ways.

| Failure | Why it invalidates the number |
| :--- | :--- |
| Translationese | MT output has unnaturally clean, English-shaped syntax; the model is scored on a distribution no native speaker writes |
| Cultural grounding | a translated "commonsense" item keeps a Western premise; the correct answer for an Indian user may differ |
| Answer-format leakage | options translated inconsistently; the correct one is often the longest or the most English-looking — models exploit it |
| Script and morphology | the original may rely on English word order; Indic languages are free-order and agglutinative, so difficulty shifts arbitrarily |

**An evaluation stack that holds up:**

1. **Native-authored items.** Written by speakers in each language, 100–300 per language per capability (QA, extraction, summarisation, arithmetic word problems, refusal).
2. **Per-language reporting, never an average.** A mean over hi/ta/te/bn/ml hides that one language is 30 points behind; report the min.
3. **Code-mixed and transliterated sets.** Real users type "ration card ka status" — a model can score well on pure Hindi and fail completely here.
4. **Human evaluation with native raters** on fluency, correctness, and *register* (formal govt Hindi vs colloquial) — automatic metrics can't see register errors, and they matter for a helpline.
5. **Tokenizer-fair protocol.** Few-shot prompts should be capped by *token* budget, not example count, or a high-fertility language silently gets less context and looks worse for the wrong reason.
6. **Task-grounded evals** for the actual product: form filling, scheme eligibility, date/amount extraction — with exact-match on structured fields.

**The uncomfortable conclusion:** for most Indic deployments, 200 native items and 3 native raters tell you more than the whole translated MMLU table.

**中文要点**：翻译基准测的是译者——翻译腔、文化前提错位、选项长度泄漏、语序/形态难度漂移，四条都会让分数失真；正确做法是母语者原生出题（每语言每能力 100–300 题）、分语言报并看最差语言、加入混说与音译集、双语者做人工评分看语域、few-shot 按 token 预算而非条数（否则高 fertility 语言吃亏）、并对表单/资格/金额做结构化精确匹配。

---

### 9. Build a Voice Activity Detector from scratch. How do you make it robust for phone-quality Indian-language audio?

**Source**: https://outcomeschool.com/blog/design-a-real-time-voice-ai-agent

**Anchor answer**

**Reference implementation (classical, then the learned upgrade)**

```python
import numpy as np

class EnergyVAD:
    def __init__(self, sr=8000, frame_ms=20, hang_ms=200, on_db=-35, off_db=-45):
        self.n = int(sr * frame_ms / 1000)
        self.hang = int(hang_ms / frame_ms)
        self.on_th, self.off_th = on_db, off_db
        self.noise_db, self.alpha = -60.0, 0.95
        self.state, self.tail = False, 0

    def frame_features(self, frame):
        x = frame.astype(np.float64)
        x = x - x.mean()                                  # kill DC offset (phone lines have plenty)
        win = np.hanning(len(x))
        X = np.abs(np.fft.rfft(x * win)) ** 2
        freqs = np.fft.rfftfreq(len(x), 1 / 8000)
        band = (freqs >= 300) & (freqs <= 3400)           # telephone band; ignore the rest
        energy_db = 10 * np.log10(X[band].sum() + 1e-10)
        zcr = np.mean(np.abs(np.diff(np.signbit(x))))     # voicing/unvoiced cue
        flatness = np.exp(np.mean(np.log(X[band] + 1e-10))) / (np.mean(X[band]) + 1e-10)
        return energy_db, zcr, flatness                   # flatness ≈ 1 ⇒ noise, ≈0 ⇒ harmonic

    def __call__(self, frame):
        e, zcr, flat = self.frame_features(frame)
        # adaptive noise floor: track only during confirmed silence
        if not self.state:
            self.noise_db = self.alpha * self.noise_db + (1 - self.alpha) * e
        snr = e - self.noise_db
        speech = snr > (self.on_th - self.noise_db + 35) and flat < 0.6 and zcr < 0.35
        if speech:
            self.state, self.tail = True, self.hang
        elif self.state:
            self.tail -= 1
            if self.tail <= 0:
                self.state = False
        return {"speech": self.state, "snr_db": snr, "energy_db": e, "flatness": flat}
```

**Key points**

- **Adaptive noise floor tracked only in silence** — a fixed threshold fails the moment the caller is on a bus; and updating during speech would drift the threshold upward and clip the utterance.
- **Band-limit to 300–3400 Hz before measuring** — phone lines carry hum and out-of-band energy that isn't speech.
- **Hysteresis: separate on/off thresholds plus a hangover** (150–250 ms). A single threshold chatters at syllable boundaries and fragments the ASR input.
- **Spectral flatness is the cheap speech/noise discriminator** — noise is flat, voiced speech is harmonic. Energy alone fails on loud non-speech (cattle, traffic, TV).
- **DC removal** matters on telephony; a DC offset inflates energy in every frame.

**Production upgrade:** take 20 log-mel frames + these features into a small CRNN/TCN (~50–200K params) trained on real telephone audio per language; a learned VAD typically halves false cut-offs versus energy-based. Keep the classical detector as the fallback for CPU-constrained deployment.

**中文要点**：只用能量阈值在电话线上必挂——要自适应噪声底（只在静音段更新）、先限带 300–3400Hz、DC 去偏、双阈值 + 150–250ms 拖尾做迟滞、并用谱平坦度区分噪声与浊音；生产上用 log-mel + 小 CRNN 换掉纯能量法，误切通常减半。

---

### 10. Whisper transcribes Hinglish poorly, often forcing output into one language or hallucinating. Why, and how would you build an ASR that handles code-mixed speech?

**Anchor answer**

**Why Whisper fails on Hinglish — three structural reasons:**

1. **One language token per 30-second window.** The decoder is conditioned on a single `<|hi|>` or `<|en|>` token. Code-switching is, by construction, outside its output space — the model must pick one and translate the other half, the "forced into one language" symptom.
2. **Training data is weakly labelled monolingual audio.** Web-scraped pairs are predominantly single-language; code-mixed speech has almost no supervision, so the model has never been rewarded for switching mid-utterance.
3. **Hallucination on low-information audio.** Trained with sequence-level losses on noisy web data, the decoder emits fluent text for silence, music, or unintelligible speech — the "subtitled YouTube" prior. On telephone audio in an unfamiliar dialect, this fires constantly.

**Building a code-mixed ASR:**

| Layer | Design |
| :--- | :--- |
| Acoustic model | CTC or RNN-T (frame-synchronous) on a shared phoneme/grapheme inventory — **no language token**, so switching costs nothing |
| Output script | decide policy: Roman transliteration for both languages (matches how users type), or native script with a parallel romanised head |
| Data | synthesise code-mixed audio: record bilingual speakers reading mixed sentences. A few hundred hours of genuine code-mixed speech beats thousands of translated hours |
| Language ID | frame-level LID as an auxiliary head (useful for downstream routing), not as a hard constraint |
| Decoding | shallow-fusion with an n-gram over romanised text; avoid a strong monolingual LM or it re-imposes one language |

**Evaluation:** WER per *segment type* (monolingual hi, monolingual en, mixed) plus a **switch-point accuracy** metric — overall WER hides the fact that switches are where all the errors live. Also measure hallucination rate on silence explicitly; transcribing 0.5% of silence as text is unusable for a helpline.

**中文要点**：Whisper 每 30 秒窗口只有一个语言 token，从结构上就无法输出混说，只能"选一个语言再翻译另一半"；加上训练数据几乎全是单语弱标注，以及对低信息音频的强先验，就出现强制单语与幻听。做法是改用无语言 token 的帧同步 CTC/RNN-T + 共享字素表、自建数百小时真实混说录音（双语者读混说句子）、帧级 LID 只作辅助头、避免强单语 LM；评测按单语/混说分段报 WER 并单列切换点准确率与静音幻听率。

---

### 11. Bulbul-style TTS has to speak code-mixed, mixed-script text naturally. What are the hard parts of text normalization and prosody for Indian-language TTS?

**Anchor answer**

**Text normalisation is where most of the quality lives, and it is language-specific.**

| Input | Hard part |
| :--- | :--- |
| `₹1,25,000` | Indian digit grouping (lakh/crore), not thousands; expand in the target language: "एक लाख पच्चीस हज़ार" |
| `12/03/2025` | day-first convention; "12 March" not "December 3" |
| `PMAY-G`, `Aadhaar`, `SMS` | acronyms are read letter-by-letter in English but as words in Hindi; may need a per-term pronunciation lexicon |
| Mixed script: `मेरा account balance` | each span must be routed to the right grapheme-to-phoneme front-end; romanised Hindi ("mera naam") must be read as Hindi, not English |
| English loanwords in Hindi | Indian-English phonology: `bank` → /bæŋk/ with retroflex-free realisation, not American vowels |
| Numbers in cases | Hindi/Tamil inflect numerals ("दो" vs "दोनों"); naive digit expansion produces ungrammatical output |

**Prosody — the part users judge instantly:**

1. **Schwa deletion.** Hindi written forms carry inherent /ə/ that is deleted in speech (`कमरा` → /kəmrɑː/ not /kəmərɑː/). Getting this wrong sounds foreign and is the single most audible defect.
2. **Aspiration and retroflex contrasts** (क/ख, त/थ, ट/ठ) must be preserved — collapsing them changes words.
3. **Phrase-level pitch contour**, not word-level: questions in Hindi rise at the end of the *phrase*, and Hindi/Tamil use pitch for focus. Per-word F0 prediction sounds robotic even with perfect phonemes.
4. **Sandhi and clitic attachment** across word boundaries; and a **code-switch prosody reset** — a speaker switching to English takes a small breath/tempo change. Models that ignore this produce a jarring splice.
5. **Speaking rate for numbers and IDs** — slow down for digits, or listeners can't transcribe them.

**Evaluation:** MOS from native speakers per language, plus a targeted "comprehension" test — can a listener write down the amount, date, and scheme name after one listen? That catches normalisation bugs MOS averages away.

**中文要点**：归一化是质量主战场——印度式数字分组（lakh/crore）、日先日期、缩写按语言决定拼读、混合脚本要按 span 路由到对应 G2P、英文借词用印度英语音系、数字还要按格变化；韵律上最致命的是印地语 schwa 删除、送气/卷舌对立、短语级而非词级语调、跨词连读与代码切换时的韵律重置。评测除 MOS 外加"听一遍能否写下金额/日期/计划名"的理解测试。

---

### 12. A state agency wants to move a paper-and-call-centre welfare-scheme service onto a multilingual assistant, on-prem for data residency. How do you scope and ship it?

**Anchor answer**

**Scope before technology.** The deliverable is not "an AI assistant" — it's a measurable reduction in unresolved calls, with an audit trail. Phase it.

| Phase | Duration | Deliverable | Gate to next phase |
| :--- | :--- | :--- | :--- |
| 0. Task inventory | 2–3 weeks | Top 20 intents by call volume, each with inputs/outputs and the current paper form | Intents covering ~70% of volume identified |
| 1. Retrieval + scripted assistant | 4–6 weeks | Deterministic flows for the top 5 intents, RAG over scheme documents, human handoff | ≥ 90% answer accuracy on a native-speaker test set |
| 2. Voice | 4–6 weeks | ASR + TTS for 2 languages, then expand; barge-in, DTMF fallback | WER and MOS thresholds per language; call-drop rate |
| 3. Autonomy | ongoing | Form filling, status lookup via backend APIs; confirmation before submission | Zero irreversible actions without explicit confirmation |

**On-prem constraints drive real decisions:**

- **Hardware budget first.** Inference on-prem means you size GPUs for peak concurrency, not average — and there's no elastic burst. Often the right answer is a 7–30B model, not the frontier model.
- **Data residency** applies to *logs and audio*, not just weights: recordings, transcripts, and eval sets must stay in-country; this rules out hosted ASR and hosted eval tooling.
- **Offline updates.** Model and index updates ship as signed bundles on a schedule; no live internet means your eval harness must run locally too.

**Failure modes I would put in the risk register:** intent drift as schemes change (needs a monthly re-index and a canary intent set); language coverage gaps in dialects (evaluate per district); citizens gaming the assistant; and the political cost of a *wrong* eligibility answer — so eligibility logic must be deterministic code, with the LLM only explaining it, never deciding it.

**中文要点**：先做任务盘点（按通话量选 Top-20 意图，覆盖约 70% 量），再分阶段：检索+脚本化流程 → 语音（分词种设 WER/MOS 门槛、DTMF 兜底）→ 有限自主（提交前必须确认）；本地化部署要先算峰值并发 GPU 预算（往往 7–30B 就够），数据驻留覆盖日志与录音，模型/索引需签名离线更新包；资格判定必须是确定性代码，LLM 只负责解释。

---

## 附：这 12 题怎么用

| 筛什么 | 挑哪几题 |
| :--- | :--- |
| 能写能跑 | 1（fertility 测量）、9（从零写 VAD） |
| 印度语言核心认知 | 2（分词是第一瓶颈）、6（低资源适配）、8（本土评测） |
| 语音工程 | 10（Hinglish ASR）、11（TTS 归一化与韵律）、7（实时语音坐席） |
| 系统设计与落地 | 4（跨语言 RAG）、3（端侧/低成本部署）、12（政府本地化交付） |
| 后训练 | 5（混合模式 + RLVR） |
