# expressive-s2s

**An agent that hears how you sound and answers in a voice that matches.**

Speech in → emotion estimated from the audio → reply generated conditioned on
that emotion → reply spoken with matching prosody, with a non-verbal cue placed
in the gap while the rest of the pipeline is still working.

It runs end to end on one laptop, and it is measured end to end. **Listen first.**
The same four real turns, laid out on the real clock, on each of the two TTS
engines — the silences you hear are the silences that were measured:

- [`demo/exchange.wav`](demo/exchange.wav) — Piper. Fast (reply at 501–1080 ms)
  and, as §3 shows, acoustically flat regardless of what was detected.
- [`demo/exchange-expressive.wav`](demo/exchange-expressive.wav) — macOS `say`.
  You can hear the voice change with the detected emotion, and you can hear the
  600 ms it costs (reply at 1133–1562 ms).

Both are the same four turns: detected anger → emphatic, disgust → neutral,
fear → sad, sad → sad, with a filled-pause cue in the gap each time.

> **The headline, in three numbers.** Emotion detection costs **45 ms** [42–53]
> as a serial stage, and **0 ms** when overlapped with the ASR decode — against
> a no-emotion control it is free to within run-to-run noise (−13 ms, sd 53–96).
> The classifier survives the loop: **79.2%** [59.5, 90.8] against actor intent
> inside the live pipeline, n = 24, versus 74.9% offline.
> And the expressive output **only measurably differs by emotion on the slow TTS
> engine** — F0 separates at p = 0.00058 on macOS `say`, and not at all on Piper
> (p = 0.17), which costs 634 ms less per utterance.

---

## The system, and the evidence behind each part

This is one project. The repo you are reading is the system; four sibling repos
are the studies that decided how it is built. Every design choice below points
at a measurement, and every measurement is one I ran.

```
  user audio ──▶ ASR (faster-whisper) ──┬──▶ EMOTION (wav2vec2) ──┐
   22050 Hz      tiny.en partials       │     6-class, per turn   │
   streamed      base.en final          │                         ▼
                                        │              LM (Llama-3.2-1B-4bit)
       ┌────────────────────────────────┘              conditioned on emotion
       │                                                         │
       ▼                                                         ▼
  CUE at 400 ms ────────────────────────────────▶  TTS with prosody preset
  ("uh", real audio, fills the gap)                  chosen by emotion
                                                             │
                                                             ▼
                                                    sounddevice, one stream
```

| Stage | What it is | The study behind it |
|---|---|---|
| **Perception** | Fine-tuned `wav2vec2-base`, 6 classes, actor-disjoint split | [emotion-label-ceiling](https://github.com/abhaymettu/emotion-label-ceiling) |
| **Timing** | Streaming loop, endpointer, per-stage budget, cue machinery | [aliveness-threshold](https://github.com/abhaymettu/aliveness-threshold) |
| **Generation** | Prosody presets on two TTS engines | [expressive-tts-audit](https://github.com/abhaymettu/expressive-tts-audit) |
| **The open path** | F0-contour transplant onto fast synthesis | [prosody-transplant](https://github.com/abhaymettu/prosody-transplant) |

**Perception carries a stated confidence limit, and that is a finding rather
than a hedge.** On audio-only CREMA-D, listener agreement is Krippendorff
α = 0.265 [0.259, 0.272], and the reliability ceiling that implies is 0.727
[0.723, 0.732] — no model scored against what listeners actually heard can
exceed it. Across three seeds there is a mean 25.6-point gap between predicting
the actor's *intent* and predicting what listeners *heard*. The checkpoint in
this loop is the s0 run: **74.9% against actor intent, 52.2% against crowd
consensus.** So "the agent detected sadness" is a claim about a label with
known, measured softness underneath it.

**Timing is measured against a real prior number, and that number moved.** The
gap definition here is the one in aliveness-threshold, unchanged: *onset of
agent speech minus offset of user speech, both silence-trimmed*, with the user's
offset re-measured by `vendor.audio.segments(merge_gap_ms=30, min_len_ms=20)` —
the end of the last speech segment, **not** the moment the endpointer noticed.
That repo first measured a median gap of 1452 ms over n = 40 on this laptop, then
**re-ran the unchanged code and got 807 ms [779–895] over n = 100**, and could
not attribute the difference to any variable it records. Its own conclusion is
that cross-run comparison to the 1452 ms figure is invalid. §1 therefore compares
against a control measured in this repo, in the same session, on the same
machine — not against either published figure.

**Generation is bounded by the engine, not by the mapping.** Piper exposes no
pitch control whatsoever. That is not a tuning problem and no preset fixes it.

---

## 1. Latency, with the emotion classifier in the path

Configuration A: TTS-rendered prompts, Piper output, classifier serial. Same gap
definition, same machine and same prompt set as the aliveness-threshold loop, so
the stage table below lines up row for row with that repo's.

**gap = 937 ms, IQR [866–989], n = 20** (mean 928, sd 75, min 819, max 1098).

| stage | ms, median [IQR] | note |
|---|---|---|
| endpoint hangover | 363 [361–365] | design constant `HANGOVER_MS = 350`; a floor, not a measurement |
| wait for in-flight partial | 6 [3–32] | final decode waits for the partial worker to release the cores |
| ASR final decode | 236 [233–250] | `base.en`, int8, CPU |
| **emotion classifier** | **45 [42–53]** | **the new stage** — wav2vec2-base, one forward pass, MPS |
| LM time to first token | 102 [98–111] | 4-bit 1B on MPS |
| LM to end of first sentence | 60 [43–69] | |
| TTS synthesis | 66 [58–90] | Piper, whole utterance |
| handoff to audio callback | 4 [2–4] | |

Startup, excluded from every turn: 8.8 s model load, then 0.6 s warmup (one ASR
decode, one emotion forward pass, one LM generation, one TTS synthesis). Turn 0
is warm. Output device BlackHole 2ch, reported latency 5.8 ms.

### What emotion detection actually costs

Three ways of asking, because the in-path stage timer alone would be the
flattering one:

| arm | schedule | gap, median [IQR] | emotion stage in path |
|---|---|---|---|
| **A** | serial, after ASR | 937 [866–989] | 45 [42–53] |
| **D** | overlapped with the ASR decode | 837 [810–882] | **0.0** |
| **E** | ablated, no classifier at all | 850 [820–916] | — |

n = 20 each. The classifier's own forward pass is 45 ms [42–53] in A and
43 ms [41–47] in D — the work is identical; only where it sits changes.

- **Serial: 45 ms as a stage, +87 ms by subtraction (A − E).** The gap between
  those two numbers is inside the between-run spread (sd 75 and 96), so 45 ms
  is the number I trust and +87 ms is the number I will not argue is precise.
- **Overlapped: free.** D is 13 ms *faster* than the no-emotion control, which
  is noise, not a speedup. The classifier fits entirely inside the 236 ms ASR
  decode with room to spare, so it costs the listener nothing.

That is the engineering answer: **on this hardware, emotion detection does not
have to cost anything.** Ship it on a worker thread started at the endpoint.

This is the same move aliveness-threshold since made on the whole downstream —
its `--fast` mode starts the ASR final decode, the LM and the TTS *inside* the
endpointer's 350 ms hangover rather than after it, discarding the result if more
speech arrives, and reaches a 386 ms median gap with 0 false endpoints in 240
turns. That is not applied here: this loop still waits out the full hangover
before doing anything, so ~360 ms of every gap above is dead time by
construction. Folding the emotion classifier into a speculative-decode loop is
the obvious next step, and it is unmeasured.

### Why there is no headline comparison to the 1452 ms baseline

937 ms against 1452 ms would be 515 ms faster *with an extra stage added*, and it
would be meaningless. **That baseline does not reproduce.** aliveness-threshold
re-ran its own unchanged code and got 807 ms [779–895] over n = 100 against the
1452 ms it recorded twenty minutes earlier, and it explicitly checked and ruled
out machine load: the run that returned 1452 ms recorded load average 16.79, the
run that returned 883 ms recorded 16.97. Whatever moved is not captured by any
variable either harness records, and its own conclusion is that any cross-run
comparison to 1452 ms is invalid.

So I do not make one. **Every claim in §1 is against arm E — no classifier at
all — run in the same session, on the same machine, interleaved with the arms it
is being compared to.** For the record: E's 850 ms [820–916] over n = 20 sits
comfortably inside that repo's re-measured 807 ms [779–895] over n = 100, which
is the consistency check you would want before believing anything else here.

The reason this matters beyond bookkeeping: a 515 ms "improvement" was sitting
there to be claimed, from a real run, using an identical gap definition, and it
would have been entirely fictitious.

### The cue fills the front of the gap

The cue ("uh", real audio from `vendor/cues.py`, 305 ms) is armed the instant
the endpointer fires, targeting 400 ms after the user stops.

| arm | first agent audio | reply | cue fired |
|---|---|---|---|
| A (TTS prompts) | **420 ms [417–424]** | 937 ms | 20/20 |
| B (real actors, Piper) | 345 ms [181–394] | 804 ms | 24/24 |
| C (real actors, `say`) | 343 ms [188–400] | 1410 ms | 24/24 |

On A the cue lands within 20 ms of target on every turn. On real human speech
it is much more variable, for a reason worth stating: the cue is scheduled off
the *endpointer's estimate* of speech offset, while the reported number is
measured against the silence-trimmed *true* offset, and on real speech those
two diverge (see §5). The cue is never counted as agent speech onset — `gap_ms`
always measures the reply.

---

## 2. The tradeoff: fast and flat, or expressive and slow

Same loop, same 24 real-actor turns, only the TTS engine changed.

| | Piper `en_US-lessac-medium` | macOS `say` (Alex) |
|---|---|---|
| **TTS stage** | **44 ms [33–62]** | **678 ms [656–689]** |
| **total gap** | **804 ms [601–856]** | **1410 ms [1200–1470]** |
| synthesis alone, fixed sentences (n=90) | 37 ms [34–42] | 662 ms [650–677] |
| mean F0 span across presets | **8.3 Hz** | **245.8 Hz** |
| F0 separates by preset? | **no** (p = 0.67 range, p = 0.99 sd) | **yes** (p = 1.4e-16, 1.1e-16) |

**634 ms of extra synthesis buys 30× the pitch range.** That is the whole
decision, and there is no third option installed on this laptop.

One note against the prior measurement: aliveness-threshold recorded the same
`say` path at **2617 ms** per utterance, four times slower than the 678 ms here.
That run was under load average 8–16; this one at ~6, on *longer* replies
(2170 ms of audio vs 1765 ms). Each `say` call is a cold subprocess, and
subprocess spawn is exactly what machine load punishes. Both numbers are real
runs; the load is the difference.

---

## 3. Does the output measurably differ by emotion?

Two questions, deliberately separated, because conflating them is the easiest
way to overclaim here.

### 3a. What the control surface can do at all (classifier not involved)

Same five sentences, every preset, 90 utterances per engine. Text held constant,
so any difference is the knobs and not word choice. Kruskal-Wallis across the
four presets:

| feature | Piper medians | H, p | `say` medians | H, p |
|---|---|---|---|---|
| f0_mean | 188.9 / 191.7 / 193.8 / 197.2 | 9.2, **0.027** | 93.4 / 215.0 / 267.5 / 339.2 | 81.9, 1.2e-17 |
| f0_sd | 48.5 / 49.7 / 50.6 / 50.9 | 0.1, **0.99** | 11.6 / 33.1 / 41.2 / 50.9 | 77.4, 1.1e-16 |
| f0_range | 150.7 / 153.2 / 155.9 / 162.8 | 1.6, **0.67** | 40.5 / 112.2 / 138.9 / 176.2 | 77.0, 1.4e-16 |
| speech_rate | 3.49 / 3.66 / 3.95 / 4.17 | 20.7, 0.00012 | 3.35 / 3.68 / 3.68 / 4.55 | 36.1, 7.2e-08 |
| duration | 1.44 / 1.52 / 1.68 / 1.74 | 35.9, 7.7e-08 | 1.32 / 1.63 / 1.63 / 1.79 | 49.8, 9e-11 |
| rms_mean_db | −25.6 / −23.7 / −23.4 / −23.1 | 44.0, 1.5e-09 | −22.6 / −17.9 / −17.9 / −17.7 | 60.5, 4.6e-13 |
| hnr | 13.3 / 13.6 / 13.7 / 14.2 | 5.1, **0.17** | 10.9 / 17.0 / 17.7 / 17.9 | 63.9, 8.8e-14 |

**Piper's expression is rate, duration and level. It is not intonation.**
f0_sd at p = 0.99 is about as null as a result gets: across four presets meant
to sound excited, sad, emphatic and neutral, the pitch variability is
indistinguishable. This independently reproduces expressive-tts-audit's finding
on a different sentence set.

### 3b. End to end — grouped by what the classifier actually detected

The reply wavs from the 24-turn real-actor runs, grouped by the emotion detected
on that turn's *user* audio. This is the real claim, and it inherits every
classifier error.

| feature | Piper: H, p | `say`: H, p |
|---|---|---|
| f0_mean | 7.7, **0.17** | 21.8, **0.00058** |
| f0_range | 7.7, **0.17** | 21.7, **0.00061** |
| rms_mean_db | 9.1, 0.10 | 15.3, 0.0093 |
| speech_rate | 3.0, 0.71 | 1.9, 0.86 |
| duration | 14.0, 0.015 | 13.8, 0.017 |

On `say`, F0 tracks the detected emotion cleanly: median f0_mean 92.6 Hz when
fear was detected, 93.6 Hz for sad, 216 Hz for disgust and neutral, 269 Hz for
anger, 336 Hz for happy. **On Piper it does not separate at all.**

Two honest caveats. The reply *text* differs between turns here, unlike §3a, so
the duration and rate rows are partly the sentence and not the prosody — which
is why they are the two rows that behave the same on both engines. And n is 3–5
per group; read §3a for the clean version and this table for the end-to-end one.

**The answer to "does the expressive output measurably differ by detected
emotion" is: yes on `say`, no on Piper.** The system works; one of its two
available voices cannot express what it is asked to.

---

## 4. Does the classifier survive being in a live loop?

Fed by real held-out CREMA-D actors — the same actors the checkpoint was
evaluated on — through the full live path (22050 Hz capture, endpointer crop,
resample back to 16 kHz, 4 s cap) instead of straight off disk.

| | accuracy vs actor intent | n |
|---|---|---|
| **in the live loop** | **0.792** [0.595, 0.908] | 24 |
| straight off disk, same actors, this loader | 0.730 | 300 |
| reported by the training run | 0.749 | 1640 |

**No detectable degradation from being in the loop.** The three intervals
overlap heavily and n = 24 is small — this rules out a *collapse*, not a
10-point drop. Per class: anger 1.00, disgust 1.00, fear 0.75, happy 0.75,
neutral 0.75, sad 0.50 (n = 4 each). Sad is the weak class, going to fear and
anger — the same confusion the offline run reports.

The 0.730 off-disk figure also confirms the vendored loader is the same model,
not a differently-wired copy of it.

### And now the part that should stop anyone from trusting it on real audio

The classifier was trained on **acted** speech. Point it at speech with no acted
emotion in it and it does not abstain — it commits.

- **Run A, 20 turns of flat synthetic prompt audio: it said `anger` 20 times out
  of 20**, confidence up to 0.96.
- **Probe, 30 synthetic utterances** across five texts and all six preset
  renderings, no ground truth available: `anger` 23, `neutral` 5, `sad` 2,
  **median confidence 0.73**.

A single unchanging TTS voice is not angry. The model is reading something —
spectral character, level, the absence of natural variability — and mapping it
confidently onto an emotion label. **Its live outputs are not reliable and are
not presented as reliable.** The CREMA-D numbers above are the ceiling of what
it can do, measured on the distribution it was built for; anything outside that
distribution is unverified at best.

---

## 5. The endpointer is much sloppier on humans than on a TTS voice

Found while reading the real-actor runs, reported because it changes how to read
every gap above.

| input | hangover, median | range | cut the user off |
|---|---|---|---|
| TTS-rendered prompts | 363 ms | 355 – 370 | 0/20 |
| real CREMA-D actors (Piper) | 288 ms | **1 – 699** | 0/24 |
| real CREMA-D actors (`say`) | 288 ms | **2 – 699** | 0/24 |

`endpoint_hangover_ms` is `t_endpoint − true speech offset` and should sit near
the 350 ms design constant. On a TTS voice it does, within 15 ms. On real people
it ranges from 1 ms to 699 ms — a 700 ms swing on a stage that is a third of the
whole gap. The live endpointer arms on chunk RMS against a running peak with an
absolute floor of −45 dBFS, tuned for a voice at steady level; a person trailing
off at the end of a sentence drops under that floor while the measurement VAD
(−55 dBFS) still counts them as speaking.

Nothing was truncated in these runs (0/48), but the margin is thin, and this is
why the real-actor gaps have IQR 255 ms against the TTS arm's 123 ms. **The
single cheapest improvement available to this system is a better endpointer**,
not a faster model.

---

## 6. Honest limitations

1. **The emotion classifier is trained on acted speech and is unreliable off
   that distribution.** §4 measures this rather than warning about it: 20/20
   confident `anger` on a neutral synthetic voice. It has not been evaluated on
   spontaneous conversational audio at all, because no labelled spontaneous
   audio was available on this machine. Treat every live emotion output as
   unverified.
2. **The label itself is soft.** Audio-only listener agreement on CREMA-D is
   α = 0.265, and the reliability ceiling is 0.727. "74.9% accurate" means
   74.9% at reproducing what one actor was *told* to perform, which is 25.6
   points away from what listeners actually *hear*.
3. **The user is an actor reading a card, not a person in a conversation.** The
   real-actor arms are a large improvement on driving the loop with a TTS voice,
   and they are still 12 fixed sentences performed on instruction. Disfluency,
   overlap, room noise and self-repair are all absent.
4. **No human ever listened to the expressive output.** Every §3 result is
   acoustic. The claim is that the conditions differ *measurably*, not that they
   differ *audibly*, and certainly not that they sound like the emotions they
   are named after.
5. **Prosody control is crude, and on Piper it is largely absent.** Four presets,
   no pitch knob on the fast engine, `pmod` inert on the slow one, and pause
   structure untested — every utterance here has zero detected pauses ≥ 100 ms,
   so the three pause features carry no information.
6. **The emotion→preset mapping is a judgement call, not a result.** Six classes
   onto four presets; `disgust` is mapped to neutral because none of the four is
   a good-faith rendering of it, and inventing one would be worse than
   abstaining.
7. **n is small.** 20–24 turns per arm, 3–5 per emotion group in §3b. Every
   figure is reported with n and spread; none of them should be read as a point
   estimate.
8. **ASR is chunked, not streaming, and the final decode throws away the
   partial's work.** That waste is inside every gap reported here. Inherited from
   aliveness-threshold, unfixed.
9. **The microphone path is only partly verified.** `--mic` opens a real input
   stream, endpoints and transcribes; no reported number comes from it. The full
   acoustic loop — speakers into the room into the mic — was not tested.
10. **The machine itself is not a stable measurement instrument.** The loop this
    one is built on measured 1452 ms and then 807 ms from identical code twenty
    minutes apart, with no recorded variable explaining it. Every number here was
    taken in one session with its own internal control, which is the mitigation,
    not a fix. Numbers from different sessions in this repo should not be
    compared to each other either.
11. **The F0 transplant did not land.** [prosody-transplant](https://github.com/abhaymettu/prosody-transplant)
    was still scaffolding when this ran (commit `490880c`), so the expressive-
    at-Piper-latency option is unmeasured and the tradeoff in §2 stands as the
    live result. If it lands, it drops into `vendor/tts.py` as a third backend
    and §2 gets a third column.

---

## Run it

```bash
uv venv --python 3.12 && uv pip install -e .

# the piper voice (gitignored, ~63 MB)
python -m piper.download_voices --download-dir models/piper en_US-lessac-medium

# the emotion checkpoint (gitignored, 378 MB) -- from emotion-label-ceiling
export EXPRESSIVE_S2S_EMOTION_CKPT=/path/to/wav2vec2-base-intended_emotion-actor-s0.pt

./run_all.sh          # every number in this README, in order
```

Individual pieces:

```bash
.venv/bin/python -m s2s.loop selfcheck --n 3        # real turns + all assertions
.venv/bin/python -m s2s.loop batch --n 20 --out runs/x.json
.venv/bin/python -m s2s.loop batch --mic --n 3      # talk to it
.venv/bin/python -m s2s.expression selfcheck
.venv/bin/python summarize.py                       # every headline number
```

Results live in [`runs/`](runs) (one JSON per turn, per arm) and [`out/`](out)
(the analyses). `summarize.py` prints every figure in this README from those
files; if a number is in the README and not in that output, it should not be in
the README.

### The self-check

`s2s.loop selfcheck` runs real turns and asserts, for each one:

- every stage timer exists, is a number, is not negative, and **the stages sum
  to the gap** within one output block (5.8 ms + 1 ms);
- the cue landed *before* the reply and was not cut off mid-word;
- the preset spoken matches the emotion detected;
- and **the classifier was fed this turn's audio.** The captured array is
  fingerprinted at capture time, the classifier returns the fingerprint of what
  it actually saw, and the two must match — and must differ across turns. A loop
  that conditions turn *N*'s reply on turn *N−1*'s audio looks perfectly healthy
  in the logs and is worthless; this is the assertion that catches it.

`s2s.expression selfcheck` asserts the acoustic feature path is actually
connected to the synthesizer: a rate change must move duration on Piper, and a
pitch change must move F0 on `say`.

## Where the code came from

Nothing here re-implements what the sibling repos already measured. Files were
copied in with an attribution header naming the source, rather than imported
through a path hack, so this repo runs standalone and the measurement code
cannot drift underneath it.

| File | Source | Modified? |
|---|---|---|
| `vendor/audio.py` | aliveness-threshold `harness/audio.py` | no |
| `vendor/cues.py` | aliveness-threshold `harness/cues.py` | no |
| `vendor/tts.py` | aliveness-threshold `harness/tts.py` | yes — `synth()` takes a prosody `cfg` and passes it to either engine |
| `vendor/features.py` | expressive-tts-audit `features.py` | yes — its self-check dropped (it imported that repo's `render.py`); `extract()` and `FEATURES` byte-identical |
| `s2s/emotion.py` | emotion-label-ceiling `modeling/finetune.py`, `modeling/common.py` | model class and label vocab verbatim; loader and per-turn API new |
| `s2s/prosody.py` | expressive-tts-audit `render.py` | preset values verbatim; emotion→preset mapping new |
| `s2s/loop.py` | aliveness-threshold `live/loop.py` | adapted — emotion stage, conditioning, prosody, cue scheduling, named playback |

Model weights are not in git.

---

Every number above came from a run that happened on this machine — Apple M4 Pro,
24 GB, macOS, Python 3.12 — on 2026-08-26. Machine load is recorded in every
result JSON, because the ASR stage is CPU-bound and the load is part of the
result.
