# expressive-s2s

**An agent that hears how you sound and answers in a voice that matches, in a
median 558 ms** [538–573], n = 20 — against its own serial control at 895 ms
[876–922] in the same session. With a faster final decode it is **464 ms**
[412–521] at a cost of 0.033 WER; with an F0 contour transplanted onto the reply
so the voice actually changes pitch, **661 ms** [639–680].

Speech in → emotion estimated from the audio → reply generated conditioned on
that emotion → reply spoken with matching prosody, with a non-verbal cue placed
in the gap while the rest of the pipeline is still working.

It runs end to end on one laptop, and it is measured end to end. **Listen first.**
Real turns, laid out on the real clock, on each of the three TTS engines — the
silences you hear are the silences that were measured:

- [`demo/exchange.wav`](demo/exchange.wav) — Piper, serial loop. Fast (reply at
  501–1080 ms) and, as §3 shows, acoustically flat regardless of what was
  detected.
- [`demo/exchange-expressive.wav`](demo/exchange-expressive.wav) — macOS `say`,
  serial loop. You can hear the voice change with the detected emotion, and you
  can hear the 600 ms it costs (reply at 1133–1562 ms).
- [`demo/exchange-transplant.wav`](demo/exchange-transplant.wav) — **Piper with
  the F0 contour transplanted onto it, on the fast path.** The voice changes and
  the reply lands at 351–1034 ms. This is the one to listen to.

The first two are the same four turns: detected anger → emphatic, disgust →
neutral, fear → sad, sad → sad, with a filled-pause cue in the gap each time.
The third is four turns from the transplant run, chosen the same way (one per
detected emotion) and with **no cue at all** — see §1.1 for why the fast path
stops firing it.

> **The headline, in four numbers.** The whole downstream pipeline — final
> decode, emotion classifier, LM and TTS — now runs **inside the endpointer's
> 350 ms hangover** instead of after it, which takes the gap from 895 ms to
> **558 ms** [538–573] with 0 false endpoints and no loss of transcript accuracy
> (§1.1). Emotion detection costs **45 ms** [42–53] as a serial stage and
> **0 ms** when overlapped — free to within run-to-run noise (§1.2).
> The classifier survives the loop: **79.2%** [59.5, 90.8] against actor intent
> inside the live pipeline, n = 24, versus 74.9% offline.
> And the tradeoff this repo used to report — fast and flat, or expressive and
> slow — **is broken by a third engine**: an F0 contour transplanted onto Piper's
> output spans **171 Hz of mean F0 for 50 ms of extra synthesis**, where macOS
> `say` spans 246 Hz for 595 ms (§2).
>
> Two things that did not get better, and are measured rather than warned about:
> the classifier still calls flat synthetic audio `anger` — 20 out of 20 turns in
> the run that first caught it, and 12 to 16 out of 20 in three more runs this
> session, at up to 0.96 confidence (§4); and the endpointer still cuts real
> people off — **3 of 24 CREMA-D turns, identically with the fast path on and
> off** (§1.3).

---

## The system, and the evidence behind each part

This is one project. The repo you are reading is the system; four sibling repos
are the studies that decided how it is built. Every design choice below points
at a measurement, and every measurement is one I ran.

```
                        80ms quiet                     endpointer fires
                            │                          (350ms quiet)
  user audio ───────────────┼──────────────────────────────┼──▶
   22050 Hz streamed        │                              │
   tiny.en partials ────────┤ partials stop here           │
                            ▼                              │
                    ┌───────────────────────────────┐      │
                    │  SPECULATION, inside the      │      │
                    │  hangover, on the audio so far│      │
                    │                               │      │
                    │  ASR final (base.en) ──┐      │      │
                    │  EMOTION (wav2vec2) ───┤      │      │
                    │   6-class, per turn    ▼      │      │
                    │      LM (Llama-3.2-1B-4bit)   │      │
                    │      conditioned on emotion   │      │
                    │              │                │      │
                    │              ▼                │      │
                    │   TTS + prosody for that      │      │
                    │   emotion (piper / +F0        │      │
                    │   transplant / say)           │      │
                    └───────────────┬───────────────┘      │
                                    │                      ▼
    speech resumed? ──▶ stale, discard, fall back    play the reply
                                                     sounddevice, one stream

  CUE ("uh", real audio) is armed for 400ms after the user stops, and stands
  down if a speculation is in flight -- at a 558ms gap it would be cut off
  mid-word by the reply it was covering.
```

| Stage | What it is | The study behind it |
|---|---|---|
| **Perception** | Fine-tuned `wav2vec2-base`, 6 classes, actor-disjoint split | [emotion-label-ceiling](https://github.com/abhaymettu/emotion-label-ceiling) |
| **Timing** | Streaming loop, endpointer, per-stage budget, cue machinery | [aliveness-threshold](https://github.com/abhaymettu/aliveness-threshold) |
| **Generation** | Prosody presets on two TTS engines | [expressive-tts-audit](https://github.com/abhaymettu/expressive-tts-audit) |
| **Expression at speed** | F0-contour transplant onto fast synthesis | [prosody-transplant](https://github.com/abhaymettu/prosody-transplant) |

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

**Generation was bounded by the engine, not by the mapping.** Piper exposes no
pitch control whatsoever. That is not a tuning problem and no preset fixes it —
so instead of a preset, a knob was bolted on from outside the engine (§2). That
is the one place in this system where the answer to a measured ceiling was to
route around it rather than report it.

---

## 1. Latency

### 1.1 The fast path: the pipeline runs inside the hangover

The endpointer waits 350 ms of trailing silence before it calls the turn over,
and the serial loop does nothing during it. `--fast` starts the final decode,
the classifier, the LM and the TTS after **80 ms** of silence, on the audio
captured so far. Nothing is guessed: everything between that snapshot and the
endpoint is silence by definition, so the speculative result is the transcript
the serial path would have produced. The only bet is that the talker has
stopped, and it is checked — if speech resumes, the snapshot is stale, the work
is thrown away and the turn falls back to the serial path.

The mechanism is [aliveness-threshold](https://github.com/abhaymettu/aliveness-threshold)'s
`FastPath`, vendored. **The change here is that the emotion classifier is inside
the speculation too**, because it conditions both the LM prompt and the TTS
preset — so the reply cannot be built early unless the perception stage is built
early with it.

Every arm below: TTS-rendered prompts, Piper output, n = 20, one session, one
machine, interleaved. Control first, control again at the end.

| arm | schedule | gap, median [IQR] | mean, sd | false endpoints | WER |
|---|---|---|---|---|---|
| **F0** | control, serial | **895 ms [876–922]** | 908, 52 | 0/20 | 0.067 |
| **F1** | `--fast --arm 80` | **624 ms [597–661]** | 640, 69 | 0/20 | 0.000 |
| **F2** | + `--emotion-parallel` | **558 ms [538–573]** | 559, 23 | 0/20 | 0.000 |
| **F3** | + `tiny.en` final decode | **464 ms [412–521]** | 468, 54 | 0/20 | 0.033 |
| **F0b** | the control again, last | **861 ms [841–877]** | 873, 58 | 0/20 | 0.000 |

**337 ms, or 38% of the gap, was dead hangover time.** F2 is the configuration
to ship: everything the listener waits for is real work, nothing is guessed, and
it never lost a turn.

- **F0b is the honesty check.** The control drifted 34 ms across the session
  (895 → 861), which is a fifth of the effect. The comparison holds.
- **WER goes down, not up, on the fast path.** 0.067 on the control against
  0.000 on F1 and F2 — one control turn's transcript differed. That is noise in
  both directions; what it rules out is the fast path being *worse*, which is
  the only thing this column is for. F3's 0.033 is the honest price of `tiny.en`
  and lands exactly on the 0.033 that repo measured for the same swap.
- **Speculation almost never misses on a TTS talker.** F2 launched 20 pipelines
  for 20 turns and served all 20 of them speculatively. F1 launched 23 for 20 —
  three turns re-armed after a within-utterance dip and one fell back to serial,
  which is why F1's sd is 69 against F2's 23.

Where the 337 ms went, stage by stage. `charged` is the critical path — each
stage billed only for what it added beyond everything already elapsed, so work
hidden inside the hangover reads 0 and the stages still sum exactly to the gap.
`work` is what that stage cost on its own clock.

| stage | F0 charged | F2 charged | F2 work |
|---|---|---|---|
| endpoint hangover | 367 [356–373] | 370 [361–374] | — |
| wait for in-flight partial | 6 [2–38] | **0** | — |
| ASR final decode | 250 [247–258] | **0** | 246 [242–250] |
| **emotion classifier** | 40 [37–47] | **0** | 45 [42–52] (fwd pass) |
| LM time to first token | 105 [98–114] | 84 [67–87] | 104 [102–109] |
| LM to end of first sentence | 56 [42–59] | 52 [41–59] | 52 [41–59] |
| TTS synthesis | 67 [59–74] | 60 [53–68] | 60 [53–68] |
| handoff to audio callback | 2 [1–4] | 3 [2–5] | — |

The decode and the classifier disappear from the bill entirely. The LM is the
stage that only *partly* hides: 104 ms of prefill, of which 84 ms is still on the
critical path, because by the time the decode and the classifier have finished
inside the hangover there is only ~20 ms of it left to hide in.

**The cue stops firing, and that is correct.** The filled pause is armed for
400 ms after the user stops and is 305 ms long; at a 558 ms gap it would be cut
off mid-word by the reply it was covering. So the cue now checks whether a
speculation is in flight and stands down if one is: 20/20 on both control arms,
**0/20 on every fast arm**. `first_audio_ms` and `gap_ms` are therefore the same
number on the fast path — there is nothing in front of the reply because there is
no longer a gap worth filling.

### 1.2 What emotion detection costs

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
have to cost anything.** Ship it on a worker thread started at the endpoint —
and, per §1.1, start that thread 270 ms before the endpoint rather than at it.

Arms A/D/E above were measured before the fast path existed; §1.1's F0/F1/F2/F3
are a separate, later session with its own control. Do not read a gap from one
table against a gap from the other — that is the mistake this repo exists to not
make twice. The arms are directly comparable *within* each table only.

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

### 1.3 Does `--arm 80` cut real people off?

**This is the caveat that travels with every number in §1.1, and it must not be
dropped.** `--arm 80` works on a TTS talker because that talker has a 65 ms
maximum internal pause, so 80 ms of silence really does mean the turn is over.
§5 of this repo measured real CREMA-D actors triggering endpoint hangovers
anywhere from **1 ms to 699 ms**. A person who pauses for 100 ms mid-sentence
arms the speculation on half an utterance.

So it was run on real people. Same 24 held-out CREMA-D actor clips, serial
control and fast arm back to back:

| arm | gap, all turns | gap, excluding truncated | false endpoints | speculations launched |
|---|---|---|---|---|
| **H0** serial control | 772 ms [578–833] | **800 ms [732–836]**, n = 21 | **3/24** | 0 |
| **H1** `--fast --arm 80` | 499 ms [400–722] | **511 ms [442–732]**, n = 21 | **3/24** | 51 for 24 turns |

Three things, in order of how much they matter.

1. **The fast path did not add a single false endpoint.** 3 of 24 on the fast
   arm, and 3 of 24 on the serial control — two of them the same clips. The
   endpointer cuts real people off at 12.5% on this material, and it does so
   with the speculation switched off. `--arm 80` is not what breaks; the
   endpointer was already broken and this is the first measurement that could
   see it.
2. **It is not a near-miss when it happens.** The three truncated turns lost
   1760, 2270 and 1700 ms of speech and transcribed as `"You"`. The shortfall
   distribution is bimodal — every other turn lost exactly 0 ms. There is no
   grey zone here to tune a threshold into.
3. **The speculation is self-limiting on human speech, which is the good news.**
   Only 14 of 24 turns were served speculatively, from 51 launched pipelines:
   a person pauses, the snapshot goes stale, the work is discarded and the turn
   falls back to serial. That is why H1's IQR runs to 732 ms — its p75 *is* the
   serial number. The cost of being wrong is wasted CPU, not a wrong answer.

**Why §5 says 0/48 and this says 3/24.** They are different detectors and only
one of them can see truncation. `s2s/endpoint_check.py` flags a turn when
`endpoint_hangover_ms < 0` — the endpointer firing before the silence-trimmed
offset *of the audio it captured*. But truncating the buffer moves that offset
earlier too, so the hangover stays positive and the detector reads clean; on all
three truncated turns here it read +5 to +8 ms. The check in §1.1/§1.3 compares
against the offset measured on the **source file, before the loop ran**, which
is aliveness-threshold's definition and the only reference that survives the
loop cutting the buffer short. §5's numbers are not wrong; the question they
answer is narrower than it looks, and this is the honest version of it.

**What is still not tested:** spontaneous conversational speech from a person
talking to this laptop through a microphone. CREMA-D actors read a fixed card
and stop cleanly. Real disfluency, self-repair and turn-holding are absent from
every number above, and the 3/24 is therefore a floor on the false-endpoint
rate, not an estimate of it.

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

## 2. The tradeoff, and the third engine that breaks it

This repo used to report a clean, unhappy choice: an engine that is fast and
acoustically flat, against an engine that is expressive and 634 ms slower per
utterance. There is now a third option, and it is not a compromise between them.

**The transplant.** Piper exposes no pitch knob, so one is bolted on from
outside: render with Piper, decompose the waveform with the WORLD vocoder into
F0 / spectral envelope / band aperiodicity, replace the F0 track with the contour
you asked for, resynthesise. Rewriting F0 while holding the spectral envelope
fixed moves pitch without moving formants. Duration and level are *not* done this
way — Piper's own `length_scale` and `volume` do those natively and correctly,
and paying a vocoder for work the engine already does right would be paying
twice. The mechanism is [prosody-transplant](https://github.com/abhaymettu/prosody-transplant),
vendored into `vendor/transplant.py`.

Same five sentences, every preset, 90 utterances per engine, text held constant:

| | Piper `en_US-lessac-medium` | **+ F0 transplant** | macOS `say` (Alex) |
|---|---|---|---|
| **synthesis, fixed sentences (n=90)** | **33 ms [31–37]** | **83 ms [78–90]** | **628 ms [623–633]** |
| mean F0 span across presets | 6.0 Hz | **171.2 Hz** | 245.8 Hz |
| F0 *range* span across presets | 15.5 Hz | **237.2 Hz** | 135.7 Hz |
| F0 separates by preset? | **no** (p = 0.61 range, 0.96 sd) | **yes** (p = 4.7e-17, 4.6e-17) | **yes** (p = 1.4e-16, 1.1e-16) |
| **total gap, serial loop (n=20)** | 895 ms [876–922] | 962 ms [918–978] | — |
| **total gap, fast path (n=20)** | **558 ms [538–573]** | **661 ms [639–680]** | — |

**50 ms of extra synthesis buys 29× the pitch range.** Against `say`'s 595 ms
for 41×. In the live loop the transplant costs **103 ms** on the fast path
(558 → 661 ms) and 67 ms on the serial one, where the same expression from `say`
cost 606 ms in this repo's earlier real-actor arms (804 → 1410 ms).

Two things it is not:

- **It is not free.** 103 ms is a real cost, and the WORLD round trip has an
  operating range: prosody-transplant measured it holding from 0.60× to 1.80× of
  the source F0, and the `sad` preset is deliberately clipped to 0.72× rather
  than the 0.433× `say`'s own sad register would imply, because below 0.60× the
  F0 tracker starts dropping voiced frames. That is a real limit and it is in
  the shipped presets, not hidden behind them.
- **It is not verified by ear.** Like every other expression number here, this
  is acoustics. See limitation 4.

The two original engines, on the same 24 real-actor turns, for the record:

| | Piper `en_US-lessac-medium` | macOS `say` (Alex) |
|---|---|---|
| **TTS stage** | **44 ms [33–62]** | **678 ms [656–689]** |
| **total gap** | **804 ms [601–856]** | **1410 ms [1200–1470]** |
| mean F0 span across presets | **8.3 Hz** | **245.8 Hz** |
| F0 separates by preset? | **no** (p = 0.67 range, p = 0.99 sd) | **yes** (p = 1.4e-16, 1.1e-16) |

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
four presets, medians in preset order **neutral / sad / emphatic / excited**
(`out/sweep-three-engines.json`, this session):

| feature | Piper | H, p | **+ transplant** | H, p | `say` | H, p |
|---|---|---|---|---|---|---|
| f0_mean | 191.9 / 189.4 / 192.9 / 195.4 | 9.4, **0.024** | 197.6 / 139.9 / 249.1 / 311.1 | 81.6, 1.4e-17 | 215.0 / 93.4 / 267.5 / 339.2 | 81.9, 1.2e-17 |
| f0_sd | 48.9 / 48.6 / 51.7 / 49.8 | 0.3, **0.96** | 48.8 / 21.4 / 92.9 / 98.1 | 79.2, 4.6e-17 | 33.1 / 11.6 / 41.2 / 50.9 | 77.4, 1.1e-16 |
| f0_range | 155.0 / 155.0 / 170.5 / 161.3 | 1.8, **0.61** | 154.5 / 69.1 / 303.1 / 306.3 | 79.1, 4.7e-17 | 112.2 / 40.5 / 138.9 / 176.2 | 77.0, 1.4e-16 |
| speech_rate | 4.0 / 3.6 / 3.5 / 4.2 | 18.3, 0.00038 | 4.0 / 3.9 / 3.9 / 4.4 | 9.5, 0.023 | 3.7 / 3.4 / 3.7 / 4.5 | 36.1, 7.2e-08 |
| duration | 1.5 / 1.7 / 1.7 / 1.4 | 33.5, 2.5e-07 | 1.5 / 1.6 / 1.5 / 1.4 | 19.4, 0.00023 | 1.6 / 1.8 / 1.6 / 1.3 | 49.8, 9e-11 |
| rms_mean_db | −23.8 / −25.6 / −23.5 / −23.4 | 39.2, 1.6e-08 | −23.0 / −23.3 / −23.9 / −24.0 | 11.3, 0.010 | −17.9 / −22.6 / −17.9 / −17.7 | 60.5, 4.6e-13 |
| hnr | 13.7 / 14.0 / 13.7 / 13.3 | 5.9, **0.11** | 14.3 / 14.4 / 13.5 / 14.6 | 5.9, **0.12** | 17.7 / 10.9 / 17.9 / 17.0 | 63.9, 8.8e-14 |

**Piper's expression is rate, duration and level. It is not intonation.**
f0_sd at p = 0.96 is about as null as a result gets: across four presets meant
to sound excited, sad, emphatic and neutral, the pitch variability is
indistinguishable. This independently reproduces expressive-tts-audit's finding
on a different sentence set, and it reproduces this repo's own earlier run of the
same sweep (p = 0.99 then, p = 0.96 now).

**The transplant turns the null rows on.** Same synthesizer, same text, same four
preset labels: f0_sd goes from p = 0.96 to p = 4.6e-17, f0_range from p = 0.61 to
p = 4.7e-17. Its f0_range span (237 Hz) is in fact *wider* than `say`'s (136 Hz),
because `range_scale` is an explicit knob there and an emergent property here.
Note the two rows that get *weaker*: rms_mean_db drops from p = 1.6e-08 to
p = 0.010 and speech_rate from 0.00038 to 0.023, because the transplant presets
lean on pitch where Piper's own presets had nothing but level and rate to lean
on. The engine did not gain expression across the board; it traded a small amount
of level separation for a very large amount of pitch separation.

### 3b. End to end — grouped by what the classifier actually detected

The reply wavs from the 24-turn real-actor runs, grouped by the emotion detected
on that turn's *user* audio. This is the real claim, and it inherits every
classifier error.

| feature | Piper: H, p | **+ transplant: H, p** | `say`: H, p |
|---|---|---|---|
| f0_mean | 7.7, **0.17** | 21.0, **0.00082** | 21.8, **0.00058** |
| f0_range | 7.7, **0.17** | 21.4, **0.00069** | 21.7, **0.00061** |
| rms_mean_db | 9.1, 0.10 | 5.0, 0.41 | 15.3, 0.0093 |
| speech_rate | 3.0, 0.71 | 3.6, 0.61 | 1.9, 0.86 |
| duration | 14.0, 0.015 | 10.8, 0.055 | 13.8, 0.017 |

On `say`, F0 tracks the detected emotion cleanly: median f0_mean 92.6 Hz when
fear was detected, 93.6 Hz for sad, 216 Hz for disgust and neutral, 269 Hz for
anger, 336 Hz for happy. **On Piper it does not separate at all.**

**The transplant reaches `say`'s end-to-end separation at Piper's latency.**
Same 24 real-actor turns, run through the transplant engine on the fast path
(`runs/tc-crema-transplant.json`, n = 24, one per detected emotion group of
2–5): median f0_mean 139.7 Hz when fear was detected, 139.8 Hz for sad, 197.6 Hz
for neutral, 201.0 Hz for disgust, 245.8 Hz for anger, 315.7 Hz for happy. That
is p = 0.00082 against `say`'s p = 0.00058 — the same claim, from an engine that
cost 103 ms of synthesis instead of 606 ms. The `sad`/`fear` floor sits at 140 Hz
rather than `say`'s 93 Hz, which is the clipped 0.72× preset from §2 showing up
end to end exactly where §2 said it would.

Its rms and duration rows go the other way (p = 0.41, 0.055 against `say`'s
0.0093, 0.017), the same trade §3a describes: this engine separates by pitch and
by little else.

Two honest caveats. The reply *text* differs between turns here, unlike §3a, so
the duration and rate rows are partly the sentence and not the prosody — which
is why they are the two rows that behave the same on both engines. And n is 3–5
per group; read §3a for the clean version and this table for the end-to-end one.

**The answer to "does the expressive output measurably differ by detected
emotion" is: yes on `say`, yes on the transplant, no on plain Piper.** It used
to be "yes on `say`, no on Piper — the system works, one of its two available
voices cannot express what it is asked to." The third engine is the fix, and it
is the fix at 103 ms rather than 606 ms.

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
- **Three more 20-turn runs on the same prompts, this session, say the same
  thing less extremely:** 16/20 `anger` + 4 `neutral` (control F0, max
  confidence 0.90), 12/20 + 8 (control F0b, 0.91), 12/20 + 8 (fast path F2,
  0.88). So the *exact* 20/20 does not reproduce — it is 12 to 20 out of 20
  across four runs of identical audio — and the finding it was reporting does:
  on five flat synthetic sentences with no emotion in them the classifier
  commits to `anger` most of the time and to `neutral` the rest, never to
  anything in between, and never with low confidence.
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
5. **Prosody control is crude.** Four presets. Plain Piper has no pitch knob at
   all, `pmod` is inert on `say`, the transplant has a pitch knob but its
   `range_scale` is a single scalar over the whole utterance rather than
   anything phrase-aware, and pause structure is untested — every utterance here has zero detected pauses ≥ 100 ms,
   so the three pause features carry no information.
6. **The emotion→preset mapping is a judgement call, not a result.** Six classes
   onto four presets; `disgust` is mapped to neutral because none of the four is
   a good-faith rendering of it, and inventing one would be worse than
   abstaining.
7. **n is small.** 20–24 turns per arm, 3–5 per emotion group in §3b. Every
   figure is reported with n and spread; none of them should be read as a point
   estimate.
8. **ASR is chunked, not streaming, and the final decode throws away the
   partial's work.** Inherited from aliveness-threshold, unfixed. The fast path
   hides that waste inside the hangover rather than removing it — the 246 ms
   decode still happens, it just no longer happens where the listener waits for
   it. On a turn that falls back to serial, the listener pays it in full.
9. **The microphone path is only partly verified.** `--mic` opens a real input
   stream, endpoints and transcribes; no reported number comes from it. The full
   acoustic loop — speakers into the room into the mic — was not tested.
10. **The machine itself is not a stable measurement instrument.** The loop this
    one is built on measured 1452 ms and then 807 ms from identical code twenty
    minutes apart, with no recorded variable explaining it. Every number here was
    taken in one session with its own internal control, which is the mitigation,
    not a fix. Numbers from different sessions in this repo should not be
    compared to each other either.
11. **The F0 transplant is measured acoustically and by one classifier, not by
    a listener.** It landed (§2, §3), it does what it claims to F0, and nobody
    has said whether the resynthesised voice sounds *good*. WORLD round trips
    are known to cost some naturalness even when they hit the target contour,
    and this repo has not measured that cost at all. Its `sad` preset is
    clipped at 0.72x for a measured reason and is therefore not as low as the
    emotion it is named for.
12. **`--arm 80` is validated on a TTS talker, not on people.** §1.3 is the
    honest version: 3/24 false endpoints on CREMA-D actors, identical with the
    fast path on and off, so the speculation is not the cause — but CREMA-D
    actors read a card and stop cleanly, and nothing here has been tested on
    spontaneous speech with real disfluency. On a talker who pauses more, the
    fast path degrades gracefully (the snapshot goes stale and the turn falls
    back) and the *endpointer* is the thing that will cut them off.
13. **The fast path is not bit-identical to the serial path.** The speculative
    decode sees the same speech with ~270 ms less trailing silence, and whisper
    can answer differently to that. What is measured is that it does not answer
    worse: WER 0.000 on both fast arms against 0.067 on the serial control,
    n = 20 each. That is a "not worse", not a "same".

---

## Run it

```bash
uv venv --python 3.12 && uv pip install -e .

# the piper voice (gitignored, ~63 MB)
python -m piper.download_voices --download-dir models/piper en_US-lessac-medium

# the emotion checkpoint (gitignored, 378 MB) -- from emotion-label-ceiling
export EXPRESSIVE_S2S_EMOTION_CKPT=/path/to/wav2vec2-base-intended_emotion-actor-s0.pt

./run_all.sh          # sections 1.2 - 5
./run_fast.sh         # sections 1.1, 1.3 and the third engine in 2 / 3
```

Individual pieces:

```bash
.venv/bin/python -m s2s.loop selfcheck --n 3            # real turns + all assertions
.venv/bin/python -m s2s.loop selfcheck --n 3 --fast --arm 80
.venv/bin/python -m s2s.loop batch --n 20 --out runs/x.json

# the fast path: decode, classifier, LM and TTS inside the endpointer's hangover
.venv/bin/python -m s2s.loop batch --n 20 --fast --arm 80 --emotion-parallel
.venv/bin/python -m s2s.loop batch --n 20 --fast --arm 80 --final-model tiny.en

# the third engine: piper with an F0 contour transplanted onto it
.venv/bin/python -m s2s.loop batch --n 20 --tts transplant --fast --arm 80

.venv/bin/python -m s2s.loop batch --mic --n 3          # talk to it
.venv/bin/python -m s2s.expression selfcheck
.venv/bin/python -m s2s.expression sweep --backends piper,transplant,say
.venv/bin/python summarize.py                           # sections 1.2 - 5
.venv/bin/python fastsummary.py                         # every fast-path number
```

Results live in [`runs/`](runs) (one JSON per turn, per arm) and [`out/`](out)
(the analyses). `summarize.py` and `fastsummary.py` between them print every
figure in this README from those files; if a number is in the README and not in
one of those two outputs, it should not be in the README.

### The self-check

`s2s.loop selfcheck` runs real turns and asserts, for each one:

- every stage timer exists, is a number, is not negative, and **the stages sum
  to the gap** within one output block (5.8 ms + 1 ms) — which is the assertion
  that keeps the critical-path accounting honest now that stages overlap;
- **no stage is charged more than it actually cost.** `stage_ms` bills the
  listener, `work_ms` records the wall clock; if a stage is ever charged more
  than its own work, the critical-path walk is billing someone else's time;
- **a speculated turn paid no dispatch.** A turn served from the fast path
  started its decode before the endpointer fired, so `asr_final_dispatch_ms`
  must be exactly 0 — that is the assertion that catches a "fast" turn that
  quietly fell back to serial while still being reported as fast;
- the cue landed *before* the reply and was not cut off mid-word;
- the preset spoken matches the emotion detected;
- and **the classifier was fed this turn's audio.** The captured array is
  fingerprinted at capture time, the classifier returns the fingerprint of what
  it actually saw, and the two must match — and must differ across turns. A loop
  that conditions turn *N*'s reply on turn *N−1*'s audio looks perfectly healthy
  in the logs and is worthless; this is the assertion that catches it.

  **The fast path made that assertion harder, not weaker.** The classifier now
  sees the *speculative snapshot*, not the whole capture, so a naive
  `sha(classifier input) == sha(capture)` would simply fail. It is asserted in
  two halves instead, and together they say something stronger: the classifier
  saw exactly what the rest of the pipeline saw (`emotion_audio_sha ==
  pipeline_audio_sha`), and what the pipeline saw is a genuine **prefix** of
  this turn's capture (`pipeline_audio_is_prefix`, an element-wise comparison,
  not a length check). A stale buffer or a previous turn's audio fails the
  second half.

`s2s.expression selfcheck` asserts the acoustic feature path is actually
connected to the synthesizer: a rate change must move duration on Piper, a pitch
change must move F0 on `say`, and the sad→excited preset pair must move F0 by
1.5x on the transplant — the last one being what would fail if the WORLD round
trip were silently handing Piper's own audio back.

## Where the code came from

Nothing here re-implements what the sibling repos already measured. Files were
copied in with an attribution header naming the source, rather than imported
through a path hack, so this repo runs standalone and the measurement code
cannot drift underneath it.

| File | Source | Modified? |
|---|---|---|
| `vendor/audio.py` | aliveness-threshold `harness/audio.py` | no |
| `vendor/cues.py` | aliveness-threshold `harness/cues.py` | no |
| `vendor/tts.py` | aliveness-threshold `harness/tts.py` | yes — `synth()` takes a prosody `cfg` and passes it down; plus a third backend, `transplant`, which is the piper backend post-filtered through `vendor/transplant.py` |
| `vendor/features.py` | expressive-tts-audit `features.py` | yes — its self-check dropped (it imported that repo's `render.py`); `extract()` and `FEATURES` byte-identical |
| `s2s/emotion.py` | emotion-label-ceiling `modeling/finetune.py`, `modeling/common.py` | model class and label vocab verbatim; loader and per-turn API new |
| `s2s/prosody.py` | expressive-tts-audit `render.py` | preset values verbatim; emotion→preset mapping new |
| `vendor/transplant.py` | prosody-transplant `transplant.py` (`1f17b9a`) | `Target`, `PRESETS`, `target_contour`, `world_transplant` verbatim; its `piper_render`/`render` **not** copied — this repo already owns a Piper voice and loading a second would double the model in memory |
| `s2s/loop.py` | aliveness-threshold `live/loop.py` | adapted — emotion stage, conditioning, prosody, cue scheduling, named playback |
| `s2s/loop.py` `FastPath`, `_critical_path` | aliveness-threshold `live/loop.py` | adapted — the arming rule, the sequence-number staleness handshake and `claim`/`reset` are unchanged in behaviour; the pipeline inside them gained the emotion stage, because the emotion conditions the LM prompt and the TTS preset and the reply cannot be built early without it |

Model weights are not in git.

---

Every number above came from a run that happened on this machine — Apple M4 Pro,
24 GB, macOS, Python 3.12 — on 2026-08-26. Machine load is recorded in every
result JSON, because the ASR stage is CPU-bound and the load is part of the
result.
