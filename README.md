# expressive-s2s

**An agent that hears how you sound and answers in a voice that matches.**

Speech in → emotion estimated from the audio → reply generated conditioned on
that emotion → reply spoken with matching prosody, with a non-verbal cue placed
in the gap while the rest of the pipeline is still working.

It runs end to end on one laptop, and it is measured end to end. **Listen
first: [`demo/exchange.wav`](demo/exchange.wav)** — four real turns, laid out on
the real clock. The silences you hear are the silences that were measured.

> **PLACEHOLDER_HEADLINE**

---

## The system, and the evidence behind each part

This is one project. The repo you are reading is the system; four sibling repos
are the studies that decided how it is built. Every design choice below points
at a measurement, and every measurement is one I ran.

```
  user audio ──▶ ASR (faster-whisper) ──┬──▶ EMOTION (wav2vec2) ──┐
   22050 Hz      tiny.en partials       │     6-class, per turn   │
   streamed      base.en final          │                        ▼
                                        │              LM (Llama-3.2-1B-4bit)
       ┌────────────────────────────────┘              conditioned on emotion
       │                                                        │
       ▼                                                        ▼
  CUE at 400 ms ────────────────────────────────▶  TTS with prosody preset
  ("uh", real audio, fills the gap)                 chosen by emotion
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

**Perception is presented with a stated confidence limit, not as a solved
component, and that is a finding not a hedge.** On audio-only CREMA-D, listener
agreement is Krippendorff α = 0.265 [0.259, 0.272], and the reliability ceiling
that implies is 0.727 [0.723, 0.732] — no model scored against what listeners
actually heard can exceed it. Across three seeds there is a mean 25.6-point gap
between predicting the actor's *intent* and predicting what listeners *heard*.
The checkpoint in this loop is the s0 run: **74.9% against actor intent, 52.2%
against crowd consensus.** So "the agent detected sadness" is a claim about a
label with known, measured softness underneath it.

**Timing is measured against a real prior number.** The gap definition here is
byte-for-byte the one in aliveness-threshold: *onset of agent speech minus
offset of user speech, both silence-trimmed*, with the user's offset
re-measured by `vendor.audio.segments(merge_gap_ms=30, min_len_ms=20)` — the
end of the last speech segment, **not** the moment the endpointer noticed. That
repo measured a median gap of **1452 ms over n = 40** on this same laptop.

**Generation is bounded by the engine, not by the mapping.** Piper exposes no
pitch control whatsoever: its four expressive presets span 2 Hz of mean F0, and
a classifier recovers them from acoustics at 49.3% [41.3, 57.4] against 25%
chance — its own `neutral` preset has recall 0.306 with a CI containing chance.
macOS `say` spans 246 Hz and is recovered at 95.8% [91.2, 98.1], but it costs
seconds per utterance. That is a real tradeoff and this repo measures both ends
of it rather than picking one and staying quiet.

---

## PLACEHOLDER_LATENCY

## PLACEHOLDER_TRADEOFF

## PLACEHOLDER_EXPRESSION

## PLACEHOLDER_PERCEPTION

## PLACEHOLDER_LIMITATIONS

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
.venv/bin/python -m s2s.loop selfcheck --n 3      # real turns + all assertions
.venv/bin/python -m s2s.loop batch --n 20 --out runs/x.json
.venv/bin/python -m s2s.loop batch --mic --n 3    # talk to it
.venv/bin/python -m s2s.expression selfcheck
.venv/bin/python summarize.py                    # every headline number
```

### The self-check

`s2s.loop selfcheck` runs real turns and asserts, for each one:

- every stage timer exists, is a number, is not negative, and **the stages sum
  to the gap** within one output block (5.8 ms + 1 ms);
- the cue landed *before* the reply and was not cut off mid-word;
- the preset spoken matches the emotion detected;
- and **the classifier was fed this turn's audio.** The captured array is
  fingerprinted at capture time, the classifier returns the fingerprint of what
  it actually saw, and the two must match — and must differ across turns. A
  loop that conditions turn *N*'s reply on turn *N−1*'s audio looks perfectly
  healthy in the logs and is worthless; this is the assertion that catches it.

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
