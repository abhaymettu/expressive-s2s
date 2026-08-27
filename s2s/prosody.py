"""Detected emotion -> the prosody the reply is spoken with, on two engines.

The knobs, their neutral values and the four preset labels are taken from
~/Desktop/Playground/expressive-tts-audit/render.py
(github.com/abhaymettu/expressive-tts-audit), which rendered 708 utterances and
swept every knob one at a time to find which ones actually move an acoustic
measurement. Its findings bound everything this repo can claim about
"expression", so they are stated here rather than discovered again:

    piper en_US-lessac-medium
        Exposes NO pitch control. Mean F0 across all four presets spans 2 Hz
        (190.0 - 191.9). Zero of 15 preset x F0-feature comparisons reach
        significance, max |g| = 0.24. Its four presets are recovered from
        acoustics at 49.3% [41.3, 57.4] against 25% chance, and its `neutral`
        preset has recall 0.306 with a CI that CONTAINS chance -- it is not
        reliably distinguishable from its own expressive presets.
        `noise_scale` changes the waveform but moves no acoustic feature.
        `length_scale` and `noise_w_scale` move duration, and that is the
        whole surface: rate, variability, level. Not intonation.

    macOS say, voice Alex
        Mean F0 spans 246 Hz (93.7 - 340.0). Presets recovered at 95.8%
        [91.2, 98.1]. But `pmod` (pitch modulation) is INERT -- levels 0
        through 130 produce byte-identical WAV files -- and `rate` is coarsely
        quantised, with 160/165/170/180 wpm all rendering to exactly 2.485 s.
        So the expression on this engine is `pbas` and `volm` doing the work.
        And it costs ~2.6 s per utterance, because each call is a cold
        subprocess (aliveness-threshold live/STATUS.md, n = 20).

There is no third option on this laptop. That is the tradeoff this repo
measures: an engine that is fast and acoustically flat, against an engine that
is expressive and two and a half seconds slow. `pmod` is kept in the say preset
because these are the audit's exact preset values and changing them would break
the link to its numbers -- but it is inert and contributes nothing.

The mapping from six CREMA-D classes onto four presets is a judgement call, not
a result. anger/happy take the two high-arousal presets, sad/fear the low one,
disgust and neutral stay neutral -- disgust because none of these four presets
is a good-faith rendering of it, and inventing one would be worse than
abstaining.
"""

from __future__ import annotations

import dataclasses

from vendor import transplant

# --- copied from expressive-tts-audit/render.py -----------------------------
PIPER_PRESETS = {
    "neutral":  dict(length_scale=1.00, noise_scale=0.667, noise_w_scale=0.80, volume=1.0),
    "excited":  dict(length_scale=0.80, noise_scale=0.900, noise_w_scale=1.10, volume=1.0),
    "sad":      dict(length_scale=1.30, noise_scale=0.450, noise_w_scale=0.55, volume=0.8),
    "emphatic": dict(length_scale=1.05, noise_scale=0.750, noise_w_scale=1.30, volume=1.0),
}

SAY_PRESETS = {
    "neutral":  dict(rate=180, pbas=50, pmod=50, volm=1.0),
    "excited":  dict(rate=225, pbas=68, pmod=85, volm=1.0),
    "sad":      dict(rate=135, pbas=36, pmod=20, volm=0.8),
    "emphatic": dict(rate=165, pbas=55, pmod=95, volm=1.0),
}

# --- the third engine, from prosody-transplant -------------------------------
# Piper with an F0 contour imposed on its output by WORLD resynthesis. These are
# the four `Target`s that repo shipped, verbatim; see vendor/transplant.py for
# how they were derived and why `sad` is clipped. The point of this backend is
# that it breaks the tradeoff the two above define: piper's four presets are
# recovered from acoustics at 52.8% [44.7, 60.8], these at 100% [97.4, 100.0]
# (n=144 each, chance 25%), for ~75 ms of extra synthesis instead of ~634 ms.
TRANSPLANT_PRESETS = {n: dataclasses.asdict(t) for n, t in transplant.PRESETS.items()}

PRESETS = {"piper": PIPER_PRESETS, "say": SAY_PRESETS,
           "transplant": TRANSPLANT_PRESETS}

EMOTION_TO_PRESET = {
    "anger":   "emphatic",
    "happy":   "excited",
    "sad":     "sad",
    "fear":    "sad",
    "neutral": "neutral",
    "disgust": "neutral",
}

# What the LM is told about how the user sounded. One clause, because the reply
# is one sentence and a longer instruction just costs prefill.
EMOTION_TO_HINT = {
    "anger":   "The user sounds angry. Be direct, acknowledge the frustration, do not apologise twice.",
    "happy":   "The user sounds cheerful. Match their energy.",
    "sad":     "The user sounds down. Be warm and brief.",
    "fear":    "The user sounds anxious. Be reassuring and concrete.",
    "disgust": "The user sounds put off. Be plain and practical.",
    "neutral": "The user sounds neutral. Be plain and helpful.",
}


def preset_for(emotion: str, backend: str = "piper") -> tuple[str, dict]:
    """(preset name, knob dict for `backend`) for a detected emotion."""
    name = EMOTION_TO_PRESET.get(emotion, "neutral")
    return name, dict(PRESETS[backend][name])


def hint_for(emotion: str) -> str:
    return EMOTION_TO_HINT.get(emotion, EMOTION_TO_HINT["neutral"])
