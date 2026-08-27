# VENDORED, UNMODIFIED LOGIC, from ~/Desktop/Playground/prosody-transplant/transplant.py
# (github.com/abhaymettu/prosody-transplant), commit 1f17b9a.
#
# `Target`, `PRESETS`, `target_contour` and `world_transplant` are copied
# verbatim. What is NOT copied is that repo's `piper_render` / `render`: this
# repo already owns a piper voice (vendor/tts.py) and re-loading a second
# PiperVoice would double the model in memory and desynchronise the two. So the
# transplant here is a *post-filter* applied to vendor/tts.py's piper output,
# which is byte-identical audio to what that repo renders.
#
# What that repo measured, on this machine, that makes this worth wiring in:
#
#   piper's own four presets, recovered from acoustics   52.8%  [44.7, 60.8]
#   the same four after the F0 transplant                100.0% [97.4, 100.0]
#   (n = 144 each, 4-way, chance 25%)
#
#   total render, piper + WORLD round trip   median 122 ms  (out/latency.csv, n=24)
#   piper alone, in that same harness        median  46 ms
#
# Which is the whole point: expression at roughly piper latency instead of the
# 678 ms macOS `say` costs to do the same job.
"""Impose an F0 contour on piper's output by WORLD analysis/resynthesis.

Piper exposes no pitch knob. This is the knob. Decompose the rendered waveform
into F0 / spectral envelope / band aperiodicity, rewrite the F0 track, put it
back together. Rewriting F0 while holding the spectral envelope fixed moves
pitch without moving formants -- a naive resample would chipmunk the voice.

Timing is deliberately NOT done here. Piper's own `length_scale` changes
duration natively, for free, with no vocoder artifact, so it is applied at
synthesis time and the transplant only touches pitch.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

SR = 22050
FRAME_PERIOD = 5.0  # ms, WORLD analysis hop


@dataclass(frozen=True)
class Target:
    """What you want the voice to do.

    f0_mean      target mean F0 in Hz (arithmetic, over voiced frames)
    range_scale  multiplies deviation from the median in the LOG domain.
                 1.0 keeps piper's own excursions, 1.6 exaggerates, 0.4 flattens
    tilt_st      semitones of declination added across the utterance,
                 negative = falls (statement), positive = rises (question)
    length_scale handed straight to piper; >1 is slower
    volume       handed straight to piper
    """

    f0_mean: float = 191.6      # piper's own measured neutral
    range_scale: float = 1.0
    tilt_st: float = 0.0
    length_scale: float = 1.0
    volume: float = 1.0


# Targets take macOS `say`'s measured F0 behaviour -- the engine expressive-tts-
# audit showed IS discriminable, at 95.8% -- and re-express it relative to
# piper's measured neutral of 191.6 Hz. `sad` is CLIPPED to 0.72x rather than
# say's 0.433x: prosody-transplant/artifacts.py measured the operating range as
# 0.60x-1.80x, and below it the F0 tracker starts losing voiced frames. That is
# a real limit of transplanting a male voice's pitch floor onto this source and
# it is reported as one, not hidden.
PRESETS = {
    "neutral":  Target(f0_mean=191.6, range_scale=1.00, tilt_st=0.0,  length_scale=1.00),
    "excited":  Target(f0_mean=301.2, range_scale=1.35, tilt_st=-2.0, length_scale=0.81),
    "emphatic": Target(f0_mean=238.3, range_scale=1.55, tilt_st=-3.0, length_scale=1.00),
    "sad":      Target(f0_mean=138.0, range_scale=0.55, tilt_st=-1.0, length_scale=1.10),
}


def target_contour(f0: np.ndarray, tgt: Target) -> np.ndarray:
    """Rewrite a measured F0 track to hit `tgt`. Unvoiced frames stay 0.

    All of it happens in log-F0, because pitch is perceived on a log scale and
    because a multiplicative shift is the transform that leaves the excursion
    *shape* intact while moving the register.

        log f0' = log(target_mean) + range_scale * (log f0 - median log f0)
                  + tilt(t)

    The median (not the mean) is the anchor so that one stray octave-error frame
    from the tracker cannot drag the whole utterance.
    """
    out = np.zeros_like(f0)
    v = f0 > 0
    n = int(v.sum())
    if n < 2:
        return out
    lf = np.log(f0[v])
    med = np.median(lf)
    pos = np.linspace(0.0, 1.0, len(f0))[v]
    tilt = (tgt.tilt_st / 12.0) * np.log(2.0) * (pos - pos.mean())
    lf_new = np.log(tgt.f0_mean) + tgt.range_scale * (lf - med) + tilt
    new = np.exp(lf_new)
    # The log-domain anchor sets the GEOMETRIC mean; f0_mean is specified (and
    # measured) as the arithmetic mean, and the two differ by Jensen's gap.
    # One multiplicative correction makes the delivered arithmetic mean exact.
    new *= tgt.f0_mean / new.mean()
    # Keep the tracker's own working band; WORLD's synthesiser degrades badly if
    # it is handed frames it could never have analysed.
    out[v] = np.clip(new, 55.0, 700.0)
    return out


def world_transplant(x: np.ndarray, tgt: Target, sr: int = SR,
                     harvest: bool = False) -> np.ndarray:
    """WORLD decompose -> rewrite F0 -> resynthesise."""
    import pyworld as pw  # noqa: PLC0415

    x = np.ascontiguousarray(x, dtype=np.float64)
    if harvest:
        f0, t = pw.harvest(x, sr, frame_period=FRAME_PERIOD)
    else:
        f0, t = pw.dio(x, sr, frame_period=FRAME_PERIOD)
        f0 = pw.stonemask(x, f0, t, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    y = pw.synthesize(target_contour(f0, tgt), sp, ap, sr, frame_period=FRAME_PERIOD)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0.99:  # WORLD can overshoot; never hand a clipped buffer downstream
        y = y * (0.99 / peak)
    return y.astype(np.float32)


def target_for(name: str, **overrides) -> Target:
    t = PRESETS[name]
    return replace(t, **overrides) if overrides else t
