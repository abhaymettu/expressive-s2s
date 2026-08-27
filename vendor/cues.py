# VENDORED, UNMODIFIED, from ~/Desktop/Playground/aliveness-threshold/harness/cues.py
# (github.com/abhaymettu/aliveness-threshold). Copied rather than imported so this
# repo runs standalone and so the measurement code cannot drift under it.
# Any change to this file is a change to the measurement and must be flagged here.
"""The non-verbal cues whose exchange rate against latency this repo measures.

Five levels, matching the study design::

    none | filled_pause | breath | backchannel | verbal_stall

Every cue is *real audio* placed at a real sample offset. Nothing here is a
text token handed to the language model.

Provenance, stated plainly because it is a limitation of the stimuli:

- ``breath`` is synthesized (band-limited noise under a breath envelope),
  not a recorded human inhale. Deterministic given the seed.
- ``filled_pause`` / ``backchannel`` / ``verbal_stall`` are TTS of "uh",
  "mm hmm", "let me see". A TTS "uh" is not a natural disfluency: it is a
  read token with clean onset and no coarticulation with the surrounding
  speech. UNVERIFIED whether listeners hear these as genuine disfluency.
  Swap in recorded cues via ``register_cue_wav()`` when they exist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import audio

CUES = ("none", "filled_pause", "breath", "backchannel", "verbal_stall")

CUE_TEXT = {
    "filled_pause": "uh",
    "backchannel": "mm hmm",
    "verbal_stall": "let me see",
}

_overrides: dict[str, Path] = {}
_cache: dict[tuple, np.ndarray] = {}


def register_cue_wav(cue: str, path) -> None:
    """Point a cue at a recorded wav instead of the synthetic default."""
    if cue not in CUES:
        raise ValueError(f"unknown cue {cue!r}; expected one of {CUES}")
    _overrides[cue] = Path(path)
    _cache.clear()


def breath(duration_ms: float = 320.0, level: float = 0.045, seed: int = 0) -> np.ndarray:
    """Synthetic inhale: noise band-limited to 400-2200 Hz, breath envelope.

    Deterministic for a given (duration, level, seed) so stimuli are reproducible.
    """
    n = audio.samples(duration_ms)
    x = np.random.default_rng(seed).standard_normal(n)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / audio.SR)
    # raised-cosine band-pass; hard bin gating rings audibly
    band = np.clip((f - 250.0) / 300.0, 0, 1) * np.clip((2600.0 - f) / 600.0, 0, 1)
    x = np.fft.irfft(spec * band, n)
    t = np.linspace(0, 1, n, endpoint=False)
    # Asymmetric envelope: quick-but-not-clicky attack, long decay. The attack
    # has to be fast enough that VAD can resolve the onset -- with a symmetric
    # sin^1.6 envelope the measured onset lagged the true one by 25ms, which ate
    # the entire error budget. This is a calibration knob, not cosmetics.
    env = np.minimum(t / 0.06, 1.0) * (1.0 - t) ** 1.3
    x = x * env
    peak = np.abs(x).max()
    return ((x / peak) * level).astype(np.float32) if peak > 0 else x.astype(np.float32)


def cue_audio(cue: str, voice=None) -> np.ndarray:
    """Audio for one cue. Empty array for ``none``. Cached per (cue, backend)."""
    if cue not in CUES:
        raise ValueError(f"unknown cue {cue!r}; expected one of {CUES}")
    if cue == "none":
        return np.zeros(0, dtype=np.float32)
    if cue in _overrides:
        return audio.trim(audio.read(_overrides[cue]))
    if cue == "breath":
        return breath()

    from . import tts  # noqa: PLC0415

    voice = voice or tts.default_voice()
    key = (cue, voice.backend, voice.name)
    if key not in _cache:
        _cache[key] = voice.synth(CUE_TEXT[cue])
    return _cache[key]


def cue_source(cue: str) -> str:
    if cue == "none":
        return "none"
    if cue in _overrides:
        return "recorded"
    return "synth" if cue == "breath" else "tts"
