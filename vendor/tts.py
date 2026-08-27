# VENDORED, MODIFIED, from ~/Desktop/Playground/aliveness-threshold/harness/tts.py
# (github.com/abhaymettu/aliveness-threshold).
#
# THE ONE MODIFICATION: `synth()` takes an optional `cfg` dict of prosody knobs
# and passes it down to whichever backend is active, so a reply can be spoken
# with prosody chosen per turn:
#
#   piper  ->  SynthesisConfig(length_scale, noise_scale, noise_w_scale, volume)
#   say    ->  the [[rate N]][[pbas N]][[pmod N]][[volm N]] embedded-command
#              prefix Apple documents in the Speech Synthesis Programming Guide
#
# THE SECOND MODIFICATION: a third backend, `transplant`. It is the piper
# backend with an F0 contour imposed on its output by WORLD analysis/resynthesis
# (vendor/transplant.py, from ~/Desktop/Playground/prosody-transplant). Piper
# has no pitch knob; this bolts one on from outside. See vendor/transplant.py
# for what that repo measured.
#
# Both knob sets and their neutral values come from
# ~/Desktop/Playground/expressive-tts-audit/render.py, which measured which of
# them actually move an acoustic feature. Upstream synthesizes every utterance
# with engine defaults and has no prosody surface at all.
# Nothing else is changed. Timing behaviour and silence trimming are untouched.
"""Text to speech, mono float32 at `audio.SR`.

Two backends, picked at import time by what is actually installed:

- ``piper``  -- preferred. Needs `piper-tts` plus a downloaded .onnx voice
  (set ``ALIVENESS_PIPER_VOICE`` to its path, or drop it in ``models/piper/``).
- ``say``    -- macOS built-in. The documented fallback, and what this machine
  is running as of the last measurement pass. Quality is fine for stimuli;
  it is not cross-platform and it is not streamable below the clause level.

Whichever is used is reported as ``tts_backend`` in every timings dict, so no
result is ever ambiguous about how its audio was made.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from . import audio

_MODELS = Path(__file__).resolve().parent.parent / "models" / "piper"


def _find_piper_voice() -> Path | None:
    env = os.environ.get("ALIVENESS_PIPER_VOICE")
    if env and Path(env).exists():
        return Path(env)
    if _MODELS.is_dir():
        found = sorted(_MODELS.glob("*.onnx"))
        if found:
            return found[0]
    return None


class Voice:
    """One TTS voice. `synth(text)` returns trimmed mono float32 at audio.SR."""

    def __init__(self, backend: str | None = None, name: str | None = None):
        self.backend = backend or self._autodetect()
        self.name = name
        self._piper = None
        if self.backend in ("piper", "transplant"):
            from piper import PiperVoice  # noqa: PLC0415

            voice_path = Path(name) if name else _find_piper_voice()
            if voice_path is None:
                raise RuntimeError(f"{self.backend} backend selected but no .onnx voice found")
            self._piper = PiperVoice.load(str(voice_path))
            self.name = str(voice_path)
        elif self.backend == "say":
            if not shutil.which("say"):
                raise RuntimeError("`say` backend selected but macOS `say` is not on PATH")
        else:
            raise ValueError(f"unknown tts backend: {self.backend}")

    @staticmethod
    def _autodetect() -> str:
        try:
            import piper  # noqa: F401,PLC0415

            if _find_piper_voice() is not None:
                return "piper"
        except ImportError:
            pass
        if shutil.which("say"):
            return "say"
        raise RuntimeError("no TTS backend available (install piper-tts, or run on macOS)")

    def synth(self, text: str, trim: bool = True, cfg: dict | None = None) -> np.ndarray:
        """Synthesize `text`. Trimmed of leading/trailing silence by default --
        an untrimmed clip silently inflates every gap we later claim to control.
        """
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        if self.backend == "transplant":
            x = self._transplant_synth(text, cfg)
        elif self.backend == "piper":
            x = self._piper_synth(text, cfg)
        else:
            x = self._say_synth(text, cfg)
        return audio.trim(x) if trim else x

    def synth_timed(self, text: str, trim: bool = True) -> tuple[np.ndarray, float]:
        """(audio, wall_ms) -- the wall clock cost of producing this clip."""
        t0 = time.perf_counter()
        x = self.synth(text, trim=trim)
        return x, (time.perf_counter() - t0) * 1000.0

    # -- backends ---------------------------------------------------------

    def _piper_synth(self, text: str, cfg: dict | None = None) -> np.ndarray:
        chunks = []
        sr = audio.SR
        syn = None
        if cfg:
            from piper import SynthesisConfig  # noqa: PLC0415

            syn = SynthesisConfig(normalize_audio=False, **cfg)
        for c in self._piper.synthesize(text, syn_config=syn):
            sr = c.sample_rate
            chunks.append(
                np.frombuffer(c.audio_int16_bytes, dtype="<i2").astype(np.float32) / 32768.0
            )
        x = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        if sr != audio.SR and len(x):
            n = int(round(len(x) * audio.SR / sr))
            x = np.interp(
                np.linspace(0, len(x) - 1, n), np.arange(len(x)), x.astype(np.float64)
            ).astype(np.float32)
        return x

    def _transplant_synth(self, text: str, cfg: dict | None = None) -> np.ndarray:
        """Piper, then the F0 contour imposed on it. `cfg` is a
        `vendor.transplant.Target` field dict -- duration and volume go to piper
        natively (it does those correctly and for free), pitch goes to WORLD."""
        from . import transplant  # noqa: PLC0415

        tgt = transplant.Target(**cfg) if cfg else transplant.PRESETS["neutral"]
        x = self._piper_synth(
            text, {"length_scale": tgt.length_scale, "volume": tgt.volume,
                   "noise_scale": 0.667, "noise_w_scale": 0.8})
        return transplant.world_transplant(x, tgt, sr=audio.SR)

    def _say_synth(self, text: str, cfg: dict | None = None) -> np.ndarray:
        if cfg:
            text = ("[[rate {rate}]][[pbas {pbas}]][[pmod {pmod}]][[volm {volm}]] "
                    .format(**cfg)) + text
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.wav"
            cmd = ["say", "-o", str(p), "--data-format=LEI16@%d" % audio.SR]
            if self.name:
                cmd += ["-v", self.name]
            cmd += ["--", text]
            subprocess.run(cmd, check=True, capture_output=True)
            return audio.read(p)


_default: Voice | None = None


def default_voice() -> Voice:
    global _default
    if _default is None:
        _default = Voice()
    return _default
