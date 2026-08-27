# VENDORED, UNMODIFIED, from ~/Desktop/Playground/aliveness-threshold/harness/audio.py
# (github.com/abhaymettu/aliveness-threshold). Copied rather than imported so this
# repo runs standalone and so the measurement code cannot drift under it.
# Any change to this file is a change to the measurement and must be flagged here.
"""Mono float32 audio at a single sample rate, plus the voice-activity
segmentation that every latency measurement in this repo is built on.

Everything downstream (gap measurement, cue onset measurement) reduces to
`segments()`. If that function is wrong, the study is wrong, so it is kept
dumb and inspectable: frame RMS, one threshold, one merge rule.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

SR = 22050  # piper voices and macOS `say` both land here natively

FRAME_MS = 5.0  # segmentation resolution; sets the floor on measurement error


def samples(ms: float, sr: int = SR) -> int:
    return int(round(ms * sr / 1000.0))


def millis(n: int, sr: int = SR) -> float:
    return 1000.0 * n / sr


def read(path, sr: int = SR) -> np.ndarray:
    """Read any soundfile-supported file as mono float32 at `sr`."""
    x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if file_sr != sr:
        # ponytail: linear resample. Sources here are all already at SR, so this
        # is a safety net, not a signal path. Swap for soxr if it ever runs hot.
        n_out = int(round(len(x) * sr / file_sr))
        x = np.interp(
            np.linspace(0, len(x) - 1, n_out, dtype=np.float64),
            np.arange(len(x), dtype=np.float64),
            x.astype(np.float64),
        ).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def write(path, x: np.ndarray, sr: int = SR) -> None:
    sf.write(str(path), np.clip(x, -1.0, 1.0).astype(np.float32), sr, subtype="PCM_16")


def frame_rms(x: np.ndarray, sr: int = SR, frame_ms: float = FRAME_MS) -> np.ndarray:
    hop = samples(frame_ms, sr)
    n = len(x) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    f = x[: n * hop].reshape(n, hop)
    return np.sqrt((f.astype(np.float64) ** 2).mean(axis=1)).astype(np.float32)


def speech_mask(
    x: np.ndarray,
    sr: int = SR,
    rel_db: float = -35.0,
    abs_db: float = -55.0,
    frame_ms: float = FRAME_MS,
) -> np.ndarray:
    """Per-frame bool. A frame is speech if it is within `rel_db` of the loudest
    frame AND above the absolute floor `abs_db` (dBFS)."""
    r = frame_rms(x, sr, frame_ms)
    if r.size == 0 or r.max() <= 0:
        return np.zeros(r.size, dtype=bool)
    return (r >= r.max() * 10 ** (rel_db / 20.0)) & (r >= 10 ** (abs_db / 20.0))


def segments(
    x: np.ndarray,
    sr: int = SR,
    merge_gap_ms: float = 60.0,
    min_len_ms: float = 30.0,
    frame_ms: float = FRAME_MS,
    **kw,
) -> list[tuple[float, float]]:
    """Speech segments as (start_ms, end_ms).

    Runs of speech frames separated by less than `merge_gap_ms` are merged, so
    inter-word pauses inside one utterance do not split it. `merge_gap_ms` must
    therefore be set below the shortest gap you intend to measure.
    """
    m = speech_mask(x, sr, frame_ms=frame_ms, **kw)
    if not m.any():
        return []
    idx = np.flatnonzero(m)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))

    out: list[list[float]] = []
    for s, e in zip(starts, ends):
        s_ms, e_ms = s * frame_ms, (e + 1) * frame_ms
        if out and s_ms - out[-1][1] < merge_gap_ms:
            out[-1][1] = e_ms
        else:
            out.append([s_ms, e_ms])
    return [(a, b) for a, b in out if b - a >= min_len_ms]


def trim(x: np.ndarray, sr: int = SR, pad_ms: float = 0.0, **kw) -> np.ndarray:
    """Strip leading/trailing silence, keeping `pad_ms` of headroom on each side.

    This is load-bearing: TTS engines emit a few hundred ms of silence around an
    utterance, and an untrimmed clip inflates every gap we claim to control.
    """
    segs = segments(x, sr, **kw)
    if not segs:
        return x
    a = max(0, samples(segs[0][0] - pad_ms, sr))
    b = min(len(x), samples(segs[-1][1] + pad_ms, sr))
    return np.ascontiguousarray(x[a:b])
