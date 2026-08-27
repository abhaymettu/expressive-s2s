"""The emotion head, lifted out of ~/Desktop/Playground/emotion-label-ceiling.

The architecture (`Model`) and the label vocabulary are copied verbatim from
`emotion-label-ceiling/modeling/finetune.py` and `modeling/common.py`
(github.com/abhaymettu/emotion-label-ceiling). They have to be, because the
checkpoint's state_dict keys are `enc.*` and `head.*` and nothing else will
load it. The checkpoint itself is the run reported there as
`wav2vec2-base-intended_emotion-actor-s0`:

    test accuracy vs actor intent      74.9%   (n = 1640 clips, 20 held-out actors)
    test accuracy vs crowd consensus   52.2%

Both numbers are from CREMA-D, which is **acted** speech: 91 actors reading 12
fixed sentences with an instructed emotion. Nothing about that guarantees the
model works on a person talking to a laptop. See README.

Weights are NOT in this repo. Point ``EXPRESSIVE_S2S_EMOTION_CKPT`` at the .pt,
or leave it and the loader will look in ``models/`` and then in the
emotion-label-ceiling checkout next door.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent

# --- copied verbatim from emotion-label-ceiling/modeling/common.py -----------
EMOTIONS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
E2I = {e: i for i, e in enumerate(EMOTIONS)}
SR = 16000
MAX_SAMPLES = 4 * SR  # p95 CREMA-D clip is 3.64s
BASE = "facebook/wav2vec2-base"

CKPT_CANDIDATES = [
    ROOT / "models" / "wav2vec2-base-intended_emotion-actor-s0.pt",
    Path.home() / "Desktop/Playground/emotion-label-ceiling/models"
    / "wav2vec2-base-intended_emotion-actor-s0.pt",
]


# --- copied verbatim from emotion-label-ceiling/modeling/finetune.py ---------
class Model(nn.Module):
    def __init__(self, name, n_cls=6, freeze_cnn=True):
        super().__init__()
        from transformers import AutoModel  # noqa: PLC0415

        self.enc = AutoModel.from_pretrained(name)
        if freeze_cnn and hasattr(self.enc, "feature_extractor"):
            self.enc.feature_extractor._freeze_parameters()
        self.head = nn.Linear(self.enc.config.hidden_size, n_cls)

    def pooled(self, x, m):
        h = self.enc(x).last_hidden_state
        f = torch.nn.functional.interpolate(m[:, None], size=h.shape[1], mode="nearest")[:, 0]
        return (h * f[..., None]).sum(1) / f.sum(1, keepdim=True).clamp(min=1)

    def forward(self, x, m):
        return self.head(self.pooled(x, m))


# --- new here ---------------------------------------------------------------

def sha(x: np.ndarray) -> str:
    """Fingerprint of one turn's captured audio.

    Load-bearing, not decoration: the selfcheck asserts the array the classifier
    actually saw hashes to the same thing as the array the turn captured. Off-by-
    one-turn conditioning is the failure mode that would make every emotion
    number here meaningless while looking completely fine in the logs.
    """
    return hashlib.sha1(np.ascontiguousarray(x, dtype=np.float32).tobytes()).hexdigest()[:16]


def find_ckpt() -> Path:
    env = os.environ.get("EXPRESSIVE_S2S_EMOTION_CKPT")
    if env:
        return Path(env)
    for p in CKPT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "no emotion checkpoint. Set EXPRESSIVE_S2S_EMOTION_CKPT, or copy "
        "wav2vec2-base-intended_emotion-actor-s0.pt into models/. Tried: "
        + ", ".join(str(p) for p in CKPT_CANDIDATES)
    )


def to_16k(x: np.ndarray, sr_in: int) -> np.ndarray:
    """Resample to the 16 kHz the encoder was trained at.

    ponytail: linear interpolation, the same safety net vendor/audio.py uses.
    wav2vec2's front end is a strided CNN over the waveform; a sharper
    anti-alias filter is not worth a dependency at this SNR. If the classifier
    ever has to be defended on resampling, swap in soxr and re-measure.
    """
    if sr_in == SR:
        return x.astype(np.float32)
    n = int(round(len(x) * SR / sr_in))
    return np.interp(
        np.linspace(0, len(x) - 1, n), np.arange(len(x)), x.astype(np.float64)
    ).astype(np.float32)


class Classifier:
    """Per-turn emotion from raw audio. One forward pass, no batching."""

    def __init__(self, ckpt: Path | None = None, device: str | None = None):
        self.ckpt = Path(ckpt) if ckpt else find_ckpt()
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = Model(BASE)
        sd = torch.load(self.ckpt, map_location="cpu")
        self.model.load_state_dict(sd)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def predict(self, x: np.ndarray, sr_in: int) -> dict:
        """Emotion for one utterance.

        Returns the label, the softmax over all six classes, the wall-clock cost,
        and the fingerprint of the audio that produced it. That last field is
        what lets the selfcheck prove the classifier saw *this* turn's audio.
        """
        t0 = time.perf_counter()
        w = to_16k(x, sr_in)
        fed = w[:MAX_SAMPLES]  # same 4s cap the model was trained under
        t = torch.from_numpy(fed)[None]
        m = torch.ones_like(t)
        logits = self.model(t.to(self.device), m.to(self.device))[0].float().cpu()
        p = torch.softmax(logits, -1).numpy()
        i = int(p.argmax())
        return {
            "emotion": EMOTIONS[i],
            "confidence": round(float(p[i]), 4),
            "probs": {e: round(float(v), 4) for e, v in zip(EMOTIONS, p)},
            "ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "audio_sha": sha(x),
            "fed_samples": int(len(fed)),
            "truncated": bool(len(w) > MAX_SAMPLES),
        }
