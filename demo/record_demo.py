"""Record one committed exchange so a hiring manager can listen before reading.

Takes a finished run's JSON and stitches the turns back into a single wav laid
out on the real clock: the user's prompt audio, then real silence for the
measured gap, then the cue (if one fired) at its measured offset, then the
reply. The gaps you hear are the gaps that were measured -- nothing is
re-timed, and nothing is compressed.

Small on purpose: 16 kHz mono PCM16, a handful of turns, so it lives in git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from vendor import audio, cues

DEMO_SR = 16000  # half the loop's 22050, for a committable file size


def pick(run, n: int, spread: bool):
    """Which turns go in the demo.

    `spread` takes one turn per distinct detected emotion, in turn order, so a
    listener hears the voice actually change instead of four turns of whatever
    the classifier happened to say first. It is a listening aid, not a result:
    the distribution over all turns is in the run JSON and in the README, and
    it is not this flattering.
    """
    # A turn the endpointer cut short is not the loop working, and playing one
    # to a listener as if it were would be the demo lying. They are counted in
    # README section 1.3, where they belong, and left out of here.
    ts = [t for t in run["turns"] if not t.get("truncated")] or run["turns"]
    if not spread:
        return ts[:n]
    seen, out = set(), []
    for t in ts:
        if t["detected_emotion"] not in seen:
            seen.add(t["detected_emotion"])
            out.append(t)
    return out[:n]


def build(run_json: str, out_wav: str, turns: int = 4, lead_ms: float = 400.0,
          spread: bool = True):
    run = json.loads(Path(run_json).read_text())
    from s2s.loop import PROMPTS, pick_voice  # noqa: PLC0415

    voice = pick_voice("piper")
    cue_a = cues.cue_audio("filled_pause", voice=voice)

    stim_dir = Path("runs/user-turns")
    out = [np.zeros(audio.samples(lead_ms), np.float32)]
    lines = []
    for t in pick(run, turns, spread):
        # prefer the exact wav the loop was fed. For a CREMA-D-driven run that is
        # a real person, which is the whole point of playing it to someone.
        stim = stim_dir / f"{t['label']}.wav"
        if stim.exists():
            prompt = audio.trim(audio.read(stim))
        else:
            prompt = voice.synth(t["label"] if t["label"] in PROMPTS else t["transcript"])
        reply = audio.read(t["reply_wav"])
        out.append(prompt)

        gap = t["gap_ms"]
        cue_at = t.get("cue_gap_ms")
        if cue_at is not None and len(cue_a):
            out.append(np.zeros(audio.samples(cue_at), np.float32))
            out.append(cue_a)
            spent = cue_at + audio.millis(len(cue_a))
            out.append(np.zeros(audio.samples(max(0.0, gap - spent)), np.float32))
        else:
            out.append(np.zeros(audio.samples(gap), np.float32))
        out.append(reply)
        out.append(np.zeros(audio.samples(700.0), np.float32))
        lines.append(
            f"  turn {t['turn']}: {t['transcript']!r}\n"
            f"    detected {t['detected_emotion']} ({t['emotion_confidence']:.2f}) "
            f"-> spoken {t['prosody_preset']}"
            + (f", cue at {cue_at:.0f}ms" if cue_at else "")
            + f", reply at {gap:.0f}ms\n"
            f"    {t['reply']!r}"
        )

    x = np.concatenate(out)
    n = int(round(len(x) * DEMO_SR / audio.SR))
    x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)),
                  x.astype(np.float64)).astype(np.float32)
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, np.clip(x, -1, 1), DEMO_SR, subtype="PCM_16")
    kb = Path(out_wav).stat().st_size / 1024
    print(f"wrote {out_wav}  ({len(x)/DEMO_SR:.1f}s, {kb:.0f} KB)")
    print("\n".join(lines))
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_json")
    ap.add_argument("out_wav")
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--first", action="store_true",
                    help="first N turns instead of one per distinct detected emotion")
    a = ap.parse_args()
    build(a.run_json, a.out_wav, a.turns, spread=not a.first)
