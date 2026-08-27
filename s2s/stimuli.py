"""Real human user turns for the loop, with a ground-truth emotion attached.

The obvious way to drive the loop is to render its own prompts with piper. That
is what aliveness-threshold did, and it is what makes the latency reproducible.
It is useless for testing an emotion classifier, because one flat synthetic
voice has no emotion to detect: whatever the classifier says, there is nothing
to be right or wrong about.

So the loop can also be driven by clips from the **held-out test actors** of
the CREMA-D split this classifier was trained on
(~/Desktop/Playground/emotion-label-ceiling). Those are real people, and each
one carries the emotion its actor was instructed to perform. That gives the
only ground truth available on this machine.

It is still acted speech read from a card, which is exactly the generalization
gap this repo has to report rather than paper over. A CREMA-D actor is closer
to a person than a TTS voice is, and further from a person than a person is.

Files are named ``{true_emotion}__{clip_id}.wav`` so the loop's existing
``--prompt-wav`` path carries the label through to the results JSON as
``label`` -- no loop change needed to score it.

    .venv/bin/python -m s2s.stimuli --n 24 --out runs/user-turns
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from s2s import emotion as emo
from vendor import audio

ELC = Path.home() / "Desktop/Playground/emotion-label-ceiling"
RUN = ELC / "modeling/runs/wav2vec2-base-intended_emotion-actor-s0"


def build(n: int = 24, out_dir="runs/user-turns", lead_ms=300.0, tail_ms=900.0, seed=0):
    """n clips balanced over the six emotions, from test actors only."""
    import pandas as pd  # noqa: PLC0415

    cfg = json.loads((RUN / "config.json").read_text())
    test_actors = set(cfg["split"]["test"])
    clips = pd.read_parquet(ELC / "data/clips.parquet")
    adir = next(p for p in [ELC / "data/audio/repo/AudioWAV", ELC / "data/raw/AudioWAV",
                            ELC / "data/raw/repo/AudioWAV"] if p.is_dir())
    te = clips[clips.actor_id.isin(test_actors)]
    assert len(te), "no test-actor clips found"

    per = max(1, n // len(emo.EMOTIONS))
    rng = np.random.default_rng(seed)
    picked = []
    for e in emo.EMOTIONS:
        g = te[te.intended_emotion == e]
        idx = rng.choice(len(g), size=min(per, len(g)), replace=False)
        picked += [g.iloc[int(i)] for i in idx]
    rng.shuffle(picked)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.wav"):
        f.unlink()
    paths = []
    for r in picked[:n]:
        x = audio.read(adir / r.wav_file)  # resamples 16k -> the loop's 22050
        # The endpointer needs trailing silence to fire on, and the leading
        # silence keeps the first partial decode from starting mid-word. Same
        # padding aliveness-threshold applies to its rendered prompts.
        x = np.concatenate([np.zeros(audio.samples(lead_ms), np.float32), audio.trim(x),
                            np.zeros(audio.samples(tail_ms), np.float32)])
        p = out / f"{r.intended_emotion}__{Path(r.wav_file).stem}.wav"
        audio.write(p, x)
        paths.append(str(p))
    print(f"{len(paths)} user turns -> {out}/  "
          f"(actors held out of this checkpoint's training set)")
    return paths


def true_emotion(label: str) -> str | None:
    """Recover the ground-truth emotion from a stimulus filename stem."""
    e = label.split("__")[0]
    return e if e in emo.EMOTIONS else None


def score(run_json: str, out_path=None) -> dict:
    """Classifier accuracy *inside the live loop*, against CREMA-D actor intent.

    This is the number that says whether the perception stage works in the
    system, as opposed to in a notebook. It is measured on the same held-out
    actors the checkpoint reports 74.9% on, so a large drop is attributable to
    the loop -- resampling, the endpointer's crop, the 4-second cap -- and not
    to a harder test set.
    """
    run = json.loads(Path(run_json).read_text())
    rows = [(true_emotion(t["label"]), t["detected_emotion"], t["emotion_confidence"])
            for t in run["turns"]]
    rows = [r for r in rows if r[0]]
    if not rows:
        raise SystemExit(f"{run_json} was not driven by CREMA-D stimuli; nothing to score")
    hit = sum(a == b for a, b, _ in rows)
    conf = {}
    for a, b, _ in rows:
        conf.setdefault(a, {}).setdefault(b, 0)
        conf[a][b] += 1
    per = {e: {"n": sum(v.values()), "recall": round(v.get(e, 0) / sum(v.values()), 3)}
           for e, v in sorted(conf.items())}
    res = {"source_run": run_json, "n": len(rows),
           "acc_vs_actor_intent": round(hit / len(rows), 4),
           "reported_offline_acc_vs_intent": 0.749,
           "reported_offline_acc_vs_crowd_consensus": 0.522,
           "per_class": per, "confusion": conf,
           "note": "same held-out CREMA-D test actors the checkpoint was evaluated on, "
                   "fed through the live loop (22050 Hz capture, endpointer crop, "
                   "resample back to 16 kHz, 4 s cap) instead of straight off disk."}
    print(f"in-loop accuracy vs actor intent: {hit}/{len(rows)} = {hit/len(rows):.3f}  "
          f"(offline, same actors, straight off disk: 0.749)")
    for e, d in per.items():
        print(f"  {e:<8} n={d['n']:<3} recall={d['recall']:.3f}")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(res, indent=2))
        print(f"wrote {out_path}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "score"])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default="runs/user-turns")
    ap.add_argument("--run")
    a = ap.parse_args()
    if a.cmd == "build":
        for q in build(a.n, a.out):
            print(" ", q)
    else:
        score(a.run, a.out if a.out != "runs/user-turns" else None)
