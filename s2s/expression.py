"""Does the expressive output actually differ by emotion? Measured, not asserted.

Three separate questions, three subcommands, because they have different
answers and conflating them would be the easiest way to overclaim here.

  sweep    Hold the text fixed, vary only the preset the reply is spoken with,
           extract acoustic features. This isolates the *expression stage*: it
           says what the prosody control surface can do at all, with the
           classifier out of the picture entirely.

  live     Take the reply wavs a real run produced, group them by the emotion
           the classifier detected on that turn's user audio, and ask whether
           the groups separate. This is the end-to-end claim, and it inherits
           every error the classifier makes.

  probe    Point the classifier at audio it was not trained on and report what
           it says. CREMA-D test clips are in-domain (it should reproduce the
           74.9% it was trained to); the piper-rendered prompts are the
           out-of-domain case, and there is no ground truth for them -- only
           the observation that a single flat synthetic voice should not
           produce six confident different emotions.

Features come from vendor/features.py (expressive-tts-audit). Group separation
is Kruskal-Wallis on the raw feature, non-parametric because n per group is
small and nothing here is normal.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path

import numpy as np

from s2s import emotion as emo
from s2s import prosody
from vendor import audio, features

ROOT = Path(__file__).resolve().parent.parent

# The features that a rate/variability/level control surface could plausibly
# move. Reported in full; this is the reading order, not a filter.
HEADLINE = ["f0_mean", "f0_sd", "f0_range", "speech_rate", "articulation_rate",
            "rms_mean_db", "rms_range_db", "duration", "pause_frac", "hnr"]


def _iqr(v):
    v = sorted(float(x) for x in v if x is not None and not np.isnan(x))
    if not v:
        return None
    q = statistics.quantiles(v, n=4) if len(v) > 3 else [v[0], statistics.median(v), v[-1]]
    return {"n": len(v), "median": round(statistics.median(v), 3),
            "p25": round(q[0], 3), "p75": round(q[2], 3)}


def _kruskal(groups: dict[str, list[float]]):
    from scipy.stats import kruskal  # noqa: PLC0415

    vals = [[x for x in v if x is not None and not np.isnan(x)] for v in groups.values()]
    vals = [v for v in vals if len(v) >= 2]
    if len(vals) < 2:
        return None
    try:
        h, p = kruskal(*vals)
    except ValueError:  # all values identical -> no variance, no separation
        return {"H": 0.0, "p": 1.0, "k": len(vals)}
    return {"H": round(float(h), 3), "p": float(p), "k": len(vals)}


def _feats(y: np.ndarray, text: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        p = Path(f.name)
    audio.write(p, y)
    try:
        return features.extract(p, text)
    finally:
        p.unlink(missing_ok=True)


def _table(rows: list[dict], by: str, out_path=None) -> dict:
    """rows: [{by: label, **features}] -> per-group stats + separation test."""
    labels = sorted({r[by] for r in rows})
    res = {"grouped_by": by, "groups": {lab: sum(1 for r in rows if r[by] == lab)
                                        for lab in labels},
           "n_total": len(rows), "features": {}}
    for f in features.FEATURES:
        g = {lab: [r[f] for r in rows if r[by] == lab] for lab in labels}
        res["features"][f] = {"per_group": {k: _iqr(v) for k, v in g.items()},
                              "kruskal": _kruskal(g)}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(res, indent=2))
        print(f"wrote {out_path}")
    return res


def _print(res: dict, keys=None):
    keys = keys or HEADLINE
    labels = list(res["groups"])
    w = max(12, max(len(k) for k in keys) + 1)
    print(f"\n{'feature':<{w}}" + "".join(f"{l[:11]:>12}" for l in labels)
          + f"{'H':>9}{'p':>10}")
    print("-" * (w + 12 * len(labels) + 19))
    for k in keys:
        d = res["features"][k]
        row = f"{k:<{w}}"
        for l in labels:
            s = d["per_group"].get(l)
            row += f"{s['median']:>12.2f}" if s else f"{'-':>12}"
        kw = d["kruskal"]
        row += f"{kw['H']:>9.2f}{kw['p']:>10.2g}" if kw else f"{'-':>9}{'-':>10}"
        print(row)
    print(f"\nn per group: " + ", ".join(f"{l}={res['groups'][l]}" for l in labels))


# --- sweep -------------------------------------------------------------------

SENTENCES = [
    "We close at six on Sunday.",
    "There is parking behind the building.",
    "The annual pass is ninety dollars.",
    "Dogs are welcome in the courtyard.",
    "Your order is at the front desk.",
]


def sweep(reps: int = 3, out_path=None, backends=("piper", "say")):
    """Same sentences, every preset, on both engines, timed.

    The classifier is not involved. This isolates two things at once: whether
    the prosody surface moves acoustics at all, and what each engine charges in
    synthesis wall time to do it. That pair is the tradeoff.
    """
    from s2s.loop import pick_voice  # noqa: PLC0415
    import time  # noqa: PLC0415

    out = {"backends": {}, "emotion_to_preset": prosody.EMOTION_TO_PRESET,
           "note": ("Text held constant across conditions, so any acoustic difference "
                    "is the control surface, not word choice. F0 columns are reported "
                    "for piper to show it cannot move them, not because a change was "
                    "expected: piper exposes no pitch knob.")}
    for backend in backends:
        voice = pick_voice(backend)
        rows, synth_ms = [], []
        for e in emo.EMOTIONS:
            name, cfg = prosody.preset_for(e, backend)
            for si, text in enumerate(SENTENCES):
                for r in range(reps):
                    t0 = time.perf_counter()
                    y = voice.synth(text, cfg=cfg)
                    synth_ms.append((time.perf_counter() - t0) * 1000.0)
                    rows.append({"emotion": e, "preset": name, "sent": si, "rep": r,
                                 **_feats(y, text)})
                    print(".", end="", flush=True)
        print()
        res = _table(rows, "preset")
        res["synth_ms"] = _iqr(synth_ms)
        res["voice"] = voice.name
        out["backends"][backend] = res
        print(f"\n=== {backend} ({voice.name}) -- synthesis {res['synth_ms']['median']:.0f} ms "
              f"[{res['synth_ms']['p25']:.0f}-{res['synth_ms']['p75']:.0f}], "
              f"n={res['synth_ms']['n']} ===")
        _print(res)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {out_path}")
    return out


# --- live --------------------------------------------------------------------

def live(run_json: str, out_path=None):
    """Reply wavs from a real run, grouped by the emotion detected that turn."""
    run = json.loads(Path(run_json).read_text())
    rows = []
    for t in run["turns"]:
        w = t.get("reply_wav")
        if not w or not Path(w).exists():
            continue
        rows.append({"emotion": t["detected_emotion"], "preset": t["prosody_preset"],
                     "turn": t["turn"], **features.extract(Path(w), t["reply"])})
    if not rows:
        raise SystemExit(f"{run_json} has no saved reply wavs; rerun with --save-wavs")
    res = _table(rows, "emotion")
    res["source_run"] = run_json
    res["n_turns_in_run"] = run["n_turns"]
    res["detected_emotion_counts"] = run["detected_emotion_counts"]
    res["note"] = (
        "Reply TEXT differs between turns here, unlike the sweep, so a difference "
        "in duration or rate is partly the sentence. Read this next to the sweep, "
        "which holds text fixed."
    )
    if out_path:
        Path(out_path).write_text(json.dumps(res, indent=2))
    _print(res)
    return res


# --- probe -------------------------------------------------------------------

def probe(n_crema: int = 300, out_path=None):
    """What the classifier says on in-domain and out-of-domain audio.

    In-domain: held-out CREMA-D test actors from the run this checkpoint came
    from. Reproducing its reported accuracy is how we know the vendored loader
    is the same model, not a differently-wired copy.

    Out-of-domain: the piper prompt utterances the live loop feeds itself, plus
    the prosody presets applied to them. There is no ground truth. The only
    thing to look at is whether a single flat synthetic voice, unchanged in
    intent, gets called six different things with high confidence.
    """
    import pandas as pd  # noqa: PLC0415

    from s2s.loop import PROMPTS, pick_voice  # noqa: PLC0415

    clf = emo.Classifier()
    res = {"ckpt": str(clf.ckpt), "device": clf.device}

    # --- in domain -----------------------------------------------------------
    elc = Path.home() / "Desktop/Playground/emotion-label-ceiling"
    cfg = json.loads((elc / "modeling/runs/wav2vec2-base-intended_emotion-actor-s0"
                      / "config.json").read_text())
    try:
        clips = pd.read_parquet(elc / "data/clips.parquet")
        adir = next(p for p in [elc / "data/audio/repo/AudioWAV", elc / "data/raw/AudioWAV",
                                elc / "data/raw/repo/AudioWAV"] if p.is_dir())
        te = clips[clips.actor_id.isin(cfg["split"]["test"])].reset_index(drop=True)
        te = te.sample(n=min(n_crema, len(te)), random_state=0)
        hit = tot = 0
        for _, r in te.iterrows():
            x = audio.read(adir / r.wav_file)
            p = clf.predict(x, audio.SR)
            hit += int(p["emotion"] == r.intended_emotion)
            tot += 1
        res["in_domain"] = {
            "source": "CREMA-D held-out test actors (actor-disjoint)",
            "n": tot, "acc_vs_intended": round(hit / tot, 4),
            "reported_by_training_run": 0.749,
            "note": "sampled subset of the 1640-clip test set, so it will not land "
                    "exactly on the reported figure; it should land near it.",
        }
        print(f"in-domain: {hit}/{tot} = {hit/tot:.3f} vs intent "
              f"(training run reported 0.749 on all 1640)")
    except Exception as e:  # noqa: BLE001
        res["in_domain"] = {"error": f"{type(e).__name__}: {e}",
                            "note": "CREMA-D audio not on this machine; in-domain "
                                    "check skipped, NOT passed"}
        print(f"in-domain check skipped: {e}")

    # --- out of domain -------------------------------------------------------
    voice = pick_voice("piper")
    ood = []
    for text in PROMPTS:
        for e in emo.EMOTIONS:
            _, c = prosody.preset_for(e, "piper")
            y = voice.synth(text, cfg=c)
            p = clf.predict(y, audio.SR)
            ood.append({"text": text, "spoken_as": prosody.EMOTION_TO_PRESET[e],
                        "pred": p["emotion"], "conf": p["confidence"]})
    counts: dict[str, int] = {}
    for r in ood:
        counts[r["pred"]] = counts.get(r["pred"], 0) + 1
    res["out_of_domain"] = {
        "source": "piper en_US-lessac-medium, the loop's own prompt utterances, "
                  "rendered under each prosody preset",
        "ground_truth": None,
        "n": len(ood), "pred_counts": counts,
        "median_confidence": round(statistics.median(r["conf"] for r in ood), 4),
        "rows": ood,
        "note": "No ground truth exists for synthetic speech with no acted intent. "
                "A wide spread of confident labels over one flat voice is evidence "
                "the classifier is reading something other than emotion.",
    }
    print(f"out-of-domain (n={len(ood)}, synthetic, no ground truth): {counts}, "
          f"median confidence {res['out_of_domain']['median_confidence']:.2f}")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(res, indent=2))
        print(f"wrote {out_path}")
    return res


def demo():
    """Self-check for the feature path, adapted from expressive-tts-audit's own.

    A rate change MUST move duration on both engines, and a pitch change MUST
    move F0 on the engine that has a pitch knob. If either fails, the extractor
    and the synthesizer are not connected and every expression number is
    fiction.
    """
    from s2s.loop import pick_voice  # noqa: PLC0415

    v = pick_voice("piper")
    text = "The annual pass is ninety dollars."
    # length_scale alone, every other knob held at neutral. The excited and sad
    # presets differ on noise_w_scale too, which moves duration the *other* way,
    # so comparing whole presets made this assertion flake at the 1.2x threshold
    # -- it was testing two knobs fighting rather than the one it names.
    lenscale = lambda s: dict(prosody.PIPER_PRESETS["neutral"], length_scale=s)  # noqa: E731
    fast = _feats(v.synth(text, cfg=lenscale(0.80)), text)
    slow = _feats(v.synth(text, cfg=lenscale(1.30)), text)
    sv = pick_voice("say")
    lo = _feats(sv.synth(text, cfg=dict(rate=180, pbas=30, pmod=50, volm=1.0)), text)
    hi = _feats(sv.synth(text, cfg=dict(rate=180, pbas=70, pmod=50, volm=1.0)), text)
    assert hi["f0_mean"] > lo["f0_mean"] * 1.5, ("say pbas moved no F0",
                                                 lo["f0_mean"], hi["f0_mean"])
    for name, d in [("excited", fast), ("sad", slow)]:
        assert set(d) == set(features.FEATURES), f"{name}: feature set drifted"
        bad = [k for k, val in d.items()
               if val is None or (isinstance(val, float) and np.isnan(val))]
        assert not bad, f"{name}: NaN features {bad}"
    assert slow["duration"] > fast["duration"] * 1.2, (fast["duration"], slow["duration"])
    assert fast["speech_rate"] > slow["speech_rate"], (fast["speech_rate"], slow["speech_rate"])

    # The third backend exists precisely to give piper the pitch knob it does
    # not have, so the same assertion that catches a disconnected `say` has to
    # be made against it -- and it is the one that would fail if the WORLD
    # round trip were silently passing piper's own audio through.
    tv = pick_voice("transplant")
    tlo = _feats(tv.synth(text, cfg=prosody.TRANSPLANT_PRESETS["sad"]), text)
    thi = _feats(tv.synth(text, cfg=prosody.TRANSPLANT_PRESETS["excited"]), text)
    assert thi["f0_mean"] > tlo["f0_mean"] * 1.5, ("the F0 transplant moved no F0",
                                                   tlo["f0_mean"], thi["f0_mean"])
    print(f"expression self-check OK  (piper duration {fast['duration']:.2f}s excited -> "
          f"{slow['duration']:.2f}s sad; say f0 {lo['f0_mean']:.0f} -> {hi['f0_mean']:.0f} Hz; "
          f"transplant f0 {tlo['f0_mean']:.0f} -> {thi['f0_mean']:.0f} Hz; "
          f"{len(features.FEATURES)} features, no NaNs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["sweep", "live", "probe", "selfcheck"])
    ap.add_argument("--run", help="run JSON for `live`")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-crema", type=int, default=300)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "sweep":
        sweep(a.reps, a.out)
    elif a.cmd == "live":
        live(a.run, a.out)
    elif a.cmd == "probe":
        probe(a.n_crema, a.out)
    else:
        demo()


if __name__ == "__main__":
    main()
