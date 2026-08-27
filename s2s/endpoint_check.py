"""How often does the live endpointer cut in before the user actually stopped?

Reported because it showed up in the CREMA-D runs and it matters twice over:

- ``endpoint_hangover_ms`` is ``t_endpoint - true speech offset``. It is
  supposed to be ~+350 ms, the design constant. A NEGATIVE value means the
  endpointer called the turn over while the silence-trimmed measurement still
  says the user was speaking -- the loop cut them off.
- When that happens the gap it reports is short for the wrong reason, and the
  audio the classifier is handed is missing the end of the utterance.

The endpointer arms on chunk RMS against a running peak with an absolute floor
of ``LIVE_FLOOR_DBFS = -45``. That floor was set for a TTS voice at a steady
level. A real person trailing off at the end of a sentence goes under it while
``vendor.audio.segments`` (floor -55 dBFS) still counts them as speaking.

    .venv/bin/python -m s2s.endpoint_check runs/b-crema-piper.json
"""

import json
import statistics
import sys
from pathlib import Path


def check(*run_jsons):
    out = {}
    for p in run_jsons:
        r = json.loads(Path(p).read_text())
        h = [t["stage_ms"]["endpoint_hangover_ms"] for t in r["turns"]]
        early = [t for t in r["turns"] if t["stage_ms"]["endpoint_hangover_ms"] < 0]
        d = {
            "n": len(h),
            "design_constant_ms": r["hangover_ms"],
            "hangover_median_ms": round(statistics.median(h), 1),
            "hangover_min_ms": round(min(h), 1),
            "hangover_max_ms": round(max(h), 1),
            "n_cut_the_user_off": len(early),
            "frac_cut_off": round(len(early) / len(h), 3),
            "turns_cut_off": [{"turn": t["turn"], "label": t["label"],
                               "hangover_ms": t["stage_ms"]["endpoint_hangover_ms"],
                               "gap_ms": t["gap_ms"],
                               "transcript": t["transcript"]} for t in early],
        }
        out[p] = d
        print(f"{p}\n  hangover median {d['hangover_median_ms']:.0f} ms "
              f"(design {d['design_constant_ms']:.0f}), range "
              f"{d['hangover_min_ms']:.0f} to {d['hangover_max_ms']:.0f}\n"
              f"  cut the user off on {d['n_cut_the_user_off']}/{d['n']} turns "
              f"({d['frac_cut_off']:.1%})")
    return out


if __name__ == "__main__":
    res = check(*sys.argv[1:])
    Path("out").mkdir(exist_ok=True)
    Path("out/endpointer.json").write_text(json.dumps(res, indent=2))
    print("wrote out/endpointer.json")
