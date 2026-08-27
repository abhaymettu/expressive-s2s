"""Print every fast-path number from the run JSONs. Nothing typed from memory.

Same job summarize.py does for the rest of the README: if a figure is in the
prose and not in this output, it should not be in the prose.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ARMS = [
    ("runs/f0-control-serial.json", "F0  control, serial"),
    ("runs/f1-fast-arm80.json", "F1  --fast --arm 80"),
    ("runs/f2-fast-arm80-emopar.json", "F2  F1 + --emotion-parallel"),
    ("runs/f3-fast-tiny.json", "F3  F2 + tiny.en final"),
    ("runs/f0b-control-serial.json", "F0b control again, end of session"),
    ("runs/t0-transplant-serial.json", "T0  F0 transplant, serial"),
    ("runs/t1-transplant-fast.json", "T1  F0 transplant, --fast --arm 80"),
    ("runs/h0-human-control.json", "H0  CREMA-D actors, serial"),
    ("runs/h1-human-fast.json", "H1  CREMA-D actors, --fast --arm 80"),
]

STAGES = ["endpoint_hangover_ms", "asr_final_dispatch_ms", "asr_final_ms",
          "emotion_ms", "lm_ttft_ms", "lm_sentence_ms", "tts_ms",
          "playback_dispatch_ms"]


def main(paths=None) -> int:
    arms = [(p, lbl) for p, lbl in ARMS if Path(p).exists()] if paths is None \
        else [(p, p) for p in paths]
    if not arms:
        print("no run JSONs yet")
        return 1

    print(f"{'arm':<36}{'gap median [IQR]':>26}{'n':>4}{'spec':>7}{'launch':>8}"
          f"{'falseEP':>9}{'cue':>6}")
    for p, lbl in arms:
        r = json.loads(Path(p).read_text())
        g = r["summary_ms"]["gap_ms"]
        sp, ep, cu = r["speculation"], r["endpointing"], r["cue"]
        print(f"{lbl:<36}{g['median']:>10.0f} ms [{g['p25']:.0f}-{g['p75']:.0f}]"
              f"{'':>{max(0, 26 - 10 - 4 - len(f'[{g['p25']:.0f}-{g['p75']:.0f}]'))}}"
              f"{g['n']:>4}"
              f"{sp['turns_served_speculatively']:>7}{sp['pipelines_launched']:>8}"
              f"{ep['false_endpoints']:>5}/{ep['n']:<3}{cu['n_fired']:>6}")

    print("\ngap, full spread")
    print(f"{'arm':<36}{'median':>8}{'p25':>8}{'p75':>8}{'min':>8}{'max':>8}"
          f"{'mean':>8}{'sd':>8}")
    for p, lbl in arms:
        g = json.loads(Path(p).read_text())["summary_ms"]["gap_ms"]
        print(f"{lbl:<36}" + "".join(f"{g[k]:>8.0f}" for k in
                                     ("median", "p25", "p75", "min", "max", "mean", "sd")))

    for p, lbl in arms:
        r = json.loads(Path(p).read_text())
        s = r["summary_ms"]
        print(f"\n{lbl}   ({r['mode']})")
        print(f"  loadavg {r['loadavg_start'][0]} -> {r['loadavg_end'][0]}   "
              f"out {r['output_device']} +{r['output_device_latency_ms']:.1f}ms   "
              f"asr {r['asr']['partial_model']}/{r['asr']['final_model']}")
        print(f"  {'stage':<26}{'charged [IQR]':>22}{'work [IQR]':>22}")
        for k in STAGES:
            if k not in s:
                continue
            c = s[k]
            w = s.get("work_ms." + k[:-3])
            wtxt = f"{w['median']:.0f} [{w['p25']:.0f}-{w['p75']:.0f}]" if w else "-"
            print(f"  {k:<26}{f'{c['median']:.0f} [{c['p25']:.0f}-{c['p75']:.0f}]':>22}"
                  f"{wtxt:>22}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
