"""Print every headline number in README.md, straight out of runs/ and out/.

Exists so nothing in the README is typed from memory. If a number is in the
README and not in this output, it should not be in the README.

    .venv/bin/python summarize.py
"""

import json
from pathlib import Path

BASE = 1452.0  # aliveness-threshold live/STATUS.md, n=40, same machine


def s(d):
    return f"{d['median']:.0f} [{d['p25']:.0f}-{d['p75']:.0f}]" if d else "-"


def run(p):
    if not Path(p).exists():
        print(f"\n### {p}  -- MISSING")
        return None
    r = json.loads(Path(p).read_text())
    m = r["summary_ms"]
    print(f"\n### {p}")
    print(f"n={r['n_turns']}  in={r['input'][:60]}  tts={r['tts']['backend']}  "
          f"emotion={r['emotion']['schedule'][:40]}")
    print(f"load {r['loadavg_start'][0]:.1f}->{r['loadavg_end'][0]:.1f}  "
          f"dev={r['output_device']} ({r['output_device_latency_ms']:.0f}ms)")
    print(f"  gap_ms          {s(m['gap_ms'])}   n={m['gap_ms']['n']}  "
          f"mean {m['gap_ms']['mean']:.0f} sd {m['gap_ms']['sd']:.0f}  "
          f"min {m['gap_ms']['min']:.0f} max {m['gap_ms']['max']:.0f}")
    print(f"  vs 1452 baseline: {m['gap_ms']['median'] - BASE:+.0f} ms")
    if "first_audio_ms" in m:
        print(f"  first_audio_ms  {s(m['first_audio_ms'])}  "
              f"(cue fired on {r['cue']['n_fired']}/{r['n_turns']} turns)")
    if "cue_gap_ms" in m:
        print(f"  cue_gap_ms      {s(m['cue_gap_ms'])}")
    print(f"  acoustic_gap_ms {s(m['acoustic_gap_ms'])}")
    for k, v in m.items():
        if k.startswith("stage_ms."):
            print(f"    {k[9:]:<24} {s(v)}")
    if "emotion_standalone_ms" in m:
        print(f"    (classifier alone)       {s(m['emotion_standalone_ms'])}")
    print(f"  detected: {r['detected_emotion_counts']}")
    return r


print("=" * 78)
print("LATENCY")
print("=" * 78)
a = run("runs/a-piper-prompts.json")
b = run("runs/b-crema-piper.json")
c = run("runs/c-crema-say.json")
d = run("runs/d-emotion-parallel.json")
e = run("runs/e-no-emotion.json")

if a and e:
    print(f"\n>> emotion cost, serial in-path stage:      "
          f"{a['summary_ms']['stage_ms.emotion_ms']['median']:.0f} ms median")
    print(f">> emotion cost, by subtraction (A - E gap): "
          f"{a['summary_ms']['gap_ms']['median'] - e['summary_ms']['gap_ms']['median']:+.0f} ms")
if a and d:
    print(f">> emotion cost, overlapped with ASR (D):    "
          f"{d['summary_ms']['stage_ms.emotion_ms']['median']:.0f} ms median in-path; "
          f"gap {d['summary_ms']['gap_ms']['median']:.0f} vs {a['summary_ms']['gap_ms']['median']:.0f}")
if b and c:
    print(f">> piper vs say gap: {b['summary_ms']['gap_ms']['median']:.0f} vs "
          f"{c['summary_ms']['gap_ms']['median']:.0f} ms  "
          f"(+{c['summary_ms']['gap_ms']['median'] - b['summary_ms']['gap_ms']['median']:.0f})")
    print(f">> piper vs say tts stage: {b['summary_ms']['stage_ms.tts_ms']['median']:.0f} vs "
          f"{c['summary_ms']['stage_ms.tts_ms']['median']:.0f} ms")

print("\n" + "=" * 78)
print("PERCEPTION")
print("=" * 78)
p = Path("out/emotion-in-loop.json")
if p.exists():
    r = json.loads(p.read_text())
    print(f"in-loop vs actor intent: {r['acc_vs_actor_intent']:.3f}  n={r['n']}  "
          f"(offline same actors: {r['reported_offline_acc_vs_intent']})")
    for k, v in r["per_class"].items():
        print(f"  {k:<8} n={v['n']:<3} recall={v['recall']:.3f}")
    print(f"  confusion: {json.dumps(r['confusion'])}")
p = Path("out/emotion-probe.json")
if p.exists():
    r = json.loads(p.read_text())
    print(f"\nin-domain straight off disk: {r['in_domain']}")
    o = r["out_of_domain"]
    print(f"out-of-domain (synthetic, no ground truth): n={o['n']} "
          f"preds={o['pred_counts']} median conf={o['median_confidence']}")

print("\n" + "=" * 78)
print("EXPRESSION")
print("=" * 78)
p = Path("out/sweep-both-engines.json")
if p.exists():
    r = json.loads(p.read_text())
    for eng, res in r["backends"].items():
        print(f"\n-- {eng}  synth {s(res['synth_ms'])} ms  n={res['n_total']}")
        for f in ["f0_mean", "f0_sd", "f0_range", "duration", "speech_rate",
                  "rms_mean_db", "hnr"]:
            g = res["features"][f]
            meds = {k: v["median"] for k, v in g["per_group"].items() if v}
            span = max(meds.values()) - min(meds.values())
            kw = g["kruskal"]
            print(f"   {f:<14} {meds}  span={span:.2f}  "
                  f"H={kw['H']:.1f} p={kw['p']:.2g}")
for name in ["out/live-piper.json", "out/live-say.json"]:
    p = Path(name)
    if not p.exists():
        continue
    r = json.loads(p.read_text())
    print(f"\n-- {name}  groups={r['groups']}")
    for f in ["f0_mean", "f0_range", "duration", "speech_rate", "rms_mean_db"]:
        g = r["features"][f]
        meds = {k: v["median"] for k, v in g["per_group"].items() if v}
        kw = g["kruskal"]
        print(f"   {f:<14} {meds}  H={kw['H']:.1f} p={kw['p']:.2g}" if kw
              else f"   {f:<14} {meds}")

print("\n" + "=" * 78)
print("ENDPOINTER")
print("=" * 78)
p = Path("out/endpointer.json")
if p.exists():
    for k, v in json.loads(p.read_text()).items():
        print(f"{k}: hangover median {v['hangover_median_ms']:.0f} ms "
              f"(design {v['design_constant_ms']:.0f}), range {v['hangover_min_ms']:.0f}"
              f" to {v['hangover_max_ms']:.0f}, cut off {v['n_cut_the_user_off']}/{v['n']}")
