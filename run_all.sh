#!/bin/bash
# Everything, in order. Every number in README.md comes from this script.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
# Route playback to BlackHole if it is installed, so a 90-turn measurement pass
# does not play aloud on a shared machine. The loop stages do not depend on the
# device; only the reported output-device latency does.
DEV=$($PY -c "import sounddevice as sd;print(next((d['name'] for d in sd.query_devices() if 'BlackHole' in d['name'] and d['max_output_channels']>0),''))")
D=(); [ -n "$DEV" ] && D=(--device "$DEV")
mkdir -p out runs

echo "== self-checks =="
$PY -m s2s.expression selfcheck
$PY -m s2s.loop selfcheck --n 3 "${D[@]}"

echo "== stimuli: real CREMA-D held-out actors as the user =="
$PY -m s2s.stimuli build --n 24

echo "== A: piper prompts + piper out, the configuration comparable to the 1452ms baseline =="
$PY -m s2s.loop batch --n 20 --out runs/a-piper-prompts.json "${D[@]}"

echo "== B: CREMA-D actors + piper out (fast, flat) =="
$PY -m s2s.loop batch --n 24 --prompt-wav-dir runs/user-turns \
    --save-wavs runs/b-wavs --out runs/b-crema-piper.json "${D[@]}"

echo "== C: CREMA-D actors + macOS say out (expressive, slow) =="
$PY -m s2s.loop batch --n 24 --tts say --prompt-wav-dir runs/user-turns \
    --save-wavs runs/c-wavs --out runs/c-crema-say.json "${D[@]}"

echo "== D: what overlapping the classifier with the ASR decode buys =="
$PY -m s2s.loop batch --n 20 --emotion-parallel --out runs/d-emotion-parallel.json "${D[@]}"

echo "== E: no emotion stage at all, to price it by subtraction =="
$PY -m s2s.loop batch --n 20 --no-emotion --out runs/e-no-emotion.json "${D[@]}"

echo "== analysis =="
$PY -m s2s.stimuli score --run runs/b-crema-piper.json --out out/emotion-in-loop.json
$PY -m s2s.expression sweep --reps 3 --out out/sweep-both-engines.json
$PY -m s2s.expression live --run runs/b-crema-piper.json --out out/live-piper.json
$PY -m s2s.expression live --run runs/c-crema-say.json --out out/live-say.json
$PY -m s2s.expression probe --out out/emotion-probe.json
$PY -m s2s.endpoint_check runs/a-piper-prompts.json runs/b-crema-piper.json runs/c-crema-say.json

echo "== demo =="
$PY -m demo.record_demo runs/b-crema-piper.json demo/exchange.wav --turns 4
$PY -m demo.record_demo runs/c-crema-say.json demo/exchange-expressive.wav --turns 4
