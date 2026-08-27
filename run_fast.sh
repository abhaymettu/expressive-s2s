#!/bin/bash
# The fast-path measurement pass: sections 1.1, 1.3, and the third engine's
# columns in 2 and 3. Every arm in one session, interleaved, control first and
# control again at the end -- because the only baseline this repo trusts is one
# measured next to the thing it is being compared against.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
DEV=$($PY -c "import sounddevice as sd;print(next((d['name'] for d in sd.query_devices() if 'BlackHole' in d['name'] and d['max_output_channels']>0),''))")
D=(); [ -n "$DEV" ] && D=(--device "$DEV")
mkdir -p runs out
N=20; NH=24
echo "CODE $(git rev-parse HEAD)"; git diff --stat | tail -1
uptime

echo "== selfchecks =="
$PY -m s2s.expression selfcheck
$PY -m s2s.loop selfcheck --n 3 "${D[@]}"
$PY -m s2s.loop selfcheck --n 3 --fast --arm 80 "${D[@]}"

echo "== F0 control serial =="
$PY -m s2s.loop batch --n $N --out runs/f0-control-serial.json "${D[@]}"
echo "== F1 fast arm80 =="
$PY -m s2s.loop batch --n $N --fast --arm 80 --out runs/f1-fast-arm80.json "${D[@]}"
echo "== F2 fast arm80 emotion-parallel =="
$PY -m s2s.loop batch --n $N --fast --arm 80 --emotion-parallel --out runs/f2-fast-arm80-emopar.json "${D[@]}"
echo "== F3 fast arm80 emotion-parallel tiny.en final =="
$PY -m s2s.loop batch --n $N --fast --arm 80 --emotion-parallel --final-model tiny.en --out runs/f3-fast-tiny.json "${D[@]}"
echo "== F0b control serial again =="
$PY -m s2s.loop batch --n $N --out runs/f0b-control-serial.json "${D[@]}"

echo "== T0 transplant serial / T1 transplant fast =="
$PY -m s2s.loop batch --n $N --tts transplant --save-wavs runs/t0-wavs --out runs/t0-transplant-serial.json "${D[@]}"
$PY -m s2s.loop batch --n $N --tts transplant --fast --arm 80 --emotion-parallel --save-wavs runs/t1-wavs --out runs/t1-transplant-fast.json "${D[@]}"

echo "== H0/H1 real CREMA-D actors =="
$PY -m s2s.loop batch --n $NH --prompt-wav-dir runs/user-turns --out runs/h0-human-control.json "${D[@]}"
$PY -m s2s.loop batch --n $NH --fast --arm 80 --emotion-parallel --prompt-wav-dir runs/user-turns --out runs/h1-human-fast.json "${D[@]}"

echo "== expression: three engines, text held constant (3a) =="
$PY -m s2s.expression sweep --reps 3 --backends piper,transplant,say --out out/sweep-three-engines.json

echo "== expression: end to end through the transplant, real actors (3b) =="
$PY -m s2s.loop batch --n $NH --tts transplant --fast --arm 80 --emotion-parallel \
    --prompt-wav-dir runs/user-turns --save-wavs runs/tc-wavs \
    --out runs/tc-crema-transplant.json "${D[@]}"
$PY -m s2s.expression live --run runs/tc-crema-transplant.json --out out/live-transplant.json

echo "CODE_AFTER $(git rev-parse HEAD)"; git diff --stat | tail -1
uptime
$PY fastsummary.py
