#!/bin/bash
# The fast-path measurement pass. Every arm in one session, on one machine,
# interleaved with its own control -- because the only baseline this repo trusts
# is one measured next to the thing it is being compared against.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
DEV=$($PY -c "import sounddevice as sd;print(next((d['name'] for d in sd.query_devices() if 'BlackHole' in d['name'] and d['max_output_channels']>0),''))")
D=(); [ -n "$DEV" ] && D=(--device "$DEV")
mkdir -p runs out
N=${N:-20}
NH=${NH:-24}

echo "== self-checks, both paths =="
$PY -m s2s.loop selfcheck --n 3 "${D[@]}"
$PY -m s2s.loop selfcheck --n 3 --fast --arm 80 "${D[@]}"

echo "== F0: control, serial. Nothing starts before the endpoint =="
$PY -m s2s.loop batch --n "$N" --out runs/f0-control-serial.json "${D[@]}"

echo "== F1: --fast --arm 80, classifier serial inside the speculation =="
$PY -m s2s.loop batch --n "$N" --fast --arm 80 --out runs/f1-fast-arm80.json "${D[@]}"

echo "== F2: --fast --arm 80 --emotion-parallel =="
$PY -m s2s.loop batch --n "$N" --fast --arm 80 --emotion-parallel \
    --out runs/f2-fast-arm80-emopar.json "${D[@]}"

echo "== F3: F2 + tiny.en for the final decode =="
$PY -m s2s.loop batch --n "$N" --fast --arm 80 --emotion-parallel \
    --final-model tiny.en --out runs/f3-fast-tiny.json "${D[@]}"

echo "== F0b: the control again, at the end, to bound within-session drift =="
$PY -m s2s.loop batch --n "$N" --out runs/f0b-control-serial.json "${D[@]}"

echo "== T: the F0-transplant backend, on the fast path =="
$PY -m s2s.loop batch --n "$N" --tts transplant --out runs/t0-transplant-serial.json "${D[@]}"
$PY -m s2s.loop batch --n "$N" --tts transplant --fast --arm 80 --emotion-parallel \
    --out runs/t1-transplant-fast.json "${D[@]}"

echo "== H0/H1: real CREMA-D actors. Does --arm 80 cut a human off? =="
$PY -m s2s.loop batch --n "$NH" --prompt-wav-dir runs/user-turns \
    --out runs/h0-human-control.json "${D[@]}"
$PY -m s2s.loop batch --n "$NH" --fast --arm 80 --emotion-parallel \
    --prompt-wav-dir runs/user-turns --out runs/h1-human-fast.json "${D[@]}"

$PY fastsummary.py
