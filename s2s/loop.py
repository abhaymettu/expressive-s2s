"""An agent that hears how you sound and answers in a voice that matches.

    audio in -> ASR -> EMOTION -> LM (conditioned) -> TTS (prosody) -> audio out
                       ^^^^^^^          ^^^^^^^^^^        ^^^^^^^^
                       new here         new here          new here

The skeleton -- the streaming input queue, the two-model ASR, the endpointer,
the token-streamed LM, the held-open output stream, and above all the *gap
definition* -- is adapted from
~/Desktop/Playground/aliveness-threshold/live/loop.py
(github.com/abhaymettu/aliveness-threshold). That file measured a median gap of
**1452 ms over n = 40** on this laptop. Every latency number here is measured
the same way so it can be put next to that one:

    gap = onset of agent speech - offset of user speech, both silence-trimmed,
    user offset re-measured with vendor.audio.segments(merge_gap_ms=30,
    min_len_ms=20) -- the end of the last speech segment, NOT the moment the
    endpointer noticed.

Three things are new, and each one is a stage on the wall clock:

1. **Emotion.** The wav2vec2 head from emotion-label-ceiling runs on the
   captured user audio, serially, after the final ASR decode. Serially on
   purpose: it keeps the "stages sum to the gap" invariant true, so the cost of
   emotion detection is a number you can read straight off the table instead of
   an argument about thread scheduling. `--emotion-parallel` runs it on a
   worker alongside the ASR decode instead, and reports what the overlap buys.
2. **Conditioning.** The detected emotion becomes one clause in the LM system
   prompt.
3. **Expression + cue.** The reply is synthesized with a piper preset chosen by
   the detected emotion, and if the reply is not ready by CUE_AT_MS after the
   user stops talking, a non-verbal cue is played into the gap first, using
   vendor/cues.py.

The cue changes what the listener hears first, so it is reported as its own
number and never folded into `gap_ms`:

    gap_ms            reply speech onset  - user speech offset   <- comparable to 1452 ms
    first_audio_ms    ANY agent audio     - user speech offset   <- what the ear waits
    acoustic_gap_ms   gap_ms + the output device's own latency
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import statistics
import threading
import time
from pathlib import Path

import numpy as np

from s2s import emotion as emo
from s2s import prosody
from vendor import audio, cues, tts

ROOT = Path(__file__).resolve().parent.parent

# --- unchanged from aliveness-threshold/live/loop.py ------------------------
CHUNK_MS = 20.0
HANGOVER_MS = 350.0
PARTIAL_EVERY_MS = 500.0
MAX_TURN_MS = 20000.0
LIVE_FLOOR_DBFS = -45.0
BLOCK = 128
MAX_TOKENS = 48
ASR_MODEL = "base.en"
PARTIAL_MODEL = "tiny.en"
LM_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
SEG_KW = {"merge_gap_ms": 30.0, "min_len_ms": 20.0}

# --- new here ---------------------------------------------------------------
SYSTEM = "You are a voice assistant. Reply in one short spoken sentence. {hint}"
# If no reply audio by this point after the user stops, put a cue in the gap.
# 400 ms is roughly the top of the range where a human turn transition still
# feels unmarked; past it the silence is the thing being noticed.
CUE_AT_MS = 400.0
CUE = "filled_pause"

PROMPTS = [
    "What time do you close on Sunday?",
    "Is there parking near the entrance?",
    "How much does the annual pass cost?",
    "Can I bring a dog inside?",
    "Where do I pick up my order?",
]


SAY_VOICE = "Alex"  # the voice expressive-tts-audit measured; pbas/volm work on it


def pick_voice(which: str = "auto"):
    """`auto` and `piper` -> the piper voice in models/piper/. `say` -> macOS
    `say` with the Alex voice, explicitly, never by autodetect fallback."""
    if which == "say":
        return tts.Voice("say", name=SAY_VOICE)
    found = sorted((ROOT / "models" / "piper").glob("*.onnx"))
    if found:
        return tts.Voice("piper", name=str(found[0]))
    if which == "piper":
        raise RuntimeError("piper requested but no .onnx voice in models/piper/")
    return tts.default_voice()


def _to_whisper(x: np.ndarray) -> np.ndarray:
    n = int(round(len(x) * 16000 / audio.SR))
    return np.interp(
        np.linspace(0, len(x) - 1, n), np.arange(len(x)), x.astype(np.float64)
    ).astype(np.float32)


class Asr:
    def __init__(self, model: str = ASR_MODEL):
        from faster_whisper import WhisperModel  # noqa: PLC0415

        self.name = model
        self.m = WhisperModel(model, device="cpu", compute_type="int8")

    def text(self, x: np.ndarray) -> str:
        segs, _ = self.m.transcribe(
            _to_whisper(x), language="en", beam_size=1,
            temperature=0.0, condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segs).strip()


class Lm:
    def __init__(self, model: str = LM_MODEL):
        from mlx_lm import load  # noqa: PLC0415

        self.name = model
        self.model, self.tok = load(model)

    def first_sentence(self, user_text: str, hint: str):
        from mlx_lm import stream_generate  # noqa: PLC0415

        prompt = self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM.format(hint=hint)},
             {"role": "user", "content": user_text}],
            add_generation_prompt=True, tokenize=False,
        )
        out, t_first, n = "", None, 0
        for r in stream_generate(self.model, self.tok, prompt, max_tokens=MAX_TOKENS):
            if t_first is None:
                t_first = time.perf_counter()
            out += r.text
            n += 1
            if re.search(r"[.!?](\s|$)", out) and len(out.strip()) > 8:
                break
        m = re.search(r"^(.*?[.!?])(\s|$)", out.strip(), re.S)
        return (m.group(1) if m else out.strip()), t_first, time.perf_counter(), n


class Player:
    """One output stream held open for the session.

    Changed from upstream: it takes a *named* clip and stamps a first-sample
    time per name, because a turn can now emit two clips (a cue, then the
    reply) and they have to be timed separately.
    """

    def __init__(self, device=None):
        import sounddevice as sd  # noqa: PLC0415

        self.buf: np.ndarray | None = None
        self.pos = 0
        self.name: str | None = None
        self.t_first: dict[str, float] = {}
        self.stream = sd.OutputStream(
            samplerate=audio.SR, channels=1, dtype="float32", blocksize=BLOCK,
            device=device, latency="low", callback=self._cb,
        )
        self.stream.start()
        self.latency_ms = float(self.stream.latency) * 1000.0
        self.device = sd.query_devices(self.stream.device)["name"]

    def _cb(self, outdata, frames, _t, _status):
        outdata[:] = 0.0
        x = self.buf
        if x is None:
            return
        n = min(frames, len(x) - self.pos)
        if n <= 0:
            self.buf = None
            return
        if self.pos == 0 and self.name not in self.t_first:
            self.t_first[self.name] = time.perf_counter()
        outdata[:n, 0] = x[self.pos : self.pos + n]
        self.pos += n

    def play(self, name: str, x: np.ndarray) -> bool:
        """Start `x`. Returns True if it preempted something still playing.

        It preempts rather than queues on purpose. Queueing would put the tail
        of a cue *inside* the reply's gap, which would silently inflate the one
        number this repo exists to compare against 1452 ms. In practice the cue
        fires at 400 ms and is ~300 ms long, so the reply at ~1500 ms preempts
        nothing; `preempted` is reported per turn so that stays a fact and not
        an assumption.
        """
        cut = self.buf is not None
        self.pos, self.name, self.buf = 0, name, x
        return cut

    def new_turn(self) -> None:
        self.t_first.clear()

    def wait(self, timeout: float = 10.0) -> None:
        end = time.perf_counter() + timeout
        while self.buf is not None and time.perf_counter() < end:
            time.sleep(0.002)

    def close(self):
        self.stream.stop()
        self.stream.close()


def _pace_wav(x: np.ndarray, q: queue.Queue, t0: float) -> None:
    n = audio.samples(CHUNK_MS)
    for i in range(0, len(x), n):
        d = (t0 + (i + n) / audio.SR) - time.perf_counter()
        if d > 0:
            time.sleep(d)
        q.put(x[i : i + n].copy())
    q.put(None)


def _mic(q: queue.Queue, stop: threading.Event, box: dict):
    import sounddevice as sd  # noqa: PLC0415

    def cb(indata, frames, _t, _s):
        if "t0" not in box:
            box["t0"] = time.perf_counter() - frames / audio.SR
        q.put(indata[:, 0].copy())

    s = sd.InputStream(
        samplerate=audio.SR, channels=1, dtype="float32",
        blocksize=audio.samples(CHUNK_MS), callback=cb, latency="low",
    )
    s.start()
    stop.wait()
    s.stop()
    s.close()


def capture(source, asr: Asr, partial_asr: Asr, clf, parallel_emotion: bool,
            on_endpoint=None) -> dict:
    """Stream in, partial-decode in the background, stop on the endpointer,
    final-decode, and classify emotion.

    `on_endpoint(t_end)` is called the instant the endpointer fires, before any
    decoding starts. That is the only moment a real system can act on "the user
    has stopped": everything after it -- ASR, emotion, LM, TTS -- is the wait
    the cue exists to fill, so arming the cue anywhere later would arm it after
    the deadline it was supposed to beat.

    `parallel_emotion` decides whether the classifier runs on a worker thread
    started at the endpoint (overlapped with the ASR final decode) or serially
    after it. Both paths classify **the same array** -- the one this function
    is about to return -- and both record its sha, which is what the selfcheck
    checks.
    """
    q: queue.Queue = queue.Queue()
    box: dict = {}
    stop_mic = threading.Event()
    if source[0] == "wav":
        t0 = time.perf_counter()
        box["t0"] = t0
        feeder = threading.Thread(target=_pace_wav, args=(source[1], q, t0), daemon=True)
    else:
        feeder = threading.Thread(target=_mic, args=(q, stop_mic, box), daemon=True)
    feeder.start()

    chunks: list[np.ndarray] = []
    lock = threading.Lock()
    stop_partials = threading.Event()
    partial = {"t": None, "text": None}

    def partial_worker():
        seen = 0
        while not stop_partials.is_set():
            with lock:
                x = np.concatenate(chunks) if chunks else None
            if x is None or len(x) - seen < audio.samples(PARTIAL_EVERY_MS):
                time.sleep(0.005)
                continue
            seen = len(x)
            txt = partial_asr.text(x)
            if txt and partial["t"] is None:
                partial["t"], partial["text"] = time.perf_counter(), txt

    pw = threading.Thread(target=partial_worker, daemon=True)
    pw.start()

    peak, last_speech, t_end, ended = 0.0, None, None, False
    deadline = time.perf_counter() + MAX_TURN_MS / 1000.0
    while not ended:
        if time.perf_counter() > deadline:
            t_end, ended = time.perf_counter(), True
            break
        try:
            c = q.get(timeout=1.0)
        except queue.Empty:
            continue
        if c is None:
            t_end, ended = time.perf_counter(), True
            break
        with lock:
            chunks.append(c)
        now = time.perf_counter()
        r = float(np.sqrt((c.astype(np.float64) ** 2).mean()))
        peak = max(peak, r)
        if r >= peak * 10 ** (-35.0 / 20.0) and r >= 10 ** (LIVE_FLOOR_DBFS / 20.0):
            last_speech = now
        elif last_speech is not None and (now - last_speech) * 1000.0 >= HANGOVER_MS:
            t_end, ended = now, True

    if on_endpoint:
        on_endpoint(t_end)

    stop_mic.set()
    stop_partials.set()
    pw.join(timeout=10.0)
    x = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    # this exact array, and nothing else, is what the classifier is allowed to see
    turn_sha = emo.sha(x)
    box_emo: dict = {}

    if clf is None:
        # ablation arm: no perception stage at all, so the emotion cost can be
        # priced by subtraction against an otherwise identical run
        box_emo = {"emotion": "neutral", "confidence": None, "probs": None,
                   "ms": 0.0, "audio_sha": turn_sha, "fed_samples": 0,
                   "truncated": False, "ablated": True}
    elif parallel_emotion:
        t_emo0 = time.perf_counter()
        ew = threading.Thread(
            target=lambda: box_emo.update(clf.predict(x, audio.SR)), daemon=True)
        ew.start()

    t_stream0 = box["t0"]
    t_final0 = time.perf_counter()
    final = asr.text(x)
    t_final = time.perf_counter()

    if clf is None:
        t_emo = time.perf_counter()
        emotion_stage_ms = 0.0
    elif parallel_emotion:
        ew.join(timeout=30.0)
        t_emo = time.perf_counter()
        # what the overlap actually cost the gap: only the part of the classifier
        # that did not fit inside the ASR decode
        emotion_stage_ms = max(0.0, round((t_emo - t_final) * 1000.0, 2))
    else:
        t_emo0 = time.perf_counter()
        box_emo.update(clf.predict(x, audio.SR))
        t_emo = time.perf_counter()
        emotion_stage_ms = round((t_emo - t_emo0) * 1000.0, 2)

    if not box_emo:
        raise RuntimeError("emotion worker produced nothing")

    segs = audio.segments(x, **SEG_KW)
    if not segs:
        raise RuntimeError("no speech found in captured input; nothing to time against")
    return {
        "audio": x,
        "audio_sha": turn_sha,
        "t_stream0": t_stream0,
        "t_speech_onset": t_stream0 + segs[0][0] / 1000.0,
        "t_speech_offset": t_stream0 + segs[-1][1] / 1000.0,
        "t_endpoint": t_end,
        "t_first_partial": partial["t"],
        "partial_text": partial["text"],
        "t_final0": t_final0,
        "t_final": t_final,
        "t_emotion": t_emo,
        "emotion": box_emo,
        "emotion_stage_ms": emotion_stage_ms,
        "emotion_standalone_ms": box_emo["ms"],
        "transcript": final,
        "input_ms": audio.millis(len(x)),
        "n_segments": len(segs),
    }


def run_turn(source, asr, partial_asr, clf, lm, voice, player, cue_audio,
             label="", use_cue=True, parallel_emotion=False, save_wav=None) -> dict:
    player.new_turn()
    fired = threading.Event()
    cue_lock = threading.Lock()

    def on_endpoint(t_end):
        """Arm the cue the moment the endpointer fires.

        The reference point is the endpointer's own estimate of speech offset,
        ``t_end - HANGOVER_MS`` -- which is what a live system actually has.
        The *reported* cue offset is still measured against the silence-trimmed
        true offset, so the number in the results is not the estimate.
        """
        if not (use_cue and len(cue_audio)):
            return

        def arm():
            d = (t_end - HANGOVER_MS / 1000.0 + CUE_AT_MS / 1000.0) - time.perf_counter()
            if d > 0:
                time.sleep(d)
            # under the lock: once the reply has claimed the stream the cue must
            # not fire behind it. Without this the cue can land AFTER the reply
            # on a slow turn and be timed as if it came first.
            with cue_lock:
                if not fired.is_set():
                    player.play("cue", cue_audio)
                    fired.set()

        threading.Thread(target=arm, daemon=True).start()

    cap = capture(source, asr, partial_asr, clf, parallel_emotion, on_endpoint)
    off = cap["t_speech_offset"]
    detected = cap["emotion"]["emotion"]
    hint = prosody.hint_for(detected)
    sentence, t_tok, t_sent, n_tok = lm.first_sentence(cap["transcript"] or "Hello?", hint)

    preset_name, cfg = prosody.preset_for(detected, voice.backend)
    t_tts0 = time.perf_counter()
    y = voice.synth(sentence, cfg=cfg)
    t_tts = time.perf_counter()
    if len(y) == 0:
        raise RuntimeError(f"TTS produced no audio for {sentence!r}")

    with cue_lock:
        cue_played = fired.is_set()
        fired.set()  # past this point the reply owns the output stream
        preempted = player.play("reply", y)
    end = time.perf_counter() + 8.0
    while "reply" not in player.t_first and time.perf_counter() < end:
        time.sleep(0.001)
    if "reply" not in player.t_first:
        raise RuntimeError("output stream never consumed the reply audio")
    t_out = player.t_first["reply"]
    player.wait()

    if save_wav:
        Path(save_wav).parent.mkdir(parents=True, exist_ok=True)
        audio.write(save_wav, y)

    ms = lambda a, b: round((a - b) * 1000.0, 2)  # noqa: E731
    stage = {
        "asr_partial_first_ms": ms(cap["t_first_partial"], cap["t_stream0"])
        if cap["t_first_partial"] else None,
        "endpoint_hangover_ms": ms(cap["t_endpoint"], off),
        "asr_final_dispatch_ms": ms(cap["t_final0"], cap["t_endpoint"]),
        "asr_final_ms": ms(cap["t_final"], cap["t_final0"]),
        "emotion_ms": cap["emotion_stage_ms"],
        "lm_ttft_ms": ms(t_tok, cap["t_emotion"]),
        "lm_sentence_ms": ms(t_sent, t_tok),
        "tts_ms": ms(t_tts, t_tts0),
        "playback_dispatch_ms": ms(t_out, t_tts),
    }
    t_cue = player.t_first.get("cue")
    return {
        "label": label,
        "transcript": cap["transcript"],
        "partial_text": cap["partial_text"],
        "reply": sentence,
        "lm_tokens": n_tok,
        "detected_emotion": detected,
        "emotion_confidence": cap["emotion"]["confidence"],
        "emotion_probs": cap["emotion"]["probs"],
        "emotion_standalone_ms": cap["emotion_standalone_ms"],
        "emotion_audio_sha": cap["emotion"]["audio_sha"],
        "input_audio_sha": cap["audio_sha"],
        "prosody_preset": preset_name,
        "prosody_cfg": cfg,
        "cue": CUE if cue_played else None,
        "cue_gap_ms": ms(t_cue, off) if t_cue else None,
        "cue_preempted_by_reply": preempted,
        "input_ms": round(cap["input_ms"], 1),
        "user_speech_ms": ms(off, cap["t_speech_onset"]),
        "reply_audio_ms": round(audio.millis(len(y)), 1),
        "reply_wav": str(save_wav) if save_wav else None,
        # comparable, by construction, to aliveness-threshold live/STATUS.md
        "gap_ms": ms(t_out, off),
        "first_audio_ms": ms(t_cue if t_cue else t_out, off),
        "acoustic_gap_ms": round(ms(t_out, off) + player.latency_ms, 2),
        "stage_ms": stage,
        "n_input_segments": cap["n_segments"],
    }


def _stats(v: list[float]) -> dict:
    v = sorted(v)
    q = statistics.quantiles(v, n=4) if len(v) > 3 else [v[0], statistics.median(v), v[-1]]
    return {
        "n": len(v), "median": round(statistics.median(v), 1),
        "p25": round(q[0], 1), "p75": round(q[2], 1),
        "iqr": round(q[2] - q[0], 1), "min": round(v[0], 1), "max": round(v[-1], 1),
        "mean": round(statistics.fmean(v), 1),
        "sd": round(statistics.stdev(v), 1) if len(v) > 1 else 0.0,
    }


def render_prompts(voice, texts, lead_ms=300.0, tail_ms=900.0):
    out = []
    for text in texts:
        p = voice.synth(text)
        x = np.concatenate([np.zeros(audio.samples(lead_ms), np.float32), p,
                            np.zeros(audio.samples(tail_ms), np.float32)])
        out.append((text, x))
    return out


def run(n_turns=20, out_path=None, device=None, mic=False, tts_backend="auto",
        use_cue=True, parallel_emotion=False, prompt_wavs=None, save_wavs=None,
        no_emotion=False) -> dict:
    load0 = os.getloadavg()
    voice = pick_voice(tts_backend)
    prompt_voice = pick_voice("auto")
    t0 = time.perf_counter()
    asr, partial_asr, lm = Asr(), Asr(PARTIAL_MODEL), Lm()
    clf = None if no_emotion else emo.Classifier()
    load_ms = (time.perf_counter() - t0) * 1000.0
    player = Player(device)

    if prompt_wavs:
        prompts = [(Path(p).stem, audio.read(p)) for p in prompt_wavs]
    else:
        prompts = render_prompts(prompt_voice, PROMPTS)
    cue_audio = cues.cue_audio(CUE, voice=prompt_voice) if use_cue else np.zeros(0, np.float32)

    t0 = time.perf_counter()
    asr.text(prompts[0][1])
    partial_asr.text(prompts[0][1])
    if clf is not None:
        clf.predict(prompts[0][1], audio.SR)
    lm.first_sentence("Hello.", prosody.hint_for("neutral"))
    voice.synth("Ready.", cfg=prosody.PRESETS[voice.backend]["neutral"])
    warm_ms = (time.perf_counter() - t0) * 1000.0

    turns = []
    try:
        for i in range(n_turns):
            text, x = prompts[i % len(prompts)]
            src = ("mic", None) if mic else ("wav", x)
            wav = str(Path(save_wavs) / f"turn{i:03d}.wav") if save_wavs else None
            t = run_turn(src, asr, partial_asr, clf, lm, voice, player, cue_audio,
                         label=text, use_cue=use_cue,
                         parallel_emotion=parallel_emotion, save_wav=wav)
            t["turn"] = i
            turns.append(t)
            # confidence is None on the ablation arm, where no classifier ran
            c = t["emotion_confidence"]
            conf = " -- " if c is None else f"{c:.2f}"
            print(f"  turn {i:2d}  gap {t['gap_ms']:7.1f}  first {t['first_audio_ms']:6.1f}  "
                  f"emo {t['detected_emotion']:<8}{conf} "
                  f"({t['stage_ms']['emotion_ms']:5.1f}ms) -> {t['prosody_preset']:<8} "
                  f"{t['reply'][:38]!r}", flush=True)
    finally:
        player.close()

    keys = ["gap_ms", "first_audio_ms", "acoustic_gap_ms", "cue_gap_ms",
            "emotion_standalone_ms"] + [f"stage_ms.{k}" for k in turns[0]["stage_ms"]]
    summary = {}
    for k in keys:
        vals = [(t["stage_ms"][k.split(".", 1)[1]] if k.startswith("stage_ms.") else t[k])
                for t in turns]
        vals = [v for v in vals if v is not None]
        if vals:
            summary[k] = _stats(vals)

    counts: dict[str, int] = {}
    for t in turns:
        counts[t["detected_emotion"]] = counts.get(t["detected_emotion"], 0) + 1

    res = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_turns": len(turns),
        "input": "microphone (live)" if mic
                 else ("wav files: " + ", ".join(Path(p).name for p in prompt_wavs)
                       if prompt_wavs else "TTS-rendered prompt, paced at 1x"),
        "asr": {"final_model": asr.name, "partial_model": partial_asr.name,
                "mode": "chunked, whole-buffer re-decode",
                "partial_every_ms": PARTIAL_EVERY_MS},
        "emotion": {"ckpt": None if clf is None else str(clf.ckpt),
                    "device": None if clf is None else clf.device,
                    "classes": emo.EMOTIONS,
                    "schedule": "ABLATED -- no classifier ran, every turn spoken neutral"
                                if clf is None else
                                ("parallel with ASR final decode" if parallel_emotion
                                 else "serial, after the ASR final decode"),
                    "trained_on": "CREMA-D, acted speech, actor-disjoint split",
                    "reported_test_acc_vs_intent": 0.749,
                    "reported_test_acc_vs_consensus": 0.522},
        "lm": {"model": lm.name, "max_tokens": MAX_TOKENS, "stop": "first sentence",
               "conditioning": "detected emotion as one clause in the system prompt"},
        "tts": {"backend": voice.backend, "voice": voice.name, "mode": "whole utterance",
                "prosody": f"{voice.backend} preset chosen by detected emotion",
                "presets": prosody.PRESETS[voice.backend],
                "control_surface_ceiling":
                    "piper exposes no pitch knob; its four presets span 2 Hz of mean F0 "
                    "and are recovered from acoustics at 49.3% vs 25% chance"
                    if voice.backend == "piper" else
                    "say spans 246 Hz of mean F0 and its presets are recovered at 95.8%, "
                    "but pmod is inert and each utterance is a cold subprocess "
                    "(~2.6s). Source: expressive-tts-audit"},
        "cue": {"enabled": use_cue, "cue": CUE, "fire_at_ms": CUE_AT_MS,
                "n_fired": sum(1 for t in turns if t["cue"])},
        "prompt_tts": {"backend": prompt_voice.backend, "voice": prompt_voice.name},
        "output_device": player.device,
        "output_device_latency_ms": round(player.latency_ms, 2),
        "model_load_ms": round(load_ms, 1),
        "loadavg_start": [round(v, 2) for v in load0],
        "loadavg_end": [round(v, 2) for v in os.getloadavg()],
        "warmup_ms": round(warm_ms, 1),
        "warmup": "one ASR decode, one emotion forward pass, one LM generation and one "
                  "TTS synthesis run before turn 0, so no measured turn pays a cold graph",
        "hangover_ms": HANGOVER_MS,
        "gap_definition": "agent REPLY speech onset - user speech offset, both silence-"
                          "trimmed, measured with vendor.audio.segments (merge_gap_ms=30, "
                          "min_len_ms=20) -- the definition in aliveness-threshold "
                          "harness/exchange.py, so gap_ms is directly comparable to the "
                          "1452 ms median in that repo's live/STATUS.md. A cue played into "
                          "the gap is NOT counted as agent speech onset; it is reported "
                          "separately as cue_gap_ms / first_audio_ms.",
        "baseline_gap_ms_median": 1452.0,
        "baseline_source": "aliveness-threshold live/STATUS.md, n=40, same machine",
        "detected_emotion_counts": counts,
        "summary_ms": summary,
        "turns": turns,
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(res, indent=2))
        print(f"\nwrote {out_path}")
    return res


def selfcheck(n_turns: int = 3, **kw) -> dict:
    """Real turns, then assert the two things that would silently invalidate
    every number in the README.

    1. Every stage timer exists, is a number, is not negative, and the stages
       sum to the gap within one output block.
    2. **The classifier was fed this turn's audio.** A live loop that
       conditions turn N's reply on turn N-1's audio looks completely healthy
       in the logs and is worthless. The turn's captured array is fingerprinted
       at capture time and the classifier returns the fingerprint of what it
       actually saw; these must match, and must be distinct across turns.
    """
    res = run(n_turns=n_turns, **kw)
    assert res["turns"], "no turns ran"
    seen_shas = set()
    for t in res["turns"]:
        n = t["turn"]
        for k in ("gap_ms", "first_audio_ms", "acoustic_gap_ms"):
            v = t[k]
            assert isinstance(v, (int, float)), f"turn {n}: {k} missing"
            assert v > 0, f"turn {n}: {k} = {v}, must be positive"
        for k, v in t["stage_ms"].items():
            assert v is not None, f"turn {n}: stage {k} never timed"
            assert v >= 0, f"turn {n}: stage {k} = {v}ms, negative"
        assert t["transcript"], f"turn {n}: empty transcript"
        assert t["reply"], f"turn {n}: empty reply"

        # -- the emotion-provenance check --
        assert t["emotion_audio_sha"] == t["input_audio_sha"], (
            f"turn {n}: classifier was fed audio {t['emotion_audio_sha']} but this turn "
            f"captured {t['input_audio_sha']} -- emotion is off by a turn"
        )
        assert t["emotion_audio_sha"] not in seen_shas, (
            f"turn {n}: audio fingerprint {t['emotion_audio_sha']} already seen on an "
            f"earlier turn -- the capture buffer is being reused, so the classifier "
            f"cannot be reacting to this turn"
        )
        seen_shas.add(t["emotion_audio_sha"])
        assert t["detected_emotion"] in emo.EMOTIONS, f"turn {n}: bad label"
        assert t["prosody_preset"] == prosody.EMOTION_TO_PRESET[t["detected_emotion"]], (
            f"turn {n}: spoke with {t['prosody_preset']} for {t['detected_emotion']}")
        if res["emotion"]["schedule"].startswith("serial"):  # noqa: SIM102
            assert t["stage_ms"]["emotion_ms"] > 0, f"turn {n}: emotion stage not timed"

        if t["cue"]:
            assert t["cue_gap_ms"] < t["gap_ms"], (
                f"turn {n}: cue at {t['cue_gap_ms']:.0f}ms landed at or after the reply "
                f"at {t['gap_ms']:.0f}ms -- it filled nothing")
            assert not t["cue_preempted_by_reply"], (
                f"turn {n}: the reply cut the cue off mid-word")

        parts = sum(t["stage_ms"][k] for k in
                    ("endpoint_hangover_ms", "asr_final_dispatch_ms", "asr_final_ms",
                     "emotion_ms", "lm_ttft_ms", "lm_sentence_ms", "tts_ms",
                     "playback_dispatch_ms"))
        assert abs(parts - t["gap_ms"]) <= audio.millis(BLOCK) + 1.0, (
            f"turn {n}: stages sum to {parts:.1f}ms but gap is {t['gap_ms']:.1f}ms")

    print(f"\nselfcheck PASS -- {len(res['turns'])} turns; every stage timer present, "
          f"positive and summing to the gap; every emotion prediction provably made from "
          f"its own turn's audio")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["batch", "selfcheck", "devices"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out")
    ap.add_argument("--save-wavs")
    ap.add_argument("--device")
    ap.add_argument("--mic", action="store_true")
    ap.add_argument("--tts", default="auto", choices=["auto", "piper", "say"])
    ap.add_argument("--no-cue", action="store_true")
    ap.add_argument("--emotion-parallel", action="store_true",
                    help="run the classifier alongside the ASR final decode")
    ap.add_argument("--prompt-wav", action="append",
                    help="use these wavs as the user turns instead of TTS prompts")
    ap.add_argument("--prompt-wav-dir",
                    help="use every wav in this directory as the user turns")
    ap.add_argument("--no-emotion", action="store_true",
                    help="ablation: skip the classifier entirely, speak everything neutral")
    a = ap.parse_args()
    if a.cmd == "devices":
        import sounddevice as sd  # noqa: PLC0415

        print(sd.query_devices())
        return
    wavs = a.prompt_wav
    if a.prompt_wav_dir:
        wavs = sorted(str(p) for p in Path(a.prompt_wav_dir).glob("*.wav"))
        if not wavs:
            raise SystemExit(f"no wavs in {a.prompt_wav_dir}")
    kw = dict(device=a.device, mic=a.mic, tts_backend=a.tts, use_cue=not a.no_cue,
              parallel_emotion=a.emotion_parallel, prompt_wavs=wavs,
              save_wavs=a.save_wavs, no_emotion=a.no_emotion)
    if a.cmd == "selfcheck":
        selfcheck(n_turns=a.n if a.n < 20 else 3, **kw)
    else:
        run(n_turns=a.n, out_path=a.out, **kw)


if __name__ == "__main__":
    main()
