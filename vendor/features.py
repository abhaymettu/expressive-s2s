# VENDORED, MODIFIED, from ~/Desktop/Playground/expressive-tts-audit/features.py
# (github.com/abhaymettu/expressive-tts-audit). THE ONE MODIFICATION: its
# demo() self-check is dropped, because it imports that repo's render.py.
# extract() and FEATURES are byte-identical to upstream.
"""Objective acoustic features. No raters, no models, no judgment.

Every feature here is a standard prosodic correlate of expressive speech.
F0 and voice quality come from Praat (via parselmouth), which is the
reference implementation for both; energy, pausing and spectral shape come
from librosa/numpy.
"""

import warnings

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call

warnings.filterwarnings("ignore", category=UserWarning)

# Praat pitch floor/ceiling wide enough for both a 90 Hz `say -v Alex [[pbas 25]]`
# and a 350 Hz `[[pbas 80]]`. A too-narrow band silently clips the range feature.
F0_FLOOR, F0_CEIL = 60.0, 500.0

FEATURES = [
    "f0_mean", "f0_sd", "f0_range", "f0_slope", "f0_iqr", "voiced_frac",
    "rms_mean_db", "rms_sd_db", "rms_range_db",
    "duration", "speech_rate", "articulation_rate",
    "pause_count", "pause_total", "pause_frac",
    "spectral_tilt", "spectral_centroid",
    "jitter_local", "shimmer_local", "hnr",
]


def extract(path, text):
    """Return a dict of acoustic features for one rendered utterance.

    `text` is needed only for rate features (words per second); it is never
    used to infer anything about the expressive condition.
    """
    snd = parselmouth.Sound(str(path))
    dur = snd.duration

    # --- F0 -----------------------------------------------------------------
    pitch = snd.to_pitch(pitch_floor=F0_FLOOR, pitch_ceiling=F0_CEIL)
    f0 = pitch.selected_array["frequency"]
    t = pitch.xs()
    voiced = f0 > 0
    fv, tv = f0[voiced], t[voiced]
    if fv.size >= 2:
        f0_mean, f0_sd = float(np.mean(fv)), float(np.std(fv))
        f0_range = float(np.percentile(fv, 95) - np.percentile(fv, 5))
        f0_iqr = float(np.percentile(fv, 75) - np.percentile(fv, 25))
        # Global contour direction, Hz/sec, over the voiced frames.
        f0_slope = float(np.polyfit(tv, fv, 1)[0])
    else:
        f0_mean = f0_sd = f0_range = f0_iqr = f0_slope = np.nan
    voiced_frac = float(np.mean(voiced))

    # --- energy -------------------------------------------------------------
    y, sr = librosa.load(str(path), sr=None, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    rms_db = 20 * np.log10(np.maximum(rms, 1e-8))
    loud = rms_db > (rms_db.max() - 40)  # ignore the silent tail
    rms_mean_db = float(np.mean(rms_db[loud]))
    rms_sd_db = float(np.std(rms_db[loud]))
    rms_range_db = float(np.percentile(rms_db[loud], 95) - np.percentile(rms_db[loud], 5))

    # --- pausing ------------------------------------------------------------
    # ponytail: energy-threshold VAD, not a forced aligner. Good enough to
    # separate inter-word silence from speech at TTS-clean SNR; swap in an
    # aligner if pause *placement* ever matters, not just pause mass.
    intervals = librosa.effects.split(y, top_db=35, frame_length=1024, hop_length=256)
    speech_time = float(sum(e - s for s, e in intervals) / sr)
    gaps = [(intervals[i + 1][0] - intervals[i][1]) / sr for i in range(len(intervals) - 1)]
    pauses = [g for g in gaps if g >= 0.10]
    pause_count = len(pauses)
    pause_total = float(sum(pauses))
    pause_frac = pause_total / dur if dur > 0 else np.nan

    n_words = len(text.split())
    speech_rate = n_words / dur if dur > 0 else np.nan
    articulation_rate = n_words / speech_time if speech_time > 0 else np.nan

    # --- spectral shape -----------------------------------------------------
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    ltas = 20 * np.log10(np.maximum(S.mean(axis=1), 1e-8))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    band = (freqs >= 100) & (freqs <= 5000)
    # dB per kHz across 0.1-5 kHz: more negative = steeper roll-off = less
    # effortful / breathier voice. Rises toward 0 under vocal effort.
    spectral_tilt = float(np.polyfit(freqs[band] / 1000.0, ltas[band], 1)[0])
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # --- voice quality ------------------------------------------------------
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", F0_FLOOR, F0_CEIL)
        jitter = float(call(pp, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3))
        shimmer = float(call([snd, pp], "Get shimmer (local)", 0, 0, 1e-4, 0.02, 1.3, 1.6))
        hnr = float(call(snd.to_harmonicity_cc(), "Get mean", 0, 0))
    except Exception:
        jitter = shimmer = hnr = np.nan

    return dict(
        f0_mean=f0_mean, f0_sd=f0_sd, f0_range=f0_range, f0_slope=f0_slope,
        f0_iqr=f0_iqr, voiced_frac=voiced_frac,
        rms_mean_db=rms_mean_db, rms_sd_db=rms_sd_db, rms_range_db=rms_range_db,
        duration=dur, speech_rate=speech_rate, articulation_rate=articulation_rate,
        pause_count=pause_count, pause_total=pause_total, pause_frac=pause_frac,
        spectral_tilt=spectral_tilt, spectral_centroid=spectral_centroid,
        jitter_local=jitter, shimmer_local=shimmer, hnr=hnr,
    )
