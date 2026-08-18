"""Stage 5 -- Strict Audio Quality Gate (highest priority, pure DSP, NO model).

Computes technical audio metrics directly from the waveform and hard-rejects on
threshold breaches from config. The composite ``quality_score`` (0..1) is also
returned as the dominant signal for final scoring, where a value below
quality_gate.minimum_quality_score forces REJECT regardless of folk relevance
(the explicit hard override lives in final_scoring.py).

Libraries: numpy, scipy (via librosa), librosa, soundfile, pyloudnorm.

INPUTS
    candidate : Candidate (uses .audio_path)
    cfg       : PipelineConfig (reads `audio` + `quality_gate`)
OUTPUT
    GateResult(stage=QUALITY_GATE) with score=quality_score and a full metrics dict.
REJECTION CODES
    TOO_SHORT, MOSTLY_SILENT, EXCESSIVE_CLIPPING, SEVERE_DISTORTION, VERY_LOW_SNR,
    EXCESSIVE_NOISE, AUDIO_DROPOUT, LOW_BANDWIDTH, CORRUPT_AUDIO
"""
from __future__ import annotations

import math
from typing import Any

from ..config import PipelineConfig
from ..logging_utils import get_logger
from ..models import Candidate, GateResult, RejectionCode, Stage

_log = get_logger("gate.quality")

_FRAME = 2048
_HOP = 512
_EPS = 1e-10


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _db(x: float) -> float:
    return 20.0 * math.log10(max(x, _EPS))


def _compute_metrics(path: str, cfg: PipelineConfig) -> dict[str, Any]:
    import numpy as np
    import librosa
    import soundfile as sf

    info = sf.info(path)
    native_sr = int(info.samplerate)
    channels = int(info.channels)
    subtype = str(info.subtype)

    y, sr = librosa.load(path, sr=None, mono=True)
    y = np.asarray(y, dtype=np.float32)
    duration = float(len(y)) / float(sr) if sr else 0.0

    if len(y) == 0 or sr == 0:
        raise ValueError("empty waveform")

    peak = float(np.max(np.abs(y)))
    rms_global = float(np.sqrt(np.mean(np.square(y))))

    # frame-wise RMS (dBFS)
    rms = librosa.feature.rms(y=y, frame_length=_FRAME, hop_length=_HOP)[0]
    rms = np.asarray(rms, dtype=np.float64)
    rms_db = 20.0 * np.log10(np.maximum(rms, _EPS))

    qg = cfg.raw["quality_gate"]
    near_sil_db = float(qg["near_silence_dbfs"])
    clip_thr = float(qg["clipping_sample_threshold"])

    near_silence_ratio = float(np.mean(rms_db < near_sil_db))
    clipping_ratio = float(np.mean(np.abs(y) >= clip_thr))

    noise_rms = float(np.percentile(rms, 5))
    p95_rms = float(np.percentile(rms, 95))
    noise_floor_dbfs = _db(noise_rms)
    snr_db = _db(rms_global) - _db(noise_rms)
    dynamic_range_db = _db(p95_rms) - _db(noise_rms)

    # dropout detection: short near-silent frames embedded in otherwise loud audio
    loud_ref = float(np.percentile(rms_db, 90))
    silent_mask = rms_db < (near_sil_db - 5.0)
    embedded = 0
    n = len(rms_db)
    win = 6
    for i in range(n):
        if not silent_mask[i]:
            continue
        lo = max(0, i - win)
        hi = min(n, i + win + 1)
        neighbourhood = np.concatenate([rms_db[lo:i], rms_db[i + 1:hi]])
        if neighbourhood.size and float(np.median(neighbourhood)) > (loud_ref - 20.0):
            embedded += 1
    dropout_ratio = float(embedded) / float(n) if n else 0.0

    # frequency bandwidth (telephone-audio detection)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)[0]
    median_rolloff_hz = float(np.median(rolloff)) if len(rolloff) else 0.0

    # LUFS integrated loudness
    lufs = None
    try:
        import pyloudnorm as pyln
        if duration >= 0.5:
            meter = pyln.Meter(sr)
            lufs = float(meter.integrated_loudness(y.astype(np.float64)))
            if not math.isfinite(lufs):
                lufs = None
    except Exception as exc:  # noqa: BLE001 - loudness is a soft signal
        _log.debug("LUFS computation skipped: %s", exc)

    return {
        "duration_sec": round(duration, 2),
        "native_sample_rate": native_sr,
        "channels": channels,
        "bit_depth": subtype,
        "peak_amplitude": round(peak, 4),
        "rms_energy": round(rms_global, 6),
        "near_silence_ratio": round(near_silence_ratio, 4),
        "clipping_ratio": round(clipping_ratio, 5),
        "snr_db": round(snr_db, 2),
        "noise_floor_dbfs": round(noise_floor_dbfs, 2),
        "dynamic_range_db": round(dynamic_range_db, 2),
        "dropout_ratio": round(dropout_ratio, 4),
        "median_rolloff_hz": round(median_rolloff_hz, 1),
        "lufs": None if lufs is None else round(lufs, 2),
    }


def _composite_score(m: dict[str, Any], cfg: PipelineConfig) -> float:
    qg = cfg.raw["quality_gate"]
    max_sil = float(qg["maximum_silence_ratio"])
    max_clip = float(qg["maximum_clipping_ratio"])
    min_snr = float(qg["minimum_snr_db"])
    max_noise = float(qg["maximum_noise_floor_dbfs"])
    min_dr = float(qg["minimum_dynamic_range_db"])
    max_drop = float(qg["dropout"]["max_dropout_ratio"])
    min_roll = float(qg["bandwidth"]["min_rolloff_hz"])
    min_lufs = float(qg["loudness"]["min_lufs"])
    max_lufs = float(qg["loudness"]["max_lufs"])

    silence_s = _clamp01(1.0 - m["near_silence_ratio"] / max(max_sil, _EPS))
    clip_s = _clamp01(1.0 - m["clipping_ratio"] / max(max_clip, _EPS))
    snr_s = _clamp01((m["snr_db"] - min_snr) / max(30.0 - min_snr, _EPS))
    noise_s = _clamp01((max_noise - m["noise_floor_dbfs"]) / 30.0 + 0.0)  # lower floor better
    dr_s = _clamp01((m["dynamic_range_db"] - min_dr) / max(40.0 - min_dr, _EPS))
    drop_s = _clamp01(1.0 - m["dropout_ratio"] / max(max_drop, _EPS))
    band_s = _clamp01(m["median_rolloff_hz"] / 8000.0)
    if m["lufs"] is None:
        loud_s = 0.7
    elif min_lufs <= m["lufs"] <= max_lufs:
        loud_s = 1.0
    else:
        loud_s = 0.4

    weights = {
        "silence": 0.18, "clip": 0.14, "snr": 0.18, "noise": 0.12,
        "dr": 0.12, "dropout": 0.10, "band": 0.10, "loud": 0.06,
    }
    score = (
        weights["silence"] * silence_s + weights["clip"] * clip_s
        + weights["snr"] * snr_s + weights["noise"] * noise_s
        + weights["dr"] * dr_s + weights["dropout"] * drop_s
        + weights["band"] * band_s + weights["loud"] * loud_s
    )
    return round(_clamp01(score), 4)


def evaluate(candidate: Candidate, cfg: PipelineConfig) -> GateResult:
    """Run the strict audio quality gate. Pure DSP, no ML.

    Returns a passing GateResult carrying score=quality_score and a metrics dict,
    or GateResult.reject(<code>, ...) on the first hard-threshold breach.
    """
    path = candidate.audio_path
    if not path:
        return GateResult.reject(Stage.QUALITY_GATE, RejectionCode.CORRUPT_AUDIO,
                                 message="no audio path")
    try:
        m = _compute_metrics(path, cfg)
    except Exception as exc:  # noqa: BLE001
        return GateResult.reject(Stage.QUALITY_GATE, RejectionCode.CORRUPT_AUDIO,
                                 message=f"quality analysis failed: {exc}")

    a = cfg.raw["audio"]
    qg = cfg.raw["quality_gate"]

    # --- hard-threshold rejects (ordered) ---
    if m["duration_sec"] < float(a["minimum_duration_sec"]):
        return GateResult.reject(Stage.QUALITY_GATE, RejectionCode.TOO_SHORT,
                                 message=f"duration {m['duration_sec']}s < {a['minimum_duration_sec']}s", **m)
    if m["duration_sec"] > float(a["maximum_duration_sec"]):
        return GateResult.reject(Stage.QUALITY_GATE, RejectionCode.TOO_LONG,
                                 message=f"duration {m['duration_sec']}s > {a['maximum_duration_sec']}s", **m)
    if m["near_silence_ratio"] > float(qg["maximum_silence_ratio"]):
        return GateResult.reject(Stage.QUALITY_GATE, RejectionCode.MOSTLY_SILENT,
                                 message=f"silence ratio {m['near_silence_ratio']:.2f}", **m)
    if m["clipping_ratio"] > float(qg["maximum_clipping_ratio"]):
        return GateResult.reject(Stage.QUALITY_GATE, RejectionCode.EXCESSIVE_CLIPPING,
                                 message=f"clipping ratio {m['clipping_ratio']:.3f}", **m)
    if m["dynamic_range_db"] < float(qg["minimum_dynamic_range_db"]):
        return GateResult.reject(Stage.QUALITY_GATE, RejectionCode.SEVERE_DISTORTION,
                                 message=f"dynamic range {m['dynamic_range_db']:.1f}dB", **m)
    if m["snr_db"] < float(qg["minimum_snr_db"]):
        return GateResult.reject(Stage.QUALITY_GATE, RejectionCode.VERY_LOW_SNR,
                                 message=f"SNR {m['snr_db']:.1f}dB", **m)
    if m["noise_floor_dbfs"] > float(qg["maximum_noise_floor_dbfs"]):
        return GateResult.reject(Stage.QUALITY_GATE, RejectionCode.EXCESSIVE_NOISE,
                                 message=f"noise floor {m['noise_floor_dbfs']:.1f}dBFS", **m)
    if m["dropout_ratio"] > float(qg["dropout"]["max_dropout_ratio"]):
        return GateResult.reject(Stage.QUALITY_GATE, RejectionCode.AUDIO_DROPOUT,
                                 message=f"dropout ratio {m['dropout_ratio']:.3f}", **m)
    if (m["median_rolloff_hz"] < float(qg["bandwidth"]["min_rolloff_hz"])
            or m["native_sample_rate"] < int(qg["min_sample_rate"])):
        return GateResult.reject(Stage.QUALITY_GATE, RejectionCode.LOW_BANDWIDTH,
                                 message=f"rolloff {m['median_rolloff_hz']:.0f}Hz / "
                                         f"sr {m['native_sample_rate']}Hz", **m)

    score = _composite_score(m, cfg)
    m["quality_score"] = score
    # Note: the score < minimum_quality_score hard override is applied explicitly
    # in final_scoring.py (not here) so quality can still contribute its metrics.
    review = score < float(qg["minimum_quality_score"])
    msg = f"quality_score={score} (dur={m['duration_sec']}s snr={m['snr_db']}dB)"
    if review:
        return GateResult.flag_review(Stage.QUALITY_GATE, score=score,
                                      message=msg + " [below min -> hard override pending]", **m)
    return GateResult.ok(Stage.QUALITY_GATE, score=score, message=msg, **m)
