"""Stage 7 -- Voice Activity Analysis (Silero VAD, CPU-native).

Measures how much continuous voice activity is present and returns it as a ratio
in [0, 1]. This is a SUPPORTING SIGNAL ONLY -- it is wired into final scoring as
one weighted input and MUST NEVER be the sole reason a candidate is rejected.

INPUTS
    candidate    : Candidate (uses .audio_path)
    model_manager: ModelManager (provides cached silero-vad model+utils)
    cfg          : PipelineConfig (reads `vad`)
OUTPUT
    GateResult(stage=VAD) with score=voice_activity_ratio (never rejects).
"""
from __future__ import annotations

from typing import Any

from ..config import PipelineConfig
from ..logging_utils import get_logger
from ..models import Candidate, GateResult, Stage

_log = get_logger("gate.vad")


def evaluate(candidate: Candidate, model_manager: Any, cfg: PipelineConfig) -> GateResult:
    """Compute the Silero voice-activity ratio for the candidate's audio.

    Returns a passing GateResult with score = (speech seconds / total seconds).
    On any failure it returns a passing result with score=None (VAD is only a
    supporting signal and never rejects).
    """
    import librosa
    import numpy as np
    import torch

    v = cfg.raw["vad"]
    target_sr = int(v.get("sampling_rate", 16000))
    threshold = float(v["threshold"])

    try:
        y, sr = librosa.load(candidate.audio_path, sr=target_sr, mono=True)
        if y.size == 0:
            raise ValueError("empty audio")
        model, utils = model_manager.silero_vad()
        get_speech_timestamps = utils[0] if isinstance(utils, (list, tuple)) else utils["get_speech_timestamps"]
        wav = torch.from_numpy(np.asarray(y, dtype=np.float32))
        speech = get_speech_timestamps(wav, model, threshold=threshold,
                                       sampling_rate=target_sr)
        total_samples = wav.shape[0]
        speech_samples = sum(seg["end"] - seg["start"] for seg in speech)
        ratio = float(speech_samples) / float(total_samples) if total_samples else 0.0
    except Exception as exc:  # noqa: BLE001 - supporting signal only
        _log.warning("VAD failed for %s: %r", candidate.recording_id, exc)
        return GateResult.ok(Stage.VAD, score=None,
                             message=f"vad error: {exc}", voice_activity_ratio=None)

    ratio = max(0.0, min(1.0, ratio))
    return GateResult.ok(Stage.VAD, score=round(ratio, 4),
                         message=f"voice_activity_ratio={ratio:.2f}",
                         voice_activity_ratio=round(ratio, 4))
