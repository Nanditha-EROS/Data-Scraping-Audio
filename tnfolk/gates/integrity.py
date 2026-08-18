"""Stage 4 -- Integrity Gate (ffprobe + soundfile).

Confirms the downloaded file decodes correctly, isn't truncated, and has a valid
audio stream BEFORE the expensive quality/ML stages run.

INPUTS
    candidate : Candidate (uses .audio_path)
    cfg       : PipelineConfig (reads download.ffprobe_path)
OUTPUT
    GateResult(stage=INTEGRITY_GATE); passed=True to proceed.
REJECTION CODES
    CORRUPT_AUDIO -- file missing/undecodable, no audio stream, or truncated
                     (ffprobe vs soundfile duration mismatch).
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Optional

from ..config import PipelineConfig
from ..logging_utils import get_logger
from ..models import Candidate, GateResult, RejectionCode, Stage

_log = get_logger("gate.integrity")


def _ffprobe_streams(path: str, ffprobe: str) -> dict[str, Any]:
    cmd = [
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()[:200]}")
    return json.loads(proc.stdout)


def evaluate(candidate: Candidate, cfg: PipelineConfig) -> GateResult:
    """Validate that the candidate's WAV is a complete, decodable audio file.

    Returns a passing GateResult with basic stream details, or
    GateResult.reject(CORRUPT_AUDIO, ...) on any integrity failure.
    """
    path = candidate.audio_path
    ffprobe = cfg.raw["download"].get("ffprobe_path") or "ffprobe"

    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return GateResult.reject(Stage.INTEGRITY_GATE, RejectionCode.CORRUPT_AUDIO,
                                 message="file missing or empty", path=path)

    # 1) soundfile: must open + report a valid PCM stream
    try:
        import soundfile as sf
        info = sf.info(path)
        sf_duration = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
        # decode a small chunk to confirm frames are actually readable
        with sf.SoundFile(path) as fh:
            fh.read(min(info.frames, info.samplerate or 16000), dtype="float32")
    except Exception as exc:  # noqa: BLE001
        return GateResult.reject(Stage.INTEGRITY_GATE, RejectionCode.CORRUPT_AUDIO,
                                 message=f"soundfile decode failed: {exc}")

    if info.frames <= 0 or info.samplerate <= 0:
        return GateResult.reject(Stage.INTEGRITY_GATE, RejectionCode.CORRUPT_AUDIO,
                                 message="no audio frames")

    # 2) ffprobe: must see an audio stream; compare durations for truncation
    ffprobe_duration: Optional[float] = None
    try:
        probe = _ffprobe_streams(path, ffprobe)
        audio_streams = [s for s in probe.get("streams", [])
                         if s.get("codec_type") == "audio"]
        if not audio_streams:
            return GateResult.reject(Stage.INTEGRITY_GATE, RejectionCode.CORRUPT_AUDIO,
                                     message="no audio stream in container")
        fmt_dur = probe.get("format", {}).get("duration")
        if fmt_dur is not None:
            ffprobe_duration = float(fmt_dur)
    except Exception as exc:  # noqa: BLE001 - ffprobe optional if soundfile ok
        _log.warning("ffprobe unavailable/failed for %s (%s); relying on soundfile",
                     candidate.recording_id, exc)

    # truncation check: significant mismatch between container and PCM durations
    if ffprobe_duration is not None and sf_duration > 0:
        diff = abs(ffprobe_duration - sf_duration)
        if diff > max(1.0, 0.15 * ffprobe_duration):
            return GateResult.reject(
                Stage.INTEGRITY_GATE, RejectionCode.CORRUPT_AUDIO,
                message=f"duration mismatch (container={ffprobe_duration:.1f}s, "
                        f"pcm={sf_duration:.1f}s) -> likely truncated",
                ffprobe_duration=ffprobe_duration, sf_duration=sf_duration,
            )

    return GateResult.ok(
        Stage.INTEGRITY_GATE,
        message=f"valid audio ({sf_duration:.1f}s, {info.samplerate}Hz, {info.channels}ch)",
        duration_sec=round(sf_duration, 2),
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        subtype=str(info.subtype),
    )
