"""Audio Chunker module for splitting long audio files into 2-5 minute sub-tracks.

When a downloaded video is longer than 5 minutes (e.g. 15-minute compilation or 45-minute album),
this module splits the audio file into individual sub-tracks at natural silence/pause boundaries.
Each sub-track is then evaluated, transcribed, and uploaded to GCS as an independent candidate.
"""
from __future__ import annotations

import os
from typing import List
# pyrefly: ignore [missing-import]
import numpy as np

from .config import PipelineConfig
from .logging_utils import get_logger
from .models import Candidate

_log = get_logger("chunker")


def split_audio_if_needed(candidate: Candidate, cfg: PipelineConfig) -> List[Candidate]:
    """Check candidate audio duration. If > chunk_threshold_sec, split into sub-candidates.
    
    Returns a list of Candidates. If no splitting is needed, returns [candidate].
    """
    path = candidate.audio_path
    if not path or not os.path.exists(path):
        return [candidate]

    chk_cfg = cfg.raw.get("chunking", {})
    if not chk_cfg.get("enabled", True):
        return [candidate]

    chunk_threshold = float(chk_cfg.get("chunk_threshold_sec", 300))
    min_chunk_dur = float(chk_cfg.get("min_chunk_duration_sec", 30))
    max_chunk_dur = float(chk_cfg.get("max_chunk_duration_sec", 180))
    target_dur = float(chk_cfg.get("target_chunk_duration_sec", 120))
    silence_db_offset = float(chk_cfg.get("silence_thresh_db", -35))
    max_chunks = int(chk_cfg.get("max_chunks_per_video", 5))

    # pyrefly: ignore [missing-import]
    import soundfile as sf
    # pyrefly: ignore [missing-import]
    import librosa

    try:
        y, sr = sf.read(path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        duration = float(len(y)) / float(sr) if sr else 0.0
    except Exception as exc:
        _log.warning("Failed to inspect audio %s for chunking: %r", candidate.recording_id, exc)
        return [candidate]

    max_allowed = float(cfg.raw.get("audio", {}).get("maximum_duration_sec", 900))
    if duration > max_allowed:
        _log.info("Audio %s duration %.1fs exceeds maximum allowed %.1fs (will be rejected as TOO_LONG)",
                  candidate.recording_id, duration, max_allowed)
        return [candidate]

    if duration <= chunk_threshold:
        return [candidate]

    _log.info("Audio %s is %.1fs (> %ds limit). Chunking into sub-tracks...", 
              candidate.recording_id, duration, int(chunk_threshold))

    # Find split points using RMS energy and silence threshold
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    max_rms = float(np.max(rms)) if len(rms) > 0 else 1e-6

    total_samples = len(y)
    target_samples = int(target_dur * sr)
    max_samples = int(max_chunk_dur * sr)
    min_samples = int(min_chunk_dur * sr)

    split_points = [0]
    current_idx = 0

    while current_idx + target_samples < total_samples:
        search_start = current_idx + min_samples
        search_end = min(current_idx + max_samples, total_samples)
        
        if search_start >= total_samples or search_end <= search_start:
            break

        # Search for quietest frame in the search window (near target_dur)
        window_start_frame = int(search_start / hop_length)
        window_end_frame = int(search_end / hop_length)
        
        if window_end_frame <= window_start_frame:
            best_sample = current_idx + target_samples
        else:
            sub_rms = rms[window_start_frame:window_end_frame]
            min_rms_frame_rel = int(np.argmin(sub_rms))
            best_frame = window_start_frame + min_rms_frame_rel
            best_sample = best_frame * hop_length

        split_points.append(best_sample)
        current_idx = best_sample

    if split_points[-1] < total_samples:
        split_points.append(total_samples)

    # Clean up split points: merge very small tail chunks into previous if under min_samples
    final_splits = [split_points[0]]
    for pt in split_points[1:]:
        if pt - final_splits[-1] < min_samples and len(final_splits) > 1:
            final_splits[-1] = pt
        else:
            final_splits.append(pt)

    if len(final_splits) <= 2 and (final_splits[-1] - final_splits[0]) <= max_samples:
        return [candidate]

    # Save chunks as new WAV files
    dir_name = os.path.dirname(path)
    sub_candidates: List[Candidate] = []

    for i in range(len(final_splits) - 1):
        start_samp = final_splits[i]
        end_samp = final_splits[i + 1]
        chunk_data = y[start_samp:end_samp]
        chunk_dur = len(chunk_data) / float(sr)

        part_num = i + 1
        part_id = f"{candidate.recording_id}_p{part_num:02d}"
        part_path = os.path.join(dir_name, f"{part_id}.wav")

        sf.write(part_path, chunk_data, sr, subtype="PCM_16")

        sub_c = Candidate(
            recording_id=part_id,
            category_slug=candidate.category_slug,
            url=candidate.url,
            title=f"{candidate.title} (Part {part_num})",
            channel=candidate.channel,
            duration_sec=round(chunk_dur, 2),
            description=candidate.description,
            audio_path=part_path,
        )
        sub_candidates.append(sub_c)

    _log.info("Successfully split %s into %d sub-tracks: %s",
              candidate.recording_id, len(sub_candidates),
              [c.recording_id for c in sub_candidates])

    # Cap the number of sub-tracks to avoid pipeline explosion from jukebox/compilation videos
    if len(sub_candidates) > max_chunks:
        _log.warning(
            "Video %s produced %d chunks; capping at %d (max_chunks_per_video). "
            "Remaining %d chunks will be skipped.",
            candidate.recording_id, len(sub_candidates), max_chunks,
            len(sub_candidates) - max_chunks
        )
        sub_candidates = sub_candidates[:max_chunks]

    return sub_candidates
