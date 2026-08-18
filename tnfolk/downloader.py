"""Stage 3 -- Download Full Audio (yt-dlp + FFmpeg).

Downloads the FULL-length audio track (no truncation) for a candidate and
converts it to a mono WAV at the configured sample rate. Wrapped in tenacity
retry with exponential backoff; a permanent failure raises DownloadError which
the orchestrator maps to REJECT(DOWNLOAD_FAILED) without killing the batch.
"""
from __future__ import annotations

import os
from typing import Any

from .config import PipelineConfig
from .logging_utils import get_logger
from .models import Candidate
from .retry import make_retrying

_log = get_logger("download")


class DownloadError(RuntimeError):
    """Raised when a candidate's audio cannot be downloaded/decoded to WAV."""


def _ydl_opts(cfg: PipelineConfig, out_template: str) -> dict[str, Any]:
    dl = cfg.raw["download"]
    opts: dict[str, Any] = {
        "format": dl["format"],
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": False,      # we WANT exceptions so retry/reject can act
        "noplaylist": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["ios", "mweb", "android"]
            }
        },
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "0",
        }],
        # force mono + target sample rate on the ffmpeg postprocessor
        "postprocessor_args": [
            "-ac", str(int(dl["target_channels"])),
            "-ar", str(int(dl["target_sample_rate"])),
        ],
    }
    cookies = dl.get("cookies_from_browser")
    if cookies:
        opts["cookiesfrombrowser"] = (cookies,)
    ffmpeg_path = dl.get("ffmpeg_path")
    if ffmpeg_path:
        opts["ffmpeg_location"] = ffmpeg_path
    return opts


def _do_download(url: str, out_template: str, expected_wav: str,
                 cfg: PipelineConfig) -> str:
    import yt_dlp

    opts = _ydl_opts(cfg, out_template)
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    if not os.path.exists(expected_wav):
        # yt-dlp sometimes emits a differently-cased/ext file; search staging dir
        stem = os.path.splitext(os.path.basename(expected_wav))[0]
        folder = os.path.dirname(expected_wav)
        for name in os.listdir(folder):
            if name.startswith(stem) and name.lower().endswith(".wav"):
                return os.path.join(folder, name)
        raise DownloadError(f"WAV not produced for {url}")
    return expected_wav


def download_audio(candidate: Candidate, cfg: PipelineConfig, staging_dir: str) -> str:
    """Download + convert a candidate's full audio to a mono WAV.

    Args:
        candidate: the video to fetch (uses .youtube_url and .recording_id).
        cfg: pipeline config (reads the `download` + `retry` sections).
        staging_dir: local scratch directory for the downloaded WAV.

    Returns:
        Absolute path to the produced WAV file.

    Raises:
        DownloadError: if the audio cannot be downloaded/converted after retries.
    """
    os.makedirs(staging_dir, exist_ok=True)
    out_template = os.path.join(staging_dir, f"{candidate.recording_id}.%(ext)s")
    expected_wav = os.path.join(staging_dir, f"{candidate.recording_id}.wav")

    if os.path.exists(expected_wav):
        _log.info("reusing staged wav for %s", candidate.recording_id)
        return expected_wav

    retrying = make_retrying(cfg, description=f"download {candidate.recording_id}")
    try:
        path = retrying(_do_download, candidate.youtube_url, out_template,
                        expected_wav, cfg)
    except Exception as exc:  # noqa: BLE001 - normalized to DownloadError
        raise DownloadError(str(exc)) from exc
    _log.info("downloaded %s -> %s", candidate.recording_id, path)
    return os.path.abspath(path)
