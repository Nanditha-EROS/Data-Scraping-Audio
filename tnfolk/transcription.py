"""Stage 11 -- Full-Audio Transcription (ACCEPT only).

Supports two backends (configured via transcription.backend in pipeline_config):

  "api"   (default) -- POSTs the audio file to a remote GPU server running
          faster-whisper. The teammate's RTX 6000 endpoint handles inference.
          This is the primary path for local-machine pipeline runs.

  "local" -- Uses the local faster-whisper CTranslate2 model (int8 CPU).
          Activate by setting backend: "local" in pipeline_config.yaml and
          uncommenting the local model keys.

Transcribes the COMPLETE accepted audio track (not sampled windows). This step
does NOT affect the ACCEPT/REVIEW/REJECT decision (already final). A transcription
failure must NEVER undo the ACCEPT or delete the saved audio -- the caller keeps
the ACCEPT and retries transcription separately.
"""
from __future__ import annotations

import os
from typing import Any

from .config import PipelineConfig
from .logging_utils import get_logger
from .retry import make_retrying

_log = get_logger("transcription")


# =========================================================================
#  Remote API backend
# =========================================================================

def _normalize_api_url(url: str) -> str:
    """Ensure the endpoint path (/api/transcribe) is present."""
    u = (url or "").strip()
    if not u:
        return ""
    if not (u.endswith("/api/transcribe") or u.endswith("/transcribe")):
        u = u.rstrip("/") + "/api/transcribe"
    return u


def _post_transcribe(audio_path: str, url: str, task: str, language: str, timeout: int) -> str:
    import requests

    _log.info("API transcribe %s -> %s", os.path.basename(audio_path), url)

    with open(audio_path, "rb") as f:
        resp = requests.post(
            url,
            files={"file": (os.path.basename(audio_path), f, "audio/wav")},
            data={"task": task, "language": language},
            timeout=timeout,
        )

    # Log the response body BEFORE raising so a 400/422 shows exactly why the
    # server rejected the request (bad field name, unsupported format, etc.)
    if not resp.ok:
        _log.error(
            "API transcribe failed (%s) for %s at %s -- server said: %s",
            resp.status_code, os.path.basename(audio_path), url, resp.text[:1000],
        )
    resp.raise_for_status()
    payload = resp.json()

    # The API may return the transcript under "text", "transcript", or "result"
    text = (
        payload.get("text")
        or payload.get("transcript")
        or payload.get("result")
        or ""
    )
    if isinstance(text, list):
        # some APIs return a list of segment dicts or strings
        parts = [
            seg["text"] if isinstance(seg, dict) else str(seg)
            for seg in text
        ]
        text = "".join(parts)

    text = text.strip()
    _log.info("API transcription done for %s (%d chars)",
              os.path.basename(audio_path), len(text))
    return text


def _transcribe_api(audio_path: str, cfg: PipelineConfig) -> str:
    """POST the audio file to the remote faster-whisper GPU API (with fallback if configured).

    Endpoint:  POST /api/transcribe  (multipart form-data)
    Fields:    file (binary), task ("transcribe"), language ("ta")
    Returns:   The transcript text from the JSON response.
    """
    t = cfg.raw["transcription"]
    primary_url = _normalize_api_url(str(t.get("api_url", "")))
    fallback_url = _normalize_api_url(str(t.get("fallback_api_url", "")))
    timeout = int(t.get("api_timeout_sec", 180))
    language = t.get("language", "ta")
    task = t.get("api_task", "transcribe")

    candidate_urls = [u for u in [primary_url, fallback_url] if u]
    if not candidate_urls:
        raise ValueError("No API URL configured for transcription backend 'api'")

    last_exc: Exception | None = None
    for idx, url in enumerate(candidate_urls):
        try:
            return _post_transcribe(audio_path, url, task, language, timeout)
        except Exception as exc:
            last_exc = exc
            if idx < len(candidate_urls) - 1:
                _log.warning(
                    "Primary transcription URL failed (%s: %s). Trying fallback URL: %s",
                    type(exc).__name__, exc, candidate_urls[idx + 1]
                )
    if last_exc:
        raise last_exc
    return ""


# =========================================================================
#  Local faster-whisper backend (preserved for fallback / offline use)
# =========================================================================

def _ct2_ready(ct2_dir: str) -> bool:
    return os.path.isdir(ct2_dir) and os.path.exists(os.path.join(ct2_dir, "model.bin"))


def _ensure_ct2_model(cfg: PipelineConfig) -> str:
    """Ensure a CTranslate2 (int8) copy of the Tamil Whisper model exists."""
    t = cfg.raw["transcription"]
    ct2_dir = cfg.abspath(t["ct2_model_dir"])
    if _ct2_ready(ct2_dir):
        return ct2_dir

    hf_id = t["hf_model_id"]
    os.makedirs(os.path.dirname(ct2_dir) or ".", exist_ok=True)
    _log.info("Converting %s -> CTranslate2 (int8) at %s", hf_id, ct2_dir)

    def _convert() -> None:
        try:
            import ctranslate2  # type: ignore # noqa: F401
            from ctranslate2.converters import TransformersConverter  # type: ignore
            converter = TransformersConverter(
                hf_id, copy_files=["tokenizer.json", "preprocessor_config.json"]
            )
            converter.convert(ct2_dir, quantization="int8", force=True)
        except ImportError:
            raise RuntimeError("ctranslate2 is not installed. Install ctranslate2 to use backend: 'local'.")

    retrying = make_retrying(cfg, description="whisper ct2 convert")
    try:
        retrying(_convert)
    except Exception:
        _log.warning("CT2 conversion with copy_files failed; retrying minimal convert")
        try:
            from ctranslate2.converters import TransformersConverter  # type: ignore
            TransformersConverter(hf_id).convert(ct2_dir, quantization="int8", force=True)
        except ImportError:
            raise RuntimeError("ctranslate2 is not installed. Install ctranslate2 to use backend: 'local'.")

    if not _ct2_ready(ct2_dir):
        raise RuntimeError(f"CT2 conversion did not produce a model at {ct2_dir}")
    return ct2_dir


def _transcribe_local(audio_path: str, model_manager: Any, cfg: PipelineConfig) -> str:
    """Transcribe using the local faster-whisper CTranslate2 model."""
    t = cfg.raw["transcription"]
    ct2_dir = _ensure_ct2_model(cfg)
    model = model_manager.whisper(ct2_dir)
    segments, info = model.transcribe(
        audio_path,
        language=t["language"],
        beam_size=int(t["beam_size"]),
        vad_filter=bool(t.get("vad_filter", True)),
    )
    parts = [seg.text for seg in segments]
    text = "".join(parts).strip()
    _log.info("local transcribed %s (%d segments, %d chars)",
              os.path.basename(audio_path), len(parts), len(text))
    return text


# =========================================================================
#  Public entry point (called by the orchestrator)
# =========================================================================

def transcribe_full(audio_path: str, model_manager: Any, cfg: PipelineConfig) -> str:
    """Transcribe the full audio track to Tamil text.

    Routes to the configured backend ("api" or "local"). Wrapped in tenacity
    retry so transient network / GPU errors back off automatically.

    Args:
        audio_path: path to the accepted WAV (full track).
        model_manager: ModelManager (only used by the local backend).
        cfg: pipeline config (reads ``transcription`` section).

    Returns:
        The concatenated transcript text (may be empty for instrumental audio).

    Raises:
        Exception: propagated to the caller, which keeps the ACCEPT and retries
        transcription separately (never deletes the saved audio).
    """
    backend = cfg.raw["transcription"].get("backend", "api")
    retrying = make_retrying(cfg, description="transcription")

    if backend == "api":
        return retrying(_transcribe_api, audio_path, cfg)
    else:
        return retrying(_transcribe_local, audio_path, model_manager, cfg)
