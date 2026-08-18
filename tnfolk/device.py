"""Device selection and a lazy, load-once model manager (CPU-only build).

Design: device is forced to "cpu" (Intel iGPU, no CUDA). Every ML model is
instantiated at most once and reused across files. GPU-heavy models are expected
to run behind a single inference worker (workers.inference / workers.transcription)
so PANNs, sentence-transformers, CLAP and faster-whisper never contend for CPU
threads simultaneously.

Heavy imports (torch, panns, sentence-transformers, laion_clap, faster-whisper)
are performed lazily inside each accessor so lightweight gates/tests remain
importable without the full ML stack installed.
"""
from __future__ import annotations

import threading
from typing import Any, Optional

from .logging_utils import get_logger

_log = get_logger("device")


def resolve_device(configured: str) -> str:
    """Resolve the configured device string to a concrete 'cpu'/'cuda'.

    "cpu"  -> always cpu (this build).
    "auto" -> cuda if torch reports a CUDA GPU, else cpu.
    "cuda" -> cuda if available, else cpu (with a warning).
    """
    configured = (configured or "cpu").lower()
    if configured == "cpu":
        return "cpu"
    try:
        import torch  # type: ignore
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if configured in ("auto", "cuda"):
        if has_cuda:
            return "cuda"
        if configured == "cuda":
            _log.warning("device=cuda requested but no CUDA GPU found; using cpu")
        return "cpu"
    return "cpu"


class ModelManager:
    """Owns every ML model, instantiating each at most once (thread-safe)."""

    def __init__(self, pipeline_config: Any, device: Optional[str] = None) -> None:
        self.cfg = pipeline_config
        self.device = device or resolve_device(pipeline_config.raw.get("device", "cpu"))
        self._lock = threading.Lock()
        self._models: dict[str, Any] = {}
        _log.info("ModelManager initialised on device=%s", self.device)

    def _get_or_create(self, key: str, factory) -> Any:
        model = self._models.get(key)
        if model is not None:
            return model
        with self._lock:
            model = self._models.get(key)
            if model is None:
                _log.info("Loading model: %s", key)
                model = factory()
                self._models[key] = model
            return model

    # -- classifier (PANNs Cnn14) ------------------------------------------
    def panns_tagger(self) -> Any:
        """Cached PANNs AudioTagging (Cnn14) on CPU."""
        def factory() -> Any:
            from panns_inference import AudioTagging  # type: ignore
            return AudioTagging(checkpoint_path=None, device=self.device)
        return self._get_or_create("panns", factory)

    # -- VAD (silero via torch.hub) ----------------------------------------
    def silero_vad(self) -> Any:
        """Return (model, utils) for silero-vad, cached."""
        def factory() -> Any:
            import torch  # type: ignore
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            model.to(self.device)
            return (model, utils)
        return self._get_or_create("silero_vad", factory)

    # -- folk relevance (sentence-transformers) ----------------------------
    def sentence_encoder(self) -> Any:
        def factory() -> Any:
            from sentence_transformers import SentenceTransformer  # type: ignore
            model_id = self.cfg.raw["folk_relevance"]["embedding_model"]
            return SentenceTransformer(model_id, device=self.device)
        return self._get_or_create("sentence_encoder", factory)

    # -- folk relevance secondary signal (LAION-CLAP) ----------------------
    def clap(self) -> Any:
        """Cached LAION-CLAP model (audio<->text). Loaded lazily; only used in the
        uncertain folk-relevance band to keep CPU load manageable."""
        def factory() -> Any:
            import laion_clap  # type: ignore
            name = self.cfg.raw["folk_relevance"]["clap"].get("model_name")
            model = laion_clap.CLAP_Module(enable_fusion=False)
            model.load_ckpt(name) if name else model.load_ckpt()
            return model
        return self._get_or_create("clap", factory)

    def whisper(self, model_dir: str) -> Any:
        def factory() -> Any:
            from faster_whisper import WhisperModel  # type: ignore
            compute_type = self.cfg.raw["transcription"]["compute_type"]
            return WhisperModel(model_dir, device=self.device, compute_type=compute_type)
        return self._get_or_create(f"whisper::{model_dir}", factory)
