"""Stage 6 -- Music / Speech / Mixed Classifier (PANNs Cnn14, CPU).

Why PANNs (Cnn14) over BEATs: Cnn14 is a compact CNN with fast CPU inference at
scale, ships pretrained on AudioSet (527 tags including Music/Singing/Speech/
Narration), and needs no GPU -- ideal for this CPU-only build.

This stage only LABELS the candidate (SONG/MUSIC, SPEECH, MIXED, UNKNOWN) and
emits music/speech probabilities as signals. It does NOT itself reject; the
content-type decision (e.g. SPEECH) is made in final_scoring.py using these
signals together with the per-category spoken-narration overrides.

INPUTS
    candidate    : Candidate (uses .audio_path)
    model_manager: ModelManager (provides the cached PANNs tagger)
    cfg          : PipelineConfig (reads `classifier`)
OUTPUT
    GateResult(stage=CLASSIFIER) with score=music_prob and details:
        {label, music_prob, speech_prob}
"""
from __future__ import annotations

from typing import Any

from ..config import PipelineConfig
from ..logging_utils import get_logger
from ..models import Candidate, GateResult, Stage

_log = get_logger("gate.classifier")

_PANNS_SR = 32000  # Cnn14 was trained at 32 kHz

_MUSIC_KEYS = (
    "music", "musical instrument", "singing", "song", "choir", "chant",
    "drum", "percussion", "guitar", "violin", "flute", "harmonica", "melody",
)
_SPEECH_KEYS = (
    "speech", "narration", "monologue", "conversation", "speech synthesizer",
    "man speaking", "woman speaking", "child speech",
)


def _load_labels() -> list[str]:
    try:
        from panns_inference.config import labels
        return list(labels)
    except Exception:  # noqa: BLE001
        return []


def _aggregate(clipwise: Any, labels: list[str]) -> tuple[float, float]:
    import numpy as np
    probs = np.asarray(clipwise).reshape(-1)
    music_p = 0.0
    speech_p = 0.0
    for i, name in enumerate(labels):
        if i >= probs.shape[0]:
            break
        low = name.lower()
        p = float(probs[i])
        if any(k in low for k in _MUSIC_KEYS):
            music_p = max(music_p, p)
        if any(k in low for k in _SPEECH_KEYS):
            speech_p = max(speech_p, p)
    return music_p, speech_p


def evaluate(candidate: Candidate, model_manager: Any, cfg: PipelineConfig) -> GateResult:
    """Classify the candidate's audio as SONG/MUSIC/SPEECH/MIXED/UNKNOWN.

    Returns a passing GateResult carrying the music/speech probabilities and the
    derived label. Never rejects here (see module docstring).
    """
    import librosa
    import numpy as np

    path = candidate.audio_path
    cl = cfg.raw["classifier"]
    try:
        y, _ = librosa.load(path, sr=_PANNS_SR, mono=True)
        if y.size == 0:
            raise ValueError("empty audio")
        audio = np.asarray(y, dtype=np.float32)[None, :]  # (1, samples)
        tagger = model_manager.panns_tagger()
        clipwise, _embedding = tagger.inference(audio)
        labels = _load_labels()
        music_p, speech_p = _aggregate(clipwise, labels)
    except Exception as exc:  # noqa: BLE001 - never kill the batch on one file
        _log.warning("classifier failed for %s: %r", candidate.recording_id, exc)
        return GateResult.ok(Stage.CLASSIFIER, score=None,
                             message=f"classifier error: {exc}",
                             label="UNKNOWN", music_prob=None, speech_prob=None)

    music_min = float(cl["music_min_prob"])
    speech_reject = float(cl["speech_reject_prob"])
    mixed_low = float(cl["mixed_low"])

    if music_p >= music_min and music_p >= speech_p:
        label = "SONG/MUSIC"
    elif speech_p >= speech_reject and speech_p > music_p:
        label = "SPEECH"
    elif music_p >= mixed_low and speech_p >= mixed_low:
        label = "MIXED"
    else:
        label = "UNKNOWN"

    return GateResult.ok(
        Stage.CLASSIFIER, score=round(music_p, 4),
        message=f"label={label} music={music_p:.2f} speech={speech_p:.2f}",
        label=label, music_prob=round(music_p, 4), speech_prob=round(speech_p, 4),
    )
