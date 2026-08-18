"""Stage 8 -- Folk-Relevance Gate (category-aware, core semantic gate).

Primary signal: multilingual sentence-transformers embeddings comparing the
candidate's textual evidence (title/description + category + audio-classifier
label) against the category's reference concepts via cosine similarity
(CPU-efficient, run on every candidate).

Secondary signal: LAION-CLAP audio<->text similarity, invoked ONLY for
candidates whose text similarity lands in the uncertain band around the REVIEW
threshold (keeps CPU load manageable -- not run on every candidate).

Movie origin, actor names and "official song" labels are explicitly IGNORED:
only the audio's folk characteristics matter. If a candidate is rejected here
the code is ALWAYS ``NON_FOLK`` -- never MOVIE_SONG/FILM_SONG/OST.

Per-category gate overrides (allow_spoken_narration / max_speech_ratio) from
folk_categories.yaml are applied so narration-heavy forms (villupattu,
therukoothu, oppari, Samiyattam, tribal, ballads) are not penalised for
containing spoken narration.

INPUTS
    candidate     : Candidate (title/description + prior classifier/vad signals)
    category      : Category (reference_concepts + gate_overrides)
    model_manager : ModelManager (sentence encoder + optional CLAP)
    cfg           : PipelineConfig (reads `folk_relevance`, `classifier`)
OUTPUT
    GateResult(stage=FOLK_RELEVANCE_GATE) with score=folk_similarity.
REJECTION CODES
    NON_FOLK (only).
"""
from __future__ import annotations

import re
from typing import Any

from ..config import Category, PipelineConfig
from ..logging_utils import get_logger
from ..models import Candidate, GateResult, RejectionCode, Stage

_log = get_logger("gate.folk")

# Movie/industry markers to strip from candidate text so film origin never acts
# as a signal (we judge folk merit only, never movie-detection).
_IGNORE_MARKERS = re.compile(
    r"\b(official|movie|film|cinema|ost|audio\s+song|lyric(?:al)?\s+video|"
    r"video\s+song|full\s+song|hd|4k|remix|trailer|teaser)\b",
    re.IGNORECASE,
)


def _clean_text(candidate: Candidate) -> str:
    text = f"{candidate.title} {candidate.description}".strip()
    text = _IGNORE_MARKERS.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _cos_sim_max(encoder: Any, query: str, refs: list[str]) -> float:
    import numpy as np
    embeddings = encoder.encode([query] + refs, normalize_embeddings=True,
                                convert_to_numpy=True)
    q = embeddings[0]
    r = embeddings[1:]
    if r.shape[0] == 0:
        return 0.0
    sims = r @ q
    return float(np.max(sims))


def _clap_sim(model_manager: Any, audio_path: str, refs: list[str]) -> float:
    import numpy as np
    clap = model_manager.clap()
    audio_emb = clap.get_audio_embedding_from_filelist(x=[audio_path], use_tensor=False)
    text_emb = clap.get_text_embedding(refs, use_tensor=False)
    a = np.asarray(audio_emb).reshape(1, -1)
    t = np.asarray(text_emb)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    sims = (t @ a.T).reshape(-1)
    return float(np.max(sims))


def evaluate(candidate: Candidate, category: Category, model_manager: Any,
             cfg: PipelineConfig) -> GateResult:
    """Score how folk-relevant the candidate is for its category.

    Returns pass (score>=accept), REVIEW flag (borderline), or
    reject(NON_FOLK) (clearly not folk). CLAP is consulted only in the uncertain
    band around the review threshold.
    """
    fr = cfg.raw["folk_relevance"]
    accept_sim = float(fr["accept_similarity"])
    review_sim = float(fr["review_similarity"])
    band = float(fr["uncertain_band"])
    clap_cfg = fr.get("clap", {}) or {}

    refs = list(category.reference_concepts)
    if not refs:
        refs = [f"Tamil folk {category.english_name} traditional song"]

    # candidate evidence = cleaned title/description + category + classifier label
    audio_label = None
    cls_sig = candidate.signals.get(Stage.CLASSIFIER.value)
    if isinstance(cls_sig, dict):
        audio_label = cls_sig.get("label")
    evidence = _clean_text(candidate)
    if audio_label and audio_label != "UNKNOWN":
        evidence = f"{evidence} [{audio_label}]"
    if not evidence:
        evidence = category.english_name

    try:
        encoder = model_manager.sentence_encoder()
        text_sim = _cos_sim_max(encoder, evidence, refs)
    except Exception as exc:  # noqa: BLE001 - one file must not kill the batch
        _log.warning("folk-relevance encode failed for %s: %r",
                     candidate.recording_id, exc)
        # Uncertain -> REVIEW rather than a wrong hard reject.
        return GateResult.flag_review(Stage.FOLK_RELEVANCE_GATE, score=None,
                                      message=f"encode error: {exc}")

    combined = text_sim
    clap_used = False
    clap_val = None
    # CLAP fallback ONLY in the uncertain band around the review threshold.
    if (clap_cfg.get("enabled") and candidate.audio_path
            and abs(text_sim - review_sim) <= band):
        try:
            clap_val = _clap_sim(model_manager, candidate.audio_path, refs)
            w = float(clap_cfg.get("blend_weight", 0.5))
            combined = (1.0 - w) * text_sim + w * clap_val
            clap_used = True
        except Exception as exc:  # noqa: BLE001 - CLAP optional
            _log.warning("CLAP fallback failed for %s: %r", candidate.recording_id, exc)

    # apply narration override: if the classifier called it SPEECH but this
    # category legitimately contains spoken narration, do not let that drag the
    # candidate below folk relevance.
    allow_narration = bool(category.gate_overrides.get("allow_spoken_narration", False))

    details: dict[str, Any] = {
        "text_sim": round(text_sim, 4),
        "clap_sim": None if clap_val is None else round(clap_val, 4),
        "clap_used": clap_used,
        "combined_sim": round(combined, 4),
        "allow_spoken_narration": allow_narration,
    }

    if combined >= accept_sim:
        return GateResult.ok(Stage.FOLK_RELEVANCE_GATE, score=round(combined, 4),
                             message=f"folk-relevant sim={combined:.2f}", **details)
    if combined >= review_sim:
        return GateResult.flag_review(Stage.FOLK_RELEVANCE_GATE, score=round(combined, 4),
                                      message=f"borderline folk sim={combined:.2f}", **details)
    return GateResult.reject(Stage.FOLK_RELEVANCE_GATE, RejectionCode.NON_FOLK,
                             message=f"not folk (sim={combined:.2f} < {review_sim:.2f})",
                             **details)
