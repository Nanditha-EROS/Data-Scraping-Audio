"""Stage 10 -- Weighted Final Decision.

Combines the quality score, folk-relevance similarity, classifier confidence and
VAD ratio into a single weighted score, then resolves ACCEPT / REVIEW / REJECT.

HARD OVERRIDE (explicit, not merely via weighting): if the audio quality score is
below quality_gate.minimum_quality_score, the candidate is REJECTED regardless of
how high its folk-relevance score is (e.g. folk=0.98, quality=0.21 -> REJECT).

Content-type rejects (SPEECH / NON_SONG) use the classifier signals together with
the per-category spoken-narration overrides. Folk shortfalls reject as NON_FOLK.

This gate NEVER promotes a candidate that was flagged REVIEW upstream to ACCEPT
(borderline confidence anywhere in the chain -> REVIEW, never auto-promoted).

INPUTS
    candidate : Candidate (reads all accumulated signals)
    category  : Category (gate_overrides)
    cfg       : PipelineConfig (reads `scoring`, `quality_gate`, `classifier`)
OUTPUT
    GateResult(stage=FINAL_SCORING); sets candidate.final_score.
    A passing (ok) result means "no objection -> ACCEPT" unless an upstream
    REVIEW flag already downgraded the candidate.
REJECTION CODES
    (quality override) MOSTLY_SILENT/EXCESSIVE_CLIPPING/SEVERE_DISTORTION/
    VERY_LOW_SNR/EXCESSIVE_NOISE/AUDIO_DROPOUT/LOW_BANDWIDTH,
    (content) SPEECH, NON_SONG, NON_FOLK
"""
from __future__ import annotations

from typing import Any, Optional

from ..config import Category, PipelineConfig
from ..logging_utils import get_logger
from ..models import Candidate, GateResult, RejectionCode, Stage

_log = get_logger("gate.scoring")


def _sig(candidate: Candidate, stage: Stage) -> dict[str, Any]:
    entry = candidate.signals.get(stage.value)
    return entry if isinstance(entry, dict) else {}


def _quality_reject_code(qd: dict[str, Any], cfg: PipelineConfig) -> RejectionCode:
    """Pick the most-appropriate code when the composite quality score is below
    the minimum (no single hard threshold was breached, so choose the weakest
    metric relative to its limit)."""
    qg = cfg.raw["quality_gate"]
    candidates: list[tuple[float, RejectionCode]] = []

    def rel(value: float, limit: float, higher_is_worse: bool) -> float:
        # normalized "badness" >0 means worse; used only for ranking
        if higher_is_worse:
            return value - limit
        return limit - value

    if qd:
        candidates.append((rel(qd.get("near_silence_ratio", 0.0),
                               float(qg["maximum_silence_ratio"]), True),
                           RejectionCode.MOSTLY_SILENT))
        candidates.append((rel(qd.get("clipping_ratio", 0.0),
                               float(qg["maximum_clipping_ratio"]), True),
                           RejectionCode.EXCESSIVE_CLIPPING))
        candidates.append((rel(qd.get("snr_db", 99.0),
                               float(qg["minimum_snr_db"]), False),
                           RejectionCode.VERY_LOW_SNR))
        candidates.append((rel(qd.get("noise_floor_dbfs", -99.0),
                               float(qg["maximum_noise_floor_dbfs"]), True),
                           RejectionCode.EXCESSIVE_NOISE))
        candidates.append((rel(qd.get("dynamic_range_db", 99.0),
                               float(qg["minimum_dynamic_range_db"]), False),
                           RejectionCode.SEVERE_DISTORTION))
        candidates.append((rel(qd.get("dropout_ratio", 0.0),
                               float(qg["dropout"]["max_dropout_ratio"]), True),
                           RejectionCode.AUDIO_DROPOUT))
        candidates.append((rel(qd.get("median_rolloff_hz", 99999.0),
                               float(qg["bandwidth"]["min_rolloff_hz"]), False),
                           RejectionCode.LOW_BANDWIDTH))
    if not candidates:
        return RejectionCode.SEVERE_DISTORTION
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def _weighted_score(values: dict[str, Optional[float]], weights: dict[str, float]) -> float:
    """Weighted mean over available (non-None) signals; missing signals drop out
    and remaining weights are renormalized."""
    num = 0.0
    den = 0.0
    for key, w in weights.items():
        v = values.get(key)
        if v is None:
            continue
        num += float(w) * float(v)
        den += float(w)
    return (num / den) if den > 0 else 0.0


def evaluate(candidate: Candidate, category: Category, cfg: PipelineConfig) -> GateResult:
    """Resolve the final weighted decision for the candidate."""
    sc = cfg.raw["scoring"]
    weights = {k: float(v) for k, v in sc["weights"].items()}
    accept_t = float(sc["accept_threshold"])
    review_t = float(sc["review_threshold"])
    min_quality = float(cfg.raw["quality_gate"]["minimum_quality_score"])
    cl = cfg.raw["classifier"]

    qd = _sig(candidate, Stage.QUALITY_GATE)
    cd = _sig(candidate, Stage.CLASSIFIER)
    fd = _sig(candidate, Stage.FOLK_RELEVANCE_GATE)
    vd = _sig(candidate, Stage.VAD)

    quality_score = qd.get("score")
    folk_sim = fd.get("score")
    music_prob = cd.get("music_prob")
    speech_prob = cd.get("speech_prob")
    label = cd.get("label")
    vad_ratio = vd.get("score")

    values = {
        "quality": quality_score,
        "folk_relevance": folk_sim,
        "music_speech": music_prob,
        "vad": vad_ratio,
    }
    weighted = round(_weighted_score(values, weights), 4)
    candidate.final_score = weighted

    details: dict[str, Any] = {
        "weighted_score": weighted, "quality_score": quality_score,
        "folk_sim": folk_sim, "music_prob": music_prob, "speech_prob": speech_prob,
        "label": label, "vad_ratio": vad_ratio,
    }

    # --- 1) HARD OVERRIDE: quality below minimum -> REJECT regardless of folk ---
    if quality_score is not None and quality_score < min_quality:
        code = _quality_reject_code(qd, cfg)
        return GateResult.reject(
            Stage.FINAL_SCORING, code,
            message=f"quality hard-override: {quality_score:.2f} < {min_quality:.2f} "
                    f"(folk={folk_sim}) -> REJECT",
            override="quality_min", **details,
        )

    # --- 2) Content-type reject (SPEECH), honoring narration overrides ---
    allow_narration = bool(category.gate_overrides.get("allow_spoken_narration", False))
    speech_reject = float(cl["speech_reject_prob"])
    music_min = float(cl["music_min_prob"])
    if label == "SPEECH" and speech_prob is not None:
        very_speechy = speech_prob > 0.92
        low_music = (music_prob or 0.0) < music_min
        if allow_narration:
            if very_speechy and low_music:
                return GateResult.reject(Stage.FINAL_SCORING, RejectionCode.SPEECH,
                                         message="spoken-dominant beyond narration tolerance",
                                         **details)
        elif speech_prob >= speech_reject and low_music:
            return GateResult.reject(Stage.FINAL_SCORING, RejectionCode.SPEECH,
                                     message=f"speech-dominant ({speech_prob:.2f})", **details)

    # --- 3) Neither music nor speech nor folk -> NON_SONG ---
    if (music_prob is not None and music_prob < music_min
            and (folk_sim is not None and folk_sim < review_t)):
        return GateResult.reject(Stage.FINAL_SCORING, RejectionCode.NON_SONG,
                                 message="no musical content and low folk relevance", **details)

    # --- 4) Weighted thresholds ---
    if weighted < review_t:
        # decide the dominant shortfall for the code
        code = RejectionCode.NON_FOLK if (folk_sim is not None and folk_sim < review_t) \
            else RejectionCode.NON_SONG
        return GateResult.reject(Stage.FINAL_SCORING, code,
                                 message=f"weighted {weighted:.2f} < {review_t:.2f}", **details)
    if weighted < accept_t:
        return GateResult.flag_review(Stage.FINAL_SCORING, score=weighted,
                                      message=f"weighted {weighted:.2f} in review band", **details)

    # No objection. Orchestrator promotes to ACCEPT only if nothing upstream
    # already flagged REVIEW (borderline is never auto-promoted).
    return GateResult.ok(Stage.FINAL_SCORING, score=weighted,
                         message=f"weighted {weighted:.2f} >= accept {accept_t:.2f}", **details)
