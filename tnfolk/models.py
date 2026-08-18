"""Shared data models, enums and the standardized rejection-code set.

These types are the common currency passed between gates. Every gate returns a
``GateResult``; the orchestrator threads a single ``Candidate`` through the
fixed gate chain and accumulates per-gate results and signals.

IMPORTANT (Design Doc constraints):
- There is NO NSFW gate and NO NSFW rejection code anywhere in this module.
- The folk-relevance gate rejects with NON_FOLK only -- never MOVIE_SONG/FILM_SONG/OST.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional


class Decision(str, enum.Enum):
    """Terminal decision buckets (Design Doc Section 8)."""

    ACCEPT = "ACCEPT"    # passes every gate, no hard-fail -> Gold Data
    REVIEW = "REVIEW"    # borderline anywhere in the chain -> manual review only
    REJECT = "REJECT"    # fails a gate outright, with a standardized code
    PENDING = "PENDING"  # not yet decided (in-flight)


class RejectionCode(str, enum.Enum):
    """The ONLY allowed rejection codes (Design Doc Section 8).

    No NSFW-related codes exist by design. Folk rejection is always NON_FOLK.
    """

    # content-type / relevance
    INTERVIEW = "INTERVIEW"
    SPEECH = "SPEECH"
    PODCAST = "PODCAST"
    NEWS = "NEWS"
    LECTURE = "LECTURE"
    DISCUSSION = "DISCUSSION"
    SERMON = "SERMON"
    STORY_ONLY = "STORY_ONLY"
    NON_SONG = "NON_SONG"
    NON_FOLK = "NON_FOLK"
    # audio quality (DSP)
    CORRUPT_AUDIO = "CORRUPT_AUDIO"
    TOO_SHORT = "TOO_SHORT"
    TOO_LONG = "TOO_LONG"
    MOSTLY_SILENT = "MOSTLY_SILENT"
    EXCESSIVE_CLIPPING = "EXCESSIVE_CLIPPING"
    SEVERE_DISTORTION = "SEVERE_DISTORTION"
    VERY_LOW_SNR = "VERY_LOW_SNR"
    EXCESSIVE_NOISE = "EXCESSIVE_NOISE"
    AUDIO_DROPOUT = "AUDIO_DROPOUT"
    LOW_BANDWIDTH = "LOW_BANDWIDTH"
    # dedupe / pipeline
    DUPLICATE_RECORDING = "DUPLICATE_RECORDING"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"


class Stage(str, enum.Enum):
    """Named pipeline stages, in canonical order. Used for resume bookkeeping."""

    SEARCH = "search"
    METADATA_GATE = "metadata_gate"
    DOWNLOAD = "download"
    INTEGRITY_GATE = "integrity_gate"
    QUALITY_GATE = "quality_gate"
    CLASSIFIER = "classifier"
    VAD = "vad"
    FOLK_RELEVANCE_GATE = "folk_relevance_gate"
    DUPLICATE_GATE = "duplicate_gate"
    FINAL_SCORING = "final_scoring"
    TRANSCRIPTION = "transcription"
    STORAGE = "storage"


# Canonical execution order (single source of truth for the orchestrator).
STAGE_ORDER: tuple[Stage, ...] = (
    Stage.SEARCH,
    Stage.METADATA_GATE,
    Stage.DOWNLOAD,
    Stage.INTEGRITY_GATE,
    Stage.QUALITY_GATE,
    Stage.CLASSIFIER,
    Stage.VAD,
    Stage.FOLK_RELEVANCE_GATE,
    Stage.DUPLICATE_GATE,
    Stage.FINAL_SCORING,
    Stage.TRANSCRIPTION,
    Stage.STORAGE,
)


@dataclass
class GateResult:
    """Uniform result returned by every gate function.

    Attributes:
        passed: True if the candidate may proceed to the next stage.
        decision: PENDING while flowing; REJECT on hard-fail; REVIEW if borderline.
        rejection_code: set iff decision == REJECT.
        score: optional 0..1 signal this gate contributes to final scoring.
        review: True if this gate flags the candidate as borderline (-> REVIEW).
        stage: which Stage produced this result.
        details: gate-specific metrics for logging/debugging (JSON-serializable).
        message: short human-readable explanation.
    """

    passed: bool
    stage: Stage
    decision: Decision = Decision.PENDING
    rejection_code: Optional[RejectionCode] = None
    score: Optional[float] = None
    review: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def __post_init__(self) -> None:
        if self.decision == Decision.REJECT and self.rejection_code is None:
            raise ValueError("REJECT GateResult must carry a rejection_code")
        if self.rejection_code is not None and self.decision != Decision.REJECT:
            raise ValueError("rejection_code only valid with decision=REJECT")

    @classmethod
    def ok(cls, stage: Stage, *, score: float | None = None,
           review: bool = False, message: str = "", **details: Any) -> "GateResult":
        return cls(passed=True, stage=stage, decision=Decision.PENDING,
                   score=score, review=review, message=message, details=details)

    @classmethod
    def reject(cls, stage: Stage, code: RejectionCode, *, message: str = "",
               **details: Any) -> "GateResult":
        return cls(passed=False, stage=stage, decision=Decision.REJECT,
                   rejection_code=code, message=message, details=details)

    @classmethod
    def flag_review(cls, stage: Stage, *, score: float | None = None,
                    message: str = "", **details: Any) -> "GateResult":
        # Passes to the next stage but is marked borderline; the orchestrator
        # will not auto-promote a REVIEW-flagged candidate to ACCEPT.
        return cls(passed=True, stage=stage, decision=Decision.REVIEW,
                   score=score, review=True, message=message, details=details)


@dataclass
class Candidate:
    """A single YouTube recording flowing through the pipeline.

    recording_id == the YouTube video id (stable, unique, resumable). It is the
    filename stem for <recording_id>.wav / <recording_id>.txt.
    """

    recording_id: str                 # youtube video id
    category_slug: str                # folder_slug from taxonomy
    title: str = ""
    description: str = ""
    channel: str = ""
    url: str = ""
    duration_sec: float = 0.0

    # populated as the candidate progresses
    audio_path: Optional[str] = None  # local wav path once downloaded
    signals: dict[str, Any] = field(default_factory=dict)   # scores per stage
    gate_results: list[GateResult] = field(default_factory=list)

    decision: Decision = Decision.PENDING
    rejection_code: Optional[RejectionCode] = None
    final_score: Optional[float] = None
    transcript_path: Optional[str] = None

    @property
    def youtube_url(self) -> str:
        return self.url or f"https://www.youtube.com/watch?v={self.recording_id}"

    def record(self, result: GateResult) -> None:
        """Append a gate result and fold its signal/decision into the candidate.

        The full per-stage signal dict (score + gate details) is stored under
        ``signals[stage]`` so later stages (e.g. final scoring) can read every
        upstream metric, not just the scalar score.
        """
        self.gate_results.append(result)
        self.signals[result.stage.value] = {
            "score": result.score,
            "review": result.review,
            **result.details,
        }
        if result.decision == Decision.REJECT:
            self.decision = Decision.REJECT
            self.rejection_code = result.rejection_code
        elif result.review and self.decision != Decision.REJECT:
            # a review flag never overrides an existing reject
            self.decision = Decision.REVIEW

    def signal_score(self, stage: "Stage") -> Optional[float]:
        """Convenience: the scalar score recorded for a stage, if any."""
        entry = self.signals.get(stage.value)
        if isinstance(entry, dict):
            return entry.get("score")
        return None
