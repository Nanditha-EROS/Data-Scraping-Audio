"""Metadata Relevance Gate -- the cheap, first, pre-download filter.

Design Doc 7.1: filters obviously-irrelevant results using title/description
keyword + fuzzy matching (rapidfuzz) against the category's keywords, BEFORE any
download happens. It also rejects obvious non-song content types.

INPUTS
    candidate : Candidate  -- uses .title and .description (metadata only, no audio)
    category  : Category   -- provides keywords + per-category gate_overrides
    cfg       : PipelineConfig -- reads the `metadata_gate` section

OUTPUT
    GateResult(stage=METADATA_GATE) with:
        - score  : best fuzzy relevance 0..1 (title/description vs category keywords)
        - passed : True if it may proceed to Download
        - decision: PENDING (pass), REVIEW (borderline relevance), or REJECT

REJECTION CODES this gate may emit (all from the standardized set):
    INTERVIEW, SPEECH, PODCAST, NEWS, LECTURE, DISCUSSION, SERMON
        -- when a hard non-song content-type term appears in title/description.
    NON_FOLK
        -- when the metadata is clearly unrelated to the folk category
           (no keyword hit and best fuzzy score well below threshold).

There is deliberately NO NSFW handling here (Design Doc: no NSFW gate/codes).
Narration-heavy categories (allow_spoken_narration) skip the speech-oriented
content-type exclusions (SPEECH/LECTURE/DISCUSSION) since those forms legitimately
contain spoken narration; hard non-music types (INTERVIEW/PODCAST/NEWS/SERMON)
still reject.
"""
from __future__ import annotations

import re
from typing import Any

# pyrefly: ignore [missing-import]
from rapidfuzz import fuzz

from ..config import Category, PipelineConfig
from ..logging_utils import get_logger
from ..models import Candidate, GateResult, RejectionCode, Stage

_log = get_logger("gate.metadata")

_TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")

# Content-type exclusions that are speech-oriented; suppressed when the category
# explicitly allows spoken narration.
_SPEECH_ORIENTED = {"SPEECH", "LECTURE", "DISCUSSION"}

# Non-Tamil language markers in titles/descriptions. If any of these appear,
# the video is clearly not a Tamil folk song -- reject BEFORE download.
# Covers Telugu, Kannada, Hindi (Devanagari), Malayalam, Bengali scripts
# plus common English giveaways from non-Tamil nursery rhyme channels.
_NON_TAMIL_SCRIPT_RE = re.compile(
    r"["
    r"\u0C00-\u0C7F"   # Telugu script
    r"\u0C80-\u0CFF"   # Kannada script
    r"\u0900-\u097F"   # Devanagari (Hindi, Marathi, Sanskrit)
    r"\u0D00-\u0D7F"   # Malayalam script
    r"\u0980-\u09FF"   # Bengali script
    r"]"
)

# English title patterns that are clearly non-Tamil language channels.
_NON_TAMIL_ENGLISH_TERMS = [
    # Telugu/Kannada folk giveaways
    "telugu", "kannada", "malayalam", "hindi", "bengali", "marathi",
    "kiswahili", "swahili",
    # Non-Tamil nursery rhyme channels
    "nursery rhyme", "nursery rhymes", "abc song", "abc songs",
    "kids song english", "kids english song",
]


def _has_tamil(text: str) -> bool:
    return bool(_TAMIL_RE.search(text))


def _term_present(term: str, haystack_lower: str, haystack_raw: str) -> bool:
    """Word-boundary match for latin terms; substring match for Tamil terms."""
    if _has_tamil(term):
        return term in haystack_raw
    t = term.lower().strip()
    if not t:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", haystack_lower) is not None


def _best_keyword_score(keywords: list[str], haystack_lower: str,
                        haystack_raw: str) -> tuple[float, str, list[tuple[float, str]]]:
    """Return (best_score_0_100, best_keyword, per_keyword_scores).

    Uses rapidfuzz partial/token matching. Tamil keywords are matched against the
    raw (original-case) haystack; latin keywords against the lowercased haystack.
    """
    best_score = 0.0
    best_kw = ""
    scores: list[tuple[float, str]] = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        if _has_tamil(kw):
            # substring gives 100; else partial_ratio on raw text
            score = 100.0 if kw in haystack_raw else float(fuzz.partial_ratio(kw, haystack_raw))
        else:
            klow = kw.lower()
            score = float(max(
                fuzz.partial_ratio(klow, haystack_lower),
                fuzz.token_set_ratio(klow, haystack_lower),
            ))
        scores.append((score, kw))
        if score > best_score:
            best_score, best_kw = score, kw
    return best_score, best_kw, scores


def evaluate(candidate: Candidate, category: Category, cfg: PipelineConfig) -> GateResult:
    """Run the metadata relevance gate. Pure function over metadata only.

    Args:
        candidate: the video candidate (title/description read; audio not needed).
        category: taxonomy entry for the search category.
        cfg: pipeline configuration (reads `metadata_gate`).

    Returns:
        GateResult for Stage.METADATA_GATE (see module docstring for codes).
    """
    mg = cfg.raw["metadata_gate"]
    threshold = float(mg["fuzzy_match_threshold"])
    min_hits = int(mg["min_keyword_hits"])
    review_margin = float(mg["review_margin"])
    exclude_terms: dict[str, list[str]] = dict(mg.get("exclude_terms") or {})

    allow_narration = bool(category.gate_overrides.get("allow_spoken_narration", False))

    title = candidate.title or ""
    desc = candidate.description or ""
    raw = f"{title}\n{desc}"
    low = raw.lower()

    # 1a) Strict Tamil-language filter -- reject BEFORE download if the title/
    #     description contains recognizable non-Tamil script or language markers.
    #     This ensures only Tamil-language folk content passes through.
    if _NON_TAMIL_SCRIPT_RE.search(raw):
        return GateResult.reject(
            Stage.METADATA_GATE, RejectionCode.NON_FOLK,
            message="non-Tamil script detected in title/description -> rejected (Tamil-only)",
        )
    for non_tamil_term in _NON_TAMIL_ENGLISH_TERMS:
        if re.search(rf"(?<![a-z0-9]){re.escape(non_tamil_term)}(?![a-z0-9])", low):
            return GateResult.reject(
                Stage.METADATA_GATE, RejectionCode.NON_FOLK,
                message=f"non-Tamil language marker '{non_tamil_term}' in title -> rejected (Tamil-only)",
            )

    # 1b) Hard content-type exclusions -> standardized rejection code.
    for code_name, terms in exclude_terms.items():
        if allow_narration and code_name in _SPEECH_ORIENTED:
            continue  # narration-heavy category legitimately contains spoken parts
        for term in terms:
            if _term_present(term, low, raw):
                try:
                    code = RejectionCode(code_name)
                except ValueError:
                    _log.warning("Unknown exclude code %s in config; skipping", code_name)
                    continue
                return GateResult.reject(
                    Stage.METADATA_GATE, code,
                    message=f"content-type term '{term}' matched -> {code_name}",
                    matched_term=term,
                )

    # 2) Keyword / fuzzy relevance against category keywords.
    keywords = category.keywords()
    best_score, best_kw, scores = _best_keyword_score(keywords, low, raw)
    hit_count = sum(1 for s, _ in scores if s >= threshold)
    norm = round(best_score / 100.0, 4)

    details: dict[str, Any] = {
        "best_score": round(best_score, 2),
        "best_keyword": best_kw,
        "hit_count": hit_count,
        "threshold": threshold,
    }

    if best_score >= threshold and hit_count >= min_hits:
        return GateResult.ok(
            Stage.METADATA_GATE, score=norm,
            message=f"relevant (best='{best_kw}' {best_score:.0f})", **details,
        )

    # Borderline band -> REVIEW (proceed but flagged; never auto-promoted).
    if best_score >= (threshold - review_margin):
        return GateResult.flag_review(
            Stage.METADATA_GATE, score=norm,
            message=f"borderline relevance (best='{best_kw}' {best_score:.0f})", **details,
        )

    # Clearly unrelated to the folk category.
    return GateResult.reject(
        Stage.METADATA_GATE, RejectionCode.NON_FOLK,
        message=f"no category keyword match (best='{best_kw}' {best_score:.0f} < {threshold:.0f})",
        **details,
    )
