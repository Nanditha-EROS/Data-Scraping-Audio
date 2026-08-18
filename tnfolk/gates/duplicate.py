"""Stage 9 -- Duplicate Detection (Chromaprint / fpcalc, algorithmic, no ML).

Fingerprints the recording with Chromaprint (`fpcalc -raw`) and compares it to
previously-seen fingerprints (persisted in a JSONL store, not a database). A
re-upload of the EXACT SAME recording is a duplicate (high fingerprint
similarity); a different performance/singer of the same traditional song is NOT
a duplicate and is kept.

INPUTS
    candidate        : Candidate (uses .audio_path, .recording_id)
    fingerprint_store: FingerprintStore (cross-run persistence)
    cfg              : PipelineConfig (reads `duplicate`)
OUTPUT
    GateResult(stage=DUPLICATE_GATE); passing adds the fingerprint to the store.
REJECTION CODES
    DUPLICATE_RECORDING
"""
from __future__ import annotations

import subprocess
from typing import Any, Optional

from ..config import PipelineConfig
from ..logging_utils import get_logger
from ..models import Candidate, GateResult, RejectionCode, Stage

_log = get_logger("gate.duplicate")


def _fpcalc_raw(path: str, length_sec: int, fpcalc: str) -> tuple[float, list[int]]:
    cmd = [fpcalc, "-raw", "-length", str(length_sec), path]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")
    if proc.returncode != 0:
        raise RuntimeError(f"fpcalc failed: {proc.stderr.strip()[:200]}")
    duration = 0.0
    fingerprint: list[int] = []
    for line in proc.stdout.splitlines():
        if line.startswith("DURATION="):
            try:
                duration = float(line.split("=", 1)[1])
            except ValueError:
                pass
        elif line.startswith("FINGERPRINT="):
            raw = line.split("=", 1)[1].strip()
            fingerprint = [int(x) for x in raw.split(",") if x.strip()]
    if not fingerprint:
        raise RuntimeError("empty fingerprint")
    return duration, fingerprint


def _similarity(a: list[int], b: list[int]) -> float:
    """Fraction of matching bits over the aligned fingerprint prefix (0..1).

    Chromaprint raw fingerprints are 32-bit sub-fingerprints; identical
    recordings have near-zero bit-error-rate.
    """
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    total_bits = n * 32
    diff_bits = 0
    for i in range(n):
        diff_bits += bin((a[i] ^ b[i]) & 0xFFFFFFFF).count("1")
    return 1.0 - (diff_bits / total_bits)


def evaluate(candidate: Candidate, fingerprint_store: Any, cfg: PipelineConfig) -> GateResult:
    """Fingerprint the candidate and reject if it duplicates a stored recording.

    On pass, the new fingerprint is added to the store so subsequent candidates
    (this run or future runs) can be deduped against it.
    """
    d = cfg.raw["duplicate"]
    threshold = float(d["fingerprint_similarity"])
    length_sec = int(d["fpcalc_length_sec"])
    fpcalc = d.get("fpcalc_path") or "fpcalc"

    try:
        duration, fp = _fpcalc_raw(candidate.audio_path, length_sec, fpcalc)
    except Exception as exc:  # noqa: BLE001 - fpcalc missing/failed: don't kill batch
        _log.warning("fingerprinting unavailable for %s (%s); skipping dedupe",
                     candidate.recording_id, exc)
        return GateResult.ok(Stage.DUPLICATE_GATE, score=None,
                             message=f"dedupe skipped: {exc}", fingerprinted=False)

    best_sim = 0.0
    best_id: Optional[str] = None
    for rid, item in fingerprint_store.items(exclude=candidate.recording_id):
        other = item.get("fingerprint") or []
        sim = _similarity(fp, other)
        if sim > best_sim:
            best_sim, best_id = sim, rid

    if best_sim >= threshold:
        return GateResult.reject(
            Stage.DUPLICATE_GATE, RejectionCode.DUPLICATE_RECORDING,
            message=f"duplicate of {best_id} (sim={best_sim:.3f})",
            duplicate_of=best_id, similarity=round(best_sim, 4),
        )

    fingerprint_store.add(candidate.recording_id, duration, fp)
    return GateResult.ok(Stage.DUPLICATE_GATE, score=None,
                         message=f"unique recording (best sim={best_sim:.3f})",
                         fingerprinted=True, best_similarity=round(best_sim, 4))
