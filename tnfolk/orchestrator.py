"""Orchestrator -- ties every stage together in the exact required order.

Order (do not reorder):
    1  YouTube Search
    2  Metadata Relevance Gate      (pre-download)
    3  Download Full Audio
    4  Integrity Gate
    5  Strict Audio Quality Gate    (highest priority; hard override in scoring)
    6  Music/Speech/Mixed Classifier (PANNs)
    7  Voice Activity Analysis      (Silero VAD; signal only)
    8  Folk-Relevance Gate          (sentence-transformers + CLAP fallback; NON_FOLK only)
    9  Duplicate Detection          (Chromaprint/fpcalc)
    10 Weighted Final Decision      (ACCEPT / REVIEW / REJECT)
    11 Full-Audio Transcription     (ACCEPT only)
    12 Generate lyrics .txt
    13 Upload (ACCEPT + REVIEW; REJECT never stored)

Reliability:
    * No database. Resume = output-file existence + JSONL run-log skip-set.
    * Every network call is retried (tenacity, exponential backoff, hard cutoff).
    * Concurrency capped per stage; inference/transcription serialized (CPU-only).
    * One file's exception never kills the batch -- it is caught, logged with the
      recording id, marked REJECT(DOWNLOAD_FAILED / CORRUPT_AUDIO), and skipped.
    * A transcription failure never undoes ACCEPT or deletes the stored audio.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import threading
from typing import Optional

from .config import Category, PipelineConfig, load_pipeline_config, load_taxonomy
from .device import ModelManager
from .logging_utils import get_logger, setup_logging
from .models import Candidate, Decision, RejectionCode, Stage
from .query_generator import generate_queries
from .runlog import FingerprintStore, RunLog
from .storage import Storage
from . import chunker, downloader, transcription
from .youtube_search import search_category_exhaust
from .downloader import DownloadError
from .gates import (
    audio_quality,
    duplicate,
    final_scoring,
    folk_relevance,
    integrity,
    metadata_relevance,
    music_speech,
    vad,
)

_log = get_logger("orchestrator")


class Pipeline:
    """Owns shared resources and runs candidates through the fixed gate chain."""

    def __init__(self, cfg: PipelineConfig, taxonomy) -> None:
        self.cfg = cfg
        self.taxonomy = taxonomy
        self.models = ModelManager(cfg)
        self.storage = Storage(cfg)
        self.runlog = RunLog(cfg.run_log_path)
        self.fingerprints = FingerprintStore(cfg.fingerprint_db_path)
        self.staging_dir = cfg.staging_dir
        os.makedirs(self.staging_dir, exist_ok=True)

        w = cfg.worker_threads
        self._quality_sema = threading.Semaphore(int(w.get("quality", 2)))
        self._inference_sema = threading.Semaphore(int(w.get("inference", 1)))
        self._transcribe_sema = threading.Semaphore(int(w.get("transcription", 1)))

    # ----------------------------------------------------------------- utils
    def _discard_staged(self, candidate: Candidate) -> None:
        path = candidate.audio_path
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    def _finalize_decision(self, candidate: Candidate) -> Decision:
        if candidate.decision == Decision.REJECT:
            return Decision.REJECT
        if candidate.decision == Decision.REVIEW:
            return Decision.REVIEW
        return Decision.ACCEPT

    def _log_result(self, candidate: Candidate, decision: Decision) -> None:
        self.runlog.append(
            candidate.recording_id, candidate.category_slug, decision.value,
            rejection_code=(candidate.rejection_code.value if candidate.rejection_code else None),
            final_score=candidate.final_score, title=candidate.title,
        )

    # ------------------------------------------------------- per-candidate run
    def process_candidate(self, candidate: Candidate, category: Category) -> Decision:
        """Run one candidate through the full chain. Never raises for a single
        file -- always resolves to a Decision and records it."""
        rid = candidate.recording_id
        try:
            # ---- resume: skip already-processed / already-stored ----
            if self.runlog.already_processed(rid):
                prev = self.runlog.decision_of(rid)
                # accepted-but-transcription-failed is handled by _retry path;
                # here we simply skip anything already logged.
                _log.info("skip %s (already processed: %s)", rid, prev)
                return Decision(prev) if prev in Decision._value2member_map_ else Decision.PENDING
            if (self.storage.output_exists(rid, category.folder_slug, Decision.ACCEPT)
                    or self.storage.output_exists(rid, category.folder_slug, Decision.REVIEW)):
                _log.info("skip %s (outputs already exist at target)", rid)
                return Decision.ACCEPT

            # ---- Stage 2: Metadata Relevance Gate (pre-download) ----
            candidate.record(metadata_relevance.evaluate(candidate, category, self.cfg))
            if candidate.decision == Decision.REJECT:
                return self._conclude(candidate, category)

            # ---- Stage 2.5: Pre-download Duration Check ----
            # candidate.duration_sec comes from YouTube search metadata -- no download needed.
            # Reject immediately if the video is already outside bounds, saving all the
            # bandwidth of downloading 200-600 MB compilations that we know will be rejected.
            if candidate.duration_sec and candidate.duration_sec > 0:
                _audio_cfg = self.cfg.raw["audio"]
                _min_dur = float(_audio_cfg.get("minimum_duration_sec", 10))
                _max_dur = float(_audio_cfg.get("maximum_duration_sec", 300))
                if candidate.duration_sec > _max_dur:
                    _log.info(
                        "pre-download REJECT %s [TOO_LONG] duration=%.0fs > max=%.0fs | %s",
                        rid, candidate.duration_sec, _max_dur, candidate.title[:60]
                    )
                    candidate.decision = Decision.REJECT
                    candidate.rejection_code = RejectionCode.TOO_LONG
                    return self._conclude(candidate, category)
                if candidate.duration_sec < _min_dur:
                    _log.info(
                        "pre-download REJECT %s [TOO_SHORT] duration=%.0fs < min=%.0fs | %s",
                        rid, candidate.duration_sec, _min_dur, candidate.title[:60]
                    )
                    candidate.decision = Decision.REJECT
                    candidate.rejection_code = RejectionCode.TOO_SHORT
                    return self._conclude(candidate, category)

            # ---- Stage 3: Download Full Audio ----
            try:
                candidate.audio_path = downloader.download_audio(
                    candidate, self.cfg, self.staging_dir)
            except DownloadError as exc:
                _log.warning("download failed %s: %s", rid, exc)
                candidate.decision = Decision.REJECT
                candidate.rejection_code = RejectionCode.DOWNLOAD_FAILED
                return self._conclude(candidate, category)

            # ---- Stage 4: Integrity Gate ----
            candidate.record(integrity.evaluate(candidate, self.cfg))
            if candidate.decision == Decision.REJECT:
                return self._conclude(candidate, category)

            # ---- Stage 4b: Audio Chunking (if long video) ----
            sub_candidates = chunker.split_audio_if_needed(candidate, self.cfg)
            if len(sub_candidates) > 1:
                self._discard_staged(candidate)
                results = []
                for sub_c in sub_candidates:
                    results.append(self._process_single_candidate(sub_c, category))
                final_dec = Decision.ACCEPT if any(d == Decision.ACCEPT for d in results) else Decision.REJECT
                self._log_result(candidate, final_dec)
                return final_dec

            return self._process_single_candidate(candidate, category)

        except Exception as exc:  # noqa: BLE001 - hard safety net; never kill batch
            _log.exception("unexpected error processing %s: %r", rid, exc)
            candidate.decision = Decision.REJECT
            candidate.rejection_code = candidate.rejection_code or RejectionCode.DOWNLOAD_FAILED
            self._discard_staged(candidate)
            decision = Decision.REJECT
            self._log_result(candidate, decision)
            return decision

    def _process_single_candidate(self, candidate: Candidate, category: Category) -> Decision:
        # ---- Stage 5: Strict Audio Quality Gate (highest priority) ----
        with self._quality_sema:
            q = audio_quality.evaluate(candidate, self.cfg)
        candidate.record(q)
        if candidate.decision == Decision.REJECT:
            return self._conclude(candidate, category)

        # Optimization: skip ML stages if quality is below minimum
        min_q = float(self.cfg.raw["quality_gate"]["minimum_quality_score"])
        quality_below_min = (q.score is not None and q.score < min_q)

        if not quality_below_min:
            # ---- Stages 6-8: ML inference (serialized, CPU-only) ----
            with self._inference_sema:
                candidate.record(music_speech.evaluate(candidate, self.models, self.cfg))
                candidate.record(vad.evaluate(candidate, self.models, self.cfg))
                folk = folk_relevance.evaluate(candidate, category, self.models, self.cfg)
            candidate.record(folk)
            if candidate.decision == Decision.REJECT:
                return self._conclude(candidate, category)

            # ---- Stage 9: Duplicate Detection ----
            candidate.record(duplicate.evaluate(candidate, self.fingerprints, self.cfg))
            if candidate.decision == Decision.REJECT:
                return self._conclude(candidate, category)

        # ---- Stage 10: Weighted Final Decision ----
        candidate.record(final_scoring.evaluate(candidate, category, self.cfg))
        return self._conclude(candidate, category)

    def _conclude(self, candidate: Candidate, category: Category) -> Decision:
        """Apply the decision: store (ACCEPT/REVIEW) or discard (REJECT), run
        transcription for ACCEPT, then record the audit line exactly once."""
        decision = self._finalize_decision(candidate)

        if decision == Decision.REJECT:
            self._discard_staged(candidate)
            _log.info("REJECT %s [%s] %s", candidate.recording_id,
                      candidate.rejection_code.value if candidate.rejection_code else "?",
                      candidate.title[:60])
            self._log_result(candidate, decision)
            return decision

        if decision == Decision.REVIEW:
            # Store REVIEW songs in the same folder as ACCEPT (not the review/ subfolder).
            # This ensures they appear alongside accepted songs in GCS for easy review.
            try:
                self.storage.store(candidate.recording_id, category.folder_slug,
                                   Decision.ACCEPT, candidate.audio_path, None,
                                   title=candidate.title, channel=candidate.channel,
                                   source_url=candidate.youtube_url)
            except Exception as exc:  # noqa: BLE001
                _log.error("failed to store REVIEW %s: %r", candidate.recording_id, exc)
            self._discard_staged(candidate)
            _log.info("REVIEW (stored to main folder) %s %s",
                      candidate.recording_id, candidate.title[:60])
            self._log_result(candidate, decision)
            return decision

        # ---- ACCEPT: transcribe (Stage 11-12) then store audio+txt (Stage 13) ----
        transcript_text: Optional[str] = None
        try:
            with self._transcribe_sema:
                transcript_text = transcription.transcribe_full(
                    candidate.audio_path, self.models, self.cfg)
        except Exception as exc:  # noqa: BLE001 - failure must NOT undo ACCEPT
            _log.error("transcription failed for ACCEPT %s (keeping accept, will retry): %r",
                       candidate.recording_id, exc)
            transcript_text = None

        try:
            self.storage.store(
                candidate.recording_id, category.folder_slug,
                Decision.ACCEPT, candidate.audio_path, transcript_text,
                title=candidate.title, channel=candidate.channel, source_url=candidate.youtube_url
            )
        except Exception as exc:  # noqa: BLE001
            _log.error("failed to store ACCEPT %s: %r", candidate.recording_id, exc)

        self._discard_staged(candidate)
        _log.info("ACCEPT %s (score=%.2f) %s", candidate.recording_id,
                  candidate.final_score or 0.0, candidate.title[:60])
        # Note: if transcript_text is None the audit still logs ACCEPT; a separate
        # retry pass can regenerate the .txt without re-deciding.
        self._log_result(candidate, decision)
        return decision

    # ---------------------------------------------------------------- category
    def run_category(self, category: Category, *, limit: Optional[int] = None) -> dict[str, int]:
        """Exhaust-search + process one category. Keeps searching YouTube until
        no new videos are found (ytsearchall, channel expansion, multi-round).
        Already-processed IDs from the run-log are excluded from discovery so
        they are never re-downloaded."""
        queries = generate_queries(category)
        _log.info("category %s: %d queries", category.folder_slug, len(queries))

        # pre-seed the skip set with everything this run-log already knows about
        already_done = {rid for rid, _ in
                        [(k, v) for k, v in self.runlog._seen.items()]}
        candidates = search_category_exhaust(category, queries, self.cfg,
                                             seen_ids=already_done)
        if limit:
            candidates = candidates[:limit]
        _log.info("category %s: %d candidates to process", category.folder_slug, len(candidates))
        return self._process_batch(candidates, category)

    def process_urls(self, urls: list[str], category: Category) -> dict[str, int]:
        """Process explicit YouTube URLs/ids under a category (bypasses search)."""
        candidates = []
        for u in urls:
            vid = _extract_video_id(u)
            candidates.append(Candidate(recording_id=vid, category_slug=category.folder_slug,
                                        title="", description="", url=u))
        return self._process_batch(candidates, category)

    def _process_batch(self, candidates: list[Candidate], category: Category) -> dict[str, int]:
        counts = {"ACCEPT": 0, "REVIEW": 0, "REJECT": 0}
        max_workers = int(self.cfg.raw["workers"]["download"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.process_candidate, c, category): c for c in candidates}
            for fut in concurrent.futures.as_completed(futures):
                c = futures[fut]
                try:
                    decision = fut.result()
                except Exception as exc:  # noqa: BLE001 - safety net
                    _log.exception("worker crashed on %s: %r", c.recording_id, exc)
                    decision = Decision.REJECT
                if decision.value in counts:
                    counts[decision.value] += 1
        _log.info("category %s done: %s", category.folder_slug, counts)
        return counts


def _extract_video_id(url_or_id: str) -> str:
    import re
    m = re.search(r"(?:v=|youtu\.be/|/shorts/|/watch/)([A-Za-z0-9_-]{11})", url_or_id)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    return url_or_id  # fall back; yt-dlp will validate


def build_pipeline(pipeline_config_path: Optional[str] = None,
                   taxonomy_path: Optional[str] = None) -> Pipeline:
    cfg = load_pipeline_config(pipeline_config_path)
    taxonomy = load_taxonomy(taxonomy_path)
    log_cfg = cfg.raw["logging"]
    setup_logging(level=log_cfg["level"], log_dir=cfg.abspath(log_cfg["dir"]),
                  log_file=log_cfg["file"])
    return Pipeline(cfg, taxonomy)


def main() -> int:
    ap = argparse.ArgumentParser(description="Tamil Nadu Folk Song Pipeline")
    ap.add_argument("--categories", help="comma-separated folder slugs (default: all)")
    ap.add_argument("--limit", type=int, default=None,
                    help="max candidates per category")
    ap.add_argument("--urls", help="comma-separated YouTube URLs/ids to process directly")
    ap.add_argument("--slug", help="category slug for --urls mode")
    ap.add_argument("--config", default=None)
    ap.add_argument("--taxonomy", default=None)
    args = ap.parse_args()

    pipe = build_pipeline(args.config, args.taxonomy)

    if args.urls:
        if not args.slug:
            ap.error("--urls requires --slug")
        category = pipe.taxonomy.get(args.slug)
        urls = [u.strip() for u in args.urls.split(",") if u.strip()]
        counts = pipe.process_urls(urls, category)
        _log.info("URL run totals: %s", counts)
        return 0

    if args.categories:
        slugs = [s.strip() for s in args.categories.split(",") if s.strip()]
        categories = [pipe.taxonomy.get(s) for s in slugs]
    else:
        categories = pipe.taxonomy.categories

    totals = {"ACCEPT": 0, "REVIEW": 0, "REJECT": 0}
    for category in categories:
        counts = pipe.run_category(category, limit=args.limit)
        for k in totals:
            totals[k] += counts.get(k, 0)
    _log.info("RUN TOTALS: %s | run-log counts: %s", totals, pipe.runlog.counts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
