"""Stages 12-13 -- lyrics .txt generation + upload/store (local or GCS).

Layout (exact, case-sensitive):
    <root>/<state>/<category>/<recording_id>.wav
    <root>/<state>/<category>/<recording_id>.txt
REVIEW items go under a `review/` segment:
    <root>/<state>/review/<category>/<recording_id>.wav[.txt]

Only ACCEPT (Gold Data) and REVIEW are stored. REJECT is never stored.

Resume (no database): ``output_exists`` checks whether the target outputs already
exist, so a candidate whose files are present is skipped without reprocessing.

Backends:
    local -> filesystem copy/write under the (project-relative or absolute) root.
    gcs   -> `gcloud storage` CLI; the first path segment of `root` is the bucket.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

from .config import PipelineConfig
from .logging_utils import get_logger
from .models import Decision

_log = get_logger("storage")


class StorageError(RuntimeError):
    """Raised when a store/upload operation fails."""


def format_lyrics_file(title: str, channel: str, category_slug: str,
                        source_url: str, raw_text: str) -> str:
    """Format lyrics into structured metadata header + [Verse] layout."""
    import re
    lines = []
    lines.append(f"Title: {title or 'Tamil Folk Song'}")
    lines.append(f"Artist: {channel or 'Unknown Artist'}")
    lines.append("Language: Tamil")
    lines.append(f"Source: {source_url or 'YouTube'}")
    lines.append(f"Category: {category_slug}")
    lines.append("")

    text = (raw_text or "").strip()
    if not text:
        lines.append("[Verse 1]")
        lines.append("[Instrumental / No Speech Detected]")
        return "\n".join(lines)

    # Split into clean sentence blocks for verses
    sentences = [s.strip() for s in re.split(r'[\n.!?|]+', text) if s.strip()]
    if not sentences:
        sentences = [text]

    verse_count = 1
    lines.append(f"[Verse {verse_count}]")
    sentence_in_verse = 0
    for s in sentences:
        lines.append(s)
        sentence_in_verse += 1
        if sentence_in_verse >= 3 and s != sentences[-1]:
            verse_count += 1
            lines.append("")
            lines.append(f"[Verse {verse_count}]")
            sentence_in_verse = 0

    return "\n".join(lines)


class Storage:
    """Local- or GCS-backed storage for accepted/review audio + lyrics."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        s = cfg.raw["storage"]
        self.backend = str(s["backend"]).lower()
        self.root = str(s["root"]).rstrip("/")
        self.state = cfg.state_name
        self.review_subdir = str(s.get("review_subdir", "review"))
        self.save_txt = bool(s.get("save_lyrics_txt", True))
        self.gcloud = s.get("gcloud_path") or "gcloud"
        if self.backend not in ("local", "gcs"):
            raise StorageError(f"unknown storage.backend: {self.backend}")

    # -- path building ------------------------------------------------------
    def _rel_dir(self, category_slug: str, review: bool) -> str:
        if review:
            return f"{self.state}/{self.review_subdir}/{category_slug}"
        return f"{self.state}/{category_slug}"

    def _local_dir(self, category_slug: str, review: bool) -> str:
        base = self.root if os.path.isabs(self.root) else self.cfg.abspath(self.root)
        return os.path.join(base, *self._rel_dir(category_slug, review).split("/"))

    def _gcs_uri(self, category_slug: str, review: bool, filename: str) -> str:
        parts = self.root.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        rel = self._rel_dir(category_slug, review)
        key = "/".join(p for p in (prefix, rel, filename) if p)
        return f"gs://{bucket}/{key}"

    # -- existence (resume) -------------------------------------------------
    def output_exists(self, recording_id: str, category_slug: str,
                      decision: Decision) -> bool:
        """True if this candidate's outputs already exist at the target path.

        For ACCEPT, requires the .wav (and .txt if save_lyrics_txt). For REVIEW,
        requires the .wav under the review path.
        """
        review = decision == Decision.REVIEW
        wav = f"{recording_id}.wav"
        txt = f"{recording_id}.txt"
        need_txt = self.save_txt and decision == Decision.ACCEPT
        if self.backend == "local":
            d = self._local_dir(category_slug, review)
            if not os.path.exists(os.path.join(d, wav)):
                return False
            if need_txt and not os.path.exists(os.path.join(d, txt)):
                return False
            return True
        # gcs
        if not self._gcs_exists(self._gcs_uri(category_slug, review, wav)):
            return False
        if need_txt and not self._gcs_exists(self._gcs_uri(category_slug, review, txt)):
            return False
        return True

    def _gcs_exists(self, uri: str) -> bool:
        proc = subprocess.run([self.gcloud, "storage", "ls", uri],
                              capture_output=True, text=True)
        return proc.returncode == 0 and bool(proc.stdout.strip())

    def _gcs_cp(self, local_path: str, uri: str) -> None:
        proc = subprocess.run([self.gcloud, "storage", "cp", local_path, uri],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise StorageError(f"gcloud cp failed: {proc.stderr.strip()[:300]}")

    # -- store --------------------------------------------------------------
    def store(self, recording_id: str, category_slug: str, decision: Decision,
              wav_path: str, transcript_text: Optional[str],
              title: str = "", channel: str = "", source_url: str = "") -> dict[str, str]:
        """Persist the ORIGINAL full audio (+ transcript for ACCEPT) to storage.

        Args:
            recording_id: youtube id (file stem).
            category_slug: taxonomy folder slug.
            decision: ACCEPT or REVIEW (REJECT must never be passed here).
            wav_path: local path to the full-length WAV to store.
            transcript_text: lyrics text for ACCEPT (None for REVIEW / instrumental).
            title: title of the song/video for metadata header.
            channel: channel/artist name for metadata header.
            source_url: full URL for metadata header.

        Returns:
            Dict with 'audio' and optionally 'transcript' destination locations.
        """
        if decision == Decision.REJECT:
            raise StorageError("REJECT must never be stored")
        review = decision == Decision.REVIEW
        wav_name = f"{recording_id}.wav"
        txt_name = f"{recording_id}.txt"
        write_txt = (self.save_txt and decision == Decision.ACCEPT
                     and transcript_text is not None)

        formatted_content = format_lyrics_file(
            title=title,
            channel=channel,
            category_slug=category_slug,
            source_url=source_url or f"https://www.youtube.com/watch?v={recording_id}",
            raw_text=transcript_text or ""
        ) if write_txt else ""

        if self.backend == "local":
            d = self._local_dir(category_slug, review)
            os.makedirs(d, exist_ok=True)
            dst_wav = os.path.join(d, wav_name)
            shutil.copy2(wav_path, dst_wav)
            out = {"audio": dst_wav}
            if write_txt:
                dst_txt = os.path.join(d, txt_name)
                with open(dst_txt, "w", encoding="utf-8") as f:
                    f.write(formatted_content)
                out["transcript"] = dst_txt
            _log.info("stored %s -> %s", recording_id, d)
            return out

        # gcs: upload wav (+ txt written to a temp file first, then removed so
        # nothing is left on local disk)
        wav_uri = self._gcs_uri(category_slug, review, wav_name)
        self._gcs_cp(wav_path, wav_uri)
        out = {"audio": wav_uri}
        if write_txt:
            tmp_txt = wav_path + ".txt"
            try:
                with open(tmp_txt, "w", encoding="utf-8") as f:
                    f.write(formatted_content)
                txt_uri = self._gcs_uri(category_slug, review, txt_name)
                self._gcs_cp(tmp_txt, txt_uri)
                out["transcript"] = txt_uri
            finally:
                if os.path.exists(tmp_txt):
                    try:
                        os.remove(tmp_txt)
                    except OSError:
                        pass
        _log.info("uploaded %s -> %s", recording_id, wav_uri)
        return out
