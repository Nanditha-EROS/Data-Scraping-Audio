"""File-based resume + audit, replacing any database.

Two lightweight, append-only JSONL files (NOT a database):

    run_log.jsonl      one line per processed video id with its final decision.
                       Primary purpose is auditing. It is also read once at
                       startup into an in-memory skip-set so already-processed
                       ids (including REJECTs, which are never stored) are not
                       redone on the next run.

    fingerprints.jsonl one line per stored recording's Chromaprint fingerprint,
                       used by the duplicate-detection gate across runs.

Authoritative resume for *stored* items is still output-file existence (see
storage.output_exists); the run-log skip-set is an efficiency layer so we do not
re-download/re-analyse rejected candidates every run.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional


class RunLog:
    """Append-only JSONL audit log + in-memory skip-set of processed ids."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._seen: dict[str, str] = {}   # recording_id -> decision
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = rec.get("recording_id")
                if rid:
                    self._seen[rid] = rec.get("decision", "")

    def already_processed(self, recording_id: str) -> bool:
        with self._lock:
            return recording_id in self._seen

    def decision_of(self, recording_id: str) -> Optional[str]:
        with self._lock:
            return self._seen.get(recording_id)

    def append(self, recording_id: str, category_slug: str, decision: str, *,
               rejection_code: str | None = None, final_score: float | None = None,
               title: str = "", extra: dict[str, Any] | None = None) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "recording_id": recording_id,
            "category_slug": category_slug,
            "title": title,
            "decision": decision,
            "rejection_code": rejection_code,
            "final_score": final_score,
        }
        if extra:
            record.update(extra)
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._seen[recording_id] = decision

    def counts(self) -> dict[str, int]:
        with self._lock:
            out: dict[str, int] = {}
            for dec in self._seen.values():
                out[dec] = out.get(dec, 0) + 1
            return out


class FingerprintStore:
    """Append-only JSONL of recording fingerprints for cross-run dedupe."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._lock = threading.Lock()
        # recording_id -> {"duration": float, "fingerprint": list[int]}
        self._items: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = rec.get("recording_id")
                if rid:
                    self._items[rid] = {
                        "duration": rec.get("duration", 0.0),
                        "fingerprint": rec.get("fingerprint", []),
                    }

    def add(self, recording_id: str, duration: float, fingerprint: list[int]) -> None:
        with self._lock:
            if recording_id in self._items:
                return
            self._items[recording_id] = {"duration": duration, "fingerprint": fingerprint}
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "recording_id": recording_id,
                    "duration": duration,
                    "fingerprint": fingerprint,
                }, ensure_ascii=False) + "\n")

    def items(self, exclude: str | None = None) -> list[tuple[str, dict[str, Any]]]:
        with self._lock:
            return [(rid, v) for rid, v in self._items.items() if rid != exclude]

    def __contains__(self, recording_id: str) -> bool:
        with self._lock:
            return recording_id in self._items
