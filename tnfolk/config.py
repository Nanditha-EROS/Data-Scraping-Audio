"""Configuration loading and validation.

Loads config/pipeline_config.yaml and config/folk_categories.yaml into typed,
attribute-accessible objects. Fails loudly on missing/invalid values so
misconfiguration is caught at startup, not mid-run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_PIPELINE_CONFIG = os.path.join(PROJECT_ROOT, "config", "pipeline_config.yaml")
DEFAULT_TAXONOMY_CONFIG = os.path.join(PROJECT_ROOT, "config", "folk_categories.yaml")


class ConfigError(RuntimeError):
    """Raised when configuration is missing or structurally invalid."""


class _Namespace(dict):
    """Dict that also supports attribute access and nested wrapping."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, _Namespace):
            value = _Namespace(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return _Namespace({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


@dataclass(frozen=True)
class Category:
    """One taxonomy entry from folk_categories.yaml."""

    id: int
    folder_slug: str
    tamil_name: str
    english_name: str
    family: str
    search_queries: dict[str, list[str]]
    reference_concepts: list[str]
    gate_overrides: dict[str, Any] = field(default_factory=dict)

    def all_queries(self) -> list[str]:
        """Flatten all query variants (Tamil/English/Tanglish/qualified)."""
        out: list[str] = []
        for variant in ("tamil", "english", "tanglish", "qualified"):
            out.extend(self.search_queries.get(variant, []))
        return out

    def keywords(self) -> list[str]:
        """Terms used by the cheap metadata gate for keyword/fuzzy matching."""
        terms = [self.tamil_name, self.english_name]
        # english_name may be "Oosal / Oonjal song" -> split into parts
        for part in self.english_name.replace("/", ",").split(","):
            part = part.strip()
            if part:
                terms.append(part)
        # include each Tamil query root
        terms.extend(self.search_queries.get("tamil", []))
        # de-dupe preserving order
        seen, uniq = set(), []
        for t in terms:
            t = t.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                uniq.append(t)
        return uniq


class PipelineConfig:
    """Typed wrapper over pipeline_config.yaml (attribute access via ``.raw``)."""

    def __init__(self, data: dict[str, Any], path: str) -> None:
        self.path = path
        self.raw = _wrap(data)
        self._validate()

    def _validate(self) -> None:
        required_top = [
            "state_name", "device", "storage", "run_log", "workers", "retry",
            "search", "download", "audio", "metadata_gate", "quality_gate",
            "classifier", "vad", "folk_relevance", "duplicate", "scoring",
            "transcription", "logging",
        ]
        missing = [k for k in required_top if k not in self.raw]
        if missing:
            raise ConfigError(f"pipeline_config.yaml missing sections: {missing}")

        q = self.raw["quality_gate"]
        if not (0.0 < float(q["minimum_quality_score"]) <= 1.0):
            raise ConfigError("quality_gate.minimum_quality_score must be in (0, 1]")

        s = self.raw["scoring"]
        if float(s["accept_threshold"]) <= float(s["review_threshold"]):
            raise ConfigError("scoring.accept_threshold must exceed review_threshold")
        weights = self.raw["scoring"]["weights"]
        wsum = sum(float(v) for v in weights.values())
        if abs(wsum - 1.0) > 1e-6:
            raise ConfigError(f"scoring.weights must sum to 1.0 (got {wsum})")

    # convenience accessors -------------------------------------------------
    @property
    def state_name(self) -> str:
        return self.raw["state_name"]

    @property
    def run_log_path(self) -> str:
        return self.abspath(str(self.raw["run_log"]["path"]))

    @property
    def fingerprint_db_path(self) -> str:
        return self.abspath(str(self.raw["fingerprint_store"]["path"]))

    @property
    def staging_dir(self) -> str:
        custom = self.raw.get("storage", {}).get("staging_dir")
        if custom:
            return self.abspath(str(custom))
        import tempfile
        return os.path.join(tempfile.gettempdir(), "tnfolk-staging")

    @property
    def worker_threads(self) -> dict[str, Any]:
        return self.raw.get("worker_threads") or self.raw.get("workers", {})

    def abspath(self, relative: str) -> str:
        """Resolve a config-relative path against the project root."""
        if os.path.isabs(relative):
            return relative
        return os.path.join(PROJECT_ROOT, relative)


class Taxonomy:
    """Typed wrapper over folk_categories.yaml."""

    def __init__(self, data: dict[str, Any], path: str) -> None:
        self.path = path
        self.state_name: str = data.get("state_name", "")
        raw_categories = data.get("categories") or []
        if not raw_categories:
            raise ConfigError(f"{path} has no categories")
        self.categories: list[Category] = [self._parse(c, path) for c in raw_categories]

        slugs = [c.folder_slug for c in self.categories]
        dupes = {s for s in slugs if slugs.count(s) > 1}
        if dupes:
            raise ConfigError(f"Duplicate folder_slug values in taxonomy: {dupes}")
        self._by_slug = {c.folder_slug: c for c in self.categories}

    @staticmethod
    def _parse(entry: dict[str, Any], path: str) -> Category:
        try:
            return Category(
                id=int(entry["id"]),
                folder_slug=str(entry["folder_slug"]),
                tamil_name=str(entry["tamil_name"]),
                english_name=str(entry["english_name"]),
                family=str(entry["family"]),
                search_queries=dict(entry.get("search_queries") or {}),
                reference_concepts=list(entry.get("reference_concepts") or []),
                gate_overrides=dict(entry.get("gate_overrides") or {}),
            )
        except KeyError as exc:
            raise ConfigError(f"Category in {path} missing field {exc}") from exc

    def get(self, slug: str) -> Category:
        try:
            return self._by_slug[slug]
        except KeyError as exc:
            raise ConfigError(f"Unknown category slug: {slug!r}") from exc

    def __len__(self) -> int:
        return len(self.categories)


def load_pipeline_config(path: str | None = None) -> PipelineConfig:
    path = path or DEFAULT_PIPELINE_CONFIG
    if not os.path.exists(path):
        raise ConfigError(f"pipeline config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return PipelineConfig(data, path)


def load_taxonomy(path: str | None = None) -> Taxonomy:
    path = path or DEFAULT_TAXONOMY_CONFIG
    if not os.path.exists(path):
        raise ConfigError(f"taxonomy config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Taxonomy(data, path)
