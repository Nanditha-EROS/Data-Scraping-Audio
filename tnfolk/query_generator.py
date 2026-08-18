"""Stage 1a -- Query Generator.

Builds ALL Tamil / English / Tanglish / qualified search-query variants for a
category from folk_categories.yaml (never hard-coded). For exhaust mode every
unique query is emitted so yt-dlp discovers as many videos as possible.
"""
from __future__ import annotations

import random
from typing import Iterable

from .config import Category

_VARIANT_ORDER = ("tamil", "english", "tanglish", "qualified")


def generate_queries(category: Category, *, seed: int | None = None) -> list[str]:
    """Return ALL de-duplicated search queries for a category (exhaust mode).

    Interleaves one query from each variant type round-robin (so searches
    always mix Tamil/English/Tanglish/qualified), then returns every unique
    query — no cap.

    Args:
        category: taxonomy entry providing the per-variant query templates.
        seed: optional RNG seed for reproducible ordering (tests).

    Returns:
        Ordered list of ALL unique query strings.
    """
    rng = random.Random(seed)

    buckets: dict[str, list[str]] = {}
    for v in _VARIANT_ORDER:
        items = list(category.search_queries.get(v, []))
        rng.shuffle(items)
        buckets[v] = items

    # interleave: one from each variant type per round
    interleaved: list[str] = []
    idx = 0
    while any(idx < len(buckets[v]) for v in _VARIANT_ORDER):
        for v in _VARIANT_ORDER:
            if idx < len(buckets[v]):
                interleaved.append(buckets[v][idx])
        idx += 1

    # de-dupe preserving order, NO cap
    seen: set[str] = set()
    out: list[str] = []
    for q in interleaved:
        key = q.strip().lower()
        if q.strip() and key not in seen:
            seen.add(key)
            out.append(q.strip())
    return out


def all_queries(categories: Iterable[Category], *, seed: int | None = None) -> dict[str, list[str]]:
    """Map each category slug -> its full query list."""
    return {
        c.folder_slug: generate_queries(c, seed=seed)
        for c in categories
    }
