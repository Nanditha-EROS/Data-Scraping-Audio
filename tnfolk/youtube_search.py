"""Stage 1b -- YouTube Search (yt-dlp) -- EXHAUST MODE.

Searches YouTube until no new videos are found for a category. Uses ytsearchall
with ALL query variants (Tamil/English/Tanglish/qualified), runs multiple rounds,
and expands the top-contributing channels to discover more. Stops only when N
consecutive rounds return zero new videos or the hard max_videos cap is hit.

Every search call is wrapped in tenacity retry with exponential backoff and a
hard cutoff. One failing query never crashes the run -- it is logged and skipped.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from .config import Category, PipelineConfig
from .logging_utils import get_logger
from .models import Candidate
from .retry import make_retrying

_log = get_logger("search")


@dataclass
class SearchHit:
    recording_id: str
    title: str
    channel: str
    channel_id: str
    duration_sec: float
    description: str
    url: str


def _yt_common_opts(cfg: PipelineConfig, max_items: Optional[int] = None) -> dict[str, Any]:
    s = cfg.raw.get("search", {})
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "skip_download": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"]
            }
        },
    }
    sleep_req = float(s.get("sleep_between_requests_sec", 1.0))
    if sleep_req > 0:
        opts["sleep_interval"] = sleep_req

    if max_items and max_items > 0:
        opts["playlistend"] = max_items

    # Prefer cookies from search section; fall back to download section
    cookies = s.get("cookies_from_browser") or cfg.raw["download"].get("cookies_from_browser")
    if cookies:
        opts["cookiesfrombrowser"] = (cookies,)
    return opts


def _flat_search(query_url: str, cfg: PipelineConfig) -> list[SearchHit]:
    """Run a single yt-dlp flat extraction (ytsearch / ytsearchall / channel URL)."""
    import yt_dlp

    s = cfg.raw.get("search", {})
    max_channel_vids = int(s.get("max_channel_videos", 300)) if query_url.startswith("http") else None

    opts = {**_yt_common_opts(cfg, max_items=max_channel_vids), "extract_flat": True}
    hits: list[SearchHit] = []
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(query_url, download=False)
    entries = (info or {}).get("entries") or []
    for e in entries:
        if not e:
            continue
        vid = e.get("id")
        if not vid:
            continue
        hits.append(SearchHit(
            recording_id=vid,
            title=e.get("title") or "",
            channel=e.get("channel") or e.get("uploader") or "",
            channel_id=e.get("channel_id") or "",
            duration_sec=float(e.get("duration") or 0.0),
            description=e.get("description") or "",
            url=e.get("url") or f"https://www.youtube.com/watch?v={vid}",
        ))
    return hits


def _top_channel_urls(hits: list[SearchHit], limit: int) -> list[str]:
    """Pick the channels that contributed the most results (likely folk uploaders)."""
    counts: dict[str, int] = {}
    for h in hits:
        cid = h.channel_id.strip()
        if cid.startswith("UC"):
            counts[cid] = counts.get(cid, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [f"https://www.youtube.com/channel/{cid}/videos" for cid, _ in ranked[:limit]]


def search_category_exhaust(category: Category, queries: list[str],
                            cfg: PipelineConfig,
                            *, seen_ids: set[str] | None = None) -> list[Candidate]:
    """Exhaust-mode: search ALL query variants with ytsearchall, do multiple
    rounds with channel expansion, stop only when YouTube has nothing new.

    Args:
        category: taxonomy entry (provides folder_slug for the candidates).
        queries: ALL query strings (no cap) from the query generator.
        cfg: pipeline config (reads `search` section).
        seen_ids: ids to exclude (already discovered / run-log).

    Returns:
        De-duplicated list of Candidate objects (metadata only; not downloaded).
    """
    s = cfg.raw["search"]
    max_rounds = int(s.get("exhaust_max_rounds", 5))
    confirm_empty = int(s.get("exhaust_confirm_empty", 2))
    max_videos = int(s.get("max_videos", 0))  # 0 = unlimited
    channel_expand = int(s.get("exhaust_channel_expand", 15))
    sleep_between_queries = float(s.get("sleep_between_searches_sec", 2))
    sleep_between_rounds = float(s.get("sleep_between_rounds_sec", 5))
    retrying = make_retrying(cfg, description="youtube_search")

    seen = set(seen_ids or ())
    all_hits: list[SearchHit] = []
    empty_streak = 0

    cap_label = str(max_videos) if max_videos > 0 else "unlimited"
    _log.info("EXHAUST MODE for %s — %d queries, max_rounds=%d, confirm_empty=%d, max_videos=%s",
              category.folder_slug, len(queries), max_rounds, confirm_empty, cap_label)

    for round_idx in range(1, max_rounds + 1):
        before = len(seen)

        # build sources: ytsearchall for each query + channel URLs from round 2+
        sources: list[tuple[str, str]] = []
        # ytsearch100 = top 100 results per query (~5 pages of 20).
        # Staying under 6 pages avoids YouTube's deep-pagination 403 rate limit.
        # 12 query variants * 100 results = up to 1200 unique candidates per round.
        max_per_q = int(s.get("max_per_query", 100))
        for q in queries:
            sources.append((f"ytsearch{max_per_q}:{q}", f"search: {q}"))
        if round_idx > 1 and channel_expand > 0:
            for url in _top_channel_urls(all_hits, channel_expand):
                sources.append((url, f"channel: {url}"))

        for source_url, label in sources:
            if max_videos > 0 and len(all_hits) >= max_videos:
                break
            try:
                hits = retrying(_flat_search, source_url, cfg)
            except Exception as exc:  # noqa: BLE001
                _log.warning("search failed for %s (category=%s): %r",
                             label, category.folder_slug, exc)
                continue
            new = 0
            for h in hits:
                if h.recording_id in seen:
                    continue
                seen.add(h.recording_id)
                all_hits.append(h)
                new += 1
                if max_videos > 0 and len(all_hits) >= max_videos:
                    break
            if new > 0:
                _log.info("round %d %-30s %s -> %d hits (%d new, %d total)",
                          round_idx, category.folder_slug, label[:50], len(hits), new, len(seen))
            if max_videos > 0 and len(all_hits) >= max_videos:
                break
            if sleep_between_queries > 0:
                time.sleep(sleep_between_queries)

        newly = len(seen) - before
        _log.info("round %d/%d for %s: +%d new (total %d)",
                  round_idx, max_rounds, category.folder_slug, newly, len(seen))

        if max_videos > 0 and len(all_hits) >= max_videos:
            _log.info("hit max_videos=%d — stopping exhaust discovery", max_videos)
            break

        if newly == 0:
            empty_streak += 1
            _log.info("no new videos (%d/%d empty rounds)", empty_streak, confirm_empty)
            if empty_streak >= confirm_empty:
                _log.info("EXHAUSTED %s — no new videos found after %d consecutive empty rounds",
                          category.folder_slug, confirm_empty)
                break
        else:
            empty_streak = 0

        if round_idx < max_rounds and sleep_between_rounds > 0:
            time.sleep(sleep_between_rounds)

    _log.info("exhaust discovery complete for %s: %d unique videos", category.folder_slug, len(all_hits))

    candidates = [
        Candidate(
            recording_id=h.recording_id,
            category_slug=category.folder_slug,
            title=h.title,
            description=h.description,
            channel=h.channel,
            url=h.url,
            duration_sec=h.duration_sec,
        )
        for h in all_hits
    ]
    return candidates
