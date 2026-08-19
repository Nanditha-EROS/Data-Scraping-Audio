"""Tenacity-based retry helpers built from the `retry` config section.

Every network call (search, download, model download) wraps itself with
``retrying(cfg)`` so failures use exponential backoff with jitter and a hard
max-attempt cutoff -- never retry forever, never crash the whole run on one bad
video. Permanent errors can be signalled by raising ``PermanentError`` which is
not retried.
"""
from __future__ import annotations

from typing import Any, Callable, Type

# pyrefly: ignore [missing-import]
from tenacity import (
    Retrying,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .logging_utils import get_logger

_log = get_logger("retry")


class PermanentError(Exception):
    """Raise to signal a non-retryable failure (stops retrying immediately)."""


def make_retrying(cfg: Any, *, description: str = "",
                  no_retry_on: tuple[Type[BaseException], ...] = ()) -> Retrying:
    """Build a tenacity ``Retrying`` controller from the retry config.

    Args:
        cfg: PipelineConfig (reads the `retry` section).
        description: label used in retry log lines.
        no_retry_on: extra exception types that must not be retried.

    Returns:
        A configured ``Retrying`` you can call as ``ret(fn, *args, **kwargs)``.
    """
    r = cfg.raw["retry"]
    exc_types: tuple[Type[BaseException], ...] = (PermanentError,) + tuple(no_retry_on)

    def _before_sleep(state: Any) -> None:
        _log.warning("retry %s: attempt %d failed (%s); backing off",
                     description or "call", state.attempt_number,
                     repr(state.outcome.exception()) if state.outcome else "?")

    return Retrying(
        stop=stop_after_attempt(int(r["max_attempts"])),
        wait=wait_exponential_jitter(
            initial=float(r["initial_backoff_sec"]),
            max=float(r["max_backoff_sec"]),
            exp_base=float(r["backoff_multiplier"]),
            jitter=float(r["jitter_sec"]),
        ),
        retry=retry_if_not_exception_type(exc_types),
        before_sleep=_before_sleep,
        reraise=True,
    )


def with_retry(cfg: Any, fn: Callable, *args: Any, description: str = "",
               no_retry_on: tuple[Type[BaseException], ...] = (), **kwargs: Any) -> Any:
    """Convenience: run ``fn(*args, **kwargs)`` under a fresh retry controller."""
    controller = make_retrying(cfg, description=description, no_retry_on=no_retry_on)
    return controller(fn, *args, **kwargs)
