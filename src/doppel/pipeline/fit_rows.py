"""Shared `--fit-rows` / auto-cap logic for gen and fit."""

from __future__ import annotations

from collections.abc import Callable

AUTO_FIT_TRIGGER_ROWS = 100_000
AUTO_FIT_CAP = 100_000
AUTO_FIT_MULTIPLIER = 5


def auto_fit_rows(
    user_value: int | None,
    source_rows: int,
    requested_rows: int,
    *,
    notify: Callable[[str], None] | None = None,
) -> int | None:
    """Pick an effective fit-row sample size.

    - User passed ``0``: opt out of capping; fit on the full source.
    - User explicitly passed ``N`` (N >= 1): honour it verbatim.
    - User omitted the flag AND source <= trigger (100k rows): no sampling.
    - User omitted the flag AND source > trigger: cap at ``min(requested_rows*5, 100k)``
      and optionally notify the user.
    """
    if user_value == 0:
        return None
    if user_value is not None:
        return user_value
    if source_rows <= AUTO_FIT_TRIGGER_ROWS:
        return None
    cap = min(requested_rows * AUTO_FIT_MULTIPLIER, AUTO_FIT_CAP)
    if notify is not None:
        notify(
            f"source has {source_rows:,} rows; sampling {cap:,} (deterministic) for fit. "
            "pass `--fit-rows 0` to disable, or `--fit-rows N` to set explicitly."
        )
    return cap
