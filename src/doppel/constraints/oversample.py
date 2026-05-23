"""Geometric oversample loop shared by reject-resample and multi-table where filtering."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def geometric_oversample(
    target: int,
    *,
    synthesize: Callable[[int], T],
    accept: Callable[[T], int],
    initial_factor: float = 1.5,
    max_factor: float = 4.0,
    on_iteration: Callable[[int, int, float], None] | None = None,
    exhausted_message: Callable[[int, int, float], str] | None = None,
) -> T:
    """Repeatedly synthesize batches until ``accept`` reports at least ``target`` kept rows.

    ``synthesize(batch_size)`` returns a batch; ``accept(batch)`` returns how many rows
    were kept from that batch (caller accumulates externally or returns cumulative).

    The loop uses geometric growth: ``batch_size = max(int(deficit * factor), 1)`` then
    ``factor *= 1.5`` until kept >= target or factor exceeds ``max_factor``.
    """
    kept = 0
    factor = initial_factor
    attempted = 0
    last: T | None = None

    while kept < target and factor <= max_factor + 1e-9:
        deficit = target - kept
        batch_size = max(int(deficit * factor), 1)
        batch = synthesize(batch_size)
        last = batch
        kept += accept(batch)
        attempted += batch_size
        if on_iteration is not None:
            on_iteration(batch_size, kept, factor)
        factor *= 1.5

    if kept < target:
        if exhausted_message is not None:
            raise ValueError(exhausted_message(target, attempted, factor))
        raise ValueError(
            f"could not reach {target} rows after {attempted} attempts "
            f"(oversample factor {factor:.1f}x)"
        )
    if last is None:
        raise RuntimeError("geometric_oversample finished without synthesizing")
    return last
