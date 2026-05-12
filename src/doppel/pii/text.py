"""Orchestration helpers: strip PII columns before fit, restore them post-sample."""

from __future__ import annotations

import polars as pl

from doppel.dataset import Table
from doppel.pii.detect import PIIDetection
from doppel.pii.fake import generate
from doppel.synth.seed import Rng


def strip(table: Table, detections: list[PIIDetection]) -> tuple[Table, list[str]]:
    """Return (table without PII columns, ordered list of original column names).

    The caller uses the original ordering to reindex the synthesized DataFrame after
    PII columns are regenerated.
    """
    if not detections:
        return table, [c.name for c in table.columns]
    pii_names = {d.name for d in detections}
    kept_columns = [c for c in table.columns if c.name not in pii_names]
    kept_data = table.data.drop(pii_names) if table.data is not None else None
    stripped = Table(
        name=table.name,
        columns=kept_columns,
        primary_key=table.primary_key,
        data=kept_data,
    )
    return stripped, [c.name for c in table.columns]


def restore(
    synthesized: pl.DataFrame,
    detections: list[PIIDetection],
    original_order: list[str],
    rng: Rng,
) -> pl.DataFrame:
    """Add fake PII columns back to a synthesized DataFrame and restore original column order."""
    out = synthesized
    for d in detections:
        out = out.with_columns(pl.Series(d.name, generate(d.entity_type, out.height, rng)))
    available = set(out.columns)
    ordered = [name for name in original_order if name in available]
    return out.select(ordered)
