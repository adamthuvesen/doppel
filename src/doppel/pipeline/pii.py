"""Optional PII detect/strip for synthesis pipelines."""

from __future__ import annotations

from collections.abc import Callable

from doppel.dataset import Table
from doppel.pii.detect import PIIDetection


def strip_pii_if_available(
    table: Table,
    *,
    on_detected: Callable[[list[PIIDetection]], None] | None = None,
) -> tuple[list[PIIDetection], Table, list[str]]:
    """Detect and strip PII when the optional ``pii`` extra is installed."""
    unchanged: tuple[list[PIIDetection], Table, list[str]] = (
        [],
        table,
        [c.name for c in table.columns],
    )
    try:
        from doppel.pii.detect import detect as detect_pii
        from doppel.pii.text import strip as strip_pii
    except ImportError:
        return unchanged
    if table.data is None:
        return unchanged
    detections = detect_pii(table.data, table.columns)
    if not detections:
        return unchanged
    if on_detected is not None:
        on_detected(detections)
    stripped, original_order = strip_pii(table, detections)
    return list(detections), stripped, original_order
