"""Optional PII detect/strip for synthesis pipelines."""

from __future__ import annotations

import importlib.util
import warnings
from collections.abc import Callable

from doppel.dataset import Table
from doppel.pii.detect import PIIDetection

# The `pii` extra pulls these. `doppel.pii.detect` / `.text` import fine without them
# (Presidio/Faker are constructed lazily inside the functions), so probing the module
# import is not enough — probe the underlying packages at the point of use.
_PII_PACKAGES = ("presidio_analyzer", "faker")


def _pii_extra_available() -> bool:
    return all(importlib.util.find_spec(pkg) is not None for pkg in _PII_PACKAGES)


def strip_pii_if_available(
    table: Table,
    *,
    on_detected: Callable[[list[PIIDetection]], None] | None = None,
) -> tuple[list[PIIDetection], Table, list[str]]:
    """Detect and strip PII when the optional ``pii`` extra is installed.

    Without the extra this is a no-op: free text is left untouched and a warning is
    emitted. Use ``doppel diff`` (DCR percentiles, per-column verbatim_rate) to check
    whether unstripped text leaked source values.
    """
    unchanged: tuple[list[PIIDetection], Table, list[str]] = (
        [],
        table,
        [c.name for c in table.columns],
    )
    if not _pii_extra_available():
        warnings.warn(
            "PII detection skipped: the optional [pii] extra is not installed. "
            "Free-text columns are passed through unmodified and may leak source "
            'values. Install with: pip install "doppeldata[pii]"',
            stacklevel=2,
        )
        return unchanged
    from doppel.pii.detect import detect as detect_pii
    from doppel.pii.text import strip as strip_pii

    if table.data is None:
        return unchanged
    detections = detect_pii(table.data, table.columns)
    if not detections:
        return unchanged
    if on_detected is not None:
        on_detected(detections)
    stripped, original_order = strip_pii(table, detections)
    return list(detections), stripped, original_order
