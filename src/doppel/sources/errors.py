"""Doppel-internal exceptions for the SQL/source layer.

Kept separate from `spec.py` so source modules can import the errors without
pulling in the dataclass machinery, and so tests can match on them.
"""

from __future__ import annotations


class WarehouseConnectionError(RuntimeError):
    """Raised when the SQL driver fails to connect or execute. The message
    contains the redacted URI (`:***@`); never the raw password."""


class RowCountProbeError(RuntimeError):
    """Raised when a row-count probe against the warehouse fails."""


__all__ = [
    "RowCountProbeError",
    "WarehouseConnectionError",
]
