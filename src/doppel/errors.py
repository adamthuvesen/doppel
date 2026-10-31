"""doppel's error types. Core code raises these; only the CLI turns them into exit codes.

`InvalidInputError` is a `ValueError`: the input (a path, URI, option, schema, or
artifact) is one doppel rejects. Every other `DoppelError` is a failure at run
time, such as a warehouse that does not answer."""

from __future__ import annotations


class DoppelError(Exception):
    """Base class for errors doppel raises on purpose."""


class InvalidInputError(DoppelError, ValueError):
    """The user's input is one doppel rejects. The CLI exits with code 2."""


class ArtifactError(InvalidInputError):
    """A `.doppel` artifact is malformed, mis-versioned, or untrusted."""


class SentinelCollisionError(InvalidInputError):
    """User data uses a column name or value that doppel reserves."""


class WarehouseConnectionError(DoppelError, RuntimeError):
    """The SQL driver failed to connect or execute. The message carries the redacted
    URI (`:***@`), never the raw password."""


class RowCountProbeError(DoppelError, RuntimeError):
    """A row-count probe against the warehouse failed."""


class PIIUnavailableError(DoppelError, RuntimeError):
    """PII detection cannot run because the pinned spaCy model is not installed."""
