"""doppel — synthetic data that looks real."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("doppeldata")
except PackageNotFoundError:  # editable install before metadata is written
    __version__ = "0.0.0+unknown"
