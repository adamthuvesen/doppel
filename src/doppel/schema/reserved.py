"""Names doppel reserves in its feature matrix, and the guards that keep user data off them."""

from __future__ import annotations

import polars as pl

from doppel.errors import SentinelCollisionError

# Encodes NULL as a category when a categorical or text column is a feature.
NULL_SENTINEL = "__doppel_null__"
# Prefix of the calendar-feature columns derived from a datetime column.
CALENDAR_FEATURE_PREFIX = "__dt_"


def reject_reserved_column_name(name: str) -> None:
    if name.startswith(CALENDAR_FEATURE_PREFIX):
        raise SentinelCollisionError(
            f"source column {name!r} starts with the reserved prefix "
            f"{CALENDAR_FEATURE_PREFIX!r}, which doppel uses internally for "
            "calendar-feature columns. Rename the column upstream."
        )


def reject_reserved_value(values: pl.Series) -> None:
    if NULL_SENTINEL in values.drop_nulls().to_list():
        raise SentinelCollisionError(
            f"column {values.name!r} contains the literal value {NULL_SENTINEL!r}, "
            "which doppel reserves to encode NULL in its feature matrix. "
            "Rename or rewrite the offending value upstream."
        )
