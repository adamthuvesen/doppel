"""Column type system — the canonical classification driving inference, synthesis, and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

    from doppel.schema.datetime import CalendarFeature


class ColumnType(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    KEY = "key"


@dataclass(frozen=True)
class Column:
    name: str
    type: ColumnType
    nullable: bool = True
    # Ordered categoricals (e.g. small < medium < large) preserve their order across synthesis.
    ordered: bool = False
    # Populated for CATEGORICAL (and bool-as-categorical); otherwise None.
    categories: tuple[object, ...] | None = None
    # Calendar features extracted from a DATETIME column for use as CART predictors on
    # downstream columns. `None` means "use the dtype default" (resolved at fit time);
    # `()` means "disabled". Non-DATETIME columns leave this at `None`.
    calendar_features: tuple[CalendarFeature, ...] | None = None

    def is_model_input(self) -> bool:
        # KEY columns are generated post-hoc, never modeled.
        return self.type is not ColumnType.KEY

    def resolved_calendar_features(self, dtype: pl.DataType) -> tuple[CalendarFeature, ...]:
        """Return the configured calendar features, or the dtype default when unset.

        - `calendar_features = None`  → default for the source dtype (Datetime/Date).
        - `calendar_features = ()`    → explicitly disabled (returns `()`).
        - `calendar_features = (...)` → exactly these features.
        """
        if self.calendar_features is not None:
            return self.calendar_features
        # Local import to avoid the StrEnum import cycle at module load.
        from doppel.schema.datetime import default_features_for

        return default_features_for(dtype)
