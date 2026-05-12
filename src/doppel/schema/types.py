"""Column type system — the canonical classification driving inference, synthesis, and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


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

    def is_model_input(self) -> bool:
        # KEY columns are generated post-hoc, never modeled.
        return self.type is not ColumnType.KEY
