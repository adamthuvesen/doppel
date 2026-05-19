"""Schema TOML — load, save, and merge with an inferred schema.

The file shape is:

    [table]
    name = "orders"
    primary_key = "order_id"

    [columns.amount]
    type = "numeric"
    nullable = false

    [columns.status]
    type = "categorical"
    ordered = true
    categories = ["draft", "submitted", "paid", "refunded"]

    [[constraints]]
    kind = "range"
    column = "amount"
    min = 0

Columns and constraints are both optional sections. `doppel schema infer` produces
a fully populated file; `doppel gen --schema` accepts partial overrides on top of
auto-inferred types.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import tomli_w
import typer
from pydantic import BaseModel, Field, field_validator

from doppel.constraints.dsl import (
    Constraint,
    DerivedConstraint,
    RangeConstraint,
)
from doppel.dataset import Table
from doppel.schema.datetime import CalendarFeature
from doppel.schema.types import Column, ColumnType


class TableMeta(BaseModel):
    name: str
    primary_key: str | None = None


_CALENDAR_ALLOWLIST = tuple(f.value for f in CalendarFeature)


class ColumnSpec(BaseModel):
    type: ColumnType
    nullable: bool = True
    ordered: bool = False
    categories: list[Any] | None = None
    # `calendar_features` accepts three TOML forms:
    #   - omitted/None      → use the dtype default at fit time.
    #   - `false`           → disabled (empty tuple after validation).
    #   - list[str]         → validated allowlist members.
    # `true` is intentionally rejected — omission already means "default".
    calendar_features: tuple[CalendarFeature, ...] | None = None

    @field_validator("calendar_features", mode="before")
    @classmethod
    def _parse_calendar_features(cls, value: object) -> tuple[CalendarFeature, ...] | None:
        if value is None:
            return None
        if value is True:
            # Pydantic would otherwise coerce `True` to a 1-element tuple via iter.
            raise ValueError(
                "calendar_features = true is not supported; omit the key for the default"
            )
        if value is False:
            return ()
        if not isinstance(value, list | tuple):
            raise ValueError(
                "calendar_features must be `false`, a list of feature names, or omitted"
            )
        items: list[CalendarFeature] = []
        for raw in value:
            if not isinstance(raw, str):
                raise ValueError(
                    f"calendar_features entries must be strings, got {type(raw).__name__}"
                )
            try:
                items.append(CalendarFeature(raw))
            except ValueError as exc:
                raise typer.BadParameter(
                    f"unknown calendar feature {raw!r}; allowed: {', '.join(_CALENDAR_ALLOWLIST)}"
                ) from exc
        return tuple(items)


class SchemaToml(BaseModel):
    table: TableMeta
    columns: dict[str, ColumnSpec] = Field(default_factory=dict)
    constraints: list[Constraint] = Field(default_factory=list)


def load(path: Path) -> SchemaToml:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return SchemaToml.model_validate(raw)


def save(schema: SchemaToml, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(to_payload(schema)), encoding="utf-8")


def from_table(table: Table) -> SchemaToml:
    return SchemaToml(
        table=TableMeta(name=table.name, primary_key=table.primary_key),
        columns={
            col.name: ColumnSpec(
                type=col.type,
                nullable=col.nullable,
                ordered=col.ordered,
                categories=list(col.categories) if col.categories is not None else None,
                # `schema infer` leaves calendar_features unset — default-by-omission.
                calendar_features=col.calendar_features,
            )
            for col in table.columns
        },
    )


def apply_overrides(inferred: Table, schema: SchemaToml) -> Table:
    """Merge user TOML overrides into the inferred Table. TOML wins per-field.

    Validation:
      - column overrides naming a column not present in the inferred data are rejected
        loudly (was previously silent).
      - any declared `primary_key` is auto-promoted to `ColumnType.KEY` so the
        synthesizer generates unique values for it rather than modelling the column.
    """
    validate_against_table(inferred, schema)
    declared_pk = schema.table.primary_key or inferred.primary_key
    columns = merge_columns(inferred.columns, schema.columns, declared_pk)
    return Table(
        name=schema.table.name,
        columns=columns,
        primary_key=declared_pk,
        data=inferred.data,
    )


def merge_columns(
    inferred: list[Column],
    overrides: dict[str, ColumnSpec],
    declared_pk: str | None,
) -> list[Column]:
    """Merge per-column TOML overrides onto inferred columns; promote declared PK to KEY.

    Shared between single-table `apply_overrides` and multi-table `to_dataset` — having
    one implementation prevents the synth's PK behavior from drifting between the two paths.
    """
    out: list[Column] = []
    for col in inferred:
        spec = overrides.get(col.name)
        if spec is None:
            merged = col
        else:
            merged = Column(
                name=col.name,
                type=spec.type,
                nullable=spec.nullable,
                ordered=spec.ordered,
                categories=(
                    tuple(spec.categories) if spec.categories is not None else col.categories
                ),
                calendar_features=spec.calendar_features,
            )
        if (
            declared_pk is not None
            and merged.name == declared_pk
            and merged.type is not ColumnType.KEY
        ):
            merged = Column(
                name=merged.name,
                type=ColumnType.KEY,
                nullable=False,
                ordered=merged.ordered,
                categories=None,
            )
        out.append(merged)
    return out


def validate_against_table(inferred: Table, schema: SchemaToml) -> None:
    """Validate schema references against an inferred table before merge or CLI check."""
    inferred_names = {c.name for c in inferred.columns}
    unknown = sorted(set(schema.columns) - inferred_names)
    if unknown:
        raise ValueError(
            f"schema declares columns not in data: {unknown}. "
            f"Available columns: {sorted(inferred_names)}"
        )

    if schema.table.primary_key is not None:
        if schema.table.primary_key not in inferred_names:
            raise ValueError(
                f"primary_key {schema.table.primary_key!r} is not present in the data. "
                f"Available columns: {sorted(inferred_names)}"
            )
        if inferred.data is not None:
            series = inferred.data[schema.table.primary_key]
            if series.null_count() > 0 or series.n_unique() != inferred.data.height:
                raise ValueError(
                    f"primary_key {schema.table.primary_key!r} must be unique and non-null"
                )

    _validate_constraints(schema.constraints, inferred_names)


def _validate_constraints(constraints: list[Constraint], column_names: set[str]) -> None:
    from doppel.constraints.derived import compile_expression

    derived_names = {c.column for c in constraints if isinstance(c, DerivedConstraint)}
    allowed_after_derived = set(column_names) | derived_names
    allowed_for_derived = set(column_names)

    for c in constraints:
        if isinstance(c, DerivedConstraint):
            try:
                compile_expression(c.expression, allowed_for_derived)
            except ValueError as exc:
                raise ValueError(f"constraint references unknown column: {exc}") from exc
            allowed_for_derived.add(c.column)
        elif isinstance(c, RangeConstraint):
            if c.column not in allowed_after_derived:
                raise ValueError(f"constraint references unknown column {c.column!r}")
        else:
            for name in (c.left, c.right):
                if name not in allowed_after_derived:
                    raise ValueError(f"constraint references unknown column {name!r}")


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    """TOML can't encode `None`. Drop keys whose value is None before serialising."""
    return {k: v for k, v in d.items() if v is not None}


def _column_payload(spec: ColumnSpec) -> dict[str, Any]:
    """Render one column spec for tomli_w: drop Nones; emit calendar_features as
    `false` (when disabled) or a list of strings (when customised)."""
    raw = spec.model_dump()
    cf = raw.pop("calendar_features", None)
    out = _drop_none(raw)
    if cf is None:
        return out
    if not cf:
        out["calendar_features"] = False
    else:
        # StrEnum members serialise to plain strings via .value.
        out["calendar_features"] = [
            f.value if isinstance(f, CalendarFeature) else str(f) for f in cf
        ]
    return out


def to_payload(schema: SchemaToml) -> dict[str, Any]:
    """Render a SchemaToml as the nested-dict shape used by tomli_w + artifact metadata."""
    payload: dict[str, Any] = {"table": _drop_none(schema.table.model_dump())}
    if schema.columns:
        payload["columns"] = {name: _column_payload(spec) for name, spec in schema.columns.items()}
    if schema.constraints:
        payload["constraints"] = [_drop_none(c.model_dump()) for c in schema.constraints]
    return payload


__all__ = [
    "ColumnSpec",
    "SchemaToml",
    "TableMeta",
    "apply_overrides",
    "from_table",
    "load",
    "merge_columns",
    "save",
    "to_payload",
    "validate_against_table",
]
