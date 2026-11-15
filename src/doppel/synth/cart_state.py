"""The stored form of a fitted `CartSynthesizer`: typed JSON plus plain numpy arrays.

Structure (columns, dtypes, flags, rates) is a strict Pydantic model. Values (pools,
encoder categories, constants) and trees are numpy arrays of bool, int64, uint64,
float64, or uint8 only. A string list is stored as its UTF-8 bytes plus int64 offsets.
Nothing here can hold an object array, so a stored synthesizer has no executable part.
"""

from __future__ import annotations

import itertools
import zoneinfo
from typing import Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, StrictBool, StrictFloat, StrictInt, StrictStr

from doppel.schema.calendar_feature import CalendarFeature
from doppel.schema.types import Column, ColumnType
from doppel.synth.tree import DecisionTree

ValueKind = Literal["bool", "int", "uint", "float", "str"]
Scalar = StrictBool | StrictInt | StrictFloat | StrictStr

_STRICT = ConfigDict(strict=True, extra="forbid", frozen=True)

_PLAIN_DTYPES: dict[str, pl.DataType] = {
    str(dtype): dtype()
    for dtype in (
        pl.Boolean,
        pl.String,
        pl.Null,
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.Int128,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
        pl.Date,
        pl.Time,
    )
}
_CATEGORY_PHYSICAL = ("UInt8", "UInt16", "UInt32")
_VALUE_DTYPES: dict[str, np.dtype] = {
    "bool": np.dtype(np.bool_),
    "int": np.dtype(np.int64),
    "uint": np.dtype(np.uint64),
    "float": np.dtype(np.float64),
}


class StoredDtype(BaseModel):
    model_config = _STRICT

    name: str
    time_unit: Literal["ns", "us", "ms"] | None = None
    time_zone: str | None = None
    enum_categories: list[str] | None = None
    categories_name: str | None = None
    categories_namespace: str | None = None
    categories_physical: str | None = None


class StoredColumn(BaseModel):
    model_config = _STRICT

    name: str
    type: ColumnType
    nullable: bool
    ordered: bool
    categories: list[Scalar] | None
    calendar_features: list[CalendarFeature] | None


class StoredColumnSynth(BaseModel):
    """One modelled column. Array keys derive from the column's position."""

    model_config = _STRICT

    source_dtype: StoredDtype
    unconditional_resample: bool
    empirical_null_rate: float
    encoder_categories: ValueKind | None
    pool: ValueKind | None
    constant: ValueKind | None
    has_null_model: bool
    has_value_model: bool


class StoredMissingFlag(BaseModel):
    model_config = _STRICT

    flag_column: str
    source_column: str
    flag_dtype: StoredDtype


class StoredCountBound(BaseModel):
    model_config = _STRICT

    low_column: str
    high_column: str


class StoredCart(BaseModel):
    model_config = _STRICT

    table_name: str
    primary_key: str | None
    columns: list[StoredColumn]
    key_dtypes: dict[str, StoredDtype]
    datetime_dtypes: dict[str, StoredDtype]
    calendar_features: dict[str, list[CalendarFeature]]
    column_synths: list[StoredColumnSynth]
    missing_flags: list[StoredMissingFlag]
    count_bounds: list[StoredCountBound]
    ordered_pairs: list[tuple[str, str]]


def encode_dtype(dtype: pl.DataType) -> StoredDtype:
    if isinstance(dtype, pl.Datetime):
        return StoredDtype(name="Datetime", time_unit=dtype.time_unit, time_zone=dtype.time_zone)
    if isinstance(dtype, pl.Duration):
        return StoredDtype(name="Duration", time_unit=dtype.time_unit)
    if isinstance(dtype, pl.Enum):
        return StoredDtype(name="Enum", enum_categories=dtype.categories.to_list())
    if isinstance(dtype, pl.Categorical):
        categories = dtype.categories
        return StoredDtype(
            name="Categorical",
            categories_name=categories.name(),
            categories_namespace=categories.namespace(),
            categories_physical=str(categories.physical()),
        )
    name = str(dtype)
    if name in _PLAIN_DTYPES and dtype == _PLAIN_DTYPES[name]:
        return StoredDtype(name=name)
    raise ValueError(f"the artifact format cannot store a column of dtype {dtype}")


def decode_dtype(stored: StoredDtype) -> pl.DataType:
    match stored.name:
        case "Datetime" if stored.time_unit is not None:
            if stored.time_zone is not None:
                _require_time_zone(stored.time_zone)
            return pl.Datetime(stored.time_unit, stored.time_zone)
        case "Duration" if stored.time_unit is not None:
            return pl.Duration(stored.time_unit)
        case "Enum" if stored.enum_categories is not None:
            return pl.Enum(stored.enum_categories)
        case "Categorical" if (
            stored.categories_name is not None
            and stored.categories_namespace is not None
            and stored.categories_physical in _CATEGORY_PHYSICAL
        ):
            return pl.Categorical(
                pl.Categories(
                    stored.categories_name or None,
                    stored.categories_namespace,
                    _PLAIN_DTYPES[stored.categories_physical],
                )
            )
        case name if name in _PLAIN_DTYPES:
            return _PLAIN_DTYPES[name]
        case _:
            raise ValueError(f"unknown or incomplete stored dtype {stored!r}")


def _require_time_zone(name: str) -> None:
    try:
        zoneinfo.ZoneInfo(name)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"stored time zone {name!r} is not a known time zone") from exc


def encode_column(col: Column) -> StoredColumn:
    categories: list[Scalar] | None = None
    if col.categories is not None:
        categories = []
        for value in col.categories:
            if not isinstance(value, bool | int | float | str):
                raise ValueError(
                    f"column {col.name!r} has a category of type {type(value).__name__}; "
                    "the artifact format stores only bool, int, float, and str categories"
                )
            categories.append(value)
    return StoredColumn(
        name=col.name,
        type=col.type,
        nullable=col.nullable,
        ordered=col.ordered,
        categories=categories,
        calendar_features=list(col.calendar_features)
        if col.calendar_features is not None
        else None,
    )


def decode_column(stored: StoredColumn) -> Column:
    return Column(
        name=stored.name,
        type=stored.type,
        nullable=stored.nullable,
        ordered=stored.ordered,
        categories=tuple(stored.categories) if stored.categories is not None else None,
        calendar_features=tuple(stored.calendar_features)
        if stored.calendar_features is not None
        else None,
    )


def put_values(arrays: dict[str, np.ndarray], key: str, values: list[object]) -> ValueKind | None:
    """Store a list of same-typed scalars under `key`. Returns None for an empty list."""
    if not values:
        return None
    kinds = {type(v) for v in values}
    if kinds == {bool}:
        arrays[key] = np.array(values, dtype=np.bool_)
        return "bool"
    if kinds == {float}:
        arrays[key] = np.array(values, dtype=np.float64)
        return "float"
    if kinds == {str}:
        encoded = [str(v).encode("utf-8") for v in values]
        lengths = np.fromiter((len(b) for b in encoded), dtype=np.int64, count=len(encoded))
        arrays[key] = np.frombuffer(b"".join(encoded), dtype=np.uint8).copy()
        arrays[f"{key}.offsets"] = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        return "str"
    if kinds == {int}:
        try:
            arrays[key] = np.array(values, dtype=np.int64)
        except OverflowError:
            pass
        else:
            return "int"
        try:
            arrays[key] = np.array(values, dtype=np.uint64)
        except OverflowError:
            raise ValueError(f"values under {key!r} do not fit in 64 bits") from None
        return "uint"
    names = ", ".join(sorted(k.__name__ for k in kinds))
    raise ValueError(f"the artifact format cannot store values of type {names} (under {key!r})")


def take_values(arrays: dict[str, np.ndarray], key: str, kind: ValueKind | None) -> list[object]:
    """Remove and decode the values `put_values` stored under `key`."""
    if kind is None:
        return []
    data = _take(arrays, key)
    if data.ndim != 1:
        raise ValueError(f"array {key!r} must be 1-D")
    if kind != "str":
        _require_dtype(key, data, _VALUE_DTYPES[kind])
        if data.size == 0:
            raise ValueError(f"array {key!r} is empty")
        return data.tolist()
    _require_dtype(key, data, np.dtype(np.uint8))
    offsets = _take(arrays, f"{key}.offsets")
    _require_dtype(f"{key}.offsets", offsets, np.dtype(np.int64))
    if (
        offsets.ndim != 1
        or offsets.size < 2
        or offsets[0] != 0
        or offsets[-1] != data.size
        or np.any(np.diff(offsets) < 0)
    ):
        raise ValueError(f"array {key!r}.offsets does not partition its string data")
    blob = data.tobytes()
    bounds = offsets.tolist()
    try:
        return [blob[a:b].decode("utf-8") for a, b in itertools.pairwise(bounds)]
    except UnicodeDecodeError as exc:
        raise ValueError(f"array {key!r} holds invalid UTF-8") from exc


def put_tree(arrays: dict[str, np.ndarray], key: str, tree: DecisionTree) -> None:
    arrays[f"{key}.left_child"] = tree.left_child
    arrays[f"{key}.right_child"] = tree.right_child
    arrays[f"{key}.feature"] = tree.feature
    arrays[f"{key}.threshold"] = tree.threshold
    arrays[f"{key}.missing_go_to_left"] = tree.missing_go_to_left


def take_tree(
    arrays: dict[str, np.ndarray], key: str, n_features: int, max_depth: int
) -> DecisionTree:
    return DecisionTree.from_arrays(
        left_child=_take(arrays, f"{key}.left_child"),
        right_child=_take(arrays, f"{key}.right_child"),
        feature=_take(arrays, f"{key}.feature"),
        threshold=_take(arrays, f"{key}.threshold"),
        missing_go_to_left=_take(arrays, f"{key}.missing_go_to_left"),
        n_features=n_features,
        max_depth=max_depth,
    )


def take_array(arrays: dict[str, np.ndarray], key: str, dtype: type[np.generic]) -> np.ndarray:
    array = _take(arrays, key)
    _require_dtype(key, array, np.dtype(dtype))
    if array.ndim != 1:
        raise ValueError(f"array {key!r} must be 1-D")
    return array


def _take(arrays: dict[str, np.ndarray], key: str) -> np.ndarray:
    try:
        return arrays.pop(key)
    except KeyError:
        raise ValueError(f"stored synthesizer is missing array {key!r}") from None


def _require_dtype(key: str, array: np.ndarray, dtype: np.dtype) -> None:
    if array.dtype != dtype:
        raise ValueError(f"array {key!r} has dtype {array.dtype}, expected {dtype}")
