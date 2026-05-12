"""Sequential CART synthesizer.

For each column in input order:
  1. Encode the null mask. First column: empirical null rate. Later columns: a
     `DecisionTreeClassifier` conditional on previously generated columns.
  2. Encode the value on the non-null subset. First column: sample with replacement from
     the observed values. Later columns: fit a `DecisionTreeRegressor` (numeric/datetime)
     or `DecisionTreeClassifier` (categorical/text), apply the trained tree to the synthetic
     feature row, then sample uniformly from the training values that landed in that leaf
     (non-parametric leaf sampling — preserves the empirical distribution within each split).
  3. Encode the column as a numeric feature for downstream columns. Categoricals/Text get
     a label-encoded code; numerics/datetime epochs pass through. Nulls are filled (median
     or `__doppel_null__` sentinel) before they enter the feature matrix.

Datetime columns are decomposed to Int64 epoch-seconds before fitting and recomposed at
output. KEY columns are not modeled; they get sequence/uuid values post-hoc.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from doppel.dataset import Dataset, Table
from doppel.schema.datetime import decompose, recompose
from doppel.schema.nullable import encode_feature
from doppel.schema.types import Column, ColumnType
from doppel.synth.seed import Rng

_MIN_SAMPLES_LEAF = 5


@dataclass
class _Encoder:
    """Maps a column's values to a numeric vector for use as a feature for downstream columns."""

    ctype: ColumnType
    code: dict[object, int] = field(default_factory=dict)

    @classmethod
    def fit(cls, ctype: ColumnType, series: pl.Series) -> _Encoder:
        if ctype in (ColumnType.CATEGORICAL, ColumnType.TEXT):
            filled = encode_feature(series, ctype)
            cats = filled.unique().sort().to_list()
            return cls(ctype=ctype, code={c: i for i, c in enumerate(cats)})
        return cls(ctype=ctype)

    def transform(self, series: pl.Series) -> np.ndarray:
        filled = encode_feature(series, self.ctype)
        if self.ctype in (ColumnType.CATEGORICAL, ColumnType.TEXT):
            return np.array(
                [self.code.get(v, -1) for v in filled.to_list()],
                dtype=np.float64,
            )
        return filled.cast(pl.Float64).to_numpy().astype(np.float64)


@dataclass
class _ColumnSynth:
    column: Column
    encoder: _Encoder
    is_first: bool
    empirical_null_rate: float
    # First-column path: pool of observed non-null values to resample.
    nonnull_pool: list[object] = field(default_factory=list)
    # Conditional path:
    null_model: DecisionTreeClassifier | None = None
    value_model: DecisionTreeRegressor | DecisionTreeClassifier | None = None
    leaf_values: dict[int, list[object]] = field(default_factory=dict)
    # Constant-target short circuit.
    constant_value: object | None = None
    has_constant: bool = False


def _fit_column(
    col: Column,
    features: pl.DataFrame,
    target: pl.Series,
    rng: Rng,
) -> _ColumnSynth:
    encoder = _Encoder.fit(col.type, target)
    n = target.len()
    null_count = target.null_count()
    null_rate = 0.0 if n == 0 else null_count / n
    nonnull = target.drop_nulls()
    nonnull_list = nonnull.to_list()
    is_first = features.width == 0

    # Free text is always sampled-with-replacement in Phase 1 — fitting a classifier
    # on a high-cardinality string target leaks raw values into a noisy model and
    # buys us nothing over empirical resampling. PII handling lands in Phase 6.
    if is_first or len(nonnull_list) <= 1 or col.type is ColumnType.TEXT:
        const_value, has_const = (
            (nonnull_list[0], True)
            if len(set(nonnull_list)) == 1 and nonnull_list
            else (None, False)
        )
        return _ColumnSynth(
            column=col,
            encoder=encoder,
            is_first=True,
            empirical_null_rate=null_rate,
            nonnull_pool=nonnull_list,
            constant_value=const_value,
            has_constant=has_const,
        )

    x_all = features.to_numpy()
    null_model: DecisionTreeClassifier | None = None
    if null_count > 0 and null_count < n:
        is_null_y = target.is_null().cast(pl.Int8).to_numpy()
        null_model = DecisionTreeClassifier(
            random_state=rng.sklearn_seed(),
            min_samples_leaf=_MIN_SAMPLES_LEAF,
        )
        null_model.fit(x_all, is_null_y)

    nonnull_mask_pl = target.is_not_null()
    x_nn = features.filter(nonnull_mask_pl).to_numpy()

    if col.type in (ColumnType.NUMERIC, ColumnType.DATETIME):
        y_arr = np.asarray(nonnull_list, dtype=np.float64)
        if np.unique(y_arr).size <= 1:
            return _ColumnSynth(
                column=col,
                encoder=encoder,
                is_first=False,
                empirical_null_rate=null_rate,
                null_model=null_model,
                constant_value=float(y_arr[0]) if y_arr.size else None,
                has_constant=y_arr.size > 0,
            )
        value_model = DecisionTreeRegressor(
            random_state=rng.sklearn_seed(),
            min_samples_leaf=_MIN_SAMPLES_LEAF,
        )
        value_model.fit(x_nn, y_arr)
        leaves = value_model.apply(x_nn)
        leaf_values: dict[int, list[object]] = defaultdict(list)
        for leaf, val in zip(leaves, y_arr.tolist(), strict=True):
            leaf_values[int(leaf)].append(val)
        return _ColumnSynth(
            column=col,
            encoder=encoder,
            is_first=False,
            empirical_null_rate=null_rate,
            null_model=null_model,
            value_model=value_model,
            leaf_values=dict(leaf_values),
        )

    # Categorical / Text classification path.
    cats = list(encoder.code.keys())
    code_for = encoder.code
    y_codes = np.array([code_for[v] for v in nonnull_list], dtype=np.int64)
    if np.unique(y_codes).size <= 1:
        return _ColumnSynth(
            column=col,
            encoder=encoder,
            is_first=False,
            empirical_null_rate=null_rate,
            null_model=null_model,
            constant_value=nonnull_list[0] if nonnull_list else None,
            has_constant=bool(nonnull_list),
        )
    value_model = DecisionTreeClassifier(
        random_state=rng.sklearn_seed(),
        min_samples_leaf=_MIN_SAMPLES_LEAF,
    )
    value_model.fit(x_nn, y_codes)
    leaves = value_model.apply(x_nn)
    leaf_values = defaultdict(list)
    for leaf, val in zip(leaves, nonnull_list, strict=True):
        leaf_values[int(leaf)].append(val)
    _ = cats  # silence unused warning; categories live in encoder.code.
    return _ColumnSynth(
        column=col,
        encoder=encoder,
        is_first=False,
        empirical_null_rate=null_rate,
        null_model=null_model,
        value_model=value_model,
        leaf_values=dict(leaf_values),
    )


def _sample_null_mask(cs: _ColumnSynth, x: np.ndarray | None, n: int, rng: Rng) -> np.ndarray:
    if cs.empirical_null_rate <= 0.0:
        return np.zeros(n, dtype=bool)
    if cs.is_first or cs.null_model is None or x is None:
        return rng.numpy.random(size=n) < cs.empirical_null_rate
    proba = np.asarray(cs.null_model.predict_proba(x))
    classes = np.asarray(cs.null_model.classes_)
    # Probability assigned to the "is-null" class (label 1).
    if 1 in classes:
        idx = int(np.where(classes == 1)[0][0])
        p_null = proba[:, idx]
    else:
        p_null = np.zeros(n, dtype=np.float64)
    return rng.numpy.random(size=n) < p_null


def _sample_values(
    cs: _ColumnSynth,
    x: np.ndarray | None,
    n_needed: int,
    rng: Rng,
) -> list[object]:
    if n_needed == 0:
        return []
    if cs.has_constant:
        return [cs.constant_value] * n_needed
    if cs.is_first or cs.value_model is None or x is None:
        idx = rng.numpy.integers(0, len(cs.nonnull_pool), size=n_needed)
        return [cs.nonnull_pool[i] for i in idx]
    leaves = cs.value_model.apply(x)
    out: list[object] = []
    for leaf in leaves.tolist():
        pool = cs.leaf_values.get(int(leaf), cs.nonnull_pool)
        out.append(pool[int(rng.numpy.integers(0, len(pool)))])
    return out


def _generate_key(col: Column, n: int, rng: Rng) -> pl.Series:
    name = col.name
    if name.lower() in {"id", "uuid"} or name.lower().endswith("_uuid"):
        return pl.Series(name, [_random_uuid_hex(rng) for _ in range(n)], dtype=pl.String)
    # Default to ascending integers. Phase 4 schema overrides can refine this.
    return pl.Series(name, list(range(1, n + 1)), dtype=pl.Int64)


def _random_uuid_hex(rng: Rng) -> str:
    # Deterministic UUIDv4 from the seeded RNG — never use uuid.uuid4() here, that
    # source is OS-random and would silently break the --seed contract.
    raw = bytearray(rng.numpy.bytes(16))
    raw[6] = (raw[6] & 0x0F) | 0x40  # version 4
    raw[8] = (raw[8] & 0x3F) | 0x80  # RFC-4122 variant
    return uuid.UUID(bytes=bytes(raw)).hex


class CartSynthesizer:
    def __init__(self) -> None:
        self._table_name: str = ""
        self._original_columns: list[Column] = []
        self._modeled_columns: list[Column] = []
        self._key_columns: list[Column] = []
        self._primary_key: str | None = None
        self._datetime_dtypes: dict[str, pl.DataType] = {}
        self._column_synths: list[_ColumnSynth] = []
        self._fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def original_columns(self) -> list[Column]:
        return self._original_columns

    @property
    def primary_key(self) -> str | None:
        return self._primary_key

    def fit(self, dataset: Dataset, rng: Rng) -> None:
        table = dataset.only()
        if table.data is None:
            raise ValueError(f"table {table.name!r} has no data attached")
        self._table_name = table.name
        self._original_columns = list(table.columns)
        self._primary_key = table.primary_key
        self._modeled_columns = [c for c in table.columns if c.is_model_input()]
        self._key_columns = [c for c in table.columns if c.type is ColumnType.KEY]

        df = self._prepare_input(table.data)

        features = pl.DataFrame()
        synths: list[_ColumnSynth] = []
        for col in self._modeled_columns:
            target = df[col.name]
            cs = _fit_column(col, features, target, rng)
            synths.append(cs)
            feat_values = cs.encoder.transform(target)
            features = features.with_columns(pl.Series(col.name, feat_values, dtype=pl.Float64))
        self._column_synths = synths
        self._fitted = True

    def sample(self, n: int, rng: Rng) -> Dataset:
        if not self._fitted:
            raise RuntimeError("CartSynthesizer.sample() called before fit()")

        features = pl.DataFrame()
        modeled_series: dict[str, pl.Series] = {}
        for col, cs in zip(self._modeled_columns, self._column_synths, strict=True):
            x = features.to_numpy() if features.width > 0 else None
            null_mask = _sample_null_mask(cs, x, n, rng)
            n_nonnull = int((~null_mask).sum())
            x_nonnull = features.filter(pl.Series(~null_mask)).to_numpy() if x is not None else None
            values = _sample_values(cs, x_nonnull, n_nonnull, rng)
            full = _interleave(values, null_mask)
            series = _build_series(col, full)
            modeled_series[col.name] = series
            feat_values = cs.encoder.transform(series)
            features = features.with_columns(pl.Series(col.name, feat_values, dtype=pl.Float64))

        # Recompose datetimes back to their original dtype.
        for name, dtype in self._datetime_dtypes.items():
            modeled_series[name] = recompose(modeled_series[name], dtype).alias(name)

        # Generate keys for KEY columns.
        for col in self._key_columns:
            modeled_series[col.name] = _generate_key(col, n, rng)

        # Restore original column order from the input table.
        ordered = [
            modeled_series[c.name] for c in self._original_columns if c.name in modeled_series
        ]
        out_df = pl.DataFrame(ordered)

        table = Table(
            name=self._table_name,
            columns=self._original_columns,
            primary_key=self._primary_key,
            data=out_df,
        )
        return Dataset.single(table)

    def _prepare_input(self, df: pl.DataFrame) -> pl.DataFrame:
        out = df
        self._datetime_dtypes = {}
        for col in self._modeled_columns:
            if col.type is ColumnType.DATETIME:
                series = out[col.name]
                self._datetime_dtypes[col.name] = series.dtype
                out = out.with_columns(decompose(series).alias(col.name))
        return out


def _interleave(nonnull_values: list[object], null_mask: np.ndarray) -> list[object | None]:
    """Place sampled values into a length-N list at positions where null_mask is False."""
    out: list[object | None] = [None] * len(null_mask)
    it = iter(nonnull_values)
    for i, is_null in enumerate(null_mask.tolist()):
        if not is_null:
            out[i] = next(it)
    return out


def _build_series(col: Column, values: list[object | None]) -> pl.Series:
    if col.type is ColumnType.DATETIME:
        # CART regressor leaf-samples produce floats; coerce to Int64 epoch before recomposition.
        coerced: list[int | None] = [None if v is None else int(v) for v in values]  # type: ignore[arg-type]
        return pl.Series(col.name, coerced, dtype=pl.Int64)
    if col.type is ColumnType.NUMERIC:
        return pl.Series(col.name, values, dtype=pl.Float64)
    return pl.Series(col.name, values)
