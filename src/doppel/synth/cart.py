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
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from doppel.dataset import Dataset, Table
from doppel.schema.datetime import (
    CalendarFeature,
    decompose,
    recompose,
)
from doppel.schema.datetime import (
    calendar_features as extract_calendar_features,
)
from doppel.schema.heuristics import (
    FLOAT_DTYPE_NAMES,
    INTEGER_DTYPE_NAMES,
    is_binary_flag,
    is_integer_dtype,
    looks_like_count_column,
)
from doppel.schema.nullable import encode_feature
from doppel.schema.nullable import null_rate as _null_rate
from doppel.schema.types import Column, ColumnType
from doppel.synth.seed import Rng

_MIN_SAMPLES_LEAF = 5
# Reserved prefix for calendar features inside the CART feature matrix. Double-underscore
# mirrors `__doppel_null__` and is collision-proof against user column names — the fit-time
# guard raises ValueError if any source column starts with this prefix.
_CALENDAR_FEATURE_PREFIX = "__dt_"
FitProgress = Callable[[int, int, str], None]


@dataclass(frozen=True)
class ColumnFitInfo:
    """Public, read-only summary of how one column was modeled. Used by --explain."""

    column: Column
    is_first: bool
    has_value_model: bool
    has_null_model: bool
    has_constant: bool
    empirical_null_rate: float
    nonnull_pool_size: int
    leaf_count: int
    # Resolved calendar features for this column. `None` for non-DATETIME columns;
    # tuple of feature names (e.g. ``("hour", "dow", "month")``) for DATETIME columns;
    # empty tuple when the user explicitly disabled calendar features via the schema.
    calendar_features: tuple[str, ...] | None = None

    @property
    def strategy(self) -> str:
        if self.has_constant:
            return "constant"
        if self.has_value_model:
            return "cart+leaf-sample" if self.has_null_model else "cart-no-nulls"
        return "empirical-resample"


@dataclass(frozen=True)
class RepairSummary:
    missing_flags: dict[str, int] = field(default_factory=dict)
    count_bounds: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.missing_flags.values()) + sum(self.count_bounds.values())


@dataclass(frozen=True)
class _MissingFlag:
    flag_column: str
    source_column: str
    flag_dtype: pl.DataType


@dataclass(frozen=True)
class _CountBound:
    low_column: str
    high_column: str


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
    source_dtype: pl.DataType
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
    source_dtype: pl.DataType,
    features: pl.DataFrame,
    target: pl.Series,
    rng: Rng,
) -> _ColumnSynth:
    encoder = _Encoder.fit(col.type, target)
    n = target.len()
    null_count = target.null_count()
    null_rate = _null_rate(target)
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
            source_dtype=source_dtype,
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
                source_dtype=source_dtype,
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
            source_dtype=source_dtype,
            is_first=False,
            empirical_null_rate=null_rate,
            null_model=null_model,
            value_model=value_model,
            leaf_values=dict(leaf_values),
            nonnull_pool=nonnull_list,
        )

    # Categorical / Text classification path.
    code_for = encoder.code
    y_codes = np.array([code_for[v] for v in nonnull_list], dtype=np.int64)
    if np.unique(y_codes).size <= 1:
        return _ColumnSynth(
            column=col,
            encoder=encoder,
            source_dtype=source_dtype,
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
    return _ColumnSynth(
        column=col,
        encoder=encoder,
        source_dtype=source_dtype,
        is_first=False,
        empirical_null_rate=null_rate,
        null_model=null_model,
        value_model=value_model,
        leaf_values=dict(leaf_values),
        nonnull_pool=nonnull_list,
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


def _generate_key(
    col: Column,
    n: int,
    rng: Rng,
    *,
    source_dtype: pl.DataType | None = None,
) -> pl.Series:
    """Generate `n` synthetic key values, preserving the source dtype.

    Strategy by source dtype:

    - UUID-named columns (`uuid` or `*_uuid`): deterministic RFC-4122 v4
      hex strings, drawn from the seeded `Rng` so two runs with the same
      seed produce byte-identical output. Never uses `uuid.uuid4()`.
    - `pl.String`: emits `f"{column_name}_{i}"` for `i` in `1..=n`. The
      format is intentionally simple and predictable; users who need a
      different shape should provide a `[[constraints]] kind = "derived"`
      block or model the column as a regular non-KEY column.
    - Integer / Float dtypes: sequential `1..=n` cast back to the source
      dtype so downstream consumers see the same numeric type.
    - Anything else: sequential `Int64` `1..=n`.
    """
    name = col.name
    # UUID-name heuristic only applies when the source dtype is string-compatible.
    # A column named `customer_uuid` with an integer source dtype should still get
    # a sequential integer — emitting strings would break the dtype round-trip.
    string_source = source_dtype is None or source_dtype == pl.String
    if string_source and _looks_like_uuid_key_name(name):
        return pl.Series(name, [_random_uuid_hex(rng) for _ in range(n)], dtype=pl.String)
    if source_dtype == pl.String:
        return pl.Series(name, [f"{name}_{i}" for i in range(1, n + 1)], dtype=pl.String)
    out = pl.Series(name, list(range(1, n + 1)), dtype=pl.Int64)
    if source_dtype is not None and str(source_dtype) in (INTEGER_DTYPE_NAMES | FLOAT_DTYPE_NAMES):
        return out.cast(source_dtype)
    return out


def _looks_like_uuid_key_name(name: str) -> bool:
    lower = name.lower()
    return lower == "uuid" or lower.endswith("_uuid")


def _random_uuid_hex(rng: Rng) -> str:
    # Deterministic UUIDv4 from the seeded RNG — never use uuid.uuid4() here, that
    # source is OS-random and would silently break the --seed contract.
    raw = bytearray(rng.numpy.bytes(16))
    raw[6] = (raw[6] & 0x0F) | 0x40  # version 4
    raw[8] = (raw[8] & 0x3F) | 0x80  # RFC-4122 variant
    return uuid.UUID(bytes=bytes(raw)).hex


def _detect_ordered_pairs(cols: list[Column], df: pl.DataFrame) -> list[tuple[str, str]]:
    """Return temporal (low_col, high_col) pairs where low_col <= high_col held for every row.

    Used to enforce impossible orderings that CART leaf-sampling can violate (e.g. a pickup
    datetime synthesised later than its corresponding dropoff datetime).
    """
    ordered: list[tuple[str, str]] = []
    candidates = [c for c in cols if c.type is ColumnType.DATETIME]
    for i, col_a in enumerate(candidates):
        for col_b in candidates[i + 1 :]:
            both_nn = df.filter(pl.col(col_a.name).is_not_null() & pl.col(col_b.name).is_not_null())
            if both_nn.height == 0:
                continue
            a = both_nn[col_a.name].cast(pl.Float64)
            b = both_nn[col_b.name].cast(pl.Float64)
            if (a <= b).all():
                ordered.append((col_a.name, col_b.name))
            elif (b <= a).all():
                ordered.append((col_b.name, col_a.name))
    return ordered


class CartSynthesizer:
    def __init__(self) -> None:
        self._table_name: str = ""
        self._original_columns: list[Column] = []
        self._modeled_columns: list[Column] = []
        self._key_columns: list[Column] = []
        self._key_dtypes: dict[str, pl.DataType] = {}
        self._primary_key: str | None = None
        self._datetime_dtypes: dict[str, pl.DataType] = {}
        # Resolved calendar features per datetime column name. Set at fit time, replayed
        # at sample time so source and synth feature matrices have the same shape.
        self._calendar_features: dict[str, tuple[CalendarFeature, ...]] = {}
        self._column_synths: list[_ColumnSynth] = []
        self._missing_flags: list[_MissingFlag] = []
        self._count_bounds: list[_CountBound] = []
        self._last_repair_summary = RepairSummary()
        # Pairs (col_a, col_b) where col_a <= col_b held for all training rows.
        # Enforced post-sampling to prevent impossible orderings (e.g. pickup > dropoff).
        self._ordered_pairs: list[tuple[str, str]] = []
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

    @property
    def last_repair_summary(self) -> RepairSummary:
        return self._last_repair_summary

    def explain_columns(self) -> list[ColumnFitInfo]:
        """Read-only summary of how each modeled column was fitted.

        Used by `doppel gen --explain` to surface per-column modeling choices
        without exposing internal synth state.
        """
        out: list[ColumnFitInfo] = []
        calendar_map = getattr(self, "_calendar_features", {})
        for cs in self._column_synths:
            cal: tuple[str, ...] | None = None
            if cs.column.type is ColumnType.DATETIME:
                resolved = calendar_map.get(cs.column.name, ())
                cal = tuple(f.value for f in resolved)
            out.append(
                ColumnFitInfo(
                    column=cs.column,
                    is_first=cs.is_first,
                    has_value_model=cs.value_model is not None,
                    has_null_model=cs.null_model is not None,
                    has_constant=cs.has_constant,
                    empirical_null_rate=cs.empirical_null_rate,
                    nonnull_pool_size=len(cs.nonnull_pool),
                    leaf_count=len(cs.leaf_values),
                    calendar_features=cal,
                )
            )
        return out

    def fit(self, dataset: Dataset, rng: Rng, progress: FitProgress | None = None) -> None:
        table = dataset.only()
        if table.data is None:
            raise ValueError(f"table {table.name!r} has no data attached")
        # Guard against user column names that would collide with the calendar-feature
        # prefix and silently corrupt the running feature matrix.
        for col in table.columns:
            if col.name.startswith(_CALENDAR_FEATURE_PREFIX):
                raise ValueError(
                    f"source column {col.name!r} starts with the reserved prefix "
                    f"{_CALENDAR_FEATURE_PREFIX!r}, which doppel uses internally for "
                    "calendar-feature columns. Rename the column upstream."
                )
        self._table_name = table.name
        self._original_columns = list(table.columns)
        self._primary_key = table.primary_key
        self._modeled_columns = [c for c in table.columns if c.is_model_input()]
        self._key_columns = [c for c in table.columns if c.type is ColumnType.KEY]
        self._key_dtypes = {
            c.name: table.data[c.name].dtype
            for c in self._key_columns
            if c.name in table.data.columns
        }

        df = self._prepare_input(table.data)
        self._ordered_pairs = _detect_ordered_pairs(self._modeled_columns, df)
        self._missing_flags = _detect_missing_flags(table.columns, table.data)
        self._count_bounds = _detect_count_bounds(table.columns, table.data)

        # Resolve calendar features per datetime column from the source dtype, then
        # extract them from the original (un-decomposed) series for use as predictors.
        self._calendar_features = {}
        source_calendar_series: dict[str, dict[str, pl.Series]] = {}
        for col in self._modeled_columns:
            if col.type is not ColumnType.DATETIME:
                continue
            source_dtype = self._datetime_dtypes.get(col.name)
            if source_dtype is None:
                continue
            resolved = col.resolved_calendar_features(source_dtype)
            self._calendar_features[col.name] = resolved
            if resolved:
                source_calendar_series[col.name] = extract_calendar_features(
                    table.data[col.name], resolved
                )

        features = pl.DataFrame()
        synths: list[_ColumnSynth] = []
        total = len(self._modeled_columns)
        for idx, col in enumerate(self._modeled_columns, start=1):
            target = df[col.name]
            source_dtype = table.data[col.name].dtype
            cs = _fit_column(col, source_dtype, features, target, rng)
            synths.append(cs)
            feat_values = cs.encoder.transform(target)
            features = features.with_columns(pl.Series(col.name, feat_values, dtype=pl.Float64))
            cal = source_calendar_series.get(col.name)
            if cal:
                features = _append_calendar_features(features, col.name, cal)
            if progress is not None:
                progress(idx, total, col.name)
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
            series = _build_series(col, full, cs.source_dtype)
            modeled_series[col.name] = series
            feat_values = cs.encoder.transform(series)
            features = features.with_columns(pl.Series(col.name, feat_values, dtype=pl.Float64))
            # Extract calendar features from the synth datetime so downstream columns
            # see the same kind of predictors they saw at fit time. The synth epoch is
            # leaf-sampled from a real source epoch, so the calendar values match a
            # real source row by construction (see design D5).
            # `getattr` keeps pre-change pickled artifacts working: they have no
            # `_calendar_features` attribute, which is equivalent to "disabled".
            if col.type is ColumnType.DATETIME:
                calendar_map = getattr(self, "_calendar_features", {})
                resolved = calendar_map.get(col.name, ())
                if resolved:
                    target_dtype = self._datetime_dtypes.get(col.name)
                    if target_dtype is not None:
                        # `series` is still Int64 epoch_s at this point; recompose so
                        # the Polars `dt.*` accessors apply to a real temporal series.
                        recomposed = recompose(series, target_dtype)
                        cal = extract_calendar_features(recomposed, resolved)
                        features = _append_calendar_features(features, col.name, cal)

        # Enforce detected ordering constraints (e.g. pickup_time <= dropoff_time).
        # Datetimes are still Int64 epoch at this point, so comparison is numeric.
        for col_a_name, col_b_name in self._ordered_pairs:
            if col_a_name not in modeled_series or col_b_name not in modeled_series:
                continue
            a = modeled_series[col_a_name]
            b = modeled_series[col_b_name]
            tmp = pl.DataFrame({col_a_name: a, col_b_name: b})
            tmp = tmp.with_columns(
                pl.when(
                    pl.col(col_b_name).is_not_null()
                    & pl.col(col_a_name).is_not_null()
                    & (pl.col(col_b_name) < pl.col(col_a_name))
                )
                .then(pl.col(col_a_name))
                .otherwise(pl.col(col_b_name))
                .alias(col_b_name)
            )
            modeled_series[col_b_name] = tmp[col_b_name]

        # Recompose datetimes back to their original dtype.
        for name, dtype in self._datetime_dtypes.items():
            modeled_series[name] = recompose(modeled_series[name], dtype).alias(name)

        # Generate keys for KEY columns.
        for col in self._key_columns:
            modeled_series[col.name] = _generate_key(
                col, n, rng, source_dtype=self._key_dtypes.get(col.name)
            )

        # Restore original column order from the input table.
        ordered = [
            modeled_series[c.name] for c in self._original_columns if c.name in modeled_series
        ]
        out_df = pl.DataFrame(ordered)
        out_df, self._last_repair_summary = _repair_output(
            out_df, self._missing_flags, self._count_bounds
        )

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


def _append_calendar_features(
    features: pl.DataFrame, source_col: str, cal: dict[str, pl.Series]
) -> pl.DataFrame:
    """Append `__dt_<source_col>_<feature>` columns to the running feature matrix.

    Each calendar feature is cast to Float64 alongside every other feature column.
    Nulls in the source datetime propagate to nulls in the calendar feature; those
    rows then go through the same null-imputation path as any other feature (median
    fill in `encode_feature` — but encode_feature is per-target-column; here we
    fill_null with 0 inline to keep the matrix sklearn-ready). Calendar values are
    bounded small integers, so 0 is a safe sentinel for the rare null case.
    """
    new_cols: list[pl.Series] = []
    for feature_name, series in cal.items():
        col_name = f"{_CALENDAR_FEATURE_PREFIX}{source_col}_{feature_name}"
        filled = series.fill_null(0).cast(pl.Float64).rename(col_name)
        new_cols.append(filled)
    if not new_cols:
        return features
    return features.with_columns(new_cols)


def _interleave(nonnull_values: list[object], null_mask: np.ndarray) -> list[object | None]:
    """Place sampled values into a length-N list at positions where null_mask is False.

    Numpy scatter via the boolean mask avoids the per-row Python loop that dominated
    sample-time CPU on million-row outputs.
    """
    out = np.empty(len(null_mask), dtype=object)
    out[null_mask] = None
    if nonnull_values:
        out[~null_mask] = np.asarray(nonnull_values, dtype=object)
    return out.tolist()


def _build_series(
    col: Column, values: list[object | None], source_dtype: pl.DataType | None = None
) -> pl.Series:
    if col.type is ColumnType.DATETIME:
        # CART regressor leaf-samples produce floats; coerce to Int64 epoch before recomposition.
        coerced: list[int | None] = [None if v is None else int(v) for v in values]  # type: ignore[arg-type]
        return pl.Series(col.name, coerced, dtype=pl.Int64)
    if col.type is ColumnType.NUMERIC:
        series = pl.Series(col.name, values, dtype=pl.Float64)
        if source_dtype is not None and str(source_dtype) in INTEGER_DTYPE_NAMES:
            return series.round(0).cast(source_dtype)
        if source_dtype is not None and str(source_dtype) in FLOAT_DTYPE_NAMES:
            return series.cast(source_dtype)
        return series
    series = pl.Series(col.name, values)
    if source_dtype is not None and source_dtype != pl.Null:
        try:
            return series.cast(source_dtype)
        except pl.exceptions.PolarsError:
            return series
    return series


def _repair_output(
    df: pl.DataFrame,
    missing_flags: list[_MissingFlag],
    count_bounds: list[_CountBound],
) -> tuple[pl.DataFrame, RepairSummary]:
    out = df
    missing_repairs: dict[str, int] = {}
    for flag in missing_flags:
        if flag.source_column not in out.columns or flag.flag_column not in out.columns:
            continue
        desired = out[flag.source_column].is_null()
        current = _flag_truth(out[flag.flag_column])
        changes = int((current != desired).sum())
        if changes == 0:
            continue
        missing_repairs[flag.flag_column] = changes
        values = desired.cast(flag.flag_dtype).alias(flag.flag_column)
        out = out.with_columns(values)

    bound_repairs: dict[str, int] = {}
    for _ in range(3):
        changed = False
        for bound in count_bounds:
            if bound.low_column not in out.columns or bound.high_column not in out.columns:
                continue
            low = out[bound.low_column]
            high = out[bound.high_column]
            mask = low.is_not_null() & high.is_not_null() & (low > high)
            changes = int(mask.sum())
            if changes == 0:
                continue
            changed = True
            label = f"{bound.low_column} <= {bound.high_column}"
            bound_repairs[label] = bound_repairs.get(label, 0) + changes
            out = out.with_columns(
                pl.when(mask)
                .then(pl.col(bound.high_column))
                .otherwise(pl.col(bound.low_column))
                .cast(low.dtype)
                .alias(bound.low_column)
            )
        if not changed:
            break

    return out, RepairSummary(missing_flags=missing_repairs, count_bounds=bound_repairs)


def _detect_missing_flags(columns: list[Column], df: pl.DataFrame) -> list[_MissingFlag]:
    by_name = {c.name: c for c in columns}
    flags: list[_MissingFlag] = []
    for col in columns:
        source = _missing_flag_source(col.name, by_name)
        if source is None or source not in df.columns or col.name not in df.columns:
            continue
        flag_series = df[col.name]
        if not is_binary_flag(flag_series):
            continue
        if (_flag_truth(flag_series) == df[source].is_null()).all():
            flags.append(_MissingFlag(col.name, source, flag_series.dtype))
    return flags


def _missing_flag_source(name: str, by_name: dict[str, Column]) -> str | None:
    upper = name.upper()
    if "_MISSING" in upper:
        idx = upper.index("_MISSING")
        prefix = name[:idx]
        if prefix in by_name:
            return prefix
        suffix = name[idx + len("_MISSING") :]
        candidate = f"{prefix}{suffix}"
        if candidate in by_name:
            return candidate
    if upper.startswith("IS_") and upper.endswith("_MISSING"):
        candidate = name[3:-8]
        if candidate in by_name:
            return candidate
    return None


def _detect_count_bounds(columns: list[Column], df: pl.DataFrame) -> list[_CountBound]:
    candidates = [
        c
        for c in columns
        if c.name in df.columns
        and c.type is ColumnType.NUMERIC
        and is_integer_dtype(df[c.name].dtype)
        and looks_like_count_column(c.name)
    ]
    bounds: list[_CountBound] = []
    for low in candidates:
        for high in candidates:
            if low.name == high.name:
                continue
            both_nn = df.filter(pl.col(low.name).is_not_null() & pl.col(high.name).is_not_null())
            if both_nn.height == 0:
                continue
            if (both_nn[low.name] <= both_nn[high.name]).all():
                bounds.append(_CountBound(low.name, high.name))
    return bounds


def _flag_truth(series: pl.Series) -> pl.Series:
    return series.fill_null(0).cast(pl.Int8) == 1
