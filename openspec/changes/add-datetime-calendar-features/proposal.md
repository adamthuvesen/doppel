## Why

CART currently sees datetime columns only as `Int64` epoch-seconds
([src/doppel/schema/datetime.py:20](../../../src/doppel/schema/datetime.py)). Hour-of-day,
day-of-week, business-hours, and monthly seasonality patterns collapse to nothing — the
tree can't split on "Friday afternoon" because epoch values cycle through that pattern at
a constant rate. Every event/transaction/login/ticket dataset suffers; the audit and
CLAUDE.md both document this as a v0.1 limitation.

Adding calendar features (`hour`, `dow`, `month` by default) to the CART feature matrix
lets downstream columns split cleanly on temporal patterns. The win is large for fidelity
(visible KS-on-dow drops on real datasets) while the change stays contained: no new
targets, no artifact-format change, no new dependencies, and consistency is preserved by
construction because the synth datetime is leaf-sampled from real source epochs.

## What Changes

- Extract calendar features from each datetime/date column at fit and sample time and
  inject them into the CART feature matrix as predictors for **downstream columns only**.
  The datetime itself is still modeled as `Int64` epoch-seconds and recomposed via the
  existing path. Calendar features are never modeled as targets.
- Default feature set:
  - `pl.Datetime` columns → `hour, dow, month` (3 Int8 features).
  - `pl.Date` columns → `dow, month` (no time-of-day).
  - `pl.Time` and `pl.Duration` → no calendar features (out of scope; deferred).
- Internal feature naming uses a `__dt_<col>_<feature>` prefix (double-underscore matches
  the existing `__doppel_null__` sentinel pattern) so user column names cannot collide.
- Schema TOML accepts a per-column override:
  - `calendar_features = false` disables features for that column.
  - `calendar_features = ["hour", "dow"]` enables only the listed allowlist members.
  - Unknown feature names raise `typer.BadParameter` at TOML load.
  - Allowlist: `hour, minute, dow, month, day_of_month, week_of_year, quarter`.
    `is_weekend` is explicitly NOT in the allowlist (CART learns it from `dow` with one split).
- `schema infer` leaves `calendar_features` unset in the generated TOML; default-by-omission.
- Polars' `dt.hour() / dt.weekday() / dt.month()` operate on the local representation for
  tz-aware datetimes — "9am Friday in NY" is captured correctly as `hour=9, dow=4` without
  any extra timezone translation. Free correctness from the existing tz-fix
  ([schema/datetime.py:31](../../../src/doppel/schema/datetime.py)).
- `doppel diff` extension: when calendar features are on for a datetime column, the diff
  report includes per-feature marginals (KS on hour/dow/month distributions) under a
  "Calendar fidelity" sub-header. Both terminal and HTML renderers; JSON report includes
  the same data.
- `doppel gen --explain` and `doppel sample --explain` show one line per datetime listing
  the enabled calendar features.

**Out of scope** (deferred or future): `is_weekend` (redundant); `pl.Time` and `pl.Duration`
columns; cyclical (sin/cos) encodings (irrelevant for CART trees); sub-second precision
(separate documented limitation); modeling calendar features as targets.

## Capabilities

### New Capabilities

- `datetime-calendar-features`: extract per-datetime calendar features (`hour`, `dow`,
  `month`, and an allowlist of opt-in extras) into the CART feature matrix for downstream
  modeling; expose schema-level opt-out/customization; surface calendar-feature fidelity
  in the `diff` report.

### Modified Capabilities

None. The `conditional-generation` and `warehouse-connectors` capabilities proposed
elsewhere are orthogonal — calendar features sit inside the synthesis pipeline and don't
touch the predicate evaluator, source/sink dispatch, or auth model.

## Impact

**Code**
- `src/doppel/schema/datetime.py` — add `CalendarFeature` (StrEnum), `default_features_for(dtype)`,
  and `calendar_features(series, features) -> dict[str, pl.Series]`. `decompose` and
  `recompose` stay unchanged.
- `src/doppel/schema/types.py` — extend `Column` with `calendar_features: tuple[CalendarFeature, ...] | None`
  (frozen dataclass; tuple over list for immutability). `None` means "use default for dtype";
  empty tuple means "disabled".
- `src/doppel/schema/toml.py` — parse the `calendar_features` field: `false` → empty tuple;
  list of strings → validated allowlist; missing → `None`. Round-trip preserves the user's
  intent.
- `src/doppel/schema/infer.py` — leave `calendar_features = None` on infer; do not write
  the field to generated TOML.
- `src/doppel/synth/cart.py` —
  - `_prepare_input` builds the calendar-feature columns alongside epoch decomposition.
  - `fit` column loop: after fitting a datetime column, append its calendar features to
    the running feature matrix used by subsequent columns.
  - `sample` column loop: after producing the synth epoch for a datetime column, extract
    calendar features from that synth epoch and append to the running feature matrix.
  - `explain_columns` exposes the active calendar-feature names per datetime column.
- `src/doppel/quality/marginals.py` — add `compute_calendar_marginals(real, synth,
  datetime_cols, features_per_col) -> dict[col_name, dict[feature_name, KSScore]]`.
- `src/doppel/report/terminal.py`, `report/html.py`, `report/json.py` — render the
  "Calendar fidelity" sub-section in each format; HTML and JSON include all per-feature
  KS values; terminal shows a compact summary.

**Tests**
- Calendar-feature extraction correctness across naive, UTC, and tz-aware datetimes.
- Fit + sample on a regression dataset where `amount` is higher on Fridays: with features
  off, dow-conditional means flatten; with features on, the Friday peak survives.
- Determinism: same `--seed` produces byte-identical output with features auto-on and
  auto-off; same seed across two runs with features on produces byte-identical output.
- Schema TOML: opt-out (`false`), customization (list), allowlist enforcement (unknown
  name → `BadParameter`).
- `pl.Date` column: only `dow` and `month` features extracted; no `hour`.
- Multi-datetime dataset: each column gets its own `__dt_<col>_*` prefix; no collision.
- Artifact roundtrip: fit + save + load + sample produces identical results — proves
  feature extraction is deterministic and the artifact format didn't change.
- Diff report: KS values for `hour`/`dow`/`month` appear in terminal output, HTML, and
  JSON when calendar features are on; absent when off.
- `--explain` lists the active features per datetime column.
- Performance smoke: 100k × 100 cols (5 datetimes) completes within 1.3× the time of
  the same dataset with features disabled.

**Dependencies**
- None. Polars `dt` accessors are already available; no new top-level deps.

**Docs**
- README Limitations: drop the bullet stating that hour/dow/business-hours patterns
  aren't preserved.
- README Usage: brief mention of the schema TOML `calendar_features` knob.
- `docs/determinism.md` (if extant): note that calendar features are deterministic pure
  functions of the source datetime; `--seed` reproducibility unchanged.

**Risk**
- Performance on wide datasets with many datetimes: 3 Int8 features × 100k rows ≈ 300KB
  per datetime column. Acceptable; opt-out is available for pathological cases.
- Polars version sensitivity: `dt.weekday()` semantics (0-6 vs 1-7) — pin the test
  expectations and document the offset in `schema/datetime.py`.
- Allowlist creep: the per-column TOML field is the right pressure valve; the auto-default
  stays at three features so the common case remains tight.
