## Context

doppel models each datetime column as `Int64` epoch-seconds. The decompose step
([src/doppel/schema/datetime.py:17](../../../src/doppel/schema/datetime.py)) converts a
temporal Polars dtype to seconds-since-epoch; the recompose step rebuilds the original
dtype after sampling, with timezone preserved via
`dt.replace_time_zone("UTC").dt.convert_time_zone(...)`. The audit's earlier #6 finding
(timezone drop on recompose) is already closed.

CART is fit column-by-column in topological order
([src/doppel/synth/cart.py:447](../../../src/doppel/synth/cart.py)). Each column produces
a `_ColumnSynth` and, via the encoder, contributes a numeric feature vector to the running
feature matrix that downstream columns see. The matrix is `pl.DataFrame` of `Float64`.

The leaf-sampling property is the load-bearing piece for this design: when CART samples
a value for a downstream column, it predicts a leaf id from the synth feature row, then
*randomly picks a real value from that leaf's training pool*. For datetime columns this
means the synth epoch is *literally a source epoch* (or one of them within the leaf). All
calendar features extracted from the synth epoch therefore inherit consistency from a
real source row — no possibility of "Tuesday at 09:30" when the source only had "Monday
at 09:30 or Friday at 14:00". This kills the consistency problem that plagues
multi-target decomposition designs.

This change lands after [add-conditional-where-filter](../add-conditional-where-filter/proposal.md)
and [add-sql-warehouse-connectors](../add-sql-warehouse-connectors/proposal.md). It is
orthogonal to both: predicate evaluation runs after sampling, and source dispatch runs
before fitting; calendar features live inside the fit/sample loop.

## Goals / Non-Goals

**Goals**
- Capture sub-day, weekly, and monthly temporal patterns in downstream column predictions
  with a minimal, contained modeling change.
- Zero artifact-format change. Existing `.doppel` files load unchanged; new artifacts are
  forward-compatible.
- Determinism preserved: calendar features are pure functions of the source datetime; no
  RNG involved.
- Schema-level control: auto-on with a TOML opt-out, per-feature customization within an
  allowlist.
- Make the fidelity gain *visible* in the diff report so users can tell whether the
  feature is working.

**Non-Goals**
- Modeling calendar features as targets. The consistency-by-construction argument depends
  on extracting from the synth epoch, not predicting independently.
- Cyclical (sin/cos) encodings. Useful for linear models, irrelevant for CART trees that
  split on raw integer values.
- `pl.Time` and `pl.Duration` columns. Neither is modeled today; adding calendar features
  for them is a separate design conversation.
- `is_weekend` as a default feature. CART learns it from `dow` with a single split; adding
  it would just duplicate the signal.
- Sub-second precision. Already documented as a v0.1 limitation; not touched.
- A per-column "auto-detect the right features" heuristic. Default by dtype, override by
  TOML. No magic.

## Decisions

### D1. Features go into the matrix only — never as targets.

**Decision.** Calendar features are predictors for downstream columns. They are never
fit as targets, never appear in `_ColumnSynth.leaf_values`, never appear in the output
DataFrame, and never get encoded as user-visible columns. The datetime itself remains
modeled as `Int64` epoch-seconds and recomposed via the existing path.

**Why.** Three properties fall out for free:
1. **Consistency.** Synth datetime is leaf-sampled from real source epochs, so its
   `hour`, `dow`, `month` match a real source row.
2. **No artifact change.** The fitted model doesn't need to store calendar feature data;
   it recomputes them at sample time from the synth epoch using deterministic Polars
   accessors.
3. **No new failure modes.** Modeling calendar features as targets would introduce
   "predicted hour is 14 but predicted epoch is 09:30" inconsistency that we'd have to
   reconcile somehow.

**Alternatives considered.**
- *Decompose into multi-target (date_epoch, hour, dow, ...).* Each modeled separately,
  then reassembled. Pure ergonomics — the model could theoretically generate combinations
  the source never had. Brings a consistency problem (which channel wins?) and a much
  bigger code change. Rejected; not worth the risk for the marginal coverage gain.
- *Add as feature AND target.* Worst of both worlds. Rejected.

### D2. Default feature set: `hour, dow, month` for Datetime; `dow, month` for Date.

**Decision.** Three features by default for `pl.Datetime`, two for `pl.Date`. Each is a
small Int8 column. The allowlist for the TOML customization path is `hour, minute, dow,
month, day_of_month, week_of_year, quarter` — `is_weekend` is intentionally absent.

**Why this set.**
- `dow`: by far the most powerful business signal (weekday vs weekend, weekly cycles).
- `hour`: essential for any sub-day data (business hours, peak times).
- `month`: monthly seasonality, holidays, end-of-quarter spikes.

Each addition past this point either duplicates an existing signal (`is_weekend` ⊂ `dow`,
`quarter` ⊂ `month`, `year` ⊂ `epoch`) or is specialized (`day_of_month` for billing,
`minute` for high-frequency data, `week_of_year` for annual peaks). These are available
on opt-in via the TOML list form.

**Alternatives.**
- *Ship five features by default (add `day_of_month`, `week_of_year`).* Wider matrix, more
  noise on small data, returns are diminishing. Rejected — the opt-in path is sufficient.
- *Auto-detect the useful set per column.* Adds a heuristic to maintain; default-by-dtype
  is predictable. Rejected.

### D3. Auto-on with schema TOML opt-out and per-column customization.

**Decision.** When `ColumnType.DATETIME` is inferred, calendar features are on by default
for that column. TOML control:

```toml
[columns.created_at]
calendar_features = false                  # disable for this column

[columns.event_time]
calendar_features = ["hour", "dow"]        # subset

[columns.billing_date]
calendar_features = ["dow", "month", "day_of_month"]   # extend with opt-in extra
```

Unknown names raise `typer.BadParameter` at TOML load with the allowlist enumerated.

**Why.**
- Auto-on means most users benefit out of the box without ceremony.
- The opt-out keeps the door open for users with weird data (e.g. epoch-as-int columns
  classified as DATETIME by mistake).
- The list form gives power users access to opt-in features without polluting the default.

**Alternatives.**
- *Strictly opt-in.* Nobody would turn it on; defeats the purpose.
- *CLI flag (`--calendar-features=auto|off|custom`).* Per-column control belongs in the
  schema. A single CLI flag is too coarse.

### D4. Internal column naming: `__dt_<col>_<feature>` prefix.

**Decision.** Calendar feature columns inside the CART feature matrix use the prefix
`__dt_<source_col>_<feature_name>` — e.g. `__dt_created_at_hour`, `__dt_created_at_dow`.
The double-underscore prefix matches the existing `__doppel_null__` sentinel pattern
in [schema/nullable.py:19](../../../src/doppel/schema/nullable.py).

A guard at fit time MUST reject any source column whose name starts with `__dt_` — the
collision would silently corrupt the feature matrix.

**Why.**
- Predictable, debuggable names when inspecting the running feature matrix.
- Collision-proof against user column names (real schemas don't lead with double
  underscores).
- Easy to filter out at output time (`for col in features.columns if not col.startswith("__dt_")`).

**Alternative.** *UUID-suffixed names.* Harder to debug; no real benefit.

### D5. Calendar features are extracted from the SYNTH epoch at sample time, not from a separate model.

**Decision.** In the `sample` column loop, after a datetime column produces its synth
epoch series, the same `calendar_features()` function used at fit time is called on the
synth series. The resulting columns are appended to the running feature matrix exactly
like at fit time.

**Why this is consistent by construction.** CART leaf-sampling guarantees the synth epoch
is a real source epoch within the leaf. Its calendar features therefore match a real
source row by definition. No reconciliation logic, no impossible combinations.

**Edge case.** When the synth value is `null` (which happens when the null model fires),
Polars' `dt.hour()` etc. return `null`, so the calendar features for that row are null.
Downstream CART models receive `null` features for that row — the existing nullable
encoding handles this cleanly via `encode_feature` in [schema/nullable.py](../../../src/doppel/schema/nullable.py).

### D6. Polars dt accessors handle timezone correctly without translation.

**Decision.** Use `dt.hour()`, `dt.weekday()`, `dt.month()` directly. For tz-aware
datetimes, these return the **local** representation (the value as displayed in the
column's timezone), which is what we want — "9am Friday in NY" stays `hour=9, dow=4`
regardless of the underlying UTC offset.

For naive datetimes, the accessors operate on whatever the source intended. For UTC
datetimes, they return UTC values. Both correct.

**Why no extra translation.** The existing tz-fix on recompose
([schema/datetime.py:31](../../../src/doppel/schema/datetime.py)) ensures the synth
datetime ends up in the same timezone as the source. The accessors-on-local-rep behavior
means we don't need to do anything special; the calendar feature numbers naturally match
between source and synth.

**Risk.** Polars' `dt.weekday()` returns 1–7 in some versions and 0–6 in others. The test
suite pins the expectation; if Polars changes semantics, tests fail loudly.

### D7. Dtype: Int8 for calendar features; cast to Float64 at the encoder boundary.

**Decision.** Calendar feature columns are emitted as `pl.Int8` (or `pl.UInt8` — pick one
and stick). At the boundary where they enter the CART feature matrix, the existing
`_Encoder.transform` path casts to Float64 alongside every other feature
([synth/cart.py:108](../../../src/doppel/synth/cart.py)).

**Why Int8 at extraction.** Smaller memory footprint (1 byte vs 8) before the cast. The
extraction function may be called many times per synth (once per datetime column per
batch); cheap dtypes win.

**Alternative.** *Float64 from the start.* Wastes memory; no benefit because CART splits
on the value regardless of dtype.

### D8. Multi-datetime ordering: features from prior datetimes are visible to later datetimes.

**Decision.** Calendar features from a datetime column are added to the running feature
matrix as soon as that column is fit/sampled. A later datetime column in topological
order will see the previous datetime's calendar features as predictors for its own epoch
model.

**Why.** Many datasets have multiple correlated datetimes (e.g. `signup_at`, `first_purchase_at`)
where the second's calendar pattern depends on the first. Letting the second see the
first's calendar features lets the model learn "purchase usually happens on the same dow
as signup". This is the same logic as letting any column see its predecessors.

**Edge case.** If a later datetime is a leaf-sampled value from the training rows of the
earlier datetime's leaves, the calendar features are again consistent by the construction
argument from D5.

### D9. Diff report: per-feature marginals under "Calendar fidelity".

**Decision.** When calendar features are on for a datetime column, `doppel diff`
computes KS distances for each enabled feature (`hour`, `dow`, `month`) on both the real
and synth dataframes and renders them in a sub-section. Both terminal and HTML; JSON
includes the same structured data under `calendar_fidelity`.

**Why.** Without this, the value of the feature is invisible. A user who turns on
calendar features has no way to confirm "yes, my weekly pattern survived" — the existing
datetime KS marginal compares whole-epoch distributions and can score "good" even when
the weekly pattern is destroyed.

**Alternative.** *Per-feature marginals only when the report explicitly asks (`--calendar-detail`).*
Conditional UX is harder to discover; default-on with a short summary is better.

### D10. `schema infer` doesn't write `calendar_features` to TOML.

**Decision.** When inferring a schema from data, the generated `schema.toml` MUST NOT
include a `calendar_features` line for any datetime column. Default-by-omission keeps
files terse; users who customize add the line themselves.

**Why.** A field that's present in 99% of files but always set to the default value is
noise. Cleaner TOML for the common case.

**Alternative.** *Write the explicit default to be self-documenting.* Trades terseness for
discoverability; the README/docs cover discoverability better.

### D11. `Column` becomes a frozen dataclass with a `calendar_features: tuple[CalendarFeature, ...] | None` field.

**Decision.** Extend `Column` in [schema/types.py](../../../src/doppel/schema/types.py)
with `calendar_features: tuple[CalendarFeature, ...] | None = None`. Semantics:
- `None` → use the dtype default at fit time.
- `()` (empty tuple) → disabled (the TOML `false` form).
- Non-empty tuple → exactly these features.

Tuple chosen over list because the dataclass is frozen; tuples preserve immutability and
hash cleanly for any future deduplication.

**Alternatives.**
- *Sentinel string `"default"`.* Worse type signature.
- *Two booleans + a list (`auto: bool`, `enabled: bool`, `features: list[str]`).* Three
  fields where one suffices. Rejected.

## Risks / Trade-offs

- [Risk] **Polars `dt.weekday()` semantic drift.** Version-to-version differences could
  shift the dow range. → Mitigation: pin test expectations; if Polars changes, tests fail
  loudly; the calendar feature extractor adds a comment noting the assumed range and the
  Polars version used.

- [Risk] **Feature matrix bloat on wide datasets.** A dataset with 10 datetime columns
  adds 30 Int8 feature columns to the matrix. → Mitigation: ~300KB per datetime per 100k
  rows is negligible; opt-out via TOML for pathological cases.

- [Risk] **Naming collision with user columns starting `__dt_`.** Highly unlikely but
  possible. → Mitigation: fit-time guard raises `ValueError` if any source column name
  starts with `__dt_`.

- [Risk] **The synth datetime is null for some rows (null_model fired).** Calendar
  features extracted from null are null. Downstream CART sees null features for those
  rows. → Mitigation: the existing `encode_feature` path handles null features cleanly
  (median fill for numeric, `__doppel_null__` for categorical). No new code needed.

- [Risk] **Users expect calendar features to be modeled as targets too** ("synth my
  events with the right hour distribution even when there's no other column to anchor
  off"). → Mitigation: README and `--explain` clearly state calendar features are
  predictors only; the diff report's calendar-fidelity section makes the actual behavior
  visible.

- [Risk] **Per-column TOML customization expands the surface to maintain.** → Mitigation:
  allowlist is enforced at parse time with a precise error message; no string-mode magic;
  the customization path covers all known-useful features so the future-feature pressure
  is contained.

## Migration Plan

No data migration. No artifact format change. Existing `.doppel` files load unchanged;
calendar features are applied on the fly at sample time. The `Column` dataclass gains a
new field with a default of `None`, which is backward compatible — existing
unpickle paths populate the field with the default.

Existing test suites pass without modification (calendar features change downstream
synthesis quality, not output shape). The performance-smoke test gates a >1.3× fit-time
regression.

Existing single-datetime fixtures will see (small) changes in synth content because the
feature matrix shape changes — these are not regressions, but golden-file tests that
assert byte-identical output across versions will need updating. The determinism test
within a version (same seed → same output twice) MUST stay green.

## Open Questions

- **Q1.** Should we expose a CLI flag (`--no-calendar-features`) for ad-hoc runs without
  editing the schema? *Recommendation:* no — per-column control belongs in the schema;
  global override via the auto-on default plus the TOML opt-out covers it.

- **Q2.** Should the diff report compute calendar-fidelity for datetime columns *even
  when calendar features are disabled* for that column? *Recommendation:* yes — the
  fidelity numbers are still informative ("you have a weekly pattern your model isn't
  capturing"). The "Calendar fidelity" section runs whenever the source column is
  datetime/date.

- **Q3.** What about a "smart default" that adds `day_of_month` when the source data
  shows a monthly billing pattern? *Recommendation:* defer. Heuristics for this are
  hard to get right; the opt-in list form gives users a clean way to add it themselves.

- **Q4.** Should the `calendar_features` allowlist include `epoch_decade` or `epoch_year`
  (years-since-epoch as an Int)? *Recommendation:* no for v1. The epoch itself already
  encodes year; adding a separate year feature is rarely useful for CART trees.

- **Q5.** Should `--explain` show the resolved feature list per column, or just the
  default-vs-customized status? *Recommendation:* show the resolved list. More
  informative; one line per datetime column.
