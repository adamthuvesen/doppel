## 1. Pre-flight

- [ ] 1.1 Verify Polars `dt.weekday()` semantics on the pinned version (0-6 or 1-7); pin the test expectation and add an inline comment in `schema/datetime.py` noting the assumed range.
- [ ] 1.2 Grep src/ and tests/ for any column name starting with `__dt_` (would block the new prefix). Expected to find none.
- [ ] 1.3 Decide on Polars dtype for the extracted features: confirm `pl.Int8` covers 0-6 (dow), 0-23 (hour), 1-12 (month), 1-31 (day_of_month), 1-53 (week_of_year), 1-4 (quarter), 0-59 (minute). All within range.

## 2. `CalendarFeature` enum and extractor

- [ ] 2.1 In `src/doppel/schema/datetime.py`, add a `CalendarFeature` StrEnum with members `HOUR, MINUTE, DOW, MONTH, DAY_OF_MONTH, WEEK_OF_YEAR, QUARTER`. (`IS_WEEKEND` NOT included.)
- [ ] 2.2 Add `default_features_for(dtype: pl.DataType) -> tuple[CalendarFeature, ...]` returning `(HOUR, DOW, MONTH)` for Datetime, `(DOW, MONTH)` for Date, `()` for Time and Duration.
- [ ] 2.3 Add `calendar_features(series: pl.Series, features: Sequence[CalendarFeature]) -> dict[str, pl.Series]` that returns one Int8 series per requested feature, keyed by feature name (e.g. `"hour" -> pl.Series(...)`).
- [ ] 2.4 Handle null inputs: when the source value is null, the calendar feature value MUST also be null. Polars dt accessors do this by default; add a regression test.
- [ ] 2.5 Unit tests:
  - tz-aware Datetime (`America/New_York`): hour reflects local time.
  - Naive Datetime: hour reflects literal value.
  - Date: dow/month extracted; hour omitted/raises if requested.
  - Time: empty extraction.
  - Duration: empty extraction.
  - All-null series: all-null feature columns of the right dtype.

## 3. Extend the `Column` dataclass

- [ ] 3.1 In `src/doppel/schema/types.py`, add `calendar_features: tuple[CalendarFeature, ...] | None = None` to `Column`. Update the docstring noting `None` = default, `()` = disabled.
- [ ] 3.2 Confirm `Column` stays frozen and hashable.
- [ ] 3.3 Add a helper `Column.resolved_calendar_features(dtype: pl.DataType) -> tuple[CalendarFeature, ...]` that returns the configured tuple or the dtype default.

## 4. TOML loader

- [ ] 4.1 In `src/doppel/schema/toml.py`, extend the column Pydantic model with a `calendar_features` field accepting `bool | list[str] | None`.
- [ ] 4.2 Add a validator: `false` → `()`; `True` → ValidationError ("use omission for default"); `list[str]` → validate every name against the allowlist, raise BadParameter with a precise message on the first unknown name.
- [ ] 4.3 Convert the validated list of strings to `tuple[CalendarFeature, ...]`.
- [ ] 4.4 `apply_overrides` propagates the field onto the inferred `Column`.
- [ ] 4.5 Unit tests:
  - `false` round-trips to `()`.
  - `["hour", "dow"]` round-trips to `(HOUR, DOW)`.
  - `["is_weekend"]` raises BadParameter naming the allowlist.
  - `True` raises ValidationError.
  - Missing field round-trips to `None`.

## 5. `schema infer` behavior

- [ ] 5.1 In `src/doppel/schema/infer.py`, leave `calendar_features = None` on inferred columns (default).
- [ ] 5.2 In TOML serialization (whichever path writes schema.toml), do NOT emit a `calendar_features` line when the value is `None`.
- [ ] 5.3 Regression test: `schema infer` on a datetime fixture produces a TOML without `calendar_features` lines.

## 6. CART pipeline — fit

- [ ] 6.1 In `src/doppel/synth/cart.py`, add a fit-time guard: raise `ValueError` if any source column name starts with `__dt_`.
- [ ] 6.2 In `_prepare_input`, after decomposing each datetime column to epoch_s, compute the resolved feature set via `Column.resolved_calendar_features(source_dtype)` and call `calendar_features(...)`. Store the resulting series in a dict keyed by `(source_col, feature_name)` for use in the fit loop.
- [ ] 6.3 In the fit column loop, after fitting a datetime column, append its calendar feature series to the running feature matrix as `__dt_<source_col>_<feature>` columns (Int8, then encoded via `_Encoder.transform` like every other feature).
- [ ] 6.4 Persist on `_ColumnSynth` (or a sibling) the resolved feature set per datetime column so sample-time extraction matches fit-time semantics.

## 7. CART pipeline — sample

- [ ] 7.1 In the sample column loop, after producing the synth epoch series for a datetime column (still as Int64 epoch_s), reconstruct a temporary Datetime/Date series at that point via `pl.from_epoch` (matching the source dtype) so calendar features can be extracted.
- [ ] 7.2 Apply the same `calendar_features(...)` call with the same resolved feature set; append the resulting Int8 columns to the running feature matrix.
- [ ] 7.3 Verify the synth feature matrix shape per column matches the fit-time shape exactly.

## 8. Diff report — calendar fidelity

- [ ] 8.1 In `src/doppel/quality/marginals.py`, add `compute_calendar_marginals(real: pl.DataFrame, synth: pl.DataFrame, datetime_cols: list[Column]) -> dict[str, dict[str, KSScore]]`. For each datetime/date column, extract calendar features on both frames using the resolved feature set, then KS-test each pair.
- [ ] 8.2 Compute calendar fidelity regardless of the column's `calendar_features` setting — informational.
- [ ] 8.3 Plumb the result through `compute_quality_summary` in `src/doppel/cli/_common.py`.
- [ ] 8.4 Render in `src/doppel/report/terminal.py` under a "Calendar fidelity" sub-section.
- [ ] 8.5 Render in `src/doppel/report/html.py` as a section after marginals.
- [ ] 8.6 Render in `src/doppel/report/json.py` under a `calendar_fidelity` top-level key.
- [ ] 8.7 Unit tests: KS values are finite for a real vs synth datetime fixture; structured shape matches the spec.

## 9. `--explain` integration

- [ ] 9.1 Extend `ColumnFitInfo` in `src/doppel/synth/cart.py` with `calendar_features: tuple[str, ...] | None` (`None` for non-datetime columns; tuple of names for datetime columns; `()` for disabled).
- [ ] 9.2 `CartSynthesizer.explain_columns` populates the new field from the per-column resolved set.
- [ ] 9.3 In the CLI `--explain` renderer (currently in `gen.py` and probably `artifact.py`), add the calendar features line per datetime column.
- [ ] 9.4 Test: `--explain` output contains `calendar=[hour, dow, month]` for a default datetime column; `calendar=[disabled]` for a `false`-configured column.

## 10. Tests — extraction correctness

- [ ] 10.1 tz-aware Datetime: extract hour and assert it matches the local hour (not UTC).
- [ ] 10.2 Naive Datetime: extract hour matches literal hour.
- [ ] 10.3 Date column: extract dow/month; no hour series produced.
- [ ] 10.4 Source value at end-of-year midnight: dow/month/hour all align with the local calendar.
- [ ] 10.5 Null input row produces null feature values; downstream encoding handles them without exception.
- [ ] 10.6 All-null source column: extraction returns all-null Int8 columns.

## 11. Tests — fidelity gain

- [ ] 11.1 Synthetic dataset: `event_time` Datetime + `amount` Float64 where source `amount` is 3× higher on Fridays. Fit with features OFF: dow-conditional mean of synth `amount` is flat (within X% of overall mean). Fit with features ON: Friday peak preserved (within Y% of source Friday mean).
- [ ] 11.2 Synthetic dataset with monthly billing pattern: `created_at` + `is_billing_day`. With features OFF, synth `is_billing_day` is uniform; with features ON (including `day_of_month`), monthly pattern preserved.
- [ ] 11.3 KS-on-dow assertion: with features ON, KS distance for `dow` between source and synth is below threshold; without features ON, KS may exceed it.

## 12. Tests — determinism

- [ ] 12.1 Same seed + features ON twice → byte-identical output (SHA256 compare).
- [ ] 12.2 Same seed + features OFF twice → byte-identical output.
- [ ] 12.3 Same seed + features ON vs OFF → outputs MAY differ (assert nothing about content; just that both runs complete).
- [ ] 12.4 Calendar feature extraction is pure (no RNG): call twice on the same input, assert identical output series.

## 13. Tests — TOML and `--explain`

- [ ] 13.1 TOML with `calendar_features = false` round-trips; synth produces no `__dt_*` columns in the feature matrix (assert via patching/spying on `_prepare_input`).
- [ ] 13.2 TOML with `calendar_features = ["hour"]` produces only `__dt_<col>_hour` (no dow/month).
- [ ] 13.3 TOML with `calendar_features = ["is_weekend"]` raises BadParameter with the allowlist enumerated.
- [ ] 13.4 `--explain` output is asserted in a CLI integration test.

## 14. Tests — multi-datetime and collisions

- [ ] 14.1 Dataset with two Datetime columns (`signup_at`, `purchase_at`): each produces its own prefixed feature set; assert no collision.
- [ ] 14.2 Source column literally named `__dt_evil`: fit raises ValueError naming the column and the reservation.
- [ ] 14.3 `pl.Date` and `pl.Datetime` in the same dataset: each gets its dtype-appropriate default set.

## 15. Tests — artifact roundtrip

- [ ] 15.1 Fit + save + load + sample with features ON: loaded artifact's sample equals the unsaved fit's sample for the same seed.
- [ ] 15.2 Pre-change artifact (a checked-in fixture or one constructed via the old code path): loads cleanly, samples correctly. (Note: the Column dataclass gains a field with a default, so older pickled Columns load unchanged.)
- [ ] 15.3 Artifact size doesn't grow by per-row calendar feature data (compare size with features ON vs OFF on the same model; expect equal within a tolerance).

## 16. Tests — diff report

- [ ] 16.1 Terminal renderer includes "Calendar fidelity" section when the input frames have a datetime column.
- [ ] 16.2 HTML renderer includes the section; assert via parsing the output.
- [ ] 16.3 JSON renderer includes `calendar_fidelity` key with the expected per-feature shape.
- [ ] 16.4 Report includes calendar fidelity even when the column's `calendar_features = false`.

## 17. Tests — performance smoke

- [ ] 17.1 Fit 100k × 100 cols (5 datetime cols, 5 numeric, 90 categorical) with features OFF: record duration.
- [ ] 17.2 Same fit with features ON: record duration.
- [ ] 17.3 Assert duration ratio ≤ 1.3 (the design's target). Skip on slow CI; mark with `@pytest.mark.slow`.

## 18. Docs

- [ ] 18.1 README Limitations: remove the bullet stating that hour/dow/business-hours patterns aren't preserved.
- [ ] 18.2 README Usage: brief subsection on calendar features and the TOML knob, with one customization example.
- [ ] 18.3 SECURITY.md: no changes needed (calendar features add no new trust surface).
- [ ] 18.4 `docs/determinism.md` (if extant): note calendar features are pure deterministic functions of the source datetime.
- [ ] 18.5 CLAUDE.md "Known limitations" section: remove the datetime limitation bullet OR rewrite to note the v1 scope (Datetime + Date, with the allowlist).

## 19. CI gates

- [ ] 19.1 `uv run ruff check src tests` clean.
- [ ] 19.2 `uv run ruff format --check src tests` clean.
- [ ] 19.3 `uv run pyright` 0 errors (strict mode).
- [ ] 19.4 `uv run pytest` green; new tests included; coverage does not regress.
- [ ] 19.5 `uv run doppel gen --help` and `uv run doppel sample --help` mention the calendar-features schema knob (or link to docs).
- [ ] 19.6 `uv run doppel schema infer <fixture>` produces a TOML without `calendar_features` lines (regression check for the omit-by-default rule).
