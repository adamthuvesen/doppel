## ADDED Requirements

### Requirement: Calendar features as downstream-only CART predictors

When a column has type `ColumnType.DATETIME` and its source dtype is `pl.Datetime`
or `pl.Date`, the synthesizer SHALL extract calendar features from the column
and inject them into the CART feature matrix as predictors for subsequent
columns. Calendar features MUST NOT be modeled as CART targets, MUST NOT appear
in `_ColumnSynth.leaf_values`, and MUST NOT appear in the synthesizer's output
DataFrame.

The datetime column itself MUST continue to be modeled as `Int64` epoch-seconds
and recomposed via the existing `decompose`/`recompose` path. The output dtype
roundtrip MUST be unchanged.

#### Scenario: Calendar features appear in the feature matrix

- **WHEN** a CART synthesizer is fit on a DataFrame with a `pl.Datetime` column `created_at` followed by a `pl.Float64` column `amount`
- **AND** calendar features are enabled for `created_at` (default behavior)
- **THEN** the feature matrix used to fit the `amount` model MUST contain columns named `__dt_created_at_hour`, `__dt_created_at_dow`, and `__dt_created_at_month`
- **AND** the values in those columns MUST match `created_at.dt.hour()`, `created_at.dt.weekday()`, and `created_at.dt.month()` (in the column's local timezone if tz-aware)

#### Scenario: Calendar features do not appear in output

- **WHEN** the synthesizer samples a Dataset
- **THEN** the output DataFrame columns MUST equal the original input columns (no `__dt_*` columns)

#### Scenario: Datetime is still modeled as epoch-seconds

- **WHEN** the synthesizer is fit on a `pl.Datetime` column with timezone `America/New_York`
- **AND** the synthesizer samples N rows
- **THEN** the output `created_at` column MUST have dtype `pl.Datetime` with timezone `America/New_York`
- **AND** the wall-clock values MUST round-trip without UTC offset shift (existing tz-fix unchanged)

### Requirement: Sample-time calendar features extracted from synth epoch

At sample time, after the synthesizer has produced the synth epoch series for
a datetime column, the synthesizer SHALL call the same calendar-feature
extraction function used at fit time on the synth series and append the
resulting columns to the running feature matrix used by subsequent columns.

Because the synth datetime is leaf-sampled from real source epoch values,
calendar features extracted from the synth epoch MUST be consistent with a real
source row by construction.

#### Scenario: Synth calendar features are valid source-row combinations

- **WHEN** the source data has datetimes only on Mondays and Fridays
- **AND** the synthesizer samples 1000 rows of a Datetime column
- **THEN** the extracted `__dt_<col>_dow` values for the synth column MUST be a subset of the source's observed `dow` values (i.e. only Mondays and Fridays)

#### Scenario: Null synth datetime produces null calendar features

- **WHEN** the synth epoch for a row is null (the null_model fired)
- **THEN** the corresponding calendar feature values for that row MUST be null
- **AND** the existing nullable encoding path MUST handle the nulls cleanly (no exceptions raised)

#### Scenario: Multi-datetime ordering

- **WHEN** the synth schema declares `signup_at` (Datetime) followed by `first_purchase_at` (Datetime)
- **AND** calendar features are enabled for both
- **THEN** the feature matrix used to fit/sample `first_purchase_at` MUST contain `__dt_signup_at_*` features
- **AND** the feature matrix used to fit/sample any column after `first_purchase_at` MUST contain both `__dt_signup_at_*` and `__dt_first_purchase_at_*` features

### Requirement: Default feature set by source dtype

The synthesizer SHALL apply the following default calendar feature sets when
the column's `calendar_features` field is `None` (the unset default):

- `pl.Datetime` columns → `(hour, dow, month)` — three Int8 features.
- `pl.Date` columns → `(dow, month)` — two Int8 features (no time-of-day).
- `pl.Time` columns → no calendar features (out of scope for v1).
- `pl.Duration` columns → no calendar features (not modeled today).

#### Scenario: Datetime column gets three features

- **WHEN** a `pl.Datetime` column has `calendar_features = None` in its schema
- **THEN** exactly three feature columns MUST be added to the matrix: `__dt_<col>_hour`, `__dt_<col>_dow`, `__dt_<col>_month`

#### Scenario: Date column gets two features

- **WHEN** a `pl.Date` column has `calendar_features = None` in its schema
- **THEN** exactly two feature columns MUST be added: `__dt_<col>_dow`, `__dt_<col>_month`
- **AND** no `__dt_<col>_hour` column MUST appear

#### Scenario: Time and Duration columns get no features

- **WHEN** a column has source dtype `pl.Time` or `pl.Duration`
- **THEN** no calendar feature columns MUST be added to the matrix
- **AND** no error MUST be raised

### Requirement: Schema TOML opt-out and customization

The `Column` dataclass SHALL gain a `calendar_features: tuple[CalendarFeature, ...] | None`
field. The schema TOML loader SHALL parse a `calendar_features` field per column
with the following semantics:

- Field missing or unset → `calendar_features = None` (use dtype default).
- `calendar_features = false` → `calendar_features = ()` (empty tuple, disabled).
- `calendar_features = ["hour", "dow"]` → `calendar_features = (CalendarFeature.HOUR, CalendarFeature.DOW)`.
- `calendar_features = true` is NOT supported (use omission for default).

The allowlist for the list form is exactly: `hour, minute, dow, month,
day_of_month, week_of_year, quarter`. `is_weekend` and any other name MUST
raise `typer.BadParameter` at TOML load with a message naming the offending
value and the allowlist.

The Pydantic model SHALL validate the list as a sequence of allowlist members
before construction.

#### Scenario: Opt-out via `false`

- **WHEN** a schema TOML contains `[columns.created_at]` with `calendar_features = false`
- **THEN** the loaded `Column.calendar_features` MUST equal the empty tuple `()`
- **AND** the synthesizer MUST add no `__dt_created_at_*` features to the matrix

#### Scenario: Customization via list

- **WHEN** the TOML contains `calendar_features = ["hour", "dow"]`
- **THEN** the loaded value MUST equal `(CalendarFeature.HOUR, CalendarFeature.DOW)`
- **AND** only `__dt_<col>_hour` and `__dt_<col>_dow` MUST appear in the matrix; no `__dt_<col>_month`

#### Scenario: Extending with opt-in extras

- **WHEN** the TOML contains `calendar_features = ["dow", "month", "day_of_month"]`
- **THEN** the loaded value MUST include `CalendarFeature.DAY_OF_MONTH`
- **AND** the matrix MUST contain `__dt_<col>_day_of_month`

#### Scenario: Unknown feature rejected at TOML load

- **WHEN** the TOML contains `calendar_features = ["is_weekend"]`
- **THEN** the schema loader MUST raise `BadParameter`
- **AND** the message MUST name `is_weekend` as unknown
- **AND** the message MUST enumerate the allowlist

#### Scenario: `schema infer` omits the field

- **WHEN** the user runs `doppel schema infer input.csv -o schema.toml` on a file with datetime columns
- **THEN** the generated `schema.toml` MUST NOT contain a `calendar_features` key for any column
- **AND** the synthesizer fit against this schema MUST use the dtype defaults

### Requirement: Collision-proof internal naming

Calendar feature columns inside the CART feature matrix MUST use the prefix
`__dt_<source_col>_<feature_name>`. At fit time, the synthesizer SHALL raise
`ValueError` if any source column name begins with `__dt_`. The double-underscore
prefix matches the existing `__doppel_null__` sentinel convention.

#### Scenario: User column named `__dt_evil` rejected

- **WHEN** a source DataFrame contains a column named `__dt_evil`
- **AND** the synthesizer's `fit` is called
- **THEN** `ValueError` MUST be raised
- **AND** the message MUST name the offending column and explain the `__dt_` reservation

### Requirement: Determinism

Calendar feature extraction MUST be a deterministic pure function of the input
series — no RNG calls. The `--seed` reproducibility contract is preserved
regardless of whether calendar features are enabled, disabled, or customized.

#### Scenario: Same seed reproduces output with features on

- **WHEN** the same source data is synthesized twice with identical `--seed` and calendar features enabled
- **THEN** the two output files MUST be byte-identical

#### Scenario: Toggling features changes output but stays deterministic

- **WHEN** the same source data is synthesized with `--seed 1` and calendar features OFF, then again with `--seed 1` and calendar features ON
- **THEN** the two outputs MAY differ (feature matrix shape changed)
- **AND** each is deterministic across repeated runs with its respective configuration

### Requirement: Diff report surfaces calendar fidelity

The `doppel diff` command SHALL compute per-feature marginal KS distances for
each datetime/date column on the real and synth DataFrames, using each column's
**resolved** calendar feature set (default if `None`, configured otherwise).
The result MUST appear in:

- The terminal report under a "Calendar fidelity" sub-header.
- The HTML report under a section visually parallel to the existing
  marginals/correlations sections.
- The JSON report under a `calendar_fidelity` key keyed by column name.

When a datetime column has `calendar_features = ()` (disabled), the diff report
SHALL still compute and display calendar fidelity for that column — it is
informative ("you have a weekly pattern your model isn't capturing") and
deciding when to render based on the column's config is more complex than
worth.

#### Scenario: Terminal report includes calendar fidelity

- **WHEN** `doppel diff real.csv synth.csv` is run AND the real frame contains a `pl.Datetime` column
- **THEN** the terminal output MUST contain a "Calendar fidelity" section
- **AND** that section MUST list KS values for `hour`, `dow`, `month` for the datetime column

#### Scenario: HTML report includes calendar fidelity

- **WHEN** `doppel diff real.csv synth.csv -o report.html` is run
- **THEN** the HTML file MUST contain a section titled "Calendar fidelity"
- **AND** that section MUST include a per-feature KS table for each datetime/date column

#### Scenario: JSON report includes calendar fidelity

- **WHEN** `doppel diff real.csv synth.csv --json report.json` is run
- **THEN** the JSON MUST contain a top-level `calendar_fidelity` key
- **AND** the value MUST be a mapping of column name → mapping of feature name → KS value

#### Scenario: Calendar fidelity computed even when features disabled

- **WHEN** the column has `calendar_features = false` in the schema
- **AND** the diff is computed against real and synth frames containing that column
- **THEN** the report MUST still include the calendar fidelity numbers for that column

### Requirement: `--explain` exposes the resolved feature list

The `doppel gen --explain` and `doppel sample --explain` outputs SHALL include
one line per datetime column listing the resolved (post-default-application)
calendar feature names. The `CartSynthesizer.explain_columns` method SHALL
expose the same information programmatically.

#### Scenario: `--explain` shows calendar features

- **WHEN** the user runs `doppel gen input.csv -n 10 -o out.csv --explain`
- **AND** `input.csv` has a `pl.Datetime` column named `event_time` with default features
- **THEN** the explain output MUST include a line like `event_time (datetime): calendar=[hour, dow, month]`

#### Scenario: `--explain` shows disabled state

- **WHEN** the column has `calendar_features = false` in the schema
- **THEN** the explain line MUST be `<col> (datetime): calendar=[disabled]` (or equivalent)

### Requirement: Polars timezone behavior leveraged without translation

Calendar feature extraction MUST use Polars' `dt.hour()`, `dt.weekday()`,
`dt.month()`, and related accessors directly. For tz-aware datetime columns,
these accessors return values in the column's local timezone, which is the
correct semantic for capturing "9am Friday in NY" patterns. No additional
timezone translation MUST be performed in calendar feature extraction.

#### Scenario: tz-aware datetime extracts local hour

- **WHEN** the source has a `pl.Datetime[μs, America/New_York]` column with values like `2024-12-06 09:00:00 EST` (= 14:00 UTC)
- **THEN** the extracted `__dt_<col>_hour` MUST equal 9 (the local hour), not 14 (the UTC hour)

#### Scenario: Naive datetime extracts as-is

- **WHEN** the source has a `pl.Datetime[μs]` column (naive) with value `2024-12-06 09:00:00`
- **THEN** the extracted `__dt_<col>_hour` MUST equal 9

### Requirement: No artifact format change

The `.doppel` artifact format MUST NOT change. Existing artifacts saved before
this change MUST load unchanged via the safe-pickle path. Calendar features
MUST be recomputed at sample time from each synth epoch using the deterministic
extraction function; no per-feature data MUST be serialized into the artifact.

#### Scenario: Pre-existing artifact loads unchanged

- **GIVEN** a `.doppel` artifact created before this change
- **WHEN** `doppel sample <artifact> -n 100 -o out.csv` is run
- **THEN** the command MUST exit 0
- **AND** the output MUST match the pre-change synthesizer's output (or its semantic equivalent)

#### Scenario: New artifact roundtrip

- **WHEN** a fit + save + load + sample cycle is run with calendar features enabled
- **THEN** the loaded artifact's sample output MUST equal the unsaved fit's sample output for the same seed
- **AND** the artifact size MUST NOT have grown by per-row calendar feature data
