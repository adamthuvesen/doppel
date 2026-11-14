# doppel

doppel fits a statistical model to a real tabular dataset and writes synthetic rows with the same fingerprint. This file fixes the names the code, the CLI, and the docs share.

## Data

**Dataset**:
A graph of tables linked by foreign keys (`Dataset` in `src/doppel/dataset.py`). A single table is a dataset with one node.
_Avoid_: Database, schema (as a data object)

**Table**:
One named set of columns and rows in a dataset (`Table`).
_Avoid_: Frame, relation

**Column type**:
The role of a column for modelling: `NUMERIC`, `CATEGORICAL`, `DATETIME`, `TEXT`, or `KEY` (`ColumnType` in `src/doppel/schema/types.py`). It is not the Polars data type.
_Avoid_: Kind, dtype

**Key column**:
A column of column type `KEY`. doppel generates fresh values for it and never models the source values.
_Avoid_: ID column

**Primary key**:
The column a table declares as unique. Declaring it makes the column a key column.

**Foreign key**:
A link from a child column to a parent column, declared in `schema.toml` as `[[foreign_keys]]`. doppel never detects one (ADR 0001).
_Avoid_: Relationship, reference

**Real data**:
The source rows doppel reads and fits.
_Avoid_: Original, production data

**Synthetic data**:
The rows doppel writes. Each row is drawn from the fitted model.
_Avoid_: Fake data (that means Faker output), generated data

## Flow

**Source**:
Where real data comes from: a file, or a DuckDB, Snowflake, or Postgres table or query (`src/doppel/sources/`).
_Avoid_: Input, reader

**Sink**:
Where synthetic data goes: a file or a DuckDB table (`src/doppel/sinks/`). A sink never overwrites its source.
_Avoid_: Output, target

**Synthesizer**:
The fitted model that writes synthetic data. Today it is `CartSynthesizer`, with `HierarchicalSynthesizer` for several tables.
_Avoid_: Model, generator

**Fit**:
Learn a synthesizer from real data.
_Avoid_: Train

**Sample**:
Draw synthetic rows from a fitted synthesizer (`doppel sample`, `Synthesizer.sample`).
_Avoid_: Generate (that is the `gen` command, which fits and samples)

**Subsample**:
Select a random subset of real rows. `--fit-rows` does it before the fit, and `--sample-rows` does it before a diff. The code names it `sample_frame` and `sample_fit`, and the flags say "sample".
_Avoid_: Sample (that draws synthetic rows)

**Artifact**:
A `.doppel` file: a fitted synthesizer saved as JSON and a plain npz. It holds real values, so it is as sensitive as its source (ADR 0007).
_Avoid_: Model file, checkpoint

**Schema file**:
The `schema.toml` a user edits to override column types, declare keys, and add constraints. `doppel schema infer` writes a first draft.
_Avoid_: Config, spec

## Synthesis

**Constraint**:
A rule that synthetic rows must meet: `range`, `inequality`, `derived`, or `where` (`src/doppel/constraints/dsl.py`).
_Avoid_: Check, validation rule

**Reject-resample**:
How doppel enforces `range`, `inequality`, and `where` constraints: it drops violating rows and draws more until it has enough. `derived` columns are computed, not rejected.
_Avoid_: Filtering, rejection sampling

**Soft repair**:
A post-fit fix of CART output that keeps a count at or below its bound and keeps a missing-value flag in step with its column (`src/doppel/synth/cart_repair.py`). It is not a constraint.
_Avoid_: Fix-up, cleanup

**Text policy**:
What happens to `TEXT` columns on output: `sample`, `hash`, `fake`, or `drop` (`TextPolicy` in `src/doppel/text_policy.py`). `hash` pseudonymises.
_Avoid_: Text mode, masking

**Calendar feature**:
A part of a datetime (hour, day of week, month) that doppel adds to the CART feature matrix as a predictor for other columns (`src/doppel/schema/calendar_feature.py`). It is not an output column.
_Avoid_: Date part, time feature

**Null sentinel**:
`__doppel_null__`, the category that stands for NULL when a categorical or text column is a feature. Source data that contains it is refused (`NULL_SENTINEL` in `src/doppel/schema/reserved.py`).
_Avoid_: Null token (that is a file value read as NULL, set with `--null-values`)

## Quality

**DCR**:
Distance to closest record: how far each synthetic row is from its nearest real row. doppel reports the 5th percentile, `dcr_p5`. A low value means near-copies.
_Avoid_: Nearest-neighbour distance

**Verbatim rate**:
For a `TEXT` column, the fraction of synthetic values that equal a real value. Reported per column as `verbatim_rate`.
_Avoid_: Leak rate, copy rate

**Exact-row-match rate**:
The share of synthetic rows that equal a real row over the modelled columns (`exact_row_match_rate` in `src/doppel/quality/row_match.py`). doppel sets no threshold for it.
_Avoid_: Duplicate rate, memorisation rate

## Name drift (code not renamed)

- Sample: `--fit-rows`, `--sample-rows`, `--sample-seed`, `sample_frame`, and `sample_fit` all subsample real rows, but their names say "sample". `Synthesizer.sample` and `doppel sample` draw synthetic rows.
- Model: `model.json` and `StoredCart` name a stored synthesizer. `has_value_model` and `_NullModel` name a per-column sub-model, which is a different thing.
- Kind: `kind` is the tag of a constraint (`Constraint.kind`) and a column in the `doctor` table. It never means column type.
