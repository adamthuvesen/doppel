# doppel

doppel fits a statistical model to a real tabular dataset and writes synthetic rows with the same fingerprint. This file fixes the names the code, the CLI, and the docs share.

**Dataset**: A graph of tables linked by foreign keys (`Dataset` in `src/doppel/dataset.py`). A single table is a dataset with one node.
_Avoid_: Database, schema (as a data object)

**Column type**: The role of a column for modelling: `NUMERIC`, `CATEGORICAL`, `DATETIME`, `TEXT`, or `KEY` (`ColumnType` in `src/doppel/schema/types.py`). It is not the Polars data type.
_Avoid_: Kind, dtype

**Key column**: A column of column type `KEY`. doppel generates fresh values for it and never models the source values. Declaring a primary key makes the column a key column.

**Foreign key**: A link from a child column to a parent column, declared in `schema.toml` as `[[foreign_keys]]`. doppel never detects one (ADR 0001).

**Synthesizer**: The fitted model that writes synthetic data. Today it is `CartSynthesizer`, with `HierarchicalSynthesizer` for several tables.
_Avoid_: Model, generator

**Sample**: Draw synthetic rows from a fitted synthesizer (`doppel sample`, `Synthesizer.sample`). `doppel gen` fits and then samples.
_Avoid_: Generate

**Artifact**: A `.doppel` file: a fitted synthesizer saved as JSON and a plain npz. It holds real values, so it is as sensitive as its source (ADR 0007).
_Avoid_: Model file, checkpoint

**Schema file**: The `schema.toml` a user edits to override column types, declare keys, and add constraints. `doppel schema infer` writes a first draft.
_Avoid_: Config, spec

**Sink**: Where synthetic data goes: a file or a DuckDB table (`src/doppel/sinks/`). A sink never writes to a file the run reads: the same CSV or Parquet path, or any table in the source's DuckDB file.
_Avoid_: Output, target

**Constraint**: A rule that synthetic rows must meet: `range`, `inequality`, `derived`, or `where` (`src/doppel/constraints/dsl.py`). doppel drops violating rows and draws more, except `derived` columns, which it computes.
_Avoid_: Check, validation rule

**Soft repair**: A post-fit fix of CART output that keeps a count at or below its bound and keeps a missing-value flag in step with its column (`src/doppel/synth/cart_repair.py`). It is not a constraint.

**Text policy**: What happens to `TEXT` columns on output: `sample`, `hash`, `fake`, or `drop` (`TextPolicy` in `src/doppel/text_policy.py`). `hash` pseudonymises.

**Null sentinel**: `__doppel_null__`, the category that stands for NULL when a categorical or text column is a feature. Source data that contains it is refused (`NULL_SENTINEL` in `src/doppel/schema/reserved.py`).
_Avoid_: Null token (a file value read as NULL, set with `--null-values`)

**DCR**: Distance to closest record: how far each synthetic row is from its nearest real row. doppel reports the 5th percentile, `dcr_p5`. A low value means near-copies.

**Verbatim rate**: For a `TEXT` column, the fraction of synthetic values that equal a real value (`verbatim_rate`).

**Exact-row-match rate**: The share of synthetic rows that equal a real row over the modelled columns (`exact_row_match_rate` in `src/doppel/quality/row_match.py`). doppel sets no threshold for it.

## Name drift (code not renamed)

- Sample: `--fit-rows`, `--sample-rows`, `--sample-seed`, `sample_frame`, and `sample_fit` all subsample real rows, but their names say "sample". `Synthesizer.sample` and `doppel sample` draw synthetic rows.
- Model: `model.json` and `StoredCart` name a stored synthesizer. `has_value_model` and `_NullModel` name a per-column sub-model.
- Kind: `kind` is the tag of a constraint (`Constraint.kind`) and a column in the `doctor` table. It never means column type.
