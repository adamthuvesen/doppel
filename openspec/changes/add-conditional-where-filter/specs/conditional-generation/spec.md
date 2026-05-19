## ADDED Requirements

### Requirement: Predicate-based filtering of synthesized output

The `doppel gen` and `doppel sample` commands SHALL accept a `--where EXPR`
option that restricts synthesized output to rows satisfying the given boolean
predicate. The predicate language MUST be a boolean expression over the
columns of a single table. The output row count MUST equal `-n N` (the
requested count) when the predicate is satisfiable within the configured
oversample budget; otherwise the command MUST fail loudly.

The `--where` option MUST compose with `--seed`: the tuple
`(input, --seed, --where, -n, --max-oversample)` MUST yield byte-identical
output across runs.

#### Scenario: Categorical equality on `gen`

- **WHEN** the user runs `doppel gen input.csv -n 10000 --where "plan=='enterprise'" --seed 1 -o out.csv`
- **AND** the source `plan` column has ≥100 rows with value `enterprise`
- **THEN** the command MUST exit 0
- **AND** the output file MUST contain exactly 10000 rows
- **AND** every row MUST have `plan == 'enterprise'`

#### Scenario: Numeric inequality on `gen`

- **WHEN** the user runs `doppel gen input.csv -n 500 --where "tenure_days > 365" --seed 1 -o out.csv`
- **AND** the source has ≥100 rows with `tenure_days > 365`
- **THEN** the command MUST exit 0
- **AND** every output row MUST have `tenure_days > 365`

#### Scenario: Combined And predicate

- **WHEN** the user runs `doppel gen input.csv -n 1000 --where "plan=='enterprise' and churned==true" --seed 1 -o out.csv`
- **AND** the source has ≥100 rows satisfying both conditions
- **THEN** every output row MUST satisfy both conditions

#### Scenario: Combined Or predicate

- **WHEN** the user runs `doppel gen input.csv -n 1000 --where "plan=='enterprise' or plan=='pro'" --seed 1 -o out.csv`
- **THEN** every output row MUST have `plan` in `{'enterprise', 'pro'}`

#### Scenario: Determinism under repeated runs

- **WHEN** the user runs the same `doppel gen` command twice with identical `--seed`, `--where`, `-n`, and `--max-oversample`
- **THEN** the two output files MUST be byte-identical

#### Scenario: `sample` honors `--where` against a fitted artifact

- **WHEN** the user runs `doppel sample model.doppel -n 5000 --where "plan=='enterprise'" --seed 1 -o out.csv`
- **AND** the fitted model can produce rows where `plan == 'enterprise'` within the oversample budget
- **THEN** the command MUST exit 0
- **AND** every output row MUST have `plan == 'enterprise'`

### Requirement: Boolean-context expression grammar

The predicate language SHALL extend the existing arithmetic AST whitelist
(`Name`, `Constant(int|float)`, `UnaryOp(USub)`, `BinOp(+,-,*,/)`) with
exactly the following nodes, in a dedicated boolean evaluation mode:

- `Compare` with exactly one operator from `{==, !=, <, <=, >, >=}` (chained
  comparisons such as `0 < x < 10` MUST be rejected).
- `BoolOp` with `And` or `Or` (mixed chains are allowed; `Not` is NOT in v1).
- `Constant(str)` and `Constant(bool)` as comparands.

The top-level node of a `--where` expression MUST be `Compare` or `BoolOp`; a
bare numeric expression at the top MUST be rejected with the message "where
expression must be a boolean predicate".

All other Python AST nodes MUST be rejected, including but not limited to:
`Call`, `Attribute`, `Subscript`, `Lambda`, `IfExp`, comprehensions, `**`,
`%`, dict/list/tuple/set literals, `is`, `is not`, `in`, `not in`,
f-strings, walrus.

#### Scenario: Single comparison accepted

- **WHEN** the evaluator compiles `tenure_days > 365` in boolean mode
- **THEN** it MUST return a `polars.Expr` of boolean type
- **AND** no error MUST be raised

#### Scenario: Chained comparison rejected

- **WHEN** the evaluator compiles `0 < x < 10` in boolean mode
- **THEN** it MUST raise `ValueError` whose message names "chained comparison"
- **AND** MUST suggest the rewrite `0 < x and x < 10`

#### Scenario: Top-level non-boolean rejected

- **WHEN** the evaluator compiles `tenure_days * 365` in boolean mode
- **THEN** it MUST raise `ValueError` whose message states "where expression must be a boolean predicate"

#### Scenario: Hostile AST nodes rejected (parametrised)

- **WHEN** the evaluator compiles any of: `__import__('os')`, `obj.attr`,
  `a[0]`, `lambda: 1`, `1 if a else 2`, `a ** 2`, `a % 2`, `[1,2]`, `{1: 2}`,
  `a is b`, `a in [1, 2, 3]`, `a not in [1, 2, 3]`, `f"{x}"`, `(x := 1)`
- **THEN** each MUST raise `ValueError` naming the rejected node type

#### Scenario: Unknown column reference rejected

- **WHEN** the evaluator compiles `unknown_col == 1` and `unknown_col` is not in the allowed-columns set
- **THEN** it MUST raise `ValueError` whose message names `unknown_col` and lists the allowed columns

#### Scenario: Numeric-mode evaluator unchanged

- **WHEN** the evaluator compiles `a + b * 2` in numeric mode (the existing `DerivedConstraint` path)
- **THEN** it MUST return a numeric `polars.Expr`
- **AND** it MUST NOT accept any `Compare`, `BoolOp`, str, or bool nodes

### Requirement: `WhereConstraint` participates in the constraint engine

A new `WhereConstraint` model with `kind == "where"` SHALL be added to the
discriminated `Constraint` union. The constraint engine SHALL partition
where constraints alongside `range`/`inequality`/`derived` and combine their
boolean masks into the same violation mask used by the reject-resample loop.
The engine's existing geometric backoff and oversample-exhaustion error
SHALL apply unchanged.

The `schema.toml` `[[constraints]]` array SHALL accept `kind = "where"`
entries with an `expression` string field.

#### Scenario: TOML round-trip

- **WHEN** a `schema.toml` contains `[[constraints]]` with `kind = "where"` and `expression = "plan == 'enterprise'"`
- **THEN** the schema loader MUST parse it into a `WhereConstraint` instance
- **AND** `doppel gen` MUST apply that constraint even without a CLI `--where` flag

#### Scenario: CLI `--where` composes with TOML constraints

- **WHEN** the user runs `doppel gen input.csv --schema s.toml -n 100 --where "tenure_days > 30" -o out.csv`
- **AND** `s.toml` contains a `range` constraint on `amount` (min=0)
- **THEN** every output row MUST satisfy both the where predicate AND the range constraint

#### Scenario: Oversample exhaustion raises the existing error

- **WHEN** `--where` matches <1% of the joint AND `--max-oversample 1.5` is set
- **THEN** the command MUST raise the existing "could not synthesize N rows satisfying constraints" error
- **AND** the error message MUST name the attempted oversample factor

### Requirement: Feasibility precheck in `gen`

`doppel gen` SHALL evaluate the `--where` expression against the source data
*before* fitting the synthesizer and react based on the match count:

- 0 matching rows: hard-fail with `typer.BadParameter` whose message names
  the input path and quotes the expression.
- 1–99 matching rows: emit a warning that fidelity will be poor and proceed.
- 100+ matching rows: proceed silently.

`doppel sample` (artifact-only) SHALL NOT perform the precheck; the engine's
oversample-exhaustion error is the fallback signal.

#### Scenario: Zero-match precheck hard-fails

- **WHEN** the user runs `doppel gen input.csv -n 100 --where "plan == 'nonexistent'" -o out.csv`
- **AND** no source row has `plan == 'nonexistent'`
- **THEN** the command MUST exit with `typer.BadParameter` (exit code 2)
- **AND** the message MUST quote the where expression
- **AND** the message MUST name the input path
- **AND** the synthesizer MUST NOT be fitted

#### Scenario: Thin-match warning, continues

- **WHEN** the source has 50 rows matching `--where`
- **THEN** a warning MUST be emitted on stderr (or via Rich console) noting low support
- **AND** the command MUST proceed to fit and sample

#### Scenario: `sample` skips the precheck

- **WHEN** the user runs `doppel sample model.doppel -n 100 --where "plan == 'nonexistent'" -o out.csv`
- **AND** the fitted model cannot produce rows where `plan == 'nonexistent'`
- **THEN** the command MUST NOT raise a precheck error
- **AND** the engine's oversample-exhaustion error MUST be the failure mode after `--max-oversample` is reached

### Requirement: `--max-oversample` CLI flag

`doppel gen` and `doppel sample` SHALL expose `--max-oversample FACTOR`
where `FACTOR` is a float `≥ 1.0`. The default MUST remain `4.0` (preserving
today's behavior). The value MUST be passed through to
`synthesize_with_constraints(max_factor=FACTOR)`.

#### Scenario: Default factor preserves current behavior

- **WHEN** the user does NOT pass `--max-oversample`
- **THEN** the engine MUST use `max_factor = 4.0`

#### Scenario: Higher factor allows rarer conditions

- **WHEN** `--max-oversample 16.0` is passed AND the condition matches 2% of the joint
- **THEN** the engine MUST exhaust up to 16× before raising

#### Scenario: Invalid factor rejected

- **WHEN** the user passes `--max-oversample 0.5`
- **THEN** the command MUST exit with `BadParameter` naming the minimum 1.0

### Requirement: Multi-table where is single-table-scoped

When `doppel gen` is invoked with a multi-table schema, the `--where`
expression SHALL reference columns from exactly one table. Cross-table
predicates MUST be rejected with `typer.BadParameter` at CLI parse time. The
table whose columns appear in the where MUST be the table to which the
filter applies; other tables (parents, siblings, children) MUST be
synthesized unconditionally, and the documentation MUST state that child
distributions do NOT track parent filters in v1.

#### Scenario: Single-table where on a multi-table schema

- **WHEN** the schema declares `users` and `orders` with an FK from `orders.user_id` to `users.id`
- **AND** the user runs `doppel gen --schema s.toml -n 100 --where "plan == 'enterprise'" -o out/`
- **AND** `plan` is a column of `users`
- **THEN** the `users` synth MUST filter to `plan == 'enterprise'`
- **AND** the `orders` synth MUST run unconditionally
- **AND** FK referential integrity MUST be preserved

#### Scenario: Cross-table where rejected

- **WHEN** the user runs `doppel gen --schema s.toml -n 100 --where "plan == 'enterprise' and amount > 100" -o out/`
- **AND** `plan` is in `users` AND `amount` is in `orders`
- **THEN** the command MUST exit with `BadParameter`
- **AND** the message MUST name both tables and the columns involved
- **AND** the synthesizer MUST NOT be fitted

#### Scenario: Multi-table where emits child-distribution warning

- **WHEN** `--where` is used in a multi-table run (even when single-table-scoped)
- **THEN** the command MUST emit a warning noting that child distributions are unconditional

### Requirement: SECURITY.md and README documentation

[SECURITY.md](../../../../SECURITY.md) SHALL list the boolean-context AST
nodes (`Compare`, `BoolOp`, str/bool constants) under the predicate-evaluator
threat model, mirroring the existing arithmetic node listing.

The README SHALL include at least one `--where` example in Quickstart or
Usage, and a Limitations bullet noting the multi-table cross-table boundary
and that child distributions do not track parent filters in v1.

#### Scenario: README quickstart includes a `--where` example

- **WHEN** a reader scans the Quickstart or Usage section
- **THEN** they MUST see at least one `doppel gen ... --where "…"` invocation

#### Scenario: SECURITY.md enumerates the boolean AST nodes

- **WHEN** a reader scans SECURITY.md's predicate-evaluator section
- **THEN** they MUST find `Compare`, `BoolOp(And, Or)`, and str/bool constants explicitly listed as allowed
- **AND** the explicit rejection of `Call`, `Attribute`, `Subscript`, `Lambda`, `IfExp`, comprehensions, `is`, `in`, and chained comparisons MUST be stated
