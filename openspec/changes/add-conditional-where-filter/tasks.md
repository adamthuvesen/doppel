## 1. Pre-flight

- [ ] 1.1 Grep the repo for any external caller of `doppel.constraints.derived` (outside `constraints/` and `tests/`); decide rename vs shim per design D2.
- [ ] 1.2 Confirm `openspec/specs/` does not exist yet; verify this change creates the first capability spec.

## 2. DSL — `WhereConstraint`

- [ ] 2.1 Add `WhereConstraint` Pydantic model in `src/doppel/constraints/dsl.py` with `kind: Literal["where"]` and `expression: str`.
- [ ] 2.2 Extend the discriminated `Constraint` union to include `WhereConstraint`.
- [ ] 2.3 Update `src/doppel/schema/toml.py` to accept `kind = "where"` entries in the `[[constraints]]` array.
- [ ] 2.4 Unit test: `kind = "where"` parses into a `WhereConstraint`; unknown `kind` still raises (regression).

## 3. Expression evaluator — extend the AST

- [ ] 3.1 Rename `src/doppel/constraints/derived.py` → `src/doppel/constraints/expr.py` (or leave the rename and add a shim per design D2 decision in 1.1).
- [ ] 3.2 Add a `mode: Literal["numeric", "boolean"]` parameter to `compile_expression` (default `"numeric"` for back-compat).
- [ ] 3.3 In boolean mode, extend `_emit` to handle: `Compare` (single op only — reject chained), `BoolOp(And, Or)`, `Constant(str)`, `Constant(bool)`. Comparands recurse via the existing numeric subgrammar plus the new string/bool constants.
- [ ] 3.4 In boolean mode, enforce that the top-level node is `Compare` or `BoolOp`; raise `ValueError("where expression must be a boolean predicate")` otherwise.
- [ ] 3.5 In numeric mode, explicitly reject `Compare`, `BoolOp`, str, and bool — the numeric path's surface MUST NOT widen.
- [ ] 3.6 Update `apply()` in the same module to take a `mode` param routed from the caller.
- [ ] 3.7 Update all import sites (`constraints/engine.py`, tests).

## 4. Engine — route `WhereConstraint`

- [ ] 4.1 Extend `_partition` in `src/doppel/constraints/engine.py` to return where constraints alongside derived/range/inequality (consider a `Partitioned` dataclass to keep the return signature clean).
- [ ] 4.2 Add `violation_mask_where(df, where_constraints) -> (mask, counts)` in `src/doppel/constraints/reject.py`, mirroring the shape of `combined_violation_mask`.
- [ ] 4.3 Update `combined_violation_mask` (or add a sibling) to OR the where mask into the existing range/inequality mask.
- [ ] 4.4 Update `synthesize_with_constraints` to compile each `WhereConstraint.expression` once in boolean mode at the start of the loop and reuse the compiled `pl.Expr` per iteration.

## 5. CLI — `--where` and `--max-oversample`

- [ ] 5.1 Add `--where EXPR` option to `src/doppel/cli/gen.py`.
- [ ] 5.2 Add `--max-oversample FACTOR` option (float, ge=1.0) to `src/doppel/cli/gen.py`; thread it through to `synthesize_with_constraints(max_factor=…)`.
- [ ] 5.3 Add the same two options to the `sample` subcommand in `src/doppel/cli/artifact.py`; thread through identically.
- [ ] 5.4 In `gen`, after schema inference but before fit, run the feasibility precheck against the source DataFrame: compile `--where` in boolean mode against the source columns, evaluate as a mask, count matches; apply 0-match BadParameter / <100-match warn / silent thresholds per design D6.
- [ ] 5.5 In `gen` with a multi-table schema, collect every `Name` in the parsed where expression, group by table via the schema column index, and reject with `BadParameter` if columns span >1 table.
- [ ] 5.6 In `gen` with a multi-table schema, emit a one-line warning that child distributions are unconditional even when the where is single-table-scoped.
- [ ] 5.7 In the engine loop, route the existing per-iteration accounting (`attempted`, `kept`, `factor`) to a one-line progress print (Rich console) when `--where` is in play.

## 6. Tests — hostile-input coverage

- [ ] 6.1 Parametrised rejection test for boolean mode: `__import__('os')`, `obj.attr`, `a[0]`, `lambda: 1`, `1 if a else 2`, `a ** 2`, `a % 2`, `[1,2]`, `{1: 2}`, `(1, 2)`, `a is b`, `a in [1,2,3]`, `a not in [1,2,3]`, `f"{x}"`, `(x := 1)`. Each must raise `ValueError` naming the rejected node type. (Closes audit gap #11.)
- [ ] 6.2 Chained-comparison rejection: `0 < x < 10` raises with message naming "chained comparison" and suggesting the `and` rewrite.
- [ ] 6.3 Top-level non-boolean rejection: `tenure_days * 365` raises with message stating "where expression must be a boolean predicate".
- [ ] 6.4 Numeric-mode regression: `_emit` in numeric mode still rejects `Compare`, `BoolOp`, str, bool — the existing `derived` path must not widen.

## 7. Tests — happy path

- [ ] 7.1 `--where "plan == 'enterprise'"` on a small categorical fixture: exit 0, all output rows match.
- [ ] 7.2 `--where "tenure_days > 365"` on a numeric column: exit 0, all output rows match.
- [ ] 7.3 Combined `and`: every row satisfies both predicates.
- [ ] 7.4 Combined `or`: every row satisfies at least one predicate.
- [ ] 7.5 TOML-declared where (no CLI flag): persisted in `schema.toml`, picked up by `gen`, applied identically.
- [ ] 7.6 CLI `--where` composes with a TOML `range` constraint: every output row satisfies both.

## 8. Tests — determinism and engine behavior

- [ ] 8.1 Determinism: two `gen` runs with identical `(seed, where, n, max-oversample)` produce byte-identical files (compare via SHA256).
- [ ] 8.2 Feasibility precheck: 0-match input raises `BadParameter` with message quoting the expression and naming the path; synthesizer is NOT fitted (assert via patching `CartSynthesizer.fit` to fail).
- [ ] 8.3 Feasibility precheck: 50-match input emits warning on stderr/console, command still exits 0.
- [ ] 8.4 Feasibility precheck: 1000-match input emits no warning.
- [ ] 8.5 Oversample exhaustion: rare condition (~0.5% match) with `--max-oversample 1.5` raises the existing "could not synthesize N rows" error.
- [ ] 8.6 `sample --where` on a fitted artifact works without source data; skipping precheck does not crash.

## 9. Tests — multi-table

- [ ] 9.1 Single-table where on a multi-table schema: parent synth filtered, child synth unconditional, FK integrity preserved.
- [ ] 9.2 Cross-table where: `BadParameter` at parse time, mentions both tables and the offending columns, synthesizer not fitted.
- [ ] 9.3 Multi-table `--where` (even single-table-scoped) emits the child-distribution warning.

## 10. Docs

- [ ] 10.1 README: add a `--where` example to the Quickstart or Usage section.
- [ ] 10.2 README Limitations: add a bullet stating that `--where` is single-table-scoped and child distributions are unconditional in v1.
- [ ] 10.3 SECURITY.md: list `Compare`, `BoolOp(And, Or)`, str/bool constants under the predicate-evaluator surface; enumerate the explicit rejects (Call, Attribute, Subscript, Lambda, IfExp, comprehensions, chained Compare, `**`, `%`, dict/list/tuple/set literals, `is`, `in`, `not in`, f-string, walrus).
- [ ] 10.4 `docs/determinism.md` (if extant): add `--where` to the determinism contract.

## 11. CI gates

- [ ] 11.1 `uv run ruff check src tests` clean.
- [ ] 11.2 `uv run ruff format --check src tests` clean.
- [ ] 11.3 `uv run pyright` 0 errors (strict mode, unchanged).
- [ ] 11.4 `uv run pytest` green; new tests included; coverage does not regress.
- [ ] 11.5 `uv run doppel gen --help` shows the new `--where` and `--max-oversample` flags with the expected descriptions.
- [ ] 11.6 `uv run doppel sample --help` shows the new `--where` and `--max-oversample` flags with the expected descriptions.
