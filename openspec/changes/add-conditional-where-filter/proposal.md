## Why

doppel can only sample iid from the joint. Users cannot ask for "10k rows where `plan='enterprise' and churned=true`" — the top capability gap per [openspec/custom/reviews/full-codebase-audit-2026-05-17.md](../../custom/reviews/full-codebase-audit-2026-05-17.md) and the single highest-leverage feature for the developer/test-fixture positioning the audit recommends. SDV, Gretel, and MOSTLY all expose this; without it, doppel is harder to reach for whenever a user needs scenario-specific test data, edge-case coverage, or a slice for a downstream ML experiment.

## What Changes

- Add `--where EXPR` to `doppel gen` and `doppel sample` (artifact-only). Same flag, same semantics in both commands.
- Add a new `WhereConstraint` kind (`kind = "where"`) to the constraint DSL so the same predicate can be declared in `schema.toml` and persisted.
- Extend the AST evaluator in `src/doppel/constraints/derived.py` to support boolean predicates: `Compare` (`==`, `!=`, `<`, `<=`, `>`, `>=`), `BoolOp` (`And`, `Or`), and `Constant(str | bool)`. Numeric-context evaluation stays exactly as it is today; the new boolean-context path is gated by an explicit caller flag — no implicit type inference. Single-comparison only; chained `0 < x < 10` is rejected.
- Route `WhereConstraint` through `synthesize_with_constraints` — it contributes to the same violation mask as `range`/`inequality`/`derived`. No new engine, no new retry loop.
- `gen` runs a **feasibility precheck** against the source data before fitting: 0 matches → hard fail with `typer.BadParameter`; <100 matches → warn but proceed; ≥100 matches → silent. `sample` (artifact-only) skips the precheck.
- Expose `--max-oversample FACTOR` on both commands so users can opt into more reject-resample work when the condition is rare. Default stays at the engine's current 4×.
- Multi-table: a `--where` expression must reference columns from exactly one table. Cross-table predicates (e.g. `users.plan='enterprise' AND orders.amount > 100`) are rejected with a clear message. Filtering a parent table does **not** propagate to child distributions — that gap is blocked on the multi-table cross-correlation work and must be documented.

**Out of scope** (deferred to future changes): rebalancing/stratification, per-segment conditional CART, named TOML `[[scenarios]]` blocks, cross-table predicates, `is`/`in`/`not in` operators.

## Capabilities

### New Capabilities

- `conditional-generation`: predicate-based filtering of synthesized output via `--where` on `gen` and `sample`, with a boolean-context AST evaluator, feasibility precheck, `WhereConstraint` DSL, and a documented multi-table scope limit.

### Modified Capabilities

None. No existing spec captures the constraint DSL yet (this repo has no `openspec/specs/` directory before this change); the new capability stands alone.

## Impact

**Code**
- `src/doppel/constraints/dsl.py` — add `WhereConstraint` and extend the discriminated union.
- `src/doppel/constraints/derived.py` — extend `_emit` and `compile_expression` with an explicit boolean-vs-numeric context flag; add the new AST node handlers.
- `src/doppel/constraints/reject.py` — add a where-mask helper that mirrors the existing `combined_violation_mask` shape.
- `src/doppel/constraints/engine.py` — route `WhereConstraint` through `_partition` and into the violation mask.
- `src/doppel/cli/gen.py` — `--where`, `--max-oversample`, single-table reference check, feasibility precheck.
- `src/doppel/cli/artifact.py` — `sample` subcommand: `--where`, `--max-oversample`, no precheck.
- `src/doppel/schema/toml.py` — accept `kind = "where"` entries in the `[[constraints]]` array.

**Tests**
- Parametrised hostile-input coverage on the expanded AST (closes audit gap #11): rejects `Call`, `Attribute`, `Subscript`, `Lambda`, `IfExp`, comprehensions, chained Compare, `**`, `%`, dict/list literals, `__import__('os')`, `is`, `in`, `not in`.
- Happy paths: categorical equality, numeric inequality, combined `And`/`Or`.
- Feasibility precheck: 0/<100/≥100-match behaviors.
- Determinism: same seed + same `--where` → byte-identical output across two runs.
- Multi-table: cross-table where rejected with a clear message; single-table where works inside a multi-table run.
- Oversample exhaustion: rare condition + low `--max-oversample` raises the existing engine error.
- `sample` with `--where` on a fitted artifact works (no source data required).

**Docs**
- README: add a `--where` example to Quickstart/Usage, and a paragraph in Limitations noting that for multi-table runs `--where` does not propagate to child distributions.
- SECURITY.md: list the new AST nodes in the predicate-evaluator surface so the threat model stays explicit.
- `docs/determinism.md` (if extant): note that `--where` composes deterministically with `--seed`.

**Dependencies**
- None. Pure stdlib `ast` + existing `polars`. No new top-level requirements.

**Risk**
- Expanding the AST whitelist is the only meaningful new surface. Mitigated by (a) keeping the boolean-vs-numeric context explicit, (b) the new hostile-input test parametrisation, and (c) the same exhaustive rejection pattern that already protects `derived`.
