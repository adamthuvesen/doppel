## Context

doppel's synthesis pipeline runs each row through a sequential CART model
([src/doppel/synth/cart.py](../../../src/doppel/synth/cart.py)) and then through a
reject-resample constraint engine
([src/doppel/constraints/engine.py](../../../src/doppel/constraints/engine.py)) for
`range` / `inequality` / `derived` constraints. The engine already has geometric
backoff, an unsatisfiable-case error, and is the cleanest insertion point for
predicate-based filtering.

The AST evaluator in [src/doppel/constraints/derived.py](../../../src/doppel/constraints/derived.py)
is a tight whitelist (`Name | Constant(int|float) | UnaryOp(-) | BinOp(+,-,*,/)`).
It emits `polars.Expr` of numeric type. Extending it to emit boolean predicates is
mechanically straightforward, but the security discipline must hold — every new AST
node needs a hostile-input test and a clear reason to be in the allowlist.

The existing change-fixes-2026-05-18 work landed dtype roundtrip, DCR scaling, and
performance regressions; this change is the first real *capability* delta after that
cleanup. The audit (`openspec/custom/issues/audit-2026-05-18.md`) also flagged that
the derived AST has thin hostile-input coverage (#11) — this change closes that gap
in the same parametrised test.

## Goals / Non-Goals

**Goals**
- Ship `--where EXPR` on `doppel gen` and `doppel sample` with byte-identical determinism under `--seed`.
- One AST, two evaluation contexts (numeric for `derived`, boolean for `where`). No second parser.
- Reuse `synthesize_with_constraints` — no new retry loop, no new orchestration code.
- Loud failure when the condition has no support in source data; loud warning when support is thin.
- Close the audit-flagged hostile-input test gap (#11) as part of the same change.

**Non-Goals**
- Rebalancing / stratified sampling. Out of scope for v1; composable from multiple `--where` runs.
- Per-segment conditional CART (refitting the model on the conditioning slice). Real model change.
- Named `[[scenarios]]` block in TOML. v2 thin wrapper over `--where`.
- Cross-table predicates (`users.plan='enterprise' AND orders.amount > 100`). Blocked on multi-table cross-correlations.
- `is`, `in`, `not in`, chained comparisons, `**`, `%`. Deferred — keep the AST narrow.
- Modifying the joint distribution. `--where` is a filter, not a conditioning mechanism; users must accept that the model still has to produce matching rows.

## Decisions

### D1. Extend the existing AST evaluator rather than build a separate predicate parser.

**Decision.** Add `Compare`, `BoolOp`, and `Constant(str | bool)` support inside
`src/doppel/constraints/derived.py` (likely renamed to `expr.py` — see D2). The
evaluator becomes context-aware via an explicit `mode: Literal["numeric", "boolean"]`
parameter on `compile_expression`. Numeric mode accepts only the current node set;
boolean mode additionally accepts `Compare` (top-level only — never inside arithmetic)
and `BoolOp(And, Or)`, with comparands restricted to the existing numeric subgrammar
plus `Constant(str | bool)`.

**Alternatives considered.**
- *Separate Pydantic-structured predicate DSL* (`{column, op, value}` records). Trivially
  safe but exposes an ugly CLI surface (`--filter "plan,==,enterprise"`) and forces
  users to learn two grammars. Two parsers to maintain. Rejected.
- *Hybrid: TOML is structured, CLI is a string mini-parser that emits the structured
  form.* Two grammars, two audit points, more code. Rejected.
- *Use `polars.SQLContext` to evaluate user SQL fragments.* Brings a different security
  surface (Polars SQL parser), no obvious win, and we lose the ability to reject
  hostile nodes by name. Rejected.

**Why the AST extension wins.** Single audit point. Single test parametrisation
catches regressions in both contexts. The audit already flagged that the hostile-input
coverage is thin (#11); doing the extension *now* lets the same test parametrisation
close that gap.

### D2. Rename `constraints/derived.py` to `constraints/expr.py`.

**Decision.** The file already evaluates expressions for `DerivedConstraint`; once it
also evaluates `WhereConstraint`, the name `derived.py` is misleading. Rename to
`expr.py` and re-export `apply`/`compile_expression` from a shim at the old path
**only** if there are external callers — check first, and if there aren't any (highly
likely; this is internal), just rename without a shim.

**Alternatives.**
- *Add a sibling `predicate.py`.* Two files, two parsers, duplicated `_emit` logic.
  Rejected — it's the same parser with one extra mode.
- *Keep the name.* The name lies about scope and will confuse future readers. Rejected.

### D3. `WhereConstraint` becomes the 4th `Constraint` subtype in the Pydantic discriminated union.

**Decision.** New model:

```python
class WhereConstraint(BaseModel):
    kind: Literal["where"] = "where"
    expression: str
    # No `column` field — a where applies across the row.
```

The union becomes `Range | Inequality | Derived | Where`. `_partition` in
`engine.py` grows a fourth return value (or moves to a single `partitioned` dataclass
to keep the signature clean).

**Alternatives.**
- *Reuse `InequalityConstraint`.* It currently encodes column-vs-column comparisons
  only. Forcing column-vs-literal into it muddles the model and breaks existing
  parsing for users who already have inequality constraints in their TOML. Rejected.
- *Embed the predicate inside `RangeConstraint` with optional `expr`.* Same muddle.
  Rejected.

### D4. Boolean-context evaluator returns `pl.Expr` that is provably boolean.

**Decision.** The boolean-context evaluator's top-level node must be a `Compare` or
`BoolOp` whose operands recursively land in `Compare`/`BoolOp`. A bare `Name` or
`BinOp(+)` at the top is rejected with a clear error: "where expression must be a
boolean predicate". This prevents a silent contract slip where the engine receives a
numeric expression and treats nonzero as truthy.

**Alternative.** *Implicitly coerce nonzero → true.* Convenient but hides bugs — a
user typoing `tenure_days * 365` instead of `tenure_days > 365` should see an error,
not get nonsense rows back. Rejected.

### D5. Where filtering reuses `synthesize_with_constraints` — no new engine.

**Decision.** `WhereConstraint` participates in the same violation mask as
`range`/`inequality`. A new helper in `reject.py` (`violation_mask_where`) builds the
boolean mask from the compiled `pl.Expr`; `combined_violation_mask` combines all
sources. The engine's existing geometric-backoff loop handles oversample; the existing
"could not synthesize N rows" error fires when `--max-oversample` is exhausted.

**Alternative.** *Filter outside the engine, after sampling.* Would duplicate the
backoff logic. Rejected.

### D6. Feasibility precheck only in `gen`, not in `sample`.

**Decision.** `gen` has the source DataFrame in memory; cheap to filter and count
matches:
- 0 matches → `typer.BadParameter("no rows in <input> satisfy the where expression …")`.
- 1 ≤ matches < 100 → `console.print` a yellow warning that fidelity will be poor.
- ≥100 matches → silent.

`sample` operates on an artifact — the source data is not available. We **could**
persist a per-column value-count histogram in the artifact, but that's a separate
design question (artifact format change, backward-compat dance). For this change,
`sample --where` skips the precheck; if the condition is empty, the engine will hit
`--max-oversample` and raise the same loud error it raises today for unsatisfiable
constraints.

**Why the asymmetry is acceptable.** `gen` is the common path for new users; the
precheck is highest-value there. Power users running `sample` against a fitted
artifact already understand the model and are tolerant of "we tried, here's the
oversample exhaustion error". Documented in the CLI help.

### D7. `--max-oversample FACTOR` exposed on both commands; default stays 4×.

**Decision.** Surface the existing internal `max_factor` parameter as a CLI flag.
Default unchanged (preserves current behavior for everyone). Type: `float`, must be
≥ 1.0. Wired through to `synthesize_with_constraints(max_factor=…)`.

**Alternative.** *Always run unbounded until satisfied or memory dies.* Unsafe.
Rejected. The current 4× ceiling is a feature, not a bug.

### D8. Multi-table: single-table reference enforced at CLI parse time.

**Decision.** When `gen` is invoked with a multi-table schema (`--schema schema.toml`,
no positional input), `--where` must reference columns that all live in one table.
The CLI parses the expression, collects every `Name` node, and groups them by which
table they appear in via the schema's column index. If they span ≥2 tables, raise
`BadParameter("--where references columns from multiple tables: …")`. The named
table's synth is then run with the where; other tables (parents or siblings) are
synthesized normally.

**Why not cross-table.** Two reasons:
1. Multi-table cross-correlations are not preserved by the current
   `HierarchicalSynthesizer` (CLAUDE.md and the audit both document this). A
   cross-table where would silently produce a misleading result — the filter would
   apply but the children's distributions wouldn't track.
2. Implementing cross-table requires either (a) joining the tables before sampling
   (which breaks the per-table CART contract) or (b) propagating filters through the
   FK edges (which needs cross-table conditional modeling — out of scope).

This must be loud in docs.

### D9. Determinism contract under `--where`.

**Decision.** `(--seed S, --where W, -n N, --max-oversample F)` is the determinism
quadruple. The engine's existing `Rng` plumbing already covers this; the new code
adds no new entropy sources. Regression test: run twice with identical flags, assert
byte-identical output (file hash compare).

**Trap to remember.** The audit flagged (#23) that `Rng.from_seed(None)` is called
from multiple places in `gen.py`. This change doesn't fix #23 but must not make it
worse — the where-mask compute and feasibility precheck use *no* RNG, so they're
seed-independent by construction.

## Risks / Trade-offs

- [Risk] **Expanding the AST whitelist widens the trust surface.** → Mitigation: explicit
  numeric-vs-boolean mode parameter; comprehensive hostile-input test that rejects every
  node we didn't add (Call, Attribute, Subscript, Lambda, IfExp, comprehensions, chained
  Compare, `**`, `%`, dict/list literals, `__import__`, `is`, `in`, `not in`). SECURITY.md
  updated.

- [Risk] **Users assume `--where` on a parent filters children's distributions in multi-table runs.**
  → Mitigation: explicit error if the where references multiple tables; README and CLI
  help spell out the limitation; a `console.print` warning when `--where` is used with
  a multi-table schema (even single-table-scoped) reminding the user that child
  distributions are unconditional.

- [Risk] **Rare conditions (<1% of source) burn CPU before failing.** → Mitigation: feasibility
  precheck in `gen` hard-fails on 0 matches and warns on <100 before fitting; the existing
  `--max-oversample` cap keeps the engine bounded; users can raise the cap explicitly if
  they accept the work.

- [Risk] **Feasibility precheck on `gen` doesn't exist on `sample`, creating UX asymmetry.**
  → Mitigation: documented in CLI help; the engine's oversample-exhaustion error is loud
  enough; persisting feasibility data in the artifact is a separate change.

- [Risk] **Boolean-context evaluator silently accepts a numeric expression at the top level.**
  → Mitigation: top-level node must be `Compare` or `BoolOp`; explicit rejection with a
  helpful error message; tested.

- [Risk] **Chained comparisons (`0 < x < 10`) feel natural to users but are out of scope.**
  → Mitigation: reject with a clear error pointing to the workaround (`0 < x AND x < 10`).
  Tested.

- [Risk] **Renaming `derived.py` to `expr.py` breaks an external caller.** → Mitigation: grep
  the repo before renaming; if any caller exists outside `constraints/` and the test suite,
  leave a re-export shim at the old path with a deprecation comment.

## Migration Plan

No data migration; no artifact format change. New CLI flags are additive. Default behavior
(no `--where`) is byte-identical to today. The constraint TOML loader gains a new `kind`
value; existing TOML files with only `range`/`inequality`/`derived` parse unchanged.

If the `derived.py → expr.py` rename happens, the only follow-up is a one-line import
update in `constraints/engine.py` and tests; CI catches anything else.

## Open Questions

- **Q1.** Do we want a structured form of `WhereConstraint` in `schema.toml` (e.g.
  separate `column`/`op`/`value` fields per predicate, ANDed together) in addition to
  the string `expression` field? Argument for: easier to validate, easier to render
  back to the user. Argument against: users will reach for the string form anyway since
  it matches the CLI flag. *Recommendation:* string form only for v1; structured form
  in a follow-up if users ask.

- **Q2.** Should `--where` on `sample` warn the user that no precheck ran? *Recommendation:*
  no — the engine error is loud enough; an unconditional warning would be noisy for the
  happy path.

- **Q3.** What's the right warning threshold (currently proposed: <100 matches)? Real-world
  test fixtures often work with 10–50 rows. *Recommendation:* keep 100 as the warn threshold,
  add a CLI flag (`--min-support N`) only if users complain.

- **Q4.** Should we emit a per-iteration progress line in the engine when oversampling under
  `--where` (so users see the rare-condition penalty)? *Recommendation:* yes, but route through
  the existing `Rich` console rather than a new logging surface. One line per iteration:
  `attempted=N kept=K factor=Fx`.
