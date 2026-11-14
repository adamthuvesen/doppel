# Constraint expressions run through a small allowlist, with size caps

Status: accepted (2026-11-12)

`derived` constraints and `--where` take user text. `src/doppel/constraints/expr.py` parses it with `ast` and walks the tree under one allowlist. It builds a `polars.Expr` and never calls `eval`, `exec`, or `compile`. Numeric mode allows `Name`, `Constant(int|float)`, `UnaryOp(-)`, and `BinOp(+, -, *, /)`. Boolean mode adds one `Compare` (never chained), `BoolOp(and|or)`, and `Constant(str|bool)`. Calls, attributes, subscripts, comprehensions, and every other node are rejected by node type.

The walker is recursive, so `_MAX_EXPRESSION_CHARS` (10,000) and `_MAX_AST_NODES` (1,000) cap the input before it runs. No source sets these numbers. They sit far above any expression a person writes, and they keep the walker inside the Python stack.

## Considered options

- Pass the text to Polars' SQL or to `eval`. Rejected: the grammar is too wide, and `eval` of user text is a code-execution path.
- A hand-written parser. Rejected: `ast` already parses Python-like text, and the allowlist is the only part we need to own.

## Consequences

- Loosening the allowlist is a security change. Add a node type only with a case that needs it and a test that proves the other node types stay rejected.
- Chained comparisons such as `0 < x < 10` are rejected. Write two comparisons joined by `and`.
