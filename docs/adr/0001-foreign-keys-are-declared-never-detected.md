# Foreign keys are declared, never detected

Status: accepted (2026-11-12)

Multi-table `gen` reads foreign keys only from `[[foreign_keys]]` blocks in `schema.toml` (`ForeignKeySpec` and `to_dataset` in `src/doppel/schema/multi.py`). Nothing in `src/doppel/schema/infer.py` or `heuristics.py` guesses a key, and `doppel schema infer` writes no foreign key. A wrong guess makes the synthesizer link tables that are not linked. The output then looks valid and is wrong. A missing declaration is a loud error instead: `_validate_fks` rejects a key whose table or column does not exist, and the table order comes only from declared edges (`Dataset` in `src/doppel/dataset.py`).

## Considered options

- Detect keys from column names and value overlap, and write them to `schema.toml` for review. Rejected: no source gave a precision we could accept, and a reviewed guess still invites a rubber stamp.
- Detect keys at run time with no review. Rejected: silent and wrong more often than it is useful.

## Consequences

- A user with many tables must write each edge by hand.
- The glossary term **Foreign key** means a declared key only.
