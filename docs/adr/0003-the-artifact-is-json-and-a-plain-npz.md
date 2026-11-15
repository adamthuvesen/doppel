# The artifact is JSON plus a plain npz, with no pickle

Status: accepted (2026-11-12)

A `.doppel` file is a gzipped tar of `manifest.json`, `model.json`, `arrays.npz`, `schema.json`, and an optional `schema.toml` (`src/doppel/artifact/save.py`). Strict Pydantic models parse the JSON (`src/doppel/synth/cart_state.py`). `numpy.load(..., allow_pickle=False)` reads the arrays, and only `src/doppel/artifact/load.py` may call it. Version 1 stored a pickled sklearn object, and loading an untrusted artifact could run code.

A fitted tree is stored as five arrays, and `DecisionTree` in `src/doppel/synth/tree.py` walks them itself. This reproduces sklearn's routing, so loading builds no sklearn object. The loader checks every member, array, and tree before sampling (`src/doppel/artifact/load.py`).

The format version is now `doppel-artifact-v2` (`ARTIFACT_VERSION` in `src/doppel/artifact/manifest.py`). There is no migration. The loader refuses a v1 artifact, which holds `synth.pickle`, before it parses any member, and `doppel sample` tells the user to refit. We chose this because nothing was published, so no user holds a v1 file, and a migrator would have to unpickle the old file, which is the risk we removed.

## Considered options

- Keep the pickle and check an allowlist of classes. Rejected: a class allowlist is the interim fix, not a guarantee, and new sklearn state breaks it.
- Use `skops` or `joblib`. Rejected: both still deserialise objects and need a trust step.
- Migrate v1 files. Rejected: see above.

## Consequences

- New synthesizer state goes into `CartSynthesizer.to_state` and `from_state`, and bumps `ARTIFACT_VERSION`.
- Ruff bans `pickle`, `marshal`, `joblib`, and `numpy.load` outside the owner file (`pyproject.toml`).
- A scikit-learn upgrade cannot break a saved artifact, because routing no longer runs through sklearn.
