# Known limitations (v0.1)

Design choices, not bugs — don't "fix" these without scope agreement.

- **Datetime modelling: epoch-seconds + calendar features.** The datetime itself is
  modeled as Int64 epoch_s; `hour`/`dow`/`month` (or `dow`/`month` for `pl.Date`) are
  injected into the CART feature matrix as predictors for downstream columns. Override
  per-column in `schema.toml` (`calendar_features = false`, or a list of allowlisted
  feature names). Sub-second precision is still dropped.
- **Multi-table cross-correlations are not preserved.** Per-table CART is fit
  independently; FK integrity holds, but "gold users place bigger orders" does not.
  `inherit_parent_features` schema flag is parsed (raises `NotImplementedError` until
  wired). See `synth/hierarchy.py` docstring.
- **Free-text columns without detected PII** are sampled with replacement and **may
  leak original strings**. `diff`'s DCR percentile + per-column verbatim_rate are the
  user-facing signal.
- **`fit` refuses detected PII.** The artifact format doesn't yet carry detection
  metadata for round-trip regeneration. Use `gen` for one-shot PII regeneration.
  v0.2 roadmap.
- **No differential privacy in v0.1.** `--epsilon` is v0.2 roadmap.

Integer + float subtypes now round-trip — Int32/Int64/UInt*/Float32/Float64 all
preserved via `_INTEGER_DTYPE_NAMES`/`_FLOAT_DTYPE_NAMES` in `synth/cart.py`.
