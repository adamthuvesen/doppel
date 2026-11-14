# A Polars null is the canonical NULL, and only the empty string reads as NULL

Status: accepted (2026-11-12)

In memory, a missing value is a Polars null. doppel never mixes it with `NaN`, `None`, or `pd.NA`. `encode_feature` (`src/doppel/schema/nullable.py`) turns a float `NaN` into null, fills a numeric feature with the median, and fills a categorical or text feature with `__doppel_null__` (`NULL_SENTINEL` in `src/doppel/schema/reserved.py`). A separate per-column model learns the null pattern, so the fill does not hide it. A source value that equals the sentinel is refused, because it would read back as NULL.

File sources read only the empty string as NULL (`null_values` defaults to `("",)` in `src/doppel/locations.py`). `--null-values NA,?` adds tokens. Reading `NA`, `N/A`, or `null` as NULL by default turned real values, such as the country code `NA` for Namibia, into NULL without a word. A user whose file uses `NA` for missing now passes the flag, which changes their output.

## Considered options

- Treat a list of common tokens as NULL. Rejected: it silently changes real data, and no list fits every file.
- Use `NaN` for missing numerics. Rejected: Polars keeps `NaN` and null apart, and a mix breaks the null model.

## Consequences

- A column named in `null_values` loses those strings as values.
- The sentinel is reserved. Rename or recode a source value that equals it.
