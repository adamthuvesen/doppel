# Determinism: all randomness comes from `Rng`, except the hash salt

Status: accepted (2026-11-12)

The same source and `--seed` give byte-identical output. Every random draw goes through `Rng` in `src/doppel/synth/seed.py`, built with `Rng.from_seed` and forked with `Rng.spawn`. Ruff bans `random`, `uuid.uuid4`, `os.urandom`, and `numpy.random` outside that file (`pyproject.toml`). Each phase (fit, sample, PII restore, text policy) roots its own `Rng` at the same seed, so a change in one phase does not shift another.

Python set order depends on the per-process hash seed, so an in-process test cannot catch an order bug. `tests/test_gen_e2e.py::test_gen_is_byte_identical_across_processes_with_different_hash_seeds` runs `gen` in separate processes with different `PYTHONHASHSEED` values.

One draw is deliberately outside the contract. `--text-policy hash` uses a random salt for each run (`resolve_hash_salt` in `src/doppel/text_policy.py`, the only user of `secrets`). A salt that `--seed` fixes would be guessable. Output of a hash-policy run is then byte-identical only with `--hash-salt VALUE`.

## Considered options

- Derive the hash salt from `--seed`. Rejected: anyone who knows the seed or tries small seeds can rebuild the mapping with a dictionary.
- Allow `random` where the result "does not matter". Rejected: this is how a contract breaks without a test noticing.

## Consequences

- The contract is not guaranteed across major NumPy or scikit-learn versions.
- See [docs/determinism.md](../determinism.md) for the phases and the SQL sampling forms.
