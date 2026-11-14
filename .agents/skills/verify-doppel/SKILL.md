---
name: verify-doppel
description: "Drives the real doppel CLI (gen, fit, sample, diff, artifact info, multi-table, DuckDB, PII) on committed seeded fixtures in a temp dir. Use to run, launch, drive, or verify doppel, or to prove a change to it works."
---

# Verify doppel

`verify` copies `src/` to a temp dir and runs the real `python -m doppel` against committed, seeded, synthetic fixtures. It never reads real data, never uses the network, and never starts a process other than the CLI under test.

## Launch / Reset

There is no long-lived process. Each run builds its own scratch dir, temp HOME, and tripwires, so a run starts at baseline and `reset` is a new run. Run from the repo root:

- `just verify all` runs every drive (about 10 s).
- `just verify` runs the drives whose paths match files changed against `main`, else the smoke drive (`gen`).
- `just verify gen diff` runs named drives.
- `just verify list` shows each drive, what it proves, and the paths it covers.
- `just verify control` plants a sink that names its own source table and passes only on its refusal.
- `just verify doctor` checks the isolation.

`just verify` needs the locked environment with every extra (`just setup`): the PII drives need the spaCy model.

## Doctor

Every run repeats these checks before its first drive: the modules import, the committed fixtures equal what `fixtures.py` writes, `doppel` imports from the temp copy and not from the checkout, `curl`, `wget`, `pip`, `pip3`, `uv`, and `ssh` resolve to tripwires on `PATH`, and the in-process guard blocks a socket and a process spawn and logs both.

## Drive

Each drive runs `sys.executable -m doppel ...` children in its own work dir with a temp HOME, a dead proxy, the tripwires first on `PATH`, and the guard (`sitecustomize`) in every child. The guard logs and blocks any network connect and any process spawn from Python. See `features/README.md` for the drive table.

A drive FAILs when any of these hold:

- a tripwire or guard line is hit that the drive did not declare, a child times out, or a child exits with an unexpected code;
- the drive finishes without asserting anything;
- the run changes the checkout (tracked, untracked, or ignored paths, HEAD, the index, or `.git/config` and hooks).

Drives run in parallel threads. The helper does not retry. A failed drive is FAIL, never `flaky`.

## Evidence

Each run writes `.verify/runs/<utc>-<sha7>[-dirty]/`: `manifest.txt` (one block per drive), `summary.json`, and one `<drive>.<n>-<step>.json` per child command (argv, exit, stdout, stderr; values of `KEY`, `TOKEN`, `SECRET`, `PASSWORD` variables redacted). The run prints `PASS|FAIL <drive> <seconds> <detail>` per drive and ends with the evidence path. `.verify/` is gitignored.

- **Control.** `just verify control` builds a DuckDB source, then re-drives the sink step with the wrong expectation: a write whose output table is the source table (`Main.ACCOUNTS` for source table `accounts`) must succeed. It passes only when the step fails with exit code 2 and the refusal `is the source file` (`REFUSAL_SINK_IS_SOURCE`). Any other failure, a tripwire hit, or a success fails the control.

## Cleanup

The scratch dir (code copy, HOME, work dirs, tripwires) is removed on every exit path. `.verify/runs/` stays. Nothing is written to the checkout.

## Helper

`verify` is the dispatcher. `harness.py` holds the isolation and the guard, `drives.py` the drives and their tokens, and `fixtures.py` the seeded generator.

- `verify` | `verify all` | `verify list` | `verify <drive>...` | `verify control` | `verify doctor`
- `verify exec -- COMMAND [ARGS...]` runs a command inside the isolation for diagnosis.
- Fixtures live in `fixtures/` and are committed. `python .agents/skills/verify-doppel/fixtures.py` rewrites them from `SEED = 20261112`. After changing the generator, rewrite and commit them: the doctor fails on drift.
- Python in this skill is its own type-check project (`pyrightconfig.json`) and its own ruff config (`ruff.toml`). `just check` runs both.
- Failure tokens are constants in `drives.py`. Refusal strings the product prints are held there too (`REFUSAL_*`) because the code has no constant for them. The install hint and the model name are imported from `doppel.pii.detect`, and the diff exit code from `doppel.cli.diff`.

## Limits

- The PII model is hidden by making `importlib.metadata.distribution` raise for `en_core_web_lg` inside the child. That is how `spacy.util.is_package` looks it up, so the product code path is real, but a model that is installed and broken is not covered.
- Detection of emails needs a real TLD: `.example` addresses are not detected as emails. The customer fixture uses `example.com`.
- Snowflake, Postgres, and `--password-cmd` need a warehouse or a subprocess and are not driven.
