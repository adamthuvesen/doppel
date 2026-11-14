# Python 3.14 only, and doppel is not published

Status: accepted (2026-11-12)

`requires-python` is `>=3.14` (`pyproject.toml`), and `.python-version` pins 3.14. doppel is not on PyPI, so no user loses support when the floor moves, and one Python version keeps the dependency set, the type checker, and the determinism claim simple. The package name in `pyproject.toml` is `doppeldata`, for a later release. Install from git: `uv tool install "doppeldata[all] @ git+https://github.com/adamthuvesen/doppel"`, or `uv sync --all-extras` in a checkout.

The spaCy model `en_core_web_lg` is not on PyPI. `[tool.uv.sources]` in `pyproject.toml` locks its release wheel URL in `uv.lock`, and doppel never downloads it at run time.

## Considered options

- Publish to PyPI. Rejected for now: the model wheel cannot be a PyPI dependency, and there is no user to serve.
- Support older Python versions. Rejected: nothing published depends on them, and each extra version costs a wheel set to test.

## Consequences

- Publishing later needs a plan for the spaCy model, for example a separate download step.
- Docs and install hints use the git command.
