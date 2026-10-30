"""connectorx is loaded by string, so ruff's import bans cannot see it.

This grep covers the gap. `scripts/check_boundaries.sh` plants a violation and requires the
first test to fail, so an emptied detector fails `just check`.
"""

from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "doppel"
# The SQL source loads the driver. `doctor` only probes that the module is present.
CONNECTORX_OWNERS = frozenset({"sources/sql.py", "cli/doctor.py"})


def _files_mentioning(token: str) -> set[str]:
    return {
        path.relative_to(SRC).as_posix()
        for path in SRC.rglob("*.py")
        if token in path.read_text(encoding="utf-8")
    }


def test_connectorx_is_named_only_in_its_owner_files() -> None:
    assert _files_mentioning("connectorx") <= CONNECTORX_OWNERS


def test_connectorx_detector_still_finds_the_real_use() -> None:
    assert "sources/sql.py" in _files_mentioning("connectorx")
