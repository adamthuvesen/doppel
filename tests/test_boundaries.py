"""Checks that ruff's import bans cannot make.

connectorx is loaded by string, so ruff cannot see it. Ruff's per-file ignores also lift every
ban in an owner file at once (a TID251 ignore cannot name one banned API), so the process and
network bans are checked here over every file. `scripts/check_boundaries.sh` plants a violation
for each test and requires it to fail, so an emptied detector fails `just check`.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "doppel"
# The SQL source loads the driver. `doctor` only probes that the module is present.
CONNECTORX_OWNERS = frozenset({"sources/sql.py", "cli/doctor.py"})
# `--password-cmd` runs the user's secret command.
PROCESS_OWNERS = frozenset({"sources/spec.py"})

PROCESS_MODULES = frozenset({"subprocess", "pty", "multiprocessing", "asyncio.subprocess"})
PROCESS_ATTRIBUTES = {
    "os": frozenset(
        {
            "system",
            "popen",
            "fork",
            "forkpty",
            "posix_spawn",
            "posix_spawnp",
            *(f"exec{suffix}" for suffix in ("l", "le", "lp", "lpe", "v", "ve", "vp", "vpe")),
            *(f"spawn{suffix}" for suffix in ("l", "le", "lp", "lpe", "v", "ve", "vp", "vpe")),
        }
    ),
    "asyncio": frozenset({"subprocess", "create_subprocess_exec", "create_subprocess_shell"}),
}
NETWORK_MODULES = frozenset(
    {"socket", "http.client", "urllib.request", "requests", "httpx", "urllib3", "aiohttp"}
)


def _files_mentioning(token: str) -> set[str]:
    return {
        path.relative_to(SRC).as_posix()
        for path in SRC.rglob("*.py")
        if token in path.read_text(encoding="utf-8")
    }


def _uses(source: str, modules: frozenset[str], attributes: dict[str, frozenset[str]]) -> list[str]:
    """Each banned name `source` imports, reaches as `module.attr`, or names in a string
    (as `importlib.import_module` would take it)."""

    def banned(name: str) -> bool:
        head, _, attr = name.rpartition(".")
        return any(name == m or name.startswith(f"{m}.") for m in modules) or (
            attr in attributes.get(head, frozenset())
        )

    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            names = [f"{node.value.id}.{node.attr}"]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names = [node.value]
        found += [name for name in names if banned(name)]
    return found


def _users(modules: frozenset[str], attributes: dict[str, frozenset[str]]) -> dict[str, list[str]]:
    users: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        found = _uses(path.read_text(encoding="utf-8"), modules, attributes)
        if found:
            users[path.relative_to(SRC).as_posix()] = found
    return users


def test_connectorx_is_named_only_in_its_owner_files() -> None:
    assert _files_mentioning("connectorx") <= CONNECTORX_OWNERS


def test_connectorx_detector_still_finds_the_real_use() -> None:
    assert "sources/sql.py" in _files_mentioning("connectorx")


def test_only_the_password_command_owner_starts_a_process() -> None:
    users = _users(PROCESS_MODULES, PROCESS_ATTRIBUTES)
    assert {f: u for f, u in users.items() if f not in PROCESS_OWNERS} == {}


def test_process_detector_still_finds_the_real_use() -> None:
    assert "subprocess" in _users(PROCESS_MODULES, PROCESS_ATTRIBUTES).get("sources/spec.py", [])


def test_no_module_opens_a_network_client() -> None:
    assert _users(NETWORK_MODULES, {}) == {}


def test_network_detector_finds_each_import_form() -> None:
    source = "import socket\nfrom http import client\nfrom urllib.request import urlopen\n"
    assert _uses(source, NETWORK_MODULES, {}) == [
        "socket",
        "http.client",
        "urllib.request",
        "urllib.request.urlopen",
    ]
