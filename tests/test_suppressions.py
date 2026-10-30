"""No file-wide type or lint disables, and every inline type ignore names its rule and a reason.

No linter rule covers either gap: ruff PGH003 and basedpyright
reportIgnoreCommentWithoutRule check the rule name, not the reason or the file-wide forms.
"""

from __future__ import annotations

import io
import re
import subprocess
import tokenize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED_ROOTS = ("src", "tests")

PYRIGHT_DIRECTIVE = re.compile(r"^#\s*pyright:\s*(?P<body>.*)$")
PYRIGHT_IGNORE_WITH_REASON = re.compile(r"^ignore\[[A-Za-z]+(?:,\s*[A-Za-z]+)*\]\s+#\s*\S.*$")
TYPE_IGNORE = re.compile(r"^#\s*type:\s*ignore")
NOQA_WITHOUT_CODE = re.compile(r"#\s*noqa\b(?!\s*:\s*[A-Z])", re.IGNORECASE)
RUFF_FILE_WIDE = re.compile(r"#\s*ruff:\s*noqa", re.IGNORECASE)


def suppression_problems(source: str) -> list[tuple[int, str]]:
    """Return ``(line, problem)`` for each comment that silences a checker too widely."""
    problems: list[tuple[int, str]] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        line, comment = token.start[0], token.string
        pyright = PYRIGHT_DIRECTIVE.match(comment)
        if pyright and not pyright.group("body").startswith("ignore"):
            problems.append((line, "file-wide pyright directive"))
        elif pyright and not PYRIGHT_IGNORE_WITH_REASON.match(pyright.group("body")):
            problems.append((line, "pyright ignore without a rule name and a reason"))
        if TYPE_IGNORE.match(comment):
            problems.append((line, "type: ignore is not read; use `# pyright: ignore[rule]`"))
        if RUFF_FILE_WIDE.search(comment):
            problems.append((line, "file-wide ruff noqa"))
        elif NOQA_WITHOUT_CODE.search(comment):
            problems.append((line, "noqa without a rule code"))
    return problems


def _tracked_python_files() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", *SCANNED_ROOTS],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    return [REPO_ROOT / name for name in listed if name.endswith(".py")]


def test_no_file_wide_disables_and_every_ignore_has_a_rule_and_a_reason() -> None:
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line}: {problem}"
        for path in _tracked_python_files()
        for line, problem in suppression_problems(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []


def test_detector_flags_each_wide_or_unexplained_suppression() -> None:
    flagged = suppression_problems(
        "# pyright: basic\n"
        "x = 1  # pyright: ignore\n"
        "y = 2  # pyright: ignore[reportAny]\n"
        "z = 3  # type: ignore[assignment]\n"
        "a = 4  # noqa\n"
        "# ruff: noqa\n"
    )

    assert [line for line, _ in flagged] == [1, 2, 3, 4, 5, 6]


def test_detector_accepts_a_rule_named_ignore_with_a_reason() -> None:
    assert suppression_problems("x = 1  # pyright: ignore[reportAny]  # untrusted input\n") == []
