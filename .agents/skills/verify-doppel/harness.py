"""Isolation, evidence, and failure tokens shared by the drives."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SKILL_DIR = Path(__file__).resolve().parent
SECRET_NAME = re.compile("KEY|TOKEN|SECRET|PASSWORD")
TRIPWIRES = ("curl", "wget", "pip", "pip3", "uv", "ssh")
TRIPWIRE_EXIT = 97
# An unroutable proxy makes an HTTP client fail fast instead of reaching the network.
DEAD_PROXY = "http://127.0.0.1:9"
CHILD_TIMEOUT_SECONDS = 90.0
# Environment variable that makes the guard hide one installed distribution from importlib.metadata.
HIDE_DISTRIBUTION_ENV = "VERIFY_HIDE_DISTRIBUTION"

# Failure tokens: stable identifiers a control or a reader matches on, never message wording.
TIMEOUT = "timeout"
TRIPWIRE_HIT = "tripwire-hit"
CHECKOUT_CHANGED = "checkout-changed"
SKIPPED = "skipped"
EARLY_EXIT = "early-exit"
NO_ASSERTIONS = "no-assertions"
CRASH = "crash"
EXIT_CODE = "unexpected-exit-code"

# Runs inside every child as sitecustomize. It records, then blocks, any network connect and any
# process spawn from Python, and can hide one installed distribution (the PII model drive).
GUARD_SOURCE = """\
import importlib.metadata
import os
import socket
import subprocess

_LOG = os.environ["VERIFY_GUARD_LOG"]


def _note(message):
    with open(_LOG, "a") as handle:
        handle.write(message + "\\n")


_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex


def _blocked_connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _note("socket.connect " + repr(address))
        raise OSError("blocked by the verify guard")
    return _connect(self, address)


def _blocked_connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _note("socket.connect_ex " + repr(address))
        return 111
    return _connect_ex(self, address)


socket.socket.connect = _blocked_connect
socket.socket.connect_ex = _blocked_connect_ex


def _blocked_popen(self, args, *a, **k):
    _note("subprocess.Popen " + repr(args))
    raise OSError("blocked by the verify guard")


subprocess.Popen.__init__ = _blocked_popen

_hidden = os.environ.get("VERIFY_HIDE_DISTRIBUTION")
if _hidden:
    _distribution = importlib.metadata.distribution

    def _distribution_without_hidden(name):
        if name.replace("_", "-").lower() == _hidden.replace("_", "-").lower():
            raise importlib.metadata.PackageNotFoundError(name)
        return _distribution(name)

    importlib.metadata.distribution = _distribution_without_hidden
"""


class DriveFailure(Exception):
    def __init__(self, token: str, detail: str) -> None:
        super().__init__(f"{token}: {detail}")
        self.token = token
        self.detail = detail


def git(*args: str, cwd: Path = ROOT) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    done = subprocess.run(
        ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "absent"


def checkout_state() -> dict[str, str]:
    """HEAD, the index, git config and hooks, and every changed, untracked, or ignored path.

    Ignored paths compare by size and mtime: .venv holds too many files to hash.
    """
    git_dir = (ROOT / git("rev-parse", "--git-common-dir")).resolve()
    state = {
        "HEAD": git("rev-parse", "HEAD"),
        "index": git("ls-files", "--stage"),
        ".git/config": _digest(git_dir / "config"),
    }
    for hook in sorted((git_dir / "hooks").glob("*")):
        state[f".git/hooks/{hook.name}"] = _digest(hook)
    listing = git("status", "--porcelain", "-z", "--untracked-files=all", "--ignored")
    for entry in filter(None, listing.split("\0")):
        code, path = entry[:2], entry[3:]
        if path.startswith(".verify/") or "__pycache__" in path.split("/"):
            continue
        target = ROOT / path
        if code == "!!":
            if target.exists():
                info = target.stat()
                state[path] = f"ignored:{info.st_size}:{info.st_mtime_ns}"
        else:
            state[path] = f"{code}:{_digest(target)}"
    return state


def changed_paths(before: Mapping[str, str]) -> list[str]:
    after = checkout_state()
    return sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key))


def squash(text: str) -> str:
    """Collapse the box drawing and line wraps of rich error panels so a token can match."""
    return re.sub(r"[\s│╭╮╰╯─]+", " ", text).strip()


class Session:
    """One scratch dir with a copy of the code, a temp HOME, the guard, and the evidence of a run."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.scratch = Path(tempfile.mkdtemp(prefix="verify-doppel-")).resolve()
        self.copy = self.scratch / "checkout"
        self.home = self.scratch / "home"
        self.tripwire_bin = self.scratch / "tripwires"
        self.guard_dir = self.scratch / "guard"
        self.tripwire_log = self.scratch / "tripwire-hits.log"
        self.guard_log = self.scratch / "guard-hits.log"
        self._lock = threading.Lock()
        self._steps = 0
        self.home.mkdir()
        self._install_tripwires()
        self._copy_code()

    def _install_tripwires(self) -> None:
        self.tripwire_bin.mkdir()
        for name in TRIPWIRES:
            shim = self.tripwire_bin / name
            shim.write_text(
                f'#!/bin/sh\necho "{name} $*" >> "{self.tripwire_log}"\nexit {TRIPWIRE_EXIT}\n'
            )
            shim.chmod(0o755)
        self.guard_dir.mkdir()
        (self.guard_dir / "sitecustomize.py").write_text(GUARD_SOURCE)

    def _copy_code(self) -> None:
        """Copy the tracked and untracked source so the children import the copy, not the checkout."""
        for rel in git(
            "ls-files", "-co", "--exclude-standard", "src", "pyproject.toml"
        ).splitlines():
            source = ROOT / rel
            if source.exists():
                target = self.copy / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target, follow_symlinks=False)

    def cleanup(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)

    def env(
        self, extra: Mapping[str, str] | None = None, guard_log: Path | None = None
    ) -> dict[str, str]:
        env = {
            "PATH": os.pathsep.join([str(self.tripwire_bin), "/usr/bin", "/bin"]),
            "HOME": str(self.home),
            "PYTHONPATH": os.pathsep.join([str(self.guard_dir), str(self.copy / "src")]),
            "PYTHONDONTWRITEBYTECODE": "1",
            "VERIFY_GUARD_LOG": str(guard_log or self.guard_log),
            "TZ": "UTC",
            "NO_COLOR": "1",
            "COLUMNS": "120",
            "HTTP_PROXY": DEAD_PROXY,
            "HTTPS_PROXY": DEAD_PROXY,
            "ALL_PROXY": DEAD_PROXY,
            "NO_PROXY": "",
            "XDG_CONFIG_HOME": str(self.home / "config"),
            "XDG_CACHE_HOME": str(self.home / "cache"),
            "XDG_DATA_HOME": str(self.home / "data"),
        }
        env.update(extra or {})
        return env

    def tripwire_hits(self, guard_log: Path | None = None) -> list[str]:
        hits: list[str] = []
        for log in (self.tripwire_log, guard_log or self.guard_log):
            if log.exists():
                hits += log.read_text().splitlines()
        return hits

    def clear_tripwire_hits(self) -> None:
        for log in (self.tripwire_log, self.guard_log):
            log.unlink(missing_ok=True)

    def run(
        self,
        drive: str,
        step: str,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        extra_env: Mapping[str, str] | None = None,
        expect_exit: int | None = 0,
        timeout: float = CHILD_TIMEOUT_SECONDS,
        guard_log: Path | None = None,
        allowed_hits: Sequence[str] = (),
    ) -> subprocess.CompletedProcess[str]:
        """Run one child in the isolated environment and record it.

        ``expect_exit=None`` skips the code check. ``allowed_hits`` lists guard lines a drive
        has declared as known, so any other hit still fails.
        """
        env = self.env(extra_env, guard_log)
        started = time.monotonic()
        try:
            done = subprocess.run(
                list(argv),
                cwd=cwd or self.scratch,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                start_new_session=True,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self._record(drive, step, argv, env, None, "", f"timed out after {timeout}s", started)
            raise DriveFailure(TIMEOUT, f"{drive}.{step} exceeded {timeout}s") from exc
        self._record(drive, step, argv, env, done.returncode, done.stdout, done.stderr, started)
        hits = [h for h in self.tripwire_hits(guard_log) if h not in allowed_hits]
        if hits:
            raise DriveFailure(TRIPWIRE_HIT, f"called {hits} (seen after {drive}.{step})")
        if expect_exit is not None and done.returncode != expect_exit:
            tail = squash(done.stderr or done.stdout)[-600:]
            raise DriveFailure(
                EXIT_CODE,
                f"{drive}.{step} exited {done.returncode}, expected {expect_exit}: {tail}",
            )
        return done

    def _record(
        self,
        drive: str,
        step: str,
        argv: Sequence[str],
        env: Mapping[str, str],
        code: int | None,
        out: str,
        err: str,
        started: float,
    ) -> None:
        secrets = [v for k, v in env.items() if SECRET_NAME.search(k) and len(v) >= 4]

        def redact(text: str) -> str:
            for secret in secrets:
                text = text.replace(secret, "<redacted>")
            return text

        with self._lock:
            self._steps += 1
            number = self._steps
        entry = {
            "argv": [redact(a) for a in argv],
            "exit": code,
            "seconds": round(time.monotonic() - started, 2),
            "stdout": redact(out[-4000:]),
            "stderr": redact(err[-4000:]),
        }
        (self.run_dir / f"{drive}.{number:02d}-{step}.json").write_text(
            json.dumps(entry, indent=2) + "\n"
        )


class Ctx:
    """What one drive sees: the session, its own work dir, and its assertion count."""

    def __init__(self, session: Session, drive: str) -> None:
        self.session = session
        self.drive = drive
        self.work = session.scratch / "work" / drive
        self.work.mkdir(parents=True)
        self.guard_log = session.scratch / f"guard-hits-{drive}.log"
        self.allowed_hits: tuple[str, ...] = ()
        self.assertions = 0

    def check(self, condition: bool, token: str, detail: str) -> None:
        self.assertions += 1
        if not condition:
            raise DriveFailure(token, detail)

    def doppel(
        self,
        step: str,
        *args: str,
        expect_exit: int | None = 0,
        extra_env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run the real CLI (`python -m doppel`) from the copy, in this drive's work dir."""
        return self.session.run(
            self.drive,
            step,
            [sys.executable, "-m", "doppel", *args],
            cwd=self.work,
            expect_exit=expect_exit,
            extra_env=extra_env,
            guard_log=self.guard_log,
            allowed_hits=self.allowed_hits,
        )

    def guard_hits(self) -> list[str]:
        """Every guard line this drive's children logged, including the allowed ones."""
        return self.session.tripwire_hits(self.guard_log)

    def parallel[T](self, jobs: Sequence[Callable[[], T]]) -> list[T]:
        """Run independent steps at once. The first failure is raised after all have finished."""
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = [pool.submit(job) for job in jobs]
            return [future.result() for future in futures]


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
