"""Repo-root discovery and subprocess helpers shared by the commands."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT_MARKER = Path("configs") / "upstreams.toml"


class OflError(Exception):
    """An expected failure: printed without a traceback, exits with `code`."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


def find_root(start: Path | None = None) -> Path:
    """Return the checkout root: the directory holding pyproject.toml and configs/upstreams.toml."""
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        if (directory / "pyproject.toml").is_file() and (directory / ROOT_MARKER).is_file():
            return directory
    raise OflError(
        f"not inside an openfrontier checkout (no pyproject.toml next to {ROOT_MARKER} above "
        f"{here}). Run ofl from a clone of https://github.com/OpenFrontierLabs/openfrontier."
    )


def run(
    argv: Sequence[str],
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """Run a command with inherited stdio and return its exit code."""
    print("+ " + " ".join(argv), flush=True)
    try:
        return subprocess.run(
            list(argv), cwd=cwd, env=None if env is None else dict(env)
        ).returncode
    except FileNotFoundError as exc:
        raise OflError(f"command not found: {argv[0]}") from exc


def python_module(module: str, *args: str) -> list[str]:
    """argv that runs `python -m module args` with this interpreter."""
    return [sys.executable, "-m", module, *args]
