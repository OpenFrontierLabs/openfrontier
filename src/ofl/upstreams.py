"""Pinned upstreams (configs/upstreams.toml): loading, fetching at the exact SHA, verification."""

from __future__ import annotations

import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ofl.repo import OflError

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
CHECKOUT_DIR = Path(".cache") / "upstreams"


@dataclass(frozen=True)
class Upstream:
    name: str
    repo: str
    ref: str
    sha: str
    license: str
    verified: str
    used_by: str


def load(root: Path) -> dict[str, Upstream]:
    raw = tomllib.loads((root / "configs" / "upstreams.toml").read_text(encoding="utf-8"))
    out: dict[str, Upstream] = {}
    for name, entry in raw.items():
        out[name] = Upstream(
            name=name,
            repo=entry["repo"],
            ref=entry["ref"],
            sha=entry["sha"],
            license=entry["license"],
            verified=entry["verified"],
            used_by=entry["used-by"],
        )
    return out


def problems(ups: dict[str, Upstream]) -> list[str]:
    out: list[str] = []
    for up in ups.values():
        if not SHA_RE.match(up.sha):
            out.append(f"configs/upstreams.toml [{up.name}]: sha must be 40 lowercase hex digits")
        if not up.repo.startswith("https://"):
            out.append(f"configs/upstreams.toml [{up.name}]: repo must be an https URL")
    return out


def checkout_path(root: Path, name: str) -> Path:
    return root / CHECKOUT_DIR / name


def _git(args: list[str], cwd: Path) -> str:
    try:
        res = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise OflError("git not found on PATH") from exc
    if res.returncode != 0:
        raise OflError(f"git {' '.join(args)} failed in {cwd}: {res.stderr.strip()}")
    return res.stdout.strip()


def head_sha(path: Path) -> str | None:
    """HEAD of a checkout, or None if `path` is not a git checkout."""
    if not (path / ".git").exists():
        return None
    return _git(["rev-parse", "HEAD"], path)


def ensure_checkout(root: Path, up: Upstream) -> Path:
    """Return a checkout of `up` at exactly `up.sha`, cloning it under .cache/ if needed."""
    path = checkout_path(root, up.name)
    current = head_sha(path)
    if current == up.sha:
        return path
    if current is not None:
        raise OflError(
            f"{path} is at {current}, expected {up.sha}. Remove it (ofl clean) and re-run."
        )
    path.mkdir(parents=True, exist_ok=True)
    print(f"fetching {up.repo} @ {up.sha} into {path}", flush=True)
    _git(["init", "-q"], path)
    _git(["remote", "add", "origin", up.repo], path)
    _git(["fetch", "-q", "--depth", "1", "origin", up.sha], path)
    _git(["checkout", "-q", "FETCH_HEAD"], path)
    got = head_sha(path)
    if got != up.sha:
        raise OflError(f"fetched {got}, expected {up.sha}")
    return path
