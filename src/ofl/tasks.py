"""Developer lifecycle: setup, check, test, fix, build, clean."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from ofl import configs, ledger
from ofl.repo import OflError, find_root, python_module, run

CLEAN_DIRS = (
    "dist",
    "build",
    ".cache",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
)
# Directories never entered when sweeping for __pycache__ and *.egg-info.
SKIP_DIRS = {".venv", ".git"}


def _uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise OflError("uv not found on PATH; install it from https://docs.astral.sh/uv/")
    return uv


def setup() -> int:
    """Create/refresh the project venv from uv.lock. Installs nothing outside the repo."""
    return run([_uv(), "sync", "--locked"], cwd=find_root())


def build() -> int:
    """Build the sdist and wheel into dist/."""
    return run([_uv(), "build"], cwd=find_root())


def test() -> int:
    return run(python_module("pytest"), cwd=find_root())


def fix() -> int:
    root = find_root()
    rc = run(python_module("ruff", "check", "--fix", "."), cwd=root)
    rc |= run(python_module("ruff", "format", "."), cwd=root)
    return rc


def check() -> int:
    """Lint, format check, types, ledger schema + example, config parsing. Runs every step."""
    root = find_root()
    failed: list[str] = []

    steps = {
        "ruff check": python_module("ruff", "check", "."),
        "ruff format --check": python_module("ruff", "format", "--check", "."),
        "mypy": python_module("mypy"),
    }
    for name, argv in steps.items():
        if run(argv, cwd=root) != 0:
            failed.append(name)

    for name, problems in (
        ("ledger schema and example", ledger.check(root)),
        ("configs, manifests and upstream pins", configs.check(root)),
    ):
        if problems:
            failed.append(name)
            print(f"FAIL {name}")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print(f"ok   {name}")

    if failed:
        print(f"check failed: {', '.join(failed)}")
        return 1
    print("check passed")
    return 0


def clean() -> int:
    """Remove caches, build output and scratch inside the repo. Leaves .venv and .git alone."""
    root = find_root()
    removed: list[Path] = []
    for name in CLEAN_DIRS:
        path = root / name
        if path.exists():
            shutil.rmtree(path)
            removed.append(path)
    for pattern in ("__pycache__", "*.egg-info"):
        for path in sorted(root.rglob(pattern)):
            if SKIP_DIRS.intersection(path.relative_to(root).parts) or not path.exists():
                continue
            shutil.rmtree(path)
            removed.append(path)
    for path in removed:
        print(f"removed {path.relative_to(root)}")
    print(f"clean: removed {len(removed)} path(s)")
    return 0


def _smoke_module(module: str, *args: str, stream: bool) -> tuple[int, float]:
    """Run one smoke module in its own process. Output goes to a log unless streaming."""
    root = find_root()
    argv = python_module(module, *args)
    start = time.perf_counter()
    if stream:
        rc = run(argv, cwd=root)
        return rc, time.perf_counter() - start
    log = root / ".cache" / "smoke" / f"{module.rsplit('.', 1)[-1]}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        rc = subprocess.run(
            argv, cwd=root, stdout=fh, stderr=subprocess.STDOUT, check=False
        ).returncode
    elapsed = time.perf_counter() - start
    lines = log.read_text(encoding="utf-8").splitlines()
    result = next((ln for ln in reversed(lines) if ln.startswith("OFL_SMOKE_RESULT")), "")
    print(result or "(no result line)")
    if rc != 0:
        print(f"--- last lines of {log.relative_to(root)} ---")
        print("\n".join(lines[-25:]))
    return rc, elapsed


def smoke() -> int:
    """Both CPU smokes: real OLMo-core run (phase 1), then reference DiLoCo (phase 3 algorithm)."""
    steps = (
        ("phase 1: olmo-core trainer, tiny model, CPU", "ofl.smoke.olmo_tiny"),
        ("phase 3 algorithm: reference DiLoCo (not prime-diloco), CPU", "ofl.smoke.diloco_ref"),
    )
    failed: list[str] = []
    for title, module in steps:
        print(f"== {title}", flush=True)
        rc, seconds = _smoke_module(module, stream=False)
        print(f"{'ok  ' if rc == 0 else 'FAIL'} {module} in {seconds:.1f} s", flush=True)
        if rc != 0:
            failed.append(module)
    if failed:
        print(f"smoke failed: {', '.join(failed)}")
        return 1
    print("smoke passed")
    return 0


def dev() -> int:
    """A longer, streaming version of the phase 1 smoke: same real OLMo-core trainer on CPU."""
    root = find_root()
    steps = configs.load_smoke_tiny(root).dev_steps
    workdir = root / ".cache" / "dev"
    shutil.rmtree(
        workdir, ignore_errors=True
    )  # a fresh run: the trainer resumes from old checkpoints
    workdir.mkdir(parents=True)
    rc, _ = _smoke_module(
        "ofl.smoke.olmo_tiny", "--steps", str(steps), "--workdir", str(workdir), stream=True
    )
    return rc
