"""Phase 1 dense launcher: builds (and optionally runs) the real OLMo-core command for one node.

The command is the pinned OLMo-core's own example script, src/examples/llm/train.py, started with
torch.distributed.run. Every option and override comes from a config in configs/phase1-dense/ and
was accepted by that script's `--dry-run` at the pinned commit; `ofl run0 verify` re-checks it.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ofl import upstreams
from ofl.configs import load_toml
from ofl.repo import OflError, find_root, run

DEFAULT_CONFIG = Path("configs") / "phase1-dense" / "run0.toml"
UPSTREAM = "olmo-core"


@dataclass(frozen=True)
class Launch:
    script_args: list[str]  # the upstream script and its arguments
    argv: list[str]  # the full torch.distributed.run command
    cwd: Path


def format_override(value: Any) -> str:
    """Render a TOML value the way the upstream's dotted-override parser reads it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def build(
    root: Path,
    config: Path,
    *,
    script_dir: Path,
    python: str,
    nproc: int,
    run_name: str | None,
    save_folder: Path,
    work_dir: Path,
    data_glob: str | None,
    dry_run: bool,
) -> Launch:
    """Assemble the launch command. Pure: touches no network and starts nothing."""
    raw = load_toml(config)
    launch = raw["launch"]
    source = raw["data"]["source"]
    overrides: dict[str, Any] = dict(raw["overrides"])

    if source == "glob":
        if not data_glob:
            raise OflError(
                f"{config.name} reads tokenised shards you provide: pass --data-glob "
                "'/path/to/*.npy'. No FineWeb-Edu shards exist yet; see docs/run0.md."
            )
        overrides["dataset.paths"] = [data_glob]
        overrides["dataset.expand_glob"] = True
    elif data_glob:
        raise OflError(f"{config.name} streams its own data mix; --data-glob does not apply")

    name = run_name or raw["meta"]["name"]
    script_args = [
        str(script_dir / launch["entrypoint"]),
        name,
        "--model-factory",
        launch["model_factory"],
        "--sequence-length",
        str(launch["sequence_length"]),
        "--save-folder",
        str(save_folder),
        "--work-dir",
        str(work_dir),
    ]
    if dry_run:
        script_args.append("--dry-run")
    script_args.extend(f"--{key}={format_override(val)}" for key, val in overrides.items())
    dist = [python, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={nproc}"]
    return Launch(script_args=script_args, argv=[*dist, *script_args], cwd=root)


def launch(args: Any) -> int:
    root = find_root()
    ups = upstreams.load(root)
    script_dir = upstreams.checkout_path(root, UPSTREAM)
    config = Path(args.config) if args.config else root / DEFAULT_CONFIG

    if not args.dry_run:
        script_dir = upstreams.ensure_checkout(root, ups[UPSTREAM])
    plan = build(
        root,
        config,
        script_dir=script_dir,
        python=args.python or sys.executable,
        nproc=args.nproc,
        run_name=args.run_name,
        save_folder=Path(args.save_folder) if args.save_folder else root / "checkpoints" / "run0",
        work_dir=Path(args.work_dir) if args.work_dir else root / ".cache" / "run0" / "work",
        data_glob=args.data_glob,
        dry_run=False,
    )
    if args.dry_run:
        print(f"# pinned upstream: {ups[UPSTREAM].repo} @ {ups[UPSTREAM].sha}")
        print(
            f"# not run. `ofl fetch {UPSTREAM}` creates {script_dir}, which holds the script below"
        )
        print(shlex.join(plan.argv))
        return 0
    return run(plan.argv, cwd=plan.cwd)


def verify(root: Path) -> int:
    """Run the pinned upstream script with `--dry-run` and our overrides; it must accept them."""
    ups = upstreams.load(root)
    script_dir = upstreams.ensure_checkout(root, ups[UPSTREAM])
    failures = 0
    for config in sorted((root / "configs" / "phase1-dense").glob("*.toml")):
        if config.name == "smoke-tiny.toml":
            continue
        plan = build(
            root,
            config,
            script_dir=script_dir,
            python=sys.executable,
            nproc=1,
            run_name=None,
            save_folder=root / ".cache" / "run0" / "verify-save",
            work_dir=root / ".cache" / "run0" / "verify-work",
            data_glob="/nonexistent/*.npy"
            if load_toml(config)["data"]["source"] == "glob"
            else None,
            dry_run=True,
        )
        # --dry-run only prints the merged config, so the distributed launcher is not needed.
        try:
            res = subprocess.run(
                [sys.executable, *plan.script_args],
                capture_output=True,
                text=True,
                check=False,
                timeout=600,
            )
        except subprocess.TimeoutExpired as exc:
            raise OflError(f"upstream --dry-run for {config.name} timed out after 600 s") from exc
        ok = res.returncode == 0
        print(
            f"{'ok  ' if ok else 'FAIL'} {config.name}: upstream accepted "
            f"{len(plan.script_args) - 2} arguments (--dry-run)"
        )
        if not ok:
            failures += 1
            print(res.stderr[-2000:])
    return 1 if failures else 0
