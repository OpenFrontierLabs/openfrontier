"""Phase 1 MoE launcher: the Hugging Face MoE handbook's Megatron-LM flags for one 8-GPU node.

Builds `torch.distributed.run ... pretrain_gpt.py <flags>` against the pinned Megatron-LM commit.
`ofl moe verify` checks every flag name in the config (and the ones added at launch) against that
commit's source.
"""

from __future__ import annotations

import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ofl import upstreams
from ofl.configs import load_toml
from ofl.repo import OflError, find_root, run

DEFAULT_CONFIG = Path("configs") / "phase1-moe" / "handbook-moe-6p9b.toml"
UPSTREAM = "megatron-lm"
# Flags `ofl moe launch` adds itself; verified together with the config's flags.
LAUNCH_FLAGS = (
    "--tokenizer-model",
    "--data-args-path",
    "--data-cache-path",
    "--save",
    "--tensorboard-dir",
)


@dataclass(frozen=True)
class MoeLaunch:
    argv: list[str]
    env: dict[str, str]
    cwd: Path


def config_flags(raw: dict[str, Any], setup_check: bool) -> list[str]:
    """Flatten [flags.*] (and [setup_check]) into an argv fragment, in file order."""
    tables = list(raw["flags"].values())
    if setup_check:
        tables.append(raw["setup_check"])
    out: list[str] = []
    for table in tables:
        for flag, value in table.items():
            if value is True:
                out.append(flag)
            elif value is False:
                continue
            else:
                out.extend([flag, str(value)])
    return out


def build(
    config: Path,
    *,
    megatron_dir: Path,
    python: str,
    tokenizer_dir: Path,
    data_args_path: Path,
    data_cache_path: Path,
    save_dir: Path,
    setup_check: bool,
) -> MoeLaunch:
    raw = load_toml(config)
    gpus = raw["launch"]["gpus_per_node"]
    flags = [
        *config_flags(raw, setup_check),
        "--tokenizer-model",
        str(tokenizer_dir),
        "--data-args-path",
        str(data_args_path),
        "--data-cache-path",
        str(data_cache_path),
        "--save",
        str(save_dir),
        "--tensorboard-dir",
        str(save_dir / "tensorboard"),
    ]
    argv = [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={gpus}",
        str(megatron_dir / raw["launch"]["entrypoint"]),
        *flags,
    ]
    env = {k: str(v) for k, v in raw.get("env", {}).items()}
    env["PYTHONPATH"] = str(megatron_dir)
    return MoeLaunch(argv=argv, env=env, cwd=megatron_dir)


def launch(args: Any) -> int:
    root = find_root()
    ups = upstreams.load(root)
    megatron_dir = upstreams.checkout_path(root, UPSTREAM)
    if not args.dry_run:
        megatron_dir = upstreams.ensure_checkout(root, ups[UPSTREAM])
        for label, path in (
            ("--tokenizer-dir", args.tokenizer_dir),
            ("--data-args-path", args.data_args_path),
        ):
            if not Path(path).exists():
                raise OflError(f"{label} {path} does not exist")
    plan = build(
        Path(args.config) if args.config else root / DEFAULT_CONFIG,
        megatron_dir=megatron_dir,
        python=args.python or sys.executable,
        tokenizer_dir=Path(args.tokenizer_dir),
        data_args_path=Path(args.data_args_path),
        data_cache_path=Path(args.data_cache_path or root / ".cache" / "moe" / "data-cache"),
        save_dir=Path(args.save or root / "checkpoints" / "moe"),
        setup_check=args.setup_check,
    )
    if args.dry_run:
        print(f"# pinned upstream: {ups[UPSTREAM].repo} @ {ups[UPSTREAM].sha}")
        print(f"# not run. `ofl fetch {UPSTREAM}` creates {megatron_dir}; the script is in it")
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in plan.env.items())
        print(f"cd {shlex.quote(str(plan.cwd))} && env {env} {shlex.join(plan.argv)}")
        return 0
    import os

    return run(plan.argv, cwd=plan.cwd, env={**os.environ, **plan.env})


def _flag_known(flag: str, texts: list[str]) -> bool:
    """A flag exists if Megatron defines it as an argparse literal, a metadata name or a field."""
    quoted = (f"'{flag}'", f'"{flag}"')
    field = flag[2:].replace("-", "_")
    candidates = {field, field.removeprefix("no_")}
    field_re = re.compile(rf"^\s+(?:{'|'.join(map(re.escape, candidates))})\s*:", re.M)
    return any(q in t for t in texts for q in quoted) or any(field_re.search(t) for t in texts)


def unknown_flags(flags: list[str], megatron_dir: Path) -> list[str]:
    """Flags (tokens starting with --) that Megatron's source under megatron_dir does not define."""
    texts = [
        p.read_text(encoding="utf-8", errors="ignore")
        for p in (megatron_dir / "megatron").rglob("*.py")
    ]
    wanted = sorted({f for f in flags if f.startswith("--")})
    return [f for f in wanted if not _flag_known(f, texts)]


def verify(root: Path) -> int:
    ups = upstreams.load(root)
    megatron_dir = upstreams.ensure_checkout(root, ups[UPSTREAM])
    raw = load_toml(root / DEFAULT_CONFIG)
    flags = config_flags(raw, setup_check=True) + list(LAUNCH_FLAGS)
    missing = unknown_flags(flags, megatron_dir)
    total = len({f for f in flags if f.startswith("--")})
    if missing:
        found = total - len(missing)
        print(f"FAIL {found}/{total} flags found in Megatron-LM {ups[UPSTREAM].sha[:9]}")
        for flag in missing:
            print(f"  unknown flag: {flag}")
        return 1
    print(f"ok   all {total} flags found in Megatron-LM {ups[UPSTREAM].sha[:9]} (name check only)")
    return 0
