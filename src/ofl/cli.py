"""The `ofl` command line. One entry point; the Makefile forwards every target here."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from ofl import deploy as deploy_mod
from ofl import doctor as doctor_mod
from ofl import moe, phases, run0, tasks, upstreams
from ofl.repo import OflError, find_root

TARGETS: dict[str, str] = {
    "setup": "uv sync from uv.lock; installs nothing outside the repo",
    "dev": "longer, streaming CPU run of the real OLMo-core trainer (tiny model)",
    "check": "ruff, ruff format --check, mypy, ledger schema + example, configs and pins",
    "test": "pytest",
    "fix": "ruff --fix and ruff format",
    "build": "uv build: sdist and wheel into dist/",
    "deploy": "ENV=dry-run lists what would be published; real publishing is not implemented",
    "smoke": "CPU, no downloads: real OLMo-core trainer run, then reference DiLoCo",
    "doctor": "GPU, CUDA, network, disk, clock and toolchain report",
    "clean": "remove caches, build output and scratch inside the repo",
    "help": "list targets",
}


def _simple(fn: Callable[[], int]) -> Callable[[argparse.Namespace], int]:
    return lambda _args: fn()


def _help(_args: argparse.Namespace) -> int:
    print("targets (make <target> or ofl <target>):")
    for name, text in TARGETS.items():
        print(f"  {name:<8} {text}")
    print("\nother commands:")
    print("  fetch NAME          clone a pinned upstream into .cache/upstreams/NAME")
    print("  upstreams           list pinned upstreams (configs/upstreams.toml)")
    print("  run0 launch|verify  phase 1 dense: OLMo-core command for one node / check it upstream")
    print("  moe launch|verify   phase 1 MoE: Megatron-LM command for one node / check its flags")
    print("  post launch         phase 2: not built")
    print("  diloco launch       phase 3: not built")
    return 0


def _deploy(args: argparse.Namespace) -> int:
    return deploy_mod.deploy(args.env)


def _fetch(args: argparse.Namespace) -> int:
    root = find_root()
    ups = upstreams.load(root)
    if args.name not in ups:
        raise OflError(f"unknown upstream {args.name!r}; known: {', '.join(ups)}", code=2)
    print(upstreams.ensure_checkout(root, ups[args.name]))
    return 0


def _list_upstreams(_args: argparse.Namespace) -> int:
    for up in upstreams.load(find_root()).values():
        print(f"{up.name:<24} {up.ref:<10} {up.sha}  {up.license.split(' (')[0]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ofl", description="Open Frontier Labs command line.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    for name, text in TARGETS.items():
        if name in ("deploy", "help"):
            continue
        p = sub.add_parser(name, help=text, description=text)
        p.set_defaults(func=_simple(_TARGET_FUNCS[name]))
    sub.add_parser("help", help=TARGETS["help"]).set_defaults(func=_help)

    p = sub.add_parser("deploy", help=TARGETS["deploy"], description=TARGETS["deploy"])
    p.add_argument("--env", required=True, help=f"one of: {', '.join(deploy_mod.ENVS)}")
    p.set_defaults(func=_deploy)

    p = sub.add_parser("fetch", help="clone a pinned upstream")
    p.add_argument("name")
    p.set_defaults(func=_fetch)
    sub.add_parser("upstreams", help="list pinned upstreams").set_defaults(func=_list_upstreams)

    run0_p = sub.add_parser("run0", help="phase 1 dense baseline (OLMo-core)")
    run0_sub = run0_p.add_subparsers(dest="action", required=True)
    p = run0_sub.add_parser("launch", help="build and run the OLMo-core command for one node")
    p.add_argument("--config", help="default: configs/phase1-dense/run0.toml")
    p.add_argument("--nproc", type=int, default=8, help="GPUs on this node (default 8)")
    p.add_argument("--python", help="interpreter with GPU torch and the pinned olmo-core installed")
    p.add_argument("--run-name")
    p.add_argument("--save-folder", help="default: checkpoints/run0")
    p.add_argument("--work-dir", help="default: .cache/run0/work")
    p.add_argument("--data-glob", help="tokenised .npy shards, for the fineweb-edu config")
    p.add_argument("--dry-run", action="store_true", help="print the command and exit")
    p.set_defaults(func=run0.launch)
    run0_sub.add_parser(
        "verify", help="fetch the pin and have the upstream script accept every override"
    ).set_defaults(func=lambda _a: run0.verify(find_root()))

    moe_p = sub.add_parser("moe", help="phase 1 MoE track (Megatron-LM, HF handbook)")
    moe_sub = moe_p.add_subparsers(dest="action", required=True)
    p = moe_sub.add_parser("launch", help="build and run the Megatron-LM command for one node")
    p.add_argument("--config", help="default: configs/phase1-moe/handbook-moe-6p9b.toml")
    p.add_argument(
        "--tokenizer-dir", required=True, help="HF tokenizer directory (Megatron --tokenizer-model)"
    )
    p.add_argument("--data-args-path", required=True, help="file of Megatron dataset prefixes")
    p.add_argument("--data-cache-path", help="default: .cache/moe/data-cache")
    p.add_argument("--save", help="checkpoint directory, default: checkpoints/moe")
    p.add_argument("--python", help="interpreter inside the Megatron training environment")
    p.add_argument(
        "--setup-check",
        action="store_true",
        help="stop after 120 iterations, as the handbook run does",
    )
    p.add_argument("--dry-run", action="store_true", help="print the command and exit")
    p.set_defaults(func=moe.launch)
    moe_sub.add_parser(
        "verify", help="fetch the pin and check every flag name against its source"
    ).set_defaults(func=lambda _a: moe.verify(find_root()))

    for phase, text in (
        ("post", "phase 2 post-training (not built)"),
        ("diloco", "phase 3 DiLoCo islands (not built)"),
    ):
        pp = sub.add_parser(phase, help=text)
        sub_p = pp.add_subparsers(dest="action", required=True)
        sub_p.add_parser("launch", help="not built").set_defaults(
            func=lambda _a, ph=phase: phases.not_built(ph)
        )
    return parser


_TARGET_FUNCS: dict[str, Callable[[], int]] = {
    "setup": tasks.setup,
    "dev": tasks.dev,
    "check": tasks.check,
    "test": tasks.test,
    "fix": tasks.fix,
    "build": tasks.build,
    "smoke": tasks.smoke,
    "doctor": lambda: doctor_mod.doctor(find_root()),
    "clean": tasks.clean,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        code: int = args.func(args)
    except OflError as exc:
        print(f"ofl: {exc}", file=sys.stderr)
        return exc.code
    return code
