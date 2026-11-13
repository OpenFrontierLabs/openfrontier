"""`python -m smoke.swarm`: the swarm PoC commands the Rust `ofl` CLI forwards here."""

from __future__ import annotations

import argparse
import sys

from smoke.configs import SmokeError
from smoke.swarm import add_parsers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ofl", description="Open Frontier Labs swarm PoC.")
    add_parsers(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        code: int = args.func(args)
    except SmokeError as exc:
        print(f"ofl: {exc}", file=sys.stderr)
        return exc.code
    return code


if __name__ == "__main__":
    raise SystemExit(main())
