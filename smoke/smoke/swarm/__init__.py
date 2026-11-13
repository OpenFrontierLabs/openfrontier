"""Swarm proof of concept: `ofl swarm coordinator|status|report` and `ofl node join`."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from smoke.swarm import coordinator as co
from smoke.swarm import node
from smoke.swarm import report as rp


def _coordinator(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir).expanduser()
    state_dir.mkdir(parents=True, exist_ok=True)
    cfg = co.RunConfig(
        nodes=args.nodes,
        rounds=args.rounds,
        inner_steps=args.inner_steps,
        batch_size=args.batch_size,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
    )
    text = co.load_text(Path(args.text) if args.text else None, state_dir)
    model_kw = {"d_model": args.d_model, "n_layers": args.n_layers, "seq_len": args.seq_len}
    reporter = rp.Reporter(args.report_url, rp.load_token()) if args.report_url else None
    coord = co.Coordinator(text, cfg, model_kw, state_dir, reporter)
    host = args.host or co.tailscale_ip()
    co.serve(coord, host, args.port)
    return 0


def format_status(status: dict[str, Any]) -> str:
    now = time.time()
    lines = [f"{'node':<28} {'device':<6} {'gpu':<36} {'GiB':>6} {'torch':<14} heartbeat"]
    for n in status["registry"].values():
        ago = now - n["last_heartbeat"]
        lines.append(
            f"{n['id']:<28} {n['device']:<6} {n['gpu_name']:<36} {n['gpu_memory_gib']:>6} "
            f"{n['torch']:<14} {ago:.0f} s ago"
        )
    lines.append(
        f"\nrun {status['run_id']}: {status['state']}, round {status['round']}/{status['rounds']}, "
        f"members {', '.join(status['members']) or '-'}, {status['params']:,} params"
    )
    lines.append(f"initial val loss {status['initial_val_loss']:.4f}")
    for r in status["log"]:
        agree = set(r["node_hashes"].values()) == {r["served_hash"]}
        lines.append(
            f"round {r['round']:>3}  val loss {r['val_loss']:.4f}  "
            f"hashes {'agree' if agree else 'DIFFER'}  {r['elapsed']:.1f} s"
        )
    return "\n".join(lines)


def _status(args: argparse.Namespace) -> int:
    if args.coordinator:
        status = node.Client(args.coordinator).request("GET", "/v1/status")
    else:
        path = Path(args.state_dir).expanduser() / "status.json"
        status = json.loads(path.read_text(encoding="utf-8"))
    print(format_status(status))
    return 0


def _join(args: argparse.Namespace) -> int:
    return node.join(args.coordinator, args.device, args.github or node.detect_github())


def _report(args: argparse.Namespace) -> int:
    """Rebuild a finished run on the site from its run directory; --github owns every node."""
    run_dir = Path(args.run_dir).expanduser()
    receipt = json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
    lines = (run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines()
    log = [json.loads(line) for line in lines if line.strip()]
    token = rp.load_token()
    reports = rp.receipt_reports(receipt, log, args.github)
    for body in reports:
        rp.post_json(args.report_url.rstrip("/"), token, body)
    print(
        f"reported run {receipt['run_id']}: {len(log)} rounds in {len(reports)} posts", flush=True
    )
    return 0


def add_parsers(sub: Any) -> None:
    swarm = sub.add_parser("swarm", help="swarm PoC: coordinator, status and report (no auth)")
    swarm_sub = swarm.add_subparsers(dest="action", required=True)
    p = swarm_sub.add_parser("coordinator", help="run the registry and DiLoCo outer steps")
    p.add_argument("--host", help="bind address (default: this machine's Tailscale IPv4)")
    p.add_argument("--port", type=int, default=8470)
    p.add_argument("--state-dir", default=str(co.DEFAULT_STATE))
    p.add_argument("--text", help="training text file (default: tiny-shakespeare, fetched once)")
    p.add_argument(
        "--report-url",
        help="site to report nodes, runs and rounds to, e.g. https://openfrontier-labs.com "
        f"(token from {rp.INGEST_ENV_VAR} or {rp.TOKEN_FILE})",
    )
    d = co.RunConfig()
    p.add_argument("--nodes", type=int, default=d.nodes)
    p.add_argument("--rounds", type=int, default=d.rounds)
    p.add_argument("--inner-steps", type=int, default=d.inner_steps)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--inner-lr", type=float, default=d.inner_lr)
    p.add_argument("--outer-lr", type=float, default=d.outer_lr)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--seq-len", type=int, default=128)
    p.set_defaults(func=_coordinator)
    p = swarm_sub.add_parser("status", help="print the node registry and run progress")
    p.add_argument("--coordinator", help="live coordinator URL (default: read --state-dir)")
    p.add_argument("--state-dir", default=str(co.DEFAULT_STATE))
    p.set_defaults(func=_status)

    p = swarm_sub.add_parser("report", help="send a finished run's receipt and log to the site")
    p.add_argument("--run-dir", required=True, help="e.g. ~/.ofl/swarm/runs/<run id>")
    p.add_argument(
        "--github", required=True, help="GitHub login every node of the run is attributed to"
    )
    p.add_argument("--report-url", required=True)
    p.set_defaults(func=_report)

    node_p = sub.add_parser("node", help="swarm PoC: join a coordinator as a training node")
    node_sub = node_p.add_subparsers(dest="action", required=True)
    p = node_sub.add_parser("join", help="register, then train DiLoCo rounds until the run ends")
    p.add_argument("--coordinator", required=True, help="e.g. http://100.83.29.21:8470")
    p.add_argument("--device", help="cuda, mps or cpu (default: detect in that order)")
    p.add_argument("--github", help="GitHub login to attach this node to (default: the `gh` login)")
    p.set_defaults(func=_join)
