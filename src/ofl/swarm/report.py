"""Report the swarm to the website: nodes, the run and its rounds, as POST /v1/swarm/ingest.

Reporting is best effort. Reports go through a queue to a background thread, so a slow or
unreachable site never holds up registration, a round or a heartbeat; a failed POST is printed
and dropped (the next report carries current state anyway). Times on the wire are Unix ms.
"""

from __future__ import annotations

import json
import os
import queue
import ssl
import threading
import time
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import certifi

from ofl.repo import OflError

INGEST_ENV_VAR = "OFL_SWARM_TOKEN"
TOKEN_FILE = Path.home() / ".ofl" / "token"
HEARTBEAT_EVERY = 30.0
BACKFILL_ROUNDS_PER_POST = 5  # the site takes at most 10 rounds a report; stay well under


def load_token(env: Mapping[str, str] = os.environ, token_file: Path = TOKEN_FILE) -> str:
    """The ingest token: OFL_SWARM_TOKEN, else ~/.ofl/token."""
    token = env.get(INGEST_ENV_VAR, "").strip()
    if not token and token_file.is_file():
        token = token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise OflError(f"no ingest token: set {INGEST_ENV_VAR} or write it to {token_file}")
    return token


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


def node_report(entry: dict[str, Any], github: str | None = None) -> dict[str, Any]:
    """A registry entry as the site's node record."""
    node: dict[str, Any] = {
        "id": entry["id"],
        "hostname": entry["hostname"],
        "device": entry["device"],
        "gpu": entry["gpu_name"],
        "memory_gib": entry["gpu_memory_gib"],
        "torch": entry["torch"],
        "first_seen": _ms(entry["registered_at"]),
        "last_heartbeat": _ms(entry["last_heartbeat"]),
    }
    handle = github or entry.get("github")
    if handle:
        node["github"] = handle
    return node


def model_name(model: dict[str, Any]) -> str:
    return f"char GPT (d{model['d_model']}, {model['n_layers']} layers)"


def run_report(
    *,
    run_id: str,
    model: dict[str, Any],
    params: int,
    rounds: int,
    status: str,
    started_at: float,
    members: list[str],
    git_sha: str,
    initial_val_loss: float,
    weights_hash: str | None = None,
    ended_at: float | None = None,
    wall_seconds: float | None = None,
) -> dict[str, Any]:
    run: dict[str, Any] = {
        "id": run_id,
        "model": model_name(model),
        "params": params,
        "rounds": rounds,
        "status": status,
        "started_at": _ms(started_at),
        "git_sha": git_sha,
        "initial_val_loss": initial_val_loss,
        "nodes": members,
    }
    if weights_hash:
        run["weights_hash"] = weights_hash
    if ended_at is not None:
        run["ended_at"] = _ms(ended_at)
    if wall_seconds is not None:
        run["wall_seconds"] = wall_seconds
    return run


def round_report(run_id: str, entry: dict[str, Any], at: float) -> dict[str, Any]:
    """One log entry (as the coordinator writes it to log.jsonl) as the site's round record."""
    return {
        "run_id": run_id,
        "round": entry["round"],
        "val_loss": entry["val_loss"],
        "hashes_agree": set(entry["node_hashes"].values()) == {entry["served_hash"]},
        "elapsed": entry["elapsed"],
        "at": _ms(at),
        "nodes": [
            {
                "node_id": n,
                "train_loss": entry["node_train_loss"][n],
                "seconds": entry["node_inner_seconds"][n],
            }
            for n in entry["node_train_loss"]
        ],
    }


def receipt_reports(
    receipt: dict[str, Any], log: list[dict[str, Any]], github: str
) -> list[dict[str, Any]]:
    """Reports that rebuild a finished run on the site from its receipt and log, every node
    attributed to `github`. Training began when the last node registered, as in the coordinator."""
    nodes = [node_report(n, github) for n in receipt["nodes"]]
    registered = [n["registered_at"] for n in receipt["nodes"]]
    began = max(registered)
    run = run_report(
        run_id=receipt["run_id"],
        model=receipt["model"],
        params=receipt["params"],
        rounds=receipt["config"]["rounds"],
        status="done",
        started_at=min(registered),
        members=[n["id"] for n in receipt["nodes"]],
        git_sha=receipt["git_sha"],
        initial_val_loss=receipt["initial_val_loss"],
        weights_hash=receipt["weights_hash"],
        ended_at=began + receipt["wall_seconds"],
        wall_seconds=receipt["wall_seconds"],
    )
    rounds = [round_report(receipt["run_id"], e, began + e["elapsed"]) for e in log]
    reports: list[dict[str, Any]] = [{"nodes": nodes, "run": run}]
    for i in range(0, len(rounds), BACKFILL_ROUNDS_PER_POST):
        reports.append({"rounds": rounds[i : i + BACKFILL_ROUNDS_PER_POST]})
    return reports


Post = Callable[[str, str, dict[str, Any]], None]


def post_json(base: str, token: str, report: dict[str, Any]) -> None:
    req = urllib.request.Request(
        f"{base}/v1/swarm/ingest",
        data=json.dumps(report).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "ofl-swarm/1",
        },
    )
    tls = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, timeout=15, context=tls) as resp:
        resp.read()


class Reporter:
    """Queues reports for a background thread. `send` and `heartbeat` never block or raise."""

    def __init__(
        self,
        url: str,
        token: str,
        post: Post = post_json,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.post = post
        self.clock = clock
        self.queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.last_heartbeat: dict[str, float] = {}
        self.failures = 0
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self) -> None:
        while (report := self.queue.get()) is not None:
            try:
                self.post(self.url, self.token, report)
            except Exception as exc:  # reporting must never take training down
                self.failures += 1
                print(f"report to {self.url} failed: {exc}", flush=True)

    def send(self, report: dict[str, Any]) -> None:
        self.queue.put(report)

    def heartbeat(self, node: dict[str, Any]) -> None:
        """Report a node's liveness, at most once per HEARTBEAT_EVERY seconds per node."""
        now = self.clock()
        last = self.last_heartbeat.get(node["id"])
        if last is not None and now - last < HEARTBEAT_EVERY:
            return
        self.last_heartbeat[node["id"]] = now
        self.send({"nodes": [node]})

    def close(self, timeout: float = 20.0) -> None:
        """Flush what is queued (bounded), then stop the thread."""
        self.queue.put(None)
        self.thread.join(timeout)
