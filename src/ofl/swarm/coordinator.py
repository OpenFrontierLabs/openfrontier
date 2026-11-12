"""`ofl swarm coordinator`: node registry plus synchronous DiLoCo outer steps over HTTP.

One round: every node of the run pulls the global weights, runs H inner AdamW steps and pushes
its pseudo-gradient (global minus local). When all have arrived the coordinator averages them
in rank order, takes one SGD-Nesterov outer step and evaluates the held-out loss.
No auth, no ledger: a proof of concept on a trusted network (Tailscale).
"""

from __future__ import annotations

import json
import socket
import ssl
import subprocess
import threading
import time
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import certifi
import torch

from ofl.repo import OflError
from ofl.swarm import model as m
from ofl.swarm import report as rp

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
DEFAULT_STATE = Path.home() / ".ofl" / "swarm"


@dataclass(frozen=True)
class RunConfig:
    nodes: int = 2
    rounds: int = 30
    inner_steps: int = 50
    batch_size: int = 32
    inner_lr: float = 1e-3
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    seed: int = 0
    eval_batches: int = 8


def load_text(path: Path | None, state_dir: Path) -> str:
    """The training text: `path`, else tiny-shakespeare fetched once into the state dir."""
    if path is not None:
        return path.read_text(encoding="utf-8")
    cached = state_dir / "data" / "tinyshakespeare.txt"
    if not cached.is_file():
        cached.parent.mkdir(parents=True, exist_ok=True)
        tls = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(SHAKESPEARE_URL, timeout=60, context=tls) as resp:
            cached.write_bytes(resp.read())
    return cached.read_text(encoding="utf-8")


def git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip() or "unknown"


class Coordinator:
    def __init__(
        self,
        text: str,
        cfg: RunConfig,
        mcfg_kw: dict[str, int],
        state_dir: Path,
        reporter: rp.Reporter | None = None,
    ):
        self.cfg = cfg
        self.reporter = reporter
        self.created = time.time()
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.state_dir = state_dir
        self.text = text
        self.chars = sorted(set(text))
        self.mcfg = m.ModelConfig(vocab=len(self.chars), **mcfg_kw)
        stoi = {c: i for i, c in enumerate(self.chars)}
        ids = torch.tensor([stoi[c] for c in text], dtype=torch.int64)
        self.split = int(0.9 * len(ids))
        self.val = ids[self.split :]

        torch.manual_seed(cfg.seed)
        self.model = m.CharGPT(self.mcfg)
        self.global_w = torch.nn.Parameter(m.get_flat(self.model))
        self.outer = torch.optim.SGD(
            [self.global_w], lr=cfg.outer_lr, momentum=cfg.outer_momentum, nesterov=True
        )
        self.run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.run_dir = state_dir / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = state_dir / "registry.json"
        self.registry: dict[str, dict[str, Any]] = (
            json.loads(self.registry_file.read_text(encoding="utf-8"))
            if self.registry_file.is_file()
            else {}
        )
        self.members: list[str] = []  # node ids of this run, in rank order
        self.round = 0
        self.state = "waiting"
        self.pending: dict[str, tuple[torch.Tensor, dict[str, Any]]] = {}
        self.log: list[dict[str, Any]] = []
        self.finals: dict[str, str] = {}
        self.started = 0.0
        self.wall = 0.0
        self.sample = ""
        self.initial_loss = self._eval()
        self.weights = m.to_bytes(self.global_w.detach())
        self.hash = m.weights_hash(self.global_w.detach())
        self.git_sha = git_sha()
        self._persist()

    # ---- evaluation and persistence -------------------------------------------------------
    def _eval(self) -> float:
        m.set_flat(self.model, self.global_w.detach())
        gen = torch.Generator().manual_seed(1234)
        with torch.no_grad():
            losses = [
                float(self.model.loss(*m.batch(self.val, gen, 32, self.mcfg.seq_len)))
                for _ in range(self.cfg.eval_batches)
            ]
        return sum(losses) / len(losses)

    def status(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "state": self.state,
            "round": self.round,
            "rounds": self.cfg.rounds,
            "weights_hash": self.hash,
            "params": int(self.global_w.numel()),
            "initial_val_loss": self.initial_loss,
            "members": self.members,
            "registry": self.registry,
            "log": self.log,
            "wall_seconds": self.wall,
        }

    def _persist(self) -> None:
        self.registry_file.write_text(json.dumps(self.registry, indent=2), encoding="utf-8")
        (self.state_dir / "status.json").write_text(
            json.dumps(self.status(), indent=2), encoding="utf-8"
        )

    def _write_receipt(self) -> None:
        receipt = {
            "run_id": self.run_id,
            "git_sha": self.git_sha,
            "config": self.cfg.__dict__,
            "model": self.mcfg.to_json(),
            "params": int(self.global_w.numel()),
            "nodes": [self.registry[n] for n in self.members],
            "initial_val_loss": self.initial_loss,
            "rounds": self.log,
            "final_val_loss": self.log[-1]["val_loss"] if self.log else None,
            "wall_seconds": self.wall,
            "weights_hash": self.hash,
            "node_final_hashes": self.finals,
            "hashes_agree": all(
                set(r["node_hashes"].values()) == {r["served_hash"]} for r in self.log
            )
            and set(self.finals.values()) <= {self.hash},
            "sample_file": "sample.txt",
        }
        (self.run_dir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    # ---- reporting to the website (best effort; Reporter never blocks or raises) -----------
    def _run_report(self, *, status: str, ended: bool = False) -> dict[str, Any]:
        return rp.run_report(
            run_id=self.run_id,
            model=self.mcfg.to_json(),
            params=int(self.global_w.numel()),
            rounds=self.cfg.rounds,
            status=status,
            started_at=self.created,
            members=list(self.members),
            git_sha=self.git_sha,
            initial_val_loss=self.initial_loss,
            weights_hash=self.hash,
            ended_at=time.time() if ended else None,
            wall_seconds=self.wall if ended else None,
        )

    def _report(self, **parts: Any) -> None:
        if self.reporter is not None:
            self.reporter.send(parts)

    # ---- protocol ---------------------------------------------------------------------------
    def register(self, info: dict[str, Any]) -> dict[str, Any]:
        node_id = f"{info['hostname']}-{info['device']}"
        with self.lock:
            now = time.time()
            prior = self.registry.get(node_id, {})
            self.registry[node_id] = {
                **info,
                "id": node_id,
                "registered_at": prior.get("registered_at", now),
                "last_heartbeat": now,
            }
            if node_id not in self.members:
                if self.state != "waiting" or len(self.members) >= self.cfg.nodes:
                    raise OflError(f"run {self.run_id} is full or already started", code=409)
                self.members.append(node_id)
            if self.state == "waiting" and len(self.members) == self.cfg.nodes:
                self.state = "training"
                self.started = time.time()
            self._persist()
            rank = self.members.index(node_id)
            if self.reporter is not None:
                self.reporter.last_heartbeat[node_id] = self.reporter.clock()
            self._report(
                nodes=[rp.node_report(self.registry[node_id])],
                run=self._run_report(status=self.state),
            )
        print(f"registered {node_id} as rank {rank}: {info.get('gpu_name')}", flush=True)
        return {
            "node_id": node_id,
            "rank": rank,
            "run_id": self.run_id,
            "model": self.mcfg.to_json(),
            "chars": "".join(self.chars),
            "train_chars": self.split,
            "nodes": self.cfg.nodes,
            "inner_steps": self.cfg.inner_steps,
            "batch_size": self.cfg.batch_size,
            "inner_lr": self.cfg.inner_lr,
            "seed": self.cfg.seed,
        }

    def heartbeat(self, node_id: str) -> None:
        with self.lock:
            if node_id not in self.registry:
                raise OflError(f"unknown node {node_id}", code=404)
            self.registry[node_id]["last_heartbeat"] = time.time()
            self._persist()
            if self.reporter is not None:
                self.reporter.heartbeat(rp.node_report(self.registry[node_id]))

    def submit(self, rnd: int, node_id: str, delta: torch.Tensor, meta: dict[str, Any]) -> None:
        with self.lock:
            if self.state != "training" or rnd != self.round or node_id not in self.members:
                raise OflError(f"not accepting round {rnd} from {node_id}", code=409)
            if delta.numel() != self.global_w.numel():
                raise OflError("pseudo-gradient has the wrong size", code=400)
            self.pending[node_id] = (delta, meta)
            if len(self.pending) < len(self.members):
                return
        # The last push of a round answers at once; the outer step, eval and (after the last
        # round) sampling run on their own thread so no request waits on them.
        threading.Thread(target=self._aggregate, args=(rnd,), daemon=True).start()

    def _aggregate(self, rnd: int) -> None:
        with self.lock:
            deltas = [self.pending[n][0] for n in self.members]
            self.global_w.grad = torch.stack(deltas).mean(0)
            self.outer.step()
            loss = self._eval()
            served = self.hash
            self.weights = m.to_bytes(self.global_w.detach())
            self.hash = m.weights_hash(self.global_w.detach())
            entry: dict[str, Any] = {
                "round": rnd,
                "val_loss": loss,
                "served_hash": served,
                "node_hashes": {n: self.pending[n][1]["weights_hash"] for n in self.members},
                "node_train_loss": {n: self.pending[n][1]["train_loss"] for n in self.members},
                "node_inner_seconds": {n: self.pending[n][1]["seconds"] for n in self.members},
                "new_hash": self.hash,
                "elapsed": time.time() - self.started,
            }
            self.log.append(entry)
            with (self.run_dir / "log.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            self.pending.clear()
            self._report(
                run=self._run_report(status="training"),
                rounds=[rp.round_report(self.run_id, entry, time.time())],
            )
            agree = set(entry["node_hashes"].values()) == {served}
            print(
                f"round {rnd:>3} val_loss {loss:.4f} hashes {'agree' if agree else 'DIFFER'}",
                flush=True,
            )
            last = rnd + 1 == self.cfg.rounds
            if not last:
                self.round += 1
            self._persist()
        if last:
            self._finish()

    def _finish(self) -> None:
        """Sample from the final weights, then open the run's end to the nodes."""
        wall = time.time() - self.started
        sampler = m.CharGPT(self.mcfg)
        m.set_flat(sampler, self.global_w.detach())
        stoi = {c: i for i, c in enumerate(self.chars)}
        prompt = "ROMEO:" if set("ROMEO:") <= stoi.keys() else self.text[:6]
        ids = m.sample(sampler, [stoi[c] for c in prompt], 600, seed=self.cfg.seed)
        with self.lock:
            self.wall = wall
            self.sample = "".join(self.chars[i] for i in ids)
            (self.run_dir / "sample.txt").write_text(self.sample + "\n", encoding="utf-8")
            (self.run_dir / "weights.f32").write_bytes(self.weights)
            self.round = self.cfg.rounds
            self.state = "done"
            self._report(run=self._run_report(status="done", ended=True))
            self._write_receipt()
            self._persist()

    def final(self, node_id: str, digest: str) -> None:
        with self.lock:
            self.finals[node_id] = digest
            self._write_receipt()
            if set(self.finals) >= set(self.members):
                self.done.set()


def _handler(coord: Coordinator) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _send(self, code: int, body: bytes, ctype: str, headers: dict[str, str]) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json", {})

        def _body(self) -> bytes:
            return self.rfile.read(int(self.headers.get("Content-Length", "0")))

        def _route(self, method: str) -> None:
            url = urlparse(self.path)
            parts = url.path.strip("/").split("/")
            try:
                if method == "GET" and url.path == "/v1/status":
                    with coord.lock:
                        self._json(coord.status())
                elif method == "GET" and url.path == "/v1/data":
                    self._send(200, coord.text.encode(), "text/plain; charset=utf-8", {})
                elif method == "GET" and url.path == "/v1/weights":
                    with coord.lock:
                        body, digest, rnd = coord.weights, coord.hash, coord.round
                    headers = {"X-Weights-Hash": digest, "X-Round": str(rnd)}
                    self._send(200, body, "application/octet-stream", headers)
                elif method == "POST" and url.path == "/v1/nodes":
                    self._json(coord.register(json.loads(self._body())))
                elif method == "POST" and parts[:2] == ["v1", "nodes"] and len(parts) == 4:
                    if parts[3] == "heartbeat":
                        coord.heartbeat(parts[2])
                    elif parts[3] == "final":
                        coord.final(parts[2], json.loads(self._body())["weights_hash"])
                    else:
                        raise OflError("not found", code=404)
                    self._json({"ok": True})
                elif method == "POST" and parts[:2] == ["v1", "rounds"] and len(parts) == 4:
                    node = parse_qs(url.query)["node"][0]
                    meta = {
                        "weights_hash": self.headers["X-Weights-Hash"],
                        "train_loss": float(self.headers["X-Train-Loss"]),
                        "seconds": float(self.headers["X-Inner-Seconds"]),
                    }
                    coord.submit(int(parts[2]), node, m.from_bytes(self._body()), meta)
                    self._json({"ok": True})
                else:
                    raise OflError("not found", code=404)
            except OflError as exc:
                self._json({"error": str(exc)}, code=exc.code)

        def do_GET(self) -> None:
            self._route("GET")

        def do_POST(self) -> None:
            self._route("POST")

    return Handler


def serve(
    coord: Coordinator, host: str, port: int, on_bound: Callable[[int], None] | None = None
) -> int:
    """Serve until every member has reported its final weights hash. Returns the bound port."""
    server = ThreadingHTTPServer((host, port), _handler(coord))
    bound = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"coordinator run {coord.run_id} on http://{host}:{bound} ({coord.run_dir})", flush=True)
    if on_bound is not None:
        on_bound(bound)
    coord.done.wait()
    if coord.reporter is not None:
        coord.reporter.close()
    server.shutdown()
    server.server_close()
    print(f"run done: receipt {coord.run_dir / 'receipt.json'}", flush=True)
    return bound


def tailscale_ip() -> str:
    out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, check=False)
    ip = out.stdout.split()[0] if out.stdout.split() else ""
    if not ip:
        raise OflError("no Tailscale IPv4 address; pass --host")
    socket.inet_aton(ip)
    return ip
