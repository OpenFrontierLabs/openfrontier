"""`ofl node join`: register with a coordinator, then pull, train H steps, push, repeat."""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import torch

from ofl.repo import OflError
from ofl.swarm import model as m


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sysctl(key: str) -> str:
    out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=False)
    return out.stdout.strip()


def detect_github() -> str | None:
    """The signed-in `gh` login, if gh is installed and authenticated."""
    try:
        out = subprocess.run(
            ["gh", "api", "user", "--jq", ".login"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    login = out.stdout.strip()
    return login if out.returncode == 0 and login else None


def describe(device: str, github: str | None = None) -> dict[str, Any]:
    """What this node is: the registry entry it sends."""
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        gpu, mem = props.name, props.total_memory
    elif device == "mps":
        gpu, mem = _sysctl("machdep.cpu.brand_string"), int(_sysctl("hw.memsize") or 0)
    else:
        gpu, mem = platform.processor() or platform.machine(), 0
    info = {
        "hostname": socket.gethostname().split(".")[0],
        "device": device,
        "gpu_name": gpu,
        "gpu_memory_gib": round(mem / 2**30, 1),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    return {**info, "github": github} if github else info


class Client:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    def request(self, method: str, path: str, body: bytes | None = None, **headers: str) -> Any:
        req = urllib.request.Request(self.base + path, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
                if resp.headers.get_content_type() == "application/json":
                    return json.loads(data)
                return data, dict(resp.headers)
        except urllib.error.HTTPError as exc:
            raise OflError(f"{method} {path}: {exc.code} {exc.read().decode()}") from exc

    def post_json(self, path: str, obj: Any) -> Any:
        return self.request(
            "POST", path, json.dumps(obj).encode(), **{"Content-Type": "application/json"}
        )


def _heartbeat(client: Client, node_id: str, stop: threading.Event) -> None:
    while not stop.wait(5.0):
        try:
            client.post_json(f"/v1/nodes/{node_id}/heartbeat", {})
        except (OSError, OflError) as exc:
            print(f"heartbeat failed: {exc}", flush=True)


def _pull(client: Client, model: m.CharGPT) -> tuple[int, str]:
    """Load the global weights; return their round and the hash of what the model now holds."""
    body, headers = client.request("GET", "/v1/weights")
    if hashlib.sha256(body).hexdigest() != headers["X-Weights-Hash"]:
        raise OflError("global weights corrupted in transit")
    m.set_flat(model, m.from_bytes(body))
    return int(headers["X-Round"]), m.weights_hash(m.get_flat(model))


def join(coordinator: str, device: str | None = None, github: str | None = None) -> int:
    client = Client(coordinator)
    dev = pick_device(device)
    info = describe(dev, github)
    cfg = client.post_json("/v1/nodes", info)
    node_id, rank = cfg["node_id"], cfg["rank"]
    who = f" for @{github}" if github else ""
    print(
        f"joined run {cfg['run_id']} as {node_id}{who} rank {rank} on {info['gpu_name']}",
        flush=True,
    )
    stop = threading.Event()
    threading.Thread(target=_heartbeat, args=(client, node_id, stop), daemon=True).start()
    try:
        return _train(client, cfg, dev, node_id, rank)
    finally:
        stop.set()


def _train(client: Client, cfg: dict[str, Any], dev: str, node_id: str, rank: int) -> int:
    text, _ = client.request("GET", "/v1/data")
    stoi = {c: i for i, c in enumerate(cfg["chars"])}
    ids = torch.tensor([stoi[c] for c in text.decode("utf-8")[: cfg["train_chars"]]])
    per = ids.numel() // cfg["nodes"]
    shard = ids[rank * per : (rank + 1) * per]
    mcfg = m.ModelConfig.from_json(cfg["model"])
    model = m.CharGPT(mcfg).to(dev)
    inner = torch.optim.AdamW(model.parameters(), lr=cfg["inner_lr"], weight_decay=0.0)
    gen = torch.Generator().manual_seed(cfg["seed"] * 1000 + rank)
    done_round = -1
    while True:
        status = client.request("GET", "/v1/status")
        if status["state"] == "done":
            break
        if status["state"] != "training" or status["round"] <= done_round:
            time.sleep(0.2)
            continue
        rnd, held = _pull(client, model)
        start = m.get_flat(model)
        t0 = time.perf_counter()
        loss = torch.zeros((), device=dev)
        for _ in range(cfg["inner_steps"]):
            x, y = m.batch(shard, gen, cfg["batch_size"], mcfg.seq_len)
            loss = model.loss(x.to(dev), y.to(dev))
            inner.zero_grad(set_to_none=True)
            loss.backward()
            inner.step()
        train_loss = float(loss.detach())  # syncs the device, so the timing below is real
        seconds = time.perf_counter() - t0
        delta = start - m.get_flat(model)
        client.request(
            "POST",
            f"/v1/rounds/{rnd}/pseudo-gradients?node={node_id}",
            m.to_bytes(delta),
            **{
                "Content-Type": "application/octet-stream",
                "X-Weights-Hash": held,
                "X-Train-Loss": repr(train_loss),
                "X-Inner-Seconds": repr(seconds),
            },
        )
        print(f"round {rnd:>3} train_loss {train_loss:.4f} inner {seconds:.2f} s", flush=True)
        done_round = rnd
    _, final = _pull(client, model)
    client.post_json(f"/v1/nodes/{node_id}/final", {"weights_hash": final})
    print(f"done; final weights {final}", flush=True)
    return 0
