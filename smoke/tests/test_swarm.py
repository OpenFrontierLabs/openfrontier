import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from smoke.configs import SmokeError
from smoke.swarm import coordinator as co
from smoke.swarm import format_status, node, report


def test_two_cpu_nodes_train_one_model_over_http(tmp_path: Path) -> None:
    text = "".join(f"the {w} sat on the mat. " for w in ("cat", "dog", "rat", "bat") * 600)
    cfg = co.RunConfig(nodes=2, rounds=5, inner_steps=15, batch_size=16, inner_lr=3e-3)
    coord = co.Coordinator(text, cfg, {"d_model": 32, "n_layers": 1, "seq_len": 32}, tmp_path)
    bound: list[int] = []
    ready = threading.Event()

    def on_bound(port: int) -> None:
        bound.append(port)
        ready.set()

    server = threading.Thread(target=co.serve, args=(coord, "127.0.0.1", 0, on_bound), daemon=True)
    server.start()
    assert ready.wait(30)
    url = f"http://127.0.0.1:{bound[0]}"
    # Two distinct node identities on one host: register under different devices.
    codes: list[int] = []
    workers = [
        threading.Thread(target=lambda d=d: codes.append(node.join(url, d)), daemon=True)
        for d in ("cpu", "cpu:0")
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join(300)
    server.join(60)
    assert codes == [0, 0]

    receipt = json.loads((coord.run_dir / "receipt.json").read_text(encoding="utf-8"))
    assert len(receipt["nodes"]) == 2
    assert len(receipt["rounds"]) == 5
    assert receipt["final_val_loss"] < 0.7 * receipt["initial_val_loss"]
    assert receipt["hashes_agree"] is True
    assert set(receipt["node_final_hashes"].values()) == {receipt["weights_hash"]}
    for r in receipt["rounds"]:
        assert r["node_train_loss"]  # both nodes really trained
    # weights moved every round
    assert len({r["new_hash"] for r in receipt["rounds"]}) == 5
    assert (coord.run_dir / "sample.txt").read_text(encoding="utf-8").startswith("the ")
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert "done, round 5/5" in format_status(status)


def _members_within(client: node.Client, ids: set[str], timeout: float) -> set[str]:
    """Poll the coordinator until every id in `ids` is a member, or `timeout` seconds pass."""
    deadline = time.monotonic() + timeout
    seen: set[str] = set()
    while time.monotonic() < deadline:
        seen = set(client.request("GET", "/v1/status")["members"])
        if ids <= seen:
            break
        time.sleep(0.05)
    return seen


def test_two_peers_discover_each_other_within_timeout(tmp_path: Path) -> None:
    text = "".join(f"the {w} sat on the mat. " for w in ("cat", "dog", "rat", "bat") * 100)
    cfg = co.RunConfig(nodes=2, rounds=1, inner_steps=1, batch_size=8, eval_batches=2)
    coord = co.Coordinator(text, cfg, {"d_model": 16, "n_layers": 1, "seq_len": 16}, tmp_path)
    bound: list[int] = []
    ready = threading.Event()

    def on_bound(port: int) -> None:
        bound.append(port)
        ready.set()

    server = threading.Thread(target=co.serve, args=(coord, "127.0.0.1", 0, on_bound), daemon=True)
    server.start()
    assert ready.wait(30)
    url = f"http://127.0.0.1:{bound[0]}"
    # Two peers on one host, distinct identities via the device string; port 0, no fixed port.
    first = node.Client(url)
    second = node.Client(url)
    first_id = first.post_json("/v1/nodes", node.describe("cpu"))["node_id"]
    # The second peer bootstraps from the first: it reads the registry before joining.
    assert {first_id} <= _members_within(second, {first_id}, 10)
    second_id = second.post_json("/v1/nodes", node.describe("cpu:0"))["node_id"]
    assert second_id != first_id
    both = {first_id, second_id}
    assert both <= _members_within(first, both, 10)
    assert both <= _members_within(second, both, 10)
    coord.done.set()
    server.join(30)


def _run_swarm(tmp_path: Path, reporter: report.Reporter, github: str | None) -> co.Coordinator:
    text = "".join(f"the {w} sat on the mat. " for w in ("cat", "dog", "rat", "bat") * 600)
    cfg = co.RunConfig(nodes=2, rounds=3, inner_steps=5, batch_size=16, inner_lr=3e-3)
    coord = co.Coordinator(
        text, cfg, {"d_model": 32, "n_layers": 1, "seq_len": 32}, tmp_path, reporter
    )
    bound: list[int] = []
    ready = threading.Event()

    def on_bound(port: int) -> None:
        bound.append(port)
        ready.set()

    server = threading.Thread(target=co.serve, args=(coord, "127.0.0.1", 0, on_bound), daemon=True)
    server.start()
    assert ready.wait(30)
    url = f"http://127.0.0.1:{bound[0]}"
    codes: list[int] = []
    workers = [
        threading.Thread(target=lambda d=d: codes.append(node.join(url, d, github)), daemon=True)
        for d in ("cpu", "cpu:0")
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join(300)
    server.join(60)
    assert codes == [0, 0]
    return coord


def test_coordinator_reports_nodes_run_and_rounds(tmp_path: Path) -> None:
    posted: list[dict[str, Any]] = []
    reporter = report.Reporter(
        "https://site.test", "t0ken", lambda _u, _t, body: posted.append(body)
    )
    coord = _run_swarm(tmp_path, reporter, "octo")
    reported_nodes = [n for body in posted for n in body.get("nodes", [])]
    assert {n["github"] for n in reported_nodes} == {"octo"}
    assert {n["id"] for n in reported_nodes} == {f"{h}-{d}" for h, d in _node_ids(coord)}
    runs = [body["run"] for body in posted if "run" in body]
    assert runs[0]["status"] in ("waiting", "training")
    assert runs[-1]["status"] == "done"
    assert runs[-1]["weights_hash"] == coord.hash
    assert runs[-1]["wall_seconds"] == coord.wall
    rounds = [r for body in posted for r in body.get("rounds", [])]
    assert [r["round"] for r in rounds] == [0, 1, 2]
    assert all(r["hashes_agree"] and len(r["nodes"]) == 2 for r in rounds)
    assert rounds[-1]["val_loss"] == coord.log[-1]["val_loss"]
    # Registry heartbeats (every 5 s) are throttled: a short run reports only registrations.
    assert sum(len(body.get("nodes", [])) for body in posted) == 2


def _node_ids(coord: co.Coordinator) -> list[tuple[str, str]]:
    return [(n["hostname"], n["device"]) for n in coord.registry.values()]


def test_failing_reports_never_stall_training(tmp_path: Path) -> None:
    def refuse(_url: str, _token: str, _body: dict[str, Any]) -> None:
        raise OSError("site unreachable")

    reporter = report.Reporter("https://site.test", "t0ken", refuse)
    coord = _run_swarm(tmp_path, reporter, None)
    receipt = json.loads((coord.run_dir / "receipt.json").read_text(encoding="utf-8"))
    assert len(receipt["rounds"]) == 3
    assert reporter.failures > 0


def test_heartbeats_are_throttled_per_node() -> None:
    now = [0.0]
    posted: list[dict[str, Any]] = []
    reporter = report.Reporter(
        "https://site.test", "t", lambda *a: posted.append(a[2]), lambda: now[0]
    )
    entry = {"id": "a-cpu"}
    for t in (0.0, 5.0, 29.0, 31.0, 40.0, 62.0):
        now[0] = t
        reporter.heartbeat(entry)
    reporter.close()
    assert len(posted) == 3  # at 0 s, 31 s and 62 s


def test_receipt_backfill_attributes_every_node_and_chunks_rounds() -> None:
    nodes = [
        {
            "id": f"h{i}-cuda",
            "hostname": f"h{i}",
            "device": "cuda",
            "gpu_name": "G",
            "gpu_memory_gib": 8.0,
            "torch": "2",
            "registered_at": 100.0 + i,
            "last_heartbeat": 400.0,
        }
        for i in range(2)
    ]
    ids = [n["id"] for n in nodes]
    log = [
        {
            "round": r,
            "val_loss": 3.0 - r / 10,
            "served_hash": "x",
            "node_hashes": dict.fromkeys(ids, "x" if r != 7 else "y"),
            "node_train_loss": dict.fromkeys(ids, 2.0),
            "node_inner_seconds": dict.fromkeys(ids, 1.5),
            "new_hash": "z",
            "elapsed": 10.0 * (r + 1),
        }
        for r in range(12)
    ]
    receipt = {
        "run_id": "r",
        "git_sha": "abc",
        "model": {"d_model": 8, "n_layers": 1},
        "params": 99,
        "config": {"rounds": 12},
        "nodes": nodes,
        "initial_val_loss": 4.0,
        "weights_hash": "w",
        "wall_seconds": 120.0,
    }
    first, *chunks = report.receipt_reports(receipt, log, "octo")
    assert [n["github"] for n in first["nodes"]] == ["octo", "octo"]
    assert first["run"]["status"] == "done" and first["run"]["nodes"] == ids
    assert first["run"]["started_at"] == 100_000 and first["run"]["ended_at"] == 101_000 + 120_000
    assert [len(c["rounds"]) for c in chunks] == [5, 5, 2]
    rounds = [r for c in chunks for r in c["rounds"]]
    assert rounds[0]["at"] == 101_000 + 10_000
    assert [r["hashes_agree"] for r in rounds].count(False) == 1 and rounds[7][
        "hashes_agree"
    ] is False


def test_token_comes_from_env_then_file(tmp_path: Path) -> None:
    f = tmp_path / "token"
    with pytest.raises(SmokeError):
        report.load_token({}, f)
    f.write_text("from-file\n", encoding="utf-8")
    assert report.load_token({}, f) == "from-file"
    assert report.load_token({report.INGEST_ENV_VAR: " from-env "}, f) == "from-env"
