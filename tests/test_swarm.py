import json
import threading
from pathlib import Path

from ofl.swarm import coordinator as co
from ofl.swarm import format_status, node


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
