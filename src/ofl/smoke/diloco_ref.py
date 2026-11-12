"""Phase 3 algorithm smoke: a reference DiLoCo in plain PyTorch on CPU.

This is NOT prime-diloco. The pinned prime-diloco needs CUDA and cannot run here, so this module
implements the DiLoCo algorithm itself (inner AdamW for H steps, averaged pseudo-gradient, outer
SGD with Nesterov momentum on the global weights) with torch.distributed/gloo and asserts:

  1. the held-out loss falls (the algorithm learns),
  2. workers genuinely diverge during the inner steps (otherwise 3 would be vacuous),
  3. after every outer sync all workers hold bit-identical weights.

Run as `python -m ofl.smoke.diloco_ref [--outer-lr X]`. Exits non-zero unless all three hold.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

from ofl.configs import SmokeDiloco, load_smoke_diloco
from ofl.repo import OflError, find_root
from ofl.smoke.toy import generate_tokens


class TinyLM(nn.Module):
    def __init__(self, cfg: SmokeDiloco) -> None:
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.sequence_length, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model,
            cfg.n_heads,
            4 * cfg.d_model,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(t))
        mask = nn.Transformer.generate_square_subsequent_mask(t)
        out: torch.Tensor = self.head(self.norm(self.blocks(h, mask=mask, is_causal=True)))
        return out


def _batch(
    stream: torch.Tensor, gen: torch.Generator, batch: int, seq: int
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, stream.numel() - seq - 1, (batch,), generator=gen)
    idx = starts[:, None] + torch.arange(seq + 1)[None, :]
    window = stream[idx]
    return window[:, :-1], window[:, 1:]


def _flat(params: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in params])


def _divergence(params: list[torch.Tensor]) -> float:
    """Largest absolute difference between any two workers' weights."""
    hi = _flat(params).clone()
    lo = hi.clone()
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    return float((hi - lo).abs().max())


def _worker(rank: int, cfg: SmokeDiloco, init_file: str, result_file: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=cfg.workers
    )
    try:
        torch.manual_seed(cfg.seed)
        model = TinyLM(cfg)
        for p in model.parameters():  # identical start on every worker
            dist.broadcast(p.data, src=0)
        params = list(model.parameters())

        # Global weights live outside the model; the outer optimiser steps them.
        global_params = [nn.Parameter(p.detach().clone()) for p in params]
        outer = torch.optim.SGD(
            global_params, lr=cfg.outer_lr, momentum=cfg.outer_momentum, nesterov=True
        )
        inner = torch.optim.AdamW(params, lr=cfg.inner_lr, weight_decay=0.0)

        # Each worker reads its own shard of the corpus; evaluation uses one fixed shared stream.
        shard = generate_tokens(200_000, cfg.vocab, cfg.noise, cfg.seed * 1000 + rank + 1)
        stream = torch.from_numpy(shard.astype("int64"))
        held = generate_tokens(20_000, cfg.vocab, cfg.noise, cfg.seed * 1000 + 999)
        held_stream = torch.from_numpy(held.astype("int64"))
        gen = torch.Generator().manual_seed(cfg.seed + rank)
        eval_x, eval_y = _batch(
            held_stream, torch.Generator().manual_seed(7), 256, cfg.sequence_length
        )

        def held_out_loss() -> float:
            with torch.no_grad():
                logits = model(eval_x)
                return float(F.cross_entropy(logits.reshape(-1, cfg.vocab), eval_y.reshape(-1)))

        initial = held_out_loss()
        rounds: list[dict[str, float]] = []
        for _ in range(cfg.outer_rounds):
            for _ in range(cfg.inner_steps):
                x, y = _batch(stream, gen, cfg.batch_size, cfg.sequence_length)
                loss = F.cross_entropy(model(x).reshape(-1, cfg.vocab), y.reshape(-1))
                inner.zero_grad(set_to_none=True)
                loss.backward()
                inner.step()
            before = _divergence(params)

            # Pseudo-gradient: how far the local weights moved from the global ones, averaged.
            delta = torch.cat(
                [
                    (g.detach() - p.detach()).reshape(-1)
                    for g, p in zip(global_params, params, strict=True)
                ]
            )
            dist.all_reduce(delta, op=dist.ReduceOp.SUM)
            delta /= cfg.workers
            offset = 0
            for g in global_params:
                n = g.numel()
                g.grad = delta[offset : offset + n].reshape(g.shape).clone()
                offset += n
            outer.step()
            with torch.no_grad():
                for g, p in zip(global_params, params, strict=True):
                    p.copy_(g)
            after = _divergence(params)
            rounds.append({"loss": held_out_loss(), "divergence_before": before, "after": after})

        if rank == 0:
            Path(result_file).write_text(
                json.dumps({"initial_loss": initial, "rounds": rounds}), encoding="utf-8"
            )
    finally:
        dist.destroy_process_group()


def evaluate(result: dict[str, Any], cfg: SmokeDiloco) -> list[str]:
    """Return the reasons this run does not pass (empty means pass)."""
    rounds = result["rounds"]
    if len(rounds) != cfg.outer_rounds:
        return [f"expected {cfg.outer_rounds} outer rounds, got {len(rounds)}"]
    problems: list[str] = []
    final = rounds[-1]["loss"]
    initial = result["initial_loss"]
    if final != final or final > cfg.loss_drop_ratio * initial:
        problems.append(
            f"held-out loss {initial:.3f} -> {final:.3f}; "
            f"required final <= {cfg.loss_drop_ratio} x initial"
        )
    for i, r in enumerate(rounds):
        if not r["divergence_before"] > 0.0:
            problems.append(f"round {i}: workers did not diverge during inner steps")
        if r["after"] != 0.0:
            problems.append(
                f"round {i}: workers differ by {r['after']:.3g} after the outer sync, expected 0"
            )
    return problems


def run(cfg: SmokeDiloco) -> dict[str, Any]:
    # Gloo picks a network interface by hostname; pin it to loopback so this works offline.
    if "GLOO_SOCKET_IFNAME" not in os.environ:
        names = {name for _, name in socket.if_nameindex()}
        for candidate in ("lo", "lo0"):
            if candidate in names:
                os.environ["GLOO_SOCKET_IFNAME"] = candidate
                break
    with tempfile.TemporaryDirectory(prefix="ofl-diloco-smoke-") as tmp:
        result_file = str(Path(tmp) / "result.json")
        mp.spawn(
            _worker,
            args=(cfg, str(Path(tmp) / "rendezvous"), result_file),
            nprocs=cfg.workers,
            join=True,
        )
        loaded: dict[str, Any] = json.loads(Path(result_file).read_text(encoding="utf-8"))
        return loaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ofl.smoke.diloco_ref")
    parser.add_argument("--outer-lr", type=float, help="override [diloco].outer_lr")
    args = parser.parse_args(argv)
    try:
        cfg = load_smoke_diloco(find_root())
    except OflError as exc:
        print(exc, file=sys.stderr)
        return exc.code
    if args.outer_lr is not None:
        cfg = dataclasses.replace(cfg, outer_lr=args.outer_lr)

    result = run(cfg)
    problems = evaluate(result, cfg)
    print(f"{'round':>5} {'held-out loss':>14} {'divergence before sync':>24} {'after sync':>11}")
    for i, r in enumerate(result["rounds"]):
        print(f"{i:>5} {r['loss']:>14.4f} {r['divergence_before']:>24.3e} {r['after']:>11.1e}")
    print(
        "OFL_SMOKE_RESULT "
        + json.dumps({"initial_loss": result["initial_loss"], "problems": problems})
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
