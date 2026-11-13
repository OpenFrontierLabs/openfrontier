"""Config loading and repo-root discovery for the Python smoke scripts.

The `ofl` CLI is Rust; these scripts run under it as `python -m smoke.<module>`
and read the same configs/ files. This module is the only Python config surface.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


class SmokeError(Exception):
    """An expected failure: printed without a traceback, exits with `code`."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


def find_root(start: Path | None = None) -> Path:
    """The checkout root: the directory holding configs/upstreams.toml."""
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        if (directory / "configs" / "upstreams.toml").is_file():
            return directory
    raise SmokeError(
        f"not inside an openfrontier checkout (no configs/upstreams.toml above {here})"
    )


def _load(root: Path, rel: str) -> dict:
    with (root / rel).open("rb") as fh:
        return tomllib.load(fh)


@dataclass(frozen=True)
class SmokeTiny:
    """configs/phase1-dense/smoke-tiny.toml"""

    factory: str
    vocab: int
    sequence_length: int
    sequences: int
    noise: float
    corpus_seed: int
    steps: int
    dev_steps: int
    batch_sequences: int
    lr: float
    warmup_steps: int
    init_seed: int
    loss_drop_ratio: float


def load_smoke_tiny(root: Path) -> SmokeTiny:
    raw = _load(root, "configs/phase1-dense/smoke-tiny.toml")
    return SmokeTiny(
        factory=raw["model"]["factory"],
        vocab=raw["model"]["vocab"],
        sequence_length=raw["data"]["sequence_length"],
        sequences=raw["data"]["sequences"],
        noise=raw["data"]["noise"],
        corpus_seed=raw["data"]["corpus_seed"],
        steps=raw["train"]["steps"],
        dev_steps=raw["train"]["dev_steps"],
        batch_sequences=raw["train"]["batch_sequences"],
        lr=raw["train"]["lr"],
        warmup_steps=raw["train"]["warmup_steps"],
        init_seed=raw["train"]["init_seed"],
        loss_drop_ratio=raw["assert"]["loss_drop_ratio"],
    )


@dataclass(frozen=True)
class SmokeDiloco:
    """configs/phase3-diloco/smoke-reference.toml"""

    workers: int
    vocab: int
    sequence_length: int
    d_model: int
    n_layers: int
    n_heads: int
    outer_rounds: int
    inner_steps: int
    inner_lr: float
    outer_lr: float
    outer_momentum: float
    batch_size: int
    noise: float
    seed: int
    loss_drop_ratio: float


def load_smoke_diloco(root: Path) -> SmokeDiloco:
    raw = _load(root, "configs/phase3-diloco/smoke-reference.toml")
    return SmokeDiloco(
        workers=raw["run"]["workers"],
        vocab=raw["model"]["vocab"],
        sequence_length=raw["model"]["sequence_length"],
        d_model=raw["model"]["d_model"],
        n_layers=raw["model"]["n_layers"],
        n_heads=raw["model"]["n_heads"],
        outer_rounds=raw["diloco"]["outer_rounds"],
        inner_steps=raw["diloco"]["inner_steps"],
        inner_lr=raw["diloco"]["inner_lr"],
        outer_lr=raw["diloco"]["outer_lr"],
        outer_momentum=raw["diloco"]["outer_momentum"],
        batch_size=raw["data"]["batch_size"],
        noise=raw["data"]["noise"],
        seed=raw["run"]["seed"],
        loss_drop_ratio=raw["assert"]["loss_drop_ratio"],
    )