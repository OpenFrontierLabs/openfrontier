"""Loading and checking everything under configs/ and manifests/."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ofl import upstreams

PHASE_DIRS = ("phase1-dense", "phase1-moe", "phase2-post", "phase3-diloco")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
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
    raw = load_toml(root / "configs" / "phase1-dense" / "smoke-tiny.toml")
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
    raw = load_toml(root / "configs" / "phase3-diloco" / "smoke-reference.toml")
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


def _is_flag_value(value: object) -> bool:
    return isinstance(value, bool | int | str)


def _check_meta(
    rel: str, raw: dict[str, Any], phase: str, ups: dict[str, upstreams.Upstream]
) -> list[str]:
    meta = raw.get("meta")
    if not isinstance(meta, dict):
        return [f"{rel}: missing [meta]"]
    out: list[str] = []
    if meta.get("phase") != phase:
        out.append(f"{rel}: [meta].phase must be {phase!r}, got {meta.get('phase')!r}")
    if meta.get("upstream") not in ups:
        out.append(f"{rel}: [meta].upstream {meta.get('upstream')!r} is not in upstreams.toml")
    return out


def _check_overrides(rel: str, raw: dict[str, Any]) -> list[str]:
    overrides = raw.get("overrides")
    if not isinstance(overrides, dict) or not overrides:
        return [f"{rel}: [overrides] missing or empty"]
    out = [
        f"{rel}: override {k!r} has unsupported value type {type(v).__name__}"
        for k, v in overrides.items()
        if not isinstance(v, bool | int | float | str | list)
    ]
    if raw.get("data", {}).get("source") not in ("mix", "glob"):
        out.append(f"{rel}: [data].source must be 'mix' or 'glob'")
    return out


def _check_moe(rel: str, raw: dict[str, Any]) -> list[str]:
    out: list[str] = []
    flags = raw.get("flags")
    if not isinstance(flags, dict):
        return [f"{rel}: missing [flags.*] tables"]
    seen: set[str] = set()
    for group, table in flags.items():
        for flag, value in table.items():
            if not flag.startswith("--"):
                out.append(f"{rel}: [flags.{group}] key {flag!r} does not start with --")
            if not _is_flag_value(value):
                out.append(f"{rel}: {flag} has unsupported value type {type(value).__name__}")
            if flag in seen:
                out.append(f"{rel}: {flag} appears twice")
            seen.add(flag)
    ep = flags.get("parallelism", {}).get("--expert-model-parallel-size")
    if not isinstance(ep, int) or ep < 1:
        out.append(f"{rel}: --expert-model-parallel-size must be a positive integer")
    if not isinstance(raw.get("launch", {}).get("gpus_per_node"), int):
        out.append(f"{rel}: [launch].gpus_per_node must be an integer")
    return out


def check_manifests(root: Path) -> list[str]:
    path = root / "manifests" / "datasets.toml"
    rel = "manifests/datasets.toml"
    try:
        raw = load_toml(path)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return [f"{rel}: {exc}"]
    out: list[str] = []
    for kind, required in (
        ("dataset", ("id", "url", "revision", "licence")),
        ("tokenizer", ("id", "url", "revision", "licence")),
        ("mix", ("name", "defined-in", "path")),
    ):
        entries = raw.get(kind, [])
        if not entries:
            out.append(f"{rel}: no [[{kind}]] entries")
        for entry in entries:
            ident = entry.get("id") or entry.get("name") or "?"
            out.extend(
                f"{rel}: [[{kind}]] {ident} missing {key!r}" for key in required if key not in entry
            )
            if "revision" in entry and not HEX40.match(entry["revision"]):
                out.append(f"{rel}: [[{kind}]] {ident} revision is not a 40-hex commit")
    return out


def check(root: Path) -> list[str]:
    """Parse every config and manifest and cross-check pins. Returns problems."""
    try:
        ups = upstreams.load(root)
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        return [f"configs/upstreams.toml: {exc}"]
    out = upstreams.problems(ups)
    out.extend(check_manifests(root))

    for phase in PHASE_DIRS:
        files = sorted((root / "configs" / phase).glob("*.toml"))
        if not files:
            out.append(f"configs/{phase}: no config files")
        for path in files:
            rel = f"configs/{phase}/{path.name}"
            try:
                raw = load_toml(path)
            except (OSError, tomllib.TOMLDecodeError) as exc:
                out.append(f"{rel}: {exc}")
                continue
            if phase == "phase1-dense" and path.name != "smoke-tiny.toml":
                out.extend(_check_meta(rel, raw, phase, ups))
                out.extend(_check_overrides(rel, raw))
            elif phase == "phase1-moe":
                out.extend(_check_meta(rel, raw, phase, ups))
                out.extend(_check_moe(rel, raw))
            elif phase == "phase2-post":
                out.extend(_check_meta(rel, raw, phase, ups))
                for stage in raw.get("stage", []):
                    if stage.get("upstream") not in ups:
                        out.append(f"{rel}: stage {stage.get('name')!r} names an unknown upstream")

    try:
        load_smoke_tiny(root)
        load_smoke_diloco(root)
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        out.append(f"smoke config: {exc!r}")
    return out
