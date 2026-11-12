"""`ofl deploy`: publishing is not implemented because no run has produced anything to publish."""

from __future__ import annotations

from pathlib import Path

from ofl.repo import OflError, find_root

ENVS = {
    "dry-run": "print what would be published and publish nothing",
    "huggingface": "publish checkpoints and weights to Hugging Face (NOT IMPLEMENTED)",
}


def find_checkpoints(root: Path) -> list[Path]:
    """Checkpoint directories under checkpoints/ (written by the run0 and moe launchers)."""
    base = root / "checkpoints"
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("step*") if p.is_dir())


def deploy(env: str) -> int:
    if env not in ENVS:
        known = ", ".join(sorted(ENVS))
        raise OflError(f"unknown --env {env!r}; known: {known}", code=2)
    if env == "huggingface":
        raise OflError(
            "not implemented yet: nothing to publish until a run has produced checkpoints. "
            "No publishing code exists.",
            code=2,
        )
    root = find_root()
    found = find_checkpoints(root)
    print("deploy --env dry-run: nothing is published.")
    if not found:
        print(
            "would publish: nothing. No checkpoints under checkpoints/ (nothing has been trained)."
        )
        return 0
    print(f"would publish {len(found)} checkpoint director{'y' if len(found) == 1 else 'ies'}:")
    for path in found:
        print(f"  {path.relative_to(root)}")
    print("publishing itself is not implemented; --env huggingface exits non-zero.")
    return 0
