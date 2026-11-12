"""Launchers for phases that are configured but not built. They fail loudly, on purpose."""

from __future__ import annotations

from ofl.repo import OflError

NOT_BUILT = {
    "post": (
        "phase 2 (post-training with open-instruct, prime-rl and verifiers) has no launcher. "
        "configs/phase2-post/post-training.toml names the upstream entry points at the pinned "
        "commits; no flags have been written or checked, and there is no base model to post-train."
    ),
    "diloco": (
        "phase 3 (DiLoCo outer loop across islands with prime-diloco) has no launcher. "
        "configs/phase3-diloco/diloco-1b.toml is a config for the pinned upstream; no island "
        "setup exists and nothing has been trained. `ofl smoke` checks the DiLoCo algorithm only."
    ),
}


def not_built(phase: str) -> int:
    raise OflError(f"not built: {NOT_BUILT[phase]}", code=2)
