# Joining a node

Short version: **single-node phase 1 contributions come first. Joining a DiLoCo island is phase 3 and is not built.** Nothing has been trained yet, so there is no run to join.

## What works today

| Command | What it does |
| --- | --- |
| `make setup` | create the project environment from `uv.lock` |
| `make doctor` | report GPU, CUDA driver versus torch build, DNS and HTTPS latency to huggingface.co and github.com, disk, clock, tool versions. Works with no GPU. |
| `make smoke` | CPU: a real tiny OLMo-core run, then the reference DiLoCo |
| `make dev` | a longer, streaming version of the first smoke |
| `cargo run -- run0 launch [--dry-run]` | the real OLMo-core command for a single node (phase 1 dense) |
| `cargo run -- moe launch [--dry-run]` | the real Megatron-LM command for a single 8-GPU node (phase 1 MoE) |
| `cargo run -- run0 verify`, `cargo run -- moe verify` | check the launchers against their pinned upstreams |
| `make deploy ENV=dry-run` | list what would be published; publishing is not implemented |

`ofl doctor` exit status: 0 unless a hard failure. Hard failures are Python older than 3.11, torch not importable, less than 2 GiB free next to the repo, and github.com or huggingface.co unreachable. Everything else (no GPU, CPU-only torch, low disk, clock skew over 30 s) is a warning or information. It measures round-trip latency only; **it does not measure bandwidth**, and it does not tell you which run you qualify for, because no run defines thresholds yet.

## Hardware, as the plan states it

These are floors that *other* projects proved, quoted from the bootstrap plan. Open Frontier Labs has set no requirements of its own yet.

- INTELLECT-1 used nodes of 8 H100-class GPUs.
- Covenant-72B required 8 B200 GPUs or equivalent per peer.
- Pluralis asks for 200 Mbit/s and 80 ms round trip on 24 GB cards.
- The plan's own reading: consumer GPUs cannot join large DiLoCo pretraining; RL rollouts and evals can take any GPU; pipeline-parallel consumer training waits for Pluralis' recipe and weights to be public.

For the phase 1 MoE track the handbook's setup is one node with 8 GPUs of about 80 GB ([phase1-moe.md](phase1-moe.md)). For the dense baseline we have not measured what fits; do not take any GPU count as a requirement.

## What contributing looks like now

1. `make setup && make doctor && make smoke`, to see the toolchain work.
2. Run a phase 1 launcher on a GPU node ([run0.md](run0.md), [phase1-moe.md](phase1-moe.md)), and report what happened as an issue or pull request, including failures and the config you used.
3. Offer compute or a cluster: open an issue.

## Planned, not built

The plan describes this flow. **None of these commands exist.**

- `pipx install ofl`: `ofl` is not published to PyPI or anywhere else; use a clone.
- `ofl join run1`: sign in with a GitHub identity and join an elastic mesh that catches you up from a live checkpoint. There is no mesh, no coordinator and no node agent.
- Automatic dropping of nodes whose updates hurt the loss, and scoring that fills `helpful_update_share` in the [ledger](../ledger/README.md).
- Rollout nodes for distributed RL verified with TOPLOC.
