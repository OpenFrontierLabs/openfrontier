# Joining a node

Short version: **single-node phase 1 contributions come first. Joining a DiLoCo island is phase 3 and is not built.** Nothing has been trained yet, so there is no run to join.

## What works today

| Command | What it does |
| --- | --- |
| `make setup` | create the project environment from `uv.lock` |
| `make doctor` | report GPU, CUDA driver versus torch build, DNS and HTTPS latency to huggingface.co and github.com, disk, clock, tool versions. Works with no GPU. |
| `make smoke` | CPU: a real tiny OLMo-core run, then the reference DiLoCo |
| `make dev` | a longer, streaming version of the first smoke |
| `uv run ofl run0 launch [--dry-run]` | the real OLMo-core command for a single node (phase 1 dense) |
| `uv run ofl moe launch [--dry-run]` | the real Megatron-LM command for a single 8-GPU node (phase 1 MoE) |
| `uv run ofl run0 verify`, `uv run ofl moe verify` | check the launchers against their pinned upstreams |
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

## Swarm proof of concept: two commands

A real but small multi-machine DiLoCo run exists as a **proof of concept**. It is not prime-diloco and not the planned mesh: one coordinator holds the global weights, nodes train a tiny char-level GPT (4.8M parameters) on tiny-shakespeare, and there is **no authentication, no ledger and no contribution scoring**. The network is Tailscale for now (anyone on the tailnet can join or push updates); a rust-libp2p transport comes later.

On the coordinator machine (binds its Tailscale IPv4, port 8470; state under `~/.ofl/swarm/`):

```sh
uv run ofl swarm coordinator --nodes 2 --rounds 30
```

On each node (detects `cuda`, then `mps`, then `cpu`):

```sh
uv run ofl node join --coordinator http://<coordinator-tailscale-ip>:8470
```

The project environment pins CPU-only torch on Linux, so a CUDA node uses its own environment with a CUDA torch build and `pip install --no-deps -e .` of this clone, then runs `ofl node join` from there. On macOS the project environment's torch already has MPS.

Each round every node pulls the global weights (checked by SHA-256), runs 50 inner AdamW steps on its own contiguous shard, and pushes its pseudo-gradient; the coordinator averages them in rank order, takes one SGD step with Nesterov momentum 0.9 (outer learning rate 0.7) and evaluates the held-out loss. Rounds are synchronous: one slow node slows the run, and a node that leaves stalls it. Nodes report the hash of the weights they actually loaded, so every round records whether all nodes held identical weights.

`uv run ofl swarm status` prints the registry (device, GPU, memory, torch, last heartbeat) and the per-round loss and hash agreement. A run directory `~/.ofl/swarm/runs/<run-id>/` holds `log.jsonl`, `receipt.json` (nodes, rounds, losses, wall time, final weights hash, every node's final hash, git SHA), `sample.txt` and `weights.f32`. The HTTP interface is described in [`src/ofl/swarm/openapi.yaml`](../src/ofl/swarm/openapi.yaml). `tests/test_swarm.py` runs the same protocol with two CPU nodes on localhost.

### Show the swarm on the website

`ofl swarm coordinator --report-url https://openfrontier-labs.com` reports nodes, the run and each round to the site, which draws them on [/dashboard](https://openfrontier-labs.com/dashboard) and, matched by GitHub handle, in the member console. The token is read from `OFL_SWARM_TOKEN` or `~/.ofl/token`. Reports go through a background queue: an unreachable site is printed and dropped, and never stalls training. `ofl node join --github <handle>` attaches a node to a member (default: your `gh` login). `ofl swarm report --run-dir ~/.ofl/swarm/runs/<run-id> --github <handle> --report-url …` sends a finished run from its receipt and log.

## Planned, not built

The plan describes this flow. **None of these commands exist.**

- `pipx install ofl`: `ofl` is not published to PyPI or anywhere else; use a clone.
- `ofl join run1`: sign in with a GitHub identity and join an elastic mesh that catches you up from a live checkpoint. Only the swarm proof of concept above exists: no identity, no elasticity, no catch-up for late joiners.
- Automatic dropping of nodes whose updates hurt the loss, and scoring that fills `helpful_update_share` in the [ledger](../ledger/README.md).
- Rollout nodes for distributed RL verified with TOPLOC.
