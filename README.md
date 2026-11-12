# Open Frontier Labs

**Open infrastructure for collective frontier intelligence: fully open models trained on community compute.**

Open Frontier Labs is what OpenAI should have been, with a pure community vibe: built in public, meant to be owned by a nonprofit, and trained on compute the community pools. No equity, no token, no closed tier.

> **Status, plainly: nothing has been trained yet.** This repository is scaffolding for the first runs. The CPU smoke tests run; the 1B baseline (Run 0) has a launcher, a config and pinned upstreams, and has not been started on any GPU. There are no results, no released weights, and no public run to join. The foundation described in [GOVERNANCE.md](GOVERNANCE.md) has not been formed.

## What is open

Everything we produce is published, as it happens:

- **Data recipes and manifests**: dataset ids, licences and revisions in [`manifests/`](manifests/).
- **Code and configs**: this repo; every upstream we build on is pinned to an exact commit in [`configs/upstreams.toml`](configs/upstreams.toml), and nothing is vendored.
- **Logs**, including failed and diverged runs.
- **Every checkpoint**, not just the last one.
- **The weights**, Apache-2.0. Datasets and logs we author are intended to be CC-BY-4.0; data we reuse keeps its upstream terms. No use-restriction licences.

Contributors and donors are credited by name in a public [compute ledger](ledger/README.md) and in [CREDITS.md](CREDITS.md).

## Roadmap

| Phase | What | Status |
| --- | --- | --- |
| **1. Single-node, fully open pipeline** | **Run 0**: a dense ~1B baseline with the official [OLMo-core](https://github.com/allenai/OLMo-core) trainer and the OLMo recipes, on one GPU node. In parallel, an **MoE track** (about 6.9B total / 1.2B active, 64 experts, top-8) following Hugging Face's *Ultimate MoE Training Handbook* and Megatron-LM. | Launchers, configs and manifests exist; nothing has run on a GPU. [Run 0](docs/run0.md), [MoE](docs/phase1-moe.md) |
| **2. Post-training** | SFT, DPO and RLVR with [open-instruct](https://github.com/allenai/open-instruct); async RL with [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) and [verifiers](https://github.com/PrimeIntellect-ai/verifiers). | Pinned and documented; launcher not built. [Details](docs/phase2-post.md) |
| **3. DiLoCo across islands** | DiLoCo outer loops ([prime-diloco](https://github.com/PrimeIntellect-ai/prime-diloco)) over single-node islands: community nodes and donated clusters. | Config and algorithm smoke only; launcher not built. [Details](docs/phase3-diloco.md) |

Later, once phase 3 exists, and not scheduled: a mid-scale community model of about 10B (Run 1), distributed RL over a swarm of rollout nodes, and a 30 to 70B model (Run 2) on donated clusters plus community nodes. We aim for best-in-class at each size, not frontier parity.

## Try it

```bash
git clone https://github.com/OpenFrontierLabs/openfrontier && cd openfrontier
make setup      # uv sync from uv.lock; needs uv, installs nothing outside the repo
make doctor     # GPU, CUDA, network latency, disk, clock, versions; works with no GPU
make smoke      # CPU, no downloads, about half a minute here: tiny OLMo-core run, then reference DiLoCo
make help       # every target
uv run ofl run0 launch --dry-run    # print the real OLMo-core command for Run 0
```

`make check` and `make test` run lint, types, schema and config checks, and the unit tests. All targets are one-line wrappers around the `ofl` CLI in `src/ofl/`.

What `make smoke` does and does not prove: it runs the pinned OLMo-core trainer on CPU with a 1.1M-parameter model and checks that the loss falls and a checkpoint is written; and it runs a **reference implementation** of DiLoCo (not prime-diloco, which needs CUDA) on two CPU workers and checks that the loss falls and the workers agree after each outer sync. It does not exercise the 1B configuration, real data or a GPU.

## Contribute

Code, compute and results are all welcome. [CONTRIBUTING.md](CONTRIBUTING.md) says how; [docs/join-a-node.md](docs/join-a-node.md) says plainly what works today (single-node phase 1 runs) and what is planned but not built (joining a shared DiLoCo run).

## Links

- Organisation: <https://github.com/OpenFrontierLabs>
- [Manifesto](MANIFESTO.md), [Governance](GOVERNANCE.md), [Credits](CREDITS.md), [Code of Conduct](CODE_OF_CONDUCT.md)
- Founding steward: [@bradlishman](https://github.com/bradlishman)

## Licence

Apache-2.0, see [LICENSE](LICENSE). Brand: Frontier Green `#66E36E`, Compute Green `#1B7F46`, Graphite `#0B0E0D`, Silver Gray `#6B7177`, Soft White `#F5F7F6`; Inter; dark first.
