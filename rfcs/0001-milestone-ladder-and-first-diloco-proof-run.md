# RFC 0001: Milestone ladder and first DiLoCo proof run

| Field | Value |
| --- | --- |
| **Status** | `proposed` |
| **Authors** | @bradlishman (founding steward); drafted by an agent (omp / claude-opus-5-5) from the community plan |
| **Created** | 2026-11-12 |
| **Layers** | 2 Node, 3 Network, 4 Training, 5 Storage, 8 Outputs |
| **Discussion** | the pull request that adds this file |
| **Tracking issue** | one tracking issue per rung, on the [roadmap](https://github.com/orgs/OpenFrontierLabs/projects/1) |

## Summary

Open Frontier Labs gets from "nothing trained yet" to a GPT-3-class open model through six public rungs. Each rung ends in something anyone can inspect: config, logs, checkpoints and a short report, including failures. The first rung that proves the idea is **Rung 1, Hello, swarm**: two or three machines, at least one of them a Mac, train one model with DiLoCo over a rust-libp2p link, and the loss falls. This RFC fixes the ladder, what "done" means at each rung, and the exact scope and pass criteria of the Rung 1 proof run.

## Motivation

The [README](../README.md) states plainly that nothing has been trained, and the [architecture doc](../docs/ARCHITECTURE.md) lists layer by layer what is missing. Contributors and compute donors need to know what they are working towards and when a step counts as finished. Without a written ladder, "progress" turns into announcements instead of artifacts.

Prior art shows the pattern works when each step is public and modest. Prime Intellect reproduced DiLoCo at 150M parameters ([OpenDiLoCo](https://arxiv.org/abs/2407.07852)), then scaled to 1.1B, then trained the 10B [INTELLECT-1](https://arxiv.org/abs/2412.01152) across five countries and released every checkpoint. BigScience used named working groups and released intermediate checkpoints. AI2's OLMo shows that releasing the whole flow (data, code, logs, checkpoints, evals) is what draws researchers.

The ladder is also the evidence trail a compute grant needs: a 256-H100-class application is credible after public reports from the smaller rungs, not before.

## Proposal

### The ladder

| Rung | What happens | Done when |
| --- | --- | --- |
| 0. Run 0 | Dense ~1B baseline on one GPU node with OLMo-core, as already configured in [`docs/run0.md`](../docs/run0.md). | Config, logs, every checkpoint and an honest report are published, failures included. |
| 1. Hello, swarm | Two or three machines, including a Mac, exchange DiLoCo updates over a rust-libp2p link. | Loss falls, workers agree after each sync, and the log is public. |
| 2. First community run | A model of about 150M parameters, trained by 8 or more volunteer nodes on at least three continents. | A run page with a node map, a credits file and a short technical report. |
| 3. About 1B across the swarm | The same recipe at 1B, compared against Run 0. | Swarm result published next to Run 0 with the same evals. |
| 4. About 10B (Run 1) | Mode A async DiLoCo plus donated clusters as islands. | Weights, checkpoints, data manifests and report released. |
| 5. 30B to 70B, then GPT-3 class | Donated clusters plus community nodes, helped by a grant backed by the earlier reports. | Open weights released with a model card crediting every contributor. |

Rungs 0 and 1 are independent and can proceed in parallel. Rung 2 needs Rung 1. Rung 3 needs Rungs 0 and 2. The target at each size is best-in-class, not frontier parity.

### What every rung publishes

A rung is done only when all of these are public:

1. the exact config and every upstream pin (`configs/`, `configs/upstreams.toml`);
2. dataset manifests with source, licence, revision and hash (`manifests/`);
3. the full training log, including restarts, divergences and failed attempts;
4. every checkpoint the run kept, not just the last one (Rung 1 is exempt: its toy model is not worth hosting);
5. a ledger file crediting every node and donor (`ledger/`, validated by `make check`);
6. a short report: what we tried, what happened, what we would change. Negative results are reports too.

Each rung has one tracking issue labelled `type/tracking` on the roadmap project. Its body lists the sub-issues and links each artifact when it exists.

### Rung 1 in detail: the first DiLoCo proof run

**Goal.** Show, with public evidence, that our stack can train one model across separately owned machines over the internet, using DiLoCo with a rust-libp2p transport. It does not show quality or scale.

**Setup.**

- Two or three workers on different networks; at least one Apple Silicon Mac (MPS) and at least one NVIDIA or CPU Linux machine.
- The model and training loop are the existing reference DiLoCo in [`smoke/smoke/diloco_ref.py`](../smoke/smoke/diloco_ref.py) scaled to a small transformer (order of 10M to 50M parameters) on a public text shard listed in `manifests/datasets.toml`.
- Inner optimiser AdamW; outer optimiser SGD with Nesterov momentum 0.9, as in the smoke test. Inner steps per round, outer learning rate and round count are chosen in the implementing PR and recorded in the config.
- Transport: pseudo-gradients cross the network through a minimal rust-libp2p peer (QUIC, TCP fallback) that replaces gloo for the all-reduce, one bootstrap peer, and a relay if a worker is behind NAT. No central parameter server; one peer may act as bootstrap only.
- Each worker signs its pseudo-gradient messages with a persistent device key (layer 2), and peers reject unsigned or wrongly signed messages.

**Pass criteria.** All of:

1. Held-out loss after the final round is below half its starting value, as in the smoke test.
2. After every outer sync, all workers hold bit-identical weights (checked by hash and logged).
3. Workers' weights genuinely diverge during inner steps (otherwise the run is not testing DiLoCo).
4. A negative control with outer learning rate 0 fails criterion 1.
5. The run survives one worker leaving and rejoining mid-run, and the log shows it.
6. The log, config, peer list (handles and coarse location only) and ledger file are published, and anyone can re-run it from the repo with documented commands.

**Not in Rung 1.** Verification of untrusted updates, contribution scoring (`helpful_update_share`), compression, Mode B, the desktop node app, and any public "join the swarm" button. Participants are known contributors.

### What this RFC does not change

Phases 1 to 3 in the README stay as the technical description of the work. The ladder is how that work is sequenced and shown in public. The README roadmap table is updated to link the ladder once this RFC is accepted.

## Alternatives

| Option | Pros | Cons |
| --- | --- | --- |
| This ladder, with Hello, swarm as the first proof | Small, visible, fast to a public artifact; matches what worked for OpenDiLoCo and INTELLECT-1 | The Rung 1 model is a toy; some will read it as not serious |
| Start at 150M community run (Rung 2) | A more impressive first result | Debugging transport, signing and training all at once with volunteers; a failure is public and confusing |
| Use prime-diloco's own transport for Rung 1 | Less new code | It needs CUDA (no Macs) and does not use libp2p, so it proves nothing about our network layer ([phase3-diloco.md](../docs/phase3-diloco.md)) |
| Single-node only until a grant arrives | Least engineering | Contradicts the mission; gives grant reviewers nothing distributed to judge |

## Risks and security

- **Overclaiming.** A toy run can be dressed up as more than it is. Mitigation: the report uses the pass criteria above and the "does not show" list verbatim.
- **Contributor machines.** Rung 1 participants run code from this repo only, never code received over the network. Messages carry tensors and signatures, not instructions.
- **Location privacy.** The published peer list carries handles and continent or country only, with consent.
- **Untrusted peers.** Out of scope for Rung 1 because participants are known. Before Rung 2 accepts volunteers, a separate RFC covers update validation, poisoning and free riders.

## Rollout

1. Merge this RFC as `accepted`; open one tracking issue per rung on the roadmap.
2. Rung 0 and Rung 1 sub-issues come from the [good first issues](https://github.com/OpenFrontierLabs/openfrontier/issues?q=is%3Aissue%20label%3A%22good%20first%20issue%22) already open (device keypair, libp2p peer, two-peer CI test, doctor bandwidth, hardware matrix).
3. Rung 1 lands as a sequence of small PRs; the run itself is a PR that adds its config, log and ledger file.
4. Changing a rung's "done when" later needs a new RFC that supersedes this one for that rung.

## Out of scope

Model architecture choices beyond Run 0's existing config, the token budget for Rungs 3 to 5, which grant to apply for, and the governance of donated clusters.

## Open questions

- Should Rung 1 use the existing reference DiLoCo or the prime-diloco algorithm code with a swapped transport? The proposal picks the reference for Mac support; implementers may argue otherwise in the implementing PR.
- Does Rung 2 require the desktop node app, or is a CLI node enough?

## References

- Douillard et al., [DiLoCo: Distributed Low-Communication Training of Language Models](https://arxiv.org/abs/2311.08105), 2023.
- Jaghouar, Ong, Hagemann, [OpenDiLoCo](https://arxiv.org/abs/2407.07852), 2024.
- Prime Intellect, [INTELLECT-1 technical report](https://arxiv.org/abs/2412.01152), 2024.
- AI2, [OLMo](https://allenai.org/olmo).
- This repo: [ARCHITECTURE.md](../docs/ARCHITECTURE.md), [run0.md](../docs/run0.md), [phase3-diloco.md](../docs/phase3-diloco.md), [ledger/README.md](../ledger/README.md).
