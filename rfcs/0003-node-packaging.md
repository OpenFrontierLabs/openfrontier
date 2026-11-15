# RFC 0003: The node is a separate Rust binary, `ofl-node`, not an extension of `ofl`

| Field | Value |
| --- | --- |
| **Status** | `proposed` |
| **Authors** | @bradlishman (founding steward); drafted by an agent (omp / claude-sonnet-5-5) |
| **Created** | 2026-11-15 |
| **Layers** | 2 Node, 3 Network, 4 Training |
| **Discussion** | [Issue #16](https://github.com/OpenFrontierLabs/openfrontier/issues/16) (opened as "RFC 0002"; that number went to CRAM) |
| **Tracking issue** | added once accepted |

## Summary

The Open Frontier Node should be the existing `ofl-node` crate grown into a daemon: a separate Rust binary that owns the device key, the rust-libp2p peer (layer 3) and every network connection, and that runs the Python trainer as a child process with no network, no key and a scratch directory only. `ofl` stays the repo developer CLI. This is the answer to the open milestone in [ARCHITECTURE.md §2](../docs/ARCHITECTURE.md#2-open-frontier-node) ("decide whether the node grows out of `ofl` or is a separate binary"). Nothing described below as proposed is built; today `node/` signs a hardware report and runs no workers.

## Motivation

[RFC 0001](0001-milestone-ladder-and-first-diloco-proof-run.md) needs a rust-libp2p peer with a persisted device key and a Mac among the workers (Rung 1), and later volunteers on three continents (Rung 2). Three pieces exist, and none is the node:

- `ofl` (`src/cli.rs`) is a developer CLI. `find_root` in `src/repo.rs` refuses to run outside a clone (it searches upward for `configs/upstreams.toml`), `smoke_python` hard-codes the Unix venv path `smoke/.venv/bin/python`, and default builds can carry opt-out Sentry reporting ([telemetry.md](../docs/telemetry.md)).
- `node/` (`ofl-node`, [node/README.md](../node/README.md)) is the device identity: an Ed25519 key kept on disk (`<config dir>/ofl-node/device.key`) and a signed hardware report. Its key type is `libp2p-identity`'s, so the same key becomes the libp2p peer id.
- The swarm proof of concept (`ofl node join`, `swarm::forward` in `src/swarm.rs`; [join-a-node.md](../docs/join-a-node.md#swarm-proof-of-concept-two-commands)) is Python over HTTP and Tailscale, which `ofl` forwards to `smoke.swarm`. It has no identity and no libp2p.

Layer 3 and the donor install story depend on the packaging choice, so the choice has to come before the libp2p work. Affected: whoever builds layers 2 and 3, and every donor who will run the node on a personal machine.

## Proposal

All of this is **proposed**.

### Decision

1. **Separate binary.** `ofl-node` is the node. It grows from `node/`; `ofl` gains no node logic.
2. **The node owns the network and the key; the trainer owns the tensors.** The node is the libp2p peer and signs with the device key. Trainers stay Python (OLMo-core, the reference DiLoCo, later prime-diloco) and run as child processes. A trainer never opens a socket and never sees the key.
3. **One join entry point.** `ofl-node join` is the donor command. It supersedes the planned `ofl join` in [ARCHITECTURE.md](../docs/ARCHITECTURE.md) and [join-a-node.md](../docs/join-a-node.md#planned-not-built), and the existing `ofl node` subcommand (`Command::Node` in `src/cli.rs`, which forwards to the Python swarm proof of concept) is retired when the libp2p run replaces that proof of concept, not renamed or aliased. Until then the proof of concept keeps `ofl node`, and the hyphenated `ofl-node` is a different binary.
4. **The name and the key path are fixed now.** `ofl-node` and `<config dir>/ofl-node/device.key` stay as they are. Changing either strands the donors' identities (node/README: moving the key aside loses the old peer id).

### The trainer boundary

- The node spawns the trainer with an explicit argv and an allow-listed environment, a per-run scratch directory as working directory, and kills it when the node exits or a user limit trips.
- Control is newline-delimited JSON on the child's stdin and stdout (progress, "pseudo-gradient written", "stop"). Tensors move as files in the scratch directory, never as stdio. Both work the same on Linux, macOS and Windows.
- Inbound data is checked by the node before the trainer sees it: size and dtype against the run's manifest, a hash, and a signature that matches a known peer id. Only safetensors reach the trainer; pickle never does. Petals states the same rule for its network: only tensors cross between peers, serialised with protobuf and never pickle ([Petals security FAQ](https://github.com/bigscience-workshop/petals/wiki/Security,-privacy,-and-AI-safety)).
- The trainer code is repo code (such as the reference DiLoCo in `smoke/smoke/diloco_ref.py`) or a pinned upstream commit from `configs/upstreams.toml`, never code received from a peer. RFC 0001 (Risks) states the same for Rung 1: participants run code from this repo only, never code received over the network.
- Averaging stays on the Python side by default: the node ferries pseudo-gradient files and hands the received ones to the trainer, so the node needs no tensor library. The wire protocol and where the reduction runs belong to the layer 3 RFC; this RFC only fixes that the trainer sees files.

### Sandboxing, per OS

The boundary above is what the node *asks for*. Whether the OS *enforces* it differs, and the node must say which case applies when it starts (`enforced`, `limits-only` or `none`) so a donor knows what protects them. What the cited sources support:

| OS | Limits and kill-with-node | Files and network confinement |
| --- | --- | --- |
| Linux | Not verified here. | bubblewrap builds a sandbox from unprivileged user namespaces, but "is not a complete, ready-made sandbox"; the protection is whatever arguments the caller passes ([README](https://github.com/containers/bubblewrap)). The node would own those arguments (proposed): read-only system, scratch read-write, no network. |
| Windows | Job objects group processes and enforce limits (working set, priority, time), account for them, and `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` ends every process when the last handle closes ([Microsoft](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects)). | Not covered by that page. Open question. |
| macOS | Not verified here. | Not verified here. Open question. |

On Windows and macOS the first release is therefore `limits-only` or `none`, and the docs say so. That is acceptable only while participants run repo-pinned code (RFC 0001, Rung 1). Before Rung 2 accepts volunteers it needs a closed answer on every OS, or an explicit rule that a platform without enforcement is not offered public runs.

### Donor install story

Target (proposed): one binary per OS and one command to join.

1. Get `ofl-node` for your OS from a release page, with a published checksum. No release exists yet ([telemetry.md](../docs/telemetry.md) says so); who signs release binaries is an open question.
2. `ofl-node join <run>` checks the machine (GPU, disk, clock, as `ofl doctor` does), creates the trainer's Python environment on first use from a lock file pinned by hash, and starts. The repo already creates such an environment with `uv sync --locked` (`ensure_smoke_env` in `src/tasks.rs`); the donor flow reuses that approach without a clone.

Prior art, for calibration. Petals' README shows three commands on Linux (install PyTorch with conda, `pip install git+...`, `python -m petals.cli.run_server`) and three on macOS (`brew install python`, `pip install git+...`, run the module); Windows goes through a WSL guide and a Docker image is offered ([README](https://github.com/bigscience-workshop/petals)). That is what "one command" looks like in practice: a few commands, and Python is the node. Psyche goes the other way: its client runs in Docker, needs Linux, an NVIDIA GPU, Docker and the NVIDIA Container Toolkit, and a distributed `run-manager` binary pulls the image and keeps it updated ([join guide](https://github.com/PsycheFoundation/psyche/blob/main/psyche-book/src/enduser/join-run.md)). Our target is tighter than either only if the binary really provisions the trainer environment itself, which is the least certain part of this RFC.

### Platforms

- **Linux, NVIDIA or CPU:** first-class.
- **macOS, Apple Silicon:** required by Rung 1. The trainer runs natively so it reaches MPS. rust-libp2p's CI compile-checks `x86_64-apple-darwin` and `x86_64-pc-windows-msvc` ([ci.yml](https://github.com/libp2p/rust-libp2p/blob/master/.github/workflows/ci.yml), `cross` job); that is a compile check, and it does not list `aarch64-apple-darwin`. The repo records no Mac or Windows build of `ofl-node` yet, so the first implementation PR builds the node on a real Apple Silicon Mac.
- **Windows:** the node should build and the key file already has a Windows branch. The trainer path is not ready: `smoke_python` assumes `bin/python`. Support is a later step, and WSL is a fallback.

### Layer 3 fit

`ofl-node` is the libp2p peer. The device key is the peer id, so layer 2 identity and layer 3 identity are one thing and every protocol message is signed by the key the ledger will name. The libp2p behaviour starts as a module of the node crate; if a second binary needs it (the bootstrap and relay peer on a VPS, [ARCHITECTURE.md §7](../docs/ARCHITECTURE.md#7-hosting)), it moves to its own workspace crate. That is a refactor, not a decision.

### What is reversible

| Choice | Reversible? |
| --- | --- |
| Separate binary versus `ofl` subcommand | Mostly. A thin `ofl` forwarder can be added later without touching the node. |
| Binary name and key path | **No** once donors have keys: changing either changes their peer ids. Fixed above. |
| Trainer boundary (stdio JSON plus files) | Versioned from its first message. Old nodes in the field still speak v1, so a break is a new version beside the old, not an edit. |
| Per-OS sandbox backends | Yes. Each is internal and replaceable. |
| Libp2p as a module versus its own crate | Yes. |
| Release channel names (package names, installer URLs) | Hard to take back once published; choose once, in the release PR. |

## Alternatives

| Option | Pros | Cons |
| --- | --- | --- |
| **This proposal:** separate Rust binary, Python trainer as subprocess | One small binary per OS; the network and the key are in Rust, matching the layer 3 plan; the trainer can be denied the network entirely; builds on `node/` as merged; no Sentry dependency in a donor daemon | The binary must provision Python environments itself; sandbox strength differs per OS; two languages in the node |
| (a) Grow `ofl` into the node | One binary and one lock file; reuses the CLI plumbing | `ofl` assumes a clone (`find_root`) and a Unix venv path; carries developer-only dependencies and optional Sentry reporting onto donor machines; a donor daemon and a dev tool then share one release cadence; the node crate already exists separately |
| (c1) Python-packaged node, Petals-style | Proven shape ([README](https://github.com/bigscience-workshop/petals)); trainer and node in one language; no subprocess boundary | `ofl` is not published to PyPI ([join-a-node.md](../docs/join-a-node.md#planned-not-built)); layer 3 is planned in Rust, so a Python node must bind a Rust peer into a wheel per OS and architecture or run a Rust sidecar anyway, which is the proposal with a Python supervisor; the key and the network live in the same process as arbitrary training dependencies |
| (c2) Container-only node, Psyche-style | Strong isolation; pinned environment; one artefact | Psyche's own guide needs Linux and an NVIDIA GPU ([join guide](https://github.com/PsycheFoundation/psyche/blob/main/psyche-book/src/enduser/join-run.md)); Docker Desktop's `--gpus` passthrough is documented for Windows with the WSL 2 backend only ([Docker docs](https://docs.docker.com/desktop/features/gpu/)), which leaves no verified path to Apple Silicon MPS, and Rung 1 needs a Mac; installing Docker is itself a step |
| Do nothing | No work | Layer 3 has no home; the PoC's HTTP and Tailscale shape would become the de facto node |

A container runner for Linux NVIDIA donors is compatible with the proposal as a second way to launch the trainer. It is not required for Rung 1 and this RFC does not commit to it.

## Risks and security

- **Uneven sandboxing.** On macOS and Windows the trainer can do whatever the donor's user can. Mitigation: the `enforced` / `limits-only` / `none` report, repo-pinned trainer code only, and the Rung 2 gate stated under the sandboxing table.
- **Supply chain of the trainer environment.** The node downloads Python wheels. Mitigation: a lock file pinned by hash, as `uv sync --locked` already does for the smokes. Hash pinning does not make a compromised upstream safe; the sandbox is the second layer.
- **Peer-supplied tensors.** A malformed file reaching the trainer's loader is an attack surface. Mitigation: the node validates size, dtype and hash first, and only safetensors cross the boundary.
- **Key exposure.** The key lives in the node process only; the trainer's environment is allow-listed and its scratch directory excludes the key directory.
- **Telemetry.** The node has no Sentry dependency today and must not gain a default-on one; a node telemetry policy is a separate decision.
- **Two-language maintenance.** Rust node and Python trainer drift apart. Mitigation: the boundary is a small versioned contract with a fake trainer in the node's tests, so the node's behaviour is tested without torch.

## Rollout

1. Merge this RFC as `proposed`. When it is accepted, the milestone in ARCHITECTURE §2 is rewritten to point here, and the `ofl join` mentions in ARCHITECTURE.md and join-a-node.md become `ofl-node join`.
2. PR: the trainer boundary in `ofl-node`: spawn with allow-listed environment and scratch directory, JSON control messages, kill-on-exit and limits. Tested with a fake trainer (CPU, no downloads, no GPU): the child sees no inherited secrets, cannot outlive the node, and a file that fails validation never reaches it.
3. PR: the libp2p peer, using the existing key (the layer 3 issues and RFC).
4. One PR per OS for a sandbox backend, each with a test that tries to escape it.
5. PR: first Apple Silicon build and `join` for a CPU or MPS trainer. This is where the Rung 1 pass criteria start being testable.
6. Release PR: prebuilt binaries, checksums and the install page. The Python swarm proof of concept and its `ofl node` forwarding are removed when the libp2p run replaces them, not aliased.

Undo: before step 6 nothing is shipped to donors, so abandoning the proposal costs the crate's new code only. After step 6 the reversible and irreversible items are the ones in the table above.

## Out of scope

The libp2p protocol and message formats, how the all-reduce is done, update verification and poisoning (RFC 0001 promises a separate RFC before volunteers join), the desktop app, GPU fields in the hardware report, node telemetry, the credit and ledger changes, and which package managers or installers to publish to.

## Open questions

- macOS: which mechanism confines a child process, and does it work for an unsigned binary a donor downloaded? Not verified in this RFC.
- Windows: how are file and network access confined (Job objects limit resources; the cited page does not cover isolation)? Not verified in this RFC.
- Who provisions Python: does `ofl-node` require `uv` on the donor's machine, or bring its own? And what is the Windows trainer path?
- Linux: bubblewrap as an external tool the donor must have, or a built-in confinement with no external dependency?
- Is a Docker runner for Linux NVIDIA donors worth a first-class place in Rung 2, given Psyche's choice and the `--gpus` limit on Docker Desktop?
- Who holds the release signing key, and where? A governance question, not a technical one.

## References

- Petals [README](https://github.com/bigscience-workshop/petals) (install commands per platform, Docker image) and [security FAQ](https://github.com/bigscience-workshop/petals/wiki/Security,-privacy,-and-AI-safety) (tensors only, no pickle, Docker or VM isolation advised, libp2p over TCP and QUIC).
- Psyche [repository](https://github.com/PsycheFoundation/psyche) and its [join guide](https://github.com/PsycheFoundation/psyche/blob/main/psyche-book/src/enduser/join-run.md) (Linux, NVIDIA, Docker, `run-manager`). The source is cited from the repository rather than the hosted docs site.
- rust-libp2p [CI workflow](https://github.com/libp2p/rust-libp2p/blob/master/.github/workflows/ci.yml) (`cross` job targets).
- bubblewrap [README](https://github.com/containers/bubblewrap); Microsoft [Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects); Docker Desktop [GPU support](https://docs.docker.com/desktop/features/gpu/).
- This repo: [RFC 0001](0001-milestone-ladder-and-first-diloco-proof-run.md), [ARCHITECTURE.md](../docs/ARCHITECTURE.md), [node/README.md](../node/README.md), [join-a-node.md](../docs/join-a-node.md), [telemetry.md](../docs/telemetry.md).
