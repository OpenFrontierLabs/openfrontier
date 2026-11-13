# Contributing

Open Frontier Labs is built in public. You do not need permission to start: read, run, open an issue, send a pull request.

Nothing has been trained yet. The repository is scaffolding for the first runs, so the most useful contributions right now are the ones that make those runs real.

Where to talk: [Discord](https://discord.gg/M4sD8T5M8D) for chat, [Discussions](https://github.com/OpenFrontierLabs/openfrontier/discussions) for questions, ideas and RFCs, issues for concrete work. The [roadmap](https://github.com/orgs/OpenFrontierLabs/projects/1) shows what is being worked on.

## Your first contribution

1. Pick an open [good first issue](https://github.com/OpenFrontierLabs/openfrontier/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22). Comment on it to say you are taking it; a maintainer assigns it to you. If nobody has replied within a day, ask in Discord.
2. Fork, branch, and make the change. Run `make check` and `make test` before you push.
3. Open a draft pull request early, even after an hour of work. Questions on a draft are welcome.
4. Every commit is signed off (see [DCO](#sign-your-work-dco)).

New to the vocabulary (DiLoCo, island, pseudo-gradient)? The [glossary](docs/GLOSSARY.md) explains the terms used across the docs in plain words.

## Contribute code

```bash
git clone https://github.com/OpenFrontierLabs/openfrontier
cd openfrontier
make setup     # uv sync from uv.lock; needs uv (https://docs.astral.sh/uv/)
make check     # lint, format, types, ledger schema, configs and pins
make test
make smoke     # CPU only: a real tiny OLMo-core run, then reference DiLoCo
make help      # every target
```

Everything is behind `make` targets that forward to `ofl`, the Python CLI in `src/ofl/`. Behaviour lives in `ofl`, not in the Makefile.

Good first areas:

- **Phase 1 runs.** Run the dense baseline or the MoE track on a single GPU node and report what happened, including failures. See [docs/run0.md](docs/run0.md) and [docs/phase1-moe.md](docs/phase1-moe.md).
- **Data.** A tokenised FineWeb-Edu shard set for the dense baseline; manifests that are correct and complete. See [manifests/](manifests/).
- **Phase 2 and 3 launchers.** They are configured but not built. Any flag you add must exist in the pinned upstream (`configs/upstreams.toml`).
- **Bugs and honesty.** If a doc claims something the code does not do, that is a bug.

Rules of the road:

- Every upstream is pinned to an exact commit. Bump a pin by changing `ref` and `sha` together and re-running `ofl run0 verify` / `ofl moe verify`.
- Do not put a flag or config key in a launcher unless it exists in the pinned upstream.
- Do not vendor upstream code.
- Do not add a command that pretends to work. If it is not built, make it fail with a message that says so.
- Keep changes small and say what you ran. Tests that only check wiring are not wanted; tests that catch real bugs are.

Merged work earns commit rights on that repository ([GOVERNANCE.md](GOVERNANCE.md)).

## Pull requests

- One logical change per PR. Small PRs are reviewed faster.
- The title says what changes, in the imperative: `Report MPS devices in ofl doctor`.
- The description says why, what you ran (`make check`, `make test`, `make smoke`, a GPU run), and links the issue (`Closes #12`).
- Run `make check`, `make test` and, for training code, `make smoke` before asking for review; say in the PR what you ran.
- PRs are squash-merged, so the PR title becomes the commit message on `main`.
- Changes that are hard to reverse (protocol messages, ledger schema, licences, layer boundaries) need an [RFC](rfcs/README.md) first.
- [CODEOWNERS](.github/CODEOWNERS) requests review from the maintainer of each layer you touch. Until a layer has its own maintainer, the founding steward reviews.

## Sign your work (DCO)

We use the [Developer Certificate of Origin](https://developercertificate.org/) (DCO), not a CLA. By adding a `Signed-off-by` line to a commit you certify that you wrote the change or otherwise have the right to submit it under the project's licence (Apache-2.0). Inbound licence equals outbound licence; you keep your copyright.

Sign off with `-s`:

```bash
git commit -s -m "Report MPS devices in ofl doctor"
```

which adds a trailer matching your git name and email:

```text
Signed-off-by: Your Name <you@example.com>
```

Maintainers check sign-off before merging. To fix a branch that lacks it:

```bash
git rebase --signoff main
git push --force-with-lease
```

Use your real name or a consistent public identity. Anonymous contributions cannot be accepted.

## What reviews a pull request

Reviews are automatic and free for public repositories. Fix what they flag, or say in the PR why not.

| Check | What it does | Required to merge |
| --- | --- | --- |
| DCO | Fails if a commit lacks a matching `Signed-off-by` ([DCO](#sign-your-work-dco)). | Yes |
| CodeRabbit | AI review and a summary of the change, configured in [`.coderabbit.yaml`](.coderabbit.yaml). Comment `@coderabbitai review` to re-run it. | No |
| CodeQL | Code scanning for security bugs (GitHub default setup). | No |
| SonarQube Cloud | Static analysis for bugs, vulnerabilities and code smells (automatic analysis, no CI). | No |

A maintainer reads every PR. The AI reviewers can be wrong; they do not replace that.

## Bot commands

A small script ([`tools/ofl-bot/`](tools/ofl-bot/)) runs every 10 minutes. It uses no model and is idempotent.

- `/assign` on an issue assigns you; `/unassign` removes you. Maintainers can add `@handle`.
- New issues get `needs-triage`; PRs get `layer/*` and `area/*` labels from the paths they change.
- A first issue or PR gets a welcome comment, and a failed DCO check gets a hint on how to fix it.
- Issues and PRs idle for 45 days get `stale` and close after 14 more days without a reply. `type/rfc`, `type/tracking`, `good first issue`, `roadmap` and `security` are exempt.
- `@all-contributors please add @handle for <contribution types>` adds someone to the README contributors list.

## Labels

| Label | Meaning |
| --- | --- |
| `layer/1-hardware` to `layer/8-outputs` | Which layer of the [architecture](docs/ARCHITECTURE.md) the work is in. |
| `area/rust`, `area/python`, `area/ml`, `area/data`, `area/evals`, `area/docs`, `area/web` | Skills the work needs. |
| `type/bug`, `type/feature`, `type/docs`, `type/rfc`, `type/tracking`, `type/compute-offer`, `type/chore` | What kind of change it is. `type/tracking` issues are milestones on the roadmap. |
| `good first issue` | Scoped for a first contribution: context, definition of done and pointers in the issue. |
| `help wanted` | Maintainers would welcome help; may need more context. |
| `priority/high`, `priority/medium`, `priority/low` | High blocks the next milestone. |
| `needs-triage` | New and not yet reviewed by a maintainer. |
| `agent-authored` | Written by an AI agent (see below). |

## AI agents

Agent-written contributions are welcome. A human is accountable for every one of them: the person who opens the PR signs off the commits and answers review.

Label agent work so it can be filtered:

- add the line `Agent: <agent> / <model>` to the commit message and the PR description, for example `Agent: omp / claude-opus-5-5`;
- add the `agent-authored` label to the PR or issue.

Agents follow the same rules as everyone else: no invented results, no commands that pretend to work, and say what was actually run.

## Contribute compute

Compute joins runs through the [compute ledger](ledger/README.md) and is credited by GitHub handle in [CREDITS.md](CREDITS.md). Today, the honest answer is:

- **Single-node phase 1 runs can be done now** with the commands in [docs/run0.md](docs/run0.md). Report results as an issue or pull request.
- **Joining a shared DiLoCo run does not exist yet.** There is no coordinator, no node agent and no `ofl join`. Phase 3 is where that gets built. See [docs/join-a-node.md](docs/join-a-node.md).
- **Donating a cluster** (as an island) or GPU time: use the [compute offer form](https://github.com/OpenFrontierLabs/openfrontier/issues/new?template=compute_offer.yml).

## Report a problem

Bugs and feature requests: [open an issue](https://github.com/OpenFrontierLabs/openfrontier/issues/new/choose) with the matching form. Questions: [Q&A Discussions](https://github.com/OpenFrontierLabs/openfrontier/discussions/categories/q-a). Security problems: privately, as described in [SECURITY.md](SECURITY.md). Conduct problems: see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Licence

Contributions are licensed under the Apache License 2.0 (see [LICENSE](LICENSE)).
