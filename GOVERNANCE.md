# Governance

Short on purpose.

## Status

Open Frontier Labs is not yet a legal entity. The intended path is a **nonprofit foundation** that holds the GitHub organisation, the repositories, the marks, the weights and any money donated. That foundation has not been formed. Until it is, the organisation is a set of public repositories under [github.com/OpenFrontierLabs](https://github.com/OpenFrontierLabs), and nothing here is a legal promise.

## Founding steward

**Brad Lishman** ([@bradlishman](https://github.com/bradlishman)) is the founding steward. The steward started the organisation, keeps it running until the foundation exists, and hands what the foundation should own over to it when it does.

## How decisions are made

- **Public by default.** Discussion, decisions and reasons live in public issues, pull requests and threads. Private channels are for conduct reports and for things that would hurt a person if posted.
- **Commit rights follow merged work.** Anyone who has had work merged in a repository earns commit rights on that repository. Rights are per repository.
- **Maintainers decide, in the open.** The people with commit rights on a repo decide what merges there. If you disagree, say so on the thread with your reasons. Where maintainers disagree and cannot settle it, the steward breaks the tie and writes the reason down.
- There are no committees and no approval boards.

## Rights follow merged work

- **Triage** (label, assign, close duplicates) after your first merged PR in a repository.
- **Write** (merge) after a handful of merged PRs in a layer, granted by that layer's maintainer or the steward.
- Rights lapse after a year with no activity and come back on request.

## Layer maintainers

The project is organised in the eight layers of [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Each layer has a maintainer who reviews its PRs and decides what merges there.

| Layer | Home | Maintainer |
| --- | --- | --- |
| 1. Contributor hardware | `openfrontier` (`ofl doctor`) | steward, until filled |
| 2. Open Frontier Node | planned `openfrontier-node` repo | steward, until filled |
| 3. Rust P2P network | planned `openfrontier-node` repo | steward, until filled |
| 4. Training engine | `openfrontier` | steward, until filled |
| 5. Storage and evidence | `openfrontier` (manifests, ledger) | steward, until filled |
| 6. Member and operator console | `website` | steward, until filled |
| 7. Hosting and relays | `website` and runbooks | steward, until filled |
| 8. Open outputs and evals | `openfrontier` and the Hugging Face org | steward, until filled |

The first contributor with merged work in a layer who wants the role becomes its maintainer, and is added to [CODEOWNERS](.github/CODEOWNERS) in the same PR that updates this table. There is no application process. A layer may have more than one maintainer.

## RFCs

Decisions that are hard to reverse (protocol messages, the ledger schema, licences, layer boundaries, the milestone ladder) go through an [RFC](rfcs/README.md): a markdown file in `rfcs/` opened as a pull request. It merges when the maintainers of the affected layers agree and nobody has an unresolved objection; the steward breaks ties and writes down why. Everything else is a normal pull request.

## Steering group

When **three layers each have an active maintainer**, a steering group forms: the steward plus one maintainer per active layer. It handles only cross-layer decisions, the brand, money and ties. Everything else stays with layer maintainers. This is a trigger, not a gate: the group forms when the people exist, and its members are listed here when it does.

When the foundation is formed, it holds the assets and the steering group becomes its technical council. No token and no equity at any stage.

## Money and compute

- Funding is meant to come from many donors and from donated compute, never from a single owner.
- Compute donors and contributors are credited publicly by name, see [CREDITS.md](CREDITS.md).
- **No token. No equity.** Nothing in this project can be bought, sold or held as a financial interest in it.

## Licences

Code and weights are Apache-2.0. Datasets and logs we author are intended to be CC-BY-4.0. Data we reuse keeps its upstream terms, recorded in [manifests/datasets.toml](manifests/datasets.toml). Use-restriction licences are not used for anything we release.

## Changing this document

Open a pull request. It is merged like any other change.
