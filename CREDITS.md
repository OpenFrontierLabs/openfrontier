# Credits

Credit here is permanent, public and tied to the repository history. There is no token and no equity.

## What is recorded

Compute contributions are recorded in a public, machine-readable **compute ledger**, one JSON file per run and period under [`ledger/`](ledger/), validated against [`ledger/schema.json`](ledger/schema.json) by `ofl check` and `ofl test`. For every node:

| Field | Meaning |
| --- | --- |
| `github_handle` | who ran the node |
| `gpu_hours` | GPU-hours the node was training in the period |
| `tokens_processed` | training tokens the node processed |
| `helpful_update_share` | fraction of the node's submitted updates that lowered the loss (0 to 1) |
| `island` | optional: the donated cluster the node belongs to |

Donated or pooled clusters are named as **islands** in the same file, with who donated them. Details, and how a ledger is produced, are in [ledger/README.md](ledger/README.md).

Model cards list compute contributors and donors by GitHub handle, rendered from the [model card template](docs/model-card-template.md). Every release names the merged-code contributors. Anyone with merged work on a release is an author on its paper, following the large named author lists of BigScience and EleutherAI.

## Status

Nothing has been trained, so there are no contributions to credit yet and no real ledger file. The only file under `ledger/` that holds data is [`ledger/example.EXAMPLE.json`](ledger/example.EXAMPLE.json), which is fake and marked as such. It names no real person.

Real credits are added here by pull request when a run produces them.

## Compute donors

None yet.

## Code contributors

Everyone with merged work is listed by the repository's commit history. A curated list will be added when there is something to curate.
