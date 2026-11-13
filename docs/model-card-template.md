# Model card template

**Status: template. Nothing has been trained and no weights have been released.** Every release gets a model card rendered from this file, following the [Hugging Face model card format](https://huggingface.co/docs/hub/model-cards); the [OLMo model cards](https://huggingface.co/allenai) are the reference shape. Fill the `<placeholders>` at release time from the run's own files. The **Contributors** and **Compute donors** sections are generated from repository data, never typed by hand (see the last two sections).

## Model details

| | |
| --- | --- |
| Name | `<run>` (for example `run0`) |
| Architecture | `<architecture>`, documented in [`docs/`](.) (for example [run0.md](run0.md)) |
| Parameters | `<n>` total, `<n>` non-embedding |
| Tokenizer | `<tokenizer id and revision>` from [`manifests/datasets.toml`](../manifests/datasets.toml) |
| Licence | [Apache-2.0](../LICENSE), no use restrictions |
| Released artefacts | weights, every checkpoint, and the run logs, all public |

## Intended use

`<what the model is for>`. Primary intended users: `<who>`. Out-of-scope use: `<what it must not be used for>`.

## Training data

Datasets, licences and Hugging Face revisions are those recorded in [`manifests/datasets.toml`](../manifests/datasets.toml) at release time:

| Dataset | Revision | Licence |
| --- | --- | --- |
| `<dataset id>` | `<revision>` | `<licence>` |

No content hashes have been computed for any dataset; a revision is a Hugging Face git commit, nothing more. `<note anything else about the data, e.g. tokenisation>`.

## Training procedure

- Run config: [`configs/<phase>/<run>.toml`](../configs/) (`<run>`), including seeds, batch, optimiser and schedule.
- Upstream code: every dependency pinned to an exact commit in [`configs/upstreams.toml`](../configs/upstreams.toml); nothing is vendored.
- Compute: the run's public record is the compute ledger file [`ledger/<run>-<period>.json`](../ledger/) (schema: [`ledger/schema.json`](../ledger/schema.json), rules: [ledger/README.md](../ledger/README.md)).

## Evaluation

`<which evals were run, on which benchmarks, with which harness and pinned commit; results table>`. Every release carries an evals and risks note; if no eval was run, say so here rather than leaving the section empty.

## Risks and limitations

`<known risks, failure modes and limitations, including anything the evals exposed>`.

## Licence

[Apache-2.0](../LICENSE). No use restrictions. Datasets and logs we author are intended to be CC-BY-4.0; data we reuse keeps its upstream terms (see [`manifests/datasets.toml`](../manifests/datasets.toml)).

## Contributors

Generated from this repository's merged pull requests (the commit history), as [CREDITS.md](../CREDITS.md) states: everyone with merged work on the release is listed by GitHub handle, following the large named author lists of BigScience and EleutherAI. **No name is ever added by hand** — the list comes from `git log` of merged work, so it cannot drift from the actual history.

## Compute donors

Generated from the compute ledger files [`ledger/<run>-<period>.json`](../ledger/), validated against [`ledger/schema.json`](../ledger/schema.json): one line per node's `github_handle` with `gpu_hours`, `tokens_processed` and `helpful_update_share`, and one line per island (donated or pooled cluster) with its `name` and `donor`. **No name is ever added by hand** — the lists come from the ledger files, which `ofl check` and `ofl test` validate (see [ledger/README.md](../ledger/README.md)).
