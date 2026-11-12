# Compute ledger

A public, machine-readable record of who contributed compute to a run. It feeds [CREDITS.md](../CREDITS.md) and the contributor lists on model cards.

**Nothing has been trained, so there is no real ledger file here.** [`example.EXAMPLE.json`](example.EXAMPLE.json) is an illustration: fake handles, fake donor, dates in the year 2000, and `"example": true`. Do not read any number in it as a result.

## Files

- `schema.json`: JSON Schema (draft 2020-12) for one ledger file.
- `example.EXAMPLE.json`: the example. A file may carry `"example": true` if and only if its name contains `EXAMPLE`; `ofl check` enforces both directions so an example cannot pass for real data.
- Real files, once a run exists, are named `<run>-<period>.json`, for example `run0-2027-01.json`.

## One file

```text
schema_version  1
run             run0, run1, ...
period          inclusive UTC start and end dates
islands[]       id, name, donor          donated or pooled clusters, named as islands
nodes[]         node_id, github_handle, island?, gpu_hours, tokens_processed, helpful_update_share
```

Rules the validator enforces beyond the schema: `node_id`s are unique within a file; a node's `island` must be declared in `islands`; `period.start` is not after `period.end`; dates are real calendar dates; `helpful_update_share` is between 0 and 1.

`helpful_update_share` is defined as the fraction of a node's submitted updates that lowered the loss. Nothing computes it yet: the scoring that would produce it is not built (see [docs/join-a-node.md](../docs/join-a-node.md)).

## Validate

```bash
make check    # validates schema.json and every ledger/*.json
make test     # tests the validator, including rejected cases
```
