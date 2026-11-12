# Contributing

Open Frontier Labs is built in public. You do not need permission to start: read, run, open an issue, send a pull request.

Nothing has been trained yet. The repository is scaffolding for the first runs, so the most useful contributions right now are the ones that make those runs real.

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

## Contribute compute

Compute joins runs through the [compute ledger](ledger/README.md) and is credited by GitHub handle in [CREDITS.md](CREDITS.md). Today, the honest answer is:

- **Single-node phase 1 runs can be done now** with the commands in [docs/run0.md](docs/run0.md). Report results as an issue or pull request.
- **Joining a shared DiLoCo run does not exist yet.** There is no coordinator, no node agent and no `ofl join`. Phase 3 is where that gets built. See [docs/join-a-node.md](docs/join-a-node.md).
- **Donating a cluster** (as an island) or GPU time: open an issue describing what you can offer.

## Report a problem

Open an issue. For conduct problems see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Licence

Contributions are licensed under the Apache License 2.0 (see [LICENSE](LICENSE)).
