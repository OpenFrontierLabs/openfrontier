# AGENTS.md

This file binds automated agents working in this checkout. Human process lives in
[CONTRIBUTING.md](CONTRIBUTING.md), [GOVERNANCE.md](GOVERNANCE.md) and [rfcs/](rfcs/README.md); read
them before your first change. Constraints only: run `make help` for targets, `ofl help` for
commands, [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the layers.

## Commits and branches

- Every commit is signed off: `git commit -s`. The DCO check is required to merge, a missing
  trailer cannot be added without rewriting history, and this checkout forbids force-push. Start
  signed or accept that the branch is dead.
- Never force-push, rebase a pushed branch, reset, clean, or delete a branch or tag, yours or
  anyone else's. A superseded branch costs nothing; a rewrite costs the PR.
- One logical change per PR. Squash merge: the PR title becomes the commit on `main`, imperative,
  e.g. `Report MPS devices in ofl doctor`, with `Closes #N` in the body.

## Gate

- Run at commit, once: `make check` (ruff, format, mypy, ledger schema, configs and pins),
  `make test`, and `make smoke` for anything touching training, data, checkpoints or the CLI's
  run paths. `make check-affected` does not exist; do not invent gates.
- Behaviour lives in `src/ofl/`. A `make` recipe stays a one-line forward to `ofl`; never put
  shell logic in the Makefile.
- CPU only, no downloads, in this environment: tests and smoke must not fetch models, datasets or
  weights, and must not require a GPU. GPU work runs on a node ([docs/run0.md](docs/run0.md)).
- A test must fail on a plausible bug. Tests that only check wiring, that a list grew, or that a
  function did not raise are rejected in review.
- `set -o pipefail` (or `${PIPESTATUS[0]}`) when you report the exit status of a piped command;
  `cmd | tail` reports `tail`'s status, not `cmd`'s.

## Dependencies

- `uv` only, `uv.lock` is the source of truth. `uv lock --check` passes; a lock change names only
  the intended package.
- Upstreams are pinned by `ref` and `sha` together in [configs/upstreams.toml](configs/upstreams.toml).
  Move both and re-run `ofl run0 verify` / `ofl moe verify`. Do not vendor upstream code.
- Do not add a flag or config key a launcher passes unless it exists in the pinned upstream.
- Dependabot alerts are fixed, never dismissed. When a dependent caps a package below its patched
  release: confirm the cap in that dependent's metadata, check whether upstream relaxed it, then
  force it with `[tool.uv] override-dependencies` and prove each consumer's real call path on the
  new major (read its source; exercise the lock acquire/release or equivalent). Write the reason,
  the pinned dependents and the alerts into the comment above the override, and delete the
  override when an upstream cap lifts.
- An `ignore:` rule in [.github/dependabot.yml](.github/dependabot.yml) written for an upstream cap
  goes stale the moment an override removes the cap: drop it in the same change, or it silences
  every later patched release in that range.

## Honesty

- No command pretends to work. An unbuilt phase fails with a message saying so; never a stub that
  succeeds, a silent no-op, or a placeholder that returns.
- When a doc claims something the code does not do, that is a bug: fix the doc or the code in the
  same PR.
- Say in the PR what you ran. Report a check you did not run as not run; never imply it passed.
- Numbers in a summary come from tool output, not memory. Quote the command that produced them.

## Hard to reverse

- An RFC first (a markdown file in `rfcs/`, opened as a PR) for protocol messages, the ledger
  schema or how credit is computed, licences, layer boundaries, a new repository, the milestone
  ladder, or what a run must publish to count as done.
- Never change `ledger/` schema, licence files or layer boundaries in a code PR.

## Workspace

- Scratch goes in `agent-temp create <slug>` and leaves with `agent-temp trash`; never a raw
  `mktemp` path paired with `rm -rf`.
- Leave files you did not create alone. Unexpected changes in `git status` belong to someone.
- Durable decisions, corrections and lessons for this repository go to the Algodyne knowledge
  graph (`assert_algodyne_facts`, actor `omp:openfrontier`) with evidence a later reader can
  check: a commit or PR URL, not a session worktree path.
