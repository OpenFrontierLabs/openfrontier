# AGENTS.md

Automated agents working in this repository read this first. Open Frontier Labs is built in
public, so treat it the way an outside contributor's agent would: it adds only what surprises an
agent, and defers to [CONTRIBUTING.md](CONTRIBUTING.md) for everything a human contributor is
already told.

## What binds you

- [CONTRIBUTING §AI agents](CONTRIBUTING.md#ai-agents): agent-written work is welcome, a human is
  accountable for it, and it is labelled — `Agent: <agent> / <model>` in the commit message and
  the PR description, plus the `agent-authored` label.
- [CONTRIBUTING §Contribute code](CONTRIBUTING.md#contribute-code): the gate (`make check`,
  `make test`, `make smoke`), upstreams pinned by `ref` and `sha` together, no vendored code, no
  launcher flag or config key the pinned upstream does not have.
- [CONTRIBUTING §Pull requests](CONTRIBUTING.md#pull-requests): one logical change, what you ran
  in the description, squash merge, an RFC for anything hard to reverse.
- [CONTRIBUTING §Sign your work](CONTRIBUTING.md#sign-your-work-dco): every commit carries a
  `Signed-off-by` matching your git identity.

## Commits

- Sign off on the first commit with `git commit -s`. Adding the trailer later means rewriting
  history, which CONTRIBUTING §Sign your work describes as a force-push. That rewrite is for your
  own unmerged branch only: never `main`, never a branch carrying someone else's commits.

## Dependencies

- Fix Dependabot alerts; never dismiss one.
- When a dependent caps a package below its patched release, first confirm the cap in that
  dependent's own metadata, then check whether upstream has relaxed it. If it has not, force the
  patched version with `[tool.uv] override-dependencies` and prove every consumer's real call path
  on the new major by reading its source and exercising that path. Write the reason, the
  dependents and the alert numbers into the comment above the override, and remove the override
  when an upstream cap lifts.
- Never add an `ignore:` entry to [.github/dependabot.yml](.github/dependabot.yml) for a version
  that has a patched release. It silences every later fix, including security fixes.

## What runs, and what it enforces

`main` has no required status checks today, so a merge is blocked by the person reviewing it, not
by GitHub. What runs on a PR:

- The DCO check: red until every commit carries a `Signed-off-by`. A maintainer will not merge
  past it.
- CodeQL, from GitHub default setup.
- `make check` and `make test`, which a reviewer expects you to have run (they are not wired into
  CI). `uv lock --check` when you touched `uv.lock`.

Guidance that review will hold you to:

- A test must fail on a plausible bug. Tests that check wiring, that a list grew, or that a call
  did not raise are not wanted.
- CPU only and no downloads in this environment: never make a test or smoke fetch weights,
  datasets or models, and never require a GPU to pass.
- `cmd | tail` reports `tail`'s status. Use `set -o pipefail` or `${PIPESTATUS[0]}` when you
  report an exit code.
- Numbers in a summary come from tool output, not memory. Quote the command.
- Report a check you did not run as not run. Never imply it passed.
