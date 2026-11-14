# Security policy

## Supported versions

Nothing has been released yet: no models, packages or tagged versions. Until a release exists, only the `main` branch is supported.

## Reporting a vulnerability

Do not open a public issue, discussion or Discord post for a security problem.

Report it privately through GitHub: open the [Security tab](https://github.com/OpenFrontierLabs/openfrontier/security) and choose **Report a vulnerability**. Private vulnerability reporting is enabled on every Open Frontier Labs repository. If you cannot use GitHub, email <openfrontierlabs@gmail.com> with "security" in the subject.

Include what is affected (file, command or URL), how to reproduce it, and the impact you see.

## What happens next

- A maintainer acknowledges the report in the advisory thread.
- We agree a fix and a disclosure date with you, fix it in a private fork, and publish a GitHub security advisory with the fix.
- You are credited in the advisory and in [CREDITS.md](CREDITS.md) unless you ask not to be.

## In scope

- Code in this repository: the `ofl` CLI, launchers, the ledger validator, CI workflows.
- Anything that could make a contributor's machine run code it did not agree to, leak a contributor's data or credentials, or let one participant corrupt another's training updates or credit.
- Supply-chain issues: a pinned upstream or dependency that is compromised.

Training-time trust between untrusted peers (poisoned updates, free riders) is a design topic as well as a security one: once the network exists, it is covered by an RFC in [`rfcs/`](rfcs/). Report concrete exploits privately here first.

## Telemetry

`ofl` compiles Sentry crash and unexpected-error reporting behind the default `telemetry` Cargo feature, activated only when an HTTPS `OFL_SENTRY_DSN` is supplied at build time (`option_env!("OFL_SENTRY_DSN")`). Source builds without `OFL_SENTRY_DSN` and `--no-default-features` builds never contact Sentry, and `OFL_SENTRY_DSN` is never committed to the repository. See [docs/telemetry.md](docs/telemetry.md) for opt-out flags, scrubbing behavior, and known limitations; report any scrubbing bypass or opt-out bug privately through the vulnerability reporting process above.
