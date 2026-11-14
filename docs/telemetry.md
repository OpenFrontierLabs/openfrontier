# Telemetry and privacy

`ofl` includes opt-out crash and unexpected-error reporting powered by [Sentry](https://sentry.io) behind the `telemetry` Cargo feature. In `Cargo.toml`, `default = ["telemetry"]` (`telemetry = ["dep:sentry"]`), so `telemetry` is the only default feature. Because volunteer machines are personal or shared workstations, telemetry is strictly scoped to panics and unexpected command failures (`OflError` with exit code != 2) and scrubs identifying information before anything leaves the host.

## When telemetry is active

A DSN is never committed to this repository. Instead, `ofl` reads `OFL_SENTRY_DSN` at compile time via `option_env!("OFL_SENTRY_DSN")` and accepts only non-empty, valid HTTPS DSNs:

- **Local source builds without `OFL_SENTRY_DSN`** (or with an empty, invalid, or non-HTTPS value): telemetry is a complete no-op (`sentry::init` is never called and no network thread is started).
- **Builds with an HTTPS `OFL_SENTRY_DSN` set** (for example, future release builds; no prebuilt releases exist yet): telemetry initializes early in `ofl::cli::main` right after CLI flag parsing, unless any opt-out signal is active. HTTP upload and shutdown timeouts are bounded to 3 seconds (`ureq` + `rustls`, reusing the CLI's existing HTTP/TLS stack) so CLI exit does not hang if the endpoint is unreachable.
- **Compiled out**: `cargo build --no-default-features` (or `cargo build --release --no-default-features`) omits the `sentry` crate entirely while keeping all other `ofl` functionality intact, since `telemetry` is the only default Cargo feature.

## Opting out

Any of these disables telemetry before `sentry::init` runs. Opt-outs are checked in the order below and are final — another variable cannot re-enable telemetry once opted out (for example, `OFL_TELEMETRY=1` does not override `DO_NOT_TRACK=1`, and `DO_NOT_TRACK=0` does not override `OFL_TELEMETRY=0`):

1. Global CLI flag: `ofl --no-telemetry <command>` (or `ofl <command> --no-telemetry`). On `ofl swarm` and `ofl node` passthrough commands, `--no-telemetry` placed before `--` disables `ofl` telemetry and is stripped before forwarding arguments to the Python swarm; `--no-telemetry` after `--` is passed through unchanged.
2. Environment variable: `OFL_TELEMETRY=0` (also accepts `false`, `off`, `no`; case-insensitive, whitespace-trimmed).
3. Standard opt-out: `DO_NOT_TRACK=1` (also accepts `true`, `on`, `yes`; case-insensitive, whitespace-trimmed).

## What is collected and what is scrubbed

Events sent when telemetry is active:

- Uncaught panics (`PanicIntegration`)
- Unexpected top-level command errors (`OflError` with exit code != 2, preserving the `source()` chain; CLI misuse errors with exit code 2 are not sent)
- Stack traces (`attach_stacktrace = true`, `AttachStacktraceIntegration`, `ProcessStacktraceIntegration`), with frame variables cleared and paths scrubbed
- Release and build environment: `release = ofl@<version>`, `environment = development | production` from the Cargo build profile
- Manual verification events from the hidden `ofl telemetry-test` maintainer command (`"ofl sentry test"` at `info` level)

Client privacy defaults:

- `send_default_pii = false`
- `server_name = None` (and cleared in `before_send`)
- Traces off: `traces_sampling_strategy = Disabled` (no performance tracing)
- Sessions off: the `release-health` feature is not enabled (no session tracking)
- No `contexts` feature: `ContextIntegration` is not compiled in, so OS kernel/version, device model/arch, and hostnames are not collected automatically
- Default integrations disabled (`default_integrations = false`); only `AttachStacktraceIntegration`, `PanicIntegration`, and `ProcessStacktraceIntegration` are registered (no `contexts`, `debug-images`, `logs`, or `metrics`).

Server-side project setting: the `ofl` Sentry project enables **Prevent Storing of IP Addresses** so Sentry discards the transport connection IP address rather than storing it (coarse country/region geo metadata may still be derived at ingest from the HTTPS connection to `sentry.io`).

Before any event is sent, `before_send` (`ofl::telemetry::scrub_event`) removes or redacts:

- `user`, `request`, `server_name`, and (if any `Device` context is present) `device.name` and `device.other`
- `frame.vars` on all stack frames (exception, raw, event, and thread stacktraces)
- Sensitive metadata keys from `tags`, `extra`, `contexts`, `breadcrumbs`, and nested objects (exact keys: `user`, `username`, `user_id`, `email`, `ip`, `ip_address`, `client_ip`, `remote_addr`, `server_name`, `hostname`, `host`, `cwd`, `pwd`, `env`, `environ`, `args`, `argv`, `cmdline`, `command_line`, `home`, `cookie`, `cookies`, `session`, `authorization`, `auth`, `key`, `dsn`, plus any key ending in `_token`, `_secret`, `_password`, `_passwd`, `_key`, `_dsn` or containing `token`, `secret`, `password`)
- Home directory paths (`$HOME`, `$USERPROFILE`, and generic `/home/<user>`, `/Users/<user>`, `C:\Users\<user>`) and local usernames (`USER`, `USERNAME`, `LOGNAME`, and the leaf segment of `$HOME`/`$USERPROFILE`; word-bounded when length >= 2), replaced with `~`
- Current working directory paths (when distinct from `$HOME` / `$USERPROFILE`), replaced with `<cwd>`
- Local hostnames (`HOSTNAME`, `COMPUTERNAME`, `/proc/sys/kernel/hostname`; word-bounded when length >= 3), replaced with `<host>`
- IPv4 and IPv6 addresses (including `ipv4:port`, `[ipv6]:port`, and unbracketed IPv6, while preserving Rust module paths such as `std::io::Error`), replaced with `<ip>`
- Sentry DSN URLs (`<redacted-dsn>`), `Bearer`/`Basic` auth headers (`<scheme> <redacted>`), secret-shaped `KEY=VALUE` / `KEY = VALUE` assignments (`<KEY>=<redacted>`), and secret-carrying CLI flags (`--token`, `--api-key`, `--apikey`, `--password`, `--passwd`, `--secret`, `--auth`, `--authorization`, `--dsn`, `--private-key`, `--access-token`, `--refresh-token`)

## Release builds and verification

No prebuilt binaries or tagged releases exist yet. When maintainers produce release builds, inject the DSN through the build environment only:

```sh
OFL_SENTRY_DSN="<project-dsn>" cargo build --release
```

Do not hardcode or commit `OFL_SENTRY_DSN` in source, Cargo config, or CI files.

To verify that a compiled binary has `OFL_SENTRY_DSN` wired and that opt-outs suppress sending, maintainers can run the hidden `telemetry-test` subcommand (omitted from `ofl --help` via `hide = true` in `src/cli.rs`):

```sh
ofl telemetry-test
```

When telemetry is active, it sends one `"ofl sentry test"` info message, flushes for up to 5 seconds, and prints `telemetry test event sent`. When disabled, opted out, or built without an HTTPS `OFL_SENTRY_DSN`, it prints `telemetry disabled (no event sent)` and sends nothing.
