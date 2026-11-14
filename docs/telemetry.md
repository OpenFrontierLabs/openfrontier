# Telemetry and privacy

`ofl` includes opt-out crash and unexpected-error reporting powered by [Sentry](https://sentry.io) behind the `telemetry` Cargo feature (enabled by default). Because volunteer machines are personal or shared workstations, telemetry is strictly scoped to panics and unexpected command failures (`OflError` with exit code != 2) and scrubs identifying information before anything leaves the host.

## When telemetry is active

A DSN is never committed to this repository. Instead, `ofl` reads `OFL_SENTRY_DSN` at compile time via `option_env!("OFL_SENTRY_DSN")`:

- **Local source builds without `OFL_SENTRY_DSN`**: telemetry is a complete no-op (`sentry::init` is never called and no network thread is started).
- **Builds with `OFL_SENTRY_DSN` set** (for example, future release builds; no prebuilt releases exist yet): telemetry initializes early in `ofl::cli::main` right after CLI flag parsing, unless any opt-out signal is active.
- **Compiled out**: `cargo build --no-default-features` omits the `sentry` crate entirely.

## Opting out

Any of these disables telemetry before `sentry::init` runs. An opt-out from any source is final (`OFL_TELEMETRY=1` does not override `DO_NOT_TRACK=1`):

1. Global CLI flag: `ofl --no-telemetry <command>`
2. Environment variable: `OFL_TELEMETRY=0` (also accepts `false`, `off`, `no`, case-insensitive)
3. Standard opt-out: `DO_NOT_TRACK=1` (also accepts `true`, `on`, `yes`)

## What is collected and what is scrubbed

Client configuration:

- `send_default_pii = false`
- `server_name = None` (and stripped in `before_send`)
- `traces_sampling_strategy = Disabled` (`0.0`, no performance tracing)
- `release = ofl@<version>`, `environment = development | production` from build profile
- No session tracking (`release-health` feature disabled)
- No `contexts` feature (`ContextIntegration` is not compiled in, so OS kernel/version, device model/arch, and hostnames are not collected)
- Default integrations limited to `AttachStacktraceIntegration`, `PanicIntegration`, and `ProcessStacktraceIntegration` (no `contexts`, `debug-images`, `logs`, or `metrics`).

Server-side project setting: the `ofl` Sentry project enables **Prevent Storing of IP Addresses** so Sentry discards the transport connection IP rather than storing IP or geolocation fields (the HTTPS request still reaches `sentry.io`).

Before any event is sent, `before_send` (`ofl::telemetry::scrub_event`) removes or redacts:

- `user`, `request`, `server_name`, and `device.name`
- `frame.vars` on all stack frames
- Sensitive metadata keys (`user`, `email`, `ip`, `hostname`, `host`, `cwd`, `env`, `args`, `token`, `secret`, `password`, `key`, `dsn`, `auth`, `cookie`, `session`) from tags, extra, contexts, and breadcrumbs
- Home directory paths (`/home/<user>`, `/Users/<user>`, `C:\Users\<user>`, `$HOME`, `$USERPROFILE`) and local usernames, replaced with `~`
- Current working directory paths, replaced with `<cwd>`
- Local hostnames, replaced with `<host>`
- IPv4 and IPv6 addresses, replaced with `<ip>`
- Sentry DSNs, `Bearer`/`Basic` auth headers, `KEY=VALUE` / `KEY = VALUE` secret assignments, and secret-carrying CLI flags (`--token`, `--api-key`, `--password`, `--secret`, `--dsn`, `--auth`, etc.)

## Future release builds

No prebuilt binaries or tagged releases exist yet. When release builds are produced, inject the DSN through the build environment only:

```sh
OFL_SENTRY_DSN="<project-dsn>" cargo build --release
```

Do not hardcode or commit `OFL_SENTRY_DSN` in source, Cargo config, or CI files.
