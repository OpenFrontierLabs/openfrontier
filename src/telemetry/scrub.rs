//! PII, local-path, IP, and secret scrubbing for telemetry events.

use std::net::{Ipv4Addr, Ipv6Addr};
use std::sync::LazyLock;

use regex::Regex;

/// Matches `/home/<user>` or `/Users/<user>` path prefixes.
#[expect(clippy::expect_used, reason = "the regex literal is compile-time valid")]
// ubs:ignore[rust.perf.regex-new,rust.ownership.unwrap-expect] -- compiled once in a LazyLock static; the expect is on a compile-time-valid literal, not runtime input.
static UNIX_HOME_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"/(?:home|Users)/[^/\s:;,\x22']+").expect("static regex is valid")
});

/// Matches Windows `C:\Users\<user>` path prefixes.
#[expect(clippy::expect_used, reason = "the regex literal is compile-time valid")]
// ubs:ignore[rust.perf.regex-new,rust.ownership.unwrap-expect] -- compiled once in a LazyLock static; the expect is on a compile-time-valid literal, not runtime input.
static WIN_HOME_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)[A-Z]:\\Users\\[^\\\s:;,"']+"#).expect("static regex is valid")
});

/// Matches Sentry DSN URLs (`https://<key>@host/<project>`).
#[expect(clippy::expect_used, reason = "the regex literal is compile-time valid")]
// ubs:ignore[rust.perf.regex-new,rust.ownership.unwrap-expect] -- compiled once in a LazyLock static; the expect is on a compile-time-valid literal, not runtime input.
static DSN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"https?://[^\s/@]+@[^\s/]+/\d+").expect("static regex is valid"));

/// Matches `Bearer <token>` or `Basic <token>` credentials.
#[expect(clippy::expect_used, reason = "the regex literal is compile-time valid")]
// ubs:ignore[rust.perf.regex-new,rust.ownership.unwrap-expect] -- compiled once in a LazyLock static; the expect is on a compile-time-valid literal, not runtime input.
static BEARER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]+").expect("static regex is valid")
});

/// Matches secret-looking `KEY=VALUE` and spaced `KEY = VALUE` assignments in text.
#[expect(clippy::expect_used, reason = "the regex literal is compile-time valid")]
// ubs:ignore[rust.perf.regex-new,rust.ownership.unwrap-expect] -- compiled once in a LazyLock static; the expect is on a compile-time-valid literal, not runtime input.
static ENV_SECRET_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|KEY|DSN|AUTH|CREDENTIAL|COOKIE|SESSION)[A-Z0-9_]*)\s*=\s*(?:"[^"]*"|'[^']*'|[^\s]+)"#,
    )
    .expect("static regex is valid")
});

/// Matches CLI flags carrying secret values (`--token <val>`, `--token=<val>`, etc.).
#[expect(clippy::expect_used, reason = "the regex literal is compile-time valid")]
// ubs:ignore[rust.perf.regex-new,rust.ownership.unwrap-expect] -- compiled once in a LazyLock static; the expect is on a compile-time-valid literal, not runtime input.
static CLI_SECRET_FLAG_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)(--(?:token|api-key|apikey|password|passwd|secret|auth|authorization|dsn|private-key|access-token|refresh-token))(?:=|\s+)(?:"[^"]*"|'[^']*'|[^\s]+)"#,
    )
    .expect("static regex is valid")
});

/// Pure inputs describing local PII/path values to scrub from telemetry events.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScrubContext {
    /// Home directory paths to replace with `~`.
    pub home_dirs: Vec<String>,
    /// Local usernames to replace with `~`.
    pub usernames: Vec<String>,
    /// Local hostnames to replace with `<host>`.
    pub hostnames: Vec<String>,
    /// Current working directory to replace with `<cwd>`.
    pub cwd: Option<String>,
}

impl ScrubContext {
    /// Collect local paths, usernames, hostname and cwd from the process environment.
    #[must_use]
    pub fn from_env() -> Self {
        let mut home_dirs = Vec::new();
        for key in ["HOME", "USERPROFILE"] {
            if let Ok(val) = std::env::var(key) {
                push_non_empty(&mut home_dirs, &val);
            }
        }
        let mut usernames = Vec::new();
        for key in ["USER", "USERNAME", "LOGNAME"] {
            if let Ok(val) = std::env::var(key) {
                push_non_empty(&mut usernames, &val);
            }
        }
        for home in &home_dirs {
            if let Some(name) = username_from_home(home) {
                push_non_empty(&mut usernames, name);
            }
        }
        let mut hostnames = Vec::new();
        for key in ["HOSTNAME", "COMPUTERNAME"] {
            if let Ok(val) = std::env::var(key) {
                push_non_empty(&mut hostnames, &val);
            }
        }
        if let Ok(proc_host) = std::fs::read_to_string("/proc/sys/kernel/hostname") {
            push_non_empty(&mut hostnames, &proc_host);
        }
        let cwd =
            std::env::current_dir().ok().map(|p| p.display().to_string()).filter(|s| !s.is_empty());

        let mut ctx = Self { home_dirs, usernames, hostnames, cwd };
        ctx.normalize();
        ctx
    }

    /// Sort replacement lists longest-first and deduplicate.
    pub fn normalize(&mut self) {
        for list in [&mut self.home_dirs, &mut self.usernames, &mut self.hostnames] {
            list.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
            list.dedup();
        }
    }
}

fn push_non_empty(out: &mut Vec<String>, raw: &str) {
    let trimmed = raw.trim();
    if !trimmed.is_empty() && !out.iter().any(|existing| existing == trimmed) {
        out.push(trimmed.to_owned());
    }
}

fn username_from_home(home: &str) -> Option<&str> {
    let trimmed = home.trim().trim_end_matches(['/', '\\']);
    let segment = trimmed.rsplit(['/', '\\']).next()?;
    if segment.is_empty() || segment == "home" || segment == "Users" { None } else { Some(segment) }
}

/// Scrub sensitive paths, usernames, hostnames, IPs, DSNs, env vars and secret CLI args from text.
#[must_use]
pub fn scrub_str(input: &str, ctx: &ScrubContext) -> String {
    let mut out = input.to_owned();

    // Redact credentialed DSNs, bearer tokens, secret env vars, and secret CLI flags first.
    out = DSN_RE.replace_all(&out, "<redacted-dsn>").into_owned();
    out = BEARER_RE.replace_all(&out, "$1 <redacted>").into_owned();
    out = ENV_SECRET_RE.replace_all(&out, "$1=<redacted>").into_owned();
    out = CLI_SECRET_FLAG_RE.replace_all(&out, "$1 <redacted>").into_owned();

    // Replace cwd before generic home dirs when cwd is distinct from $HOME.
    if let Some(cwd) = ctx.cwd.as_deref() {
        let trimmed = cwd.trim().trim_end_matches(['/', '\\']);
        if trimmed.len() > 1
            && !ctx.home_dirs.iter().any(|h| h.trim_end_matches(['/', '\\']) == trimmed)
        {
            out = out.replace(trimmed, "<cwd>");
        }
    }

    // Replace known home directories (longest first) and generic OS home-dir prefixes with `~`.
    for home in &ctx.home_dirs {
        let trimmed = home.trim().trim_end_matches(['/', '\\']);
        if trimmed.len() > 1 {
            out = out.replace(trimmed, "~");
        }
    }
    out = UNIX_HOME_RE.replace_all(&out, "~").into_owned();
    out = WIN_HOME_RE.replace_all(&out, "~").into_owned();

    // Replace known usernames (word-bounded when len >= 2) with `~`.
    for user in &ctx.usernames {
        let trimmed = user.trim();
        if trimmed.len() >= 2 {
            out = replace_whole_word(&out, trimmed, "~");
        }
    }

    // Replace known hostnames (word-bounded when len >= 3) with `<host>`.
    for host in &ctx.hostnames {
        let trimmed = host.trim();
        if trimmed.len() >= 3 {
            out = replace_whole_word(&out, trimmed, "<host>");
        }
    }

    scrub_ips_in_text(&out)
}

/// Replace valid IPv4 and IPv6 tokens (including `[ipv6]:port` and `ipv4:port`) with `<ip>`
/// while preserving Rust module paths like `core::f64::consts::PI` and `db::query`.
fn scrub_ips_in_text(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while !rest.is_empty() {
        let split_at = rest.find(|c: char| !is_ip_token_char(c)).unwrap_or(rest.len());
        if split_at == 0 {
            if let Some(ch) = rest.chars().next() {
                out.push(ch);
                let next_idx = ch.len_utf8();
                rest = rest.get(next_idx..).unwrap_or("");
            } else {
                break;
            }
            continue;
        }
        let (token, tail) = rest.split_at(split_at);
        let body = token.trim_end_matches('.');
        let trailing_dots = token.get(body.len()..).unwrap_or("");
        if is_ip_token(body) {
            out.push_str("<ip>");
            out.push_str(trailing_dots);
        } else {
            out.push_str(token);
        }
        rest = tail;
    }
    out
}

const fn is_ip_token_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | ':' | '[' | ']' | '%' | '_')
}

fn is_ip_token(token: &str) -> bool {
    is_ip_candidate(token) || is_ip_candidate(token.trim_end_matches([':', '.']))
}

fn is_ip_candidate(cleaned: &str) -> bool {
    if let Some(rest) = cleaned.strip_prefix('[') {
        let Some((inside, port_part)) = rest.split_once(']') else {
            return false;
        };
        if let Some(port) = port_part.strip_prefix(':') {
            if port.is_empty() || !port.chars().all(|c| c.is_ascii_digit()) {
                return false;
            }
        } else if !port_part.is_empty() {
            return false;
        }
        let host = inside.split('%').next().unwrap_or(inside);
        return host.parse::<Ipv6Addr>().is_ok();
    }
    if cleaned.parse::<Ipv4Addr>().is_ok() {
        return true;
    }
    if let Some((addr, port)) = cleaned.rsplit_once(':')
        && !port.is_empty()
        && port.chars().all(|c| c.is_ascii_digit())
        && addr.parse::<Ipv4Addr>().is_ok()
    {
        return true;
    }
    // Unbracketed IPv6: require at least 2 colons and reject bare `a::b` with no digits
    // unless it starts with `::` (like `::1`) or has >=3 colons (like `fe80::1` vs `db::query`).
    if cleaned.matches(':').count() >= 2 && !cleaned.contains("::") {
        return cleaned.parse::<Ipv6Addr>().is_ok();
    }
    if cleaned.contains("::")
        && (cleaned.starts_with("::") || cleaned.bytes().any(|b| b.is_ascii_digit()))
    {
        return cleaned.parse::<Ipv6Addr>().is_ok();
    }
    false
}

/// Replace occurrences of `needle` bounded by non-alphanumeric characters.
fn replace_whole_word(haystack: &str, needle: &str, replacement: &str) -> String {
    if needle.is_empty() {
        return haystack.to_owned();
    }
    let mut result = String::with_capacity(haystack.len());
    let mut rest = haystack;
    while let Some((before, after)) = rest.split_once(needle) {
        let left_ok = before.chars().next_back().is_none_or(|c| !is_word_char(c));
        let right_ok = after.chars().next().is_none_or(|c| !is_word_char(c));
        result.push_str(before);
        if left_ok && right_ok {
            result.push_str(replacement);
        } else {
            result.push_str(needle);
        }
        rest = after;
    }
    result.push_str(rest);
    result
}

const fn is_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Check whether a metadata key name looks like PII, env, argv, or a secret.
#[must_use]
pub fn is_sensitive_key(key: &str) -> bool {
    let lower = key.trim().to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "user"
            | "username"
            | "user_id"
            | "email"
            | "ip"
            | "ip_address"
            | "client_ip"
            | "remote_addr"
            | "server_name"
            | "hostname"
            | "host"
            | "cwd"
            | "pwd"
            | "env"
            | "environ"
            | "args"
            | "argv"
            | "cmdline"
            | "command_line"
            | "home"
            | "cookie"
            | "cookies"
            | "session"
            | "authorization"
            | "auth"
            | "key"
            | "dsn"
    ) || lower.ends_with("_token")
        || lower.ends_with("_secret")
        || lower.ends_with("_password")
        || lower.ends_with("_passwd")
        || lower.ends_with("_key")
        || lower.ends_with("_dsn")
        || lower.contains("token")
        || lower.contains("secret")
        || lower.contains("password")
}

#[cfg(feature = "telemetry")]
mod event_scrub {
    use sentry::protocol::{Context, Event, Map, Stacktrace, Value};

    use super::{ScrubContext, is_sensitive_key, scrub_str};

    /// Scrub PII, local paths, hostnames, IPs, env vars and secret args from an event.
    #[must_use]
    pub fn scrub_event(mut event: Event<'static>, ctx: &ScrubContext) -> Event<'static> {
        event.user = None;
        event.request = None;
        event.server_name = None;
        event.message = event.message.take().map(|m| scrub_str(&m, ctx));
        event.culprit = event.culprit.take().map(|c| scrub_str(&c, ctx));
        event.transaction = event.transaction.take().map(|t| scrub_str(&t, ctx));

        if let Some(logentry) = &mut event.logentry {
            logentry.message = scrub_str(&logentry.message, ctx);
            for param in &mut logentry.params {
                scrub_value(param, ctx);
            }
        }

        scrub_string_map(&mut event.tags, ctx);
        scrub_value_map(&mut event.extra, ctx);
        scrub_contexts(&mut event.contexts, ctx);

        for crumb in &mut event.breadcrumbs.values {
            crumb.message = crumb.message.take().map(|m| scrub_str(&m, ctx));
            scrub_value_map(&mut crumb.data, ctx);
        }

        for exc in &mut event.exception.values {
            exc.value = exc.value.take().map(|v| scrub_str(&v, ctx));
            if let Some(st) = &mut exc.stacktrace {
                scrub_stacktrace(st, ctx);
            }
            if let Some(st) = &mut exc.raw_stacktrace {
                scrub_stacktrace(st, ctx);
            }
        }

        if let Some(st) = &mut event.stacktrace {
            scrub_stacktrace(st, ctx);
        }

        for thread in &mut event.threads.values {
            thread.name = thread.name.take().map(|n| scrub_str(&n, ctx));
            if let Some(st) = &mut thread.stacktrace {
                scrub_stacktrace(st, ctx);
            }
            if let Some(st) = &mut thread.raw_stacktrace {
                scrub_stacktrace(st, ctx);
            }
        }

        event
    }

    fn scrub_stacktrace(st: &mut Stacktrace, ctx: &ScrubContext) {
        for frame in &mut st.frames {
            frame.abs_path = frame.abs_path.take().map(|p| scrub_str(&p, ctx));
            frame.filename = frame.filename.take().map(|p| scrub_str(&p, ctx));
            frame.context_line = frame.context_line.take().map(|l| scrub_str(&l, ctx));
            for line in &mut frame.pre_context {
                *line = scrub_str(line, ctx);
            }
            for line in &mut frame.post_context {
                *line = scrub_str(line, ctx);
            }
            frame.vars.clear();
        }
    }

    fn scrub_contexts(contexts: &mut Map<String, Context>, ctx: &ScrubContext) {
        contexts.retain(|key, _| !is_sensitive_key(key));
        for context in contexts.values_mut() {
            match context {
                Context::Device(device) => {
                    device.name = None;
                    device.other.clear();
                }
                Context::Os(os) => {
                    scrub_value_map(&mut os.other, ctx);
                }
                Context::Other(map) => {
                    scrub_value_map(map, ctx);
                }
                _ => {}
            }
        }
    }

    fn scrub_string_map(map: &mut Map<String, String>, ctx: &ScrubContext) {
        map.retain(|key, _| !is_sensitive_key(key));
        for val in map.values_mut() {
            *val = scrub_str(val, ctx);
        }
    }

    fn scrub_value_map(map: &mut Map<String, Value>, ctx: &ScrubContext) {
        map.retain(|key, _| !is_sensitive_key(key));
        for val in map.values_mut() {
            scrub_value(val, ctx);
        }
    }

    fn scrub_value(value: &mut Value, ctx: &ScrubContext) {
        match value {
            Value::String(s) => *s = scrub_str(s, ctx),
            Value::Array(arr) => {
                for item in arr {
                    scrub_value(item, ctx);
                }
            }
            Value::Object(obj) => {
                obj.retain(|key, _| !is_sensitive_key(key));
                for val in obj.values_mut() {
                    scrub_value(val, ctx);
                }
            }
            Value::Null | Value::Bool(_) | Value::Number(_) => {}
        }
    }
}

#[cfg(feature = "telemetry")]
pub use event_scrub::scrub_event;

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test assertions panic on failure")]
mod tests {
    use super::*;

    #[test]
    fn scrub_str_removes_paths_users_hosts_ips_and_secrets() {
        let mut ctx = ScrubContext {
            home_dirs: vec!["/home/alice".to_owned()],
            usernames: vec!["alice".to_owned()],
            hostnames: vec!["gpu-node-01".to_owned()],
            cwd: Some("/tmp/alice-work/repo".to_owned()),
        };
        ctx.normalize();

        let raw = "user alice on gpu-node-01 (192.168.1.42, [fd12:3456:789a::1]:8000, fe80::1, ::1) \
                   in /tmp/alice-work/repo failed reading /home/alice/.ofl/config with \
                   OFL_SENTRY_DSN=https://secret@o1.ingest.sentry.io/9 and \
                   api_key = \"toml_secret_val\" and SENTRY_AUTH_TOKEN=sntrys_abc123 \
                   --token mysecret --api-key='k123' Authorization: Bearer tok_xyz \
                   at std::io::Error core::f64::consts::PI db::query v0.1.0";
        let cleaned = scrub_str(raw, &ctx);

        assert!(!cleaned.contains("alice"), "username leaked: {cleaned}");
        assert!(!cleaned.contains("gpu-node-01"), "hostname leaked: {cleaned}");
        assert!(!cleaned.contains("192.168.1.42"), "ip leaked: {cleaned}");
        assert!(!cleaned.contains("fd12:3456:789a::1"), "bracketed ipv6 leaked: {cleaned}");
        assert!(!cleaned.contains("fe80::1"), "ipv6 leaked: {cleaned}");
        assert!(!cleaned.contains("::1"), "loopback ipv6 leaked: {cleaned}");
        assert!(!cleaned.contains("/tmp/alice-work/repo"), "cwd leaked: {cleaned}");
        assert!(!cleaned.contains("secret"), "dsn secret leaked: {cleaned}");
        assert!(!cleaned.contains("sntrys_abc123"), "env token leaked: {cleaned}");
        assert!(!cleaned.contains("toml_secret_val"), "spaced config secret leaked: {cleaned}");
        assert!(!cleaned.contains("mysecret"), "cli token leaked: {cleaned}");
        assert!(!cleaned.contains("k123"), "cli api-key leaked: {cleaned}");
        assert!(!cleaned.contains("tok_xyz"), "bearer token leaked: {cleaned}");
        assert!(cleaned.contains("~/"), "expected ~ for home dir: {cleaned}");
        assert!(cleaned.contains("<cwd>"), "expected <cwd>: {cleaned}");
        assert!(cleaned.contains("<host>"), "expected <host>: {cleaned}");
        assert!(cleaned.contains("<ip>"), "expected <ip>: {cleaned}");
        assert!(cleaned.contains("std::io::Error"), "rust path must stay intact: {cleaned}");
        assert!(
            cleaned.contains("core::f64::consts::PI"),
            "core::f64 path must stay intact: {cleaned}"
        );
        assert!(cleaned.contains("db::query"), "db::query path must stay intact: {cleaned}");
        assert!(cleaned.contains("v0.1.0"), "version triple must stay intact: {cleaned}");
        assert!(is_sensitive_key("key"));
        assert!(is_sensitive_key("dsn"));

        let trailing_period = scrub_str(
            "seen 10.20.30.40. then 10.20.30.41:8080. and [fd12:3456:789a::2]:9000. and fe80::1. and 2001:db8:: and 2001:db8::.",
            &ctx,
        );
        assert_eq!(trailing_period, "seen <ip>. then <ip>. and <ip>. and <ip>. and <ip> and <ip>.");
    }

    #[test]
    fn scrub_str_handles_macos_and_windows_home_paths() {
        let ctx = ScrubContext::default();
        let mac = scrub_str("panic at /Users/brad/coding/openfrontier/src/cli.rs:10", &ctx);
        assert_eq!(mac, "panic at ~/coding/openfrontier/src/cli.rs:10");

        let win = scrub_str(r"failed opening C:\Users\brad\.ofl\token", &ctx);
        assert_eq!(win, r"failed opening ~\.ofl\token");
    }

    #[cfg(feature = "telemetry")]
    #[test]
    fn client_options_privacy_defaults_and_before_send_scrubs_event() {
        use sentry::TracesSamplingStrategy;
        use sentry::protocol::{Context, Value};

        use crate::telemetry::build_client_options;

        let ctx = ScrubContext {
            home_dirs: vec!["/home/volunteer".to_owned()],
            usernames: vec!["volunteer".to_owned()],
            hostnames: vec!["rig-42".to_owned()],
            cwd: Some("/home/volunteer/openfrontier".to_owned()),
        };

        let opts =
            build_client_options(Some("https://public@example.invalid/1"), false, None, None, ctx)
                .unwrap();

        assert!(!opts.send_default_pii);
        assert!(opts.server_name.is_none());
        assert!(opts.attach_stacktrace);
        assert!(matches!(opts.traces_sampling_strategy, TracesSamplingStrategy::Disabled));
        assert_eq!(opts.release.as_deref(), Some(concat!("ofl@", env!("CARGO_PKG_VERSION"))));

        let before_send = opts.before_send.as_ref().unwrap();
        let scrubbed = before_send(dirty_test_event()).unwrap();

        assert!(scrubbed.user.is_none());
        assert!(scrubbed.request.is_none());
        assert!(scrubbed.server_name.is_none());
        assert!(!scrubbed.tags.contains_key("user"));
        assert_eq!(scrubbed.tags.get("command").map(String::as_str), Some("doctor"));
        assert!(!scrubbed.extra.contains_key("OFL_SWARM_TOKEN"));
        assert_eq!(scrubbed.extra.get("detail"), Some(&Value::String("~/foo".to_owned())));

        let msg = scrubbed.message.as_deref().unwrap();
        assert!(!msg.contains("volunteer"));
        assert!(!msg.contains("rig-42"));
        assert!(msg.contains("<cwd>/x"));

        let dev = match scrubbed.contexts.get("device") {
            Some(Context::Device(dev)) => dev,
            other => unreachable!("unexpected device context: {other:?}"),
        };
        assert!(dev.name.is_none());
        assert_eq!(dev.arch.as_deref(), Some("x86_64"));

        let frame = &scrubbed.exception.values[0].stacktrace.as_ref().unwrap().frames[0];
        assert!(frame.vars.is_empty());
        assert_eq!(frame.abs_path.as_deref(), Some("<cwd>/src/cli.rs"));
    }

    #[cfg(feature = "telemetry")]
    fn dirty_test_event() -> sentry::protocol::Event<'static> {
        use sentry::protocol::{
            Breadcrumb, Context, DeviceContext, Event, Exception, Frame, Stacktrace, User, Value,
        };
        let mut event = Event::new();
        event.server_name = Some("rig-42".into());
        event.user = Some(User {
            username: Some("volunteer".to_owned()),
            email: Some("v@example.com".to_owned()),
            ip_address: Some("10.0.0.5".parse().unwrap()),
            ..Default::default()
        });
        event.request = Some(sentry::protocol::Request {
            url: Some("https://secret-host/path?token=abc".parse().unwrap()),
            ..Default::default()
        });
        event.message =
            Some("error for volunteer on rig-42 at /home/volunteer/openfrontier/x".to_owned());
        event.tags.insert("user".to_owned(), "volunteer".to_owned());
        event.tags.insert("command".to_owned(), "doctor".to_owned());
        event.extra.insert("OFL_SWARM_TOKEN".to_owned(), Value::String("tok".to_owned()));
        event.extra.insert("detail".to_owned(), Value::String("/home/volunteer/foo".to_owned()));
        event.contexts.insert(
            "device".to_owned(),
            Context::Device(Box::new(DeviceContext {
                name: Some("rig-42".to_owned()),
                arch: Some("x86_64".to_owned()),
                ..Default::default()
            })),
        );
        event.breadcrumbs.values.push(Breadcrumb {
            message: Some("running --token abc in /home/volunteer/openfrontier".to_owned()),
            ..Default::default()
        });
        event.exception.values.push(Exception {
            ty: "OflError".to_owned(),
            value: Some("cannot read /home/volunteer/.ofl/secret".to_owned()),
            stacktrace: Some(Stacktrace {
                frames: vec![Frame {
                    abs_path: Some("/home/volunteer/openfrontier/src/cli.rs".to_owned()),
                    filename: Some("/home/volunteer/openfrontier/src/cli.rs".to_owned()),
                    vars: std::iter::once((
                        "api_key".to_owned(),
                        Value::String("secret".to_owned()),
                    ))
                    .collect(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        });
        event
    }
}
