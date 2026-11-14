//! Opt-out Sentry error and panic telemetry with strict PII scrubbing.
//!
//! Telemetry is compiled behind the default `telemetry` feature and activated
//! only when an HTTPS `OFL_SENTRY_DSN` is present at compile time and the user
//! has not opted out via `--no-telemetry`, `OFL_TELEMETRY=0|false|off|no`, or
//! `DO_NOT_TRACK=1`.

pub mod scrub;

#[cfg(feature = "telemetry")]
pub use scrub::scrub_event;
pub use scrub::{ScrubContext, is_sensitive_key, scrub_str};

/// Build-time Sentry DSN injected from the `OFL_SENTRY_DSN` environment variable.
#[cfg(feature = "telemetry")]
const BUILD_DSN: Option<&str> = option_env!("OFL_SENTRY_DSN");

/// Lifetime guard for the telemetry client; dropping flushes pending events.
pub struct TelemetryGuard {
    #[cfg(feature = "telemetry")]
    inner: Option<sentry::ClientInitGuard>,
}

impl TelemetryGuard {
    /// A no-op guard used when telemetry is disabled or compiled out.
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            #[cfg(feature = "telemetry")]
            inner: None,
        }
    }

    /// Whether an active Sentry client is bound and enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        #[cfg(feature = "telemetry")]
        {
            self.inner.as_ref().is_some_and(sentry::ClientInitGuard::is_enabled)
        }
        #[cfg(not(feature = "telemetry"))]
        {
            false
        }
    }
}

/// Decide whether telemetry is opted out.
///
/// Any opt-out signal disables telemetry (`--no-telemetry`,
/// `OFL_TELEMETRY=0|false|off|no`, or `DO_NOT_TRACK=1|true|on|yes`).
/// An opt-out from any source is final and cannot be overridden by another
/// variable (for example, `OFL_TELEMETRY=1` does not override `DO_NOT_TRACK=1`).
#[must_use]
pub fn is_opted_out(
    cli_no_telemetry: bool,
    ofl_telemetry: Option<&str>,
    do_not_track: Option<&str>,
) -> bool {
    if cli_no_telemetry {
        return true;
    }
    if let Some(raw) = ofl_telemetry
        && matches!(raw.trim().to_ascii_lowercase().as_str(), "0" | "false" | "off" | "no")
    {
        return true;
    }
    if let Some(raw) = do_not_track
        && matches!(raw.trim().to_ascii_lowercase().as_str(), "1" | "true" | "on" | "yes")
    {
        return true;
    }
    false
}

/// Check opt-out status from the CLI flag and live process environment variables.
#[must_use]
pub fn is_opted_out_from_env(cli_no_telemetry: bool) -> bool {
    let ofl_telemetry = std::env::var("OFL_TELEMETRY").ok();
    let do_not_track = std::env::var("DO_NOT_TRACK").ok();
    is_opted_out(cli_no_telemetry, ofl_telemetry.as_deref(), do_not_track.as_deref())
}

/// Initialize telemetry as early as possible after CLI flag parsing.
///
/// Returns a disabled guard when `--no-telemetry`, `OFL_TELEMETRY`, or
/// `DO_NOT_TRACK` opts out, when `OFL_SENTRY_DSN` was unset at build time, or
/// when the `telemetry` Cargo feature is disabled.
#[must_use]
pub fn init(cli_no_telemetry: bool) -> TelemetryGuard {
    #[cfg(feature = "telemetry")]
    {
        let ofl_telemetry = std::env::var("OFL_TELEMETRY").ok();
        let do_not_track = std::env::var("DO_NOT_TRACK").ok();
        init_with_config(
            BUILD_DSN,
            cli_no_telemetry,
            ofl_telemetry.as_deref(),
            do_not_track.as_deref(),
            ScrubContext::from_env(),
        )
    }
    #[cfg(not(feature = "telemetry"))]
    {
        let _ = cli_no_telemetry;
        TelemetryGuard::disabled()
    }
}

/// Capture an unexpected top-level error (preserving its `source()` chain).
pub fn capture_error(err: &(dyn std::error::Error + 'static)) {
    #[cfg(feature = "telemetry")]
    {
        let _event_id = sentry::capture_error(err);
    }
    #[cfg(not(feature = "telemetry"))]
    {
        let _ = err;
    }
}

/// Send one `"ofl sentry test"` info message and flush if telemetry is active.
///
/// Returns `true` when telemetry is active and the event was queued and flushed.
#[must_use]
pub fn send_test_event(guard: &TelemetryGuard) -> bool {
    #[cfg(feature = "telemetry")]
    {
        let Some(client) = guard.inner.as_ref() else {
            return false;
        };
        if !client.is_enabled() {
            return false;
        }
        let id = sentry::capture_message("ofl sentry test", sentry::Level::Info);
        if id.is_nil() {
            return false;
        }
        client.flush(Some(std::time::Duration::from_secs(5)))
    }
    #[cfg(not(feature = "telemetry"))]
    {
        let _ = guard;
        false
    }
}

#[cfg(feature = "telemetry")]
mod sentry_impl {
    use std::borrow::Cow;
    use std::sync::Arc;
    use std::time::Duration;

    use sentry::integrations::backtrace::{
        AttachStacktraceIntegration, ProcessStacktraceIntegration,
    };
    use sentry::integrations::panic::PanicIntegration;
    use sentry::transports::UreqHttpTransportOptions;
    use sentry::types::{Dsn, Scheme};
    use sentry::{
        ClientOptions, Integration, TracesSamplingStrategy, Transport, TransportFactory,
        TransportOptions,
    };
    use ureq::tls::{TlsConfig, TlsProvider};
    use ureq::{Agent, Proxy};

    use super::{ScrubContext, TelemetryGuard, is_opted_out, scrub_event};

    /// Maximum wall-clock time for a single Sentry HTTP envelope upload.
    const TRANSPORT_TIMEOUT: Duration = Duration::from_secs(3);

    /// Transport factory that bounds `ureq` request/connect deadlines so CLI exit cannot stall.
    struct BoundedUreqTransportFactory;

    impl TransportFactory for BoundedUreqTransportFactory {
        fn create_transport_with_options(&self, options: TransportOptions) -> Arc<dyn Transport> {
            let mut builder =
                Agent::config_builder().timeout_global(Some(TRANSPORT_TIMEOUT)).tls_config(
                    TlsConfig::builder()
                        .provider(TlsProvider::Rustls)
                        .disable_verification(options.accept_invalid_certs)
                        .build(),
                );
            if let Some(proxy_url) =
                options.https_proxy.as_deref().or(options.http_proxy.as_deref())
                && let Ok(proxy) = Proxy::new(proxy_url)
            {
                builder = builder.proxy(Some(proxy));
            }
            let agent = builder.build().new_agent();
            Arc::new(UreqHttpTransportOptions::from(options).with_agent(agent).build())
        }
    }

    /// Parse a build-time DSN if present, non-empty, and using HTTPS.
    pub fn parse_dsn(raw: Option<&str>) -> Option<Dsn> {
        raw.map(str::trim)
            .filter(|s| !s.is_empty())
            .and_then(|s| s.parse::<Dsn>().ok())
            .filter(|dsn| dsn.scheme() == Scheme::Https)
    }

    /// Build `ClientOptions` when enabled, or `None` when disabled/opted out/no DSN.
    pub fn build_client_options(
        raw_dsn: Option<&str>,
        cli_no_telemetry: bool,
        ofl_telemetry: Option<&str>,
        do_not_track: Option<&str>,
        mut scrub_ctx: ScrubContext,
    ) -> Option<ClientOptions> {
        if is_opted_out(cli_no_telemetry, ofl_telemetry, do_not_track) {
            return None;
        }
        let dsn = parse_dsn(raw_dsn)?;
        scrub_ctx.normalize();

        let integrations: Vec<Arc<dyn Integration>> = vec![
            Arc::new(AttachStacktraceIntegration),
            Arc::new(PanicIntegration::default()),
            Arc::new(ProcessStacktraceIntegration),
        ];

        let environment = if cfg!(debug_assertions) { "development" } else { "production" };
        let before_send_ctx = Arc::new(scrub_ctx);

        let mut opts = ClientOptions::default();
        opts.dsn = Some(dsn);
        opts.release = sentry::release_name!();
        opts.environment = Some(Cow::Borrowed(environment));
        opts.server_name = None;
        opts.send_default_pii = false;
        opts.attach_stacktrace = true;
        opts.traces_sampling_strategy = TracesSamplingStrategy::Disabled;
        opts.default_integrations = false;
        opts.integrations = integrations;
        opts.transport = Some(Arc::new(BoundedUreqTransportFactory));
        opts.shutdown_timeout = TRANSPORT_TIMEOUT;
        opts.before_send = Some(Arc::new(move |event| Some(scrub_event(event, &before_send_ctx))));
        Some(opts)
    }

    /// Initialize Sentry from explicit parameters (testable without mutating process env).
    #[must_use]
    pub fn init_with_config(
        raw_dsn: Option<&str>,
        cli_no_telemetry: bool,
        ofl_telemetry: Option<&str>,
        do_not_track: Option<&str>,
        scrub_ctx: ScrubContext,
    ) -> TelemetryGuard {
        let Some(opts) =
            build_client_options(raw_dsn, cli_no_telemetry, ofl_telemetry, do_not_track, scrub_ctx)
        else {
            return TelemetryGuard::disabled();
        };
        TelemetryGuard { inner: Some(sentry::init(opts)) }
    }
}

#[cfg(feature = "telemetry")]
pub use sentry_impl::{build_client_options, init_with_config};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opt_out_flag_beats_everything() {
        assert!(is_opted_out(true, None, None));
        assert!(is_opted_out(true, Some("1"), Some("0")));
        assert!(is_opted_out(true, Some("true"), None));
    }

    #[test]
    fn opt_out_ofl_telemetry_values() {
        for off in ["0", "false", "FALSE", "off", "Off", "no", " 0 "] {
            assert!(is_opted_out(false, Some(off), None), "{off:?} should opt out");
            assert!(
                is_opted_out(false, Some(off), Some("0")),
                "{off:?} should override DO_NOT_TRACK=0"
            );
        }
        for on in ["1", "true", "on", "yes"] {
            assert!(!is_opted_out(false, Some(on), None), "{on:?} should allow");
            assert!(
                is_opted_out(false, Some(on), Some("1")),
                "DO_NOT_TRACK=1 is final even when OFL_TELEMETRY={on:?}"
            );
        }
    }

    #[test]
    fn opt_out_do_not_track_precedence() {
        assert!(is_opted_out(false, None, Some("1")));
        assert!(is_opted_out(false, None, Some("true")));
        assert!(is_opted_out(false, Some(""), Some("1")));
        assert!(is_opted_out(false, Some("maybe"), Some("1")));
        assert!(is_opted_out(false, Some("1"), Some("1")));
        assert!(!is_opted_out(false, None, Some("0")));
        assert!(!is_opted_out(false, None, Some("false")));
        assert!(!is_opted_out(false, None, None));
    }

    #[cfg(feature = "telemetry")]
    #[test]
    fn client_options_disabled_without_dsn_or_on_opt_out() {
        let ctx = ScrubContext::default();
        assert!(build_client_options(None, false, None, None, ctx.clone()).is_none());
        assert!(build_client_options(Some(""), false, None, None, ctx.clone()).is_none());
        assert!(build_client_options(Some("not-a-dsn"), false, None, None, ctx.clone()).is_none());
        assert!(
            build_client_options(
                Some("http://public@example.invalid/1"),
                false,
                None,
                None,
                ctx.clone()
            )
            .is_none(),
            "cleartext http DSN must be rejected"
        );
        assert!(
            build_client_options(
                Some("https://public@example.invalid/1"),
                true,
                None,
                None,
                ctx.clone()
            )
            .is_none()
        );
        assert!(
            build_client_options(
                Some("https://public@example.invalid/1"),
                false,
                Some("off"),
                None,
                ctx.clone()
            )
            .is_none()
        );
        assert!(
            build_client_options(
                Some("https://public@example.invalid/1"),
                false,
                None,
                Some("1"),
                ctx
            )
            .is_none()
        );
    }
}
