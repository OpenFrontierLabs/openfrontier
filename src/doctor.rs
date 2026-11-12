//! `ofl doctor`: machine, toolchain and network checks. Works without a GPU.
//!
//! Status meanings:
//!   ok    as expected
//!   info  informational, never affects the exit code
//!   warn  something a run would want fixed; exit code stays 0
//!   FAIL  hard failure, exit code 1: Python older than 3.11, torch not
//!         importable, less than 2 GiB free next to the repo, or github.com /
//!         huggingface.co unreachable.
//!
//! Nothing here measures bandwidth. The network rows report DNS time and HTTPS
//! round-trip time only.

use std::net::ToSocketAddrs;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::error::Result;
use crate::repo::{find_root, run_captured, smoke_python};

/// Hosts probed for reachability and clock skew.
const HOSTS: [&str; 2] = ["huggingface.co", "github.com"];
/// Minimum Python version (major, minor).
const MIN_PYTHON: (u32, u32) = (3, 11);
/// Free disk below this is a hard failure.
const FAIL_FREE_GIB: f64 = 2.0;
/// Free disk below this is a warning: room for data and checkpoints, a heuristic.
const WARN_FREE_GIB: f64 = 20.0;
/// Clock skew beyond this warns.
const CLOCK_WARN_SECONDS: f64 = 30.0;
/// HTTPS round-trip samples per host.
const HTTPS_SAMPLES: usize = 3;
/// Timeout for short helper commands.
const HELPER_TIMEOUT: Duration = Duration::from_secs(15);
/// Timeout for one HTTPS probe.
const HTTPS_TIMEOUT: Duration = Duration::from_secs(8);
/// Timeout for importing torch (it is heavy).
const TORCH_TIMEOUT: Duration = Duration::from_secs(60);
/// Bytes per GiB.
const GIB: f64 = 1_073_741_824.0;

/// One doctor report line.
#[derive(Clone, Debug)]
pub struct Row {
    /// ok | info | warn | FAIL.
    pub status: &'static str,
    /// Check name.
    pub name: String,
    /// Human detail.
    pub detail: String,
}

/// One HTTPS probe result.
#[derive(Clone, Debug)]
pub struct HttpsProbe {
    /// Whether the request completed.
    pub ok: bool,
    /// Round-trip seconds.
    pub seconds: f64,
    /// Server Date header as unix seconds, when present and parseable.
    pub server_date: Option<i64>,
    /// Failure description when not ok.
    pub error: String,
}

/// Torch build facts.
#[derive(Clone, Debug)]
pub struct TorchInfo {
    /// torch.__version__.
    pub version: String,
    /// torch.version.cuda, None for CPU builds.
    pub cuda_build: Option<String>,
    /// `torch.cuda.is_available()`.
    pub cuda_available: bool,
}

/// A helper-command probe: (exit code, trimmed stdout).
type RunProbe = Box<dyn Fn(&[String]) -> (i32, String)>;
/// A DNS probe: lookup seconds, or None on failure.
type DnsProbe = Box<dyn Fn(&str) -> Option<f64>>;
/// A distribution-version probe: version string, or None.
type VersionProbe = Box<dyn Fn(&str) -> Option<String>>;

/// What doctor reads from the machine. Tests substitute fakes for the probes.
pub struct Host {
    /// Python version (major, minor, patch).
    pub python: (u32, u32, u32),
    /// Run a helper command: (exit code, trimmed stdout).
    pub run: RunProbe,
    /// Free bytes on the filesystem holding a path.
    pub disk_free_bytes: Box<dyn Fn(&Path) -> u64>,
    /// DNS lookup seconds, or None on failure.
    pub dns: DnsProbe,
    /// One HTTPS probe.
    pub https: Box<dyn Fn(&str) -> HttpsProbe>,
    /// Torch info, or None when torch is not importable.
    pub torch: Box<dyn Fn() -> Option<TorchInfo>>,
    /// Installed Python distribution version, or None.
    pub version: VersionProbe,
    /// Current unix seconds.
    pub now: Box<dyn Fn() -> i64>,
}

impl Host {
    /// The real machine, reading the trainer stack from the smoke interpreter.
    #[must_use]
    pub fn real(root: &Path) -> Self {
        let python = smoke_python(root);
        // ubs:ignore[rust.collections.clone-any] -- each probe closure owns its interpreter path; two clones at construction, not in a loop.
        let torch_python = python.clone();
        // ubs:ignore[rust.collections.clone-any] -- each probe closure owns its interpreter path; two clones at construction, not in a loop.
        let version_python = python.clone();
        Self {
            python: python_version(&python),
            run: Box::new(|argv| run_captured(argv, HELPER_TIMEOUT)),
            disk_free_bytes: Box::new(disk_free),
            dns: Box::new(dns_seconds),
            https: Box::new(https_probe),
            torch: Box::new(move || torch_info(&torch_python)),
            version: Box::new(move |name| dist_version(&version_python, name)),
            now: Box::new(unix_now),
        }
    }
}

/// Run doctor and print the report.
///
/// # Errors
///
/// [`OflError`] when the repo root cannot be found.
pub fn doctor(repo: Option<&Path>) -> Result<i32> {
    let root = match repo {
        Some(path) => path.to_path_buf(),
        None => find_root(None)?,
    };
    let rows = collect(&Host::real(&root), &root);
    println!("{}", render(&rows));
    let failed: Vec<&str> =
        rows.iter().filter(|r| r.status == "FAIL").map(|r| r.name.as_str()).collect();
    let warned = rows.iter().filter(|r| r.status == "warn").count();
    if failed.is_empty() {
        println!("\ndoctor: no hard failures; {warned} warning(s)");
    } else {
        println!("\ndoctor: {} hard failure(s): {}", failed.len(), failed.join(", "));
    }
    Ok(exit_code(&rows))
}

/// Collect every report row.
#[must_use]
pub fn collect(host: &Host, repo: &Path) -> Vec<Row> {
    let mut rows = toolchain_rows(host);
    let torch_info = (host.torch)();
    rows.extend(gpu_rows(host, torch_info.as_ref()));
    rows.push(disk_row(host, repo));
    rows.extend(network_rows(host));
    rows.push(Row {
        status: "info",
        name: "run fit".to_owned(),
        detail:
            "not computed: no run is open to join, so there are no thresholds to compare against"
                .to_owned(),
    });
    rows
}

/// Python, uv, torch and Python distribution rows.
fn toolchain_rows(host: &Host) -> Vec<Row> {
    let py = format!("{}.{}.{}", host.python.0, host.python.1, host.python.2);
    let need = format!("{}.{}", MIN_PYTHON.0, MIN_PYTHON.1);
    let ok = (host.python.0, host.python.1) >= MIN_PYTHON;
    let mut rows = vec![Row {
        status: if ok { "ok" } else { "FAIL" },
        name: "python".to_owned(),
        detail: format!("{py} (need >= {need})"),
    }];
    let (rc, out) = (host.run)(&["uv".to_owned(), "--version".to_owned()]);
    rows.push(if rc == 0 {
        Row { status: "ok", name: "uv".to_owned(), detail: out }
    } else {
        Row { status: "warn", name: "uv".to_owned(), detail: "not found on PATH".to_owned() }
    });
    match (host.torch)() {
        Some(info) => {
            rows.push(Row { status: "ok", name: "torch".to_owned(), detail: info.version });
        }
        None => rows.push(Row {
            status: "FAIL",
            name: "torch".to_owned(),
            detail: "not importable".to_owned(),
        }),
    }
    for pkg in ["ai2-olmo-core", "jsonschema"] {
        // ubs:ignore[rust.perf.string-alloc-in-loop] -- two fixed package names; building report rows, not a hot path.
        let name = pkg.to_owned();
        match (host.version)(pkg) {
            Some(ver) => rows.push(Row { status: "ok", name, detail: ver }),
            // ubs:ignore[rust.perf.string-alloc-in-loop] -- two fixed package names; building report rows, not a hot path.
            None => rows.push(Row { status: "warn", name, detail: "not installed".to_owned() }),
        }
    }
    rows
}

/// GPU and CUDA rows from nvidia-smi and torch.
fn gpu_rows(host: &Host, torch_info: Option<&TorchInfo>) -> Vec<Row> {
    let argv: Vec<String> =
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
    let (rc, out) = (host.run)(&argv);
    if rc != 0 || out.is_empty() {
        return vec![
            Row {
                status: "warn",
                name: "gpu".to_owned(),
                detail: "no GPU: CPU smoke only (nvidia-smi not found or reported none)".to_owned(),
            },
            Row { status: "info", name: "cuda".to_owned(), detail: "n/a without a GPU".to_owned() },
        ];
    }
    let lines: Vec<&str> = out.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let first = lines.first().copied().unwrap_or("");
    let name = first.split(',').next().unwrap_or("").trim();
    let mut rows = vec![Row {
        status: "ok",
        name: "gpu".to_owned(),
        detail: format!("{} x {name}", lines.len()),
    }];
    for (i, line) in lines.iter().enumerate() {
        // ubs:ignore[rust.perf.string-alloc-in-loop] -- one report row per GPU; building the doctor table, not a hot path.
        rows.push(Row { status: "info", name: format!("gpu {i}"), detail: (*line).to_owned() });
    }
    let driver = first.split(',').next_back().unwrap_or("").trim();
    rows.push(cuda_row(driver, torch_info));
    rows
}

/// The CUDA row comparing the driver with this environment's torch build.
fn cuda_row(driver: &str, torch_info: Option<&TorchInfo>) -> Row {
    let Some(info) = torch_info else {
        return Row {
            status: "info",
            name: "cuda".to_owned(),
            detail: format!("driver {driver}; torch not importable"),
        };
    };
    let Some(cuda_build) = &info.cuda_build else {
        return Row {
            status: "info",
            name: "cuda".to_owned(),
            detail: format!(
                "driver {driver}; this environment's torch {} is a CPU build (expected: GPU runs \
                 use their own environment, see docs/run0.md)",
                info.version
            ),
        };
    };
    let state = if info.cuda_available { "usable" } else { "NOT usable from torch" };
    Row {
        status: if info.cuda_available { "ok" } else { "warn" },
        name: "cuda".to_owned(),
        detail: format!(
            "driver {driver}; torch {} built for CUDA {cuda_build}, {state}",
            info.version
        ),
    }
}

/// Free-disk row with the fail/warn thresholds.
fn disk_row(host: &Host, repo: &Path) -> Row {
    // Free bytes fit comfortably in f64 for any real disk; precision loss is immaterial.
    #[expect(clippy::cast_precision_loss, reason = "disk byte counts are far below 2^52")]
    // ubs:ignore[rust.casts.as-casts] -- byte count to f64 for a GiB ratio; values are far below 2^52, no meaningful truncation.
    let free = (host.disk_free_bytes)(repo) as f64 / GIB;
    let status = if free < FAIL_FREE_GIB {
        "FAIL"
    } else if free < WARN_FREE_GIB {
        "warn"
    } else {
        "ok"
    };
    Row {
        status,
        name: "disk".to_owned(),
        detail: format!("{free:.1} GiB free at {}", repo.display()),
    }
}

/// DNS, HTTPS latency and clock rows.
fn network_rows(host: &Host) -> Vec<Row> {
    let mut rows = Vec::new();
    let mut skews: Vec<f64> = Vec::new();
    let mut skew_hosts: Vec<&str> = Vec::new();
    for name in HOSTS {
        let Some(dns) = (host.dns)(name) else {
            // ubs:ignore[rust.perf.string-alloc-in-loop] -- one report row per probed host; building the doctor table, not a hot path.
            rows.push(Row {
                status: "FAIL",
                name: format!("net {name}"),
                detail: "DNS lookup failed".to_owned(),
            });
            continue;
        };
        let probes: Vec<HttpsProbe> = (0..HTTPS_SAMPLES).map(|_| (host.https)(name)).collect();
        let good: Vec<&HttpsProbe> = probes.iter().filter(|p| p.ok).collect();
        if good.is_empty() {
            let err = probes.last().map(|p| p.error.as_str()).unwrap_or_default();
            // ubs:ignore[rust.perf.string-alloc-in-loop] -- one report row per probed host; building the doctor table, not a hot path.
            rows.push(Row {
                status: "FAIL",
                name: format!("net {name}"),
                detail: format!("HTTPS failed: {err}"),
            });
            continue;
        }
        let median = median_seconds(&good);
        // ubs:ignore[rust.perf.string-alloc-in-loop] -- one report row per probed host; building the doctor table, not a hot path.
        rows.push(Row {
            status: "ok",
            name: format!("net {name}"),
            detail: format!(
                "DNS {:.0} ms; HTTPS round trip incl. connect, median of {}: {:.0} ms (latency \
                 only, bandwidth not measured)",
                dns * 1000.0,
                good.len(),
                median * 1000.0
            ),
        });
        if let Some(date) = good.iter().find_map(|p| p.server_date) {
            // Clock skew in seconds is tiny; the i64->f64 cast is exact here.
            #[expect(clippy::cast_precision_loss, reason = "skew seconds are far below 2^52")]
            // ubs:ignore[rust.casts.as-casts] -- skew seconds to f64; values are far below 2^52, no meaningful truncation.
            let skew = ((host.now)() - date) as f64;
            skews.push(skew);
            skew_hosts.push(name);
        }
    }
    rows.push(clock_row(&skews, &skew_hosts));
    rows
}

/// Median of probe round-trips.
fn median_seconds(good: &[&HttpsProbe]) -> f64 {
    let mut secs: Vec<f64> = good.iter().map(|p| p.seconds).collect();
    secs.sort_by(f64::total_cmp);
    let mid = secs.len() / 2;
    // Callers pass only non-empty probe sets; the fallback keeps this panic-free regardless.
    match secs.get(mid) {
        Some(&center) if secs.len() % 2 == 1 => center,
        Some(&center) => f64::midpoint(secs.get(mid - 1).copied().unwrap_or(center), center),
        None => 0.0,
    }
}

/// Clock skew judged by the host that agrees best (a cached CDN Date inflates one host).
fn clock_row(skews: &[f64], skew_hosts: &[&str]) -> Row {
    if skews.is_empty() {
        return Row {
            status: "info",
            name: "clock".to_owned(),
            detail: "not checked: no server Date header received".to_owned(),
        };
    }
    let best = skews.iter().copied().min_by(|a, b| a.abs().total_cmp(&b.abs())).unwrap_or(0.0);
    let status = if best.abs() > CLOCK_WARN_SECONDS { "warn" } else { "ok" };
    let per_host: Vec<String> =
        skew_hosts.iter().zip(skews).map(|(h, s)| format!("{h} {s:+.1} s")).collect();
    Row {
        status,
        name: "clock".to_owned(),
        detail: format!("{best:+.1} s (smallest of: {})", per_host.join(", ")),
    }
}

/// Render the report table.
#[must_use]
pub fn render(rows: &[Row]) -> String {
    let width = rows.iter().map(|r| r.name.len()).max().unwrap_or(0);
    let mut lines = vec![format!("{:<6}  {:<width$}  DETAIL", "STATUS", "CHECK")];
    for r in rows {
        // ubs:ignore[rust.perf.string-alloc-in-loop] -- one rendered line per report row; building the doctor table, not a hot path.
        lines.push(format!("{:<6}  {:<width$}  {}", r.status, r.name, r.detail));
    }
    lines.join("\n")
}

/// 1 when any row is a hard failure, else 0.
#[must_use]
pub fn exit_code(rows: &[Row]) -> i32 {
    i32::from(rows.iter().any(|r| r.status == "FAIL"))
}

/// The smoke interpreter's Python version, via `python -c`.
fn python_version(python: &str) -> (u32, u32, u32) {
    let argv = vec![
        python.to_owned(),
        "-c".to_owned(),
        "import sys; print('.'.join(map(str, sys.version_info[:3])))".to_owned(),
    ];
    let (rc, out) = run_captured(&argv, HELPER_TIMEOUT);
    if rc != 0 {
        return (0, 0, 0);
    }
    let parts: Vec<u32> = out.split('.').filter_map(|p| p.parse().ok()).collect();
    match parts.as_slice() {
        [a, b, c] => (*a, *b, *c),
        _ => (0, 0, 0),
    }
}

/// Free bytes on the filesystem holding `path`.
fn disk_free(path: &Path) -> u64 {
    rustix::fs::statvfs(path).map_or(0, |stat| stat.f_bavail.saturating_mul(stat.f_frsize))
}

/// DNS lookup time for a host, or None on failure.
fn dns_seconds(host: &str) -> Option<f64> {
    let start = Instant::now();
    let addr = format!("{host}:443");
    let mut addrs = addr.to_socket_addrs().ok()?;
    addrs.next().map(|_| start.elapsed().as_secs_f64())
}

/// One HTTPS HEAD probe with the server Date header.
fn https_probe(host: &str) -> HttpsProbe {
    let start = Instant::now();
    let agent =
        ureq::Agent::config_builder().timeout_global(Some(HTTPS_TIMEOUT)).build().new_agent();
    let url = format!("https://{host}/");
    match agent.head(&url).call() {
        Ok(resp) => {
            let date =
                resp.headers().get("date").and_then(|v| v.to_str().ok()).and_then(parse_http_date);
            HttpsProbe {
                ok: true,
                seconds: start.elapsed().as_secs_f64(),
                server_date: date,
                error: String::new(),
            }
        }
        Err(e) => HttpsProbe {
            ok: false,
            seconds: start.elapsed().as_secs_f64(),
            server_date: None,
            error: e.to_string(),
        },
    }
}

/// Parse an HTTP Date header to unix seconds.
fn parse_http_date(text: &str) -> Option<i64> {
    let time = httpdate::parse_http_date(text).ok()?;
    let secs = time.duration_since(UNIX_EPOCH).ok()?.as_secs();
    i64::try_from(secs).ok()
}

/// Torch build facts from the smoke interpreter.
fn torch_info(python: &str) -> Option<TorchInfo> {
    let script = "import torch, json; print(json.dumps([torch.__version__, torch.version.cuda, \
                  bool(torch.cuda.is_available())]))";
    let argv = vec![python.to_owned(), "-c".to_owned(), script.to_owned()];
    let (rc, out) = run_captured(&argv, TORCH_TIMEOUT);
    if rc != 0 {
        return None;
    }
    let parsed: serde_json::Value = serde_json::from_str(&out).ok()?;
    let arr = parsed.as_array()?;
    Some(TorchInfo {
        version: arr.first()?.as_str()?.to_owned(),
        cuda_build: arr.get(1).and_then(|v| v.as_str()).map(str::to_owned),
        cuda_available: arr.get(2).and_then(serde_json::Value::as_bool).unwrap_or(false),
    })
}

/// Installed Python distribution version from the smoke interpreter.
fn dist_version(python: &str, name: &str) -> Option<String> {
    let script = format!("import importlib.metadata as m; print(m.version({name:?}))");
    let argv = vec![python.to_owned(), "-c".to_owned(), script];
    let (rc, out) = run_captured(&argv, HELPER_TIMEOUT);
    if rc == 0 && !out.is_empty() { Some(out) } else { None }
}

/// Current unix seconds.
fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
}
