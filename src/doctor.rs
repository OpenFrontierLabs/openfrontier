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
//! Network rows report DNS time and HTTPS round-trip time. Bandwidth is opt-in
//! (`ofl doctor --bandwidth`): one bounded download from a pinned public object,
//! reported as MB/s. Upload is not measured.

use std::io::Read;
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
/// Public object timed by `--bandwidth`: `model.safetensors` of `openai-community/gpt2`
/// at a pinned revision, so the object cannot change under the measurement. Checked
/// with curl on 2026-11-15: HEAD (after the CDN redirect) gave HTTP 200 and
/// `content-length: 548105171`; a `Range: bytes=0-33554431` request gave HTTP 206 and
/// `content-range: bytes 0-33554431/548105171`.
const BANDWIDTH_URL: &str = "https://huggingface.co/openai-community/gpt2/resolve/\
                             607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors";
/// Most bytes `--bandwidth` moves: a 32 MiB range of that file, not the whole 548 MB.
const BANDWIDTH_CAP_BYTES: u64 = 32 * 1024 * 1024;
/// `--bandwidth` stops reading this long after the request starts, keeping what it got.
const BANDWIDTH_DEADLINE: Duration = Duration::from_secs(8);
/// Hard stop for a stalled `--bandwidth` connection, a little after the deadline.
const BANDWIDTH_HARD_TIMEOUT: Duration = Duration::from_secs(10);
/// Bytes per megabyte (decimal), the unit throughput is reported in.
const MEGABYTE: f64 = 1_000_000.0;

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

/// One bounded download measurement.
#[derive(Clone, Debug)]
pub struct BandwidthProbe {
    /// URL that was fetched.
    pub url: String,
    /// Body bytes received.
    pub bytes: u64,
    /// Seconds spent receiving those bytes, counted from the response headers.
    pub seconds: f64,
    /// Whether reading stopped at the time limit before the byte cap.
    pub timed_out: bool,
    /// Failure description, when the download failed or no data arrived in time.
    pub error: Option<String>,
}

/// Torch build facts.
#[derive(Clone, Debug, Default)]
pub struct TorchInfo {
    /// torch.__version__.
    pub version: String,
    /// torch.version.cuda, None for CPU builds.
    pub cuda_build: Option<String>,
    /// `torch.cuda.is_available()`; true on `ROCm` builds too.
    pub cuda_available: bool,
    /// `torch.version.hip`, None unless this is a `ROCm` build.
    pub hip_build: Option<String>,
    /// `torch.backends.mps.is_available()`.
    pub mps_available: bool,
    /// Names of the `torch.cuda` devices (NVIDIA or `ROCm`).
    pub devices: Vec<String>,
    /// Total memory in bytes per `torch.cuda` device, None when torch does not expose it.
    pub device_memory_bytes: Vec<Option<u64>>,
    /// `torch.mps.recommended_max_memory()` in bytes, when available.
    pub mps_recommended_max_bytes: Option<u64>,
}

/// A helper-command probe: (exit code, trimmed stdout).
type RunProbe = Box<dyn Fn(&[String]) -> (i32, String)>;
/// A DNS probe: lookup seconds, or None on failure.
type DnsProbe = Box<dyn Fn(&str) -> Option<f64>>;
/// A distribution-version probe: version string, or None.
type VersionProbe = Box<dyn Fn(&str) -> Option<String>>;
/// A bandwidth probe: one bounded download.
type BandwidthRun = Box<dyn Fn() -> BandwidthProbe>;

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
    /// Bounded download probe; None when `--bandwidth` was not requested.
    pub bandwidth: Option<BandwidthRun>,
}

impl Host {
    /// The real machine, reading the trainer stack from the smoke interpreter.
    #[must_use]
    pub fn real(root: &Path, bandwidth: bool) -> Self {
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
            bandwidth: bandwidth.then(|| -> BandwidthRun { Box::new(bandwidth_probe) }),
        }
    }
}

/// Run doctor and print the report. `bandwidth` adds one bounded download.
///
/// # Errors
///
/// [`OflError`] when the repo root cannot be found.
pub fn doctor(repo: Option<&Path>, bandwidth: bool) -> Result<i32> {
    let root = match repo {
        Some(path) => path.to_path_buf(),
        None => find_root(None)?,
    };
    let rows = collect(&Host::real(&root, bandwidth), &root);
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
    rows.push(bandwidth_row(host));
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

/// One report row.
fn row(status: &'static str, name: &str, detail: impl Into<String>) -> Row {
    Row { status, name: name.to_owned(), detail: detail.into() }
}

/// Memory as GiB text, or a note when torch does not expose it.
fn format_gib(bytes: Option<u64>) -> String {
    match bytes {
        Some(b) if b > 0 => format!("{:.1} GiB", gib(b)),
        _ => "memory not exposed by torch".to_owned(),
    }
}

/// Bytes to GiB for display.
#[expect(clippy::cast_precision_loss, reason = "device memory is far below 2^52 bytes")]
// ubs:ignore[rust.casts.as-casts] -- byte count to f64 for a GiB display value; far below 2^52.
fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB
}

/// GPU, CUDA, `ROCm` and MPS rows: nvidia-smi first, then what torch sees (`ROCm`, MPS).
fn gpu_rows(host: &Host, torch_info: Option<&TorchInfo>) -> Vec<Row> {
    let argv: Vec<String> =
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
    let (rc, out) = (host.run)(&argv);
    if rc == 0 && !out.is_empty() {
        return nvidia_rows(&out, torch_info);
    }
    match torch_info {
        Some(info) if info.hip_build.is_some() => rocm_rows(info),
        Some(info) if info.mps_available => mps_rows(host, info),
        _ => vec![
            row("warn", "gpu", "no GPU: CPU smoke only (no CUDA, ROCm or MPS device detected)"),
            row("info", "cuda", "n/a without a GPU"),
            row("info", "rocm", NO_ROCM),
            row("info", "mps", NO_MPS),
        ],
    }
}

/// Detail for the `ROCm` row when there is no `ROCm` device.
const NO_ROCM: &str = "not present (no ROCm device)";
/// Detail for the MPS row when there is no Apple MPS device.
const NO_MPS: &str = "not present (Apple MPS not available)";

/// Rows for a machine where nvidia-smi reports GPUs.
fn nvidia_rows(out: &str, torch_info: Option<&TorchInfo>) -> Vec<Row> {
    let lines: Vec<&str> = out.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let first = lines.first().copied().unwrap_or("");
    let name = first.split(',').next().unwrap_or("").trim();
    let mut rows = vec![row("ok", "gpu", format!("{} x {name} (backend: cuda)", lines.len()))];
    for (i, line) in lines.iter().enumerate() {
        // ubs:ignore[rust.perf.string-alloc-in-loop] -- one report row per GPU; building the doctor table, not a hot path.
        rows.push(row("info", &format!("gpu {i}"), *line));
    }
    let driver = first.split(',').next_back().unwrap_or("").trim();
    rows.push(cuda_row(driver, torch_info));
    rows.push(row("info", "rocm", NO_ROCM));
    rows.push(row("info", "mps", NO_MPS));
    rows
}

/// Rows for a `ROCm` torch build: usable devices, or a warning when it finds none.
fn rocm_rows(info: &TorchInfo) -> Vec<Row> {
    let hip = info.hip_build.as_deref().unwrap_or("");
    let Some(first) = info.devices.first().filter(|_| info.cuda_available) else {
        return vec![
            row("warn", "gpu", "no GPU: CPU smoke only (ROCm torch build found no usable device)"),
            row("info", "cuda", "n/a (ROCm build)"),
            row(
                "warn",
                "rocm",
                format!("torch {} built for ROCm {hip}, NOT usable from torch", info.version),
            ),
            row("info", "mps", NO_MPS),
        ];
    };
    let memory = |i: usize| format_gib(info.device_memory_bytes.get(i).copied().flatten());
    let count = info.devices.len();
    let mut rows =
        vec![row("ok", "gpu", format!("{count} x {first} ({}, backend: rocm)", memory(0)))];
    for (i, device) in info.devices.iter().enumerate() {
        // ubs:ignore[rust.perf.string-alloc-in-loop] -- one report row per GPU; building the doctor table, not a hot path.
        rows.push(row("info", &format!("gpu {i}"), format!("{device}, {}, ROCm {hip}", memory(i))));
    }
    rows.push(row("info", "cuda", "n/a (ROCm build)"));
    rows.push(row(
        "ok",
        "rocm",
        format!("torch {} built for ROCm {hip}, usable ({count} device(s))", info.version),
    ));
    rows.push(row("info", "mps", NO_MPS));
    rows
}

/// Rows for Apple Silicon with a usable MPS backend.
fn mps_rows(host: &Host, info: &TorchInfo) -> Vec<Row> {
    let argv: Vec<String> =
        ["sysctl", "-n", "machdep.cpu.brand_string"].iter().map(|s| (*s).to_owned()).collect();
    let (rc, out) = (host.run)(&argv);
    let chip = if rc == 0 && !out.trim().is_empty() { out.trim() } else { "Apple Silicon" };
    let memory = match info.mps_recommended_max_bytes {
        Some(b) if b > 0 => format!("{} recommended max", format_gib(Some(b))),
        _ => "unified memory (size not exposed by torch)".to_owned(),
    };
    vec![
        row("ok", "gpu", format!("1 x {chip} ({memory}, backend: mps)")),
        row("info", "cuda", "n/a (Apple MPS)"),
        row("info", "rocm", NO_ROCM),
        row(
            "ok",
            "mps",
            format!("Apple MPS usable from torch {} ({chip}, {memory})", info.version),
        ),
    ]
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
                 only)",
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

/// Download throughput row, or an info row when `--bandwidth` was not requested.
fn bandwidth_row(host: &Host) -> Row {
    let name = "bandwidth".to_owned();
    let Some(run) = &host.bandwidth else {
        return Row {
            status: "info",
            name,
            detail: "not measured; run `ofl doctor --bandwidth` to time one bounded download"
                .to_owned(),
        };
    };
    let probe = run();
    let url = &probe.url;
    if let Some(error) = &probe.error {
        let moved =
            if probe.bytes == 0 { String::new() } else { format!(" after {} bytes", probe.bytes) };
        return Row {
            status: "warn",
            name,
            detail: format!("download failed{moved}: {error} (source {url})"),
        };
    }
    let Some(rate) = mb_per_second(probe.bytes, probe.seconds) else {
        return Row {
            status: "warn",
            name,
            detail: format!(
                "too little to time: {} bytes in {:.3} s (source {url})",
                probe.bytes, probe.seconds
            ),
        };
    };
    let stop = if probe.timed_out {
        format!("stopped at the {} s time limit", BANDWIDTH_DEADLINE.as_secs())
    } else {
        format!("read the {} MiB range", BANDWIDTH_CAP_BYTES / (1024 * 1024))
    };
    Row {
        status: "ok",
        name,
        detail: format!(
            "download {rate:.1} MB/s ({:.0} Mbit/s): {} bytes in {:.1} s from {url}; {stop}; \
             one connection, timed from the response headers; upload not measured",
            rate * 8.0,
            probe.bytes,
            probe.seconds
        ),
    }
}

/// Decimal megabytes per second, or None when nothing was moved or no time elapsed.
fn mb_per_second(bytes: u64, seconds: f64) -> Option<f64> {
    if bytes == 0 || seconds <= 0.0 {
        return None;
    }
    // Byte counts here are capped at 32 MiB, far below 2^52.
    #[expect(clippy::cast_precision_loss, reason = "byte counts are far below 2^52")]
    // ubs:ignore[rust.casts.as-casts] -- byte count to f64; capped at 32 MiB, no meaningful truncation.
    let megabytes = bytes as f64 / MEGABYTE;
    Some(megabytes / seconds)
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

/// Whether a body-read error is a timeout. `ureq` 3 reports its own `Error::Timeout`
/// (the global timeout) as an `io::Error` of kind `Other` wrapping the `ureq` error, so the
/// kind alone is not enough.
#[must_use]
pub fn is_timeout(e: &std::io::Error) -> bool {
    e.kind() == std::io::ErrorKind::TimedOut
        || e.get_ref()
            .and_then(|inner| inner.downcast_ref::<ureq::Error>())
            .is_some_and(|err| matches!(err, ureq::Error::Timeout(_)))
}

/// One bounded download of [`BANDWIDTH_URL`]: at most [`BANDWIDTH_CAP_BYTES`] and
/// [`BANDWIDTH_DEADLINE`] after the request starts. A stall past the deadline keeps the
/// bytes already received; no data at all is an error.
fn bandwidth_probe() -> BandwidthProbe {
    let start = Instant::now();
    let mut probe = BandwidthProbe {
        url: BANDWIDTH_URL.to_owned(),
        bytes: 0,
        seconds: 0.0,
        timed_out: false,
        error: None,
    };
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(BANDWIDTH_HARD_TIMEOUT))
        .build()
        .new_agent();
    let range = format!("bytes=0-{}", BANDWIDTH_CAP_BYTES - 1);
    let resp = match agent.get(BANDWIDTH_URL).header("Range", &range).call() {
        Ok(resp) => resp,
        Err(e) => {
            probe.error = Some(e.to_string());
            return probe;
        }
    };
    // `take` bounds the read even when the server ignores Range and sends the whole file.
    let mut reader = resp.into_body().into_reader().take(BANDWIDTH_CAP_BYTES);
    let mut buf = vec![0_u8; 64 * 1024];
    let body_start = Instant::now();
    loop {
        if start.elapsed() >= BANDWIDTH_DEADLINE {
            probe.timed_out = true;
            break;
        }
        match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => probe.bytes += u64::try_from(n).unwrap_or(0),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            // Only a timeout is a time-limited measurement; any other read error (a reset,
            // say) is a failure whatever the elapsed time, and must not read as a clean result.
            Err(e) if is_timeout(&e) => {
                probe.timed_out = true;
                break;
            }
            Err(e) => {
                probe.error = Some(e.to_string());
                break;
            }
        }
    }
    probe.seconds = body_start.elapsed().as_secs_f64();
    if probe.timed_out && probe.bytes == 0 {
        probe.error = Some(format!("no data received within {} s", BANDWIDTH_DEADLINE.as_secs()));
    }
    probe
}

/// Parse an HTTP Date header to unix seconds.
fn parse_http_date(text: &str) -> Option<i64> {
    let time = httpdate::parse_http_date(text).ok()?;
    let secs = time.duration_since(UNIX_EPOCH).ok()?.as_secs();
    i64::try_from(secs).ok()
}

/// Probe run in the smoke interpreter: torch version, CUDA/`ROCm` devices and MPS, as one JSON object.
const TORCH_PROBE: &str = r#"
import json, torch
cuda = bool(torch.cuda.is_available())
count = int(torch.cuda.device_count()) if cuda else 0
def memory(i):
    try:
        return int(torch.cuda.get_device_properties(i).total_memory)
    except RuntimeError:
        return None
mps = getattr(torch.backends, "mps", None)
mps_ok = bool(mps is not None and mps.is_available())
mps_max = None
if mps_ok and hasattr(torch.mps, "recommended_max_memory"):
    try:
        mps_max = int(torch.mps.recommended_max_memory())
    except RuntimeError:
        pass
print(json.dumps({
    "version": torch.__version__,
    "cuda": torch.version.cuda,
    "hip": getattr(torch.version, "hip", None),
    "cuda_available": cuda,
    "devices": [torch.cuda.get_device_name(i) for i in range(count)],
    "memory": [memory(i) for i in range(count)],
    "mps_available": mps_ok,
    "mps_max": mps_max,
}))
"#;

/// Torch build facts from the smoke interpreter.
fn torch_info(python: &str) -> Option<TorchInfo> {
    let argv = vec![python.to_owned(), "-c".to_owned(), TORCH_PROBE.to_owned()];
    let (rc, out) = run_captured(&argv, TORCH_TIMEOUT);
    if rc != 0 {
        return None;
    }
    parse_torch_info(&out)
}

/// Parse the [`TORCH_PROBE`] output.
#[must_use]
pub fn parse_torch_info(out: &str) -> Option<TorchInfo> {
    let v: serde_json::Value = serde_json::from_str(out).ok()?;
    let text = |key: &str| v.get(key).and_then(|x| x.as_str()).map(str::to_owned);
    let flag = |key: &str| v.get(key).and_then(serde_json::Value::as_bool).unwrap_or(false);
    let list = |key: &str| v.get(key).and_then(|x| x.as_array()).cloned().unwrap_or_default();
    Some(TorchInfo {
        version: text("version")?,
        cuda_build: text("cuda"),
        cuda_available: flag("cuda_available"),
        hip_build: text("hip"),
        mps_available: flag("mps_available"),
        devices: list("devices").iter().filter_map(|x| x.as_str().map(str::to_owned)).collect(),
        device_memory_bytes: list("memory").iter().map(serde_json::Value::as_u64).collect(),
        mps_recommended_max_bytes: v.get("mps_max").and_then(serde_json::Value::as_u64),
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
