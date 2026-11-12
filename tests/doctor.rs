//! Doctor report tests with fake machine probes.

use std::path::Path;

use ofl::doctor::{Host, HttpsProbe, Row, TorchInfo, collect, exit_code};

/// A fixed "now" for clock-skew tests: 2026-11-12T12:00:00Z.
const NOW: i64 = 1_794_484_800;
const GIB: u64 = 1 << 30;

fn fake_host() -> Host {
    Host {
        python: (3, 12, 0),
        run: Box::new(|argv: &[String]| {
            if argv.first().map(String::as_str) == Some("uv") {
                (0, "uv 0.0.0".to_owned())
            } else {
                (127, String::new())
            }
        }),
        disk_free_bytes: Box::new(|_: &Path| 500 * GIB),
        dns: Box::new(|_: &str| Some(0.004)),
        https: Box::new(|_: &str| HttpsProbe {
            ok: true,
            seconds: 0.12,
            server_date: Some(NOW),
            error: String::new(),
        }),
        torch: Box::new(|| {
            Some(TorchInfo {
                version: "2.13.0+cpu".to_owned(),
                cuda_build: None,
                cuda_available: false,
            })
        }),
        version: Box::new(|_: &str| Some("1.0".to_owned())),
        now: Box::new(|| NOW),
    }
}

fn by_name(rows: &[Row]) -> std::collections::HashMap<String, Row> {
    rows.iter().map(|r| (r.name.clone(), r.clone())).collect()
}

#[test]
fn no_gpu_machine_warns_and_exits_zero() {
    let rows = collect(&fake_host(), Path::new("."));
    let table = by_name(&rows);
    assert_eq!(table["gpu"].status, "warn");
    assert!(table["gpu"].detail.contains("CPU smoke only"));
    assert_eq!(exit_code(&rows), 0);
}

#[test]
fn gpu_with_cpu_only_torch_is_informational() {
    let mut host = fake_host();
    host.run = Box::new(|argv: &[String]| {
        if argv.first().map(String::as_str) == Some("nvidia-smi") {
            (0, "NVIDIA H100 80GB HBM3, 81559 MiB, 560.35.03".to_owned())
        } else {
            (0, "uv 0.0.0".to_owned())
        }
    });
    let rows = collect(&host, Path::new("."));
    let table = by_name(&rows);
    assert_eq!(table["gpu"].status, "ok");
    assert!(table["gpu"].detail.starts_with("1 x NVIDIA H100"));
    assert_eq!(table["cuda"].status, "info");
    assert!(table["cuda"].detail.contains("560.35.03"));
    assert!(table["cuda"].detail.contains("CPU build"));
}

#[test]
fn gpu_present_but_torch_cannot_use_it_warns() {
    let mut host = fake_host();
    host.run = Box::new(|argv: &[String]| {
        if argv.first().map(String::as_str) == Some("nvidia-smi") {
            (0, "NVIDIA H100 80GB HBM3, 81559 MiB, 560.35.03".to_owned())
        } else {
            (0, "uv 0.0.0".to_owned())
        }
    });
    host.torch = Box::new(|| {
        Some(TorchInfo {
            version: "2.13.0+cu130".to_owned(),
            cuda_build: Some("13.0".to_owned()),
            cuda_available: false,
        })
    });
    let rows = collect(&host, Path::new("."));
    let cuda = &by_name(&rows)["cuda"];
    assert_eq!(cuda.status, "warn");
    assert!(cuda.detail.contains("13.0"));
    assert_eq!(exit_code(&rows), 0);
}

#[test]
fn old_python_is_a_hard_failure() {
    let mut host = fake_host();
    host.python = (3, 10, 9);
    let rows = collect(&host, Path::new("."));
    assert_eq!(by_name(&rows)["python"].status, "FAIL");
    assert_eq!(exit_code(&rows), 1);
}

#[test]
fn missing_torch_is_a_hard_failure() {
    let mut host = fake_host();
    host.torch = Box::new(|| None);
    let rows = collect(&host, Path::new("."));
    assert_eq!(by_name(&rows)["torch"].status, "FAIL");
    assert_eq!(exit_code(&rows), 1);
}

#[test]
fn disk_thresholds() {
    let mut host = fake_host();
    host.disk_free_bytes = Box::new(|_: &Path| GIB);
    assert_eq!(by_name(&collect(&host, Path::new(".")))["disk"].status, "FAIL");
    host.disk_free_bytes = Box::new(|_: &Path| 10 * GIB);
    assert_eq!(by_name(&collect(&host, Path::new(".")))["disk"].status, "warn");
}

#[test]
fn unreachable_network_is_a_hard_failure_per_host() {
    let mut host = fake_host();
    host.dns = Box::new(|_: &str| None);
    let rows = collect(&host, Path::new("."));
    let table = by_name(&rows);
    assert_eq!(table["net github.com"].status, "FAIL");
    assert_eq!(table["net huggingface.co"].status, "FAIL");
    assert_eq!(exit_code(&rows), 1);
}

#[test]
fn https_failure_reports_the_error() {
    let mut host = fake_host();
    host.https = Box::new(|_: &str| HttpsProbe {
        ok: false,
        seconds: 8.0,
        server_date: None,
        error: "timed out".to_owned(),
    });
    let rows = collect(&host, Path::new("."));
    let row = &by_name(&rows)["net github.com"];
    assert_eq!(row.status, "FAIL");
    assert!(row.detail.contains("timed out"));
}

#[test]
fn latency_is_the_median_of_samples_and_bandwidth_is_not_claimed() {
    let samples = std::cell::RefCell::new([0.5, 0.1, 0.2, 0.5, 0.1, 0.2].into_iter());
    let mut host = fake_host();
    host.https = Box::new(move |_: &str| {
        let seconds = samples.borrow_mut().next().unwrap_or(0.1);
        HttpsProbe { ok: true, seconds, server_date: Some(NOW), error: String::new() }
    });
    let rows = collect(&host, Path::new("."));
    let row = &by_name(&rows)["net huggingface.co"];
    assert!(row.detail.contains("200 ms"));
    assert!(row.detail.contains("bandwidth not measured"));
}

#[test]
fn clock_skew_on_every_host_warns() {
    let mut host = fake_host();
    host.https = Box::new(|_: &str| HttpsProbe {
        ok: true,
        seconds: 0.1,
        server_date: Some(NOW - 300),
        error: String::new(),
    });
    let rows = collect(&host, Path::new("."));
    let clock = &by_name(&rows)["clock"];
    assert_eq!(clock.status, "warn");
    assert!(clock.detail.contains("+300.0 s"));
}

#[test]
fn one_stale_cached_date_does_not_trigger_a_clock_warning() {
    let mut host = fake_host();
    host.https = Box::new(|h: &str| {
        let date = if h == "huggingface.co" { NOW - 300 } else { NOW };
        HttpsProbe { ok: true, seconds: 0.1, server_date: Some(date), error: String::new() }
    });
    let rows = collect(&host, Path::new("."));
    let clock = &by_name(&rows)["clock"];
    assert_eq!(clock.status, "ok");
    assert!(clock.detail.contains("+0.0 s"));
    assert_eq!(by_name(&collect(&fake_host(), Path::new(".")))["clock"].status, "ok");
}
