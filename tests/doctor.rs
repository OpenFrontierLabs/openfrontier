#![cfg(test)]
#![expect(clippy::unwrap_used, reason = "test assertions panic on failure")]
//! Doctor report tests with fake machine probes.

use std::path::Path;

use ofl::doctor::{
    BandwidthProbe, Host, HttpsProbe, Row, TorchInfo, collect, exit_code, is_timeout,
    parse_torch_info,
};

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
            Some(TorchInfo { version: "2.13.0+cpu".to_owned(), ..TorchInfo::default() })
        }),
        version: Box::new(|_: &str| Some("1.0".to_owned())),
        now: Box::new(|| NOW),
        bandwidth: None,
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
    for name in ["cuda", "rocm", "mps"] {
        assert_eq!(table[name].status, "info", "{name}");
    }
    assert!(table["rocm"].detail.contains("not present"));
    assert!(table["mps"].detail.contains("not present"));
    assert_eq!(exit_code(&rows), 0);
}

#[test]
fn mps_device_is_reported_when_available() {
    let mut host = fake_host();
    host.run = Box::new(|argv: &[String]| match argv.first().map(String::as_str) {
        Some("sysctl") => (0, "Apple M3 Max".to_owned()),
        Some("uv") => (0, "uv 0.0.0".to_owned()),
        _ => (127, String::new()),
    });
    host.torch = Box::new(|| {
        Some(TorchInfo {
            version: "2.13.0".to_owned(),
            mps_available: true,
            mps_recommended_max_bytes: Some(48 * GIB),
            ..TorchInfo::default()
        })
    });
    let rows = collect(&host, Path::new("."));
    let table = by_name(&rows);
    assert_eq!(table["gpu"].status, "ok");
    assert!(table["gpu"].detail.contains("Apple M3 Max"));
    assert!(table["gpu"].detail.contains("48.0 GiB recommended max"));
    assert!(table["gpu"].detail.contains("backend: mps"));
    assert_eq!(table["mps"].status, "ok");
    assert!(table["mps"].detail.contains("Apple MPS usable"));
    assert_eq!(table["cuda"].status, "info");
    assert_eq!(table["rocm"].status, "info");
    assert_eq!(exit_code(&rows), 0);
}

#[test]
fn rocm_device_is_reported_when_available() {
    let mut host = fake_host();
    host.torch = Box::new(|| {
        Some(TorchInfo {
            version: "2.13.0+rocm6.2".to_owned(),
            cuda_available: true,
            hip_build: Some("6.2.41133".to_owned()),
            devices: vec!["AMD Instinct MI300X".to_owned()],
            device_memory_bytes: vec![Some(192 * GIB)],
            ..TorchInfo::default()
        })
    });
    let rows = collect(&host, Path::new("."));
    let table = by_name(&rows);
    assert_eq!(table["gpu"].status, "ok");
    assert!(table["gpu"].detail.contains("1 x AMD Instinct MI300X"));
    assert!(table["gpu"].detail.contains("192.0 GiB"));
    assert!(table["gpu"].detail.contains("backend: rocm"));
    assert!(table["gpu 0"].detail.contains("ROCm 6.2.41133"));
    assert_eq!(table["rocm"].status, "ok");
    assert!(table["rocm"].detail.contains("usable"));
    assert_eq!(table["cuda"].status, "info");
    assert_eq!(table["mps"].status, "info");
    assert_eq!(exit_code(&rows), 0);
}

#[test]
fn rocm_build_without_a_usable_device_warns() {
    let mut host = fake_host();
    host.torch = Box::new(|| {
        Some(TorchInfo {
            version: "2.13.0+rocm6.2".to_owned(),
            hip_build: Some("6.2.41133".to_owned()),
            ..TorchInfo::default()
        })
    });
    let rows = collect(&host, Path::new("."));
    let table = by_name(&rows);
    assert_eq!(table["gpu"].status, "warn");
    assert_eq!(table["rocm"].status, "warn");
    assert!(table["rocm"].detail.contains("NOT usable"));
    assert_eq!(exit_code(&rows), 0);
}

#[test]
fn torch_probe_output_parses_for_cuda_rocm_mps_and_cpu() {
    let cpu = r#"{"version":"2.13.0+cpu","cuda":null,"hip":null,"cuda_available":false,
        "devices":[],"memory":[],"mps_available":false,"mps_max":null}"#;
    let info = parse_torch_info(cpu).unwrap();
    assert!(!info.cuda_available && !info.mps_available && info.hip_build.is_none());

    let mps = r#"{"version":"2.13.0","cuda":null,"hip":null,"cuda_available":false,
        "devices":[],"memory":[],"mps_available":true,"mps_max":38654705664}"#;
    let info = parse_torch_info(mps).unwrap();
    assert!(info.mps_available);
    assert_eq!(info.mps_recommended_max_bytes, Some(36 * GIB));

    let rocm = r#"{"version":"2.13.0+rocm6.2","cuda":null,"hip":"6.2.41133","cuda_available":true,
        "devices":["AMD Radeon PRO W7900"],"memory":[51539607552],"mps_available":false,"mps_max":null}"#;
    let info = parse_torch_info(rocm).unwrap();
    assert_eq!(info.hip_build.as_deref(), Some("6.2.41133"));
    assert_eq!(info.devices, ["AMD Radeon PRO W7900"]);
    assert_eq!(info.device_memory_bytes, [Some(48 * GIB)]);

    let cuda = r#"{"version":"2.13.0+cu124","cuda":"12.4","hip":null,"cuda_available":true,
        "devices":["NVIDIA H100"],"memory":[null],"mps_available":false,"mps_max":null}"#;
    let info = parse_torch_info(cuda).unwrap();
    assert_eq!(info.cuda_build.as_deref(), Some("12.4"));
    assert_eq!(info.device_memory_bytes, [None]);
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
            ..TorchInfo::default()
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
fn latency_is_the_median_of_samples() {
    let samples = std::cell::RefCell::new([0.5, 0.1, 0.2, 0.5, 0.1, 0.2].into_iter());
    let mut host = fake_host();
    host.https = Box::new(move |_: &str| {
        let seconds = samples.borrow_mut().next().unwrap_or(0.1);
        HttpsProbe { ok: true, seconds, server_date: Some(NOW), error: String::new() }
    });
    let rows = collect(&host, Path::new("."));
    let row = &by_name(&rows)["net huggingface.co"];
    assert!(row.detail.contains("200 ms"));
}

const SOURCE: &str = "https://example.invalid/object.bin";

/// A host whose bandwidth probe returns `probe` instead of touching the network.
fn bandwidth_host(probe: BandwidthProbe) -> Host {
    let mut host = fake_host();
    host.bandwidth = Some(Box::new(move || probe.clone()));
    host
}

fn probe(bytes: u64, seconds: f64) -> BandwidthProbe {
    BandwidthProbe { url: SOURCE.to_owned(), bytes, seconds, timed_out: false, error: None }
}

#[test]
fn bandwidth_is_off_by_default_and_says_how_to_turn_it_on() {
    let rows = collect(&fake_host(), Path::new("."));
    let row = &by_name(&rows)["bandwidth"];
    assert_eq!(row.status, "info");
    assert!(row.detail.contains("not measured"));
    assert!(row.detail.contains("--bandwidth"));
}

#[test]
fn throughput_is_decimal_megabytes_per_second_over_the_body_time() {
    // 32 MiB in 4 s: 8.39 MB/s decimal (a MiB/s figure would read 8.0), 67 Mbit/s.
    let rows = collect(&bandwidth_host(probe(33_554_432, 4.0)), Path::new("."));
    let row = &by_name(&rows)["bandwidth"];
    assert_eq!(row.status, "ok");
    assert!(row.detail.starts_with("download 8.4 MB/s (67 Mbit/s): 33554432 bytes in 4.0 s"));
    assert!(row.detail.contains(SOURCE));
    assert!(row.detail.contains("upload not measured"));
    assert_eq!(exit_code(&rows), 0);
}

#[test]
fn hitting_the_time_limit_keeps_the_partial_measurement() {
    let mut slow = probe(4_000_000, 8.0);
    slow.timed_out = true;
    let rows = collect(&bandwidth_host(slow), Path::new("."));
    let row = &by_name(&rows)["bandwidth"];
    assert_eq!(row.status, "ok");
    assert!(row.detail.starts_with("download 0.5 MB/s (4 Mbit/s): 4000000 bytes in 8.0 s"));
    assert!(row.detail.contains("time limit"));
}

#[test]
fn download_failure_is_a_warning_with_the_reason_and_source() {
    let mut failed = probe(0, 0.0);
    failed.error = Some("no data received within 8 s".to_owned());
    let rows = collect(&bandwidth_host(failed), Path::new("."));
    let row = &by_name(&rows)["bandwidth"];
    assert_eq!(row.status, "warn");
    assert!(row.detail.contains("download failed: no data received within 8 s"));
    assert!(row.detail.contains(SOURCE));
    assert!(!row.detail.contains("MB/s"));
    // A failed bandwidth check must not turn a reachable machine into a hard failure.
    assert_eq!(exit_code(&rows), 0);
}

#[test]
fn failure_midway_reports_bytes_moved_and_no_throughput() {
    let mut cut = probe(1_048_576, 0.7);
    cut.error = Some("connection reset".to_owned());
    let rows = collect(&bandwidth_host(cut), Path::new("."));
    let row = &by_name(&rows)["bandwidth"];
    assert_eq!(row.status, "warn");
    assert!(row.detail.contains("download failed after 1048576 bytes: connection reset"));
    assert!(!row.detail.contains("MB/s"));
}

#[test]
fn an_unmeasurable_download_never_divides_by_zero() {
    for (bytes, seconds) in [(1_000_000, 0.0), (0, 2.0)] {
        let rows = collect(&bandwidth_host(probe(bytes, seconds)), Path::new("."));
        let row = &by_name(&rows)["bandwidth"];
        assert_eq!(row.status, "warn", "{bytes} bytes in {seconds} s");
        assert!(row.detail.contains("too little to time"));
        assert!(!row.detail.contains("inf"));
    }
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

#[test]
fn only_timeouts_count_as_hitting_the_bandwidth_time_limit() {
    use std::io::{Error, ErrorKind};
    // ureq 3 wraps its global timeout as io::ErrorKind::Other around the ureq error.
    assert!(is_timeout(&ureq::Error::Timeout(ureq::Timeout::Global).into_io()));
    assert!(is_timeout(&Error::from(ErrorKind::TimedOut)));
    // A reset, an unrelated wrapped ureq error and a plain Other are failures, not time limits.
    assert!(!is_timeout(&Error::from(ErrorKind::ConnectionReset)));
    assert!(!is_timeout(&ureq::Error::ConnectionFailed.into_io()));
    assert!(!is_timeout(&Error::other("boom")));
}
