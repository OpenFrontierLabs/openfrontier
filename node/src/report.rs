//! The hardware report: only fields this process can read without guessing.
//!
//! No GPU fields yet: `ofl doctor` gets them from `nvidia-smi` and torch, and a signed report
//! must not carry values this crate cannot read itself.

use serde::{Deserialize, Serialize};

use crate::error::Result;

pub const SCHEMA: &str = "ofl.node.hardware-report/v1";

/// Field order here is the order in the signed JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareReport {
    pub schema: String,
    pub generated_at_unix: u64,
    pub os: String,
    pub arch: String,
    /// Logical CPUs available to this process; `null` when the platform will not say.
    pub cpu_threads: Option<u64>,
    /// Total physical memory in bytes; `null` where this crate cannot read it (not Linux).
    pub memory_bytes: Option<u64>,
}

impl HardwareReport {
    /// Read the current machine.
    #[must_use]
    pub fn collect(now_unix: u64) -> Self {
        Self {
            schema: SCHEMA.to_owned(),
            generated_at_unix: now_unix,
            os: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
            cpu_threads: std::thread::available_parallelism().ok().map(|n| n.get() as u64),
            memory_bytes: std::fs::read_to_string("/proc/meminfo")
                .ok()
                .and_then(|text| parse_mem_total(&text)),
        }
    }

    /// Compact JSON, keys in declaration order. These exact bytes are what gets signed.
    ///
    /// # Errors
    ///
    /// [`NodeError::Json`]; not reachable for this struct, but never swallowed.
    pub fn to_signed_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }
}

/// `MemTotal` from `/proc/meminfo` text, converted from kB (really KiB) to bytes.
#[must_use]
pub fn parse_mem_total(meminfo: &str) -> Option<u64> {
    let line = meminfo.lines().find_map(|l| l.strip_prefix("MemTotal:"))?;
    let kib: u64 = line.trim().strip_suffix("kB")?.trim().parse().ok()?;
    kib.checked_mul(1024)
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "test assertions panic on failure")]

    use super::{HardwareReport, SCHEMA, parse_mem_total};

    #[test]
    fn mem_total_is_converted_from_kib_to_bytes() {
        let text = "MemTotal:       16384256 kB\nMemFree:         1000 kB\n";
        assert_eq!(parse_mem_total(text), Some(16_384_256 * 1024));
    }

    #[test]
    fn mem_total_absent_or_malformed_is_none_not_zero() {
        assert_eq!(parse_mem_total("MemFree: 5 kB\n"), None);
        assert_eq!(parse_mem_total("MemTotal: lots kB\n"), None);
        assert_eq!(parse_mem_total("MemTotal: 5 MB\n"), None);
    }

    #[test]
    fn signed_json_is_compact_in_field_order_and_round_trips() {
        let report = HardwareReport {
            schema: SCHEMA.to_owned(),
            generated_at_unix: 7,
            os: "linux".to_owned(),
            arch: "x86_64".to_owned(),
            cpu_threads: Some(8),
            memory_bytes: None,
        };
        let json = report.to_signed_json().expect("serialize");
        assert_eq!(
            json,
            r#"{"schema":"ofl.node.hardware-report/v1","generated_at_unix":7,"os":"linux","arch":"x86_64","cpu_threads":8,"memory_bytes":null}"#
        );
        assert_eq!(serde_json::from_str::<HardwareReport>(&json).expect("parse"), report);
    }
}
