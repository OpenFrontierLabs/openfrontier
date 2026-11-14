#![expect(clippy::expect_used, reason = "test assertions panic on failure")]
//! Black-box tests: the real `ofl-node` binary against a temp key file. No network.

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

use ofl_node::envelope::SignedReport;
use tempfile::TempDir;

fn node(key: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ofl-node"))
        .arg("--key-file")
        .arg(key)
        .args(args)
        .output()
        .expect("spawn ofl-node")
}

fn stdout(out: &Output) -> String {
    String::from_utf8(out.stdout.clone()).expect("utf8 stdout")
}

fn stderr(out: &Output) -> String {
    String::from_utf8(out.stderr.clone()).expect("utf8 stderr")
}

fn signed(key: &Path) -> SignedReport {
    let out = node(key, &["sign"]);
    assert!(out.status.success(), "{}", stderr(&out));
    serde_json::from_str(&stdout(&out)).expect("envelope json")
}

fn verify(envelope: &SignedReport, dir: &TempDir) -> Output {
    let file = dir.path().join("envelope.json");
    fs::write(&file, serde_json::to_string(envelope).expect("serialize")).expect("write");
    node(&dir.path().join("unused.key"), &["verify", file.to_str().expect("utf8 path")])
}

#[test]
fn keygen_is_idempotent_and_keeps_the_same_key() {
    let dir = TempDir::new().expect("tempdir");
    let key = dir.path().join("nested/device.key");
    let first = node(&key, &["keygen"]);
    assert!(first.status.success(), "{}", stderr(&first));
    assert!(stdout(&first).starts_with("created "));
    let bytes = fs::read(&key).expect("key file");

    let second = node(&key, &["keygen"]);
    assert!(second.status.success(), "{}", stderr(&second));
    assert!(stdout(&second).starts_with("exists "));
    assert_eq!(fs::read(&key).expect("key file"), bytes, "second keygen rewrote the key");
    let ids = |o: &Output| stdout(o).lines().find(|l| l.starts_with("peer_id ")).map(str::to_owned);
    assert_eq!(ids(&first), ids(&second));
    assert!(ids(&first).is_some());
}

#[test]
fn output_never_contains_the_secret_key() {
    let dir = TempDir::new().expect("tempdir");
    let key = dir.path().join("device.key");
    let secret_hex = {
        assert!(node(&key, &["keygen"]).status.success());
        // The Ed25519 secret seed is the 32 bytes after the 4-byte protobuf header.
        let bytes = fs::read(&key).expect("key file");
        assert_eq!(bytes[..4], [0x08, 0x01, 0x12, 0x40], "unexpected key file layout");
        assert_eq!(bytes.len(), 68);
        hex::encode(&bytes[4..36])
    };
    for args in [&["keygen"][..], &["sign"][..]] {
        let out = node(&key, args);
        assert!(!stdout(&out).contains(&secret_hex) && !stderr(&out).contains(&secret_hex));
    }
}

#[cfg(unix)]
#[test]
fn key_file_is_0600_and_loose_permissions_are_refused() {
    use std::os::unix::fs::PermissionsExt;
    let dir = TempDir::new().expect("tempdir");
    let key = dir.path().join("device.key");
    assert!(node(&key, &["keygen"]).status.success());
    let mode = |p: &Path| fs::metadata(p).expect("stat").permissions().mode() & 0o777;
    assert_eq!(mode(&key), 0o600);

    fs::set_permissions(&key, fs::Permissions::from_mode(0o644)).expect("chmod");
    for args in [&["keygen"][..], &["sign"][..]] {
        let out = node(&key, args);
        assert!(!out.status.success(), "{args:?} accepted a world-readable key");
        assert!(stderr(&out).contains("chmod 600"), "{}", stderr(&out));
    }
}

#[cfg(unix)]
#[test]
fn config_dir_is_created_private() {
    use std::os::unix::fs::PermissionsExt;
    let dir = TempDir::new().expect("tempdir");
    let key = dir.path().join("cfg/device.key");
    assert!(node(&key, &["keygen"]).status.success());
    let mode = fs::metadata(key.parent().expect("parent")).expect("stat").permissions().mode();
    assert_eq!(mode & 0o777, 0o700);
}

#[test]
fn corrupt_key_file_is_an_error_and_is_not_replaced() {
    let dir = TempDir::new().expect("tempdir");
    let key = dir.path().join("device.key");
    assert!(node(&key, &["keygen"]).status.success());
    let mut bytes = fs::read(&key).expect("key file");
    bytes.truncate(10);
    fs::write(&key, &bytes).expect("corrupt");
    let out = node(&key, &["sign"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("corrupt"), "{}", stderr(&out));
    assert_eq!(fs::read(&key).expect("key file"), bytes, "a corrupt key was overwritten");
}

#[test]
fn sign_then_verify_round_trips_and_reports_the_signing_peer() {
    let dir = TempDir::new().expect("tempdir");
    let key = dir.path().join("device.key");
    let keygen = node(&key, &["keygen"]);
    let peer = stdout(&keygen)
        .lines()
        .find_map(|l| l.strip_prefix("peer_id ").map(str::to_owned))
        .expect("peer_id line");
    let envelope = signed(&key);
    let out = verify(&envelope, &dir);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(stdout(&out).starts_with(&format!("ok peer_id {peer}\n")), "{}", stdout(&out));
}

#[test]
fn tampered_report_fails_verification() {
    let dir = TempDir::new().expect("tempdir");
    let key = dir.path().join("device.key");
    let mut envelope = signed(&key);
    assert!(envelope.report.contains("\"os\":"));
    envelope.report = envelope.report.replacen("\"cpu_threads\":", "\"cpu_threads\":9", 1);
    let out = verify(&envelope, &dir);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("signature does not match"), "{}", stderr(&out));
}

#[test]
fn reserialised_report_with_different_whitespace_fails_because_exact_bytes_are_signed() {
    let dir = TempDir::new().expect("tempdir");
    let key = dir.path().join("device.key");
    let mut envelope = signed(&key);
    envelope.report = envelope.report.replacen(',', ", ", 1); // same JSON value, different bytes;
    assert!(!verify(&envelope, &dir).status.success());
}

#[test]
fn signature_from_another_key_fails_verification() {
    let dir = TempDir::new().expect("tempdir");
    let mut envelope = signed(&dir.path().join("a.key"));
    let other = signed(&dir.path().join("b.key"));
    assert_ne!(envelope.public_key, other.public_key);
    envelope.public_key = other.public_key;
    let out = verify(&envelope, &dir);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("signature does not match"), "{}", stderr(&out));
}

#[test]
fn malformed_envelope_is_a_clear_error() {
    let dir = TempDir::new().expect("tempdir");
    let file = dir.path().join("bad.json");
    fs::write(&file, "{\"report\": 1}").expect("write");
    let out = node(&dir.path().join("k"), &["verify", file.to_str().expect("utf8")]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("not a signed report envelope"), "{}", stderr(&out));
}
