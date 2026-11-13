#![cfg(test)]
#![expect(clippy::unwrap_used, reason = "test assertions panic on failure")]
//! Bounded child-run tests: the pipe-drain wait must not deadlock on large output.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use ofl::repo::{run_captured, wait_piped};

/// A child that writes far more than the ~64 KiB pipe buffer to stderr must
/// still exit promptly: the drain threads keep the pipe empty while we poll.
/// Before the drain fix this deadlocked and surfaced as a spurious timeout.
#[test]
fn large_stderr_does_not_deadlock_the_bounded_wait() {
    // 200_000 bytes, well past any OS pipe buffer.
    let script = "import sys; sys.stderr.write('x' * 200000); sys.stderr.flush()";
    let child = Command::new("python3")
        .args(["-c", script])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let out = wait_piped(child, Duration::from_secs(30)).unwrap();
    assert!(!out.timed_out, "child deadlocked against a full stderr pipe");
    assert_eq!(out.code, 0);
    assert_eq!(out.stderr.len(), 200_000);
}

/// Both pipes drain concurrently, and the exit code is preserved.
#[test]
fn stdout_and_stderr_are_both_captured() {
    let script = "import sys; sys.stdout.write('out' * 50000); sys.stderr.write('err' * 50000)";
    let child = Command::new("python3")
        .args(["-c", script])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let out = wait_piped(child, Duration::from_secs(30)).unwrap();
    assert!(!out.timed_out);
    assert_eq!(out.code, 0);
    assert_eq!(out.stdout.len(), 150_000);
    assert_eq!(out.stderr.len(), 150_000);
}

/// A child that outlives the limit is killed and reported as timed out.
#[test]
fn slow_child_is_killed_at_the_limit() {
    let child = Command::new("python3")
        .args(["-c", "import time; time.sleep(60)"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let out = wait_piped(child, Duration::from_millis(300)).unwrap();
    assert!(out.timed_out);
}

/// A grandchild that outlives the killed parent keeps the pipe open. The wait
/// must still return at the limit, not block joining the drain threads for the
/// grandchild's full lifetime. With the old join-after-kill this took ~30 s.
#[test]
fn grandchild_holding_the_pipe_does_not_outlast_the_limit() {
    let child = Command::new("sh")
        .args(["-c", "sleep 30 & sleep 30"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let start = Instant::now();
    let out = wait_piped(child, Duration::from_secs(1)).unwrap();
    let elapsed = start.elapsed();
    assert!(out.timed_out);
    assert!(elapsed < Duration::from_secs(10), "wait blocked on the grandchild: {elapsed:?}");
}

/// `run_captured` drains stdout while waiting, so a large stdout cannot deadlock
/// the helper (the same poll-then-read shape the drain fix removed).
#[test]
fn run_captured_drains_large_stdout() {
    let argv = vec![
        "python3".to_owned(),
        "-c".to_owned(),
        "import sys; sys.stdout.write('y' * 200000)".to_owned(),
    ];
    let (code, stdout) = run_captured(&argv, Duration::from_secs(30));
    assert_eq!(code, 0);
    assert_eq!(stdout.len(), 200_000);
}
