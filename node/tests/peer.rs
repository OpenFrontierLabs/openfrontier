#![expect(clippy::expect_used, reason = "test assertions panic on failure")]
//! Black-box tests of `ofl-node peer`: real processes on 127.0.0.1, no internet. Every wait is
//! bounded and every child is killed when its handle drops, so a failure never hangs or leaks.

use std::cell::Cell;
use std::io::{BufRead, BufReader, Read};
use std::net::UdpSocket;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use tempfile::TempDir;

const WAIT: Duration = Duration::from_secs(60);

/// A running `ofl-node peer`. Dropping it kills the process.
struct PeerProc {
    child: Child,
    lines: Receiver<String>,
    /// Everything printed so far, stdout and stderr, for failure messages.
    log: Vec<String>,
    _key_dir: TempDir,
}

impl PeerProc {
    fn start(args: &[&str]) -> Self {
        Self::start_in(TempDir::new().expect("tempdir"), args)
    }

    /// Start with the device key at `key_dir/device.key` (made if absent).
    fn start_in(key_dir: TempDir, args: &[&str]) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_ofl-node"))
            .arg("--key-file")
            .arg(key_dir.path().join("device.key"))
            .arg("peer")
            .args(["--interval-secs", "1"])
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn ofl-node peer");
        let (tx, lines) = mpsc::channel();
        let stdout = child.stdout.take().expect("stdout pipe");
        let stderr = child.stderr.take().expect("stderr pipe");
        forward(stdout, "", tx.clone());
        forward(stderr, "stderr: ", tx);
        Self { child, lines, log: Vec::new(), _key_dir: key_dir }
    }

    /// The first output line, from now or earlier, for which `pick` returns a value. Fails with
    /// the full log if none arrives in time or the process exits first.
    fn wait_for<T>(&mut self, what: &str, pick: impl Fn(&str) -> Option<T>) -> T {
        let found = self.poll(&pick);
        assert!(found.is_some(), "did not see {what} within {WAIT:?} (or the peer exited)\n{self}");
        found.expect("asserted just above")
    }

    fn poll<T>(&mut self, pick: &impl Fn(&str) -> Option<T>) -> Option<T> {
        if let Some(found) = self.log.iter().find_map(|l| pick(l)) {
            return Some(found);
        }
        let deadline = Instant::now() + WAIT;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            // A timeout or a closed channel (the process exited) both end the wait.
            let line = self.lines.recv_timeout(left).ok()?;
            let found = pick(&line);
            self.log.push(line);
            if found.is_some() {
                return found;
            }
        }
    }

    fn peer_id(&mut self) -> String {
        self.wait_for("peer_id line", |l| l.strip_prefix("peer_id ").map(str::to_owned))
    }

    /// A `listening` multiaddr on loopback; it ends in `/p2p/<id>`.
    fn loopback_addr(&mut self) -> String {
        self.wait_for("a loopback listening line", |l| {
            l.strip_prefix("listening ")
                .filter(|a| a.starts_with("/ip4/127.0.0.1/"))
                .map(str::to_owned)
        })
    }

    /// Wait until a listing shows `other` with the given `connected`/`routing_table` flags.
    fn wait_for_listed(&mut self, other: &str, connected: bool, routing_table: bool) {
        let want = format!("peer {other} connected={connected} routing_table={routing_table}");
        self.wait_for(&want, |l| (l == want).then_some(()));
    }
}

impl std::fmt::Display for PeerProc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "--- peer output ---\n{}", self.log.join("\n"))
    }
}

impl Drop for PeerProc {
    fn drop(&mut self) {
        drop(self.child.kill());
        drop(self.child.wait());
    }
}

fn forward(stream: impl Read + Send + 'static, prefix: &'static str, tx: mpsc::Sender<String>) {
    thread::spawn(move || {
        for line in BufReader::new(stream).lines().map_while(std::result::Result::ok) {
            if tx.send(format!("{prefix}{line}")).is_err() {
                return;
            }
        }
    });
}

/// Start a listener, then a joiner that bootstraps from the listener's printed multiaddr.
/// Returns `(listener, listener id, joiner, joiner id)`.
fn pair(listener_args: &[&str], joiner_args: &[&str]) -> (PeerProc, String, PeerProc, String) {
    let mut listener =
        PeerProc::start(&[&["--listen", "--host", "127.0.0.1"], listener_args].concat());
    let listener_id = listener.peer_id();
    let addr = listener.loopback_addr();
    assert!(addr.ends_with(&format!("/p2p/{listener_id}")), "{addr} is not the listener's id");

    let mut joiner = PeerProc::start(&[&["--bootstrap", &addr], joiner_args].concat());
    let joiner_id = joiner.peer_id();
    assert_ne!(listener_id, joiner_id, "two peers share an identity");
    (listener, listener_id, joiner, joiner_id)
}

#[test]
fn quic_joiner_and_bootstrap_list_each_other() {
    let (mut listener, listener_id, mut joiner, joiner_id) = pair(&[], &[]);
    // The default transport is QUIC: the address the listener advertises says so.
    let quic = listener.loopback_addr();
    assert!(quic.contains("/udp/") && quic.contains("/quic-v1/"), "not QUIC: {quic}");
    // The joiner holds the bootstrap peer in its routing table; the bootstrap peer sees a
    // client-mode joiner as connected only (Kademlia admits servers).
    joiner.wait_for_listed(&listener_id, true, true);
    listener.wait_for_listed(&joiner_id, true, false);
}

#[test]
fn tcp_joiner_and_bootstrap_list_each_other() {
    let (mut listener, listener_id, mut joiner, joiner_id) = pair(&["--tcp"], &["--tcp"]);
    let tcp = listener.loopback_addr();
    assert!(tcp.contains("/tcp/") && !tcp.contains("quic"), "not TCP: {tcp}");
    joiner.wait_for_listed(&listener_id, true, true);
    listener.wait_for_listed(&joiner_id, true, false);
}

#[test]
fn listening_joiner_enters_the_bootstraps_routing_table() {
    // Both peers are Kademlia servers, so Identify hands the bootstrap peer a dialable address
    // for the joiner and Kademlia admits it.
    let (mut listener, listener_id, mut joiner, joiner_id) =
        pair(&[], &["--listen", "--host", "127.0.0.1"]);
    listener.wait_for_listed(&joiner_id, true, true);
    joiner.wait_for_listed(&listener_id, true, true);
}

#[test]
fn quic_peer_cannot_use_a_tcp_bootstrap_address() {
    let mut listener = PeerProc::start(&["--listen", "--host", "127.0.0.1", "--tcp"]);
    let listener_id = listener.peer_id();
    let addr = listener.loopback_addr();
    let mut joiner = PeerProc::start(&["--bootstrap", &addr]);
    joiner.wait_for("a dial error", |l| l.starts_with("stderr: error: dial").then_some(()));
    // Kademlia keeps the bootstrap address in its table although it never connected, and the
    // listing says so: in the table, not connected.
    joiner.wait_for_listed(&listener_id, false, true);
    // Let a few more listings pass, then check none ever reported a connection.
    let listings = Cell::new(0);
    joiner.wait_for("three listings", |l| {
        listings.set(listings.get() + usize::from(l.starts_with("peers ")));
        (listings.get() >= 3).then_some(())
    });
    assert!(
        joiner.log.iter().all(|l| !l.contains("connected=true")),
        "a QUIC peer connected over TCP?\n{joiner}"
    );
}

#[test]
fn joiner_started_before_its_bootstrap_peer_connects_once_it_is_up() {
    // Fix the listener's identity and port first, so the joiner can be given an address nobody
    // is listening on yet.
    let key_dir = TempDir::new().expect("tempdir");
    let keygen = Command::new(env!("CARGO_BIN_EXE_ofl-node"))
        .arg("--key-file")
        .arg(key_dir.path().join("device.key"))
        .arg("keygen")
        .output()
        .expect("spawn ofl-node keygen");
    let listener_id = String::from_utf8(keygen.stdout)
        .expect("utf8 stdout")
        .lines()
        .find_map(|l| l.strip_prefix("peer_id ").map(str::to_owned))
        .expect("keygen peer_id line");
    let port = UdpSocket::bind("127.0.0.1:0").expect("bind").local_addr().expect("addr").port();
    let addr = format!("/ip4/127.0.0.1/udp/{port}/quic-v1/p2p/{listener_id}");

    let mut joiner = PeerProc::start(&["--bootstrap", &addr]);
    let joiner_id = joiner.peer_id();
    joiner
        .wait_for("the first dial to fail", |l| l.starts_with("stderr: error: dial").then_some(()));
    let port = port.to_string();
    let mut listener =
        PeerProc::start_in(key_dir, &["--listen", "--host", "127.0.0.1", "--port", &port]);
    assert_eq!(listener.peer_id(), listener_id);

    joiner.wait_for_listed(&listener_id, true, true);
    listener.wait_for_listed(&joiner_id, true, false);
}

fn run_failing(args: &[&str], key: &Path) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_ofl-node"))
        .arg("--key-file")
        .arg(key)
        .arg("peer")
        .args(args)
        .output()
        .expect("spawn ofl-node");
    assert!(!out.status.success(), "{args:?} was accepted");
    String::from_utf8(out.stderr).expect("utf8 stderr")
}

#[test]
fn bad_arguments_are_refused_before_any_network_use() {
    let dir = TempDir::new().expect("tempdir");
    let key = dir.path().join("device.key");
    let no_id = run_failing(&["--bootstrap", "/ip4/127.0.0.1/udp/1/quic-v1"], &key);
    assert!(no_id.contains("/p2p/"), "{no_id}");
    let id_only = run_failing(
        &["--bootstrap", "/p2p/12D3KooWD3eckifWpRn9wQpMG9R9hX3sD158z7EqHWmweQAtpuZd"],
        &key,
    );
    assert!(id_only.contains("no address"), "{id_only}");
    let neither = run_failing(&[], &key);
    assert!(neither.contains("--listen") && neither.contains("--bootstrap"), "{neither}");
    assert!(!key.exists(), "a refused command still created a key");
}
