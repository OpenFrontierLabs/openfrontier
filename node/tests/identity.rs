#![expect(clippy::expect_used, reason = "test assertions panic on failure")]
//! Library-level tests for the key file: concurrent creation. No network.

use std::thread;

use ofl_node::identity::load_or_create;
use tempfile::TempDir;

#[test]
fn concurrent_first_runs_in_one_process_all_get_the_same_key() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("a/b/device.key");
    let peers: Vec<String> = thread::scope(|scope| {
        // All sixteen are started before any is joined, so they really race.
        let mut handles = Vec::new();
        for _ in 0..16 {
            handles.push(scope.spawn(|| {
                let (keypair, _) = load_or_create(&path).expect("load_or_create");
                keypair.public().to_peer_id().to_string()
            }));
        }
        handles.into_iter().map(|h| h.join().expect("thread")).collect()
    });
    assert!(
        peers.windows(2).all(|w| w[0] == w[1]),
        "racing first runs made different keys: {peers:?}"
    );
    let leftovers: Vec<_> = std::fs::read_dir(path.parent().expect("parent"))
        .expect("read_dir")
        .map(|e| e.expect("entry").file_name())
        .collect();
    assert_eq!(leftovers, ["device.key"], "staging files were left behind");
}
