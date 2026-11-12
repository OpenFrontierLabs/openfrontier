#![expect(clippy::expect_used, reason = "test helper failures panic")]
//! Shared helpers for the integration tests: the real repo root.

use std::path::{Path, PathBuf};

use ofl::repo::find_root;

/// The checkout root these tests run in.
pub fn root() -> PathBuf {
    find_root(Some(Path::new(env!("CARGO_MANIFEST_DIR")))).expect("tests run inside the checkout")
}
