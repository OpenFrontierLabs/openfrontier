//! One error type for the crate; every message says what to do next.

use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, NodeError>;

#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error("no config directory for this platform; pass --key-file")]
    NoConfigDir,
    #[error("{op} {path}: {source}")]
    Io { op: &'static str, path: PathBuf, source: std::io::Error },
    #[error(
        "key file {path} has permissions {mode:04o}; it must not be accessible by group or \
         others. Fix: chmod 600 {path}"
    )]
    KeyFilePermissions { path: PathBuf, mode: u32 },
    #[error("key file {path} is not a regular file")]
    KeyFileNotRegular { path: PathBuf },
    #[error(
        "key file {path} is corrupt or not an Ed25519 keypair ({reason}). Move it aside and run \
         keygen to make a new identity; the old peer id is then lost"
    )]
    KeyFileCorrupt { path: PathBuf, reason: String },
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("signing failed: {0}")]
    Sign(String),
    #[error("not a signed report envelope: {0}")]
    BadEnvelope(String),
    #[error("signature does not match the report and public key")]
    BadSignature,
    #[error(
        "bootstrap address {addr}: {reason}. Expected a full multiaddr that ends in the peer id, \
         such as /ip4/127.0.0.1/udp/4001/quic-v1/p2p/12D3Koo..."
    )]
    BadBootstrap { addr: String, reason: String },
    #[error("peer network: {0}")]
    Network(String),
}

impl NodeError {
    #[must_use]
    pub fn io(op: &'static str, path: &std::path::Path, source: std::io::Error) -> Self {
        Self::Io { op, path: path.to_path_buf(), source }
    }
}
