//! Open Frontier Node device identity: a persisted Ed25519 keypair that signs a hardware report.
//!
//! The key type is `libp2p-identity`'s Ed25519, so the same key is the node's libp2p peer id later.

pub mod envelope;
pub mod error;
pub mod identity;
pub mod report;

pub use error::{NodeError, Result};
