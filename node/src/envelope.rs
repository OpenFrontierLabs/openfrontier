//! The signed envelope: `{report, public_key, signature}`.
//!
//! - `report` is a JSON *string* holding the exact UTF-8 bytes that were signed (the compact JSON
//!   of a [`HardwareReport`], see [`HardwareReport::to_signed_json`]). A verifier checks the
//!   signature over those bytes as they arrive and only then parses them. Nothing is
//!   re-serialised, so no JSON canonicalisation rules have to agree between signer and verifier.
//! - `public_key` is hex of the libp2p protobuf encoding of the Ed25519 public key (self
//!   describing; the peer id derives from it).
//! - `signature` is hex of the Ed25519 signature over [`DOMAIN`] followed by the `report` bytes.
//!   The domain prefix stops a signature made here from being valid for any other message type
//!   signed by the same key.

use libp2p_identity::{Keypair, PublicKey};
use serde::{Deserialize, Serialize};

use crate::error::{NodeError, Result};
use crate::report::{HardwareReport, SCHEMA};

pub const DOMAIN: &[u8] = b"ofl-node/hardware-report/v1\n";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedReport {
    pub report: String,
    pub public_key: String,
    pub signature: String,
}

/// A verified envelope: whose key signed which report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verified {
    pub peer_id: String,
    pub report: HardwareReport,
}

fn signing_input(report: &str) -> Vec<u8> {
    [DOMAIN, report.as_bytes()].concat()
}

/// Sign `report` with `keypair`.
///
/// # Errors
///
/// JSON or signing failure.
pub fn sign(keypair: &Keypair, report: &HardwareReport) -> Result<SignedReport> {
    let report = report.to_signed_json()?;
    let signature =
        keypair.sign(&signing_input(&report)).map_err(|e| NodeError::Sign(e.to_string()))?;
    Ok(SignedReport {
        report,
        public_key: hex::encode(keypair.public().encode_protobuf()),
        signature: hex::encode(signature),
    })
}

/// Check the signature against the embedded public key, then parse the signed report.
///
/// This proves the report was signed by the holder of `public_key`, not who that is: compare
/// the returned peer id with one you already trust.
///
/// # Errors
///
/// [`NodeError::BadEnvelope`] for undecodable fields, [`NodeError::BadSignature`] for a
/// signature that does not match, [`NodeError::Json`] when the signed bytes are not a report.
pub fn verify(envelope: &SignedReport) -> Result<Verified> {
    let key_bytes = hex::decode(&envelope.public_key)
        .map_err(|e| NodeError::BadEnvelope(format!("public_key is not hex: {e}")))?;
    let public = PublicKey::try_decode_protobuf(&key_bytes)
        .map_err(|e| NodeError::BadEnvelope(format!("public_key does not decode: {e}")))?;
    let signature = hex::decode(&envelope.signature)
        .map_err(|e| NodeError::BadEnvelope(format!("signature is not hex: {e}")))?;
    if !public.verify(&signing_input(&envelope.report), &signature) {
        return Err(NodeError::BadSignature);
    }
    let report: HardwareReport = serde_json::from_str(&envelope.report)?;
    if report.schema != SCHEMA {
        return Err(NodeError::BadEnvelope(format!("unknown report schema {:?}", report.schema)));
    }
    Ok(Verified { peer_id: public.to_peer_id().to_string(), report })
}
