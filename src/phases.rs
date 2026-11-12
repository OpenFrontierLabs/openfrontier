//! Launchers for phases that are configured but not built. They fail loudly, on purpose.

use crate::error::{OflError, Result};

/// Refuse to launch a phase whose launcher does not exist.
///
/// # Errors
///
/// Always: [`OflError`] with code 2 explaining what is missing.
pub fn not_built(phase: &str) -> Result<i32> {
    let message = match phase {
        "post" => {
            "not built: phase 2 (post-training with open-instruct, prime-rl and verifiers) has no \
             launcher. configs/phase2-post/post-training.toml names the upstream entry points at \
             the pinned commits; no flags have been written or checked, and there is no base \
             model to post-train."
        }
        "diloco" => {
            "not built: phase 3 (DiLoCo outer loop across islands with prime-diloco) has no \
             launcher. configs/phase3-diloco/diloco-1b.toml is a config for the pinned upstream; \
             no island setup exists and nothing has been trained. `ofl smoke` checks the DiLoCo \
             algorithm only."
        }
        other => return Err(OflError::misuse(format!("unknown phase {other:?}"))),
    };
    Err(OflError::misuse(message))
}
