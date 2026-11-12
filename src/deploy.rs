//! `ofl deploy`: publishing is not implemented because no run has produced anything to publish.

use std::path::{Path, PathBuf};

use crate::error::{OflError, Result};
use crate::repo::find_root;

/// The environments `--env` accepts, with their one-line descriptions.
pub const ENVS: [(&str, &str); 2] = [
    ("dry-run", "print what would be published and publish nothing"),
    ("huggingface", "publish checkpoints and weights to Hugging Face (NOT IMPLEMENTED)"),
];

/// Checkpoint directories under `checkpoints/` (written by the run0 and moe launchers).
#[must_use]
pub fn find_checkpoints(root: &Path) -> Vec<PathBuf> {
    let base = root.join("checkpoints");
    if !base.is_dir() {
        return Vec::new();
    }
    let mut found = Vec::new();
    collect_step_dirs(&base, &mut found);
    found.sort();
    found
}

/// Recursively collect directories named `step*` under `dir`.
fn collect_step_dirs(dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if path.file_name().is_some_and(|n| n.to_string_lossy().starts_with("step")) {
            // ubs:ignore[rust.collections.clone-in-loop,rust.collections.clone-any] -- the path is both stored and recursed into; one clone is required, not a per-iteration waste.
            found.push(path.clone());
        }
        collect_step_dirs(&path, found);
    }
}

/// Deploy: only `dry-run` works; `huggingface` refuses because nothing is implemented.
///
/// # Errors
///
/// [`OflError`] with code 2 for an unknown env or the unimplemented `huggingface` env.
// ubs:ignore[rust.ownership.println-macro] -- dry-run report is the command's user-facing output; a CLI, not a library log.
pub fn deploy(env: &str) -> Result<i32> {
    if !ENVS.iter().any(|(name, _)| *name == env) {
        let known: Vec<&str> = ENVS.iter().map(|(name, _)| *name).collect();
        return Err(OflError::misuse(format!(
            "unknown --env {env:?}; known: {}",
            known.join(", ")
        )));
    }
    if env == "huggingface" {
        return Err(OflError::misuse(
            "not implemented yet: nothing to publish until a run has produced checkpoints. \
             No publishing code exists.",
        ));
    }
    let root = find_root(None)?;
    let found = find_checkpoints(&root);
    println!("deploy --env dry-run: nothing is published.");
    if found.is_empty() {
        println!(
            "would publish: nothing. No checkpoints under checkpoints/ (nothing has been trained)."
        );
        return Ok(0);
    }
    let noun = if found.len() == 1 { "directory" } else { "directories" };
    println!("would publish {} checkpoint {noun}:", found.len());
    for path in &found {
        println!("  {}", path.strip_prefix(&root).unwrap_or(path).display());
    }
    println!("publishing itself is not implemented; --env huggingface exits non-zero.");
    Ok(0)
}
