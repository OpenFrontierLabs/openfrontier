//! Pinned upstreams (`configs/upstreams.toml`): loading, fetching at the exact
//! SHA, verification.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;
use std::time::Duration;

use regex::Regex;

use crate::error::{OflError, Result};

/// Where fetched upstream checkouts live, relative to the repo root.
const CHECKOUT_DIR: &str = ".cache/upstreams";
/// Timeout for each git helper invocation.
const GIT_TIMEOUT: Duration = Duration::from_secs(900);
/// A pinned commit: 40 lowercase hex digits.
#[expect(clippy::expect_used, reason = "the regex literal is compile-time valid")]
// ubs:ignore[rust.perf.regex-new,rust.ownership.unwrap-expect] -- compiled once in a LazyLock static; the expect is on a compile-time-valid literal, not runtime input.
pub(crate) static SHA_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("^[0-9a-f]{40}$").expect("static regex is valid"));

/// One pinned upstream from `configs/upstreams.toml`.
#[derive(Clone, Debug)]
pub struct Upstream {
    /// Table name, e.g. `olmo-core`.
    pub name: String,
    /// https repository URL.
    pub repo: String,
    /// Human reference: tag, branch or short sha.
    pub reference: String,
    /// Exact pinned commit.
    pub sha: String,
    /// Licence as recorded at verification.
    pub license: String,
    /// Date the pin was verified.
    pub verified: String,
    /// Which phases use it.
    pub used_by: String,
}

/// Load every upstream entry, in file order.
///
/// # Errors
///
/// [`OflError`] when the file cannot be read or an entry lacks a required key.
pub fn load(root: &Path) -> Result<Vec<Upstream>> {
    let path = root.join("configs").join("upstreams.toml");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| OflError::new(format!("{}: {e}", path.display())))?;
    let raw: toml::Table =
        toml::from_str(&text).map_err(|e| OflError::new(format!("{}: {e}", path.display())))?;
    raw.iter().map(|(name, entry)| parse(name, entry)).collect()
}

/// Parse one `[name]` table into an [`Upstream`].
fn parse(name: &str, entry: &toml::Value) -> Result<Upstream> {
    let field = |key: &str| -> Result<String> {
        entry.get(key).and_then(toml::Value::as_str).map(str::to_owned).ok_or_else(|| {
            OflError::new(format!("configs/upstreams.toml [{name}]: missing {key:?}"))
        })
    };
    Ok(Upstream {
        name: name.to_owned(),
        repo: field("repo")?,
        reference: field("ref")?,
        sha: field("sha")?,
        license: field("license")?,
        verified: field("verified")?,
        used_by: field("used-by")?,
    })
}

/// Structural problems with the pins themselves.
#[must_use]
pub fn problems(ups: &[Upstream]) -> Vec<String> {
    let mut out = Vec::new();
    for up in ups {
        // ubs:ignore[rust.perf.string-alloc-in-loop] -- building one diagnostic message per pin; a handful of upstreams, not a hot path.
        let rel = format!("configs/upstreams.toml [{}]", up.name);
        if !SHA_RE.is_match(&up.sha) {
            // ubs:ignore[rust.perf.string-alloc-in-loop] -- diagnostic message construction, not a hot path.
            out.push(format!("{rel}: sha must be 40 lowercase hex digits"));
        }
        if !up.repo.starts_with("https://") {
            // ubs:ignore[rust.perf.string-alloc-in-loop] -- diagnostic message construction, not a hot path.
            out.push(format!("{rel}: repo must be an https URL"));
        }
    }
    out
}

/// The checkout path for an upstream (it may not exist yet).
#[must_use]
pub fn checkout_path(root: &Path, name: &str) -> PathBuf {
    root.join(CHECKOUT_DIR).join(name)
}

/// HEAD of a checkout, or `None` when `path` is not a git checkout.
///
/// # Errors
///
/// [`OflError`] when git fails on an existing checkout.
pub fn head_sha(path: &Path) -> Result<Option<String>> {
    if !path.join(".git").exists() {
        return Ok(None);
    }
    Ok(Some(git(&["rev-parse", "HEAD"], path)?))
}

/// Return a checkout of `up` at exactly `up.sha`, cloning under `.cache/` if needed.
///
/// # Errors
///
/// [`OflError`] when an existing checkout is at another commit, git fails, or
/// the fetched HEAD does not match the pin.
pub fn ensure_checkout(root: &Path, up: &Upstream) -> Result<PathBuf> {
    let path = checkout_path(root, &up.name);
    let current = head_sha(&path)?;
    if current.as_deref() == Some(up.sha.as_str()) {
        return Ok(path);
    }
    if let Some(current) = current {
        return Err(OflError::new(format!(
            "{} is at {current}, expected {}. Remove it (ofl clean) and re-run.",
            path.display(),
            up.sha
        )));
    }
    std::fs::create_dir_all(&path)
        .map_err(|e| OflError::new(format!("cannot create {}: {e}", path.display())))?;
    // ubs:ignore[rust.ownership.println-macro] -- user-facing fetch progress on a long clone; CLI output, not a library log.
    println!("fetching {} @ {} into {}", up.repo, up.sha, path.display());
    git(&["init", "-q"], &path)?;
    git(&["remote", "add", "origin", &up.repo], &path)?;
    git(&["fetch", "-q", "--depth", "1", "origin", &up.sha], &path)?;
    git(&["checkout", "-q", "FETCH_HEAD"], &path)?;
    let got = head_sha(&path)?;
    if got.as_deref() != Some(up.sha.as_str()) {
        return Err(OflError::new(format!(
            "fetched {}, expected {}",
            got.unwrap_or_else(|| "<none>".to_owned()),
            up.sha
        )));
    }
    Ok(path)
}

/// Run one git command in `cwd`, returning trimmed stdout.
fn git(args: &[&str], cwd: &Path) -> Result<String> {
    let mut cmd = Command::new("git");
    cmd.args(args)
        .current_dir(cwd)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let child = cmd.spawn().map_err(|e| OflError::new(format!("git cannot run: {e}")))?;
    // Bound the clone/fetch and drain both pipes so large output cannot deadlock it.
    let out = crate::repo::wait_piped(child, GIT_TIMEOUT)?;
    if out.timed_out {
        return Err(OflError::new(format!(
            "git {} timed out after {} s",
            args.join(" "),
            GIT_TIMEOUT.as_secs()
        )));
    }
    if out.code != 0 {
        return Err(OflError::new(format!(
            "git {} failed in {}: {}",
            args.join(" "),
            cwd.display(),
            out.stderr.trim()
        )));
    }
    Ok(out.stdout.trim().to_owned())
}
