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
static SHA_RE: LazyLock<Regex> =
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
        let rel = format!("configs/upstreams.toml [{}]", up.name);
        if !SHA_RE.is_match(&up.sha) {
            out.push(format!("{rel}: sha must be 40 lowercase hex digits"));
        }
        if !up.repo.starts_with("https://") {
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
    let start = std::time::Instant::now();
    let mut cmd = Command::new("git");
    cmd.args(args)
        .current_dir(cwd)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| OflError::new(format!("git cannot run: {e}")))?;
    // Bound the clone/fetch: git network operations must not hang forever.
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if start.elapsed() < GIT_TIMEOUT => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(OflError::new(format!(
                    "git {} timed out after {} s",
                    args.join(" "),
                    GIT_TIMEOUT.as_secs()
                )));
            }
            Err(e) => return Err(OflError::new(format!("git wait failed: {e}"))),
        }
    }
    let out =
        child.wait_with_output().map_err(|e| OflError::new(format!("git output failed: {e}")))?;
    if !out.status.success() {
        return Err(OflError::new(format!(
            "git {} failed in {}: {}",
            args.join(" "),
            cwd.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
}
