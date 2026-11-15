//! Load or create the device keypair on disk.
//!
//! Format: the libp2p protobuf encoding of the keypair (what rust-libp2p and kubo store), raw
//! bytes, no text wrapper. The file holds the secret key; it is created 0600 on Unix and refused
//! on load if group or others can access it.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use directories::ProjectDirs;
use libp2p_identity::Keypair;

use crate::error::{NodeError, Result};

const KEY_FILE_NAME: &str = "device.key";

/// Counts staging files made by this process; see [`create_exclusive`].
static ATTEMPT: AtomicU64 = AtomicU64::new(0);

/// Whether [`load_or_create`] made a new key or found an existing one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    Created,
    Existing,
}

/// Default key path: `<platform config dir>/ofl-node/device.key`.
///
/// # Errors
///
/// [`NodeError::NoConfigDir`] when the platform reports no home directory.
pub fn default_key_path() -> Result<PathBuf> {
    let dirs =
        ProjectDirs::from("org", "OpenFrontierLabs", "ofl-node").ok_or(NodeError::NoConfigDir)?;
    Ok(dirs.config_dir().join(KEY_FILE_NAME))
}

/// Return the keypair at `path`, generating and persisting one first if the file does not exist.
///
/// An existing key is never overwritten: creation links a fully written temp file into place, so
/// two racing first runs end with one key and a concurrent reader never sees a partial file.
///
/// # Errors
///
/// I/O failures, a key file that is not a regular file, one with loose permissions (Unix), or
/// one whose contents are not an Ed25519 keypair.
pub fn load_or_create(path: &Path) -> Result<(Keypair, Origin)> {
    match load(path) {
        Ok(keypair) => return Ok((keypair, Origin::Existing)),
        Err(NodeError::Io { source, .. }) if source.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    let keypair = Keypair::generate_ed25519();
    let encoded = keypair
        .to_protobuf_encoding()
        .map_err(|e| NodeError::Sign(format!("cannot encode keypair: {e}")))?;
    if create_exclusive(path, &encoded)? {
        Ok((keypair, Origin::Created))
    } else {
        // Lost a creation race: the winner's key is the identity.
        load(path).map(|k| (k, Origin::Existing))
    }
}

/// Read and validate the key file.
///
/// # Errors
///
/// See [`load_or_create`]; a missing file is an [`NodeError::Io`] with kind `NotFound`.
pub fn load(path: &Path) -> Result<Keypair> {
    // Stat before opening: opening a FIFO with no writer blocks, so a non-regular file must be
    // refused first. The descriptor is checked again below against a swap in between.
    let before = fs::metadata(path).map_err(|e| NodeError::io("stat key file", path, e))?;
    if !before.is_file() {
        return Err(NodeError::KeyFileNotRegular { path: path.to_path_buf() });
    }
    let mut file = File::open(path).map_err(|e| NodeError::io("open key file", path, e))?;
    let meta = file.metadata().map_err(|e| NodeError::io("stat key file", path, e))?;
    if !meta.is_file() {
        return Err(NodeError::KeyFileNotRegular { path: path.to_path_buf() });
    }
    check_mode(path, &meta)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(|e| NodeError::io("read key file", path, e))?;
    Keypair::from_protobuf_encoding(&bytes)
        .ok()
        .filter(|k| k.key_type() == libp2p_identity::KeyType::Ed25519)
        .ok_or_else(|| NodeError::KeyFileCorrupt {
            path: path.to_path_buf(),
            reason: format!("{} bytes do not decode", bytes.len()),
        })
}

#[cfg(unix)]
fn check_mode(path: &Path, meta: &fs::Metadata) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mode = meta.permissions().mode() & 0o777;
    // Group and other bits (the low six) must all be clear.
    if mode.trailing_zeros() >= 6 {
        Ok(())
    } else {
        Err(NodeError::KeyFilePermissions { path: path.to_path_buf(), mode })
    }
}

#[cfg(not(unix))]
fn check_mode(_path: &Path, _meta: &fs::Metadata) -> Result<()> {
    // No POSIX mode bits; the file inherits the user profile directory's ACL.
    Ok(())
}

/// Write `bytes` to `path` with owner-only permissions unless `path` exists. `Ok(false)` means it
/// already existed and was left untouched. On `Ok(true)` the file and its directory entry have
/// been flushed to disk.
fn create_exclusive(path: &Path, bytes: &[u8]) -> Result<bool> {
    let dir = dir_of(path);
    create_private_dir(dir)?;
    // One staging name per attempt (process id plus a counter), so concurrent attempts in one
    // process never share or delete each other's file.
    let attempt = ATTEMPT.fetch_add(1, Ordering::Relaxed);
    let tmp = dir.join(format!(".{KEY_FILE_NAME}.{}.{attempt}.tmp", std::process::id()));
    let result = write_private(&tmp, bytes).and_then(|()| match fs::hard_link(&tmp, path) {
        Ok(()) => sync_dir(dir).map(|()| true),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
        Err(e) => Err(NodeError::io("create key file", path, e)),
    });
    // The temp file is only a staging name; the key file (if made) is a second link to it.
    drop(fs::remove_file(&tmp));
    result
}

/// The directory that holds `path`, `.` for a bare file name.
fn dir_of(path: &Path) -> &Path {
    path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."))
}

/// Flush a directory's entries to disk, so a file created in it survives a crash.
#[cfg(unix)]
fn sync_dir(dir: &Path) -> Result<()> {
    File::open(dir).and_then(|d| d.sync_all()).map_err(|e| NodeError::io("sync directory", dir, e))
}

/// Directories cannot be opened for syncing on this platform; nothing to do.
#[cfg(not(unix))]
fn sync_dir(_dir: &Path) -> Result<()> {
    Ok(())
}

fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path).map_err(|e| NodeError::io("create key file", path, e))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|e| NodeError::io("write key file", path, e))
}

fn create_private_dir(dir: &Path) -> Result<()> {
    // Directories this call will create, deepest first, so their entries can be flushed too.
    let mut missing = Vec::new();
    let mut cur = dir;
    while !cur.exists() {
        missing.push(cur);
        match cur.parent().filter(|p| !p.as_os_str().is_empty()) {
            Some(parent) => cur = parent,
            None => break,
        }
    }
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(dir).map_err(|e| NodeError::io("create config dir", dir, e))?;
    for created in missing {
        sync_dir(created)?;
        sync_dir(dir_of(created))?;
    }
    Ok(())
}
