//! Repo-root discovery, subprocess helpers and shell quoting shared by the commands.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use crate::error::{OflError, Result};

/// A checkout root is the directory holding this marker.
const ROOT_MARKER: &str = "configs/upstreams.toml";
/// Poll interval while waiting on a timed child.
const POLL_INTERVAL: Duration = Duration::from_millis(50);
/// Exit code reported when a helper command cannot run or is killed.
const HELPER_FAIL_CODE: i32 = 127;

/// Return the checkout root: the directory holding `configs/upstreams.toml`,
/// searching upward from `start` (default: cwd).
///
/// # Errors
///
/// [`OflError`] when the cwd cannot be read or no such directory exists above `start`.
pub fn find_root(start: Option<&Path>) -> Result<PathBuf> {
    let cwd =
        std::env::current_dir().map_err(|e| OflError::new(format!("cannot read cwd: {e}")))?;
    let here = start.unwrap_or(&cwd);
    let resolved = here
        .canonicalize()
        .map_err(|e| OflError::new(format!("cannot resolve {}: {e}", here.display())))?;
    for dir in resolved.ancestors() {
        if dir.join(ROOT_MARKER).is_file() {
            return Ok(dir.to_path_buf());
        }
    }
    Err(OflError::new(format!(
        "not inside an openfrontier checkout (no {ROOT_MARKER} above {}). \
         Run ofl from a clone of https://github.com/OpenFrontierLabs/openfrontier.",
        here.display()
    )))
}

/// The interpreter that has the pinned trainer stack: the smoke venv when it
/// exists, else `python3` on PATH.
#[must_use]
pub fn smoke_python(root: &Path) -> String {
    let venv = root.join("smoke/.venv/bin/python");
    if venv.exists() {
        return venv.display().to_string();
    }
    "python3".to_owned()
}

/// A subprocess to run: argv plus optional working directory and environment.
#[derive(Debug)]
pub struct RunSpec<'a> {
    /// Program and arguments; the program is `argv[0]`.
    pub argv: &'a [String],
    /// Working directory; `None` inherits the caller's.
    pub cwd: Option<&'a Path>,
    /// Full environment; `None` inherits the caller's.
    pub env: Option<&'a [(String, String)]>,
}

impl<'a> RunSpec<'a> {
    /// A command with inherited stdio and environment.
    #[must_use]
    pub const fn inherited(argv: &'a [String]) -> Self {
        Self { argv, cwd: None, env: None }
    }
}

/// Run a command with inherited stdio and return its exit code.
///
/// Prints `+ <argv>` first. `timeout` is `None` only for training launches,
/// which run until they finish; a fired timeout is an error.
///
/// # Errors
///
/// [`OflError`] when the program is missing, the spawn fails, or the timeout fires.
pub fn run(spec: &RunSpec<'_>, timeout: Option<Duration>) -> Result<i32> {
    if spec.argv.is_empty() {
        return Err(OflError::new("empty command"));
    }
    let joined = shlex::try_join(spec.argv.iter().map(String::as_str))
        .map_err(|e| OflError::new(format!("argv cannot be shell-quoted: {e}")))?;
    // ubs:ignore[rust.ownership.println-macro] -- echoing the command before running it is the CLI's documented behavior (like make/shell -x).
    println!("+ {joined}");
    let mut cmd = command(spec);
    let child = spawn(&mut cmd, spec.argv)?;
    wait(child, spec.argv, timeout)
}

/// Run a short helper capturing stdout: (exit code, trimmed stdout).
/// A missing program, failed spawn or fired timeout is code 127 with empty
/// output, never an error.
#[must_use]
pub fn run_captured(argv: &[String], timeout: Duration) -> (i32, String) {
    let Some((program, rest)) = argv.split_first() else {
        return (HELPER_FAIL_CODE, String::new());
    };
    // ubs:ignore[rust.security.command-executable] -- trusted operator CLI: the executable is caller/config argv (git, uv, python, upstream trainers), the tool's purpose; no shell, no lower-trust input, same as the Python subprocess.run it replaces.
    let mut cmd = Command::new(program);
    cmd.args(rest).stdout(Stdio::piped()).stderr(Stdio::null());
    let Ok(mut child) = cmd.spawn() else {
        return (HELPER_FAIL_CODE, String::new());
    };
    let Some(code) = poll_child(&mut child, timeout) else {
        return (HELPER_FAIL_CODE, String::new());
    };
    let stdout = child
        .stdout
        .take()
        .map(|mut pipe| {
            let mut buf = String::new();
            let _ = pipe.read_to_string(&mut buf);
            buf
        })
        .unwrap_or_default();
    (code, stdout.trim().to_owned())
}

/// Build the configured command with inherited stdio.
fn command(spec: &RunSpec<'_>) -> Command {
    // `run` rejects an empty argv before calling this, so split_first always yields.
    let Some((program, args)) = spec.argv.split_first() else {
        return Command::new("");
    };
    // ubs:ignore[rust.security.command-executable] -- trusted operator CLI: the executable is caller/config argv (cargo, git, uv, python, upstream trainers); no shell, no lower-trust input. Same trust boundary as run_captured.
    let mut cmd = Command::new(program);
    cmd.args(args);
    if let Some(cwd) = spec.cwd {
        cmd.current_dir(cwd);
    }
    if let Some(env) = spec.env {
        cmd.env_clear().envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    }
    cmd
}

/// Spawn, mapping a missing program to [`OflError`].
fn spawn(cmd: &mut Command, argv: &[String]) -> Result<Child> {
    let program = argv.first().map_or("<empty>", String::as_str);
    cmd.spawn().map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => OflError::new(format!("command not found: {program}")),
        _ => OflError::new(format!("cannot run {program}: {e}")),
    })
}

/// Wait for the child, enforcing the timeout when one is set.
fn wait(mut child: Child, argv: &[String], timeout: Option<Duration>) -> Result<i32> {
    let program = argv.first().map_or("<empty>", String::as_str);
    let Some(limit) = timeout else {
        let status =
            child.wait().map_err(|e| OflError::new(format!("cannot wait for {program}: {e}")))?;
        return Ok(status.code().unwrap_or(1));
    };
    poll_child(&mut child, limit)
        .ok_or_else(|| OflError::new(format!("{program} timed out after {} s", limit.as_secs())))
}

/// Poll until the child exits or `limit` elapses; `None` means it was killed on timeout.
fn poll_child(child: &mut Child, limit: Duration) -> Option<i32> {
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status.code().unwrap_or(1)),
            Ok(None) if start.elapsed() < limit => std::thread::sleep(POLL_INTERVAL),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            Err(_) => return Some(1),
        }
    }
}

/// Captured output and exit status of a bounded child run.
#[derive(Debug)]
pub struct PipedOutput {
    /// Exit code (1 when the child was killed on timeout or signaled).
    pub code: i32,
    /// Everything the child wrote to stdout.
    pub stdout: String,
    /// Everything the child wrote to stderr.
    pub stderr: String,
    /// True when the child was killed because `limit` elapsed.
    pub timed_out: bool,
}

/// Wait for a child, draining its piped output on background threads.
///
/// Draining keeps a large output from blocking the child against a full pipe
/// buffer (the deadlock a poll-then-read loop hits). Kills the child on timeout.
///
/// # Errors
///
/// [`OflError`] when the child cannot be waited on or a drain thread panics.
pub fn wait_piped(mut child: Child, limit: Duration) -> Result<PipedOutput> {
    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();
    let out_handle = std::thread::spawn(move || drain_stdout(stdout_pipe));
    let err_handle = std::thread::spawn(move || drain_stderr(stderr_pipe));
    let start = Instant::now();
    let mut timed_out = false;
    let code = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status.code().unwrap_or(1),
            Ok(None) if start.elapsed() < limit => std::thread::sleep(POLL_INTERVAL),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                timed_out = true;
                break 1;
            }
            // ubs:ignore[rust.perf.string-alloc-in-loop] -- error message on the wait-failure exit path, not a per-iteration allocation.
            Err(e) => return Err(OflError::new(format!("wait failed: {e}"))),
        }
    };
    if timed_out {
        // A grandchild (git-remote-https, torchrun workers) may still hold the
        // pipes open; joining the drain threads would block past the timeout.
        // Drop the handles and report no output rather than hang.
        return Ok(PipedOutput { code, stdout: String::new(), stderr: String::new(), timed_out });
    }
    let stdout = out_handle.join().map_err(|_| OflError::new("stdout drain panicked"))?;
    let stderr = err_handle.join().map_err(|_| OflError::new("stderr drain panicked"))?;
    Ok(PipedOutput { code, stdout, stderr, timed_out })
}

/// Read the stdout pipe to EOF, returning its bytes as lossy UTF-8.
fn drain_stdout(pipe: Option<std::process::ChildStdout>) -> String {
    let Some(mut pipe) = pipe else {
        return String::new();
    };
    let mut buf = Vec::new();
    let _ = pipe.read_to_end(&mut buf);
    String::from_utf8_lossy(&buf).into_owned()
}

/// Read the stderr pipe to EOF, returning its bytes as lossy UTF-8.
fn drain_stderr(pipe: Option<std::process::ChildStderr>) -> String {
    let Some(mut pipe) = pipe else {
        return String::new();
    };
    let mut buf = Vec::new();
    let _ = pipe.read_to_end(&mut buf);
    String::from_utf8_lossy(&buf).into_owned()
}

/// argv that runs `python -m module args` with the given interpreter.
#[must_use]
pub fn python_module(python: &str, module: &str, args: &[String]) -> Vec<String> {
    let mut command = vec![python.to_owned(), "-m".to_owned(), module.to_owned()];
    command.extend(args.iter().cloned());
    command
}
