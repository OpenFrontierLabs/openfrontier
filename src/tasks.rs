//! Developer lifecycle: setup, check, test, fix, build, clean, smoke, dev.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::error::{OflError, Result};
use crate::repo::{RunSpec, find_root, run};
use crate::{configs, ledger};

/// Directories `clean` removes outright.
const CLEAN_DIRS: [&str; 7] =
    ["dist", "build", ".cache", ".ruff_cache", ".mypy_cache", ".pytest_cache", "target"];
/// Directory names never entered when sweeping for `__pycache__` and `*.egg-info`.
const SKIP_DIRS: [&str; 3] = [".venv", ".git", "target"];
/// Timeout for one smoke subprocess.
const SMOKE_TIMEOUT: Duration = Duration::from_secs(1800);
/// Exit code used when a smoke subprocess times out.
const TIMEOUT_CODE: i32 = 124;
/// Log lines shown when a smoke fails.
const FAIL_TAIL_LINES: usize = 25;

/// Locate uv or fail with installation guidance.
fn uv() -> Result<String> {
    which("uv").ok_or_else(|| {
        OflError::new("uv not found on PATH; install it from https://docs.astral.sh/uv/")
    })
}

/// Find an executable on PATH.
fn which(name: &str) -> Option<String> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        // ubs:ignore[rust.security.path-traversal] -- standard PATH executable lookup: name is a fixed tool name ("uv"), dirs come from the operator's own PATH.
        let candidate = dir.join(name);
        if candidate.is_file() {
            // ubs:ignore[rust.perf.string-alloc-in-loop] -- returns on the first match; the allocation happens once, not per iteration.
            return Some(candidate.display().to_string());
        }
    }
    None
}

/// Create/refresh the Python smoke environment from `smoke/uv.lock`.
///
/// # Errors
///
/// [`OflError`] when uv is missing or the sync fails.
pub fn setup() -> Result<i32> {
    let root = find_root(None)?;
    let smoke_dir = root.join("smoke");
    let argv = vec![uv()?, "sync".to_owned(), "--locked".to_owned()];
    let spec = RunSpec { argv: &argv, cwd: Some(&smoke_dir), env: None };
    run(&spec, None)
}

/// Build the `ofl` binary.
///
/// # Errors
///
/// [`OflError`] when cargo fails to start.
pub fn build() -> Result<i32> {
    let root = find_root(None)?;
    let argv = vec!["cargo".to_owned(), "build".to_owned()];
    let spec = RunSpec { argv: &argv, cwd: Some(&root), env: None };
    run(&spec, None)
}

/// Run the Rust test suite, then the swarm tests in the Python smoke environment.
///
/// # Errors
///
/// [`OflError`] when cargo or the smoke environment fails to start.
pub fn test() -> Result<i32> {
    let root = find_root(None)?;
    let argv = vec!["cargo".to_owned(), "test".to_owned()];
    let spec = RunSpec { argv: &argv, cwd: Some(&root), env: None };
    let rust = run(&spec, None)?;
    ensure_smoke_env(&root)?;
    let smoke_dir = root.join("smoke");
    let python = crate::repo::smoke_python(&root);
    let argv = crate::repo::python_module(&python, "pytest", &[]);
    let spec = RunSpec { argv: &argv, cwd: Some(&smoke_dir), env: None };
    Ok(rust | run(&spec, None)?)
}

/// Auto-fix lint and format.
///
/// # Errors
///
/// [`OflError`] when cargo fails to start.
pub fn fix() -> Result<i32> {
    let root = find_root(None)?;
    let clippy_argv = vec![
        "cargo".to_owned(),
        "clippy".to_owned(),
        "--fix".to_owned(),
        "--allow-dirty".to_owned(),
        "--allow-staged".to_owned(),
    ];
    let clippy = RunSpec { argv: &clippy_argv, cwd: Some(&root), env: None };
    let mut rc = run(&clippy, None)?;
    let fmt_argv = vec!["cargo".to_owned(), "fmt".to_owned()];
    let fmt = RunSpec { argv: &fmt_argv, cwd: Some(&root), env: None };
    rc |= run(&fmt, None)?;
    Ok(rc)
}

/// Lint, format check, types, ledger schema + example, config parsing. Runs every step.
///
/// # Errors
///
/// [`OflError`] when a subprocess cannot start.
pub fn check() -> Result<i32> {
    let root = find_root(None)?;
    let mut failed: Vec<&str> = Vec::new();
    for (name, argv) in lint_steps() {
        let spec = RunSpec { argv: &argv, cwd: Some(&root), env: None };
        if run(&spec, None)? != 0 {
            failed.push(name);
        }
    }
    for (name, problems) in [
        ("ledger schema and example", ledger::check(&root)),
        ("configs, manifests and upstream pins", configs::check(&root)),
    ] {
        if problems.is_empty() {
            println!("ok   {name}");
        } else {
            failed.push(name);
            println!("FAIL {name}");
            for problem in problems {
                println!("  - {problem}");
            }
        }
    }
    if !failed.is_empty() {
        println!("check failed: {}", failed.join(", "));
        return Ok(1);
    }
    println!("check passed");
    Ok(0)
}

/// The lint/format/type subprocesses `check` runs.
fn lint_steps() -> Vec<(&'static str, Vec<String>)> {
    vec![
        ("cargo fmt --check", vec!["cargo".to_owned(), "fmt".to_owned(), "--check".to_owned()]),
        (
            "cargo clippy",
            vec![
                "cargo".to_owned(),
                "clippy".to_owned(),
                "--all-targets".to_owned(),
                "--".to_owned(),
                "-D".to_owned(),
                "warnings".to_owned(),
            ],
        ),
    ]
}

/// Remove caches, build output and scratch inside the repo. Leaves `.venv` and `.git` alone.
///
/// # Errors
///
/// [`OflError`] when the repo root cannot be found.
pub fn clean() -> Result<i32> {
    let root = find_root(None)?;
    let mut removed: Vec<PathBuf> = Vec::new();
    for name in CLEAN_DIRS {
        // ubs:ignore[rust.security.path-traversal] -- name is a fixed constant from CLEAN_DIRS, joined under the repo root; no untrusted segment.
        let path = root.join(name);
        if path.exists() {
            std::fs::remove_dir_all(&path)
                .map_err(|e| OflError::new(format!("cannot remove {}: {e}", path.display())))?;
            removed.push(path);
        }
    }
    sweep(&root, &mut removed)?;
    for path in &removed {
        println!("removed {}", path.strip_prefix(&root).unwrap_or(path).display());
    }
    println!("clean: removed {} path(s)", removed.len());
    Ok(0)
}

/// Sweep `__pycache__` and `*.egg-info` directories, skipping vendor dirs.
fn sweep(dir: &Path, removed: &mut Vec<PathBuf>) -> Result<()> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| OflError::new(format!("cannot read {}: {e}", dir.display())))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        if SKIP_DIRS.contains(&name.as_str()) {
            continue;
        }
        if name == "__pycache__" || name.ends_with(".egg-info") {
            std::fs::remove_dir_all(&path)
                .map_err(|e| OflError::new(format!("cannot remove {}: {e}", path.display())))?;
            removed.push(path);
            continue;
        }
        sweep(&path, removed)?;
    }
    Ok(())
}

/// Ensure the Python smoke environment exists, syncing it from `smoke/uv.lock`
/// on first use so `ofl smoke` works right after clone.
pub(crate) fn ensure_smoke_env(root: &Path) -> Result<()> {
    if root.join("smoke/.venv/bin/python").exists() {
        return Ok(());
    }
    println!(
        "smoke environment missing; running uv sync in smoke/ (first run downloads CPU torch)"
    );
    let smoke_dir = root.join("smoke");
    let argv = vec![uv()?, "sync".to_owned(), "--locked".to_owned()];
    let spec = RunSpec { argv: &argv, cwd: Some(&smoke_dir), env: None };
    let rc = run(&spec, None)?;
    if rc != 0 {
        return Err(OflError::new("uv sync failed for the smoke environment; run `make setup`"));
    }
    Ok(())
}

/// Both CPU smokes: real OLMo-core run (phase 1), then reference `DiLoCo` (phase 3 algorithm).
///
/// # Errors
///
/// [`OflError`] when the repo root or smoke environment cannot be found.
pub fn smoke() -> Result<i32> {
    let root = find_root(None)?;
    ensure_smoke_env(&root)?;
    let steps = [
        ("phase 1: olmo-core trainer, tiny model, CPU", "olmo_tiny"),
        ("phase 3 algorithm: reference DiLoCo (not prime-diloco), CPU", "diloco_ref"),
    ];
    let mut failed: Vec<&str> = Vec::new();
    for (title, module) in steps {
        println!("== {title}");
        let (rc, seconds) = smoke_module(&root, module, &[], false)?;
        let status = if rc == 0 { "ok  " } else { "FAIL" };
        println!("{status} {module} in {seconds:.1} s");
        if rc != 0 {
            failed.push(module);
        }
    }
    if !failed.is_empty() {
        println!("smoke failed: {}", failed.join(", "));
        return Ok(1);
    }
    println!("smoke passed");
    Ok(0)
}

/// A longer, streaming version of the phase 1 smoke: same real OLMo-core trainer on CPU.
///
/// # Errors
///
/// [`OflError`] when the config or smoke environment cannot be read.
pub fn dev() -> Result<i32> {
    let root = find_root(None)?;
    ensure_smoke_env(&root)?;
    let steps = configs::load_smoke_tiny(&root)?.dev_steps;
    let workdir = root.join(".cache/dev");
    if workdir.exists() {
        // A fresh run: the trainer resumes from old checkpoints.
        std::fs::remove_dir_all(&workdir)
            .map_err(|e| OflError::new(format!("cannot clear {}: {e}", workdir.display())))?;
    }
    std::fs::create_dir_all(&workdir)
        .map_err(|e| OflError::new(format!("cannot create {}: {e}", workdir.display())))?;
    let args = [
        "--steps".to_owned(),
        steps.to_string(),
        "--workdir".to_owned(),
        workdir.display().to_string(),
    ];
    let (rc, _) = smoke_module(&root, "olmo_tiny", &args, true)?;
    Ok(rc)
}

/// Run one smoke module in its own process; output goes to a log unless streaming.
fn smoke_module(root: &Path, module: &str, args: &[String], stream: bool) -> Result<(i32, f64)> {
    let python = crate::repo::smoke_python(root);
    let command = crate::repo::python_module(&python, &format!("smoke.{module}"), args);
    let start = Instant::now();
    if stream {
        let spec = RunSpec { argv: &command, cwd: Some(root), env: None };
        let rc = run(&spec, None)?;
        return Ok((rc, start.elapsed().as_secs_f64()));
    }
    let log = root.join(".cache/smoke").join(format!("{module}.log"));
    std::fs::create_dir_all(log.parent().unwrap_or(root))
        .map_err(|e| OflError::new(format!("cannot create smoke log dir: {e}")))?;
    let rc = run_logged(&command, root, &log)?;
    let elapsed = start.elapsed().as_secs_f64();
    report_smoke(&log, rc);
    Ok((rc, elapsed))
}

/// Run a smoke subprocess writing stdout+stderr to a log, bounded by a timeout.
fn run_logged(argv: &[String], cwd: &Path, log: &Path) -> Result<i32> {
    let Some((program, rest)) = argv.split_first() else {
        return Err(OflError::new("empty smoke command"));
    };
    let file = std::fs::File::create(log)
        .map_err(|e| OflError::new(format!("cannot create {}: {e}", log.display())))?;
    let stderr =
        file.try_clone().map_err(|e| OflError::new(format!("cannot clone log handle: {e}")))?;
    // ubs:ignore[rust.security.command-executable] -- the smoke interpreter from the repo's own venv; trusted, fixed by ofl.
    let mut child = Command::new(program)
        .args(rest)
        .current_dir(cwd)
        .stdout(Stdio::from(file))
        .stderr(Stdio::from(stderr))
        .spawn()
        .map_err(|e| OflError::new(format!("cannot run {program}: {e}")))?;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status.code().unwrap_or(1)),
            Ok(None) if start.elapsed() < SMOKE_TIMEOUT => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return Ok(TIMEOUT_CODE);
            }
            Err(_) => return Ok(1),
        }
    }
}

/// Print the smoke result line, and the log tail on failure.
fn report_smoke(log: &Path, rc: i32) {
    let text = std::fs::read_to_string(log).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    let result = lines
        .iter()
        .rev()
        .find(|l| l.starts_with("OFL_SMOKE_RESULT"))
        .copied()
        .unwrap_or("(no result line)");
    println!("{result}");
    if rc != 0 {
        println!("--- last lines of {} ---", log.display());
        let start = lines.len().saturating_sub(FAIL_TAIL_LINES);
        // ubs:ignore[rust.panic.direct-indexing] -- start = len - N clamped at 0, so start <= len; the slice is always in bounds.
        for line in &lines[start..] {
            println!("{line}");
        }
    }
}
