//! Phase 1 dense launcher: builds (and optionally runs) the real OLMo-core
//! command for one node.
//!
//! The command is the pinned OLMo-core's own example script,
//! `src/examples/llm/train.py`, started with `torch.distributed.run`. Every
//! option and override comes from a config in `configs/phase1-dense/` and was
//! accepted by that script's `--dry-run` at the pinned commit; `ofl run0 verify`
//! re-checks it.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::configs::load_toml;
use crate::error::{OflError, Result};
use crate::repo::{self, RunSpec, find_root, run};
use crate::upstreams;

/// Default config, relative to the repo root.
const DEFAULT_CONFIG: &str = "configs/phase1-dense/run0.toml";
/// The upstream this launcher drives.
const UPSTREAM: &str = "olmo-core";
/// Timeout for one upstream `--dry-run` acceptance check.
const VERIFY_TIMEOUT: Duration = Duration::from_secs(600);
/// Characters of upstream stderr kept when a verify check fails.
const STDERR_TAIL_CHARS: usize = 2000;

/// A fully assembled launch: the upstream script args and the torchrun wrapper.
#[derive(Clone, Debug)]
pub struct Launch {
    /// The upstream script and its arguments.
    pub script_args: Vec<String>,
    /// The full `torch.distributed.run` command.
    pub argv: Vec<String>,
    /// Working directory for the launch.
    pub cwd: PathBuf,
}

/// Paths and identity the launcher injects into every command.
#[derive(Clone, Debug)]
pub struct Dirs {
    /// The pinned upstream checkout holding the entrypoint script.
    pub script_dir: PathBuf,
    /// Checkpoint destination.
    pub save_folder: PathBuf,
    /// Trainer scratch directory.
    pub work_dir: PathBuf,
}

/// Everything `build` needs beyond the config itself.
#[derive(Clone, Debug)]
pub struct Options {
    /// Interpreter with GPU torch and the pinned olmo-core installed.
    pub python: String,
    /// GPUs on this node.
    pub nproc: u32,
    /// Run name override; `None` uses `[meta].name`.
    pub run_name: Option<String>,
    /// Tokenised shards for glob configs.
    pub data_glob: Option<String>,
    /// Append `--dry-run` to the upstream script.
    pub dry_run: bool,
}

/// Render a TOML value the way the upstream's dotted-override parser reads it.
#[must_use]
pub fn format_override(value: &toml::Value) -> String {
    match value {
        toml::Value::Boolean(b) => if *b { "true" } else { "false" }.to_owned(),
        toml::Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(format_override).collect();
            format!("[{}]", parts.join(","))
        }
        toml::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Assemble the launch command. Pure: touches no network and starts nothing.
///
/// # Errors
///
/// [`OflError`] when the config cannot be read, a glob config has no
/// `--data-glob`, or a mix config is given one.
pub fn build(root: &Path, config: &Path, dirs: &Dirs, opts: &Options) -> Result<Launch> {
    let raw = load_toml(config)?;
    let launch = raw
        .get("launch")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| OflError::new(format!("{}: missing [launch]", config.display())))?;
    let source = crate::configs::path(&raw, "data/source")
        .and_then(toml::Value::as_str)
        .ok_or_else(|| OflError::new(format!("{}: missing [data].source", config.display())))?;
    let mut overrides: Vec<(String, toml::Value)> = raw
        .get("overrides")
        .and_then(toml::Value::as_table)
        .map(|t| t.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();
    apply_data_source(config, source, opts.data_glob.as_deref(), &mut overrides)?;
    let name = opts
        .run_name
        .clone()
        .or_else(|| {
            crate::configs::path(&raw, "meta/name").and_then(toml::Value::as_str).map(str::to_owned)
        })
        .ok_or_else(|| OflError::new(format!("{}: missing [meta].name", config.display())))?;
    let script_args = script_args(launch, &name, dirs, opts, &overrides)?;
    let mut argv = vec![
        opts.python.clone(),
        "-m".to_owned(),
        "torch.distributed.run".to_owned(),
        "--standalone".to_owned(),
        format!("--nproc-per-node={}", opts.nproc),
    ];
    argv.extend(script_args.iter().cloned());
    Ok(Launch { script_args, argv, cwd: root.to_path_buf() })
}

/// Set or reject `--data-glob` according to the config's data source.
fn apply_data_source(
    config: &Path,
    source: &str,
    data_glob: Option<&str>,
    overrides: &mut Vec<(String, toml::Value)>,
) -> Result<()> {
    match source {
        "glob" => {
            let Some(glob) = data_glob else {
                return Err(OflError::new(format!(
                    "{} reads tokenised shards you provide: pass --data-glob '/path/to/*.npy'. \
                     No FineWeb-Edu shards exist yet; see docs/run0.md.",
                    config.display()
                )));
            };
            overrides.push((
                "dataset.paths".to_owned(),
                toml::Value::Array(vec![toml::Value::String(glob.to_owned())]),
            ));
            overrides.push(("dataset.expand_glob".to_owned(), toml::Value::Boolean(true)));
            Ok(())
        }
        _ if data_glob.is_some() => Err(OflError::new(format!(
            "{} streams its own data mix; --data-glob does not apply",
            config.display()
        ))),
        _ => Ok(()),
    }
}

/// The upstream script and its arguments, in the order the Python launcher used.
fn script_args(
    launch: &toml::Table,
    name: &str,
    dirs: &Dirs,
    opts: &Options,
    overrides: &[(String, toml::Value)],
) -> Result<Vec<String>> {
    let field = |key: &str| -> Result<String> {
        launch
            .get(key)
            .map(|v| match v {
                toml::Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .ok_or_else(|| OflError::new(format!("[launch].{key} missing")))
    };
    let mut args = vec![
        dirs.script_dir.join(field("entrypoint")?).display().to_string(),
        name.to_owned(),
        "--model-factory".to_owned(),
        field("model_factory")?,
        "--sequence-length".to_owned(),
        field("sequence_length")?,
        "--save-folder".to_owned(),
        dirs.save_folder.display().to_string(),
        "--work-dir".to_owned(),
        dirs.work_dir.display().to_string(),
    ];
    if opts.dry_run {
        args.push("--dry-run".to_owned());
    }
    for (key, value) in overrides {
        args.push(format!("--{key}={}", format_override(value)));
    }
    Ok(args)
}

/// CLI arguments for `ofl run0 launch`.
#[derive(Clone, Debug)]
pub struct LaunchArgs {
    /// Config override; `None` uses the default run0 config.
    pub config: Option<PathBuf>,
    /// GPUs on this node.
    pub nproc: u32,
    /// Interpreter override.
    pub python: Option<String>,
    /// Run name override.
    pub run_name: Option<String>,
    /// Checkpoint destination override.
    pub save_folder: Option<PathBuf>,
    /// Trainer scratch override.
    pub work_dir: Option<PathBuf>,
    /// Tokenised shards for glob configs.
    pub data_glob: Option<String>,
    /// Print the command instead of running it.
    pub dry_run: bool,
}

/// Launch (or print) the Run 0 command.
///
/// # Errors
///
/// [`OflError`] when the config is invalid or the upstream fetch fails.
pub fn launch(args: &LaunchArgs) -> Result<i32> {
    let root = find_root(None)?;
    let ups = upstreams::load(&root)?;
    let up = ups
        .iter()
        .find(|u| u.name == UPSTREAM)
        .ok_or_else(|| OflError::new(format!("configs/upstreams.toml has no [{UPSTREAM}]")))?;
    let mut script_dir = upstreams::checkout_path(&root, UPSTREAM);
    let config = args.config.clone().unwrap_or_else(|| root.join(DEFAULT_CONFIG));
    if !args.dry_run {
        script_dir = upstreams::ensure_checkout(&root, up)?;
    }
    let dirs = Dirs {
        script_dir,
        save_folder: args.save_folder.clone().unwrap_or_else(|| root.join("checkpoints/run0")),
        work_dir: args.work_dir.clone().unwrap_or_else(|| root.join(".cache/run0/work")),
    };
    let opts = Options {
        python: args.python.clone().unwrap_or_else(|| repo::smoke_python(&root)),
        nproc: args.nproc,
        run_name: args.run_name.clone(),
        data_glob: args.data_glob.clone(),
        dry_run: false,
    };
    let plan = build(&root, &config, &dirs, &opts)?;
    if args.dry_run {
        println!("# pinned upstream: {} @ {}", up.repo, up.sha);
        println!(
            "# not run. `ofl fetch {UPSTREAM}` creates {}, which holds the script below",
            dirs.script_dir.display()
        );
        println!("{}", shlex::try_join(plan.argv.iter().map(String::as_str)).unwrap_or_default());
        return Ok(0);
    }
    let spec = RunSpec { argv: &plan.argv, cwd: Some(&plan.cwd), env: None };
    run(&spec, None)
}

/// Run the pinned upstream script with `--dry-run` and our overrides; it must accept them.
///
/// # Errors
///
/// [`OflError`] when the upstream fetch fails or a check times out.
pub fn verify(root: &Path) -> Result<i32> {
    let ups = upstreams::load(root)?;
    let up = ups
        .iter()
        .find(|u| u.name == UPSTREAM)
        .ok_or_else(|| OflError::new(format!("configs/upstreams.toml has no [{UPSTREAM}]")))?;
    let script_dir = upstreams::ensure_checkout(root, up)?;
    let mut failures = 0;
    for config in dense_configs(root) {
        let ok = verify_one(root, &config, &script_dir)?;
        if !ok {
            failures += 1;
        }
    }
    Ok(i32::from(failures > 0))
}

/// Sorted dense configs except the smoke config.
fn dense_configs(root: &Path) -> Vec<PathBuf> {
    let dir = root.join("configs/phase1-dense");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension().is_some_and(|e| e == "toml")
                && p.file_name().is_some_and(|n| n != "smoke-tiny.toml")
        })
        .collect();
    files.sort();
    files
}

/// Verify one config against the upstream script. Returns whether it passed.
fn verify_one(root: &Path, config: &Path, script_dir: &Path) -> Result<bool> {
    let raw = load_toml(config)?;
    let source =
        crate::configs::path(&raw, "data/source").and_then(toml::Value::as_str).unwrap_or("");
    let dirs = Dirs {
        script_dir: script_dir.to_path_buf(),
        save_folder: root.join(".cache/run0/verify-save"),
        work_dir: root.join(".cache/run0/verify-work"),
    };
    let opts = Options {
        python: repo::smoke_python(root),
        nproc: 1,
        run_name: None,
        data_glob: (source == "glob").then(|| "/nonexistent/*.npy".to_owned()),
        dry_run: true,
    };
    let plan = build(root, config, &dirs, &opts)?;
    // --dry-run only prints the merged config, so the distributed launcher is not needed.
    let (code, stderr) = run_verify(&plan.script_args)?;
    let name = config.file_name().unwrap_or_default().to_string_lossy();
    let count = plan.script_args.len().saturating_sub(2);
    if code == 0 {
        println!("ok   {name}: upstream accepted {count} arguments (--dry-run)");
        return Ok(true);
    }
    println!("FAIL {name}: upstream accepted {count} arguments (--dry-run)");
    let tail: String = stderr
        .chars()
        .rev()
        .take(STDERR_TAIL_CHARS)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    println!("{tail}");
    Ok(false)
}

/// Run the upstream script capturing stderr for failure reporting, bounded.
fn run_verify(argv: &[String]) -> Result<(i32, String)> {
    use std::process::{Command, Stdio};
    let mut child = Command::new(&argv[0])
        .args(&argv[1..])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| OflError::new(format!("cannot run {}: {e}", argv[0])))?;
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if start.elapsed() < VERIFY_TIMEOUT => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(OflError::new(format!(
                    "upstream --dry-run timed out after {} s",
                    VERIFY_TIMEOUT.as_secs()
                )));
            }
            Err(e) => return Err(OflError::new(format!("wait failed: {e}"))),
        }
    }
    let out = child.wait_with_output().map_err(|e| OflError::new(format!("output failed: {e}")))?;
    Ok((out.status.code().unwrap_or(1), String::from_utf8_lossy(&out.stderr).into_owned()))
}
