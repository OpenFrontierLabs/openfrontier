//! Phase 1 `MoE` launcher: the Hugging Face `MoE` handbook's Megatron-LM flags for
//! one 8-GPU node.
//!
//! Builds `torch.distributed.run ... pretrain_gpt.py <flags>` against the pinned
//! Megatron-LM commit. `ofl moe verify` checks every flag name in the config
//! (and the ones added at launch) against that commit's source.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;

use crate::configs::load_toml;
use crate::error::{OflError, Result};
use crate::repo::{self, RunSpec, find_root, run};
use crate::upstreams;

/// Default config, relative to the repo root.
const DEFAULT_CONFIG: &str = "configs/phase1-moe/handbook-moe-6p9b.toml";
/// The upstream this launcher drives.
const UPSTREAM: &str = "megatron-lm";
/// Flags `ofl moe launch` adds itself; verified together with the config's flags.
const LAUNCH_FLAGS: [&str; 5] =
    ["--tokenizer-model", "--data-args-path", "--data-cache-path", "--save", "--tensorboard-dir"];

/// A fully assembled `MoE` launch.
#[derive(Clone, Debug)]
pub struct MoeLaunch {
    /// The full `torch.distributed.run` command.
    pub argv: Vec<String>,
    /// Environment variables the config sets (merged over the caller's at launch).
    pub env: Vec<(String, String)>,
    /// Working directory: the Megatron checkout.
    pub cwd: PathBuf,
}

/// Paths the launcher injects into every command.
#[derive(Clone, Debug)]
pub struct MoeDirs {
    /// The pinned Megatron-LM checkout.
    pub megatron_dir: PathBuf,
    /// HF tokenizer directory (`--tokenizer-model`).
    pub tokenizer_dir: PathBuf,
    /// File of Megatron dataset prefixes (`--data-args-path`).
    pub data_args_path: PathBuf,
    /// Megatron data cache (`--data-cache-path`).
    pub data_cache_path: PathBuf,
    /// Checkpoint destination (`--save`).
    pub save_dir: PathBuf,
}

/// Flatten `[flags.*]` (and `[setup_check]`) into an argv fragment, in file order.
///
/// # Errors
///
/// [`OflError`] when `[flags]` or a flag table is missing.
pub fn config_flags(raw: &toml::Table, setup_check: bool) -> Result<Vec<String>> {
    let flags = raw
        .get("flags")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| OflError::new("missing [flags.*] tables"))?;
    let mut out = Vec::new();
    for table in flags.values() {
        let Some(table) = table.as_table() else { continue };
        extend_flags(&mut out, table);
    }
    if setup_check && let Some(table) = raw.get("setup_check").and_then(toml::Value::as_table) {
        extend_flags(&mut out, table);
    }
    Ok(out)
}

/// Append one flag table: bool true is bare, false is skipped, else flag + value.
fn extend_flags(out: &mut Vec<String>, table: &toml::Table) {
    for (flag, value) in table {
        match value {
            // ubs:ignore[rust.collections.clone-in-loop,rust.collections.clone-any] -- building the argv once from the config; each flag name is owned into the command vector.
            toml::Value::Boolean(true) => out.push(flag.clone()),
            toml::Value::Boolean(false) => {}
            other => {
                // ubs:ignore[rust.collections.clone-in-loop,rust.collections.clone-any] -- building the argv once from the config; each flag name is owned into the command vector.
                out.push(flag.clone());
                out.push(toml_value_string(other));
            }
        }
    }
}

/// Render a non-bool TOML flag value as Megatron's argparse reads it.
fn toml_value_string(value: &toml::Value) -> String {
    match value {
        // ubs:ignore[rust.collections.clone-any] -- the argv owns its strings; one clone per config value at build time.
        toml::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Assemble the launch command. Pure: touches no network and starts nothing.
///
/// # Errors
///
/// [`OflError`] when the config cannot be read or lacks `[launch]`/`[flags]`.
pub fn build(config: &Path, dirs: &MoeDirs, python: &str, setup_check: bool) -> Result<MoeLaunch> {
    let raw = load_toml(config)?;
    let gpus = crate::configs::path(&raw, "launch/gpus_per_node")
        .and_then(toml::Value::as_integer)
        .ok_or_else(|| {
            OflError::new(format!("{}: missing [launch].gpus_per_node", config.display()))
        })?;
    let entrypoint =
        crate::configs::path(&raw, "launch/entrypoint").and_then(toml::Value::as_str).ok_or_else(
            || OflError::new(format!("{}: missing [launch].entrypoint", config.display())),
        )?;
    let mut flags = config_flags(&raw, setup_check)?;
    flags.extend([
        "--tokenizer-model".to_owned(),
        dirs.tokenizer_dir.display().to_string(),
        "--data-args-path".to_owned(),
        dirs.data_args_path.display().to_string(),
        "--data-cache-path".to_owned(),
        dirs.data_cache_path.display().to_string(),
        "--save".to_owned(),
        dirs.save_dir.display().to_string(),
        "--tensorboard-dir".to_owned(),
        dirs.save_dir.join("tensorboard").display().to_string(),
    ]);
    let mut argv = vec![
        python.to_owned(),
        "-m".to_owned(),
        "torch.distributed.run".to_owned(),
        "--standalone".to_owned(),
        "--nnodes=1".to_owned(),
        format!("--nproc_per_node={gpus}"),
        dirs.megatron_dir.join(entrypoint).display().to_string(),
    ];
    argv.extend(flags);
    let mut env: Vec<(String, String)> = raw
        .get("env")
        .and_then(toml::Value::as_table)
        // ubs:ignore[rust.collections.clone-any] -- the launch env owns its keys; one clone per config env var at build time.
        .map(|t| t.iter().map(|(k, v)| (k.clone(), toml_value_string(v))).collect())
        .unwrap_or_default();
    env.push(("PYTHONPATH".to_owned(), dirs.megatron_dir.display().to_string()));
    Ok(MoeLaunch { argv, env, cwd: dirs.megatron_dir.clone() })
}

/// CLI arguments for `ofl moe launch`.
#[derive(Clone, Debug)]
pub struct LaunchArgs {
    /// Config override; `None` uses the default handbook config.
    pub config: Option<PathBuf>,
    /// HF tokenizer directory (required).
    pub tokenizer_dir: PathBuf,
    /// File of Megatron dataset prefixes (required).
    pub data_args_path: PathBuf,
    /// Data cache override.
    pub data_cache_path: Option<PathBuf>,
    /// Checkpoint destination override.
    pub save: Option<PathBuf>,
    /// Interpreter inside the Megatron training environment.
    pub python: Option<String>,
    /// Stop after 120 iterations, as the handbook run does.
    pub setup_check: bool,
    /// Print the command instead of running it.
    pub dry_run: bool,
}

/// Launch (or print) the `MoE` command.
///
/// # Errors
///
/// [`OflError`] when required inputs are missing, the config is invalid, or the
/// upstream fetch fails.
pub fn launch(args: &LaunchArgs) -> Result<i32> {
    let root = find_root(None)?;
    let ups = upstreams::load(&root)?;
    let up = ups
        .iter()
        .find(|u| u.name == UPSTREAM)
        .ok_or_else(|| OflError::new(format!("configs/upstreams.toml has no [{UPSTREAM}]")))?;
    let mut megatron_dir = upstreams::checkout_path(&root, UPSTREAM);
    if !args.dry_run {
        megatron_dir = upstreams::ensure_checkout(&root, up)?;
        for (label, path) in
            [("--tokenizer-dir", &args.tokenizer_dir), ("--data-args-path", &args.data_args_path)]
        {
            if !path.exists() {
                // ubs:ignore[rust.perf.string-alloc-in-loop] -- error message on the missing-input exit path, not a per-iteration allocation.
                return Err(OflError::new(format!("{label} {} does not exist", path.display())));
            }
        }
    }
    let dirs = MoeDirs {
        megatron_dir,
        tokenizer_dir: args.tokenizer_dir.clone(),
        data_args_path: args.data_args_path.clone(),
        data_cache_path: args
            .data_cache_path
            .clone()
            .unwrap_or_else(|| root.join(".cache/moe/data-cache")),
        save_dir: args.save.clone().unwrap_or_else(|| root.join("checkpoints/moe")),
    };
    let config = args.config.clone().unwrap_or_else(|| root.join(DEFAULT_CONFIG));
    let python = args.python.clone().unwrap_or_else(|| repo::smoke_python(&root));
    let plan = build(&config, &dirs, &python, args.setup_check)?;
    if args.dry_run {
        print_dry_run(up, &plan);
        return Ok(0);
    }
    let env = merged_env(&plan.env);
    let spec = RunSpec { argv: &plan.argv, cwd: Some(&plan.cwd), env: Some(&env) };
    run(&spec, None)
}

/// Print the dry-run command with its environment.
fn print_dry_run(up: &upstreams::Upstream, plan: &MoeLaunch) {
    println!("# pinned upstream: {} @ {}", up.repo, up.sha);
    println!(
        "# not run. `ofl fetch {UPSTREAM}` creates {}; the script is in it",
        plan.cwd.display()
    );
    let env: Vec<String> = plan
        .env
        .iter()
        .map(|(k, v)| format!("{k}={}", shlex::try_quote(v).unwrap_or_default()))
        .collect();
    let argv = shlex::try_join(plan.argv.iter().map(String::as_str)).unwrap_or_default();
    println!(
        "cd {} && env {} {argv}",
        shlex::try_quote(&plan.cwd.display().to_string()).unwrap_or_default(),
        env.join(" ")
    );
}

/// The caller's environment with the config's variables applied on top.
fn merged_env(overrides: &[(String, String)]) -> Vec<(String, String)> {
    let mut env: Vec<(String, String)> = std::env::vars().collect();
    for (key, value) in overrides {
        if let Some(slot) = env.iter_mut().find(|(k, _)| k == key) {
            slot.1.clone_from(value);
        } else {
            // ubs:ignore[rust.collections.clone-in-loop] -- merging a handful of config env vars over the process env once at launch.
            env.push((key.clone(), value.clone()));
        }
    }
    env
}

/// Matches a Python dataclass/metadata field declaration (`    name: type`) at
/// the start of a line; compiled once.
#[expect(clippy::expect_used, reason = "the regex literal is compile-time valid")]
// ubs:ignore[rust.perf.regex-new,rust.ownership.unwrap-expect] -- compiled once in a LazyLock static; the expect is on a compile-time-valid literal, not runtime input.
static FIELD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\s+(\w+)\s*:").expect("static regex is valid"));

/// Collect every field name Megatron's source declares, once per verify run.
fn declared_fields(texts: &[String]) -> HashSet<String> {
    let mut fields = HashSet::new();
    for text in texts {
        for cap in FIELD_RE.captures_iter(text) {
            if let Some(name) = cap.get(1) {
                // ubs:ignore[rust.perf.string-alloc-in-loop] -- collecting the declared-field set once per verify run; each name is owned into the set.
                fields.insert(name.as_str().to_owned());
            }
        }
    }
    fields
}

/// A flag exists if Megatron defines it as an argparse literal or a dataclass
/// field (or the field behind a `--no-` negation).
fn flag_known(flag: &str, texts: &[String], fields: &HashSet<String>) -> bool {
    let single = format!("'{flag}'");
    let double = format!("\"{flag}\"");
    if texts.iter().any(|t| t.contains(&single) || t.contains(&double)) {
        return true;
    }
    let field = flag.trim_start_matches('-').replace('-', "_");
    if fields.contains(&field) {
        return true;
    }
    field.strip_prefix("no_").is_some_and(|stripped| fields.contains(stripped))
}

/// Flags (tokens starting with `--`) that Megatron's source under `megatron_dir` does not define.
///
/// # Errors
///
/// [`OflError`] when the source tree cannot be read.
pub fn unknown_flags(flags: &[String], megatron_dir: &Path) -> Result<Vec<String>> {
    let texts = read_sources(megatron_dir)?;
    let fields = declared_fields(&texts);
    let mut wanted: Vec<&str> =
        flags.iter().filter(|f| f.starts_with("--")).map(String::as_str).collect();
    wanted.sort_unstable();
    wanted.dedup();
    Ok(wanted.into_iter().filter(|f| !flag_known(f, &texts, &fields)).map(str::to_owned).collect())
}

/// Read every `.py` file under `megatron_dir/megatron`.
fn read_sources(megatron_dir: &Path) -> Result<Vec<String>> {
    let base = megatron_dir.join("megatron");
    let mut texts = Vec::new();
    collect_py(&base, &mut texts)?;
    Ok(texts)
}

/// Recursively collect Python source text.
fn collect_py(dir: &Path, texts: &mut Vec<String>) -> Result<()> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| OflError::new(format!("cannot read {}: {e}", dir.display())))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_py(&path, texts)?;
        } else if path.extension().is_some_and(|e| e == "py") {
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            texts.push(text);
        }
    }
    Ok(())
}

/// Check every config flag (and the launch-added ones) against the pinned source.
///
/// # Errors
///
/// [`OflError`] when the upstream fetch fails or the config cannot be read.
pub fn verify(root: &Path) -> Result<i32> {
    let ups = upstreams::load(root)?;
    let up = ups
        .iter()
        .find(|u| u.name == UPSTREAM)
        .ok_or_else(|| OflError::new(format!("configs/upstreams.toml has no [{UPSTREAM}]")))?;
    let megatron_dir = upstreams::ensure_checkout(root, up)?;
    let raw = load_toml(&root.join(DEFAULT_CONFIG))?;
    let mut flags = config_flags(&raw, true)?;
    flags.extend(LAUNCH_FLAGS.iter().map(|f| (*f).to_owned()));
    let missing = unknown_flags(&flags, &megatron_dir)?;
    let total = flags
        .iter()
        .filter(|f| f.starts_with("--"))
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    if !missing.is_empty() {
        let found = total.saturating_sub(missing.len());
        println!("FAIL {found}/{total} flags found in Megatron-LM {}", &up.sha[..9]);
        for flag in &missing {
            println!("  unknown flag: {flag}");
        }
        return Ok(1);
    }
    println!("ok   all {total} flags found in Megatron-LM {} (name check only)", &up.sha[..9]);
    Ok(0)
}
