//! Loading and checking everything under `configs/` and `manifests/`.

use std::path::{Path, PathBuf};

use crate::error::{OflError, Result};
use crate::upstreams::{self, Upstream};

/// Phase config directories, in roadmap order.
const PHASE_DIRS: [&str; 4] = ["phase1-dense", "phase1-moe", "phase2-post", "phase3-diloco"];

/// `configs/phase1-dense/smoke-tiny.toml`, fully typed.
#[derive(Clone, Debug)]
pub struct SmokeTiny {
    pub factory: String,
    pub vocab: i64,
    pub sequence_length: i64,
    pub sequences: i64,
    pub noise: f64,
    pub corpus_seed: i64,
    pub steps: i64,
    pub dev_steps: i64,
    pub batch_sequences: i64,
    pub lr: f64,
    pub warmup_steps: i64,
    pub init_seed: i64,
    pub loss_drop_ratio: f64,
}

/// `configs/phase3-diloco/smoke-reference.toml`, fully typed.
#[derive(Clone, Debug)]
pub struct SmokeDiloco {
    pub workers: i64,
    pub vocab: i64,
    pub sequence_length: i64,
    pub d_model: i64,
    pub n_layers: i64,
    pub n_heads: i64,
    pub outer_rounds: i64,
    pub inner_steps: i64,
    pub inner_lr: f64,
    pub outer_lr: f64,
    pub outer_momentum: f64,
    pub batch_size: i64,
    pub noise: f64,
    pub seed: i64,
    pub loss_drop_ratio: f64,
}

/// Read and parse one TOML file.
///
/// # Errors
///
/// [`OflError`] when the file cannot be read or parsed.
pub fn load_toml(path: &Path) -> Result<toml::Table> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| OflError::new(format!("{}: {e}", path.display())))?;
    toml::from_str(&text).map_err(|e| OflError::new(format!("{}: {e}", path.display())))
}

/// Look up a slash-separated key path in a TOML table (`"data/source"`).
#[must_use]
pub fn path<'a>(raw: &'a toml::Table, keys: &str) -> Option<&'a toml::Value> {
    let mut current: Option<&toml::Value> = None;
    for (i, key) in keys.split('/').enumerate() {
        current = if i == 0 { raw.get(key) } else { current.and_then(|v| v.get(key)) };
    }
    current
}

/// Load the tiny OLMo-core smoke config.
///
/// # Errors
///
/// [`OflError`] when the file is missing, unparseable, or a key has the wrong type.
pub fn load_smoke_tiny(root: &Path) -> Result<SmokeTiny> {
    let raw = load_toml(&root.join("configs/phase1-dense/smoke-tiny.toml"))?;
    let int = |table: &str, key: &str| integer(&raw, table, key);
    let float = |table: &str, key: &str| real(&raw, table, key);
    Ok(SmokeTiny {
        factory: string(&raw, "model", "factory")?,
        vocab: int("model", "vocab")?,
        sequence_length: int("data", "sequence_length")?,
        sequences: int("data", "sequences")?,
        noise: float("data", "noise")?,
        corpus_seed: int("data", "corpus_seed")?,
        steps: int("train", "steps")?,
        dev_steps: int("train", "dev_steps")?,
        batch_sequences: int("train", "batch_sequences")?,
        lr: float("train", "lr")?,
        warmup_steps: int("train", "warmup_steps")?,
        init_seed: int("train", "init_seed")?,
        loss_drop_ratio: float("assert", "loss_drop_ratio")?,
    })
}

/// Load the reference `DiLoCo` smoke config.
///
/// # Errors
///
/// [`OflError`] when the file is missing, unparseable, or a key has the wrong type.
pub fn load_smoke_diloco(root: &Path) -> Result<SmokeDiloco> {
    let raw = load_toml(&root.join("configs/phase3-diloco/smoke-reference.toml"))?;
    let int = |table: &str, key: &str| integer(&raw, table, key);
    let float = |table: &str, key: &str| real(&raw, table, key);
    Ok(SmokeDiloco {
        workers: int("run", "workers")?,
        vocab: int("model", "vocab")?,
        sequence_length: int("model", "sequence_length")?,
        d_model: int("model", "d_model")?,
        n_layers: int("model", "n_layers")?,
        n_heads: int("model", "n_heads")?,
        outer_rounds: int("diloco", "outer_rounds")?,
        inner_steps: int("diloco", "inner_steps")?,
        inner_lr: float("diloco", "inner_lr")?,
        outer_lr: float("diloco", "outer_lr")?,
        outer_momentum: float("diloco", "outer_momentum")?,
        batch_size: int("data", "batch_size")?,
        noise: float("data", "noise")?,
        seed: int("run", "seed")?,
        loss_drop_ratio: float("assert", "loss_drop_ratio")?,
    })
}

/// A required string field inside a table.
fn string(raw: &toml::Table, table: &str, key: &str) -> Result<String> {
    raw.get(table)
        .and_then(|t| t.get(key))
        .and_then(toml::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            OflError::new(format!("smoke config: [{table}].{key} missing or not a string"))
        })
}

/// A required integer field inside a table.
fn integer(raw: &toml::Table, table: &str, key: &str) -> Result<i64> {
    raw.get(table).and_then(|t| t.get(key)).and_then(toml::Value::as_integer).ok_or_else(|| {
        OflError::new(format!("smoke config: [{table}].{key} missing or not an integer"))
    })
}

/// A required float field inside a table (integers are accepted too).
fn real(raw: &toml::Table, table: &str, key: &str) -> Result<f64> {
    let value = raw
        .get(table)
        .and_then(|t| t.get(key))
        .ok_or_else(|| OflError::new(format!("smoke config: [{table}].{key} missing")))?;
    // Config numbers are small; the i64->f64 cast cannot lose meaningful precision.
    #[expect(clippy::cast_precision_loss, reason = "config values are far below 2^52")]
    // ubs:ignore[rust.casts.as-casts] -- config integer to f64; values are far below 2^52, no meaningful truncation.
    let as_float = value.as_integer().map(|i| i as f64);
    value
        .as_float()
        .or(as_float)
        .ok_or_else(|| OflError::new(format!("smoke config: [{table}].{key} not a number")))
}

/// Parse every config and manifest and cross-check pins. Returns problems.
///
/// # Errors
///
/// Never: problems are returned as strings, matching the Python `ofl check`.
#[must_use]
pub fn check(root: &Path) -> Vec<String> {
    let ups = match upstreams::load(root) {
        Ok(ups) => ups,
        Err(e) => return vec![e.message],
    };
    let mut out = upstreams::problems(&ups);
    out.extend(check_manifests(root));
    for phase in PHASE_DIRS {
        out.extend(check_phase(root, phase, &ups));
    }
    out.extend(check_smoke_loads(root));
    out
}

/// Check one phase directory's configs.
fn check_phase(root: &Path, phase: &str, ups: &[Upstream]) -> Vec<String> {
    let dir = root.join("configs").join(phase);
    let files = toml_files(&dir);
    let mut out = Vec::new();
    if files.is_empty() {
        out.push(format!("configs/{phase}: no config files"));
    }
    for path in files {
        // ubs:ignore[rust.perf.string-alloc-in-loop] -- one diagnostic path label per config file; a handful of files, not a hot path.
        let rel =
            format!("configs/{phase}/{}", path.file_name().unwrap_or_default().to_string_lossy());
        let raw = match load_toml(&path) {
            Ok(raw) => raw,
            Err(e) => {
                out.push(e.message);
                continue;
            }
        };
        out.extend(check_one(phase, &path, &rel, &raw, ups));
    }
    out
}

/// Dispatch per-phase checks for one parsed config.
fn check_one(
    phase: &str,
    path: &Path,
    rel: &str,
    raw: &toml::Table,
    ups: &[Upstream],
) -> Vec<String> {
    // Only the phase-3 configs are in the upstream's own format and carry no [meta].
    let mut out =
        if phase == "phase3-diloco" { Vec::new() } else { check_meta(rel, raw, phase, ups) };
    match phase {
        "phase1-dense" if path.file_name().is_some_and(|n| n != "smoke-tiny.toml") => {
            out.extend(check_overrides(rel, raw));
        }
        "phase1-moe" => out.extend(check_moe(rel, raw)),
        "phase2-post" => out.extend(check_post_stages(rel, raw, ups)),
        _ => {}
    }
    out
}

/// `[meta]` must name the phase and a known upstream.
fn check_meta(rel: &str, raw: &toml::Table, phase: &str, ups: &[Upstream]) -> Vec<String> {
    let Some(meta) = raw.get("meta").and_then(toml::Value::as_table) else {
        return vec![format!("{rel}: missing [meta]")];
    };
    let mut out = Vec::new();
    let got_phase = meta.get("phase").and_then(toml::Value::as_str);
    if got_phase != Some(phase) {
        out.push(format!("{rel}: [meta].phase must be {phase:?}, got {got_phase:?}"));
    }
    let upstream = meta.get("upstream").and_then(toml::Value::as_str);
    if !ups.iter().any(|u| Some(u.name.as_str()) == upstream) {
        out.push(format!("{rel}: [meta].upstream {upstream:?} is not in upstreams.toml"));
    }
    out
}

/// Dense launch configs need overrides and a valid data source.
fn check_overrides(rel: &str, raw: &toml::Table) -> Vec<String> {
    let Some(overrides) = raw.get("overrides").and_then(toml::Value::as_table) else {
        return vec![format!("{rel}: [overrides] missing or empty")];
    };
    if overrides.is_empty() {
        return vec![format!("{rel}: [overrides] missing or empty")];
    }
    let mut out: Vec<String> = overrides
        .iter()
        .filter(|(_, v)| !is_override_value(v))
        .map(|(k, v)| format!("{rel}: override {k:?} has unsupported value type {}", type_name(v)))
        .collect();
    let source = raw.get("data").and_then(|d| d.get("source")).and_then(toml::Value::as_str);
    if !matches!(source, Some("mix" | "glob")) {
        out.push(format!("{rel}: [data].source must be 'mix' or 'glob'"));
    }
    out
}

/// Override values the upstream dotted-override parser accepts.
const fn is_override_value(v: &toml::Value) -> bool {
    matches!(
        v,
        toml::Value::Boolean(_)
            | toml::Value::Integer(_)
            | toml::Value::Float(_)
            | toml::Value::String(_)
            | toml::Value::Array(_)
    )
}

/// `MoE` configs need flag tables, unique flags and parallelism integers.
fn check_moe(rel: &str, raw: &toml::Table) -> Vec<String> {
    let Some(flags) = raw.get("flags").and_then(toml::Value::as_table) else {
        return vec![format!("{rel}: missing [flags.*] tables")];
    };
    let mut out = Vec::new();
    let mut seen: Vec<&str> = Vec::new();
    for (group, table) in flags {
        let Some(table) = table.as_table() else { continue };
        for (flag, value) in table {
            if !flag.starts_with("--") {
                // ubs:ignore[rust.perf.string-alloc-in-loop] -- diagnostic message construction, not a hot path.
                out.push(format!("{rel}: [flags.{group}] key {flag:?} does not start with --"));
            }
            if !is_flag_value(value) {
                // ubs:ignore[rust.perf.string-alloc-in-loop] -- diagnostic message construction, not a hot path.
                out.push(format!("{rel}: {flag} has unsupported value type {}", type_name(value)));
            }
            if seen.contains(&flag.as_str()) {
                // ubs:ignore[rust.perf.string-alloc-in-loop] -- diagnostic message construction, not a hot path.
                out.push(format!("{rel}: {flag} appears twice"));
            }
            seen.push(flag);
        }
    }
    out.extend(check_moe_parallelism(rel, raw, flags));
    out
}

/// `MoE` expert parallelism and GPU count must be positive integers.
fn check_moe_parallelism(rel: &str, raw: &toml::Table, flags: &toml::Table) -> Vec<String> {
    let mut out = Vec::new();
    let ep = flags
        .get("parallelism")
        .and_then(|t| t.get("--expert-model-parallel-size"))
        .and_then(toml::Value::as_integer);
    if ep.is_none_or(|v| v < 1) {
        out.push(format!("{rel}: --expert-model-parallel-size must be a positive integer"));
    }
    let gpus =
        raw.get("launch").and_then(|l| l.get("gpus_per_node")).and_then(toml::Value::as_integer);
    if gpus.is_none() {
        out.push(format!("{rel}: [launch].gpus_per_node must be an integer"));
    }
    out
}

/// Flag values Megatron's argparse accepts: bool, int or string.
const fn is_flag_value(v: &toml::Value) -> bool {
    matches!(v, toml::Value::Boolean(_) | toml::Value::Integer(_) | toml::Value::String(_))
}

/// Phase 2 stages must each name a known upstream.
fn check_post_stages(rel: &str, raw: &toml::Table, ups: &[Upstream]) -> Vec<String> {
    let Some(stages) = raw.get("stage").and_then(toml::Value::as_array) else {
        return Vec::new();
    };
    stages
        .iter()
        .filter(|stage| {
            let up = stage.get("upstream").and_then(toml::Value::as_str);
            !ups.iter().any(|u| Some(u.name.as_str()) == up)
        })
        .map(|stage| {
            let name = stage.get("name").and_then(toml::Value::as_str).unwrap_or("?");
            format!("{rel}: stage {name:?} names an unknown upstream")
        })
        .collect()
}

/// Both smoke configs must load with the right types.
fn check_smoke_loads(root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    if let Err(e) = load_smoke_tiny(root) {
        out.push(format!("smoke config: {e}"));
    }
    if let Err(e) = load_smoke_diloco(root) {
        out.push(format!("smoke config: {e}"));
    }
    out
}

/// Manifest entries need their required keys and 40-hex revisions.
fn check_manifests(root: &Path) -> Vec<String> {
    let path = root.join("manifests").join("datasets.toml");
    let rel = "manifests/datasets.toml";
    let raw = match load_toml(&path) {
        Ok(raw) => raw,
        Err(e) => return vec![e.message],
    };
    let mut out = Vec::new();
    for (kind, required) in [
        ("dataset", ["id", "url", "revision", "licence"].as_slice()),
        ("tokenizer", ["id", "url", "revision", "licence"].as_slice()),
        ("mix", ["name", "defined-in", "path"].as_slice()),
    ] {
        out.extend(check_manifest_kind(&raw, rel, kind, required));
    }
    out
}

/// Check one `[[kind]]` array in the manifest.
fn check_manifest_kind(raw: &toml::Table, rel: &str, kind: &str, required: &[&str]) -> Vec<String> {
    let entries = raw.get(kind).and_then(toml::Value::as_array);
    let Some(entries) = entries else {
        return vec![format!("{rel}: no [[{kind}]] entries")];
    };
    if entries.is_empty() {
        return vec![format!("{rel}: no [[{kind}]] entries")];
    }
    let mut out = Vec::new();
    for entry in entries {
        let ident = entry
            .get("id")
            .or_else(|| entry.get("name"))
            .and_then(toml::Value::as_str)
            .unwrap_or("?");
        for key in required {
            if entry.get(*key).is_none() {
                // ubs:ignore[rust.perf.string-alloc-in-loop] -- diagnostic message construction, not a hot path.
                out.push(format!("{rel}: [[{kind}]] {ident} missing {key:?}"));
            }
        }
        if let Some(rev) = entry.get("revision").and_then(toml::Value::as_str)
            && !upstreams::SHA_RE.is_match(rev)
        {
            // ubs:ignore[rust.perf.string-alloc-in-loop] -- diagnostic message construction, not a hot path.
            out.push(format!("{rel}: [[{kind}]] {ident} revision is not a 40-hex commit"));
        }
    }
    out
}

/// Sorted `*.toml` files directly inside `dir`.
fn toml_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "toml"))
        .collect();
    files.sort();
    files
}

/// TOML value type name for error messages.
const fn type_name(v: &toml::Value) -> &'static str {
    match v {
        toml::Value::String(_) => "str",
        toml::Value::Integer(_) => "int",
        toml::Value::Float(_) => "float",
        toml::Value::Boolean(_) => "bool",
        toml::Value::Datetime(_) => "datetime",
        toml::Value::Array(_) => "list",
        toml::Value::Table(_) => "dict",
    }
}
