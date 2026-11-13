//! Validation of `ledger/*.json` against `ledger/schema.json` plus cross-reference rules.

use std::path::Path;

use serde_json::Value;

use crate::error::{OflError, Result};

/// Schema location relative to the repo root.
const SCHEMA_RELPATH: &str = "ledger/schema.json";

/// Read and parse the ledger schema.
///
/// # Errors
///
/// [`OflError`] when the file cannot be read, is not JSON, or is not an object.
pub fn load_schema(root: &Path) -> Result<Value> {
    let path = root.join(SCHEMA_RELPATH);
    let text = std::fs::read_to_string(&path)
        .map_err(|e| OflError::new(format!("{SCHEMA_RELPATH}: {e}")))?;
    let schema: Value = serde_json::from_str(&text)
        .map_err(|e| OflError::new(format!("{SCHEMA_RELPATH} is not valid JSON: {e}")))?;
    if !schema.is_object() {
        return Err(OflError::new(format!("{SCHEMA_RELPATH} must be a JSON object")));
    }
    Ok(schema)
}

/// Schema validation plus rules JSON Schema cannot express. Returns problems.
#[must_use]
pub fn validate_document(doc: &Value, schema: &Value, filename: &str) -> Vec<String> {
    let validator =
        jsonschema::draft202012::options().should_validate_formats(true).offline().build(schema);
    let validator = match validator {
        Ok(v) => v,
        Err(e) => return vec![format!("{filename}: schema cannot be compiled: {e}")],
    };
    let mut problems: Vec<String> = validator
        .iter_errors(doc)
        .map(|err| {
            let path = err.instance_path().as_str();
            let path = if path.is_empty() { "<root>".to_owned() } else { path.to_owned() };
            format!("{filename}: {path}: {err}")
        })
        .collect();
    if !problems.is_empty() {
        return problems;
    }
    problems.extend(cross_rules(doc, filename));
    problems
}

/// Rules beyond the schema: example marker, period order, unique ids, island refs.
fn cross_rules(doc: &Value, filename: &str) -> Vec<String> {
    let mut out = Vec::new();
    out.extend(example_marker_rule(doc, filename));
    out.extend(period_rule(doc, filename));
    out.extend(unique_islands_rule(doc, filename));
    out.extend(node_rules(doc, filename));
    out
}

/// `example: true` exactly when the filename contains EXAMPLE.
fn example_marker_rule(doc: &Value, filename: &str) -> Vec<String> {
    let is_example_name = filename.contains("EXAMPLE");
    let marked = doc.get("example").and_then(Value::as_bool).unwrap_or(false);
    if marked != is_example_name {
        return vec![format!(
            "{filename}: 'example': true must be set exactly when the filename contains EXAMPLE"
        )];
    }
    Vec::new()
}

/// `period.start` must not be after `period.end` (both already schema-checked dates).
fn period_rule(doc: &Value, filename: &str) -> Vec<String> {
    let start = doc.pointer("/period/start").and_then(Value::as_str);
    let end = doc.pointer("/period/end").and_then(Value::as_str);
    if let (Some(start), Some(end)) = (start, end)
        && start > end
    {
        return vec![format!("{filename}: period.start is after period.end")];
    }
    Vec::new()
}

/// Island ids must be unique within the file.
fn unique_islands_rule(doc: &Value, filename: &str) -> Vec<String> {
    let ids: Vec<&str> = doc
        .get("islands")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|i| i.get("id").and_then(Value::as_str)).collect())
        .unwrap_or_default();
    if has_duplicates(&ids) {
        return vec![format!("{filename}: duplicate island id")];
    }
    Vec::new()
}

/// Node ids must be unique and every island reference must be declared.
fn node_rules(doc: &Value, filename: &str) -> Vec<String> {
    let Some(nodes) = doc.get("nodes").and_then(Value::as_array) else {
        return Vec::new();
    };
    let ids: Vec<&str> =
        nodes.iter().filter_map(|n| n.get("node_id").and_then(Value::as_str)).collect();
    let mut out = Vec::new();
    if has_duplicates(&ids) {
        out.push(format!("{filename}: duplicate node_id"));
    }
    let island_ids: Vec<&str> = doc
        .get("islands")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|i| i.get("id").and_then(Value::as_str)).collect())
        .unwrap_or_default();
    for node in nodes {
        out.extend(island_reference_rule(node, &island_ids, filename));
    }
    out
}

/// One node's island reference, when present, must be declared.
fn island_reference_rule(node: &Value, island_ids: &[&str], filename: &str) -> Vec<String> {
    let Some(island) = node.get("island").and_then(Value::as_str) else {
        return Vec::new();
    };
    if island_ids.contains(&island) {
        return Vec::new();
    }
    let node_id = node.get("node_id").and_then(Value::as_str).unwrap_or("?");
    vec![format!("{filename}: node {node_id} references unknown island {island:?}")]
}

/// True when any id appears more than once.
fn has_duplicates(ids: &[&str]) -> bool {
    let mut seen: Vec<&str> = Vec::with_capacity(ids.len());
    for id in ids {
        if seen.contains(id) {
            return true;
        }
        seen.push(id);
    }
    false
}

/// Validate the schema itself and every `ledger/*.json` data file.
#[must_use]
pub fn check(root: &Path) -> Vec<String> {
    let schema = match load_schema(root) {
        Ok(schema) => schema,
        Err(e) => return vec![e.message],
    };
    if let Err(e) = jsonschema::meta::validate(&schema) {
        return vec![format!("{SCHEMA_RELPATH}: {e}")];
    }
    let mut problems = Vec::new();
    let data_files = ledger_files(root);
    if data_files.is_empty() {
        problems.push("ledger/: no example file to validate against the schema".to_owned());
    }
    for path in data_files {
        problems.extend(check_file(&path, &schema));
    }
    problems
}

/// Validate one ledger data file.
fn check_file(path: &Path, schema: &Value) -> Vec<String> {
    let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) => return vec![format!("ledger/{name}: {e}")],
    };
    let doc: Value = match serde_json::from_str(&text) {
        Ok(doc) => doc,
        Err(e) => return vec![format!("ledger/{name}: {e}")],
    };
    validate_document(&doc, schema, &name)
}

/// Sorted ledger data files (everything but the schema itself).
fn ledger_files(root: &Path) -> Vec<std::path::PathBuf> {
    let Ok(entries) = std::fs::read_dir(root.join("ledger")) else {
        return Vec::new();
    };
    let mut files: Vec<std::path::PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension().is_some_and(|e| e == "json")
                && p.file_name().is_some_and(|n| n != "schema.json")
        })
        .collect();
    files.sort();
    files
}
