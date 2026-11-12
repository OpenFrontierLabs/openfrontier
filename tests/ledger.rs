#![cfg(test)]
#![expect(clippy::unwrap_used, reason = "test assertions panic on failure")]
//! Ledger schema and cross-reference validation tests.

mod common;

use std::path::Path;

use ofl::ledger;
use serde_json::{Value, json};

fn schema() -> Value {
    ledger::load_schema(&common::root()).unwrap()
}

fn example() -> Value {
    let path = common::root().join("ledger/example.EXAMPLE.json");
    let text = std::fs::read_to_string(path).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn problems(doc: &Value, name: &str) -> Vec<String> {
    ledger::validate_document(doc, &schema(), name)
}

#[test]
fn shipped_ledger_files_validate() {
    assert_eq!(ledger::check(&common::root()), Vec::<String>::new());
}

#[test]
fn example_is_marked_as_an_example_and_uses_fake_data() {
    let doc = example();
    assert_eq!(doc["example"], json!(true));
    for node in doc["nodes"].as_array().unwrap() {
        assert!(node["github_handle"].as_str().unwrap().starts_with("example-"));
    }
}

#[test]
fn real_ledger_file_may_not_claim_to_be_an_example() {
    assert!(problems(&example(), "run0-2026-11.json").iter().any(|p| p.contains("EXAMPLE")));
}

#[test]
fn example_file_must_carry_the_example_marker() {
    let mut doc = example();
    doc.as_object_mut().unwrap().remove("example");
    assert!(problems(&doc, "x.EXAMPLE.json").iter().any(|p| p.contains("EXAMPLE")));
}

#[test]
fn invalid_node_fields_are_rejected() {
    let cases: Vec<(&str, Value)> = vec![
        ("helpful_update_share", json!(1.5)),
        ("helpful_update_share", json!(-0.1)),
        ("gpu_hours", json!(-1)),
        ("tokens_processed", json!(1.5)),
        ("github_handle", json!("-bad-handle")),
        ("github_handle", json!("has space")),
        ("node_id", json!("Upper")),
    ];
    for (field, value) in cases {
        let mut doc = example();
        doc["nodes"][0][field] = value.clone();
        assert!(!problems(&doc, "x.EXAMPLE.json").is_empty(), "field {field} value {value}");
    }
}

#[test]
fn missing_node_field_is_rejected() {
    let mut doc = example();
    doc["nodes"][0].as_object_mut().unwrap().remove("tokens_processed");
    assert!(problems(&doc, "x.EXAMPLE.json").iter().any(|p| p.contains("tokens_processed")));
}

#[test]
fn unknown_keys_are_rejected() {
    let mut doc = example();
    doc["nodes"][0]["email"] = json!("someone@example.com");
    assert_ne!(problems(&doc, "x.EXAMPLE.json"), Vec::<String>::new());
}

#[test]
fn node_must_reference_a_declared_island() {
    let mut doc = example();
    doc["nodes"][0]["island"] = json!("no-such-island");
    assert!(problems(&doc, "x.EXAMPLE.json").iter().any(|p| p.contains("unknown island")));
}

#[test]
fn duplicate_node_ids_are_rejected() {
    let mut doc = example();
    let first = doc["nodes"][0].clone();
    doc["nodes"].as_array_mut().unwrap().push(first);
    assert!(problems(&doc, "x.EXAMPLE.json").iter().any(|p| p.contains("duplicate node_id")));
}

#[test]
fn period_must_not_run_backwards() {
    let mut doc = example();
    doc["period"] = json!({"start": "2000-02-01", "end": "2000-01-01"});
    assert!(problems(&doc, "x.EXAMPLE.json").iter().any(|p| p.contains("after period.end")));
}

#[test]
fn impossible_calendar_date_is_rejected() {
    let mut doc = example();
    doc["period"]["end"] = json!("2000-02-31");
    assert_ne!(problems(&doc, "x.EXAMPLE.json"), Vec::<String>::new());
}

#[test]
fn check_flags_a_directory_with_no_data_file() {
    let tmp = tempfile::tempdir().unwrap();
    let ledger_dir = tmp.path().join("ledger");
    std::fs::create_dir_all(&ledger_dir).unwrap();
    let schema_text = std::fs::read_to_string(common::root().join("ledger/schema.json")).unwrap();
    std::fs::write(ledger_dir.join("schema.json"), schema_text).unwrap();
    assert!(ledger::check(tmp.path()).iter().any(|p| p.contains("no example file")));
}

#[test]
fn duplicate_island_ids_are_rejected() {
    let mut doc = example();
    let first = doc["islands"][0].clone();
    doc["islands"].as_array_mut().unwrap().push(first);
    assert!(problems(&doc, "x.EXAMPLE.json").iter().any(|p| p.contains("duplicate island id")));
}

/// Guard: the schema file itself must stay a valid Draft 2020-12 document.
#[test]
fn schema_is_valid_draft_2020_12() {
    let root = common::root();
    let path: &Path = &root.join("ledger/schema.json");
    assert!(path.is_file());
    let schema = ledger::load_schema(&root).unwrap();
    assert!(jsonschema::meta::validate(&schema).is_ok());
}
