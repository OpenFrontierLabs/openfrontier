#![cfg(test)]
#![expect(clippy::unwrap_used, reason = "test assertions panic on failure")]
//! CLI contract, deploy and repo-root tests.

mod common;

use std::path::Path;

use ofl::repo::find_root;
use ofl::{cli, deploy};

/// A broken CLI definition must fail the suite, not panic on every invocation.
///
/// Regression gate for the clap auto-help collision.
#[test]
fn cli_definition_is_internally_consistent() {
    cli::debug_assert();
}

#[test]
fn makefile_has_exactly_the_documented_targets_as_one_line_wrappers() {
    let text = std::fs::read_to_string(common::root().join("Makefile")).unwrap();
    let phony_line = text.lines().find(|l| l.starts_with(".PHONY:")).unwrap();
    let phony: std::collections::BTreeSet<&str> =
        phony_line.trim_start_matches(".PHONY:").split_whitespace().collect();
    let targets: std::collections::BTreeSet<&str> =
        cli::TARGETS.iter().map(|(name, _)| *name).collect();
    assert_eq!(phony, targets);
    let recipes: Vec<&str> = text.lines().filter(|l| l.starts_with('\t')).map(str::trim).collect();
    assert_ne!(recipes, Vec::<&str>::new());
    for recipe in &recipes {
        assert!(recipe.starts_with("@cargo run -q -- "), "recipe: {recipe}");
    }
    assert!(text.contains("ofl deploy --env $(ENV)") || text.contains("deploy --env $(ENV)"));
}

#[test]
fn deploy_dry_run_publishes_nothing() {
    // The test process runs inside the checkout, so find_root succeeds.
    assert_eq!(deploy::deploy("dry-run").unwrap(), 0);
}

#[test]
fn deploy_real_env_refuses_with_nonzero_exit() {
    let err = deploy::deploy("huggingface").unwrap_err();
    assert_eq!(err.code, 2);
    assert!(err.message.contains("not implemented yet: nothing to publish"));
}

#[test]
fn deploy_unknown_env_lists_the_known_ones() {
    let err = deploy::deploy("prod").unwrap_err();
    assert_eq!(err.code, 2);
    assert!(err.message.contains("dry-run"));
    assert!(err.message.contains("huggingface"));
}

#[test]
fn deploy_dry_run_lists_checkpoints_that_exist() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("checkpoints/run0/step2000")).unwrap();
    std::fs::create_dir_all(tmp.path().join("checkpoints/run0/notes")).unwrap();
    let found = deploy::find_checkpoints(tmp.path());
    let names: Vec<String> =
        found.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
    assert_eq!(names, vec!["step2000".to_owned()]);
}

#[test]
fn find_root_outside_a_checkout_is_an_error() {
    let tmp = tempfile::tempdir().unwrap();
    let err = find_root(Some(tmp.path())).unwrap_err();
    assert!(err.message.contains("not inside an openfrontier checkout"));
}

#[test]
fn find_root_finds_the_checkout_from_a_subdirectory() {
    let root = common::root();
    let nested = root.join("configs/phase1-dense");
    assert_eq!(find_root(Some(&nested)).unwrap(), root);
}

/// The unbuilt phases must refuse with code 2 and a "not built" message.
#[test]
fn unbuilt_phases_fail_loudly() {
    for phase in ["post", "diloco"] {
        let err = ofl::phases::not_built(phase).unwrap_err();
        assert_eq!(err.code, 2);
        assert!(err.message.contains("not built"));
    }
}

#[test]
fn help_lists_every_target() {
    // `help` prints to stdout; capture via the public target table instead of
    // spawning, asserting every documented target has a description.
    for (name, text) in cli::TARGETS {
        assert_ne!(name, "");
        assert_ne!(text, "");
    }
    assert_eq!(cli::TARGETS.len(), 11);
}

#[test]
fn python_module_builds_the_expected_argv() {
    let argv = ofl::repo::python_module(
        "/py",
        "smoke.olmo_tiny",
        &["--steps".to_owned(), "60".to_owned()],
    );
    assert_eq!(argv, vec!["/py", "-m", "smoke.olmo_tiny", "--steps", "60"]);
}

#[test]
fn smoke_python_prefers_the_venv_when_present() {
    let tmp = tempfile::tempdir().unwrap();
    let venv_bin = tmp.path().join("smoke/.venv/bin");
    std::fs::create_dir_all(&venv_bin).unwrap();
    std::fs::write(venv_bin.join("python"), "").unwrap();
    assert!(ofl::repo::smoke_python(tmp.path()).ends_with("smoke/.venv/bin/python"));
    assert_eq!(ofl::repo::smoke_python(Path::new("/nonexistent-ofl-root")), "python3");
}
