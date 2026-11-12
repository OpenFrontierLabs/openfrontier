#![expect(clippy::unwrap_used, reason = "test assertions panic on failure")]
//! Upstream pin loading and checkout-refusal tests.

mod common;

use std::path::Path;
use std::process::Command;

use ofl::upstreams::{self, Upstream};

#[test]
fn shipped_upstreams_load_and_have_no_problems() {
    let ups = upstreams::load(&common::root()).unwrap();
    assert!(ups.len() >= 8);
    assert_eq!(upstreams::problems(&ups), Vec::<String>::new());
    assert!(ups.iter().any(|u| u.name == "olmo-core"));
    assert!(ups.iter().any(|u| u.name == "megatron-lm"));
}

#[test]
fn problems_flags_bad_sha_and_non_https_repo() {
    let bad = Upstream {
        name: "x".to_owned(),
        repo: "http://example.com/x".to_owned(),
        reference: "main".to_owned(),
        sha: "not-a-sha".to_owned(),
        license: "MIT".to_owned(),
        verified: "2026-11-12".to_owned(),
        used_by: "test".to_owned(),
    };
    let problems = upstreams::problems(&[bad]);
    assert_eq!(problems.len(), 2);
    assert!(problems.iter().any(|p| p.contains("40 lowercase hex")));
    assert!(problems.iter().any(|p| p.contains("https URL")));
}

#[test]
fn existing_checkout_at_the_wrong_commit_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let repo = tmp.path().join(".cache/upstreams/olmo-core");
    std::fs::create_dir_all(&repo).unwrap();
    let git = |args: &[&str]| {
        let status = Command::new("git")
            .args(["-c", "user.name=t", "-c", "user.email=t@example.com"])
            .args(args)
            .current_dir(&repo)
            .status()
            .unwrap();
        assert!(status.success());
    };
    git(&["init", "-q"]);
    git(&["commit", "-q", "--allow-empty", "-m", "x"]);
    let up = Upstream {
        name: "olmo-core".to_owned(),
        repo: "https://example.invalid/x".to_owned(),
        reference: "v0".to_owned(),
        sha: "a".repeat(40),
        license: "MIT".to_owned(),
        verified: "d".to_owned(),
        used_by: "u".to_owned(),
    };
    let err = upstreams::ensure_checkout(tmp.path(), &up).unwrap_err();
    assert!(err.message.contains(&format!("expected {}", "a".repeat(40))));
}

#[test]
fn head_sha_is_none_for_a_non_checkout() {
    let tmp = tempfile::tempdir().unwrap();
    assert_eq!(upstreams::head_sha(tmp.path()).unwrap(), None);
    assert_eq!(upstreams::head_sha(Path::new("/nonexistent-ofl-path")).unwrap(), None);
}

#[test]
fn checkout_path_is_under_cache_upstreams() {
    let path = upstreams::checkout_path(Path::new("/repo"), "olmo-core");
    assert_eq!(path, Path::new("/repo/.cache/upstreams/olmo-core"));
}
