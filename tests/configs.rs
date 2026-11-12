#![cfg(test)]
#![expect(clippy::unwrap_used, reason = "test assertions panic on failure")]
//! Config, manifest and launcher command-building tests.

mod common;

use std::path::{Path, PathBuf};

use ofl::configs::{self, load_smoke_diloco, load_smoke_tiny};
use ofl::error::OflError;
use ofl::{moe, run0, upstreams};

/// Copy `configs/` and `manifests/` into a fresh tree (the copied
/// `configs/upstreams.toml` is the root marker), so config checks can mutate
/// them without touching the real checkout.
fn copy_tree(dest: &Path) -> PathBuf {
    let src = common::root();
    for name in ["configs", "manifests"] {
        copy_dir(&src.join(name), &dest.join(name));
    }
    dest.to_path_buf()
}

/// Recursively copy a directory.
fn copy_dir(src: &Path, dest: &Path) {
    std::fs::create_dir_all(dest).unwrap();
    for entry in std::fs::read_dir(src).unwrap().flatten() {
        let path = entry.path();
        let target = dest.join(entry.file_name());
        if path.is_dir() {
            copy_dir(&path, &target);
        } else {
            std::fs::copy(&path, &target).unwrap();
        }
    }
}

fn dense_plan(root: &Path, config: &str) -> Result<run0::Launch, OflError> {
    dense_plan_with(root, config, None, false)
}

fn dense_plan_with(
    root: &Path,
    config: &str,
    data_glob: Option<&str>,
    dry_run: bool,
) -> Result<run0::Launch, OflError> {
    let dirs = run0::Dirs {
        script_dir: PathBuf::from("/up"),
        save_folder: PathBuf::from("/save"),
        work_dir: PathBuf::from("/work"),
    };
    let opts = run0::Options {
        python: "python".to_owned(),
        nproc: 8,
        run_name: None,
        data_glob: data_glob.map(str::to_owned),
        dry_run,
    };
    run0::build(root, &root.join("configs/phase1-dense").join(config), &dirs, &opts)
}

fn moe_plan(root: &Path, setup_check: bool) -> Result<moe::MoeLaunch, OflError> {
    let dirs = moe::MoeDirs {
        megatron_dir: PathBuf::from("/m"),
        tokenizer_dir: PathBuf::from("/tok"),
        data_args_path: PathBuf::from("/data-args.txt"),
        data_cache_path: PathBuf::from("/cache"),
        save_dir: PathBuf::from("/save"),
    };
    moe::build(
        &root.join("configs/phase1-moe/handbook-moe-6p9b.toml"),
        &dirs,
        "python",
        setup_check,
    )
}

#[test]
fn shipped_configs_and_manifests_pass() {
    assert_eq!(configs::check(&common::root()), Vec::<String>::new());
}

#[test]
fn smoke_configs_load_with_expected_types() {
    let root = common::root();
    let tiny = load_smoke_tiny(&root).unwrap();
    assert!(tiny.steps >= 20 && (0.0..1.0).contains(&tiny.loss_drop_ratio));
    let reference = load_smoke_diloco(&root).unwrap();
    assert!(reference.workers >= 2 && (reference.outer_momentum - 0.9).abs() < f64::EPSILON);
}

#[test]
fn upstream_pins_are_full_shas_and_olmo_core_matches_smoke_project() {
    let root = common::root();
    let ups = upstreams::load(&root).unwrap();
    assert_eq!(upstreams::problems(&ups), Vec::<String>::new());
    let smoke = std::fs::read_to_string(root.join("smoke/pyproject.toml")).unwrap();
    let parsed: toml::Table = toml::from_str(&smoke).unwrap();
    let pinned = configs::path(&parsed, "tool/uv/sources/ai2-olmo-core/rev")
        .and_then(toml::Value::as_str)
        .unwrap();
    let olmo = ups.iter().find(|u| u.name == "olmo-core").unwrap();
    assert_eq!(pinned, olmo.sha);
}

#[test]
fn config_with_unknown_upstream_is_reported() {
    let tmp = tempfile::tempdir().unwrap();
    let tree = copy_tree(tmp.path());
    let path = tree.join("configs/phase1-moe/handbook-moe-6p9b.toml");
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, text.replace("upstream = \"megatron-lm\"", "upstream = \"nope\""))
        .unwrap();
    assert!(configs::check(&tree).iter().any(|p| p.contains("nope")));
}

#[test]
fn manifest_revision_must_be_a_commit() {
    let tmp = tempfile::tempdir().unwrap();
    let tree = copy_tree(tmp.path());
    let path = tree.join("manifests/datasets.toml");
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, text.replace("87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "main"))
        .unwrap();
    assert!(configs::check(&tree).iter().any(|p| p.contains("not a 40-hex")));
}

#[test]
fn unparseable_config_is_reported() {
    let tmp = tempfile::tempdir().unwrap();
    let tree = copy_tree(tmp.path());
    std::fs::write(tree.join("configs/phase2-post/post-training.toml"), "[[[").unwrap();
    assert!(configs::check(&tree).iter().any(|p| p.contains("post-training.toml")));
}

#[test]
fn run0_command_targets_the_pinned_olmo_core_example() {
    let plan = dense_plan(&common::root(), "run0.toml").unwrap();
    let argv = &plan.argv;
    assert_eq!(argv[..3], ["python", "-m", "torch.distributed.run"]);
    assert!(argv.contains(&"--nproc-per-node=8".to_owned()));
    let factory = argv.iter().position(|a| a == "--model-factory").unwrap();
    assert_eq!(argv[factory + 1], "olmo2_1B_v2");
    assert!(argv.contains(&"/up/src/examples/llm/train.py".to_owned()));
    assert!(argv.contains(&"--data_loader.global_batch_size=2097152".to_owned()));
    assert!(argv.contains(&"--train_module.optim.betas=[0.9,0.95]".to_owned()));
    assert!(argv.contains(&"--trainer.callbacks.lm_evaluator.enabled=false".to_owned()));
    assert!(!argv.contains(&"--dry-run".to_owned()));
}

#[test]
fn run0_dry_run_flag_is_forwarded_to_the_upstream_script() {
    let plan = dense_plan_with(&common::root(), "run0.toml", None, true).unwrap();
    assert!(plan.script_args.contains(&"--dry-run".to_owned()));
}

#[test]
fn fineweb_config_requires_user_supplied_shards() {
    let root = common::root();
    let err = dense_plan(&root, "run0-fineweb-edu.toml").unwrap_err();
    assert!(err.message.contains("--data-glob"));
    let plan = dense_plan_with(&root, "run0-fineweb-edu.toml", Some("/d/*.npy"), false).unwrap();
    assert!(plan.argv.contains(&"--dataset.paths=[\"/d/*.npy\"]".to_owned()));
    assert!(plan.argv.contains(&"--dataset.expand_glob=true".to_owned()));
}

#[test]
fn mix_config_rejects_a_data_glob() {
    let err = dense_plan_with(&common::root(), "run0.toml", Some("/d/*.npy"), false).unwrap_err();
    assert!(err.message.contains("does not apply"));
}

#[test]
fn format_override_renders_upstream_syntax() {
    assert_eq!(run0::format_override(&toml::Value::Boolean(true)), "true");
    let betas = toml::Value::Array(vec![toml::Value::Float(0.9), toml::Value::Float(0.95)]);
    assert_eq!(run0::format_override(&betas), "[0.9,0.95]");
    assert_eq!(run0::format_override(&toml::Value::String("null".into())), "null");
    assert_eq!(run0::format_override(&toml::Value::Integer(2000)), "2000");
}

#[test]
fn moe_command_matches_the_handbook_model_and_parallelism() {
    let plan = moe_plan(&common::root(), false).unwrap();
    let argv = &plan.argv;
    assert_eq!(argv[0], "python");
    assert!(argv.contains(&"--nproc_per_node=8".to_owned()));
    let val = |flag: &str| -> String {
        let idx = argv.iter().position(|a| a == flag).unwrap();
        argv[idx + 1].clone()
    };
    assert_eq!(val("--num-experts"), "64");
    assert_eq!(val("--moe-router-topk"), "8");
    assert_eq!(val("--moe-ffn-hidden-size"), "1024");
    assert_eq!(val("--expert-model-parallel-size"), "8");
    assert_eq!(val("--num-layers"), "16");
    assert_eq!(val("--hidden-size"), "2048");
    assert_eq!(val("--tokenizer-model"), "/tok");
    assert!(argv.contains(&"--swiglu".to_owned()));
    assert!(!argv.contains(&"--exit-interval".to_owned()));
}

#[test]
fn moe_setup_check_adds_the_handbook_stop_condition() {
    let plan = moe_plan(&common::root(), true).unwrap();
    let idx = plan.argv.iter().position(|a| a == "--exit-interval").unwrap();
    assert_eq!(plan.argv[idx + 1], "120");
}

#[test]
fn moe_env_points_pythonpath_at_the_checkout() {
    let plan = moe_plan(&common::root(), false).unwrap();
    let env: std::collections::HashMap<_, _> = plan.env.iter().cloned().collect();
    assert_eq!(env.get("PYTHONPATH").unwrap(), "/m");
    assert_eq!(env.get("CUDA_DEVICE_MAX_CONNECTIONS").unwrap(), "1");
}

#[test]
fn unknown_flags_detects_missing_argparse_names_and_accepts_dataclass_fields() {
    let tmp = tempfile::tempdir().unwrap();
    let pkg = tmp.path().join("megatron/training");
    std::fs::create_dir_all(&pkg).unwrap();
    std::fs::write(pkg.join("arguments.py"), "group.add_argument('--num-experts', type=int)\n")
        .unwrap();
    std::fs::write(
        pkg.join("config.py"),
        "class C:\n    tensorboard_dir: str | None = None\n    mmap_bin_files: bool = True\n",
    )
    .unwrap();
    let flags: Vec<String> =
        ["--num-experts", "--tensorboard-dir", "--no-mmap-bin-files", "--made-up-flag"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
    assert_eq!(moe::unknown_flags(&flags, tmp.path()).unwrap(), vec!["--made-up-flag".to_owned()]);
}
