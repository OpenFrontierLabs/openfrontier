import shutil
import tomllib
from pathlib import Path

import pytest

from ofl import configs, moe, run0, upstreams
from ofl.repo import OflError


def copy_tree(root: Path, dest: Path) -> Path:
    for name in ("configs", "manifests"):
        shutil.copytree(root / name, dest / name)
    (dest / "pyproject.toml").write_text("", encoding="utf-8")
    return dest


def test_shipped_configs_and_manifests_pass(root: Path) -> None:
    assert configs.check(root) == []


def test_smoke_configs_load_with_expected_types(root: Path) -> None:
    tiny = configs.load_smoke_tiny(root)
    assert tiny.steps >= 20 and 0 < tiny.loss_drop_ratio < 1
    ref = configs.load_smoke_diloco(root)
    assert ref.workers >= 2 and ref.outer_momentum == 0.9


def test_upstream_pins_are_full_shas_and_olmo_core_matches_pyproject(root: Path) -> None:
    ups = upstreams.load(root)
    assert upstreams.problems(ups) == []
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    pinned = pyproject["tool"]["uv"]["sources"]["ai2-olmo-core"]["rev"]
    assert pinned == ups["olmo-core"].sha


def test_config_with_wrong_phase_or_unknown_upstream_is_reported(
    root: Path, tmp_path: Path
) -> None:
    tree = copy_tree(root, tmp_path)
    path = tree / "configs" / "phase1-moe" / "handbook-moe-6p9b.toml"
    path.write_text(
        path.read_text(encoding="utf-8").replace('upstream = "megatron-lm"', 'upstream = "nope"'),
        encoding="utf-8",
    )
    assert any("nope" in p for p in configs.check(tree))


def test_manifest_revision_must_be_a_commit(root: Path, tmp_path: Path) -> None:
    tree = copy_tree(root, tmp_path)
    path = tree / "manifests" / "datasets.toml"
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace("87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "main"), encoding="utf-8"
    )
    assert any("not a 40-hex" in p for p in configs.check(tree))


def test_unparseable_config_is_reported(root: Path, tmp_path: Path) -> None:
    tree = copy_tree(root, tmp_path)
    (tree / "configs" / "phase2-post" / "post-training.toml").write_text("[[[", encoding="utf-8")
    assert any("post-training.toml" in p for p in configs.check(tree))


def dense_plan(root: Path, config: str, **kwargs: object) -> run0.Launch:
    params: dict[str, object] = {
        "script_dir": Path("/up"),
        "python": "python",
        "nproc": 8,
        "run_name": None,
        "save_folder": Path("/save"),
        "work_dir": Path("/work"),
        "data_glob": None,
        "dry_run": False,
    }
    params.update(kwargs)
    return run0.build(root, root / "configs" / "phase1-dense" / config, **params)  # type: ignore[arg-type]


def test_run0_command_targets_the_pinned_olmo_core_example(root: Path) -> None:
    plan = dense_plan(root, "run0.toml")
    argv = plan.argv
    assert argv[:3] == ["python", "-m", "torch.distributed.run"]
    assert "--nproc-per-node=8" in argv
    assert argv[argv.index("--model-factory") + 1] == "olmo2_1B_v2"
    assert "/up/src/examples/llm/train.py" in argv
    assert "--data_loader.global_batch_size=2097152" in argv
    assert "--train_module.optim.betas=[0.9,0.95]" in argv
    assert "--trainer.callbacks.lm_evaluator.enabled=false" in argv
    assert "--dry-run" not in argv


def test_run0_dry_run_flag_is_forwarded_to_the_upstream_script(root: Path) -> None:
    assert "--dry-run" in dense_plan(root, "run0.toml", dry_run=True).script_args


def test_fineweb_config_requires_user_supplied_shards(root: Path) -> None:
    with pytest.raises(OflError, match="--data-glob"):
        dense_plan(root, "run0-fineweb-edu.toml")
    plan = dense_plan(root, "run0-fineweb-edu.toml", data_glob="/d/*.npy")
    assert '--dataset.paths=["/d/*.npy"]' in plan.argv
    assert "--dataset.expand_glob=true" in plan.argv


def test_mix_config_rejects_a_data_glob(root: Path) -> None:
    with pytest.raises(OflError, match="does not apply"):
        dense_plan(root, "run0.toml", data_glob="/d/*.npy")


def test_format_override_renders_upstream_syntax() -> None:
    assert run0.format_override(True) == "true"
    assert run0.format_override([0.9, 0.95]) == "[0.9,0.95]"
    assert run0.format_override("null") == "null"
    assert run0.format_override(2000) == "2000"


def moe_plan(root: Path, **kwargs: object) -> moe.MoeLaunch:
    params: dict[str, object] = {
        "megatron_dir": Path("/m"),
        "python": "python",
        "tokenizer_dir": Path("/tok"),
        "data_args_path": Path("/data-args.txt"),
        "data_cache_path": Path("/cache"),
        "save_dir": Path("/save"),
        "setup_check": False,
    }
    params.update(kwargs)
    return moe.build(root / "configs" / "phase1-moe" / "handbook-moe-6p9b.toml", **params)  # type: ignore[arg-type]


def test_moe_command_matches_the_handbook_model_and_parallelism(root: Path) -> None:
    argv = moe_plan(root).argv
    assert argv[0] == "python"
    assert "--nproc_per_node=8" in argv

    def val(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    assert val("--num-experts") == "64"
    assert val("--moe-router-topk") == "8"
    assert val("--moe-ffn-hidden-size") == "1024"
    assert val("--expert-model-parallel-size") == "8"
    assert val("--num-layers") == "16"
    assert val("--hidden-size") == "2048"
    assert val("--tokenizer-model") == "/tok"
    assert "--swiglu" in argv  # boolean flags carry no value
    assert "--exit-interval" not in argv


def test_moe_setup_check_adds_the_handbook_stop_condition(root: Path) -> None:
    argv = moe_plan(root, setup_check=True).argv
    assert argv[argv.index("--exit-interval") + 1] == "120"


def test_moe_env_points_pythonpath_at_the_checkout(root: Path) -> None:
    env = moe_plan(root).env
    assert env["PYTHONPATH"] == "/m"
    assert env["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"


def test_unknown_flags_detects_missing_argparse_names_and_accepts_dataclass_fields(
    tmp_path: Path,
) -> None:
    pkg = tmp_path / "megatron" / "training"
    pkg.mkdir(parents=True)
    (pkg / "arguments.py").write_text(
        "group.add_argument('--num-experts', type=int)\n", encoding="utf-8"
    )
    (pkg / "config.py").write_text(
        "class C:\n    tensorboard_dir: str | None = None\n    mmap_bin_files: bool = True\n",
        encoding="utf-8",
    )
    flags = ["--num-experts", "--tensorboard-dir", "--no-mmap-bin-files", "--made-up-flag"]
    assert moe.unknown_flags(flags, tmp_path) == ["--made-up-flag"]
