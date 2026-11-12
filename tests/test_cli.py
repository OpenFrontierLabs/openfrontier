import re
from pathlib import Path

import pytest

from ofl import cli
from ofl.repo import OflError


def test_makefile_has_exactly_the_documented_targets_as_one_line_wrappers(root: Path) -> None:
    text = (root / "Makefile").read_text(encoding="utf-8")
    phony = re.search(r"^\.PHONY:(.*)$", text, re.M)
    assert phony is not None
    assert set(phony.group(1).split()) == set(cli.TARGETS)
    # every recipe line is a single `uv run ofl ...` call, no shell logic
    recipes = [ln.strip() for ln in text.splitlines() if ln.startswith("\t")]
    assert recipes
    assert all(re.fullmatch(r"@uv run ofl [\w$@() -]+", r) for r in recipes), recipes
    assert "ofl deploy --env $(ENV)" in text


def test_every_target_parses_and_resolves_to_a_function() -> None:
    parser = cli.build_parser()
    for target in cli.TARGETS:
        argv = [target, "--env", "dry-run"] if target == "deploy" else [target]
        assert callable(parser.parse_args(argv).func)


def test_help_lists_every_target(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["help"]) == 0
    out = capsys.readouterr().out
    for target in cli.TARGETS:
        assert re.search(rf"^\s+{target}\s", out, re.M), target


@pytest.mark.parametrize("phase", ["post", "diloco"])
def test_unbuilt_phases_fail_loudly(phase: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([phase, "launch"]) == 2
    assert "not built" in capsys.readouterr().err


def test_deploy_requires_an_env() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["deploy"])
    assert exc.value.code == 2


def test_deploy_dry_run_publishes_nothing(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(root)
    assert cli.main(["deploy", "--env", "dry-run"]) == 0
    assert "nothing is published" in capsys.readouterr().out


def test_deploy_real_env_refuses_with_nonzero_exit(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["deploy", "--env", "huggingface"]) == 2
    assert "not implemented yet: nothing to publish" in capsys.readouterr().err


def test_deploy_unknown_env_lists_the_known_ones(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["deploy", "--env", "prod"]) == 2
    err = capsys.readouterr().err
    assert "dry-run" in err
    assert "huggingface" in err


def test_deploy_dry_run_lists_checkpoints_that_exist(tmp_path: Path) -> None:
    from ofl import deploy

    (tmp_path / "checkpoints" / "run0" / "step2000").mkdir(parents=True)
    (tmp_path / "checkpoints" / "run0" / "notes").mkdir()
    found = deploy.find_checkpoints(tmp_path)
    assert [p.name for p in found] == ["step2000"]


def test_find_root_outside_a_checkout_is_an_error(tmp_path: Path) -> None:
    from ofl.repo import find_root

    with pytest.raises(OflError, match="not inside an openfrontier checkout"):
        find_root(tmp_path)
