import subprocess
from pathlib import Path

import pytest

from ofl import upstreams
from ofl.repo import OflError


def test_existing_checkout_at_the_wrong_commit_is_refused(tmp_path: Path) -> None:
    repo = tmp_path / ".cache" / "upstreams" / "olmo-core"
    repo.mkdir(parents=True)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q"], cwd=repo, check=True, timeout=60)
    subprocess.run(
        [*git, "commit", "-q", "--allow-empty", "-m", "x"], cwd=repo, check=True, timeout=60
    )
    up = upstreams.Upstream(
        "olmo-core", "https://example.invalid/x", "v0", "a" * 40, "MIT", "d", "u"
    )
    with pytest.raises(OflError, match="expected " + "a" * 40):
        upstreams.ensure_checkout(tmp_path, up)
