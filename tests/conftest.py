from pathlib import Path

import pytest

from ofl.repo import find_root


@pytest.fixture(scope="session")
def root() -> Path:
    return find_root(Path(__file__).parent)
