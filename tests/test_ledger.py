import copy
import json
from pathlib import Path
from typing import Any

import pytest

from ofl import ledger


@pytest.fixture
def schema(root: Path) -> dict[str, Any]:
    return ledger.load_schema(root)


@pytest.fixture
def example(root: Path) -> dict[str, Any]:
    path = root / "ledger" / "example.EXAMPLE.json"
    try:
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        pytest.fail(f"{path} is not valid JSON: {exc}")
    return loaded


def problems(doc: Any, schema: dict[str, Any], name: str = "x.EXAMPLE.json") -> list[str]:
    return ledger.validate_document(doc, schema, name)


def test_shipped_ledger_files_validate(root: Path) -> None:
    assert ledger.check(root) == []


def test_example_is_marked_as_an_example_and_uses_fake_data(example: dict[str, Any]) -> None:
    assert example["example"] is True
    assert all(n["github_handle"].startswith("example-") for n in example["nodes"])


def test_real_ledger_file_may_not_claim_to_be_an_example(
    example: dict[str, Any], schema: dict[str, Any]
) -> None:
    assert any("EXAMPLE" in p for p in problems(example, schema, "run0-2026-11.json"))


def test_example_file_must_carry_the_example_marker(
    example: dict[str, Any], schema: dict[str, Any]
) -> None:
    del example["example"]
    assert any("EXAMPLE" in p for p in problems(example, schema))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("helpful_update_share", 1.5),
        ("helpful_update_share", -0.1),
        ("gpu_hours", -1),
        ("tokens_processed", 1.5),
        ("github_handle", "-bad-handle"),
        ("github_handle", "has space"),
        ("node_id", "Upper"),
    ],
)
def test_invalid_node_fields_are_rejected(
    field: str, value: object, example: dict[str, Any], schema: dict[str, Any]
) -> None:
    example["nodes"][0][field] = value
    assert problems(example, schema)


def test_missing_node_field_is_rejected(example: dict[str, Any], schema: dict[str, Any]) -> None:
    del example["nodes"][0]["tokens_processed"]
    assert any("tokens_processed" in p for p in problems(example, schema))


def test_unknown_keys_are_rejected(example: dict[str, Any], schema: dict[str, Any]) -> None:
    example["nodes"][0]["email"] = "someone@example.com"
    assert problems(example, schema)


def test_node_must_reference_a_declared_island(
    example: dict[str, Any], schema: dict[str, Any]
) -> None:
    example["nodes"][0]["island"] = "no-such-island"
    assert any("unknown island" in p for p in problems(example, schema))


def test_duplicate_node_ids_are_rejected(example: dict[str, Any], schema: dict[str, Any]) -> None:
    example["nodes"].append(copy.deepcopy(example["nodes"][0]))
    assert any("duplicate node_id" in p for p in problems(example, schema))


def test_period_must_not_run_backwards(example: dict[str, Any], schema: dict[str, Any]) -> None:
    example["period"] = {"start": "2000-02-01", "end": "2000-01-01"}
    assert any("after period.end" in p for p in problems(example, schema))


def test_impossible_calendar_date_is_rejected(
    example: dict[str, Any], schema: dict[str, Any]
) -> None:
    example["period"]["end"] = "2000-02-31"
    assert problems(example, schema)


def test_check_flags_a_directory_with_no_data_file(root: Path, tmp_path: Path) -> None:
    (tmp_path / "ledger").mkdir()
    (tmp_path / "ledger" / "schema.json").write_text(
        (root / "ledger" / "schema.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert any("no example file" in p for p in ledger.check(tmp_path))
