"""Validation of ledger/*.json against ledger/schema.json plus cross-reference rules."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

SCHEMA_RELPATH = Path("ledger") / "schema.json"


def load_schema(root: Path) -> dict[str, Any]:
    schema = json.loads((root / SCHEMA_RELPATH).read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise ValueError("ledger/schema.json must be a JSON object")
    return schema


def validate_document(doc: Any, schema: dict[str, Any], filename: str) -> list[str]:
    """Schema validation plus rules JSON Schema cannot express. Returns problems."""
    validator = Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)
    problems = [
        f"{filename}: {'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
        for err in sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path))
    ]
    if problems:
        return problems

    is_example_name = "EXAMPLE" in filename
    if bool(doc.get("example")) != is_example_name:
        problems.append(
            f"{filename}: 'example': true must be set exactly when the filename contains EXAMPLE"
        )
    if date.fromisoformat(doc["period"]["start"]) > date.fromisoformat(doc["period"]["end"]):
        problems.append(f"{filename}: period.start is after period.end")

    island_ids = [i["id"] for i in doc.get("islands", [])]
    if len(set(island_ids)) != len(island_ids):
        problems.append(f"{filename}: duplicate island id")
    node_ids = [n["node_id"] for n in doc["nodes"]]
    if len(set(node_ids)) != len(node_ids):
        problems.append(f"{filename}: duplicate node_id")
    for node in doc["nodes"]:
        island = node.get("island")
        if island is not None and island not in island_ids:
            problems.append(
                f"{filename}: node {node['node_id']} references unknown island {island!r}"
            )
    return problems


def check(root: Path) -> list[str]:
    """Validate the schema itself and every ledger/*.json data file."""
    try:
        schema = load_schema(root)
        Draft202012Validator.check_schema(schema)
    except (OSError, ValueError, SchemaError) as exc:
        return [f"{SCHEMA_RELPATH}: {exc}"]

    problems: list[str] = []
    data_files = sorted(p for p in (root / "ledger").glob("*.json") if p.name != "schema.json")
    if not data_files:
        problems.append("ledger/: no example file to validate against the schema")
    for path in data_files:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"ledger/{path.name}: {exc}")
            continue
        problems.extend(validate_document(doc, schema, path.name))
    return problems
