#!/usr/bin/env python3
"""
Create or update a Databricks Genie space from Genie/*. config files.

Uses the Genie REST API (serialized_space) per:
https://docs.databricks.com/api/workspace/genie/createspace

Run from the repository root after Unity Catalog tables exist (e.g. after pipeline job).
Uses `databricks bundle` target for auth and resolved variables (catalog, schema, warehouse).

Usage:
  python3 scripts/deploy_genie_space.py --target sandbox

Prints the space id on the last line as: GENIE_SPACE_ID=<id>
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Tables aligned with Genie/README.md and Genie/tables.md (fact + gold + scenario)
GENIE_TABLE_NAMES = [
    "scrap_transactions_silver",
    "eight_d_reports_silver",
    "scrap_daily_summary_gold",
    "scrap_monthly_summary_gold",
    "scrap_by_plant_gold",
    "scrap_by_supplier_gold",
    "scrap_by_defect_gold",
    "supplier_scenario_gold",
    "kpi_dashboard_gold",
]


def _run_json(cmd: list[str]) -> dict:
    out = subprocess.check_output(cmd, cwd=REPO_ROOT, text=True)
    return json.loads(out)


def _bundle_variables(target: str) -> dict[str, str]:
    summary = _run_json(
        ["databricks", "bundle", "summary", "--target", target, "-o", "json"]
    )
    raw = summary.get("variables") or {}
    resolved: dict[str, str] = {}
    for key, meta in raw.items():
        if isinstance(meta, dict) and "value" in meta:
            resolved[key] = str(meta["value"])
        elif isinstance(meta, str):
            resolved[key] = meta
    return resolved


def _space_title(target: str) -> str:
    return f"Scrap Reduction Genie ({target})"


def _space_description() -> str:
    return (
        "AI assistant for manufacturing scrap analytics: costs, plants, suppliers, defects, "
        "8D quality reports, and scenarios. Tables are registered from this bundle's Unity Catalog schema."
    )


def _new_instruction_id() -> str:
    # Match typical Genie id length (32 hex chars)
    return secrets.token_hex(16)


def _build_serialized_space(catalog: str, schema: str, instructions_text: str) -> str:
    # API requires data_sources.tables sorted by identifier (lexicographic on full name).
    identifiers = sorted(f"{catalog}.{schema}.{name}" for name in GENIE_TABLE_NAMES)
    tables = [{"identifier": ident} for ident in identifiers]
    body = {
        "version": 1,
        "data_sources": {"tables": tables},
        "instructions": {
            "text_instructions": [
                {
                    "id": _new_instruction_id(),
                    "content": [instructions_text],
                }
            ],
            "example_question_sqls": [],
        },
    }
    return json.dumps(body, separators=(",", ":"))


def _list_spaces(target: str) -> list[dict]:
    out = subprocess.check_output(
        ["databricks", "genie", "list-spaces", "-t", target, "-o", "json"],
        cwd=REPO_ROOT,
        text=True,
    )
    data = json.loads(out)
    return data.get("spaces") or []


def _find_space_id(spaces: list[dict], title: str) -> str | None:
    for s in spaces:
        if s.get("title") == title:
            return s.get("space_id")
    return None


def _write_tmp_json(payload: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".json", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def _databricks_json(cmd: list[str]) -> dict:
    out = subprocess.check_output(cmd, cwd=REPO_ROOT, text=True)
    return json.loads(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy Genie space for scrap demo bundle")
    parser.add_argument(
        "--target",
        required=True,
        help="Databricks bundle target (e.g. dev, sandbox)",
    )
    parser.add_argument(
        "--instructions",
        default=str(REPO_ROOT / "Genie" / "instructions_short.txt"),
        help="Path to instructions text file",
    )
    args = parser.parse_args()

    vars_ = _bundle_variables(args.target)
    catalog = vars_.get("catalog", "scrap_catalog")
    schema = vars_.get("schema", "scrap_demo")
    warehouse_id = vars_.get("warehouse_id")
    if not warehouse_id:
        print("ERROR: warehouse_id not resolved from bundle for this target.", file=sys.stderr)
        return 1

    instructions_path = Path(args.instructions)
    if not instructions_path.is_file():
        print(f"ERROR: Instructions file not found: {instructions_path}", file=sys.stderr)
        return 1
    instructions_text = instructions_path.read_text(encoding="utf-8").strip()
    if not instructions_text:
        print("ERROR: Instructions file is empty.", file=sys.stderr)
        return 1

    serialized = _build_serialized_space(catalog, schema, instructions_text)
    title = _space_title(args.target)
    description = _space_description()

    spaces = _list_spaces(args.target)
    existing_id = _find_space_id(spaces, title)

    if existing_id:
        print(f"Updating Genie space {existing_id} ({title})...", file=sys.stderr)
        upd_path = _write_tmp_json(
            {
                "serialized_space": serialized,
                "warehouse_id": warehouse_id,
                "title": title,
                "description": description,
            }
        )
        try:
            _databricks_json(
                [
                    "databricks",
                    "genie",
                    "update-space",
                    existing_id,
                    "--json",
                    f"@{upd_path}",
                    "-t",
                    args.target,
                    "-o",
                    "json",
                ]
            )
        finally:
            os.unlink(upd_path)
        space_id = existing_id
    else:
        print(f"Creating Genie space ({title})...", file=sys.stderr)
        create_path = _write_tmp_json(
            {
                "title": title,
                "description": description,
                "warehouse_id": warehouse_id,
                "serialized_space": serialized,
            }
        )
        try:
            resp = _databricks_json(
                [
                    "databricks",
                    "genie",
                    "create-space",
                    "--json",
                    f"@{create_path}",
                    "-t",
                    args.target,
                    "-o",
                    "json",
                ]
            )
        finally:
            os.unlink(create_path)
        space_id = resp.get("space_id")
        if not space_id:
            print(f"ERROR: Unexpected API response: {resp}", file=sys.stderr)
            return 1

    print(f"GENIE_SPACE_ID={space_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
