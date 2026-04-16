#!/usr/bin/env python3
"""
Grant Unity Catalog privileges to a Databricks App's service principal so SQL and Genie can read tables.

Genie **space** access for the app identity is granted by the bundle `genie_space` app resource
(permission CAN_RUN), not by UC SQL. This script only applies SELECT/USAGE on the UC schema.

Run after the app resource exists (bundle deploy) and tables exist (pipeline job). Requires the caller's
identity to have GRANT on the catalog/schema (e.g. schema owner or metastore admin).

Usage:
  python3 scripts/grant_app_uc_access.py --target sandbox
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from typing import Any


def _bundle_var(target: str, key: str) -> str:
    out = subprocess.check_output(
        [
            "databricks",
            "bundle",
            "summary",
            "--target",
            target,
            "-o",
            "json",
        ],
        text=True,
    )
    data = json.loads(out)
    v = (data.get("variables") or {}).get(key) or {}
    return str(v.get("value") or "").strip()


def _cli_json(cmd: list[str]) -> dict[str, Any]:
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out) if out.strip() else {}


def _statement_final_state(target: str, data: dict[str, Any]) -> dict[str, Any]:
    """Poll until the SQL statement finishes (Statement Execution API may return PENDING)."""
    sid = data.get("statement_id")
    state = (data.get("status") or {}).get("state") or ""
    for _ in range(90):
        if state in ("SUCCEEDED", "FAILED", "CANCELED"):
            return data
        if not sid:
            break
        time.sleep(2)
        out = subprocess.check_output(
            [
                "databricks",
                "api",
                "get",
                f"/api/2.0/sql/statements/{sid}",
                "-t",
                target,
                "-o",
                "json",
            ],
            text=True,
        )
        data = json.loads(out) if out.strip() else {}
        state = (data.get("status") or {}).get("state") or ""
    return data


def _run_sql_statement(target: str, warehouse_id: str, statement: str) -> None:
    body = {
        "warehouse_id": warehouse_id,
        "statement": statement,
        "wait_timeout": "50s",
        "on_wait_timeout": "CONTINUE",
    }
    out = subprocess.check_output(
        [
            "databricks",
            "api",
            "post",
            "/api/2.0/sql/statements",
            "--json",
            json.dumps(body),
            "-t",
            target,
            "-o",
            "json",
        ],
        text=True,
    )
    data = json.loads(out) if out.strip() else {}
    data = _statement_final_state(target, data)
    state = (data.get("status") or {}).get("state") or ""
    if state != "SUCCEEDED":
        err = (data.get("status") or {}).get("error") or data
        raise RuntimeError(f"Statement did not succeed (state={state!r}): {err}")


def main() -> int:
    p = argparse.ArgumentParser(description="UC grants for app service principal")
    p.add_argument("--target", required=True, help="Bundle target (e.g. sandbox)")
    args = p.parse_args()
    target = args.target
    app_name = f"scrap-reduction-{target}"

    catalog = _bundle_var(target, "catalog")
    schema = _bundle_var(target, "schema")
    warehouse_id = _bundle_var(target, "warehouse_id")
    if not catalog or not schema or not warehouse_id:
        print(
            "ERROR: Missing catalog, schema, or warehouse_id from bundle summary.",
            file=sys.stderr,
        )
        return 1

    try:
        app = _cli_json(
            ["databricks", "apps", "get", app_name, "-t", target, "-o", "json"]
        )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: databricks apps get failed: {e}", file=sys.stderr)
        return 1

    principal = (app.get("service_principal_name") or "").strip()
    if not principal:
        print(
            "ERROR: App JSON has no service_principal_name; redeploy the app or check API version.",
            file=sys.stderr,
        )
        return 1

    # UC SQL: quote identifiers; principal display name may contain spaces — use backticks only around identifiers.
    cat = catalog.replace("`", "``")
    sch = schema.replace("`", "``")
    pr = principal.replace("`", "``")

    # USAGE/SELECT on schema is sufficient for typical Genie + warehouse queries on `main` (USE CATALOG on main is often default).
    stmts = [
        f"GRANT USAGE ON SCHEMA `{cat}`.`{sch}` TO `{pr}`",
        f"GRANT SELECT ON ALL TABLES IN SCHEMA `{cat}`.`{sch}` TO `{pr}`",
        f"GRANT SELECT ON FUTURE TABLES IN SCHEMA `{cat}`.`{sch}` TO `{pr}`",
    ]

    print(
        f"Granting UC access for app {app_name} principal `{principal}` on {catalog}.{schema}...",
        file=sys.stderr,
    )
    for stmt in stmts:
        print(f"  {stmt}", file=sys.stderr)
        try:
            _run_sql_statement(target, warehouse_id, stmt)
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"ERROR: Statement failed ({e}).", file=sys.stderr)
            return 1

    print("UC grants applied.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
