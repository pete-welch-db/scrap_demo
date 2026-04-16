#!/usr/bin/env python3
"""
Verify the Databricks App service principal has CAN USE on the bundle SQL warehouse.

`resources/apps.yml` attaches `sql_warehouse` with `permission: CAN_USE`, but some workspaces
still require this ACL to appear on the warehouse — users may add it in the UI (Compute →
SQL Warehouse → Permissions). This script checks via CLI so deploy logs show a clear pass/fail.

Usage:
  python3 scripts/check_app_warehouse_access.py --target sandbox
  python3 scripts/check_app_warehouse_access.py --target sandbox --profile fe-sandbox-scrapdemo
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys


def _prefix(profile: str | None) -> list[str]:
    return ["databricks", "-p", profile] if profile else ["databricks"]


def _bundle_var(target: str, key: str, profile: str | None) -> str:
    out = subprocess.check_output(
        _prefix(profile)
        + ["bundle", "summary", "--target", target, "-o", "json"],
        text=True,
    )
    data = json.loads(out)
    v = (data.get("variables") or {}).get(key) or {}
    return str(v.get("value") or "").strip()


def main() -> int:
    p = argparse.ArgumentParser(description="Check app SP CAN USE on SQL warehouse")
    p.add_argument("--target", required=True)
    p.add_argument("--profile", default=None)
    args = p.parse_args()
    target = args.target
    profile = args.profile
    app_name = f"scrap-reduction-{target}"
    wid = _bundle_var(target, "warehouse_id", profile)
    if not wid:
        print("ERROR: warehouse_id missing from bundle summary.", file=sys.stderr)
        return 1

    app = json.loads(
        subprocess.check_output(
            _prefix(profile) + ["apps", "get", app_name, "-t", target, "-o", "json"],
            text=True,
        )
    )
    sp_uuid = (app.get("service_principal_client_id") or "").strip()
    if not sp_uuid:
        print("ERROR: app has no service_principal_client_id.", file=sys.stderr)
        return 1

    perm = json.loads(
        subprocess.check_output(
            _prefix(profile)
            + ["warehouses", "get-permissions", wid, "-t", target, "-o", "json"],
            text=True,
        )
    )
    acl = perm.get("access_control_list") or []
    for entry in acl:
        if (entry.get("service_principal_name") or "").strip() != sp_uuid:
            continue
        for ap in entry.get("all_permissions") or []:
            if ap.get("permission_level") == "CAN_USE" and not ap.get("inherited"):
                print(
                    f"OK: App `{app_name}` SP `{sp_uuid}` has CAN_USE on warehouse `{wid}`.",
                    file=sys.stderr,
                )
                return 0
        break

    print(
        f"WARNING: App `{app_name}` SP `{sp_uuid}` does not have explicit CAN_USE on warehouse `{wid}`. "
        f"Add it: SQL Warehouses → {wid} → Permissions → add app SP → Can use. "
        f"Or rely on bundle app resource `sql_warehouse` after redeploy.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
