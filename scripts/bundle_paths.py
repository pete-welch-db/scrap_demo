#!/usr/bin/env python3
"""Emit workspace paths from `databricks bundle summary` for shell deploy scripts."""
import argparse
import json
import subprocess
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument(
        "key",
        choices=(
            "app_source",
            "warehouse_id",
            "profile",
            "file_path",
            "root_path",
            "catalog",
            "schema",
        ),
        help="Which value to print",
    )
    args = p.parse_args()
    out = subprocess.check_output(
        ["databricks", "bundle", "summary", "--target", args.target, "-o", "json"],
        text=True,
    )
    data = json.loads(out)
    ws = data.get("workspace") or {}
    if args.key == "app_source":
        fp = ws.get("file_path", "").rstrip("/")
        print(f"{fp}/app" if fp else "", end="")
    elif args.key == "warehouse_id":
        v = (data.get("variables") or {}).get("warehouse_id") or {}
        print(v.get("value", ""), end="")
    elif args.key == "profile":
        print(ws.get("profile", ""), end="")
    elif args.key == "file_path":
        print(ws.get("file_path", ""), end="")
    elif args.key == "root_path":
        print(ws.get("root_path", ""), end="")
    elif args.key == "catalog":
        v = (data.get("variables") or {}).get("catalog") or {}
        print(v.get("value", ""), end="")
    elif args.key == "schema":
        v = (data.get("variables") or {}).get("schema") or {}
        print(v.get("value", ""), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
