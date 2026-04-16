#!/usr/bin/env python3
"""Legacy helper: keep resources/genie_app_resource.yml as a no-op placeholder.

Genie is attached in resources/apps.yml (genie_space block). This script only
rewrites the placeholder file so nothing under resources/ redefines
scrap_intelligence_app (bundle forbids duplicate resource keys).

Optional: --clear writes an empty resources map (default state).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "resources" / "genie_app_resource.yml"

PLACEHOLDER = """# Placeholder: Genie app attachment lives in resources/apps.yml (genie_space block).
# This file remains so include patterns still resolve; do not redefine apps here.
resources: {}
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="sandbox", help="Unused; kept for CLI compatibility")
    p.add_argument("--space-id", default="", help="Unused; use bundle deploy --var genie_space_id=...")
    p.add_argument("--clear", action="store_true", help="Write placeholder (default)")
    args = p.parse_args()

    OUT.write_text(PLACEHOLDER, encoding="utf-8")
    print(f"Wrote placeholder {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
