#!/usr/bin/env bash
# =============================================================================
# Scrap Reduction Platform — full stack deploy (one command)
# =============================================================================
# Deploys and runs, in order:
#   1. Bundle (Lakeview dashboards, DLT pipeline defs, serverless jobs, app resource)
#   2. SQL warehouse — start (Pro / serverless warehouse for jobs + dashboards)
#   3. Setup job — bronze + schema prep
#   4. Pipeline job — DLT silver/gold + ML + metric views + dashboard refresh task
#   5. Genie space — create/update + second bundle deploy with GENIE_SPACE_ID
#   6. App — start compute if needed + snapshot deploy of synced Streamlit code
#
# Usage:
#   ./deploy.sh [target]
#   DEPLOY_TARGET=sandbox ./deploy.sh
#
# Optional environment:
#   DEPLOY_TARGET   — default target if no positional arg (default: dev)
#   SKIP_GENIE=1    — skip Genie space + genie bundle var
#   SKIP_JOBS=1     — bundle + app only (no setup/pipeline jobs, no Genie)
#   SKIP_APP=1      — skip Databricks Apps snapshot deploy
#   SKIP_WAREHOUSE=1 — do not start SQL warehouse before jobs
#   SKIP_UC_GRANTS=1 — skip scripts/grant_app_uc_access.py (UC grants for the app service principal)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

TARGET="${1:-${DEPLOY_TARGET:-dev}}"
APP_NAME="scrap-reduction-${TARGET}"

echo "🚀 Scrap Reduction — full deploy | target=${TARGET}"
echo "=================================================="

echo "📋 Validating bundle..."
databricks bundle validate --target "$TARGET"

echo ""
echo "📦 Stage 0: Deploy bundle (dashboards, pipelines, jobs, app definition)..."
databricks bundle deploy --target "$TARGET"

if [[ "${SKIP_JOBS:-}" == "1" ]]; then
  echo ""
  echo "⏭️  SKIP_JOBS=1 — skipping warehouse, jobs, Genie (bundle-only deploy above)."
  if [[ "${SKIP_APP:-}" != "1" ]]; then
    APP_SRC="$(python3 scripts/bundle_paths.py --target "$TARGET" app_source)"
    echo "📱 Deploying app from ${APP_SRC}..."
    databricks apps start "$APP_NAME" -t "$TARGET" 2>/dev/null || true
    databricks apps deploy "$APP_NAME" -t "$TARGET" --source-code-path "$APP_SRC" --skip-validation --timeout 45m
  fi
  databricks bundle summary --target "$TARGET"
  exit 0
fi

WID="$(python3 scripts/bundle_paths.py --target "$TARGET" warehouse_id)"
if [[ -n "$WID" && "${SKIP_WAREHOUSE:-}" != "1" ]]; then
  echo ""
  echo "🏭 Starting SQL warehouse ${WID} (for jobs + dashboard SQL)..."
  databricks warehouses start "$WID" -t "$TARGET" --timeout 25m || true
fi

echo ""
echo "⚙️  Stage 1: Setup job (catalog/schema/bronze)..."
databricks bundle run scrap_setup_job --target "$TARGET"

echo ""
echo "🔄 Stage 2: Pipeline job (DLT + ML + metrics + dashboard refresh task)..."
databricks bundle run scrap_pipeline_job --target "$TARGET"

if [[ "${SKIP_GENIE:-}" == "1" ]]; then
  echo ""
  echo "⏭️  Skipping Genie (SKIP_GENIE=1)"
else
  echo ""
  echo "🧞 Stage 3: Genie space + bundle (GENIE_SPACE_ID)..."
  # Progress on stderr; last stdout line is GENIE_SPACE_ID=<id>
  GENIE_LINE="$(python3 scripts/deploy_genie_space.py --target "$TARGET" | tail -1)"
  GENIE_ID="${GENIE_LINE#GENIE_SPACE_ID=}"
  if [[ -z "$GENIE_ID" || "$GENIE_ID" == "$GENIE_LINE" ]]; then
    echo "ERROR: Could not parse Genie space id. Last line: $GENIE_LINE" >&2
    exit 1
  fi
  # Genie attachment is defined in resources/apps.yml; pass the new space id into the bundle.
  databricks bundle deploy --target "$TARGET" --var "genie_space_id=$GENIE_ID"
fi

if [[ "${SKIP_APP:-}" == "1" ]]; then
  echo ""
  echo "⏭️  Skipping app deploy (SKIP_APP=1)"
else
  APP_SRC="$(python3 scripts/bundle_paths.py --target "$TARGET" app_source)"
  echo ""
  echo "📱 Stage 4: Databricks App snapshot (${APP_NAME})..."
  echo "   Source: ${APP_SRC}"
  databricks apps start "$APP_NAME" -t "$TARGET" 2>/dev/null || true
  sleep 5
  databricks apps deploy "$APP_NAME" -t "$TARGET" --source-code-path "$APP_SRC" --skip-validation --timeout 45m

  if [[ "${SKIP_UC_GRANTS:-}" != "1" ]]; then
    echo ""
    echo "🔐 Stage 5: UC grants for app service principal (${APP_NAME})..."
    python3 scripts/grant_app_uc_access.py --target "$TARGET" || {
      echo "WARNING: grant_app_uc_access.py failed (insufficient GRANT rights or API). Run manually:" >&2
      echo "  python3 scripts/grant_app_uc_access.py --target $TARGET" >&2
    }
  fi
fi

echo ""
echo "✅ Full deploy finished."
echo ""
databricks bundle summary --target "$TARGET"
