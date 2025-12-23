#!/bin/bash
# ===========================================
# Scrap Reduction Platform - Deploy Script
# ===========================================
# Two-stage deployment:
#   1. Setup job - Creates catalog, schema, bronze data
#   2. Pipeline job - DLT transformations + ML training
#
# Usage:
#   ./deploy.sh [target]
#
# Examples:
#   ./deploy.sh          # Deploy to default (dev) target
#   ./deploy.sh dev      # Deploy to dev target
#   ./deploy.sh prod     # Deploy to prod target

set -e

TARGET="${1:-dev}"

echo "🚀 Deploying Scrap Reduction Platform to target: $TARGET"
echo "=============================================="

# Validate the bundle first
echo "📋 Validating bundle configuration..."
databricks bundle validate --target "$TARGET"

# Deploy the bundle
echo ""
echo "📦 Deploying bundle..."
databricks bundle deploy --target "$TARGET"

# Stage 1: Run setup job (creates catalog, schema, bronze data)
echo ""
echo "⚙️ Stage 1: Running setup job (creates catalog, schema, bronze data)..."
databricks bundle run scrap_setup_job --target "$TARGET"

# Stage 2: Run pipeline job (DLT + ML)
echo ""
echo "🔄 Stage 2: Running pipeline job (DLT transformations + ML training)..."
databricks bundle run scrap_pipeline_job --target "$TARGET"

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Pipeline stages completed:"
echo "  1. ✅ Setup: Catalog, schema, and bronze data created"
echo "  2. ✅ DLT: Silver and Gold tables created"
echo "  3. ✅ ML: All models trained"
