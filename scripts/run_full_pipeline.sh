#!/usr/bin/env bash
set -e

echo "🚀 Starting FULL pipeline..."

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate

python -m src.pipelines.full_pipeline

echo "✅ FULL pipeline completed successfully"