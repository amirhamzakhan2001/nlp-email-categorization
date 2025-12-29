#!/usr/bin/env bash
set -e

echo "🔄 Starting INCREMENTAL pipeline..."

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate

python -m src.pipelines.incremental_pipeline

echo "✅ INCREMENTAL pipeline completed successfully"