#!/usr/bin/env bash
# Run medeval CLI for segmentation evaluation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

echo "=== Running medeval CLI evaluation ==="
echo "Working directory: $(pwd)"
echo ""

python3 -m medeval.cli.main evaluate \
    --task segmentation \
    --manifest demo_external_user/manifest_seg.csv \
    --config demo_external_user/config_seg.yaml \
    --output demo_external_user/out_cli \
    -vv

echo ""
echo "=== CLI evaluation complete ==="
echo "Output directory: demo_external_user/out_cli/"
ls -la demo_external_user/out_cli/

