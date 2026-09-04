#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CONFIG="${1:?Usage: scripts/serve_checkpoint.sh CONFIG CHECKPOINT_DIR}"
CHECKPOINT_DIR="${2:?Usage: scripts/serve_checkpoint.sh CONFIG CHECKPOINT_DIR}"

cd "$MAMBAACT_ROOT/openpi"
exec uv run python scripts/serve_policy_from_checkpoint.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --config-name "$CONFIG" \
  --port "${PORT:-8000}" \
  --num-denoise-steps "${NUM_DENOISE_STEPS:-10}"
