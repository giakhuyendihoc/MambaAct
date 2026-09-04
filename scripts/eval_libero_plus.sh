#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${1:?Usage: scripts/eval_libero_plus.sh MODEL CHECKPOINT_DIR CLASSIFICATION_JSON}"
CHECKPOINT_DIR="${2:?Usage: scripts/eval_libero_plus.sh MODEL CHECKPOINT_DIR CLASSIFICATION_JSON}"
CLASSIFICATION_JSON="${3:?Usage: scripts/eval_libero_plus.sh MODEL CHECKPOINT_DIR CLASSIFICATION_JSON}"
RUN_ID="${RUN_ID:-${MODEL}_libero_plus}"
REPORT_DIR="${REPORT_DIR:-$MAMBAACT_OUTPUT_ROOT/reports/libero_plus/$RUN_ID}"

export RUN_ID REPORT_DIR
TRIALS=1 bash "$MAMBAACT_ROOT/scripts/eval_libero.sh" "$MODEL" "$CHECKPOINT_DIR"

python "$MAMBAACT_ROOT/scripts/summarize_libero_plus.py" \
  --metrics-dir "$REPORT_DIR" \
  --classification-json "$CLASSIFICATION_JSON" \
  --output-dir "$REPORT_DIR" \
  --model-label "$MODEL" \
  --trials 1 \
  --output-prefix "${MODEL}_libero_plus_seed7"
