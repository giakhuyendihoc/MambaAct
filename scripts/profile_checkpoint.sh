#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${1:?Usage: scripts/profile_checkpoint.sh MODEL CHECKPOINT_DIR}"
CHECKPOINT_DIR="${2:?Usage: scripts/profile_checkpoint.sh MODEL CHECKPOINT_DIR}"
IFS=$'\t' read -r CONFIG DEFAULT_RUN MODEL_KIND < <(resolve_pi05_model "$MODEL")

REPORT_DIR="${REPORT_DIR:-$MAMBAACT_OUTPUT_ROOT/reports/profiling/$DEFAULT_RUN}"
mkdir -p "$REPORT_DIR"
EXTRA_ARGS=()
if [[ "${PROFILE_FLOPS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--profile-flops)
fi

cd "$MAMBAACT_ROOT/openpi"
uv run python scripts/check_policy_inference.py \
  --config-name "$CONFIG" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --pytorch-device "${DEVICE:-cuda}" \
  --num-denoise-steps "${NUM_DENOISE_STEPS:-10}" \
  --warmup-runs "${WARMUP_RUNS:-3}" \
  --timed-runs "${TIMED_RUNS:-10}" \
  --component-timing \
  --metrics-out-path "$REPORT_DIR/component_timing.json" \
  --table-out-path "$REPORT_DIR/component_timing.md" \
  --model-label "$MODEL" \
  "${EXTRA_ARGS[@]}"

echo "Timing report: $REPORT_DIR/component_timing.md"
