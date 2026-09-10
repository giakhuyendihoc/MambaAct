#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
LEROBOT_ROOT="${MAMBAACT_LEROBOT_ROOT:?Set MAMBAACT_LEROBOT_ROOT to the patched LeRobot checkout}"
POLICY="${1:?Provide a pretrained-model directory}"
LABEL="${2:-smolvla_25k}"
REPORT_DIR="${TIMING_OUTPUT_DIR:-$MAMBAACT_OUTPUT_ROOT/smolvla/${LABEL}_timing}"
export PYTHONPATH="$LEROBOT_ROOT/src:$MAMBAACT_ROOT/mamba:${PYTHONPATH:-}"
mkdir -p "$REPORT_DIR"
python "$MAMBAACT_ROOT/scripts/profile_smolvla_component_timing.py" \
  --policy-path "$POLICY" --model-label "$LABEL" --device cuda \
  --image-keys observation.images.camera1 observation.images.camera2 --state-dim 8 \
  --n-action-steps 10 --warmup-runs 3 --timed-runs 10 \
  --metrics-out "$REPORT_DIR/component_timing.json" --table-out "$REPORT_DIR/component_timing.md"
