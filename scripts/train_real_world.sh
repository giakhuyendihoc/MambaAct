#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${1:?Usage: scripts/train_real_world.sh MODEL TASK}"
TASK="${2:?Usage: scripts/train_real_world.sh MODEL TASK}"

case "$MODEL:$TASK" in
  mambaact:tube) CONFIG=mambaact_tube ;;
  pi05:tube) CONFIG=pi05_tube_baseline ;;
  mambaact_l12:tube) CONFIG=mambaact_tube_l12 ;;
  mambaact:pour) CONFIG=mambaact_pour ;;
  pi05:pour) CONFIG=pi05_pour_baseline ;;
  mambaact_l12:pour) CONFIG=mambaact_pour_l12 ;;
  *)
    echo "Unsupported model/task combination: $MODEL $TASK" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-${MODEL}_${TASK}}"
STEPS="${STEPS:-30000}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$MAMBAACT_OUTPUT_ROOT/checkpoints}"
ASSETS_ROOT="${ASSETS_ROOT:-$MAMBAACT_OUTPUT_ROOT/assets}"

if [[ ! -s "$MAMBAACT_PI05_CHECKPOINT/model.safetensors" ]]; then
  echo "Missing converted PI0.5 checkpoint: $MAMBAACT_PI05_CHECKPOINT/model.safetensors" >&2
  exit 1
fi

SCOPE_ARGS=(--action-train-scope fullvision_fullvlm)
if [[ "$MODEL" != "pi05" ]]; then
  SCOPE_ARGS=(--mamba-train-scope fullvision_fullvlm)
fi

export PI_MAMBA_SKIP_OPTIMIZER_CHECKPOINT=1

cd "$MAMBAACT_ROOT/openpi"
uv run python scripts/compute_norm_stats.py \
  --config-name "$CONFIG" \
  --assets-base-dir "$ASSETS_ROOT"

uv run python scripts/train_pytorch.py "$CONFIG" \
  --exp-name "$RUN_NAME" \
  --batch-size "${BATCH_SIZE:-16}" \
  --gradient-accumulation-steps "${GRAD_ACCUM:-1}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --num-train-steps "$STEPS" \
  --save-interval "$STEPS" \
  --seed "${TRAIN_SEED:-42}" \
  --checkpoint-base-dir "$CHECKPOINT_ROOT" \
  --assets-base-dir "$ASSETS_ROOT" \
  --pytorch-weight-path "$MAMBAACT_PI05_CHECKPOINT" \
  --initializer-checkpoint "$MAMBAACT_PI05_CHECKPOINT" \
  --lr-schedule.warmup-steps 0 \
  --lr-schedule.peak-lr 5e-5 \
  --lr-schedule.decay-steps "$STEPS" \
  --lr-schedule.decay-lr 5e-6 \
  --model.action-horizon 50 \
  --model.train-vision-encoder \
  --mamba-action-lr 5e-5 \
  --mamba-vision-lr 5e-5 \
  "${SCOPE_ARGS[@]}"

echo "Final checkpoint: $CHECKPOINT_ROOT/$CONFIG/$RUN_NAME/$STEPS"
