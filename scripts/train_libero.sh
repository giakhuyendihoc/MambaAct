#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${1:-mambaact}"
IFS=$'\t' read -r CONFIG DEFAULT_RUN MODEL_KIND < <(resolve_pi05_model "$MODEL")

RUN_NAME="${RUN_NAME:-$DEFAULT_RUN}"
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TRAIN_SEED="${TRAIN_SEED:-42}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$MAMBAACT_OUTPUT_ROOT/checkpoints}"
ASSETS_ROOT="${ASSETS_ROOT:-$MAMBAACT_OUTPUT_ROOT/assets}"

if [[ ! -s "$MAMBAACT_PI05_CHECKPOINT/model.safetensors" ]]; then
  echo "Missing converted PI0.5 checkpoint: $MAMBAACT_PI05_CHECKPOINT/model.safetensors" >&2
  echo "Follow docs/REPRODUCIBILITY.md to prepare the initializer." >&2
  exit 1
fi

SCOPE_ARGS=(
  --action-train-scope fullvision_fullvlm
  --mamba-action-lr 5e-5
  --mamba-vision-lr 5e-5
)
if [[ "$MODEL_KIND" == "mamba" ]]; then
  SCOPE_ARGS=(
    --mamba-train-scope fullvision_fullvlm
    --mamba-action-lr 5e-5
    --mamba-vision-lr 5e-5
  )
fi

export PI_MAMBA_SKIP_OPTIMIZER_CHECKPOINT=1

cd "$MAMBAACT_ROOT/openpi"
uv run python scripts/compute_norm_stats.py \
  --config-name "$CONFIG" \
  --assets-base-dir "$ASSETS_ROOT"

uv run python scripts/train_pytorch.py "$CONFIG" \
  --exp-name "$RUN_NAME" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --num-workers "$NUM_WORKERS" \
  --num-train-steps "$STEPS" \
  --save-interval "$STEPS" \
  --seed "$TRAIN_SEED" \
  --checkpoint-base-dir "$CHECKPOINT_ROOT" \
  --assets-base-dir "$ASSETS_ROOT" \
  --pytorch-weight-path "$MAMBAACT_PI05_CHECKPOINT" \
  --initializer-checkpoint "$MAMBAACT_PI05_CHECKPOINT" \
  --lr-schedule.warmup-steps 0 \
  --lr-schedule.peak-lr 5e-5 \
  --lr-schedule.decay-steps "$STEPS" \
  --lr-schedule.decay-lr 5e-6 \
  --model.train-vision-encoder \
  "${SCOPE_ARGS[@]}"

echo "Final checkpoint: $CHECKPOINT_ROOT/$CONFIG/$RUN_NAME/$STEPS"
