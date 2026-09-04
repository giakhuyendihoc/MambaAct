#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${1:-mambaact}"
LEROBOT_ROOT="${MAMBAACT_LEROBOT_ROOT:?Set MAMBAACT_LEROBOT_ROOT to the patched LeRobot checkout}"
STEPS="${STEPS:-30000}"
TRIALS="${TRIALS:-50}"

case "$MODEL" in
+  baseline)
    RUN_NAME="${RUN_NAME:-smolvla_baseline}"
    EXTRA_POLICY_ARGS=()
    ;;
  mambaact)
    RUN_NAME="${RUN_NAME:-smolvla_mambaact}"
    EXTRA_POLICY_ARGS=(
      --policy.use_mamba_action_expert=true
      --policy.mamba_depth=4
      --policy.mamba_width=768
      --policy.mamba_context_tokens=64
      --policy.mamba_d_state=64
      --policy.mamba_expand=1
      --policy.mamba_headdim=64
      --policy.mamba_chunk_size=64
    )
    ;;
  *)
+    echo "Expected MODEL to be 'baseline' or 'mambaact'." >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-$MAMBAACT_OUTPUT_ROOT/smolvla/$RUN_NAME}"
export PYTHONPATH="$LEROBOT_ROOT/src:$MAMBAACT_ROOT/mamba:${PYTHONPATH:-}"

python -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task=libero_10 \
  --output_dir="$OUTPUT_DIR" \
  --steps="$STEPS" \
  --batch_size="${BATCH_SIZE:-16}" \
  --num_workers="${NUM_WORKERS:-8}" \
  --save_freq="$STEPS" \
  --env_eval_freq=0 \
  --eval_steps=0 \
  --seed="${TRAIN_SEED:-42}" \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.num_steps=10 \
  --policy.optimizer_lr=1e-4 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps="$STEPS" \
  --policy.scheduler_decay_lr=2.5e-6 \
  "${EXTRA_POLICY_ARGS[@]}"

printf -v STEP_DIR '%06d' "$STEPS"
FINAL_POLICY="$OUTPUT_DIR/checkpoints/$STEP_DIR/pretrained_model"
python -m lerobot.scripts.lerobot_eval \
  --output_dir="$OUTPUT_DIR/eval50" \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes="$TRIALS" \
  --policy.path="$FINAL_POLICY" \
  --policy.n_action_steps=10 \
  --seed="${EVAL_SEED:-7}"
