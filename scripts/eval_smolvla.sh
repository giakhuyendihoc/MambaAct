#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
LEROBOT_ROOT="${MAMBAACT_LEROBOT_ROOT:?Set MAMBAACT_LEROBOT_ROOT to the patched LeRobot checkout}"
LIBERO_ROOT="${MAMBAACT_LIBERO_ROOT:?Set MAMBAACT_LIBERO_ROOT to the installed LIBERO checkout}"
export PYTHONPATH="$LEROBOT_ROOT/src:$MAMBAACT_ROOT/mamba:$LIBERO_ROOT:${PYTHONPATH:-}"
MODEL="${1:-released}"
case "$MODEL" in
  released)
    POLICY="${2:-$MAMBAACT_OUTPUT_ROOT/smolvla/inputs/released/31d453f7edd78c839a8bbc39744a292686daf0de}"
    if [[ $# -lt 2 ]]; then
      python "$MAMBAACT_ROOT/scripts/download_smolvla.py" released \
        --output-root "$MAMBAACT_OUTPUT_ROOT/smolvla/inputs"
    fi
    ;;
  mambaact) POLICY="${2:-$MAMBAACT_OUTPUT_ROOT/smolvla/mambaact_25k/checkpoints/025000/pretrained_model}" ;;
  *) echo "Usage: $0 {released|mambaact} [checkpoint-directory]" >&2; exit 2 ;;
esac
[[ -s "$POLICY/model.safetensors" ]] || { echo "Model weights missing: $POLICY" >&2; exit 1; }
LABEL="smolvla_${MODEL}_25k"
REPORT_DIR="${EVAL_OUTPUT_DIR:-$MAMBAACT_OUTPUT_ROOT/smolvla/${LABEL}_eval50_seed7}"
[[ ! -e "$REPORT_DIR" ]] || { echo "Evaluation output already exists: $REPORT_DIR" >&2; exit 1; }
RENAME_MAP='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
python -m lerobot.scripts.lerobot_eval \
  --output_dir="$REPORT_DIR" --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --env.max_parallel_tasks=1 --env.observation_height=360 --env.observation_width=360 \
  --eval.batch_size=1 --eval.n_episodes=50 --policy.path="$POLICY" \
  --policy.chunk_size=50 --policy.n_action_steps=10 --policy.num_steps=10 \
  --policy.use_amp=false --policy.resize_imgs_with_padding='[512,512]' \
  --rename_map="$RENAME_MAP" --seed=7
TIMING_OUTPUT_DIR="$REPORT_DIR" bash "$MAMBAACT_ROOT/scripts/profile_smolvla.sh" "$POLICY" "$LABEL"
