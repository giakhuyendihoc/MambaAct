#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
if [[ "${1:-mambaact}" != mambaact ]]; then
  echo "Only MambaAct is trained. Evaluate the released baseline with scripts/eval_smolvla.sh released." >&2
  exit 2
fi
LEROBOT_ROOT="${MAMBAACT_LEROBOT_ROOT:?Set MAMBAACT_LEROBOT_ROOT to the patched LeRobot checkout}"
LIBERO_ROOT="${MAMBAACT_LIBERO_ROOT:?Set MAMBAACT_LIBERO_ROOT to the installed LIBERO checkout}"
export PYTHONPATH="$LEROBOT_ROOT/src:$MAMBAACT_ROOT/mamba:$LIBERO_ROOT:${PYTHONPATH:-}"
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=1
OUTPUT_DIR="${OUTPUT_DIR:-$MAMBAACT_OUTPUT_ROOT/smolvla/mambaact_25k}"
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Training output already exists: $OUTPUT_DIR" >&2; exit 1; }
INPUT_ROOT="$MAMBAACT_OUTPUT_ROOT/smolvla/inputs"
python "$MAMBAACT_ROOT/scripts/download_smolvla.py" training --output-root "$INPUT_ROOT"
BASE_POLICY="$INPUT_ROOT/base/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
DATASET_ROOT="$INPUT_ROOT/dataset/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
RENAME_MAP='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
INPUT_FEATURES='{"observation.images.camera1":{"type":"VISUAL","shape":[3,256,256]},"observation.images.camera2":{"type":"VISUAL","shape":[3,256,256]},"observation.state":{"type":"STATE","shape":[8]}}'
python -m lerobot.scripts.lerobot_train \
  --policy.path="$BASE_POLICY" --policy.input_features="$INPUT_FEATURES" --policy.output_features=null \
  --policy.push_to_hub=false --policy.repo_id=giakhuyendihoc/smolvla-mambaact-libero-25k \
  --policy.freeze_vision_encoder=false --policy.train_expert_only=false --policy.train_state_proj=true \
  --policy.use_amp=false --policy.chunk_size=50 --policy.n_action_steps=50 --policy.num_steps=10 \
  --policy.optimizer_lr=1e-4 --policy.optimizer_betas='[0.9,0.95]' --policy.optimizer_eps=1e-8 \
  --policy.optimizer_weight_decay=1e-10 --policy.optimizer_grad_clip_norm=10 \
  --policy.scheduler_warmup_steps=1000 --policy.scheduler_decay_steps=25000 --policy.scheduler_decay_lr=2.5e-6 \
  --policy.use_mamba_action_expert=true --policy.mamba_depth=4 --policy.mamba_width=768 \
  --policy.mamba_context_tokens=64 --policy.mamba_d_state=64 --policy.mamba_expand=1 \
  --policy.mamba_headdim=64 --policy.mamba_chunk_size=64 \
  --dataset.repo_id=lerobot/libero --dataset.root="$DATASET_ROOT" \
  --dataset.revision=a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 \
  --dataset.video_backend=torchcodec --dataset.image_transforms.enable=false --rename_map="$RENAME_MAP" \
  --env.type=libero --env.task=libero_10 --output_dir="$OUTPUT_DIR" \
  --steps=25000 --batch_size=32 --num_workers=8 --save_freq=25000 \
  --env_eval_freq=0 --eval_steps=0 --log_freq=100 --seed=1000 --wandb.enable=false
FINAL_POLICY="$OUTPUT_DIR/checkpoints/025000/pretrained_model"
[[ -s "$FINAL_POLICY/model.safetensors" ]] || { echo "Final 25k checkpoint missing." >&2; exit 1; }
bash "$MAMBAACT_ROOT/scripts/eval_smolvla.sh" mambaact "$FINAL_POLICY"
