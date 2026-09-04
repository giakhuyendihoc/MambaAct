#!/usr/bin/env bash

set -euo pipefail

MAMBAACT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAMBAACT_OUTPUT_ROOT="${MAMBAACT_OUTPUT_ROOT:-$MAMBAACT_ROOT/artifacts}"
MAMBAACT_PI05_CHECKPOINT="${MAMBAACT_PI05_CHECKPOINT:-$MAMBAACT_OUTPUT_ROOT/checkpoints/pi05_base_pytorch}"

export PI_MAMBA_RESOURCE_ROOT="$MAMBAACT_OUTPUT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-$MAMBAACT_OUTPUT_ROOT/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export WANDB_DIR="${WANDB_DIR:-$MAMBAACT_OUTPUT_ROOT/wandb}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p "$MAMBAACT_OUTPUT_ROOT" "$HF_HOME" "$WANDB_DIR"

resolve_pi05_model() {
  case "$1" in
    mambaact)
      printf '%s\t%s\t%s\n' "mambaact_libero" "mambaact_l18" "mamba"
      ;;
    pi05)
      printf '%s\t%s\t%s\n' "pi05_libero_baseline" "pi05_l18" "pi05"
      ;;
    mambaact_l12)
      printf '%s\t%s\t%s\n' "mambaact_libero_l12" "mambaact_l12" "mamba"
      ;;
    mamba_cross_attention)
      printf '%s\t%s\t%s\n' "mambaact_libero_cross_attention" "mamba_cross_attention" "mamba"
      ;;
    transformer_gate)
      printf '%s\t%s\t%s\n' "transformer_action_libero_gated" "transformer_gate" "mamba"
      ;;
    transformer_cross_attention)
      printf '%s\t%s\t%s\n' "transformer_action_libero_cross_attention" "transformer_cross_attention" "mamba"
      ;;
    *)
      echo "Unknown model '$1'." >&2
      echo "Expected: mambaact, pi05, mambaact_l12, mamba_cross_attention, transformer_gate, or transformer_cross_attention." >&2
      return 2
      ;;
  esac
}
