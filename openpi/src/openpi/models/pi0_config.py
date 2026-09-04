import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"
    # Optional PyTorch LoRA overrides. If lora_rank is set and lora_alpha is
    # left unset, alpha follows rank, matching the existing OpenPI defaults.
    lora_rank: int | None = None
    lora_alpha: float | None = None
    # Optional PyTorch Vision-LoRA. This applies adapters to the SigLIP vision tower.
    vision_lora_rank: int | None = None
    vision_lora_alpha: float | None = None
    # Heavy ablation: unfreeze the full vision tower instead of only Vision-LoRA adapters.
    train_vision_encoder: bool = False
    # Optional structured-pruning overrides for paired PaliGemma/action-expert
    # transformer depth. These are used by PruneVLA OpenPI-PyTorch checkpoints.
    paligemma_depth: int | None = None
    action_expert_depth: int | None = None
    # Optional structured-pruning override for the SigLIP vision encoder depth.
    # This is used by SigLIP-pruned OpenPI-PyTorch checkpoints.
    vision_depth: int | None = None
    pruned_layers: tuple[int, ...] = ()

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        if self.lora_rank is not None and self.lora_rank <= 0:
            raise ValueError(f"lora_rank must be positive, got {self.lora_rank}")
        if self.lora_alpha is not None and self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")
        if self.vision_lora_rank is not None and self.vision_lora_rank <= 0:
            raise ValueError(f"vision_lora_rank must be positive, got {self.vision_lora_rank}")
        if self.vision_lora_alpha is not None and self.vision_lora_alpha <= 0:
            raise ValueError(f"vision_lora_alpha must be positive, got {self.vision_lora_alpha}")
        if self.paligemma_depth is not None and self.paligemma_depth <= 0:
            raise ValueError(f"paligemma_depth must be positive, got {self.paligemma_depth}")
        if self.action_expert_depth is not None and self.action_expert_depth <= 0:
            raise ValueError(f"action_expert_depth must be positive, got {self.action_expert_depth}")
        if self.vision_depth is not None and self.vision_depth <= 0:
            raise ValueError(f"vision_depth must be positive, got {self.vision_depth}")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
