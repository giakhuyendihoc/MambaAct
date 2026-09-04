import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class Pi0MambaConfig(pi0_config.Pi0Config):
    """PyTorch-only PI0/PI0.5 variant with a Mamba action expert."""

    mamba_width: int | None = None
    mamba_d_state: int = 128
    mamba_expand: int = 2
    mamba_headdim: int = 64
    mamba_chunk_size: int = 64
    mamba_is_mimo: bool = False
    mamba_depth: int | None = None
    # Action-token sequence mixer. "mamba" is the main model; "transformer"
    # keeps the same ctxD/gated interface for mixer ablations.
    mamba_action_mixer: Literal["mamba", "transformer"] = "mamba"
    mamba_context_num_heads: int = 8
    mamba_context_dropout: float = 0.0
    # Optional memory optimization: compress the PaliGemma prefix sequence to
    # this many context tokens before every Mamba action block attends to it.
    # None preserves the full prefix sequence for exact old-run reproducibility.
    mamba_context_compression_tokens: int | None = None
    # Advanced override for the compressed context split. If omitted, the
    # PyTorch model derives a vision/language/global split from K.
    mamba_context_vision_tokens: int | None = None
    mamba_context_language_tokens: int | None = None
    mamba_context_history_tokens: int | None = None
    mamba_context_global_tokens: int | None = None
    mamba_context_use_mlp: bool = True
    # Optional real VLM early-exit. If set to N, the prefix language model runs
    # exactly the first N PaliGemma language layers, skips the remaining layers,
    # then feeds that early hidden state to ctxD.
    mamba_vlm_early_exit_layer: int | None = None
    # Context conditioning used inside each Mamba action block. "cross_attn"
    # preserves the original ctxD behavior; "gated" pools ctxD tokens once and
    # uses lightweight FiLM gates around the Mamba recurrence; "gated_cross_attn"
    # keeps token-level ctxD attention but gates the cross-attention residual.
    mamba_context_conditioning: str = "cross_attn"
    mamba_mlp_dim: int | None = None
    # Hidden size for context gate/FiLM MLPs only. Defaults to mamba_mlp_dim
    # so existing configs keep identical behavior.
    mamba_context_gate_mlp_dim: int | None = None
    mamba_history_length: int = 0
    mamba_history_depth: int = 1
    # Optional override for low-dimensional history context. By default the
    # history compressor consumes state and action histories with action_dim
    # features each. Famba uses wider state history and no action history.
    mamba_history_state_dim: int | None = None
    mamba_history_action_dim: int | None = None
    # If enabled, the action-side Mamba stack consumes prefix representations
    # from matched VLM layer indices instead of only the final prefix output.
    mamba_layerwise_context: bool = False
    mamba_layerwise_context_indices: tuple[int, ...] = ()
    # If enabled, collect selected VLM hidden states, concatenate their vision
    # tokens first and language tokens second, then compress once with ctxD.
    mamba_multilayer_context: bool = False
    mamba_multilayer_context_indices: tuple[int, ...] = ()
    mamba_multilayer_context_add_layer_embed: bool = True

    def __post_init__(self):
        super().__post_init__()
        positive_int_fields = {
            "mamba_width": self.mamba_width,
            "mamba_d_state": self.mamba_d_state,
            "mamba_expand": self.mamba_expand,
            "mamba_headdim": self.mamba_headdim,
            "mamba_chunk_size": self.mamba_chunk_size,
            "mamba_depth": self.mamba_depth,
            "mamba_context_num_heads": self.mamba_context_num_heads,
            "mamba_context_compression_tokens": self.mamba_context_compression_tokens,
            "mamba_mlp_dim": self.mamba_mlp_dim,
            "mamba_context_gate_mlp_dim": self.mamba_context_gate_mlp_dim,
            "mamba_history_depth": self.mamba_history_depth,
            "mamba_history_state_dim": self.mamba_history_state_dim,
        }
        for field_name, value in positive_int_fields.items():
            if value is not None and value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")

        nonnegative_int_fields = {
            "mamba_context_vision_tokens": self.mamba_context_vision_tokens,
            "mamba_context_language_tokens": self.mamba_context_language_tokens,
            "mamba_context_history_tokens": self.mamba_context_history_tokens,
            "mamba_context_global_tokens": self.mamba_context_global_tokens,
            "mamba_history_length": self.mamba_history_length,
            "mamba_history_action_dim": self.mamba_history_action_dim,
        }
        for field_name, value in nonnegative_int_fields.items():
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be nonnegative, got {value}")
        if self.mamba_vlm_early_exit_layer is not None and self.mamba_vlm_early_exit_layer <= 0:
            raise ValueError(f"mamba_vlm_early_exit_layer must be positive, got {self.mamba_vlm_early_exit_layer}")
        valid_context_conditioning = {"cross_attn", "gated", "gated_cross_attn"}
        valid_action_mixers = {"mamba", "transformer"}
        if self.mamba_action_mixer not in valid_action_mixers:
            raise ValueError(
                f"Unsupported mamba_action_mixer={self.mamba_action_mixer!r}; "
                f"expected one of {sorted(valid_action_mixers)}"
            )

        if self.mamba_context_conditioning not in valid_context_conditioning:
            raise ValueError(
                "mamba_context_conditioning must be one of "
                f"{sorted(valid_context_conditioning)}, got {self.mamba_context_conditioning!r}"
            )

        split_values = (
            self.mamba_context_vision_tokens,
            self.mamba_context_language_tokens,
            self.mamba_context_history_tokens,
            self.mamba_context_global_tokens,
        )
        if self.mamba_context_compression_tokens is None and any(value is not None for value in split_values):
            raise ValueError(
                "mamba_context_compression_tokens must be set when explicit context split tokens are provided"
            )
        if self.mamba_context_compression_tokens is not None and all(value is not None for value in split_values):
            split_total = sum(split_values)
            if split_total != self.mamba_context_compression_tokens:
                raise ValueError(
                    "Explicit context split must sum to mamba_context_compression_tokens: "
                    f"{split_total} vs {self.mamba_context_compression_tokens}"
                )
        if (self.mamba_context_history_tokens or 0) > 0 and self.mamba_history_length <= 0:
            raise ValueError("mamba_history_length must be positive when mamba_context_history_tokens is positive")
        def validate_layer_indices(field_name: str, indices: tuple[int, ...]) -> None:
            previous = -1
            for layer_idx in indices:
                if layer_idx < 0:
                    raise ValueError(f"{field_name} must be nonnegative")
                if layer_idx <= previous:
                    raise ValueError(f"{field_name} must be strictly increasing")
                previous = layer_idx

        validate_layer_indices("mamba_layerwise_context_indices", self.mamba_layerwise_context_indices)
        validate_layer_indices("mamba_multilayer_context_indices", self.mamba_multilayer_context_indices)
        if self.mamba_layerwise_context and self.mamba_multilayer_context:
            raise ValueError("mamba_layerwise_context and mamba_multilayer_context are mutually exclusive")
        if self.mamba_vlm_early_exit_layer is not None and (
            self.mamba_layerwise_context or self.mamba_multilayer_context
        ):
            raise ValueError(
                "mamba_vlm_early_exit_layer is only supported with the shared final/early ctxD path, "
                "not layerwise or multilayer context"
            )

        width = self.mamba_width
        if width is not None:
            if width % self.mamba_headdim != 0:
                raise ValueError(f"mamba_width must be divisible by mamba_headdim: {width} vs {self.mamba_headdim}")
            if width % self.mamba_context_num_heads != 0:
                raise ValueError(
                    "mamba_width must be divisible by mamba_context_num_heads: "
                    f"{width} vs {self.mamba_context_num_heads}"
                )

    @override
    def create(self, rng: at.KeyArrayLike):
        raise NotImplementedError("Pi0MambaConfig is PyTorch-only; use scripts/train_pytorch.py.")

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        observation_spec, action_spec = super().inputs_spec(batch_size=batch_size)
        if (self.mamba_context_history_tokens or 0) <= 0:
            return observation_spec, action_spec

        with at.disable_typechecking():
            history_state_dim = self.mamba_history_state_dim or self.action_dim
            history_action_dim = self.action_dim if self.mamba_history_action_dim is None else self.mamba_history_action_dim
            observation_spec = dataclasses.replace(
                observation_spec,
                state_history=jax.ShapeDtypeStruct(
                    [batch_size, self.mamba_history_length, history_state_dim],
                    jnp.float32,
                ),
                action_history=(
                    jax.ShapeDtypeStruct([batch_size, self.mamba_history_length, history_action_dim], jnp.float32)
                    if history_action_dim > 0
                    else None
                ),
                history_mask=jax.ShapeDtypeStruct([batch_size, self.mamba_history_length], jnp.bool_),
            )
        return observation_spec, action_spec
