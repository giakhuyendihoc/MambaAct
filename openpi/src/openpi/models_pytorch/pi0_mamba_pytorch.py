import dataclasses
from functools import lru_cache
import logging
import math
from pathlib import Path
import sys
from typing import Literal

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

import openpi.models.gemma as _gemma
import openpi.models_pytorch.lora_pytorch as _lora_pytorch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


@lru_cache(maxsize=1)
def _load_mamba3_cls():
    workspace_root = Path(__file__).resolve().parents[4]
    local_mamba = workspace_root / "mamba"
    if local_mamba.exists():
        local_mamba_path = str(local_mamba)
        if local_mamba_path in sys.path:
            sys.path.remove(local_mamba_path)
        sys.path.insert(0, local_mamba_path)

    try:
        from mamba_ssm import Mamba3
    except ImportError as exc:
        raise ImportError(
            "Mamba3 is required for MambaAct. Run `bash scripts/setup.sh` from the repository root "
            "or install the vendored `mamba` package in the active environment."
        ) from exc

    logging.info("Using the Mamba3 SISO action mixer")
    return Mamba3


class PaliGemmaPrefixModel(nn.Module):
    """PaliGemma image/language prefix encoder without the transformer action expert."""

    def __init__(
        self,
        vlm_config,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        vision_depth: int | None = None,
    ):
        super().__init__()
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = False
        vlm_config_hf.text_config.adarms_cond_dim = None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"
        if vision_depth is not None:
            vlm_config_hf.vision_config.num_hidden_layers = vision_depth

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def _prefix_attention(
        self,
        layer,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        key_states = modeling_gemma.repeat_kv(key_states, layer.self_attn.num_key_value_groups)
        value_states = modeling_gemma.repeat_kv(value_states, layer.self_attn.num_key_value_groups)
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, : key_states.shape[-2]].to(dtype=query_states.dtype)
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0 if not self.training else layer.self_attn.attention_dropout,
            scale=layer.self_attn.scaling,
        )
        return attn_output.transpose(1, 2).contiguous()

    def _forward_prefix_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        language_model = self.paligemma.language_model
        layer = language_model.layers[layer_idx]

        residual = hidden_states
        hidden_states, gate = layer.input_layernorm(hidden_states, cond=None)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_states = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        dummy_tensor = torch.zeros(
            query_states.shape[0],
            query_states.shape[2],
            query_states.shape[-1],
            device=query_states.device,
            dtype=query_states.dtype,
        )
        cos, sin = language_model.rotary_emb(dummy_tensor, position_ids)
        query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            unsqueeze_dim=1,
        )

        attn_output = self._prefix_attention(layer, query_states, key_states, value_states, attention_mask)
        attn_output = attn_output.reshape(*input_shape, -1)
        if attn_output.dtype != layer.self_attn.o_proj.weight.dtype:
            attn_output = attn_output.to(dtype=layer.self_attn.o_proj.weight.dtype)
        attn_output = layer.self_attn.o_proj(attn_output)
        hidden_states = modeling_gemma._gated_residual(residual, attn_output, gate)  # noqa: SLF001

        residual = hidden_states
        hidden_states, gate = layer.post_attention_layernorm(hidden_states, cond=None)
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype=torch.bfloat16)
        hidden_states = layer.mlp(hidden_states)
        return modeling_gemma._gated_residual(residual, hidden_states, gate)  # noqa: SLF001

    def _forward_prefix_layerwise(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor,
        num_layers: int | None = None,
    ) -> torch.Tensor:
        if not getattr(self, "_prefix_layerwise_logged", False):
            logging.info("Using layer-wise checkpointed PaliGemma prefix encoder for PI0MambaPytorch")
            self._prefix_layerwise_logged = True

        language_model = self.paligemma.language_model
        hidden_states = inputs_embeds
        if language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype=torch.bfloat16)

        total_layers = language_model.config.num_hidden_layers
        layers_to_run = total_layers if num_layers is None else num_layers
        if layers_to_run < 1 or layers_to_run > total_layers:
            raise ValueError(f"num_layers must be in [1, {total_layers}], got {layers_to_run}")

        use_gradient_checkpointing = (
            getattr(language_model, "gradient_checkpointing", False)
            and self.training
            and torch.is_grad_enabled()
        )
        for layer_idx in range(layers_to_run):
            if use_gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    self._forward_prefix_layer,
                    layer_idx,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                hidden_states = self._forward_prefix_layer(layer_idx, hidden_states, attention_mask, position_ids)

        hidden_states, _ = language_model.norm(hidden_states, cond=None)
        return hidden_states

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if inputs_embeds is None or inputs_embeds[0] is None:
            raise ValueError("PaliGemmaPrefixModel only encodes prefix inputs.")

        if past_key_values is None and not use_cache:
            prefix_output = self._forward_prefix_layerwise(
                inputs_embeds[0],
                attention_mask,
                position_ids,
            )
            return [prefix_output, None], None

        prefix_output = self.paligemma.language_model.forward(
            inputs_embeds=inputs_embeds[0],
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return [prefix_output.last_hidden_state, None], prefix_output.past_key_values


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        normed = x_float * torch.rsqrt(torch.mean(x_float * x_float, dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(dtype=dtype)


class MambaActionBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        d_state: int,
        expand: int,
        headdim: int,
        chunk_size: int,
        is_mimo: bool,
        context_num_heads: int,
        context_dropout: float,
        mlp_dim: int,
        context_conditioning: str,
        context_gate_mlp_dim: int | None = None,
    ):
        super().__init__()
        if context_conditioning not in {"cross_attn", "gated", "gated_cross_attn"}:
            raise ValueError(f"Unsupported Mamba context conditioning: {context_conditioning}")
        self.context_conditioning = context_conditioning
        mamba3_cls = _load_mamba3_cls()
        self.cond_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

        gate_mlp_dim = context_gate_mlp_dim or mlp_dim

        if context_conditioning in {"cross_attn", "gated_cross_attn"}:
            self.context_norm = RMSNorm(d_model)
            self.context_attn = nn.MultiheadAttention(
                d_model,
                context_num_heads,
                dropout=context_dropout,
                batch_first=True,
            )
        else:
            self.context_norm = None
            self.context_attn = None

        if context_conditioning == "gated":
            self.context_gate_norm = RMSNorm(d_model)
            self.context_gate_mlp = nn.Sequential(
                nn.Linear(d_model, gate_mlp_dim),
                nn.SiLU(),
                nn.Linear(gate_mlp_dim, 3 * d_model),
            )
            final_gate = self.context_gate_mlp[-1]
            nn.init.zeros_(final_gate.weight)
            nn.init.zeros_(final_gate.bias)
            final_gate.bias.data[2 * d_model :] = math.log(0.2 / 0.8)
        elif context_conditioning == "gated_cross_attn":
            self.context_gate_norm = RMSNorm(d_model)
            self.context_gate_mlp = nn.Sequential(
                nn.Linear(d_model, gate_mlp_dim),
                nn.SiLU(),
                nn.Linear(gate_mlp_dim, d_model),
            )
            final_gate = self.context_gate_mlp[-1]
            nn.init.zeros_(final_gate.weight)
            nn.init.constant_(final_gate.bias, -2.0)
        else:
            self.context_gate_norm = None
            self.context_gate_mlp = None

        self.mamba_norm = RMSNorm(d_model)
        self.mamba = mamba3_cls(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
            is_mimo=is_mimo,
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, d_model),
        )

    def forward(
        self,
        action_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        context_key_padding_mask: torch.Tensor,
        cond: torch.Tensor | None,
        context_vector: torch.Tensor | None = None,
        context_cache: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        if cond is not None:
            cond = cond.to(dtype=self.cond_proj.weight.dtype)
            action_tokens = action_tokens + self.cond_proj(F.silu(cond)).to(dtype=action_tokens.dtype).unsqueeze(1)

        if self.context_conditioning == "cross_attn":
            context_tokens = context_tokens.to(dtype=action_tokens.dtype)
            attn_out, _ = self.context_attn(
                query=self.context_norm(action_tokens),
                key=context_tokens,
                value=context_tokens,
                key_padding_mask=context_key_padding_mask,
                need_weights=False,
            )
            action_tokens = action_tokens + attn_out
            action_tokens = action_tokens + self.mamba(self.mamba_norm(action_tokens))
        elif self.context_conditioning == "gated_cross_attn":
            if context_vector is None:
                raise ValueError("context_vector is required when mamba_context_conditioning='gated_cross_attn'")
            action_tokens = action_tokens + self.mamba(self.mamba_norm(action_tokens))
            context_tokens = context_tokens.to(dtype=action_tokens.dtype)
            attn_out, _ = self.context_attn(
                query=self.context_norm(action_tokens),
                key=context_tokens,
                value=context_tokens,
                key_padding_mask=context_key_padding_mask,
                need_weights=False,
            )
            if context_cache is None:
                (gate,) = self.prepare_context_cache(context_vector, output_dtype=action_tokens.dtype)
            else:
                (gate,) = context_cache
                gate = gate.to(dtype=action_tokens.dtype)
            action_tokens = action_tokens + gate[:, None, :] * attn_out
        else:
            if context_vector is None:
                raise ValueError("context_vector is required when mamba_context_conditioning='gated'")
            mamba_out = self.mamba(self.mamba_norm(action_tokens))
            if context_cache is None:
                scale, shift, gate = self.prepare_context_cache(context_vector, output_dtype=action_tokens.dtype)
            else:
                scale, shift, gate = context_cache
                scale = scale.to(dtype=action_tokens.dtype)
                shift = shift.to(dtype=action_tokens.dtype)
                gate = gate.to(dtype=action_tokens.dtype)
            mamba_out = (1 + scale[:, None, :]) * mamba_out + shift[:, None, :]
            action_tokens = action_tokens + gate[:, None, :] * mamba_out

        return action_tokens + self.ffn(self.ffn_norm(action_tokens))

    def prepare_context_cache(
        self,
        context_vector: torch.Tensor,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...] | None:
        if self.context_conditioning not in {"gated", "gated_cross_attn"}:
            return None
        if context_vector is None:
            raise ValueError("context_vector is required to prepare gated context cache")

        output_dtype = output_dtype or context_vector.dtype
        gate_input = self.context_gate_norm(context_vector.to(dtype=output_dtype))
        gate_input = gate_input.to(dtype=self.context_gate_mlp[0].weight.dtype)
        gate_values = self.context_gate_mlp(gate_input).to(dtype=output_dtype)
        if self.context_conditioning == "gated_cross_attn":
            return (torch.sigmoid(gate_values),)
        scale, shift, gate = gate_values.chunk(3, dim=-1)
        return scale, shift, torch.sigmoid(gate)


class TransformerActionBlock(nn.Module):
    """Action-token transformer block for ctxD action-mixer ablations."""

    def __init__(
        self,
        *,
        d_model: int,
        context_num_heads: int,
        context_dropout: float,
        mlp_dim: int,
        context_conditioning: str,
        context_gate_mlp_dim: int | None = None,
    ):
        super().__init__()
        if context_conditioning not in {"gated", "cross_attn"}:
            raise ValueError(
                "Transformer ctxD ablation supports mamba_context_conditioning='gated' or 'cross_attn'; "
                f"got {context_conditioning!r}"
            )
        self.context_conditioning = context_conditioning
        self.cond_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

        gate_mlp_dim = context_gate_mlp_dim or mlp_dim
        self.attn_norm = RMSNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model,
            context_num_heads,
            dropout=context_dropout,
            batch_first=True,
        )
        if context_conditioning == "cross_attn":
            self.context_norm = RMSNorm(d_model)
            self.context_attn = nn.MultiheadAttention(
                d_model,
                context_num_heads,
                dropout=context_dropout,
                batch_first=True,
            )
            self.context_gate_norm = None
            self.context_gate_mlp = None
        else:
            self.context_norm = None
            self.context_attn = None
            self.context_gate_norm = RMSNorm(d_model)
            self.context_gate_mlp = nn.Sequential(
                nn.Linear(d_model, gate_mlp_dim),
                nn.SiLU(),
                nn.Linear(gate_mlp_dim, 3 * d_model),
            )
            final_gate = self.context_gate_mlp[-1]
            nn.init.zeros_(final_gate.weight)
            nn.init.zeros_(final_gate.bias)
            final_gate.bias.data[2 * d_model :] = math.log(0.2 / 0.8)

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, d_model),
        )

    def forward(
        self,
        action_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        context_key_padding_mask: torch.Tensor,
        cond: torch.Tensor | None,
        context_vector: torch.Tensor | None = None,
        context_cache: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        if cond is not None:
            cond = cond.to(dtype=self.cond_proj.weight.dtype)
            action_tokens = action_tokens + self.cond_proj(F.silu(cond)).to(dtype=action_tokens.dtype).unsqueeze(1)

        attn_in = self.attn_norm(action_tokens)
        attn_out, _ = self.self_attn(
            query=attn_in,
            key=attn_in,
            value=attn_in,
            need_weights=False,
        )
        if self.context_conditioning == "cross_attn":
            action_tokens = action_tokens + attn_out
            context_tokens = context_tokens.to(dtype=action_tokens.dtype)
            ctx_out, _ = self.context_attn(
                query=self.context_norm(action_tokens),
                key=context_tokens,
                value=context_tokens,
                key_padding_mask=context_key_padding_mask,
                need_weights=False,
            )
            action_tokens = action_tokens + ctx_out
        else:
            if context_vector is None:
                raise ValueError("context_vector is required for transformer pure-gated ctxD conditioning")
            if context_cache is None:
                scale, shift, gate = self.prepare_context_cache(context_vector, output_dtype=action_tokens.dtype)
            else:
                scale, shift, gate = context_cache
                scale = scale.to(dtype=action_tokens.dtype)
                shift = shift.to(dtype=action_tokens.dtype)
                gate = gate.to(dtype=action_tokens.dtype)
            attn_out = (1 + scale[:, None, :]) * attn_out + shift[:, None, :]
            action_tokens = action_tokens + gate[:, None, :] * attn_out
        return action_tokens + self.ffn(self.ffn_norm(action_tokens))

    def prepare_context_cache(
        self,
        context_vector: torch.Tensor,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...] | None:
        if self.context_conditioning != "gated":
            return None
        output_dtype = output_dtype or context_vector.dtype
        gate_input = self.context_gate_norm(context_vector.to(dtype=output_dtype))
        gate_input = gate_input.to(dtype=self.context_gate_mlp[0].weight.dtype)
        scale, shift, gate = self.context_gate_mlp(gate_input).to(dtype=output_dtype).chunk(3, dim=-1)
        return scale, shift, torch.sigmoid(gate)


class HistoryMambaBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        d_state: int,
        expand: int,
        headdim: int,
        chunk_size: int,
        is_mimo: bool,
        mlp_dim: int,
    ):
        super().__init__()
        mamba3_cls = _load_mamba3_cls()
        self.mamba_norm = RMSNorm(d_model)
        self.mamba = mamba3_cls(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
            is_mimo=is_mimo,
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, d_model),
        )

    def forward(self, history_tokens: torch.Tensor) -> torch.Tensor:
        history_tokens = history_tokens + self.mamba(self.mamba_norm(history_tokens))
        return history_tokens + self.ffn(self.ffn_norm(history_tokens))


class HistoryMambaCompressor(nn.Module):
    def __init__(self, config, *, model_action_dim: int, d_model: int, mlp_dim: int):
        super().__init__()
        self.history_tokens = config.mamba_context_history_tokens or 0
        self.history_length = config.mamba_history_length
        self.model_action_dim = model_action_dim
        self.history_state_dim = getattr(config, "mamba_history_state_dim", None) or model_action_dim
        action_dim_override = getattr(config, "mamba_history_action_dim", None)
        self.history_action_dim = model_action_dim if action_dim_override is None else action_dim_override
        self.step_proj = nn.Linear(self.history_state_dim + self.history_action_dim + 1, d_model)
        self.blocks = nn.ModuleList(
            [
                HistoryMambaBlock(
                    d_model=d_model,
                    d_state=config.mamba_d_state,
                    expand=config.mamba_expand,
                    headdim=config.mamba_headdim,
                    chunk_size=config.mamba_chunk_size,
                    is_mimo=config.mamba_is_mimo,
                    mlp_dim=mlp_dim,
                )
                for _ in range(config.mamba_history_depth)
            ]
        )
        self.final_norm = RMSNorm(d_model)

    def _empty_context(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_ctx = torch.zeros(batch_size, self.history_tokens, self.step_proj.out_features, device=device, dtype=dtype)
        history_mask = torch.zeros(batch_size, self.history_tokens, device=device, dtype=torch.bool)
        return history_ctx, history_mask

    @staticmethod
    def _segment_mean_pool(
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        num_segments: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compressed_tokens = []
        compressed_masks = []
        seq_len = tokens.shape[1]
        for token_idx in range(num_segments):
            start = token_idx * seq_len // num_segments
            end = (token_idx + 1) * seq_len // num_segments
            segment_tokens = tokens[:, start:end, :]
            segment_mask = valid_mask[:, start:end].bool()
            segment_weights = segment_mask.to(dtype=segment_tokens.dtype).unsqueeze(-1)
            denom = segment_weights.sum(dim=1).clamp_min(1)
            compressed_tokens.append((segment_tokens * segment_weights).sum(dim=1) / denom)
            compressed_masks.append(segment_mask.any(dim=1))
        return torch.stack(compressed_tokens, dim=1), torch.stack(compressed_masks, dim=1)

    def forward(
        self,
        state_history: torch.Tensor | None,
        action_history: torch.Tensor | None,
        history_mask: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = self.step_proj.weight.dtype
        if self.history_tokens <= 0:
            return self._empty_context(batch_size=batch_size, device=device, dtype=dtype)
        if state_history is None:
            return self._empty_context(batch_size=batch_size, device=device, dtype=dtype)

        state_history = state_history.to(device=device, dtype=dtype)
        if history_mask is None:
            history_mask = torch.ones(state_history.shape[:2], device=device, dtype=torch.bool)
        else:
            history_mask = history_mask.to(device=device, dtype=torch.bool)

        state_history = state_history[..., : self.history_state_dim]
        if state_history.shape[-1] < self.history_state_dim:
            state_history = F.pad(state_history, (0, self.history_state_dim - state_history.shape[-1]))
        history_parts = [state_history]
        if self.history_action_dim > 0:
            if action_history is None:
                action_history = torch.zeros(
                    state_history.shape[0],
                    state_history.shape[1],
                    self.history_action_dim,
                    device=device,
                    dtype=dtype,
                )
            else:
                action_history = action_history.to(device=device, dtype=dtype)[..., : self.history_action_dim]
                if action_history.shape[-1] < self.history_action_dim:
                    action_history = F.pad(action_history, (0, self.history_action_dim - action_history.shape[-1]))
            history_parts.append(action_history)
        seq_len = state_history.shape[1]
        rel_time = torch.arange(-seq_len, 0, device=device, dtype=dtype).view(1, seq_len, 1)
        rel_time = rel_time / max(seq_len, 1)
        rel_time = rel_time.expand(state_history.shape[0], -1, -1)
        history_parts.append(rel_time)

        step_inputs = torch.cat(history_parts, dim=-1)
        hidden = self.step_proj(step_inputs)
        valid_weights = history_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        hidden = hidden * valid_weights
        for block in self.blocks:
            hidden = block(hidden)
            hidden = hidden * valid_weights
        hidden = self.final_norm(hidden)
        return self._segment_mean_pool(hidden, history_mask, self.history_tokens)


class MambaActionExpert(nn.Module):
    def __init__(self, config, *, prefix_dim: int, action_dim: int):
        super().__init__()
        depth = config.mamba_depth or _gemma.get_config(config.action_expert_variant).depth
        mlp_dim = config.mamba_mlp_dim or action_dim * 4
        context_gate_mlp_dim = config.mamba_context_gate_mlp_dim
        self.context_compression_tokens = config.mamba_context_compression_tokens
        (
            self.context_vision_tokens,
            self.context_language_tokens,
            self.context_history_tokens,
            self.context_global_tokens,
        ) = self._resolve_context_token_counts(config)
        self.context_proj = nn.Linear(prefix_dim, action_dim)
        self.context_use_mlp = config.mamba_context_use_mlp
        self.context_conditioning = config.mamba_context_conditioning
        logging.info("PI-Mamba context conditioning: %s", self.context_conditioning)
        if self.context_compression_tokens is not None:
            self.vision_type_embed = nn.Parameter(torch.zeros(1, 1, action_dim))
            self.language_type_embed = nn.Parameter(torch.zeros(1, 1, action_dim))
            self.history_type_embed = nn.Parameter(torch.zeros(1, 1, action_dim))
            self.global_type_embed = nn.Parameter(torch.zeros(1, 1, action_dim))
            if self.context_use_mlp:
                self.context_norm = RMSNorm(action_dim)
                self.context_mlp = nn.Sequential(
                    nn.Linear(action_dim, mlp_dim),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(mlp_dim, action_dim),
                )
            else:
                self.context_norm = None
                self.context_mlp = None
        else:
            self.vision_type_embed = None
            self.language_type_embed = None
            self.history_type_embed = None
            self.global_type_embed = None
            self.context_norm = None
            self.context_mlp = None

        if self.context_history_tokens > 0:
            self.history_compressor = HistoryMambaCompressor(
                config,
                model_action_dim=config.action_dim,
                d_model=action_dim,
                mlp_dim=mlp_dim,
            )
        else:
            self.history_compressor = None

        if self.context_global_tokens > 0:
            self.global_context_queries = nn.Parameter(torch.randn(self.context_global_tokens, action_dim) * 0.02)
            self.global_context_attn = nn.MultiheadAttention(
                action_dim,
                config.mamba_context_num_heads,
                dropout=config.mamba_context_dropout,
                batch_first=True,
            )
        else:
            self.global_context_queries = None
            self.global_context_attn = None

        self.action_mixer = getattr(config, "mamba_action_mixer", "mamba")
        logging.info("PI-Mamba action mixer: %s", self.action_mixer)
        if self.action_mixer == "mamba":
            self.blocks = nn.ModuleList(
                [
                    MambaActionBlock(
                        d_model=action_dim,
                        d_state=config.mamba_d_state,
                        expand=config.mamba_expand,
                        headdim=config.mamba_headdim,
                        chunk_size=config.mamba_chunk_size,
                        is_mimo=config.mamba_is_mimo,
                        context_num_heads=config.mamba_context_num_heads,
                        context_dropout=config.mamba_context_dropout,
                        mlp_dim=mlp_dim,
                        context_conditioning=self.context_conditioning,
                        context_gate_mlp_dim=context_gate_mlp_dim,
                    )
                    for _ in range(depth)
                ]
            )
        elif self.action_mixer == "transformer":
            self.blocks = nn.ModuleList(
                [
                    TransformerActionBlock(
                        d_model=action_dim,
                        context_num_heads=config.mamba_context_num_heads,
                        context_dropout=config.mamba_context_dropout,
                        mlp_dim=mlp_dim,
                        context_conditioning=self.context_conditioning,
                        context_gate_mlp_dim=context_gate_mlp_dim,
                    )
                    for _ in range(depth)
                ]
            )
        else:
            raise ValueError(f"Unsupported mamba_action_mixer={self.action_mixer!r}")
        self.final_norm = RMSNorm(action_dim)

    @staticmethod
    def _resolve_context_token_counts(config) -> tuple[int, int, int, int]:
        total_tokens = config.mamba_context_compression_tokens
        if total_tokens is None:
            return 0, 0, 0, 0

        global_tokens = config.mamba_context_global_tokens
        if global_tokens is None:
            global_tokens = min(max(2, total_tokens // 16), total_tokens)

        history_tokens = config.mamba_context_history_tokens
        if history_tokens is None:
            history_tokens = 0

        language_tokens = config.mamba_context_language_tokens
        if language_tokens is None:
            language_tokens = min(max(1, total_tokens // 4), max(total_tokens - global_tokens - history_tokens, 0))

        vision_tokens = config.mamba_context_vision_tokens
        if vision_tokens is None:
            vision_tokens = max(total_tokens - language_tokens - history_tokens - global_tokens, 0)

        split_total = vision_tokens + language_tokens + history_tokens + global_tokens
        if split_total != total_tokens:
            raise ValueError(
                "Mamba context split must sum to mamba_context_compression_tokens: "
                f"{split_total} vs {total_tokens}"
            )
        return vision_tokens, language_tokens, history_tokens, global_tokens

    def _segment_mean_pool(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        num_segments: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_segments <= 0:
            empty_tokens = tokens.new_zeros(tokens.shape[0], 0, tokens.shape[-1])
            empty_mask = valid_mask.new_zeros(valid_mask.shape[0], 0, dtype=torch.bool)
            return empty_tokens, empty_mask

        compressed_tokens = []
        compressed_masks = []
        seq_len = tokens.shape[1]
        for token_idx in range(num_segments):
            start = token_idx * seq_len // num_segments
            end = (token_idx + 1) * seq_len // num_segments
            segment_tokens = tokens[:, start:end, :]
            segment_mask = valid_mask[:, start:end].bool()
            segment_weights = segment_mask.to(dtype=segment_tokens.dtype).unsqueeze(-1)
            denom = segment_weights.sum(dim=1).clamp_min(1)
            compressed_tokens.append((segment_tokens * segment_weights).sum(dim=1) / denom)
            compressed_masks.append(segment_mask.any(dim=1))

        return torch.stack(compressed_tokens, dim=1), torch.stack(compressed_masks, dim=1)

    @staticmethod
    def split_segment_counts(total_segments: int, num_parts: int) -> tuple[int, ...]:
        if num_parts <= 0:
            raise ValueError(f"num_parts must be positive, got {num_parts}")
        base = total_segments // num_parts
        remainder = total_segments % num_parts
        return tuple(base + (1 if idx < remainder else 0) for idx in range(num_parts))

    def summarize_context_layer(
        self,
        prefix_tokens: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        language_token_count: int,
        *,
        vision_segments: int,
        language_segments: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        projected_prefix = self.context_proj(prefix_tokens.to(dtype=self.context_proj.weight.dtype))
        prefix_pad_masks = prefix_pad_masks.bool()
        language_token_count = min(language_token_count, projected_prefix.shape[1])
        vision_token_count = projected_prefix.shape[1] - language_token_count

        vision_ctx, vision_mask = self._segment_mean_pool(
            projected_prefix[:, :vision_token_count],
            prefix_pad_masks[:, :vision_token_count],
            vision_segments,
        )
        language_ctx, language_mask = self._segment_mean_pool(
            projected_prefix[:, vision_token_count:],
            prefix_pad_masks[:, vision_token_count:],
            language_segments,
        )
        return vision_ctx, vision_mask, language_ctx, language_mask

    def prepare_context_from_projected_summaries(
        self,
        vision_tokens: torch.Tensor,
        vision_mask: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        state_history: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.context_compression_tokens is None:
            projected_prefix = torch.cat([vision_tokens, language_tokens], dim=1)
            prefix_mask = torch.cat([vision_mask, language_mask], dim=1).bool()
            return projected_prefix, ~prefix_mask

        blind_pooling = (
            self.context_language_tokens == 0
            and self.context_history_tokens == 0
            and self.context_global_tokens == 0
        )
        if blind_pooling:
            projected_prefix = torch.cat([vision_tokens, language_tokens], dim=1)
            prefix_mask = torch.cat([vision_mask, language_mask], dim=1).bool()
            vision_ctx, vision_mask = self._segment_mean_pool(
                projected_prefix,
                prefix_mask,
                self.context_vision_tokens,
            )
            vision_ctx = vision_ctx + self.vision_type_embed.to(dtype=vision_ctx.dtype)
            context_parts = [vision_ctx]
            mask_parts = [vision_mask]
        else:
            projected_prefix = torch.cat([vision_tokens, language_tokens], dim=1)
            prefix_mask = torch.cat([vision_mask, language_mask], dim=1).bool()
            vision_ctx, vision_mask = self._segment_mean_pool(
                vision_tokens,
                vision_mask,
                self.context_vision_tokens,
            )
            language_ctx, language_mask = self._segment_mean_pool(
                language_tokens,
                language_mask,
                self.context_language_tokens,
            )
            vision_ctx = vision_ctx + self.vision_type_embed.to(dtype=vision_ctx.dtype)
            language_ctx = language_ctx + self.language_type_embed.to(dtype=language_ctx.dtype)
            context_parts = [vision_ctx, language_ctx]
            mask_parts = [vision_mask, language_mask]

        if self.context_history_tokens > 0:
            history_ctx, history_valid_mask = self.history_compressor(
                state_history,
                action_history,
                history_mask,
                batch_size=projected_prefix.shape[0],
                device=projected_prefix.device,
            )
            history_ctx = history_ctx + self.history_type_embed.to(dtype=history_ctx.dtype)
            context_parts.append(history_ctx)
            mask_parts.append(history_valid_mask)
        if self.context_global_tokens > 0:
            global_queries = self.global_context_queries.to(dtype=projected_prefix.dtype)
            global_queries = global_queries.unsqueeze(0).expand(projected_prefix.shape[0], -1, -1)
            global_ctx, _ = self.global_context_attn(
                query=global_queries,
                key=projected_prefix,
                value=projected_prefix,
                key_padding_mask=~prefix_mask,
                need_weights=False,
            )
            global_ctx = global_ctx + self.global_type_embed.to(dtype=global_ctx.dtype)
            global_mask = prefix_mask.new_ones(projected_prefix.shape[0], self.context_global_tokens)
            context_parts.append(global_ctx)
            mask_parts.append(global_mask)

        context_tokens = torch.cat(context_parts, dim=1)
        context_mask = torch.cat(mask_parts, dim=1)
        if self.context_mlp is not None:
            context_tokens = context_tokens + self.context_mlp(self.context_norm(context_tokens))
        return context_tokens, ~context_mask.bool()

    def prepare_context(
        self,
        prefix_tokens: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        language_token_count: int,
        state_history: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected_prefix = self.context_proj(prefix_tokens.to(dtype=self.context_proj.weight.dtype))
        prefix_pad_masks = prefix_pad_masks.bool()
        if self.context_compression_tokens is None:
            return projected_prefix, ~prefix_pad_masks

        language_token_count = min(language_token_count, projected_prefix.shape[1])
        vision_token_count = projected_prefix.shape[1] - language_token_count
        return self.prepare_context_from_projected_summaries(
            projected_prefix[:, :vision_token_count],
            prefix_pad_masks[:, :vision_token_count],
            projected_prefix[:, vision_token_count:],
            prefix_pad_masks[:, vision_token_count:],
            state_history=state_history,
            action_history=action_history,
            history_mask=history_mask,
        )

    def pool_context(
        self,
        context_tokens: torch.Tensor,
        context_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = ~context_key_padding_mask.bool()
        weights = valid_mask.to(dtype=context_tokens.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1)
        return (context_tokens * weights).sum(dim=1) / denom

    def prepare_context_cache(
        self,
        context_tokens: torch.Tensor,
        context_key_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, ...] | None, ...]] | None:
        if self.context_conditioning not in {"gated", "gated_cross_attn"}:
            return None
        context_vector = self.pool_context(context_tokens, context_key_padding_mask)
        block_caches = tuple(
            block.prepare_context_cache(context_vector, output_dtype=context_tokens.dtype) for block in self.blocks
        )
        return context_vector, block_caches

    def prepare_layerwise_context_caches(
        self,
        contexts: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[tuple[torch.Tensor, tuple[torch.Tensor, ...] | None], ...] | None:
        if self.context_conditioning not in {"gated", "gated_cross_attn"}:
            return None
        if len(contexts) != len(self.blocks):
            raise ValueError(
                f"Layerwise context count must match blocks for cache: {len(contexts)} vs {len(self.blocks)}"
            )
        caches = []
        for block, (context_tokens, context_key_padding_mask) in zip(self.blocks, contexts, strict=True):
            context_vector = self.pool_context(context_tokens, context_key_padding_mask)
            block_cache = block.prepare_context_cache(context_vector, output_dtype=context_tokens.dtype)
            caches.append((context_vector, block_cache))
        return tuple(caches)

    def forward(
        self,
        suffix_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        context_key_padding_mask: torch.Tensor,
        cond: torch.Tensor | None,
        context_cache: tuple[torch.Tensor, tuple[tuple[torch.Tensor, ...] | None, ...]] | None = None,
    ) -> torch.Tensor:
        hidden = suffix_tokens
        context_vector = None
        block_caches = None
        if context_cache is not None:
            context_vector, block_caches = context_cache
        elif self.context_conditioning in {"gated", "gated_cross_attn"}:
            context_vector = self.pool_context(context_tokens, context_key_padding_mask)
        for index, block in enumerate(self.blocks):
            block_cache = block_caches[index] if block_caches is not None else None
            hidden = block(hidden, context_tokens, context_key_padding_mask, cond, context_vector, block_cache)
        return self.final_norm(hidden)


class PI0MambaPytorch(PI0Pytorch):
    """PI0/PI0.5 PyTorch model with a Mamba3 SISO action expert."""

    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        if config.paligemma_depth is not None:
            paligemma_config = dataclasses.replace(paligemma_config, depth=config.paligemma_depth)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        action_width = config.mamba_width or action_expert_config.width

        self.paligemma_backbone = PaliGemmaPrefixModel(
            paligemma_config,
            precision=config.dtype,
            vision_depth=config.vision_depth,
        )
        paligemma_lora_layers = _lora_pytorch.apply_gemma_lora(
            self.paligemma_backbone.paligemma.language_model,
            paligemma_config,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
        )
        vision_lora_layers = _lora_pytorch.apply_vision_lora(
            self.paligemma_backbone.paligemma.vision_tower,
            rank=config.vision_lora_rank,
            alpha=config.vision_lora_alpha,
        )
        if paligemma_lora_layers:
            logging.info("Applied PyTorch LoRA to PI0MambaPytorch PaliGemma: %d linear layers", paligemma_lora_layers)
        if vision_lora_layers:
            logging.info(
                "Applied PyTorch Vision-LoRA to PI0MambaPytorch: vision=%d linear layers",
                vision_lora_layers,
            )

        self.action_in_proj = nn.Linear(config.action_dim, action_width)
        self.action_out_proj = nn.Linear(action_width, config.action_dim)
        self.mamba_action_expert = MambaActionExpert(
            config,
            prefix_dim=paligemma_config.width,
            action_dim=action_width,
        )
        self.multilayer_context_layer_embed = None
        if config.mamba_multilayer_context and config.mamba_multilayer_context_add_layer_embed:
            self.multilayer_context_layer_embed = nn.Embedding(paligemma_config.depth, paligemma_config.width)
            nn.init.normal_(self.multilayer_context_layer_embed.weight, std=0.02)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_width, action_width)
            self.time_mlp_out = nn.Linear(action_width, action_width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_width)
            self.action_time_mlp_in = nn.Linear(2 * action_width, action_width)
            self.action_time_mlp_out = nn.Linear(action_width, action_width)

        self._cast_action_path(config.dtype)

        torch.set_float32_matmul_precision("high")
        if config.pytorch_compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)

        self.gradient_checkpointing_enabled = False
        self._layerwise_context_indices: tuple[int, ...] | None = None
        self._multilayer_context_indices: tuple[int, ...] | None = None
        self._early_exit_logged = False

        msg = (
            "transformers_replace is not installed correctly. Please install it with "
            "`uv pip install transformers==4.53.2` and "
            "`cp -r ./src/openpi/models_pytorch/transformers_replace/* "
            ".venv/lib/python3.11/site-packages/transformers/`."
        )
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def _cast_action_path(self, precision: Literal["bfloat16", "float32"]) -> None:
        if precision == "float32":
            return
        if precision != "bfloat16":
            raise ValueError(f"Invalid precision: {precision}")

        action_modules = [self.action_in_proj, self.action_out_proj, self.mamba_action_expert]
        if self.pi05:
            action_modules.extend([self.time_mlp_in, self.time_mlp_out])
        else:
            action_modules.extend([self.state_proj, self.action_time_mlp_in, self.action_time_mlp_out])
        for module in action_modules:
            module.to(dtype=torch.bfloat16)
        logging.info("Cast PI0MambaPytorch action path to bfloat16")

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        self.paligemma_backbone.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_backbone.paligemma.vision_tower.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI0MambaPytorch PaliGemma backbone")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.paligemma_backbone.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_backbone.paligemma.vision_tower.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI0MambaPytorch PaliGemma backbone")

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_backbone.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_backbone.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], len(att_masks))

        return embs, pad_masks, att_masks

    def _preprocess_observation(self, observation, *, train=True):
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
            observation.state_history,
            observation.action_history,
            observation.history_mask,
        )

    def configure_trainable_parameters(self):
        has_lora = "lora" in self.config.paligemma_variant or self.config.vision_lora_rank is not None
        if not has_lora:
            _lora_pytorch.log_trainable_parameters(self, prefix="PI0MambaPytorch full-finetune ")
            return

        trainable_substrings = (
            "lora_",
            "action_in_proj",
            "action_out_proj",
            "state_proj",
            "action_time_mlp",
            "time_mlp",
            "mamba_action_expert",
        )
        _lora_pytorch.set_requires_grad_by_name(self, trainable_substrings=trainable_substrings)
        if self.config.train_vision_encoder:
            for name, param in self.named_parameters():
                if "vision_tower" in name or "vision_model" in name:
                    param.requires_grad = True
        _lora_pytorch.log_trainable_parameters(self, prefix="PI0MambaPytorch ")

    def load_pretrained_weights(self, weight_path: str, *, strict: bool = False):
        return _lora_pytorch.load_state_dict_with_lora_remap(
            self,
            weight_path,
            key_prefix_rewrites=(("paligemma_with_expert.paligemma.", "paligemma_backbone.paligemma."),),
            strict=strict,
        )

    def _resolve_vlm_early_exit_layer(self) -> int | None:
        early_exit_layer = getattr(self.config, "mamba_vlm_early_exit_layer", None)
        if early_exit_layer is None:
            return None
        total_layers = self.paligemma_backbone.paligemma.language_model.config.num_hidden_layers
        if early_exit_layer < 1 or early_exit_layer > total_layers:
            raise ValueError(f"mamba_vlm_early_exit_layer must be in [1, {total_layers}], got {early_exit_layer}")
        if not self._early_exit_logged:
            logging.info(
                "PI-Mamba VLM early-exit: vlm_early_exit_layer=%d num_vlm_layers_executed=%d num_vlm_layers_skipped=%d",
                early_exit_layer,
                early_exit_layer,
                total_layers - early_exit_layer,
            )
            self._early_exit_logged = True
        return early_exit_layer

    def _encode_prefix(self, prefix_embs, prefix_pad_masks, prefix_att_masks) -> torch.Tensor:
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_backbone.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        early_exit_layer = self._resolve_vlm_early_exit_layer()

        def forward_prefix(prefix_embs, prefix_att_2d_masks_4d, prefix_position_ids):
            return self.paligemma_backbone._forward_prefix_layerwise(
                prefix_embs,
                prefix_att_2d_masks_4d,
                prefix_position_ids,
                num_layers=early_exit_layer,
            )

        return self._apply_checkpoint(forward_prefix, prefix_embs, prefix_att_2d_masks_4d, prefix_position_ids)

    def _prepare_mamba_context(
        self,
        prefix_out,
        prefix_pad_masks,
        language_token_count: int,
        state_history=None,
        action_history=None,
        history_mask=None,
    ):
        return self.mamba_action_expert.prepare_context(
            prefix_out,
            prefix_pad_masks,
            language_token_count,
            state_history=state_history,
            action_history=action_history,
            history_mask=history_mask,
        )

    def _prefix_backbone_requires_grad(self) -> bool:
        return any(param.requires_grad for param in self.paligemma_backbone.parameters())

    def _resolve_multilayer_context_indices(self) -> tuple[int, ...]:
        if self._multilayer_context_indices is not None:
            return self._multilayer_context_indices

        num_prefix_layers = self.paligemma_backbone.paligemma.language_model.config.num_hidden_layers
        indices = tuple(getattr(self.config, "mamba_multilayer_context_indices", ()) or ())
        if not indices:
            raise ValueError("mamba_multilayer_context_indices must be set when mamba_multilayer_context is enabled")
        if len(set(indices)) != len(indices):
            raise ValueError(f"mamba_multilayer_context_indices must be unique: {indices}")
        if any(layer_idx < 0 or layer_idx >= num_prefix_layers for layer_idx in indices):
            raise ValueError(
                f"mamba_multilayer_context_indices must be within [0, {num_prefix_layers - 1}]: {indices}"
            )
        if tuple(sorted(indices)) != indices:
            raise ValueError(f"mamba_multilayer_context_indices must be sorted ascending: {indices}")

        self._multilayer_context_indices = indices
        logging.info("Using shared multi-layer PI-Mamba ctxD context at VLM layers: %s", indices)
        return indices

    def _encode_prefix_multilayer_context(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        language_token_count: int,
        state_history=None,
        action_history=None,
        history_mask=None,
        detach_prefix_context: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_backbone.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        language_model = self.paligemma_backbone.paligemma.language_model
        hidden_states = prefix_embs
        if language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype=torch.bfloat16)

        context_indices = self._resolve_multilayer_context_indices()
        context_index_set = set(context_indices)
        vision_segment_counts = self.mamba_action_expert.split_segment_counts(
            self.mamba_action_expert.context_vision_tokens,
            len(context_indices),
        )
        language_segment_counts = self.mamba_action_expert.split_segment_counts(
            self.mamba_action_expert.context_language_tokens,
            len(context_indices),
        )
        vision_summaries: list[torch.Tensor] = []
        vision_summary_masks: list[torch.Tensor] = []
        language_summaries: list[torch.Tensor] = []
        language_summary_masks: list[torch.Tensor] = []
        use_gradient_checkpointing = (
            getattr(language_model, "gradient_checkpointing", False)
            and self.training
            and torch.is_grad_enabled()
            and not detach_prefix_context
        )

        for layer_idx in range(language_model.config.num_hidden_layers):
            if use_gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    self.paligemma_backbone._forward_prefix_layer,
                    layer_idx,
                    hidden_states,
                    prefix_att_2d_masks_4d,
                    prefix_position_ids,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            elif detach_prefix_context:
                with torch.no_grad():
                    hidden_states = self.paligemma_backbone._forward_prefix_layer(
                        layer_idx,
                        hidden_states,
                        prefix_att_2d_masks_4d,
                        prefix_position_ids,
                    )
                hidden_states = hidden_states.detach()
            else:
                hidden_states = self.paligemma_backbone._forward_prefix_layer(
                    layer_idx,
                    hidden_states,
                    prefix_att_2d_masks_4d,
                    prefix_position_ids,
                )

            if layer_idx in context_index_set:
                summary_idx = len(vision_summaries)
                context_source = hidden_states.detach() if detach_prefix_context else hidden_states
                if self.multilayer_context_layer_embed is not None:
                    layer_embed = self.multilayer_context_layer_embed.weight[layer_idx]
                    layer_embed = layer_embed.to(device=context_source.device, dtype=context_source.dtype)
                    context_source = context_source + layer_embed.view(1, 1, -1)
                vision_ctx, vision_mask, language_ctx, language_mask = self.mamba_action_expert.summarize_context_layer(
                    context_source,
                    prefix_pad_masks,
                    language_token_count,
                    vision_segments=vision_segment_counts[summary_idx],
                    language_segments=language_segment_counts[summary_idx],
                )
                vision_summaries.append(vision_ctx)
                vision_summary_masks.append(vision_mask)
                language_summaries.append(language_ctx)
                language_summary_masks.append(language_mask)

        if len(vision_summaries) != len(context_indices):
            raise RuntimeError(f"Expected {len(context_indices)} multi-layer context states, got {len(vision_summaries)}")

        return self.mamba_action_expert.prepare_context_from_projected_summaries(
            torch.cat(vision_summaries, dim=1),
            torch.cat(vision_summary_masks, dim=1),
            torch.cat(language_summaries, dim=1),
            torch.cat(language_summary_masks, dim=1),
            state_history=state_history,
            action_history=action_history,
            history_mask=history_mask,
        )

    def _resolve_layerwise_context_indices(self) -> tuple[int, ...]:
        if self._layerwise_context_indices is not None:
            return self._layerwise_context_indices

        num_prefix_layers = self.paligemma_backbone.paligemma.language_model.config.num_hidden_layers
        num_mamba_blocks = len(self.mamba_action_expert.blocks)
        configured_indices = tuple(getattr(self.config, "mamba_layerwise_context_indices", ()) or ())

        if configured_indices:
            indices = configured_indices
        elif num_mamba_blocks == num_prefix_layers:
            indices = tuple(range(num_prefix_layers))
        elif num_mamba_blocks < num_prefix_layers:
            if num_mamba_blocks == 1:
                indices = (num_prefix_layers - 1,)
            else:
                indices = tuple(
                    round(block_idx * (num_prefix_layers - 1) / (num_mamba_blocks - 1))
                    for block_idx in range(num_mamba_blocks)
                )
            logging.warning(
                "mamba_layerwise_context_indices not set; mapping %d Mamba blocks to VLM layers %s",
                num_mamba_blocks,
                indices,
            )
        else:
            raise ValueError(
                "Layerwise Mamba context needs no more Mamba blocks than VLM layers unless explicit "
                f"indices are provided: mamba_blocks={num_mamba_blocks}, vlm_layers={num_prefix_layers}"
            )

        if len(indices) != num_mamba_blocks:
            raise ValueError(
                "mamba_layerwise_context_indices length must match Mamba depth: "
                f"{len(indices)} vs {num_mamba_blocks}"
            )
        if len(set(indices)) != len(indices):
            raise ValueError(f"mamba_layerwise_context_indices must be unique: {indices}")
        if any(layer_idx < 0 or layer_idx >= num_prefix_layers for layer_idx in indices):
            raise ValueError(
                f"mamba_layerwise_context_indices must be within [0, {num_prefix_layers - 1}]: {indices}"
            )
        if tuple(sorted(indices)) != indices:
            raise ValueError(f"mamba_layerwise_context_indices must be sorted ascending: {indices}")

        self._layerwise_context_indices = indices
        logging.info("Using layerwise PI-Mamba context at VLM layers: %s", indices)
        return indices

    def _encode_prefix_layerwise_contexts(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        language_token_count: int,
        state_history=None,
        action_history=None,
        history_mask=None,
        detach_prefix_context: bool = False,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_backbone.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        language_model = self.paligemma_backbone.paligemma.language_model
        hidden_states = prefix_embs
        if language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype=torch.bfloat16)

        context_indices = self._resolve_layerwise_context_indices()
        context_index_set = set(context_indices)
        contexts: list[tuple[torch.Tensor, torch.Tensor]] = []
        use_gradient_checkpointing = (
            getattr(language_model, "gradient_checkpointing", False)
            and self.training
            and torch.is_grad_enabled()
            and not detach_prefix_context
        )

        for layer_idx in range(language_model.config.num_hidden_layers):
            if layer_idx in context_index_set:
                context_source = hidden_states.detach() if detach_prefix_context else hidden_states
                contexts.append(
                    self._prepare_mamba_context(
                        context_source,
                        prefix_pad_masks,
                        language_token_count,
                        state_history=state_history,
                        action_history=action_history,
                        history_mask=history_mask,
                    )
                )

            if use_gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    self.paligemma_backbone._forward_prefix_layer,
                    layer_idx,
                    hidden_states,
                    prefix_att_2d_masks_4d,
                    prefix_position_ids,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            elif detach_prefix_context:
                with torch.no_grad():
                    hidden_states = self.paligemma_backbone._forward_prefix_layer(
                        layer_idx,
                        hidden_states,
                        prefix_att_2d_masks_4d,
                        prefix_position_ids,
                    )
                hidden_states = hidden_states.detach()
            else:
                hidden_states = self.paligemma_backbone._forward_prefix_layer(
                    layer_idx,
                    hidden_states,
                    prefix_att_2d_masks_4d,
                    prefix_position_ids,
                )

        if len(contexts) != len(context_indices):
            raise RuntimeError(f"Expected {len(context_indices)} layerwise contexts, got {len(contexts)}")
        return contexts

    def _run_mamba_action_expert(
        self,
        suffix_embs,
        context_tokens,
        context_key_padding_mask,
        adarms_cond,
        context_cache=None,
    ):
        def mamba_expert_func(suffix_embs, context_tokens, context_key_padding_mask, adarms_cond):
            return self.mamba_action_expert(
                suffix_embs,
                context_tokens,
                context_key_padding_mask,
                adarms_cond,
                context_cache=context_cache,
            )

        if self.gradient_checkpointing_enabled and self.training and context_cache is None:
            return torch.utils.checkpoint.checkpoint(
                mamba_expert_func,
                suffix_embs,
                context_tokens,
                context_key_padding_mask,
                adarms_cond,
                use_reentrant=True,
                preserve_rng_state=False,
            )
        return mamba_expert_func(suffix_embs, context_tokens, context_key_padding_mask, adarms_cond)

    def _run_mamba_action_block(
        self,
        block: MambaActionBlock,
        hidden,
        context_tokens,
        context_key_padding_mask,
        adarms_cond,
        context_cache=None,
    ):
        def block_func(hidden, context_tokens, context_key_padding_mask, adarms_cond):
            context_vector = None
            block_cache = None
            if context_cache is not None:
                context_vector, block_cache = context_cache
            elif self.mamba_action_expert.context_conditioning in {"gated", "gated_cross_attn"}:
                context_vector = self.mamba_action_expert.pool_context(context_tokens, context_key_padding_mask)
            return block(hidden, context_tokens, context_key_padding_mask, adarms_cond, context_vector, block_cache)

        if self.gradient_checkpointing_enabled and self.training and context_cache is None:
            return torch.utils.checkpoint.checkpoint(
                block_func,
                hidden,
                context_tokens,
                context_key_padding_mask,
                adarms_cond,
                use_reentrant=True,
                preserve_rng_state=False,
            )
        return block_func(hidden, context_tokens, context_key_padding_mask, adarms_cond)

    def _run_mamba_action_expert_layerwise(self, suffix_embs, contexts, adarms_cond, context_caches=None):
        hidden = suffix_embs
        if len(contexts) != len(self.mamba_action_expert.blocks):
            raise ValueError(
                f"Layerwise context count must match Mamba blocks: {len(contexts)} vs "
                f"{len(self.mamba_action_expert.blocks)}"
            )
        for index, (block, (context_tokens, context_key_padding_mask)) in enumerate(
            zip(
                self.mamba_action_expert.blocks,
                contexts,
                strict=True,
            )
        ):
            context_cache = context_caches[index] if context_caches is not None else None
            hidden = self._run_mamba_action_block(
                block,
                hidden,
                context_tokens,
                context_key_padding_mask,
                adarms_cond,
                context_cache=context_cache,
            )
        return self.mamba_action_expert.final_norm(hidden)

    def predict_velocity(self, observation, actions, noise=None, time=None) -> Tensor:
        """Predict the flow-matching velocity for a shared noise/time sample."""
        images, img_masks, lang_tokens, lang_masks, state, state_history, action_history, history_mask = (
            self._preprocess_observation(observation, train=True)
        )

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions

        detach_prefix_context = self.training and not self._prefix_backbone_requires_grad()
        if detach_prefix_context:
            with torch.no_grad():
                prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                    images, img_masks, lang_tokens, lang_masks
                )
            prefix_embs = prefix_embs.detach()
        else:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
        suffix_embs, _, _, adarms_cond = self.embed_suffix(state, x_t, time)
        if self.config.mamba_multilayer_context:
            context_tokens, context_key_padding_mask = self._encode_prefix_multilayer_context(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                language_token_count=lang_masks.shape[1],
                state_history=state_history,
                action_history=action_history,
                history_mask=history_mask,
                detach_prefix_context=detach_prefix_context,
            )
            suffix_out = self._run_mamba_action_expert(
                suffix_embs,
                context_tokens,
                context_key_padding_mask,
                adarms_cond,
            )
        elif self.config.mamba_layerwise_context:
            contexts = self._encode_prefix_layerwise_contexts(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                language_token_count=lang_masks.shape[1],
                state_history=state_history,
                action_history=action_history,
                history_mask=history_mask,
                detach_prefix_context=detach_prefix_context,
            )
            suffix_out = self._run_mamba_action_expert_layerwise(suffix_embs, contexts, adarms_cond)
        else:
            if detach_prefix_context:
                with torch.no_grad():
                    prefix_out = self._encode_prefix(prefix_embs, prefix_pad_masks, prefix_att_masks)
                prefix_out = prefix_out.detach()
            else:
                prefix_out = self._encode_prefix(prefix_embs, prefix_pad_masks, prefix_att_masks)
            context_tokens, context_key_padding_mask = self._prepare_mamba_context(
                prefix_out,
                prefix_pad_masks,
                language_token_count=lang_masks.shape[1],
                state_history=state_history,
                action_history=action_history,
                history_mask=history_mask,
            )
            del prefix_out

            suffix_out = self._run_mamba_action_expert(
                suffix_embs,
                context_tokens,
                context_key_padding_mask,
                adarms_cond,
            )
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        return self.action_out_proj(suffix_out.to(dtype=self.action_out_proj.weight.dtype))

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        u_t = noise - actions
        v_t = self.predict_velocity(observation, actions, noise=noise, time=time)
        return F.mse_loss(u_t.float(), v_t.float(), reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state, state_history, action_history, history_mask = (
            self._preprocess_observation(observation, train=False)
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        context_cache = None
        layerwise_context_caches = None
        if self.config.mamba_multilayer_context:
            context_tokens, context_key_padding_mask = self._encode_prefix_multilayer_context(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                language_token_count=lang_masks.shape[1],
                state_history=state_history,
                action_history=action_history,
                history_mask=history_mask,
            )
            contexts = None
            context_cache = self.mamba_action_expert.prepare_context_cache(context_tokens, context_key_padding_mask)
        elif self.config.mamba_layerwise_context:
            contexts = self._encode_prefix_layerwise_contexts(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                language_token_count=lang_masks.shape[1],
                state_history=state_history,
                action_history=action_history,
                history_mask=history_mask,
            )
            context_tokens = None
            context_key_padding_mask = None
            layerwise_context_caches = self.mamba_action_expert.prepare_layerwise_context_caches(contexts)
        else:
            prefix_out = self._encode_prefix(prefix_embs, prefix_pad_masks, prefix_att_masks)
            context_tokens, context_key_padding_mask = self._prepare_mamba_context(
                prefix_out,
                prefix_pad_masks,
                language_token_count=lang_masks.shape[1],
                state_history=state_history,
                action_history=action_history,
                history_mask=history_mask,
            )
            contexts = None
            context_cache = self.mamba_action_expert.prepare_context_cache(context_tokens, context_key_padding_mask)

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                context_tokens,
                context_key_padding_mask,
                x_t,
                expanded_time,
                context_cache=context_cache,
                layerwise_contexts=contexts,
                layerwise_context_caches=layerwise_context_caches,
            )

            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        context_tokens,
        context_key_padding_mask,
        x_t,
        timestep,
        *,
        context_cache=None,
        layerwise_contexts=None,
        layerwise_context_caches=None,
    ):
        suffix_embs, _, _, adarms_cond = self.embed_suffix(state, x_t, timestep)
        if layerwise_contexts is not None:
            suffix_out = self._run_mamba_action_expert_layerwise(
                suffix_embs,
                layerwise_contexts,
                adarms_cond,
                context_caches=layerwise_context_caches,
            )
        else:
            suffix_out = self._run_mamba_action_expert(
                suffix_embs,
                context_tokens,
                context_key_padding_mask,
                adarms_cond,
                context_cache=context_cache,
            )
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        return self.action_out_proj(suffix_out.to(dtype=self.action_out_proj.weight.dtype))
