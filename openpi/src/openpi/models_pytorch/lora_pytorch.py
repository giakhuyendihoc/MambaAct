import logging
import math
from collections.abc import Sequence
from pathlib import Path

import safetensors.torch
import torch
from torch import nn


logger = logging.getLogger("openpi")


_DEFAULT_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

_DEFAULT_VISION_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "o_proj",
    "fc1",
    "fc2",
)


class LoRALinear(nn.Module):
    """Drop-in LoRA wrapper for a torch Linear layer."""

    def __init__(self, base_layer: nn.Linear, *, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        lora_dtype = base_layer.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, base_layer.in_features, dtype=lora_dtype))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank, dtype=lora_dtype))
        with torch.no_grad():
            init_A = torch.empty_like(self.lora_A, dtype=torch.float32)
            nn.init.kaiming_uniform_(init_A, a=math.sqrt(5))
            self.lora_A.copy_(init_A.to(dtype=lora_dtype))

        for param in self.base_layer.parameters():
            param.requires_grad = False

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        lora_x = self.dropout(x).to(dtype=self.lora_A.dtype)
        lora = torch.matmul(torch.matmul(lora_x, self.lora_A.t()), self.lora_B.t()) * self.scaling
        return base + lora.to(dtype=base.dtype)


def _get_lora_config(
    model_config,
    kind: str,
    *,
    rank_override: int | None = None,
    alpha_override: float | None = None,
):
    config = model_config.lora_configs.get(kind)
    if config is None:
        return None
    rank = rank_override or config.rank
    alpha = alpha_override if alpha_override is not None else (float(rank) if rank_override is not None else config.alpha)
    return {"rank": rank, "alpha": alpha}


def apply_gemma_lora(
    root: nn.Module,
    model_config,
    *,
    lora_rank: int | None = None,
    lora_alpha: float | None = None,
    target_suffixes: Sequence[str] = _DEFAULT_TARGET_SUFFIXES,
    dropout: float = 0.0,
) -> int:
    """Apply LoRA to Gemma attention/MLP Linear layers in-place."""

    attn_config = _get_lora_config(model_config, "attn", rank_override=lora_rank, alpha_override=lora_alpha)
    ffn_config = _get_lora_config(model_config, "ffn", rank_override=lora_rank, alpha_override=lora_alpha)
    if attn_config is None and ffn_config is None:
        return 0

    replacements: list[tuple[str, nn.Linear, dict[str, float]]] = []
    for module_name, module in root.named_modules():
        if isinstance(module, LoRALinear):
            continue
        if not isinstance(module, nn.Linear):
            continue
        suffix = module_name.rsplit(".", maxsplit=1)[-1]
        if suffix not in target_suffixes:
            continue
        if suffix in {"q_proj", "k_proj", "v_proj", "o_proj"} and attn_config is not None:
            replacements.append((module_name, module, attn_config))
        elif suffix in {"gate_proj", "up_proj", "down_proj"} and ffn_config is not None:
            replacements.append((module_name, module, ffn_config))

    for module_name, module, lora_kwargs in replacements:
        parent_name, child_name = module_name.rsplit(".", maxsplit=1)
        parent = root.get_submodule(parent_name)
        setattr(parent, child_name, LoRALinear(module, dropout=dropout, **lora_kwargs))

    return len(replacements)


def apply_vision_lora(
    root: nn.Module,
    *,
    rank: int | None,
    alpha: float | None = None,
    target_suffixes: Sequence[str] = _DEFAULT_VISION_TARGET_SUFFIXES,
    dropout: float = 0.0,
) -> int:
    """Apply LoRA to SigLIP/vision tower Linear layers in-place."""

    if rank is None:
        return 0

    lora_kwargs = {"rank": rank, "alpha": float(alpha if alpha is not None else rank)}
    target_suffix_set = set(target_suffixes)
    replacements: list[tuple[str, nn.Linear]] = []
    for module_name, module in root.named_modules():
        if isinstance(module, LoRALinear):
            continue
        if not isinstance(module, nn.Linear):
            continue
        suffix = module_name.rsplit(".", maxsplit=1)[-1]
        if suffix in target_suffix_set:
            replacements.append((module_name, module))

    for module_name, module in replacements:
        parts = module_name.rsplit(".", maxsplit=1)
        if len(parts) == 2:
            parent_name, child_name = parts
            parent = root.get_submodule(parent_name)
        else:
            parent = root
            child_name = module_name
        setattr(parent, child_name, LoRALinear(module, dropout=dropout, **lora_kwargs))

    return len(replacements)


def set_requires_grad_by_name(
    module: nn.Module,
    *,
    trainable_substrings: Sequence[str],
    frozen_substrings: Sequence[str] = (),
) -> None:
    """Set requires_grad using name substrings; frozen substrings win first."""

    for name, param in module.named_parameters():
        if any(selector in name for selector in frozen_substrings):
            param.requires_grad = False
        else:
            param.requires_grad = any(selector in name for selector in trainable_substrings)


def log_trainable_parameters(module: nn.Module, *, prefix: str = "") -> None:
    total = 0
    trainable = 0
    for param in module.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    pct = 100.0 * trainable / max(total, 1)
    logger.info("%strainable parameters: %.2fM / %.2fM (%.2f%%)", prefix, trainable / 1e6, total / 1e6, pct)


def _candidate_keys(key: str, key_prefix_rewrites: Sequence[tuple[str, str]]) -> list[str]:
    candidates = [key]
    for old_prefix, new_prefix in key_prefix_rewrites:
        if key.startswith(old_prefix):
            candidates.append(f"{new_prefix}{key[len(old_prefix):]}")
    return candidates


def _match_key(candidate: str, model_state: dict[str, torch.Tensor]) -> str | None:
    if candidate in model_state:
        return candidate
    if candidate.endswith(".weight"):
        lora_candidate = f"{candidate[:-len('.weight')]}.base_layer.weight"
        if lora_candidate in model_state:
            return lora_candidate
    if candidate.endswith(".bias"):
        lora_candidate = f"{candidate[:-len('.bias')]}.base_layer.bias"
        if lora_candidate in model_state:
            return lora_candidate
    return None


def load_state_dict_with_lora_remap(
    module: nn.Module,
    weight_path: str | Path,
    *,
    key_prefix_rewrites: Sequence[tuple[str, str]] = (),
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load a safetensors state dict, accepting base-checkpoint keys for LoRA-wrapped Linear layers."""

    state = safetensors.torch.load_file(str(weight_path), device="cpu")
    model_state = module.state_dict()
    matched: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    shape_mismatch: list[str] = []

    for key, tensor in state.items():
        matched_key = None
        for candidate in _candidate_keys(key, key_prefix_rewrites):
            matched_key = _match_key(candidate, model_state)
            if matched_key is not None:
                break
        if matched_key is None:
            unexpected.append(key)
            continue
        if model_state[matched_key].shape != tensor.shape:
            shape_mismatch.append(
                f"{key} -> {matched_key}: {tuple(tensor.shape)} != {tuple(model_state[matched_key].shape)}"
            )
            continue
        matched[matched_key] = tensor

    incompatible = module.load_state_dict(matched, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected.extend(incompatible.unexpected_keys)

    if shape_mismatch:
        unexpected.extend(shape_mismatch)

    if strict and (missing or unexpected):
        raise RuntimeError(
            "Checkpoint did not match model exactly. "
            f"Missing keys: {missing[:20]} Unexpected keys: {unexpected[:20]}"
        )

    logger.info(
        "Loaded %d tensors from %s (%d missing, %d unexpected/skipped)",
        len(matched),
        weight_path,
        len(missing),
        len(unexpected),
    )
    if missing:
        logger.info("First missing keys: %s", missing[:20])
    if unexpected:
        logger.info("First unexpected/skipped keys: %s", unexpected[:20])

    return missing, unexpected
