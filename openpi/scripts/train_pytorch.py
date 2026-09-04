"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import gc
import json
import logging
import os
import pathlib
import platform
import re
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models.pi0_mamba_config
import openpi.models_pytorch.pi0_mamba_pytorch
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def get_model_named_parameters(model):
    """Get named parameters from model, handling DDP wrapper."""
    return (
        model.module.named_parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.named_parameters()
    )


def get_trainable_parameters(model):
    return [param for param in get_model_parameters(model) if param.requires_grad]


def get_parameter_metrics(model):
    params = list(get_model_parameters(model))
    total_params = sum(param.numel() for param in params)
    trainable_params = sum(param.numel() for param in params if param.requires_grad)
    trainable_pct = trainable_params / total_params * 100 if total_params else 0.0
    return {
        "model_total_params": total_params,
        "model_trainable_params": trainable_params,
        "model_trainable_pct": trainable_pct,
        "model_total_params_b": total_params / 1e9,
        "model_trainable_params_m": trainable_params / 1e6,
    }


def log_parameter_metrics_to_wandb(parameter_metrics: dict[str, float], *, step: int = 0):
    payload = {
        "model/total_params": parameter_metrics["model_total_params"],
        "model/trainable_params": parameter_metrics["model_trainable_params"],
        "model/trainable_pct": parameter_metrics["model_trainable_pct"],
        "model/total_params_b": parameter_metrics["model_total_params_b"],
        "model/trainable_params_m": parameter_metrics["model_trainable_params_m"],
        "efficiency/model_total_params_b": parameter_metrics["model_total_params_b"],
        "efficiency/model_trainable_params_m": parameter_metrics["model_trainable_params_m"],
        "efficiency/model_trainable_pct": parameter_metrics["model_trainable_pct"],
    }
    wandb.log(payload, step=step)
    for key, value in payload.items():
        wandb.summary[key] = value


def create_ema_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Create an EMA copy for trainable floating-point parameters."""

    ema_state: dict[str, torch.Tensor] = {}
    for name, param in get_model_named_parameters(model):
        if param.requires_grad and param.is_floating_point():
            ema_state[name] = param.detach().clone()
    return ema_state


@torch.no_grad()
def update_ema_state(model: torch.nn.Module, ema_state: dict[str, torch.Tensor], decay: float) -> None:
    """Update EMA weights from the current trainable parameters."""

    one_minus_decay = 1.0 - decay
    for name, param in get_model_named_parameters(model):
        ema_param = ema_state.get(name)
        if ema_param is None:
            continue
        ema_param.mul_(decay).add_(param.detach(), alpha=one_minus_decay)


def get_ema_metrics(ema_state: dict[str, torch.Tensor] | None) -> dict[str, float | int]:
    if ema_state is None:
        return {
            "ema_tracked_tensors": 0,
            "ema_tracked_params": 0,
            "ema_tracked_params_m": 0.0,
            "ema_storage_gb": 0.0,
        }
    tracked_params = sum(tensor.numel() for tensor in ema_state.values())
    storage_bytes = sum(tensor.numel() * tensor.element_size() for tensor in ema_state.values())
    return {
        "ema_tracked_tensors": len(ema_state),
        "ema_tracked_params": tracked_params,
        "ema_tracked_params_m": tracked_params / 1e6,
        "ema_storage_gb": storage_bytes / 1024**3,
    }


@torch.no_grad()
def _copy_ema_to_model_for_save(
    model: torch.nn.Module,
    ema_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Swap EMA weights into the model and return CPU raw-weight backups."""

    backups: dict[str, torch.Tensor] = {}
    for name, param in get_model_named_parameters(model):
        ema_param = ema_state.get(name)
        if ema_param is None:
            continue
        backups[name] = param.detach().cpu().clone()
        param.copy_(ema_param.to(device=param.device, dtype=param.dtype))
    return backups


@torch.no_grad()
def _restore_model_after_ema_save(model: torch.nn.Module, backups: dict[str, torch.Tensor]) -> None:
    for name, param in get_model_named_parameters(model):
        backup = backups.get(name)
        if backup is None:
            continue
        param.copy_(backup.to(device=param.device, dtype=param.dtype))


def freeze_pytorch_vision_encoder(model: torch.nn.Module) -> tuple[int, int]:
    """Freeze SigLIP/vision tower parameters while leaving other trainable params unchanged."""

    frozen_params = 0
    frozen_tensors = 0
    for name, param in model.named_parameters():
        if "vision_tower" in name or "vision_model" in name:
            if param.requires_grad:
                frozen_params += param.numel()
                frozen_tensors += 1
            param.requires_grad = False
    return frozen_params, frozen_tensors


_MAMBA_ACTION_SUBSTRINGS = (
    "action_in_proj",
    "action_out_proj",
    "state_proj",
    "action_time_mlp",
    "time_mlp",
    "mamba_action_expert",
    "multilayer_context_layer_embed",
)
_PI0_ACTION_EXPERT_SUBSTRINGS = (
    "paligemma_with_expert.gemma_expert",
    "action_in_proj",
    "action_out_proj",
    "state_proj",
    "action_time_mlp",
    "time_mlp",
)
_ACTION_GROUP_SUBSTRINGS = _MAMBA_ACTION_SUBSTRINGS + _PI0_ACTION_EXPERT_SUBSTRINGS
_LORA_SUBSTRINGS = ("lora_",)
_VISION_SUBSTRINGS = ("vision_tower", "vision_model")
_VLM_LAYER_INDEX_RE = re.compile(
    r"(?:paligemma_backbone\.paligemma(?:\.model)?\.language_model\.layers|"
    r"paligemma_with_expert\.paligemma(?:\.model)?\.language_model\.layers)\.(\d+)\."
)
_FINAL_INITIALIZER_ALLOWLIST = (
    "pi05_base_pytorch",
    "pi05_base",
    "prunevla_openpi",
    "prunevla_siglip_pruned",
    "siglip_pruned",
    "openpi-assets/checkpoints/pi05_base",
)


def _name_matches(name: str, substrings: tuple[str, ...]) -> bool:
    return any(substring in name for substring in substrings)


def _parse_int_csv(raw_value: str) -> frozenset[int]:
    if not raw_value.strip():
        return frozenset()
    values = []
    for raw_item in raw_value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        values.append(int(item))
    return frozenset(values)


def _load_int_list_from_json(path: str | pathlib.Path) -> frozenset[int]:
    json_path = pathlib.Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"VLM LoRA layer selection JSON does not exist: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"VLM LoRA layer selection JSON must contain an object: {json_path}")

    raw_layers = data.get("selected_vlm_lora_layers") or data.get("mamba_vlm_lora_layers")
    if raw_layers is None and isinstance(data.get("selected"), list):
        raw_layers = [
            item.get("layer_index")
            for item in data["selected"]
            if isinstance(item, dict) and item.get("layer_index") is not None
        ]
    if raw_layers is None:
        raise ValueError(
            "Could not find selected VLM LoRA layers in JSON. Expected key "
            "'selected_vlm_lora_layers', 'mamba_vlm_lora_layers', or selected[*].layer_index. "
            f"Path: {json_path}"
        )
    if not isinstance(raw_layers, list):
        raise ValueError(f"Selected VLM LoRA layers must be a list in {json_path}")
    return frozenset(int(layer) for layer in raw_layers)


def _resolve_selected_vlm_lora_layers(config: _config.TrainConfig) -> frozenset[int]:
    if config.mamba_vlm_lora_layers.strip():
        return _parse_int_csv(config.mamba_vlm_lora_layers)
    if config.mamba_vlm_lora_layers_json:
        return _load_int_list_from_json(config.mamba_vlm_lora_layers_json)
    return frozenset()


def _selected_vlm_lora_matches(name: str, selected_layers: frozenset[int]) -> bool:
    if not selected_layers:
        return True
    match = _VLM_LAYER_INDEX_RE.search(name)
    return match is not None and int(match.group(1)) in selected_layers


def _format_vlm_lora_layers(selected_layers: frozenset[int]) -> str:
    return "all" if not selected_layers else ",".join(str(layer) for layer in sorted(selected_layers))


def _mamba_executed_vlm_layer_count(model: torch.nn.Module) -> int | None:
    early_exit_layer = getattr(model.config, "mamba_vlm_early_exit_layer", None)
    if early_exit_layer is not None:
        return int(early_exit_layer)
    language_model = getattr(getattr(model.paligemma_backbone, "paligemma", None), "language_model", None)
    model_config = getattr(language_model, "config", None)
    num_hidden_layers = getattr(model_config, "num_hidden_layers", None)
    return int(num_hidden_layers) if num_hidden_layers is not None else None


def _mamba_full_vlm_param_matches(name: str, executed_layer_count: int | None) -> bool:
    if "paligemma_backbone.paligemma" not in name:
        return False
    if _name_matches(name, _VISION_SUBSTRINGS):
        return False
    layer_match = _VLM_LAYER_INDEX_RE.search(name)
    if layer_match is not None:
        return executed_layer_count is None or int(layer_match.group(1)) < executed_layer_count
    return "language_model.embed_tokens" in name or "language_model.norm" in name or "multi_modal_projector" in name


def _configure_mamba_train_scope(
    model: torch.nn.Module,
    scope: str,
    *,
    selected_vlm_lora_layers: frozenset[int] = frozenset(),
) -> bool:
    """Apply PI-Mamba staged trainability. Returns True if it handled the model."""
    model_to_configure = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    if not isinstance(model_to_configure, openpi.models_pytorch.pi0_mamba_pytorch.PI0MambaPytorch):
        return False

    if scope == "default":
        return False
    if scope not in {"action_only", "action_lora", "action_vision_lora", "fullvision_lora", "fullvision_fullvlm"}:
        raise ValueError(f"Unsupported PI-Mamba train scope: {scope}")

    for _, param in model_to_configure.named_parameters():
        param.requires_grad = False

    executed_vlm_layer_count = _mamba_executed_vlm_layer_count(model_to_configure)
    enabled_counts = {"action": 0, "vlm_lora": 0, "vision_lora": 0, "vision": 0, "vlm_full": 0}
    for name, param in model_to_configure.named_parameters():
        is_action = _name_matches(name, _MAMBA_ACTION_SUBSTRINGS)
        is_lora = _name_matches(name, _LORA_SUBSTRINGS)
        is_vision = _name_matches(name, _VISION_SUBSTRINGS)
        is_vision_lora = is_lora and is_vision
        is_vlm_lora = is_lora and not is_vision
        is_selected_vlm_lora = is_vlm_lora and _selected_vlm_lora_matches(name, selected_vlm_lora_layers)
        is_full_vlm = scope == "fullvision_fullvlm" and _mamba_full_vlm_param_matches(name, executed_vlm_layer_count)
        trainable = is_action
        trainable = trainable or (scope in {"action_lora", "fullvision_lora", "fullvision_fullvlm"} and is_selected_vlm_lora)
        trainable = trainable or (scope == "action_vision_lora" and is_vision_lora)
        trainable = trainable or (scope in {"fullvision_lora", "fullvision_fullvlm"} and is_vision)
        trainable = trainable or is_full_vlm
        param.requires_grad = trainable
        if trainable and is_action:
            enabled_counts["action"] += param.numel()
        elif trainable and is_vision_lora:
            enabled_counts["vision_lora"] += param.numel()
        elif trainable and is_vlm_lora:
            enabled_counts["vlm_lora"] += param.numel()
        elif trainable and is_vision:
            enabled_counts["vision"] += param.numel()
        elif trainable and is_full_vlm:
            enabled_counts["vlm_full"] += param.numel()

    logging.info(
        "Applied PI-Mamba train scope '%s': action/context=%.2fM, vlm_lora=%.2fM, vision_lora=%.2fM, vision=%.2fM, vlm_full=%.2fM%s%s",
        scope,
        enabled_counts["action"] / 1e6,
        enabled_counts["vlm_lora"] / 1e6,
        enabled_counts["vision_lora"] / 1e6,
        enabled_counts["vision"] / 1e6,
        enabled_counts["vlm_full"] / 1e6,
        f", selected_vlm_lora_layers={_format_vlm_lora_layers(selected_vlm_lora_layers)}"
        if scope in {"action_lora", "fullvision_lora", "fullvision_fullvlm"}
        else "",
        f", executed_vlm_layers={executed_vlm_layer_count}" if scope == "fullvision_fullvlm" else "",
    )
    return True


def _pi0_full_vlm_param_matches(name: str) -> bool:
    if "paligemma_with_expert.paligemma" not in name:
        return False
    if _name_matches(name, _VISION_SUBSTRINGS):
        return False
    return (
        "language_model" in name
        or "multi_modal_projector" in name
    )


def _configure_action_train_scope(model: torch.nn.Module, scope: str) -> bool:
    """Apply non-Mamba PI0/PI0.5 action-expert trainability scopes."""
    model_to_configure = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    if not isinstance(model_to_configure, openpi.models_pytorch.pi0_pytorch.PI0Pytorch):
        return False

    if scope == "default":
        return False
    if scope not in {"action_only", "fullvision_lora", "fullvision_fullvlm"}:
        raise ValueError(f"Unsupported action train scope: {scope}")

    for _, param in model_to_configure.named_parameters():
        param.requires_grad = False

    enabled_counts = {"action": 0, "vlm_lora": 0, "vision_lora": 0, "vision": 0, "vlm_full": 0}
    for name, param in model_to_configure.named_parameters():
        is_action = _name_matches(name, _PI0_ACTION_EXPERT_SUBSTRINGS)
        is_lora = _name_matches(name, _LORA_SUBSTRINGS)
        is_vision = _name_matches(name, _VISION_SUBSTRINGS)
        is_vision_lora = is_lora and is_vision
        is_vlm_lora = is_lora and not is_vision
        is_full_vlm = scope == "fullvision_fullvlm" and _pi0_full_vlm_param_matches(name)
        trainable = is_action
        trainable = trainable or (scope in {"fullvision_lora", "fullvision_fullvlm"} and (is_lora or is_vision))
        trainable = trainable or is_full_vlm
        param.requires_grad = trainable
        if trainable and is_action:
            enabled_counts["action"] += param.numel()
        elif trainable and is_vision_lora:
            enabled_counts["vision_lora"] += param.numel()
        elif trainable and is_vlm_lora:
            enabled_counts["vlm_lora"] += param.numel()
        elif trainable and is_vision:
            enabled_counts["vision"] += param.numel()
        elif trainable and is_full_vlm:
            enabled_counts["vlm_full"] += param.numel()

    logging.info(
        "Applied action train scope '%s': action_expert=%.2fM, vlm_lora=%.2fM, vision_lora=%.2fM, vision=%.2fM, vlm_full=%.2fM",
        scope,
        enabled_counts["action"] / 1e6,
        enabled_counts["vlm_lora"] / 1e6,
        enabled_counts["vision_lora"] / 1e6,
        enabled_counts["vision"] / 1e6,
        enabled_counts["vlm_full"] / 1e6,
    )
    return True


def _parameter_group_name(name: str) -> str:
    if _name_matches(name, _ACTION_GROUP_SUBSTRINGS):
        return "action"
    if _name_matches(name, _VISION_SUBSTRINGS):
        return "vision"
    if _name_matches(name, _LORA_SUBSTRINGS):
        return "lora"
    return "other"


def _build_optimizer_param_groups(
    model: torch.nn.Module,
    config: _config.TrainConfig,
    peak_lr: float,
    include_frozen_groups: set[str] | None = None,
) -> tuple[list[dict], list[torch.nn.Parameter]]:
    include_frozen_groups = include_frozen_groups or set()
    trainable_named_params = [
        (name, param)
        for name, param in get_model_named_parameters(model)
        if param.requires_grad or _parameter_group_name(name) in include_frozen_groups
    ]
    if not trainable_named_params:
        raise ValueError("No trainable parameters found. Check LoRA/freeze configuration.")

    use_custom_groups = any(
        value is not None for value in (config.mamba_action_lr, config.mamba_lora_lr, config.mamba_vision_lr)
    ) or config.mamba_train_scope != "default" or config.action_train_scope != "default" or bool(include_frozen_groups)

    if not use_custom_groups:
        params = [param for _, param in trainable_named_params]
        return [{"params": params, "lr": peak_lr, "initial_lr": peak_lr, "name": "default"}], params

    lr_by_group = {
        "action": config.mamba_action_lr if config.mamba_action_lr is not None else peak_lr,
        "lora": config.mamba_lora_lr if config.mamba_lora_lr is not None else peak_lr,
        "vision": config.mamba_vision_lr if config.mamba_vision_lr is not None else peak_lr,
        "other": peak_lr,
    }
    params_by_group: dict[str, list[torch.nn.Parameter]] = {"action": [], "lora": [], "vision": [], "other": []}
    counts_by_group = dict.fromkeys(params_by_group, 0)
    for name, param in trainable_named_params:
        group_name = _parameter_group_name(name)
        params_by_group[group_name].append(param)
        counts_by_group[group_name] += param.numel()

    groups = []
    flat_params = []
    for group_name in ("action", "lora", "vision", "other"):
        params = params_by_group[group_name]
        if not params:
            continue
        lr = lr_by_group[group_name]
        groups.append({"params": params, "lr": lr, "initial_lr": lr, "name": group_name})
        flat_params.extend(params)
        logging.info("Optimizer group '%s': %.2fM params, initial_lr=%.2e", group_name, counts_by_group[group_name] / 1e6, lr)

    return groups, flat_params


def _parse_optional_float(raw_value: str) -> float | None:
    raw_value = raw_value.strip()
    if not raw_value:
        return None
    return float(raw_value)


def _parse_stage_vlm_lora_layers(raw_value: str) -> frozenset[int] | None:
    raw_value = raw_value.strip()
    if not raw_value or raw_value == "global":
        return None
    if raw_value == "all":
        return frozenset()

    layers = []
    for raw_item in re.split(r"[|+]", raw_value):
        item = raw_item.strip()
        if not item:
            continue
        layers.append(int(item))
    if not layers:
        raise ValueError(f"Empty VLM LoRA layer override: {raw_value!r}")
    return frozenset(layers)


def _parse_mamba_stage_schedule(schedule: str) -> list[dict[str, float | int | str | frozenset[int] | None]]:
    if not schedule.strip():
        return []
    stages = []
    for raw_stage_item in schedule.split(","):
        raw_stage = raw_stage_item.strip()
        if not raw_stage:
            continue
        parts = raw_stage.split(":")
        if len(parts) not in {5, 6}:
            raise ValueError(
                "Each mamba stage must be start:scope:action_lr:lora_lr:vision_lr[:vlm_lora_layers], "
                f"got {raw_stage!r}"
            )
        start_step = int(parts[0])
        scope = parts[1]
        if scope not in {"action_only", "action_lora", "action_vision_lora", "fullvision_lora", "fullvision_fullvlm"}:
            raise ValueError(f"Unsupported mamba stage scope: {scope}")
        stages.append(
            {
                "start_step": start_step,
                "scope": scope,
                "action_lr": _parse_optional_float(parts[2]),
                "lora_lr": _parse_optional_float(parts[3]),
                "vision_lr": _parse_optional_float(parts[4]),
                "vlm_lora_layers": _parse_stage_vlm_lora_layers(parts[5]) if len(parts) == 6 else None,
            }
        )
    stages = sorted(stages, key=lambda stage: int(stage["start_step"]))
    if stages and stages[0]["start_step"] != 0:
        raise ValueError("mamba_stage_schedule must start at step 0")
    return stages


def _mamba_stage_for_step(
    stages: list[dict[str, float | int | str | frozenset[int] | None]],
    step: int,
) -> tuple[int, dict[str, float | int | str | frozenset[int] | None]]:
    if not stages:
        raise ValueError("No Mamba stages configured")
    active_index = 0
    for idx, stage in enumerate(stages):
        if int(stage["start_step"]) <= step:
            active_index = idx
    return active_index, stages[active_index]


def _mamba_stage_superset_groups(stages: list[dict[str, float | int | str | frozenset[int] | None]]) -> set[str]:
    groups = {"action"}
    if any(stage["scope"] in {"action_lora", "fullvision_lora", "fullvision_fullvlm"} for stage in stages):
        groups.add("lora")
    if any(stage["scope"] in {"action_vision_lora", "fullvision_lora", "fullvision_fullvlm"} for stage in stages):
        groups.add("vision")
    if any(stage["scope"] == "fullvision_fullvlm" for stage in stages):
        groups.add("other")
    return groups


def _mamba_stage_superset_scope(stages: list[dict[str, float | int | str | frozenset[int] | None]]) -> str:
    if any(stage["scope"] == "fullvision_fullvlm" for stage in stages):
        return "fullvision_fullvlm"
    if any(stage["scope"] == "fullvision_lora" for stage in stages):
        return "fullvision_lora"
    if any(stage["scope"] == "action_vision_lora" for stage in stages):
        if any(stage["scope"] == "action_lora" for stage in stages):
            return "fullvision_lora"
        return "action_vision_lora"
    if any(stage["scope"] == "action_lora" for stage in stages):
        return "action_lora"
    return "action_only"


def _apply_mamba_stage_lrs(
    optimizer: torch.optim.Optimizer,
    stage: dict[str, float | int | str | frozenset[int] | None],
    config: _config.TrainConfig,
    peak_lr: float,
):
    lr_by_group = {
        "action": stage["action_lr"] if stage["action_lr"] is not None else config.mamba_action_lr,
        "lora": stage["lora_lr"] if stage["lora_lr"] is not None else config.mamba_lora_lr,
        "vision": stage["vision_lr"] if stage["vision_lr"] is not None else config.mamba_vision_lr,
        "other": None,
    }
    for group in optimizer.param_groups:
        name = group.get("name", "other")
        group["initial_lr"] = lr_by_group.get(name) or peak_lr


def _mamba_stage_vlm_lora_layers(
    stage: dict[str, float | int | str | frozenset[int] | None],
    selected_vlm_lora_layers: frozenset[int],
) -> frozenset[int]:
    stage_layers = stage.get("vlm_lora_layers")
    if stage_layers is None:
        return selected_vlm_lora_layers
    if not isinstance(stage_layers, frozenset):
        raise TypeError(f"Unexpected VLM LoRA layer override: {stage_layers!r}")
    return stage_layers


def _set_optimizer_lrs(optimizer: torch.optim.Optimizer, scheduled_lr: float, peak_lr: float):
    scale = scheduled_lr / peak_lr if peak_lr > 0 else 1.0
    for group in optimizer.param_groups:
        group["lr"] = group.get("initial_lr", peak_lr) * scale


def _parse_extra_save_steps(config: _config.TrainConfig) -> set[int]:
    steps: set[int] = set()
    if config.save_initial_checkpoint:
        steps.add(0)
    for raw_step_item in str(config.extra_save_steps or "").split(","):
        raw_step = raw_step_item.strip()
        if not raw_step:
            continue
        steps.add(int(raw_step))
    return steps


def _validate_final_initializer(config: _config.TrainConfig):
    if config.experiment_mode != "final":
        return
    initializer = config.initializer_checkpoint or config.pytorch_weight_path
    if initializer is None:
        return
    normalized = str(initializer)
    if any(allowed in normalized for allowed in _FINAL_INITIALIZER_ALLOWLIST):
        return
    raise ValueError(
        "Final/report PI-Mamba runs must start from a base or architecture-converted base initializer. "
        f"Refusing initializer_checkpoint/pytorch_weight_path={normalized!r}. Use experiment_mode='search' for "
        "continuation runs from fine-tuned checkpoints."
    )


def _hardware_metadata(device: torch.device | None = None) -> dict[str, str | int | float | bool | None]:
    metadata: dict[str, str | int | float | bool | None] = {
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if torch.cuda.is_available():
        metadata["cuda_device_count"] = torch.cuda.device_count()
        device_index = None
        if device is not None and device.type == "cuda":
            device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        metadata.update(
            {
                "cuda_device_index": int(device_index),
                "gpu_name": properties.name,
                "gpu_total_memory_gb": properties.total_memory / 1024**3,
                "gpu_multiprocessor_count": properties.multi_processor_count,
            }
        )
    else:
        metadata["cuda_device_count"] = 0
    return metadata


def _experiment_metadata(config: _config.TrainConfig, device: torch.device | None = None) -> dict[str, str | int | float | bool | None]:
    initializer = config.initializer_checkpoint or config.pytorch_weight_path
    metadata: dict[str, str | int | float | bool | None] = {
        "experiment_mode": config.experiment_mode,
        "initializer_checkpoint": initializer,
        "warm_start_type": config.warm_start_type,
        "training_schedule": config.training_schedule,
        "total_optimizer_steps": config.total_optimizer_steps or config.num_train_steps,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
        "num_workers": config.num_workers,
        "save_interval": config.save_interval,
        "pytorch_training_precision": config.pytorch_training_precision,
        "pytorch_gradient_checkpointing_requested": config.pytorch_gradient_checkpointing,
        "skip_optimizer_checkpoint": os.environ.get("PI_MAMBA_SKIP_OPTIMIZER_CHECKPOINT", "0"),
        "resume_requested": config.resume,
        "model_only_resume_allowed": _env_flag("PI_MAMBA_ALLOW_MODEL_ONLY_RESUME"),
        "action_train_scope": config.action_train_scope,
        "mamba_train_scope": config.mamba_train_scope,
        "mamba_stage_schedule": config.mamba_stage_schedule,
        "mamba_vlm_lora_layers": config.mamba_vlm_lora_layers,
        "mamba_vlm_lora_layers_json": config.mamba_vlm_lora_layers_json,
        "distill_teacher_config": config.distill_teacher_config,
        "distill_teacher_checkpoint": config.distill_teacher_checkpoint,
        "distill_beta": config.distill_beta,
        "distill_teacher_lora_rank": config.distill_teacher_lora_rank,
        "distill_teacher_lora_alpha": config.distill_teacher_lora_alpha,
    }
    metadata.update(_hardware_metadata(device))
    return metadata


def log_run_metadata_to_wandb(metadata: dict[str, str | int | float | bool | None]):
    config_payload = {f"run_metadata/{key}": value for key, value in metadata.items()}
    wandb.config.update(config_payload, allow_val_change=True)
    for key, value in metadata.items():
        wandb.summary[f"run_metadata/{key}"] = value


def _build_pytorch_model(config: _config.TrainConfig):
    if isinstance(config.model, openpi.models.pi0_mamba_config.Pi0MambaConfig):
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
        return openpi.models_pytorch.pi0_mamba_pytorch.PI0MambaPytorch(model_cfg), model_cfg

    if isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    else:
        # Convert dataclass to Pi0Config if needed.
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )

    return openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg), model_cfg


def _load_pytorch_weights(model, pytorch_weight_path: str):
    weight_path = pathlib.Path(pytorch_weight_path)
    model_path = weight_path if weight_path.suffix == ".safetensors" else weight_path / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"PyTorch weight file does not exist: {model_path}")

    model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    if hasattr(model_to_load, "load_pretrained_weights"):
        model_to_load.load_pretrained_weights(str(model_path), strict=False)
    else:
        safetensors.torch.load_model(model_to_load, model_path)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def _nested_getattr(obj, path: str):
    current = obj
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _collect_architecture_metrics(model: torch.nn.Module) -> dict[str, int | str | None]:
    model_to_inspect = _unwrap_model(model)

    def _config_depth(path: str) -> int | None:
        module = _nested_getattr(model_to_inspect, path)
        config = getattr(module, "config", None)
        depth = getattr(config, "num_hidden_layers", None)
        return int(depth) if depth is not None else None

    model_config = getattr(model_to_inspect, "config", None)
    return {
        "model_class": type(model_to_inspect).__name__,
        "vlm_text_layers": _config_depth("paligemma_with_expert.paligemma.language_model")
        or _config_depth("paligemma_backbone.paligemma.language_model"),
        "vision_layers": _config_depth("paligemma_with_expert.paligemma.vision_tower.vision_model")
        or _config_depth("paligemma_backbone.paligemma.vision_tower.vision_model"),
        "action_expert_layers": _config_depth("paligemma_with_expert.gemma_expert")
        or _config_depth("mamba_action_expert"),
        "mamba_early_exit_layer": getattr(model_config, "mamba_vlm_early_exit_layer", None),
        "mamba_context_conditioning": getattr(model_config, "mamba_context_conditioning", None),
        "mamba_context_tokens": getattr(model_config, "mamba_context_tokens", None),
    }


def _build_distillation_teacher(config: _config.TrainConfig, device: torch.device) -> torch.nn.Module | None:
    if config.distill_beta <= 0:
        return None
    if not config.distill_teacher_config:
        raise ValueError("--distill-teacher-config is required when --distill-beta > 0")
    if not config.distill_teacher_checkpoint:
        raise ValueError("--distill-teacher-checkpoint is required when --distill-beta > 0")

    teacher_base_config = _config.get_config(config.distill_teacher_config)
    teacher_model_config = teacher_base_config.model
    if config.distill_teacher_lora_rank is not None:
        teacher_model_config = dataclasses.replace(
            teacher_model_config,
            lora_rank=config.distill_teacher_lora_rank,
            lora_alpha=(
                config.distill_teacher_lora_alpha
                if config.distill_teacher_lora_alpha is not None
                else float(config.distill_teacher_lora_rank)
            ),
        )

    teacher_config = dataclasses.replace(
        teacher_base_config,
        exp_name=f"{config.exp_name}_teacher",
        model=teacher_model_config,
        pytorch_training_precision=config.pytorch_training_precision,
        assets_base_dir=config.assets_base_dir,
        checkpoint_base_dir=config.checkpoint_base_dir,
        wandb_enabled=False,
    )
    teacher, _ = _build_pytorch_model(teacher_config)
    teacher = teacher.to(device)
    _load_pytorch_weights(teacher, config.distill_teacher_checkpoint)
    teacher.eval()
    teacher.requires_grad_(requires_grad=False)
    if hasattr(teacher, "gradient_checkpointing_disable"):
        teacher.gradient_checkpointing_disable()
    if not hasattr(teacher, "predict_velocity"):
        raise TypeError(f"Distillation teacher {type(teacher).__name__} does not implement predict_velocity().")
    logging.info(
        "Loaded frozen distillation teacher: config=%s checkpoint=%s beta=%.3f lora_rank=%s lora_alpha=%s",
        config.distill_teacher_config,
        config.distill_teacher_checkpoint,
        config.distill_beta,
        getattr(teacher_model_config, "lora_rank", None),
        getattr(teacher_model_config, "lora_alpha", None),
    )
    return teacher


def _compute_training_losses(
    model: torch.nn.Module,
    observation,
    actions: torch.Tensor,
    teacher: torch.nn.Module | None,
    distill_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if teacher is None:
        losses = model(observation, actions)
        return losses, {}

    student = _unwrap_model(model)
    if not hasattr(student, "predict_velocity"):
        raise TypeError(f"Distillation student {type(student).__name__} does not implement predict_velocity().")

    noise = student.sample_noise(actions.shape, actions.device)
    flow_time = student.sample_time(actions.shape[0], actions.device)
    student_v = student.predict_velocity(observation, actions, noise=noise, time=flow_time)
    fm_losses = F.mse_loss((noise - actions).float(), student_v.float(), reduction="none")

    with torch.no_grad():
        teacher_v = teacher.predict_velocity(observation, actions, noise=noise, time=flow_time)
        teacher_v = teacher_v.to(device=student_v.device)

    distill_losses = F.mse_loss(student_v.float(), teacher_v.float(), reduction="none")
    losses = fm_losses + distill_beta * distill_losses
    return losses, {
        "fm_loss": float(fm_losses.mean().detach().item()),
        "distill_loss": float(distill_losses.mean().detach().item()),
    }


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def _profile_synchronize(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _profile_start(device: torch.device, sync_cuda: bool) -> float:
    _profile_synchronize(device, sync_cuda)
    return time.perf_counter()


def _profile_elapsed(start_time: float, device: torch.device, sync_cuda: bool) -> float:
    _profile_synchronize(device, sync_cuda)
    return time.perf_counter() - start_time


def _profile_percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return float(ordered[index])


def _write_profile_timing_summary(
    records: list[dict[str, float | int]],
    output_path: str | pathlib.Path,
    *,
    metadata: dict[str, float | int | str | None],
):
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timing_keys = [key for key in records[0] if key.endswith("_s") or key in {"samples_per_second"}] if records else []
    summary = {}
    for key in timing_keys:
        values = [float(record[key]) for record in records if record.get(key) is not None]
        if not values:
            continue
        summary[key] = {
            "mean": float(sum(values) / len(values)),
            "p50": _profile_percentile(values, 0.50),
            "p90": _profile_percentile(values, 0.90),
            "min": float(min(values)),
            "max": float(max(values)),
            "total": float(sum(values)),
        }

    payload = {"metadata": metadata, "summary": summary, "records": records}
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = output_path.with_suffix(".csv")
    if records:
        columns = list(records[0])
        lines = [",".join(columns)]
        for record in records:
            lines.append(",".join(str(record.get(column, "")) for column in columns))
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logging.info("Profile timing summary written to %s", output_path)
    if records:
        logging.info("Profile timing records CSV written to %s", csv_path)
    for key in (
        "total_step_s",
        "data_wait_s",
        "device_transfer_s",
        "stage_lr_s",
        "forward_loss_s",
        "backward_s",
        "grad_clip_s",
        "optimizer_s",
        "grad_cleanup_s",
        "checkpoint_s",
    ):
        if key in summary:
            logging.info(
                "PROFILE %-18s mean=%.4fs p50=%.4fs p90=%.4fs",
                key,
                summary[key]["mean"],
                summary[key]["p50"],
                summary[key]["p90"],
            )


def _build_efficiency_metrics(
    *,
    global_step: int,
    config: _config.TrainConfig,
    run_start_time: float,
    training_start_time: float,
    first_step_time: float | None,
    parameter_metrics: dict[str, float | int],
    effective_batch_size: int,
    checkpoint_size_bytes: int,
    device: torch.device,
) -> dict[str, float | int | None]:
    now = time.time()
    elapsed_run_s = now - run_start_time
    elapsed_training_s = now - training_start_time
    elapsed_first_step_s = None if first_step_time is None else now - first_step_time
    steps_per_second = global_step / elapsed_training_s if elapsed_training_s > 0 else None
    if elapsed_first_step_s is not None and elapsed_first_step_s > 0:
        steps_per_second_after_first_step = max(0, global_step - 1) / elapsed_first_step_s
    else:
        steps_per_second_after_first_step = None

    metrics: dict[str, float | int | None] = {
        **parameter_metrics,
        "global_step": global_step,
        "num_train_steps": config.num_train_steps,
        "effective_batch_size": effective_batch_size,
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "checkpoint_size_gb": checkpoint_size_bytes / 1024**3,
        "elapsed_run_s": elapsed_run_s,
        "elapsed_run_hours": elapsed_run_s / 3600,
        "elapsed_training_s": elapsed_training_s,
        "elapsed_training_hours": elapsed_training_s / 3600,
        "elapsed_after_first_step_s": elapsed_first_step_s,
        "elapsed_after_first_step_hours": None if elapsed_first_step_s is None else elapsed_first_step_s / 3600,
        "steps_per_second": steps_per_second,
        "samples_per_second": None if steps_per_second is None else steps_per_second * effective_batch_size,
        "steps_per_second_after_first_step": steps_per_second_after_first_step,
        "samples_per_second_after_first_step": (
            None if steps_per_second_after_first_step is None else steps_per_second_after_first_step * effective_batch_size
        ),
    }

    if torch.cuda.is_available() and device.type == "cuda":
        metrics["cuda_peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
        metrics["cuda_peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1024**3

    return metrics


def save_checkpoint(
    model,
    optimizer,
    global_step,
    config,
    is_main,
    data_config,
    *,
    run_start_time: float,
    training_start_time: float,
    first_step_time: float | None,
    parameter_metrics: dict[str, float | int],
    effective_batch_size: int,
    device: torch.device,
    force: bool = False,
    ema_state: dict[str, torch.Tensor] | None = None,
    ema_decay: float | None = None,
):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    if _env_flag("PI_MAMBA_DISABLE_CHECKPOINTS"):
        return

    # Only save if it's time to save or if it's the final step
    extra_save_steps = _parse_extra_save_steps(config)
    should_save = (
        force
        or global_step in extra_save_steps
        or (global_step % config.save_interval == 0 and global_step > 0)
        or global_step == config.num_train_steps
    )
    if should_save:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors).
        # When EMA is enabled, model.safetensors is the eval-ready EMA checkpoint.
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        model_path = tmp_ckpt_dir / "model.safetensors"
        ema_backups = None
        if ema_state is not None:
            logging.info("Saving EMA model weights at step %d with decay %.6f", global_step, ema_decay or 0.0)
            ema_backups = _copy_ema_to_model_for_save(model, ema_state)
        try:
            safetensors.torch.save_model(model_to_save, model_path)
        finally:
            if ema_backups is not None:
                _restore_model_after_ema_save(model, ema_backups)
                del ema_backups
        checkpoint_size_bytes = model_path.stat().st_size

        # Optimizer state is useful for resume, but large full-vision search runs
        # may only need eval-usable model checkpoints under tight project quotas.
        skip_optimizer_checkpoint = os.environ.get("PI_MAMBA_SKIP_OPTIMIZER_CHECKPOINT", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if skip_optimizer_checkpoint:
            (tmp_ckpt_dir / "optimizer.pt.skipped").write_text(
                "Skipped because PI_MAMBA_SKIP_OPTIMIZER_CHECKPOINT=1\n", encoding="utf-8"
            )
            logging.info("Skipping optimizer state checkpoint by PI_MAMBA_SKIP_OPTIMIZER_CHECKPOINT")
        else:
            torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        efficiency_metrics = _build_efficiency_metrics(
            global_step=global_step,
            config=config,
            run_start_time=run_start_time,
            training_start_time=training_start_time,
            first_step_time=first_step_time,
            parameter_metrics=parameter_metrics,
            effective_batch_size=effective_batch_size,
            checkpoint_size_bytes=checkpoint_size_bytes,
            device=device,
        )

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        ema_metrics = get_ema_metrics(ema_state)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
            "efficiency_metrics": efficiency_metrics,
            "experiment_metadata": _experiment_metadata(config, device=device),
            "checkpoint_uses_ema_weights": ema_state is not None,
            "ema_decay": ema_decay,
            "ema_metrics": ema_metrics,
            **_experiment_metadata(config, device=device),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")
        if ema_state is not None:
            logging.info(
                "EMA checkpoint enabled: decay=%.6f, tracked=%.2fM params, storage=%.2fGB",
                ema_decay or 0.0,
                ema_metrics["ema_tracked_params_m"],
                ema_metrics["ema_storage_gb"],
            )
        logging.info(
            "Efficiency metrics at step %d: model=%.2fB params, trainable=%.2fM, "
            "training_hours=%.4f, checkpoint=%.2fGB, peak_vram=%.2fGB",
            global_step,
            efficiency_metrics["model_total_params_b"],
            efficiency_metrics["model_trainable_params_m"],
            efficiency_metrics["elapsed_training_hours"],
            efficiency_metrics["checkpoint_size_gb"],
            efficiency_metrics.get("cuda_peak_allocated_gb", 0.0) or 0.0,
        )

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb_payload = {
                "checkpoint_step": global_step,
                "efficiency/model_total_params_b": efficiency_metrics["model_total_params_b"],
                "efficiency/model_trainable_params_m": efficiency_metrics["model_trainable_params_m"],
                "efficiency/model_trainable_pct": efficiency_metrics["model_trainable_pct"],
                "efficiency/training_hours": efficiency_metrics["elapsed_training_hours"],
                "efficiency/checkpoint_size_gb": efficiency_metrics["checkpoint_size_gb"],
                "efficiency/steps_per_second": efficiency_metrics["steps_per_second"],
                "efficiency/samples_per_second": efficiency_metrics["samples_per_second"],
                "ema/enabled": int(ema_state is not None),
                "ema/decay": 0.0 if ema_decay is None else ema_decay,
                "ema/tracked_params_m": ema_metrics["ema_tracked_params_m"],
                "ema/storage_gb": ema_metrics["ema_storage_gb"],
            }
            if "cuda_peak_allocated_gb" in efficiency_metrics:
                wandb_payload["efficiency/cuda_peak_allocated_gb"] = efficiency_metrics["cuda_peak_allocated_gb"]
                wandb_payload["efficiency/cuda_peak_reserved_gb"] = efficiency_metrics["cuda_peak_reserved_gb"]
            wandb.log(wandb_payload, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
            optimizer.load_state_dict(optimizer_state_dict)
            del optimizer_state_dict
        elif _env_flag("PI_MAMBA_ALLOW_MODEL_ONLY_RESUME"):
            logging.warning(
                "Optimizer checkpoint is missing at %s; continuing with freshly initialized optimizer state "
                "because PI_MAMBA_ALLOW_MODEL_ONLY_RESUME=1",
                ckpt_dir,
            )
        else:
            raise FileNotFoundError(
                f"No optimizer checkpoint found at {ckpt_dir}. Set PI_MAMBA_ALLOW_MODEL_ONLY_RESUME=1 "
                "only when a model-only continuation is acceptable."
            )

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    _validate_final_initializer(config)
    mamba_stages = _parse_mamba_stage_schedule(config.mamba_stage_schedule)
    selected_vlm_lora_layers = _resolve_selected_vlm_lora_layers(config)
    if mamba_stages and config.mamba_train_scope != "default":
        raise ValueError("Use either mamba_stage_schedule or mamba_train_scope, not both.")
    run_start_time = time.time()
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
        run_metadata = _experiment_metadata(config, device=device)
        logging.info("Run metadata: %s", json.dumps(run_metadata, sort_keys=True, default=str))
        if config.wandb_enabled:
            log_run_metadata_to_wandb(run_metadata)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    if config.batch_size % world_size != 0:
        raise ValueError(f"batch_size must be divisible by world_size: {config.batch_size} vs {world_size}")
    if config.gradient_accumulation_steps < 1:
        raise ValueError(f"gradient_accumulation_steps must be >= 1, got {config.gradient_accumulation_steps}")
    per_device_batch_size = config.batch_size // world_size
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    logging.info(
        "Using batch size per GPU: "
        f"{per_device_batch_size} (micro-batch total across {world_size} GPUs: {config.batch_size}, "
        f"gradient_accumulation_steps={config.gradient_accumulation_steps}, "
        f"optimizer-step batch size={effective_batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    model, model_cfg = _build_pytorch_model(config)
    model = model.to(device)

    if config.pytorch_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    elif not config.pytorch_gradient_checkpointing:
        enable_gradient_checkpointing = False
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        logging.info("Gradient checkpointing disabled by config")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if mamba_stages:
        _configure_mamba_train_scope(
            model,
            _mamba_stage_superset_scope(mamba_stages),
            # DDP is constructed after this call. Keep every LoRA adapter that any
            # later stage might unfreeze in the DDP/optimizer parameter universe.
            selected_vlm_lora_layers=frozenset(),
        )
    elif _configure_mamba_train_scope(
        model, config.mamba_train_scope, selected_vlm_lora_layers=selected_vlm_lora_layers
    ):
        pass
    elif _configure_action_train_scope(model, config.action_train_scope):
        pass
    elif hasattr(model, "configure_trainable_parameters"):
        model.configure_trainable_parameters()

    if config.pytorch_freeze_vision_encoder:
        frozen_params, frozen_tensors = freeze_pytorch_vision_encoder(model)
        logging.info(
            "Frozen PyTorch vision encoder parameters: %.2fM across %d tensors",
            frozen_params / 1e6,
            frozen_tensors,
        )

    parameter_metrics = get_parameter_metrics(model)
    if is_main:
        logging.info(
            "Model parameters%s: total=%.2fB trainable=%.2fM trainable_pct=%.2f%%",
            " (staged superset)" if mamba_stages else "",
            parameter_metrics["model_total_params_b"],
            parameter_metrics["model_trainable_params_m"],
            parameter_metrics["model_trainable_pct"],
        )
        if config.wandb_enabled:
            log_parameter_metrics_to_wandb(parameter_metrics, step=0)

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")
        _load_pytorch_weights(model, config.pytorch_weight_path)
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    distillation_teacher = _build_distillation_teacher(config, device)

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters. PI-Mamba staged training may use separate LR groups.
    optimizer_param_groups, trainable_params = _build_optimizer_param_groups(
        model,
        config,
        peak_lr,
        include_frozen_groups=_mamba_stage_superset_groups(mamba_stages) if mamba_stages else None,
    )
    optim = torch.optim.AdamW(
        optimizer_param_groups,
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    ema_state: dict[str, torch.Tensor] | None = None
    if config.ema_decay is not None:
        if not 0.0 < config.ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {config.ema_decay}")
        ema_state = create_ema_state(model)
        ema_metrics = get_ema_metrics(ema_state)
        if is_main:
            logging.info(
                "Enabled PyTorch EMA: decay=%.6f, tracked=%.2fM params across %d tensors, storage=%.2fGB",
                config.ema_decay,
                ema_metrics["ema_tracked_params_m"],
                ema_metrics["ema_tracked_tensors"],
                ema_metrics["ema_storage_gb"],
            )
            if config.wandb_enabled:
                wandb.log(
                    {
                        "ema/enabled": 1,
                        "ema/decay": config.ema_decay,
                        "ema/tracked_params_m": ema_metrics["ema_tracked_params_m"],
                        "ema/storage_gb": ema_metrics["ema_storage_gb"],
                    },
                    step=global_step,
                )
                wandb.summary["ema/enabled"] = 1
                wandb.summary["ema/decay"] = config.ema_decay
                wandb.summary["ema/tracked_params_m"] = ema_metrics["ema_tracked_params_m"]
                wandb.summary["ema/storage_gb"] = ema_metrics["ema_storage_gb"]

    active_mamba_stage_index: int | None = None
    if mamba_stages:
        active_mamba_stage_index, active_mamba_stage = _mamba_stage_for_step(mamba_stages, global_step)
        _configure_mamba_train_scope(
            model,
            str(active_mamba_stage["scope"]),
            selected_vlm_lora_layers=_mamba_stage_vlm_lora_layers(active_mamba_stage, selected_vlm_lora_layers),
        )
        _apply_mamba_stage_lrs(optim, active_mamba_stage, config, peak_lr)
        parameter_metrics = get_parameter_metrics(model)
        if is_main:
            logging.info(
                "Active PI-Mamba stage at step %d: scope=%s, action_lr=%s, lora_lr=%s, vision_lr=%s, vlm_lora_layers=%s",
                global_step,
                active_mamba_stage["scope"],
                active_mamba_stage["action_lr"] or config.mamba_action_lr or peak_lr,
                active_mamba_stage["lora_lr"] or config.mamba_lora_lr or peak_lr,
                active_mamba_stage["vision_lr"] or config.mamba_vision_lr or peak_lr,
                _format_vlm_lora_layers(_mamba_stage_vlm_lora_layers(active_mamba_stage, selected_vlm_lora_layers))
                if str(active_mamba_stage["scope"]) in {"action_lora", "fullvision_lora"}
                else "n/a",
            )
            logging.info(
                "Active-stage parameters: total=%.2fB trainable=%.2fM trainable_pct=%.2f%%",
                parameter_metrics["model_total_params_b"],
                parameter_metrics["model_trainable_params_m"],
                parameter_metrics["model_trainable_pct"],
            )

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    training_start_time = time.time()
    first_step_time: float | None = None
    log_interval_start_time = time.time()
    infos = []  # Collect stats over log interval
    accum_losses: list[float] = []
    accum_loss_breakdowns: list[dict[str, float]] = []
    micro_step = 0
    optim.zero_grad(set_to_none=True)
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            "Training config: "
            f"batch_size={config.batch_size}, gradient_accumulation_steps={config.gradient_accumulation_steps}, "
            f"effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        if mamba_stages:
            logging.info("PI-Mamba stage LR mode: %s", config.mamba_stage_lr_mode)
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        if ema_state is not None:
            logging.info("EMA: enabled for PyTorch training with decay %.6f; checkpoints use EMA weights", config.ema_decay)
        else:
            logging.info("EMA: disabled for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")
        logging.info(
            "Experiment metadata: mode=%s, warm_start_type=%s, initializer=%s, schedule=%s, scope=%s, total_optimizer_steps=%s",
            config.experiment_mode,
            config.warm_start_type,
            config.initializer_checkpoint or config.pytorch_weight_path,
            config.training_schedule,
            config.mamba_train_scope,
            config.total_optimizer_steps or config.num_train_steps,
        )
        if distillation_teacher is not None:
            logging.info(
                "Distillation: teacher_config=%s teacher_checkpoint=%s beta=%.3f",
                config.distill_teacher_config,
                config.distill_teacher_checkpoint,
                config.distill_beta,
            )

    if global_step == 0 and 0 in _parse_extra_save_steps(config):
        save_checkpoint(
            model,
            optim,
            global_step,
            config,
            is_main,
            data_config,
            run_start_time=run_start_time,
            training_start_time=training_start_time,
            first_step_time=first_step_time,
            parameter_metrics=parameter_metrics,
            effective_batch_size=effective_batch_size,
            device=device,
            force=True,
            ema_state=ema_state,
            ema_decay=config.ema_decay,
        )

    # Training loop - iterate until we reach num_train_steps
    disable_tqdm = os.environ.get("PI_MAMBA_DISABLE_TQDM", "0").lower() in {"1", "true", "yes", "on"}
    pbar = (
        tqdm.tqdm(
            total=config.num_train_steps,
            initial=global_step,
            desc="Training",
            disable=disable_tqdm or not is_main,
        )
        if is_main and not disable_tqdm
        else None
    )
    if is_main and disable_tqdm:
        logging.info("PI-Mamba tqdm progress bar disabled by PI_MAMBA_DISABLE_TQDM")

    profile_enabled = is_main and _env_flag("PI_MAMBA_PROFILE_TIMING")
    profile_steps = int(os.environ.get("PI_MAMBA_PROFILE_STEPS", "80"))
    profile_warmup_steps = int(os.environ.get("PI_MAMBA_PROFILE_WARMUP_STEPS", "20"))
    profile_sync_cuda = _env_flag("PI_MAMBA_PROFILE_CUDA_SYNC", "1")
    profile_records: list[dict[str, float | int]] = []
    profile_summary_written = False
    profile_output_path = os.environ.get(
        "PI_MAMBA_PROFILE_OUTPUT",
        str(config.checkpoint_dir / "profile_timing_summary.json"),
    )
    profile_architecture_metrics = _collect_architecture_metrics(model) if profile_enabled else {}
    if profile_enabled:
        logging.info(
            "Profile timing enabled: warmup_steps=%d, records=%d, cuda_sync=%s, output=%s",
            profile_warmup_steps,
            profile_steps,
            profile_sync_cuda,
            profile_output_path,
        )
        logging.info("Profile architecture metrics: %s", profile_architecture_metrics)

    profile_data_wait_start = time.perf_counter()

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            if profile_enabled:
                profile_step_total_start = _profile_start(device, profile_sync_cuda)
                profile_data_wait_s = time.perf_counter() - profile_data_wait_start
                profile_transfer_start = _profile_start(device, profile_sync_cuda)
            else:
                profile_step_total_start = time.perf_counter()
                profile_data_wait_s = 0.0
                profile_transfer_start = profile_step_total_start

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901
            profile_device_transfer_s = (
                _profile_elapsed(profile_transfer_start, device, profile_sync_cuda) if profile_enabled else 0.0
            )

            profile_stage_lr_start = _profile_start(device, profile_sync_cuda) if profile_enabled else time.perf_counter()
            if mamba_stages:
                next_stage_index, next_stage = _mamba_stage_for_step(mamba_stages, global_step)
                if next_stage_index != active_mamba_stage_index:
                    _configure_mamba_train_scope(
                        model,
                        str(next_stage["scope"]),
                        selected_vlm_lora_layers=_mamba_stage_vlm_lora_layers(next_stage, selected_vlm_lora_layers),
                    )
                    _apply_mamba_stage_lrs(optim, next_stage, config, peak_lr)
                    parameter_metrics = get_parameter_metrics(model)
                    active_mamba_stage_index = next_stage_index
                    if is_main:
                        logging.info(
                            "Switched PI-Mamba stage at step %d: scope=%s, action_lr=%s, lora_lr=%s, vision_lr=%s, vlm_lora_layers=%s",
                            global_step,
                            next_stage["scope"],
                            next_stage["action_lr"] or config.mamba_action_lr or peak_lr,
                            next_stage["lora_lr"] or config.mamba_lora_lr or peak_lr,
                            next_stage["vision_lr"] or config.mamba_vision_lr or peak_lr,
                            _format_vlm_lora_layers(_mamba_stage_vlm_lora_layers(next_stage, selected_vlm_lora_layers))
                            if str(next_stage["scope"]) in {"action_lora", "fullvision_lora"}
                            else "n/a",
                        )
                        logging.info(
                            "Active-stage parameters: total=%.2fB trainable=%.2fM trainable_pct=%.2f%%",
                            parameter_metrics["model_total_params_b"],
                            parameter_metrics["model_trainable_params_m"],
                            parameter_metrics["model_trainable_pct"],
                        )

            # Update LR. Differential groups either follow the historical global cosine
            # scale or stay at the active stage's requested LR.
            if not mamba_stages or config.mamba_stage_lr_mode == "global_cosine":
                _set_optimizer_lrs(optim, lr_schedule(global_step), peak_lr)
            elif config.mamba_stage_lr_mode == "constant":
                _set_optimizer_lrs(optim, peak_lr, peak_lr)
            else:
                raise ValueError(f"Unsupported mamba_stage_lr_mode: {config.mamba_stage_lr_mode}")
            profile_stage_lr_s = (
                _profile_elapsed(profile_stage_lr_start, device, profile_sync_cuda) if profile_enabled else 0.0
            )

            # Forward pass
            profile_forward_start = _profile_start(device, profile_sync_cuda) if profile_enabled else time.perf_counter()
            losses, loss_breakdown = _compute_training_losses(
                model, observation, actions, distillation_teacher, config.distill_beta
            )
            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()
            profile_forward_loss_s = (
                _profile_elapsed(profile_forward_start, device, profile_sync_cuda) if profile_enabled else 0.0
            )

            # Backward pass
            profile_backward_start = _profile_start(device, profile_sync_cuda) if profile_enabled else time.perf_counter()
            (loss / config.gradient_accumulation_steps).backward()
            profile_backward_s = (
                _profile_elapsed(profile_backward_start, device, profile_sync_cuda) if profile_enabled else 0.0
            )
            accum_losses.append(loss.detach().item())
            if loss_breakdown:
                accum_loss_breakdowns.append(loss_breakdown)
            micro_step += 1

            if micro_step % config.gradient_accumulation_steps != 0:
                profile_data_wait_start = time.perf_counter()
                continue

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            profile_grad_clip_start = _profile_start(device, profile_sync_cuda) if profile_enabled else time.perf_counter()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.optimizer.clip_gradient_norm)
            profile_grad_clip_s = (
                _profile_elapsed(profile_grad_clip_start, device, profile_sync_cuda) if profile_enabled else 0.0
            )

            # Optimizer step
            profile_optimizer_start = _profile_start(device, profile_sync_cuda) if profile_enabled else time.perf_counter()
            optim.step()
            optim.zero_grad(set_to_none=True)
            profile_optimizer_s = (
                _profile_elapsed(profile_optimizer_start, device, profile_sync_cuda) if profile_enabled else 0.0
            )

            if ema_state is not None and config.ema_decay is not None:
                update_ema_state(model, ema_state, config.ema_decay)

            # Clear gradients more aggressively
            profile_grad_cleanup_start = _profile_start(device, profile_sync_cuda) if profile_enabled else time.perf_counter()
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None
            profile_grad_cleanup_s = (
                _profile_elapsed(profile_grad_cleanup_start, device, profile_sync_cuda) if profile_enabled else 0.0
            )

            step_loss = sum(accum_losses) / len(accum_losses)
            accum_losses = []
            step_loss_breakdown: dict[str, float] = {}
            if accum_loss_breakdowns:
                for key in accum_loss_breakdowns[0]:
                    step_loss_breakdown[key] = sum(item[key] for item in accum_loss_breakdowns) / len(
                        accum_loss_breakdowns
                    )
                accum_loss_breakdowns = []
            completed_step = global_step + 1
            if first_step_time is None:
                first_step_time = time.time()

            # Collect stats
            if is_main:
                infos.append(
                    {
                        "loss": step_loss,
                        "learning_rate": optim.param_groups[0]["lr"],
                        "learning_rates": {
                            pg.get("name", f"group_{idx}"): pg["lr"] for idx, pg in enumerate(optim.param_groups)
                        },
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        **step_loss_breakdown,
                    }
                )

            if is_main and (completed_step % config.log_interval == 0 or completed_step == 1):
                elapsed = time.time() - log_interval_start_time
                logged_steps = len(infos)

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / logged_steps
                avg_lr = sum(info["learning_rate"] for info in infos) / logged_steps

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                avg_extra_losses: dict[str, float] = {}
                for key in ("fm_loss", "distill_loss"):
                    vals = [info[key] for info in infos if key in info]
                    if vals:
                        avg_extra_losses[key] = sum(vals) / len(vals)
                logging.info(
                    f"step={completed_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={completed_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": completed_step,
                        "time_per_step": elapsed / logged_steps,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    log_payload.update(avg_extra_losses)
                    for idx, pg in enumerate(optim.param_groups):
                        log_payload[f"learning_rate/{pg.get('name', f'group_{idx}')}"] = pg["lr"]
                    wandb.log(log_payload, step=completed_step)

                log_interval_start_time = time.time()
                infos = []  # Reset stats collection

            global_step = completed_step
            # Save checkpoint using the new mechanism
            profile_checkpoint_start = _profile_start(device, profile_sync_cuda) if profile_enabled else time.perf_counter()
            save_checkpoint(
                model,
                optim,
                global_step,
                config,
                is_main,
                data_config,
                run_start_time=run_start_time,
                training_start_time=training_start_time,
                first_step_time=first_step_time,
                parameter_metrics=parameter_metrics,
                effective_batch_size=effective_batch_size,
                device=device,
                ema_state=ema_state,
                ema_decay=config.ema_decay,
            )
            profile_checkpoint_s = (
                _profile_elapsed(profile_checkpoint_start, device, profile_sync_cuda) if profile_enabled else 0.0
            )
            profile_total_step_s = (
                _profile_elapsed(profile_step_total_start, device, profile_sync_cuda) if profile_enabled else 0.0
            )
            if (
                profile_enabled
                and completed_step > profile_warmup_steps
                and len(profile_records) < profile_steps
            ):
                profile_records.append(
                    {
                        "record_index": len(profile_records),
                        "step": completed_step,
                        "data_wait_s": float(profile_data_wait_s),
                        "device_transfer_s": float(profile_device_transfer_s),
                        "stage_lr_s": float(profile_stage_lr_s),
                        "forward_loss_s": float(profile_forward_loss_s),
                        "backward_s": float(profile_backward_s),
                        "grad_clip_s": float(profile_grad_clip_s),
                        "optimizer_s": float(profile_optimizer_s),
                        "grad_cleanup_s": float(profile_grad_cleanup_s),
                        "checkpoint_s": float(profile_checkpoint_s),
                        "total_step_s": float(profile_total_step_s),
                        "samples_per_second": float(effective_batch_size / profile_total_step_s)
                        if profile_total_step_s > 0
                        else 0.0,
                    }
                )
                if len(profile_records) == profile_steps and not profile_summary_written:
                    _write_profile_timing_summary(
                        profile_records,
                        profile_output_path,
                        metadata={
                            "config_name": config.name,
                            "exp_name": config.exp_name,
                            "host": platform.node(),
                            "batch_size": config.batch_size,
                            "gradient_accumulation_steps": config.gradient_accumulation_steps,
                            "effective_batch_size": effective_batch_size,
                            "num_train_steps": config.num_train_steps,
                            "profile_warmup_steps": profile_warmup_steps,
                            "profile_steps": profile_steps,
                            "profile_cuda_sync": int(profile_sync_cuda),
                            **parameter_metrics,
                            **profile_architecture_metrics,
                        },
                    )
                    profile_summary_written = True

            profile_data_wait_start = time.perf_counter()

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{step_loss:.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    if profile_enabled and profile_records and not profile_summary_written:
        _write_profile_timing_summary(
            profile_records,
            profile_output_path,
            metadata={
                "config_name": config.name,
                "exp_name": config.exp_name,
                "host": platform.node(),
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_steps": config.num_train_steps,
                "profile_warmup_steps": profile_warmup_steps,
                "profile_steps": len(profile_records),
                "profile_cuda_sync": int(profile_sync_cuda),
                **parameter_metrics,
                **profile_architecture_metrics,
            },
        )

    if is_main:
        final_training_seconds = time.time() - training_start_time
        final_training_hours = final_training_seconds / 3600.0
        logging.info(
            "Final training wall time: %.0f seconds (%.4f hours)",
            final_training_seconds,
            final_training_hours,
        )
        if config.wandb_enabled:
            wandb.log(
                {
                    "efficiency/final_training_seconds": final_training_seconds,
                    "efficiency/final_training_hours": final_training_hours,
                },
                step=global_step,
            )
            wandb.summary["efficiency/final_training_seconds"] = final_training_seconds
            wandb.summary["efficiency/final_training_hours"] = final_training_hours

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
