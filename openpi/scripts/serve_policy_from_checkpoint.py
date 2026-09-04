import dataclasses
import logging
import pathlib
import socket
from typing import Any

import torch
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


def _torch_load_metadata(metadata_path: pathlib.Path) -> dict[str, Any]:
    try:
        return torch.load(metadata_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(metadata_path, map_location="cpu")


def load_train_config_from_checkpoint(
    checkpoint_dir: pathlib.Path,
    *,
    config_name: str | None = None,
    lora_rank: int | None = None,
    lora_alpha: float | None = None,
) -> _config.TrainConfig:
    """Load a TrainConfig and restore checkpoint-time model overrides when available."""

    checkpoint_dir = checkpoint_dir.resolve()
    metadata_path = checkpoint_dir / "metadata.pt"
    metadata_config: dict[str, Any] | None = None

    if metadata_path.exists():
        metadata = _torch_load_metadata(metadata_path)
        metadata_config = metadata.get("config")
        if not isinstance(metadata_config, dict):
            metadata_config = None

    resolved_config_name = config_name
    if resolved_config_name is None and metadata_config is not None:
        resolved_config_name = metadata_config.get("name")
    if resolved_config_name is None:
        raise ValueError(
            "Could not infer config name. Pass --config-name, or use a checkpoint with metadata.pt."
        )

    train_config = _config.get_config(resolved_config_name)

    if metadata_config is not None:
        model_config = metadata_config.get("model")
        if isinstance(model_config, dict):
            for key, value in model_config.items():
                if hasattr(train_config.model, key):
                    object.__setattr__(train_config.model, key, value)

    if lora_rank is not None:
        object.__setattr__(train_config.model, "lora_rank", lora_rank)
    if lora_alpha is not None:
        object.__setattr__(train_config.model, "lora_alpha", lora_alpha)

    return train_config


@dataclasses.dataclass
class Args:
    checkpoint_dir: pathlib.Path
    config_name: str | None = None
    default_prompt: str | None = None
    port: int = 8000
    pytorch_device: str | None = None
    num_denoise_steps: int = 10
    lora_rank: int | None = None
    lora_alpha: float | None = None
    record: bool = False


def main(args: Args) -> None:
    train_config = load_train_config_from_checkpoint(
        args.checkpoint_dir,
        config_name=args.config_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    logging.info("Serving config: %s", train_config.name)
    logging.info("Serving checkpoint: %s", args.checkpoint_dir)

    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
        sample_kwargs={"num_steps": args.num_denoise_steps},
        pytorch_device=args.pytorch_device,
    )
    _log_policy_efficiency_context(policy, args.checkpoint_dir)
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s, port: %d)", hostname, local_ip, args.port)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


def _log_policy_efficiency_context(policy: _policy.Policy, checkpoint_dir: pathlib.Path) -> None:
    model = getattr(policy, "_model", None)
    if model is not None:
        total_params = sum(param.numel() for param in model.parameters())
        trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        logging.info(
            "Model parameters: total=%.2fM trainable=%.2fM trainable_pct=%.2f%%",
            total_params / 1e6,
            trainable_params / 1e6,
            trainable_params / total_params * 100 if total_params else 0.0,
        )

    weight_path = pathlib.Path(checkpoint_dir) / "model.safetensors"
    if weight_path.exists():
        logging.info("Checkpoint weight size: %.2fGB", weight_path.stat().st_size / 1024**3)

    if torch.cuda.is_available():
        device = torch.device(getattr(policy, "_pytorch_device", "cuda"))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            logging.info(
                "CUDA memory after model load: allocated=%.2fGB reserved=%.2fGB",
                torch.cuda.memory_allocated(device) / 1024**3,
                torch.cuda.memory_reserved(device) / 1024**3,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
