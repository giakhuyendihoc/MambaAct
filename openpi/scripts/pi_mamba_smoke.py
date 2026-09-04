import os
import pathlib
import tempfile

import safetensors.torch
import torch
import tyro

from openpi.models import model as _model
from openpi.training import config as _config

import train_pytorch


def _fake_observation(model_config, batch_size: int, device: torch.device) -> _model.Observation:
    images = {
        key: torch.zeros(batch_size, 3, 224, 224, dtype=torch.float32, device=device)
        for key in _model.IMAGE_KEYS
    }
    image_masks = {
        key: torch.ones(batch_size, dtype=torch.bool, device=device)
        for key in _model.IMAGE_KEYS
    }
    image_masks["right_wrist_0_rgb"] = torch.zeros(batch_size, dtype=torch.bool, device=device)

    return _model.Observation(
        images=images,
        image_masks=image_masks,
        state=torch.zeros(batch_size, model_config.action_dim, dtype=torch.float32, device=device),
        tokenized_prompt=torch.ones(batch_size, model_config.max_token_len, dtype=torch.long, device=device),
        tokenized_prompt_mask=torch.ones(batch_size, model_config.max_token_len, dtype=torch.bool, device=device),
    )


def main(
    config_name: str,
    *,
    batch_size: int = 1,
    train_steps: int = 1,
    device: str = "cuda",
    skip_weights: bool = False,
    checkpoint_roundtrip: bool = True,
):
    cfg = _config.get_config(config_name)
    torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model, model_cfg = train_pytorch._build_pytorch_model(cfg)  # noqa: SLF001
    model = model.to(torch_device)
    if hasattr(model, "configure_trainable_parameters"):
        model.configure_trainable_parameters()
    if not skip_weights and cfg.pytorch_weight_path is not None:
        train_pytorch._load_pytorch_weights(model, cfg.pytorch_weight_path)  # noqa: SLF001

    observation = _fake_observation(model_cfg, batch_size, torch_device)
    actions = torch.zeros(batch_size, model_cfg.action_horizon, model_cfg.action_dim, device=torch_device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=cfg.lr_schedule.peak_lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    model.train()
    for step in range(train_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(observation, actions).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        optimizer.step()
        print(f"step={step} loss={loss.item():.6f}")

    if checkpoint_roundtrip:
        tmp_root = pathlib.Path(os.environ.get("PI_MAMBA_RESOURCE_ROOT", tempfile.gettempdir())) / "tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_dir:
            path = pathlib.Path(tmp_dir) / "model.safetensors"
            safetensors.torch.save_model(model, path)
            safetensors.torch.load_model(model, path)
            print(f"checkpoint_roundtrip={path}")


if __name__ == "__main__":
    tyro.cli(main)
