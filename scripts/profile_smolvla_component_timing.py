#!/usr/bin/env python3
"""Component timing probe for SmolVLA LIBERO checkpoints.

This times the same model path used by the LeRobot eval job, but avoids a full
rollout. It builds one canonical LIBERO-shaped batch, measures prefix/context
setup separately from the flow-matching denoise/action-expert loop, and writes a
small JSON/Markdown report.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.configs import FeatureType
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def ms_stats(values: list[float]) -> dict[str, float]:
    values = list(values)
    if not values:
        return {}
    sorted_values = sorted(values)
    p95_index = min(len(sorted_values) - 1, max(0, math.ceil(0.95 * len(sorted_values)) - 1))
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": sorted_values[p95_index],
        "min": min(values),
        "max": max(values),
    }


def tokenize(policy: SmolVLAPolicy, text: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    tokenized = tokenizer(
        text if text.endswith("\n") else text + "\n",
        max_length=policy.config.tokenizer_max_length,
        truncation=True,
        padding=policy.config.pad_language_to,
        padding_side="right",
        return_tensors="pt",
    )
    return tokenized["input_ids"].to(device), tokenized["attention_mask"].to(device, dtype=torch.bool)


def build_batch(
    policy: SmolVLAPolicy,
    device: torch.device,
    task: str,
    image_keys: list[str] | None = None,
    state_dim: int | None = None,
) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    if image_keys is not None:
        unknown = set(image_keys) - set(policy.config.image_features)
        if unknown:
            raise ValueError(f"Unknown image features: {sorted(unknown)}")
    for key, feature in policy.config.input_features.items():
        if feature.type is FeatureType.VISUAL:
            if image_keys is not None and key not in image_keys:
                continue
            shape = tuple(feature.shape)
            if len(shape) != 3:
                raise ValueError(f"Expected visual feature {key} to have C,H,W shape, got {shape}")
            batch[key] = torch.full((1, *shape), 0.5, dtype=torch.float32, device=device)
    if state_dim is None:
        state_dim = policy.config.robot_state_feature.shape[0] if policy.config.robot_state_feature else 7
    batch[OBS_STATE] = torch.zeros((1, state_dim), dtype=torch.float32, device=device)
    tokens, mask = tokenize(policy, task, device)
    batch[OBS_LANGUAGE_TOKENS] = tokens
    batch[OBS_LANGUAGE_ATTENTION_MASK] = mask
    return batch


@torch.inference_mode()
def timed_sample(policy: SmolVLAPolicy, batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, float]:
    model = policy.model
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    actions_shape = (state.shape[0], policy.config.chunk_size, policy.config.max_action_dim)
    noise = model.sample_noise(actions_shape, device)

    sync(device)
    total_start = time.perf_counter()

    prefix_start = time.perf_counter()
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    context_tokens = None
    context_key_padding_mask = None
    context_cache = None
    past_key_values = None
    if model.mamba_action_expert is not None:
        prefix_out = model._encode_prefix_for_mamba(prefix_embs, prefix_pad_masks, prefix_att_masks)
        context_tokens, context_key_padding_mask = model.mamba_action_expert.prepare_context(
            prefix_out, prefix_pad_masks
        )
        context_cache = model.mamba_action_expert.prepare_context_cache(
            context_tokens, context_key_padding_mask
        )
    else:
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=policy.config.use_cache,
            fill_kv_cache=True,
        )
    sync(device)
    prefix_ms = (time.perf_counter() - prefix_start) * 1000.0

    denoise_start = time.perf_counter()
    x_t = noise
    dt = -1.0 / policy.config.num_steps
    for step in range(policy.config.num_steps):
        time_value = 1.0 + step * dt
        time_tensor = torch.tensor(time_value, dtype=torch.float32, device=device).expand(state.shape[0])
        v_t = model.denoise_step(
            x_t=x_t,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            timestep=time_tensor,
            context_tokens=context_tokens,
            context_key_padding_mask=context_key_padding_mask,
            context_cache=context_cache,
        )
        x_t = x_t + dt * v_t
    sync(device)
    denoise_ms = (time.perf_counter() - denoise_start) * 1000.0
    total_ms = (time.perf_counter() - total_start) * 1000.0
    return {
        "model_ms": total_ms,
        "prefix_context_ms": prefix_ms,
        "denoise_ms": denoise_ms,
        "denoise_ms_per_step": denoise_ms / policy.config.num_steps,
        "unaccounted_ms": total_ms - prefix_ms - denoise_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--table-out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--timed-runs", type=int, default=10)
    parser.add_argument("--task", default="put the bowl on the plate")
    parser.add_argument("--image-keys", nargs="+")
    parser.add_argument("--state-dim", type=int)
    parser.add_argument("--n-action-steps", type=int)
    args = parser.parse_args()

    if args.device != "cpu" and not torch.cuda.is_available():
        parser.error("CUDA is required for GPU-comparable timing")
    if args.warmup_runs < 0 or args.timed_runs < 1:
        parser.error("Use nonnegative warmups and at least one timed query")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    if args.n_action_steps is not None:
        if not 1 <= args.n_action_steps <= policy.config.chunk_size:
            raise ValueError("Executed actions must be between 1 and chunk_size")
        policy.config.n_action_steps = args.n_action_steps
    policy.config.device = str(device)
    policy.to(device)
    policy.eval()
    batch = build_batch(policy, device, args.task, args.image_keys, args.state_dim)

    for _ in range(args.warmup_runs):
        timed_sample(policy, batch, device)

    rows = [timed_sample(policy, batch, device) for _ in range(args.timed_runs)]
    model_ms = [row["model_ms"] for row in rows]
    prefix_ms = [row["prefix_context_ms"] for row in rows]
    denoise_ms = [row["denoise_ms"] for row in rows]
    denoise_step_ms = [row["denoise_ms_per_step"] for row in rows]
    prefix_median = statistics.median(prefix_ms)
    denoise_median = statistics.median(denoise_ms)
    denom = max(prefix_median + denoise_median, 1e-9)

    result = {
        "model_label": args.model_label,
        "backend": "smolvla",
        "warmup_runs": args.warmup_runs,
        "timed_runs": args.timed_runs,
        "num_denoise_steps": policy.config.num_steps,
        "chunk_size": policy.config.chunk_size,
        "executed_actions": policy.config.n_action_steps,
        "image_keys": [key for key in batch if key in policy.config.image_features],
        "component_timing": {
            "prefix_context_ms": ms_stats(prefix_ms),
            "denoise_ms": ms_stats(denoise_ms),
            "denoise_ms_per_step": ms_stats(denoise_step_ms),
            "prefix_percent": {"median": 100.0 * prefix_median / denom},
            "denoise_percent": {"median": 100.0 * denoise_median / denom},
        },
        "latency": {
            "model_infer_ms": ms_stats(model_ms),
        },
        "frequency": {
            "query_hz": 1000.0 / max(statistics.median(model_ms), 1e-9),
            "action_hz": 1000.0 * policy.config.n_action_steps / max(statistics.median(model_ms), 1e-9),
        },
        "raw_runs": rows,
    }

    metrics_out = Path(args.metrics_out)
    table_out = Path(args.table_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    table_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        f"# {args.model_label} Component Timing",
        "",
        "| Model | Backend | Model ms median | Query Hz | Action Hz | Prefix/context ms | Denoise ms | Denoise ms/step | Prefix % | Denoise % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {args.model_label} | SmolVLA | "
            f"{statistics.median(model_ms):.2f} | {result['frequency']['query_hz']:.2f} | {result['frequency']['action_hz']:.2f} | "
            f"{prefix_median:.2f} | {denoise_median:.2f} | {statistics.median(denoise_step_ms):.2f} | "
            f"{result['component_timing']['prefix_percent']['median']:.1f}% | {result['component_timing']['denoise_percent']['median']:.1f}% |"
        ),
        "",
        f"Metrics JSON: `{metrics_out.name}`",
    ]
    table_out.write_text("\n".join(lines) + "\n")
    print(table_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
