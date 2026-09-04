import dataclasses
import datetime as dt
import json
import logging
import pathlib
import re
import time
from typing import Any

import numpy as np
import torch
import tyro

from openpi.policies import libero_policy
from openpi.policies import policy_config as _policy_config
from serve_policy_from_checkpoint import load_train_config_from_checkpoint


@dataclasses.dataclass
class Args:
    checkpoint_dir: pathlib.Path
    config_name: str | None = None
    default_prompt: str | None = None
    pytorch_device: str | None = None
    num_denoise_steps: int = 2
    action_horizon: int | None = None
    lora_rank: int | None = None
    lora_alpha: float | None = None
    warmup_runs: int = 2
    timed_runs: int = 10
    profile_flops: bool = False
    component_timing: bool = False
    train_log: pathlib.Path | None = None
    metrics_out_path: pathlib.Path | None = None
    table_out_path: pathlib.Path | None = None
    model_label: str | None = None


def main(args: Args) -> None:
    checkpoint_metadata = _load_checkpoint_metadata(args.checkpoint_dir)
    train_config = load_train_config_from_checkpoint(
        args.checkpoint_dir,
        config_name=args.config_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    if args.action_horizon is not None:
        if not hasattr(train_config.model, "action_horizon"):
            raise ValueError(f"Config model does not expose action_horizon: {type(train_config.model)}")
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, action_horizon=args.action_horizon),
        )
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
        sample_kwargs={"num_steps": args.num_denoise_steps},
        pytorch_device=args.pytorch_device,
    )

    example = libero_policy.make_libero_example()
    last_result = None
    device = _policy_device(policy)
    component_timer = _install_component_timers(policy, device) if args.component_timing else None
    if device is not None and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(args.warmup_runs):
            if component_timer is not None:
                component_timer.reset()
            last_result = policy.infer(example)
            _assert_finite_actions(last_result)
    if torch.cuda.is_available() and (args.pytorch_device is None or "cuda" in args.pytorch_device):
        torch.cuda.synchronize()

    model_infer_ms = []
    wall_ms = []
    component_records = []
    with torch.inference_mode():
        for _ in range(args.timed_runs):
            if component_timer is not None:
                component_timer.reset()
            start_time = time.monotonic()
            result = policy.infer(example)
            if torch.cuda.is_available() and (args.pytorch_device is None or "cuda" in args.pytorch_device):
                torch.cuda.synchronize()
            wall_ms.append((time.monotonic() - start_time) * 1000)
            _assert_finite_actions(result)
            model_infer_ms.append(float(result.get("policy_timing", {}).get("infer_ms", float("nan"))))
            if component_timer is not None:
                component_records.append(component_timer.snapshot())
            last_result = result

    if last_result is None:
        raise RuntimeError("No inference runs were executed")

    actions = np.asarray(last_result["actions"])
    action_horizon = actions.shape[0]
    model_infer_summary = _latency_summary(model_infer_ms)
    wall_summary = _latency_summary(wall_ms)
    model_metrics = _collect_model_metrics(policy, args.checkpoint_dir, checkpoint_metadata)
    training_metrics = _collect_training_metrics(checkpoint_metadata, args.train_log)
    component_metrics = _component_timing_summary(component_records, model_infer_ms, args.num_denoise_steps)
    approx_gflops = None
    if args.profile_flops:
        approx_gflops = _profile_policy_gflops(policy, example, device)

    metrics = {
        "model_label": args.model_label or train_config.name,
        "config": train_config.name,
        "checkpoint": str(args.checkpoint_dir),
        "global_step": checkpoint_metadata.get("global_step"),
        "warmup_runs": args.warmup_runs,
        "timed_runs": args.timed_runs,
        "num_denoise_steps": args.num_denoise_steps,
        "requested_action_horizon": args.action_horizon,
        "action_horizon": action_horizon,
        "actions_shape": list(actions.shape),
        "actions_mean": float(actions.mean()),
        "actions_std": float(actions.std()),
        "latency": {
            "model_infer_ms": model_infer_summary,
            "wall_ms": wall_summary,
            "model_infer_per_action_ms": _latency_summary([value / action_horizon for value in model_infer_ms]),
            "wall_per_action_ms": _latency_summary([value / action_horizon for value in wall_ms]),
        },
        "frequency": _frequency_metrics(model_infer_summary, wall_summary, action_horizon),
        "component_timing": component_metrics,
        "model": model_metrics,
        "training": training_metrics,
        "approx_gflops": approx_gflops,
    }
    metrics["table_row"] = _build_table_row(metrics)

    print(f"config={train_config.name}")
    print(f"checkpoint={args.checkpoint_dir}")
    print(f"warmup_runs={args.warmup_runs}")
    print(f"timed_runs={args.timed_runs}")
    print(f"num_denoise_steps={args.num_denoise_steps}")
    if args.action_horizon is not None:
        print(f"action_horizon_override={args.action_horizon}")
    print(f"actions_shape={actions.shape}")
    print(f"actions_mean={actions.mean():.6f}")
    print(f"actions_std={actions.std():.6f}")
    _print_latency("model_infer_ms", model_infer_ms)
    _print_latency("wall_ms", wall_ms)
    _print_latency("model_infer_per_action_ms", [value / action_horizon for value in model_infer_ms])
    _print_latency("wall_per_action_ms", [value / action_horizon for value in wall_ms])
    if component_metrics is not None:
        print(f"prefix_context_ms_median={component_metrics['prefix_context_ms']['median']:.2f}")
        print(f"denoise_ms_median={component_metrics['denoise_ms']['median']:.2f}")
        print(f"denoise_ms_per_step_median={component_metrics['denoise_ms_per_step']['median']:.2f}")
        print(f"prefix_percent_median={component_metrics['prefix_percent']['median']:.1f}")
        print(f"denoise_percent_median={component_metrics['denoise_percent']['median']:.1f}")
        print(f"unaccounted_ms_median={component_metrics['unaccounted_ms']['median']:.2f}")
    print(f"model_query_hz={metrics['frequency']['model_query_hz']:.4f}")
    print(f"model_action_generation_hz={metrics['frequency']['model_action_generation_hz']:.4f}")
    print(f"wall_query_hz={metrics['frequency']['wall_query_hz']:.4f}")
    print(f"wall_action_generation_hz={metrics['frequency']['wall_action_generation_hz']:.4f}")
    print(f"model_total_params_b={model_metrics.get('model_total_params_b', float('nan')):.4f}")
    print(f"model_trainable_params_m={model_metrics.get('model_trainable_params_m', float('nan')):.2f}")
    print(f"checkpoint_size_gb={model_metrics.get('checkpoint_size_gb', float('nan')):.2f}")
    if training_metrics.get("training_time_hours") is not None:
        print(f"training_time_hours={training_metrics['training_time_hours']:.4f}")
    if approx_gflops is not None:
        print(f"approx_gflops={approx_gflops:.2f}")
    print("table_row_markdown=" + _format_table_row(metrics["table_row"]))

    if args.metrics_out_path is not None:
        args.metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out_path.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"metrics_out_path={args.metrics_out_path}")
    if args.table_out_path is not None:
        args.table_out_path.parent.mkdir(parents=True, exist_ok=True)
        args.table_out_path.write_text(_format_table(metrics["table_row"]))
        print(f"table_out_path={args.table_out_path}")


class _ComponentTimer:
    def __init__(self, device: torch.device | None):
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.values = {
            "prefix_context_ms": 0.0,
            "denoise_ms": 0.0,
        }
        self.calls = {
            "prefix_context_calls": 0,
            "denoise_calls": 0,
        }

    def measure(self, key: str, fn, *args, **kwargs):
        if self.device is not None and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        start = time.monotonic()
        result = fn(*args, **kwargs)
        if self.device is not None and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self.values[key] += (time.monotonic() - start) * 1000
        self.calls[f"{key.removesuffix('_ms')}_calls"] = self.calls.get(f"{key.removesuffix('_ms')}_calls", 0) + 1
        return result

    def snapshot(self) -> dict[str, float | int]:
        return {**self.values, **self.calls}


def _install_component_timers(policy, device: torch.device | None) -> _ComponentTimer | None:
    model = getattr(policy, "_model", None)
    if model is None:
        return None

    timer = _ComponentTimer(device)

    def wrap(obj, name: str, key: str, classifier=None) -> None:
        if obj is None or not hasattr(obj, name):
            return
        original = getattr(obj, name)
        if getattr(original, "_pi_mamba_component_wrapped", False):
            return

        def wrapped(*args, **kwargs):
            measured_key = classifier(args, kwargs) if classifier is not None else key
            if measured_key is None:
                return original(*args, **kwargs)
            return timer.measure(measured_key, original, *args, **kwargs)

        wrapped._pi_mamba_component_wrapped = True  # type: ignore[attr-defined]
        setattr(obj, name, wrapped)

    def paligemma_forward_classifier(_args, kwargs):
        inputs = kwargs.get("inputs_embeds")
        if isinstance(inputs, (list, tuple)) and len(inputs) >= 2 and inputs[0] is not None and inputs[1] is None:
            return "prefix_context_ms"
        return None

    wrap(model, "embed_prefix", "prefix_context_ms")
    wrap(model, "denoise_step", "denoise_ms")

    # PI-Mamba prefix path: embed prefix, run VLM prefix/early exit, compress ctxD, and cache context.
    for name in (
        "_encode_prefix",
        "_encode_prefix_multilayer_context",
        "_encode_prefix_layerwise_contexts",
        "_prepare_mamba_context",
    ):
        wrap(model, name, "prefix_context_ms")
    expert = getattr(model, "mamba_action_expert", None)
    wrap(expert, "prepare_context_cache", "prefix_context_ms")
    wrap(expert, "prepare_layerwise_context_caches", "prefix_context_ms")

    # PI0.5 prefix path: after embed_prefix, the first PaliGemma forward builds prefix KV cache.
    paligemma_with_expert = getattr(model, "paligemma_with_expert", None)
    wrap(paligemma_with_expert, "forward", "prefix_context_ms", classifier=paligemma_forward_classifier)

    return timer


def _component_timing_summary(
    records: list[dict[str, float | int]],
    model_infer_ms: list[float],
    num_denoise_steps: int,
) -> dict[str, Any] | None:
    if not records:
        return None

    prefix_values = [float(record.get("prefix_context_ms", 0.0)) for record in records]
    denoise_values = [float(record.get("denoise_ms", 0.0)) for record in records]
    accounted_values = [p + d for p, d in zip(prefix_values, denoise_values, strict=True)]
    denoise_per_step = [d / num_denoise_steps for d in denoise_values]
    prefix_percent = [100.0 * p / a if a > 0 else float("nan") for p, a in zip(prefix_values, accounted_values, strict=True)]
    denoise_percent = [100.0 * d / a if a > 0 else float("nan") for d, a in zip(denoise_values, accounted_values, strict=True)]
    unaccounted = [m - a for m, a in zip(model_infer_ms, accounted_values, strict=True)]
    prefix_calls = [float(record.get("prefix_context_calls", 0)) for record in records]
    denoise_calls = [float(record.get("denoise_calls", 0)) for record in records]

    return {
        "prefix_context_ms": _latency_summary(prefix_values),
        "denoise_ms": _latency_summary(denoise_values),
        "denoise_ms_per_step": _latency_summary(denoise_per_step),
        "prefix_percent": _latency_summary(prefix_percent),
        "denoise_percent": _latency_summary(denoise_percent),
        "unaccounted_ms": _latency_summary(unaccounted),
        "prefix_context_calls": _latency_summary(prefix_calls),
        "denoise_calls": _latency_summary(denoise_calls),
    }

def _assert_finite_actions(result: dict) -> None:
    actions = np.asarray(result["actions"])
    if not np.isfinite(actions).all():
        raise RuntimeError("Policy returned non-finite actions")


def _print_latency(name: str, values: list[float]) -> None:
    summary = _latency_summary(values)
    print(f"{name}_mean={summary['mean']:.2f}")
    print(f"{name}_median={summary['median']:.2f}")
    print(f"{name}_p95={summary['p95']:.2f}")
    print(f"{name}_min={summary['min']:.2f}")
    print(f"{name}_max={summary['max']:.2f}")


def _latency_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _frequency_metrics(
    model_infer_summary: dict[str, float],
    wall_summary: dict[str, float],
    action_horizon: int,
) -> dict[str, float]:
    model_query_hz = _hz_from_ms(model_infer_summary["median"])
    wall_query_hz = _hz_from_ms(wall_summary["median"])
    return {
        "model_query_hz": model_query_hz,
        "wall_query_hz": wall_query_hz,
        "model_action_generation_hz": model_query_hz * action_horizon,
        "wall_action_generation_hz": wall_query_hz * action_horizon,
    }


def _hz_from_ms(milliseconds: float) -> float:
    return 1000.0 / milliseconds if milliseconds > 0 else float("nan")


def _load_checkpoint_metadata(checkpoint_dir: pathlib.Path) -> dict[str, Any]:
    metadata_path = checkpoint_dir / "metadata.pt"
    if not metadata_path.exists():
        return {}
    try:
        return torch.load(metadata_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(metadata_path, map_location="cpu")


def _policy_device(policy) -> torch.device | None:
    device_name = getattr(policy, "_pytorch_device", None)
    if device_name is not None:
        return torch.device(device_name)
    model = getattr(policy, "_model", None)
    if model is None:
        return None
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def _collect_model_metrics(policy, checkpoint_dir: pathlib.Path, metadata: dict[str, Any]) -> dict[str, float | int | None]:
    model = getattr(policy, "_model", None)
    metrics = dict(metadata.get("efficiency_metrics") or {})
    if model is not None:
        if hasattr(model, "configure_trainable_parameters"):
            model.configure_trainable_parameters()
        params = list(model.parameters())
        total_params = sum(param.numel() for param in params)
        trainable_params = sum(param.numel() for param in params if param.requires_grad)
        metrics.update(
            {
                "model_total_params": total_params,
                "model_trainable_params": trainable_params,
                "model_trainable_pct": trainable_params / total_params * 100 if total_params else 0.0,
                "model_total_params_b": total_params / 1e9,
                "model_trainable_params_m": trainable_params / 1e6,
            }
        )

    weight_path = checkpoint_dir / "model.safetensors"
    if weight_path.exists():
        checkpoint_size_bytes = weight_path.stat().st_size
        metrics["checkpoint_size_bytes"] = checkpoint_size_bytes
        metrics["checkpoint_size_gb"] = checkpoint_size_bytes / 1024**3

    device = _policy_device(policy)
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        metrics["cuda_peak_allocated_gb_inference"] = torch.cuda.max_memory_allocated(device) / 1024**3
        metrics["cuda_peak_reserved_gb_inference"] = torch.cuda.max_memory_reserved(device) / 1024**3

    return metrics


def _collect_training_metrics(metadata: dict[str, Any], train_log: pathlib.Path | None) -> dict[str, Any]:
    efficiency_metrics = metadata.get("efficiency_metrics") or {}
    training_time_hours = (
        efficiency_metrics.get("elapsed_training_hours")
        or efficiency_metrics.get("elapsed_after_first_step_hours")
        or efficiency_metrics.get("elapsed_run_hours")
    )
    metrics: dict[str, Any] = {
        "training_time_hours": training_time_hours,
        "source": "checkpoint_metadata" if training_time_hours is not None else None,
    }
    if train_log is not None:
        parsed = _parse_training_time_from_log(train_log, metadata.get("global_step"))
        metrics.update(parsed)
        if parsed.get("training_time_hours") is not None:
            metrics["training_time_hours"] = parsed["training_time_hours"]
            metrics["source"] = "train_log"
    return metrics


def _parse_training_time_from_log(train_log: pathlib.Path, global_step: int | None) -> dict[str, Any]:
    if not train_log.exists():
        raise FileNotFoundError(f"Training log does not exist: {train_log}")

    created = None
    train_config = None
    first_step = None
    final_step = None
    final_save = None
    step_pattern = re.compile(rf"step={global_step}\s") if global_step is not None else None

    for raw_line in train_log.read_text(errors="replace").replace("\r", "\n").splitlines():
        timestamp = _parse_log_time(raw_line)
        if timestamp is None:
            continue
        if "Created experiment checkpoint directory" in raw_line:
            created = timestamp
        if "Training config:" in raw_line:
            train_config = timestamp
        if re.search(r"step=1\s", raw_line):
            first_step = timestamp
        if step_pattern is not None and step_pattern.search(raw_line):
            final_step = timestamp
        if global_step is not None and f"Saved checkpoint at step {global_step}" in raw_line:
            final_save = timestamp

    end_time = final_save or final_step
    start_time = train_config or first_step or created
    training_time_hours = None if start_time is None or end_time is None else _elapsed_hours(start_time, end_time)
    return {
        "training_time_hours": training_time_hours,
        "created_to_final_save_hours": None if created is None or final_save is None else _elapsed_hours(created, final_save),
        "train_config_to_final_save_hours": (
            None if train_config is None or final_save is None else _elapsed_hours(train_config, final_save)
        ),
        "first_step_to_final_save_hours": None if first_step is None or final_save is None else _elapsed_hours(first_step, final_save),
        "first_step_to_final_step_hours": None if first_step is None or final_step is None else _elapsed_hours(first_step, final_step),
        "log_path": str(train_log),
    }


def _parse_log_time(line: str) -> dt.datetime | None:
    match = re.search(r"(\d{2}:\d{2}:\d{2}\.\d{3})", line)
    if match is None:
        return None
    return dt.datetime.strptime(match.group(1), "%H:%M:%S.%f")


def _elapsed_hours(start: dt.datetime, end: dt.datetime) -> float:
    while end < start:
        end += dt.timedelta(days=1)
    return (end - start).total_seconds() / 3600


def _profile_policy_gflops(policy, example: dict, device: torch.device | None) -> float:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize(device)

    with torch.inference_mode():
        with torch.profiler.profile(activities=activities, with_flops=True) as profiler:
            result = policy.infer(example)
            _assert_finite_actions(result)
            if device is not None and device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(device)

    total_flops = sum((getattr(event, "flops", 0) or 0) for event in profiler.key_averages())
    return float(total_flops) / 1e9


def _build_table_row(metrics: dict[str, Any]) -> dict[str, Any]:
    model_metrics = metrics["model"]
    training_metrics = metrics["training"]
    return {
        "Model": metrics["model_label"],
        "Model Size": f"{model_metrics.get('model_total_params_b', float('nan')):.2f}B",
        "Trainable Params": f"{model_metrics.get('model_trainable_params_m', float('nan')):.2f}M",
        "Training Time (hours)": _format_optional(training_metrics.get("training_time_hours"), precision=3),
        "GFLOPs": _format_optional(metrics.get("approx_gflops"), precision=2),
        "Inference Speed (ms)": f"{metrics['latency']['model_infer_ms']['median']:.2f}",
        "Inference Speed (Hz)": f"{metrics['frequency']['model_query_hz']:.3f}",
        "Action Gen. (Hz)": f"{metrics['frequency']['model_action_generation_hz']:.3f}",
        "Checkpoint Size": f"{model_metrics.get('checkpoint_size_gb', float('nan')):.2f}GB",
    }


def _format_optional(value: Any, *, precision: int) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}f}"


def _format_table_row(row: dict[str, Any]) -> str:
    return "| " + " | ".join(str(value) for value in row.values()) + " |"


def _format_table(row: dict[str, Any]) -> str:
    header = "| " + " | ".join(row.keys()) + " |\n"
    separator = "| " + " | ".join("---" for _ in row) + " |\n"
    return header + separator + _format_table_row(row) + "\n"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
