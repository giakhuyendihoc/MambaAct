"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Mapping, Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.pi0_mamba_config as pi0_mamba_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.ur5_policy as ur5_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


def _pi_mamba_resource_path(*parts: str) -> str:
    root = os.environ.get("PI_MAMBA_RESOURCE_ROOT")
    if root is None:
        root = pathlib.Path(__file__).resolve().parents[4] / "resources"
    return str(pathlib.Path(root).joinpath(*parts))


def _prunevla_checkpoint_path(*parts: str) -> str:
    root = os.environ.get("PRUNEVLA_CHECKPOINT_ROOT")
    if root is not None:
        return str(pathlib.Path(root).joinpath(*parts))
    return _pi_mamba_resource_path("checkpoints", *parts)


def _pruned_pi05_config(
    *,
    name: str,
    method: str,
    k: int,
    pruned_layers: tuple[int, ...],
    finetune_mode: str = "lora",
) -> "TrainConfig":
    depth = 18 - k
    if finetune_mode not in {"lora", "full"}:
        raise ValueError(f"Unsupported pruned pi0.5 finetune mode: {finetune_mode}")
    is_lora = finetune_mode == "lora"
    return TrainConfig(
        name=name,
        project_name="prunevla",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora" if is_lora else "gemma_2b",
            action_expert_variant="gemma_300m_lora" if is_lora else "gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=32 if is_lora else None,
            lora_alpha=32.0 if is_lora else None,
            paligemma_depth=depth,
            action_expert_depth=depth,
            pruned_layers=pruned_layers,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=4 if is_lora else 1,
        gradient_accumulation_steps=1 if is_lora else 32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0 if is_lora else 10_000,
            peak_lr=5e-5,
            decay_steps=30_000 if is_lora else 1_000_000,
            decay_lr=5e-6 if is_lora else 5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "prunevla_openpi", f"pi05_{method}_k{k}"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    )


def _component_pruned_pi05_config(
    *,
    name: str,
    checkpoint_name: str,
    pruned_vlm_layers: tuple[int, ...],
    pruned_action_expert_layers: tuple[int, ...],
    finetune_mode: str = "lora",
) -> "TrainConfig":
    if finetune_mode not in {"lora", "full"}:
        raise ValueError(f"Unsupported component-pruned pi0.5 finetune mode: {finetune_mode}")
    is_lora = finetune_mode == "lora"
    base_depth = 18
    paligemma_depth = base_depth - len(pruned_vlm_layers)
    action_expert_depth = base_depth - len(pruned_action_expert_layers)
    return TrainConfig(
        name=name,
        project_name="prunevla",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora" if is_lora else "gemma_2b",
            action_expert_variant="gemma_300m_lora" if is_lora else "gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=32 if is_lora else None,
            lora_alpha=32.0 if is_lora else None,
            paligemma_depth=paligemma_depth,
            action_expert_depth=action_expert_depth,
            pruned_layers=tuple(sorted(set(pruned_vlm_layers) | set(pruned_action_expert_layers))),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=4 if is_lora else 1,
        gradient_accumulation_steps=1 if is_lora else 32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0 if is_lora else 10_000,
            peak_lr=5e-5,
            decay_steps=30_000 if is_lora else 1_000_000,
            decay_lr=5e-6 if is_lora else 5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "prunevla_openpi", checkpoint_name),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    )


def _ur5_camera_shifts_data_config() -> "LeRobotUR5DataConfig":
    return LeRobotUR5DataConfig(
        repo_id="arturaah/ur5_lab_test_tube_camera_shifts",
        base_config=DataConfig(prompt_from_task=True),
        use_delta_joint_actions=True,
        image_key="observation.images.context_left_rgb",
        wrist_image_key="observation.images.wrist_right_rgb",
    )


def _tube_pi05_base_data_config() -> "LeRobotUR5DataConfig":
    return LeRobotUR5DataConfig(
        repo_id="ur5_lab_test_tube_camera_shifts",
        assets=AssetsConfig(asset_id="ur5_lab_test_tube_camera_shifts"),
        base_config=DataConfig(prompt_from_task=True),
        use_delta_joint_actions=True,
        image_key="observation.images.context_left_rgb",
        wrist_image_key="observation.images.wrist_right_rgb",
    )


def _pour_pi05_base_data_config() -> "LeRobotUR5DataConfig":
    return LeRobotUR5DataConfig(
        repo_id="ur5_place_and_pour_nuts_camera_shifts",
        assets=AssetsConfig(asset_id="ur5_place_and_pour_nuts_camera_shifts"),
        base_config=DataConfig(prompt_from_task=True),
        use_delta_joint_actions=True,
        image_key="observation.images.context_left_rgb",
        wrist_image_key="observation.images.wrist_right_rgb",
    )


def _tube_force_depth_joint_data_config() -> "LeRobotUR5DataConfig":
    return LeRobotUR5DataConfig(
        repo_id="giakhuyendihoc/ur5_tube_full_tcp240_converted",
        base_config=DataConfig(prompt_from_task=True),
        use_delta_joint_actions=True,
        image_key="observation.images.front",
        wrist_image_key="observation.images.wrist",
    )


def _pour_full_tcp240_noforce_data_config() -> "LeRobotUR5DataConfig":
    return LeRobotUR5DataConfig(
        repo_id="giakhuyendihoc/ur5_pour_full_tcp240_noforce",
        base_config=DataConfig(prompt_from_task=True),
        use_delta_joint_actions=True,
        image_key="observation.images.front",
        wrist_image_key="observation.images.wrist",
    )


def _catheter_force_depth_eef_data_config() -> "LeRobotUR5EEFDataConfig":
    return LeRobotUR5EEFDataConfig(
        repo_id="giakhuyendihoc/catheter_force_depth_eef",
        base_config=DataConfig(prompt_from_task=True),
        tcp_z_offset_m=0.041,
        image_key="observation.images.front",
        wrist_image_key="observation.images.wrist",
    )


def _catheter_force_depth_joint_force_data_config() -> "LeRobotUR5JointForceDataConfig":
    return LeRobotUR5JointForceDataConfig(
        repo_id="giakhuyendihoc/catheter_force_depth_eef",
        base_config=DataConfig(prompt_from_task=True),
        use_delta_joint_actions=True,
        image_key="observation.images.front",
        wrist_image_key="observation.images.wrist",
    )


def _pruned_pi05_ur5_camera_shifts_config(
    *,
    name: str,
    method: str,
    k: int,
    pruned_layers: tuple[int, ...],
    finetune_mode: str = "lora",
) -> "TrainConfig":
    depth = 18 - k
    if finetune_mode not in {"lora", "full"}:
        raise ValueError(f"Unsupported pruned pi0.5 UR5 finetune mode: {finetune_mode}")
    is_lora = finetune_mode == "lora"
    return TrainConfig(
        name=name,
        project_name="prunevla",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora" if is_lora else "gemma_2b",
            action_expert_variant="gemma_300m_lora" if is_lora else "gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=32 if is_lora else None,
            lora_alpha=32.0 if is_lora else None,
            paligemma_depth=depth,
            action_expert_depth=depth,
            pruned_layers=pruned_layers,
        ),
        data=_ur5_camera_shifts_data_config(),
        batch_size=4 if is_lora else 1,
        gradient_accumulation_steps=1 if is_lora else 32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0 if is_lora else 10_000,
            peak_lr=5e-5,
            decay_steps=30_000 if is_lora else 1_000_000,
            decay_lr=5e-6 if is_lora else 5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "prunevla_openpi", f"pi05_{method}_k{k}"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    )


def _pruned_pi05_ur5_camera_shifts_eef_config(
    *,
    name: str,
    method: str,
    k: int,
    pruned_layers: tuple[int, ...],
    finetune_mode: str = "lora",
) -> "TrainConfig":
    depth = 18 - k
    if finetune_mode not in {"lora", "full"}:
        raise ValueError(f"Unsupported pruned pi0.5 UR5 EEF finetune mode: {finetune_mode}")
    is_lora = finetune_mode == "lora"
    return TrainConfig(
        name=name,
        project_name="prunevla",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora" if is_lora else "gemma_2b",
            action_expert_variant="gemma_300m_lora" if is_lora else "gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=32 if is_lora else None,
            lora_alpha=32.0 if is_lora else None,
            paligemma_depth=depth,
            action_expert_depth=depth,
            pruned_layers=pruned_layers,
        ),
        data=LeRobotUR5EEFDataConfig(
            repo_id="arturaah/ur5_lab_test_tube_camera_shifts",
            base_config=DataConfig(prompt_from_task=True),
            tcp_z_offset_m=0.041,
            image_key="observation.images.context_left_rgb",
            wrist_image_key="observation.images.wrist_right_rgb",
        ),
        batch_size=4 if is_lora else 1,
        gradient_accumulation_steps=1 if is_lora else 32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0 if is_lora else 10_000,
            peak_lr=5e-5,
            decay_steps=30_000 if is_lora else 1_000_000,
            decay_lr=5e-6 if is_lora else 5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "prunevla_openpi", f"pi05_{method}_k{k}"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    )


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)
    # Names of state keys that can be queried as a past window for history-aware models.
    state_sequence_keys: Sequence[str] = ()
    # Optional per-key timestamp offsets, expressed in frame offsets. This is
    # useful when a target sequence needs a different length from the default
    # action horizon, e.g. EEF pose needs current pose plus H future poses.
    delta_timestamp_offsets: Mapping[str, Sequence[int]] = dataclasses.field(default_factory=dict)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


def _uses_mamba_history(model_config: _model.BaseModelConfig) -> bool:
    return (getattr(model_config, "mamba_context_history_tokens", None) or 0) > 0


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        use_history = _uses_mamba_history(model_config)
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_structure = {
            "observation/image": "image",
            "observation/wrist_image": "wrist_image",
            "observation/state": "state",
            "actions": "actions",
            "prompt": "prompt",
        }
        if use_history:
            repack_structure.update(
                {
                    "observation/state_is_pad": "state_is_pad",
                    "actions_is_pad": "actions_is_pad",
                }
            )

        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(repack_structure)
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        input_transforms = []
        if use_history:
            input_transforms.append(
                _transforms.SplitLeRobotHistory(
                    history_length=model_config.mamba_history_length,
                )
            )
        input_transforms.append(libero_policy.LiberoInputs(model_type=model_config.model_type))
        data_transforms = _transforms.Group(
            inputs=input_transforms,
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            state_sequence_keys=("state",),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUR5DataConfig(DataConfigFactory):
    """Data config for the UR5 lab test tube LeRobot dataset.

    The dataset stores absolute joint/gripper targets in ``action``. We train the
    arm joints as deltas against the current state and keep the gripper absolute,
    matching the usual pi0/pi0.5 action convention for joint-position datasets.
    """

    use_delta_joint_actions: bool = True
    image_key: str = "observation.images.context_left_rgb"
    wrist_image_key: str = "observation.images.wrist_right_rgb"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        use_history = _uses_mamba_history(model_config)
        repack_structure = {
            "observation/image": self.image_key,
            "observation/wrist_image": self.wrist_image_key,
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }
        if use_history:
            repack_structure.update(
                {
                    "observation/state_is_pad": "observation.state_is_pad",
                    "actions_is_pad": "action_is_pad",
                }
            )

        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(repack_structure)
            ]
        )

        delta_action_mask = _transforms.make_bool_mask(6, -1)
        input_transforms = []
        if use_history:
            input_transforms.append(
                _transforms.SplitLeRobotHistory(
                    history_length=model_config.mamba_history_length,
                    delta_action_mask=delta_action_mask if self.use_delta_joint_actions else None,
                )
            )
        input_transforms.append(ur5_policy.UR5Inputs(model_type=model_config.model_type))
        data_transforms = _transforms.Group(
            inputs=input_transforms,
            outputs=[ur5_policy.UR5Outputs()],
        )
        if self.use_delta_joint_actions:
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            state_sequence_keys=("observation.state",),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUR5JointForceDataConfig(DataConfigFactory):
    """UR5 joint-space data config with force/torque history for Famba.

    State before padding is 13D: six joints, gripper, and six force/torque
    channels. Actions remain 7D joint/gripper targets; the first six action
    dimensions are converted to deltas against the current joint state.
    """

    use_delta_joint_actions: bool = True
    image_key: str = "observation.images.context_left_rgb"
    wrist_image_key: str = "observation.images.wrist_right_rgb"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        use_history = _uses_mamba_history(model_config)
        repack_structure = {
            "observation/image": self.image_key,
            "observation/wrist_image": self.wrist_image_key,
            "observation/state": "observation.state",
            "observation/force_torque": "observation.force_torque",
            "actions": "action",
            "prompt": "prompt",
        }
        if use_history:
            repack_structure.update(
                {
                    "observation/state_is_pad": "observation.state_is_pad",
                    "observation/force_torque_is_pad": "observation.force_torque_is_pad",
                }
            )

        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(repack_structure)])
        history_length = model_config.mamba_history_length if use_history else 0
        data_transforms = _transforms.Group(
            inputs=[
                ur5_policy.BuildUR5JointForceStateActions(history_length=history_length),
                ur5_policy.UR5Inputs(model_type=model_config.model_type),
            ],
            outputs=[ur5_policy.UR5Outputs()],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)
        state_sequence_keys = ("observation.state", "observation.force_torque") if use_history else ()

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            state_sequence_keys=state_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUR5EEFDataConfig(DataConfigFactory):
    """UR5 dataset config that trains Pi0.5 to output TCP-corrected EEF pose.

    State/action before padding are 8D:
      [x, y, z, qx, qy, qz, qw, gripper]

    The dataset EEF pose is corrected from the old 199 mm TCP to the new 240 mm
    TCP by shifting 0.041 m along the local tool +Z axis.
    """

    tcp_z_offset_m: float = 0.041
    image_key: str = "observation.images.context_left_rgb"
    wrist_image_key: str = "observation.images.wrist_right_rgb"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        use_history = _uses_mamba_history(model_config)
        if use_history:
            raise ValueError("LeRobotUR5EEFDataConfig does not currently support history-aware models.")

        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": self.image_key,
                        "observation/wrist_image": self.wrist_image_key,
                        "observation/state": "observation.state",
                        "eef_pose_sequence": "observation.ee_pose",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[
                ur5_policy.BuildUR5EEFStateActions(tcp_z_offset_m=self.tcp_z_offset_m),
                ur5_policy.UR5EEFInputs(model_type=model_config.model_type),
            ],
            outputs=[ur5_policy.UR5EEFOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action", "observation.ee_pose"),
            delta_timestamp_offsets={"observation.ee_pose": tuple(range(model_config.action_horizon + 1))},
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"
    # Enable activation checkpointing for PyTorch training. This reduces memory
    # usage at the cost of extra recomputation.
    pytorch_gradient_checkpointing: bool = True
    pytorch_freeze_vision_encoder: bool = False

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of micro-batches to accumulate before each optimizer step.
    gradient_accumulation_steps: int = 1
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of optimizer steps to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # Experiment provenance. Search runs may warm-start from fine-tuned checkpoints for strategy discovery.
    # Final/report runs must start from a base or architecture-converted base initializer.
    experiment_mode: Literal["search", "final", "ablation"] = "search"
    initializer_checkpoint: str | None = None
    warm_start_type: Literal["base", "finetuned_search_only", "ablation"] | None = None
    training_schedule: str | None = None
    total_optimizer_steps: int | None = None

    # Optional action-expert distillation. When enabled, the trainer builds a frozen
    # PyTorch teacher policy and adds beta * MSE(student_velocity, teacher_velocity)
    # using the same flow-matching noise/time sample as the student.
    distill_teacher_config: str | None = None
    distill_teacher_checkpoint: str | None = None
    distill_beta: float = 0.0
    distill_teacher_lora_rank: int | None = None
    distill_teacher_lora_alpha: float | None = None

    # PI-Mamba staged training controls. The default preserves each model's existing trainability behavior.
    # Generic PyTorch action-expert-only control for non-Mamba PI0/PI0.5 action-expert ablations.
    action_train_scope: Literal["default", "action_only", "fullvision_lora", "fullvision_fullvlm"] = "default"
    mamba_train_scope: Literal[
        "default",
        "action_only",
        "action_lora",
        "action_vision_lora",
        "fullvision_lora",
        "fullvision_fullvlm",
    ] = "default"
    mamba_action_lr: float | None = None
    mamba_lora_lr: float | None = None
    mamba_vision_lr: float | None = None
    # Optional comma-separated list of PaliGemma language layer indices whose LoRA
    # adapters should be trainable in action_lora/fullvision_lora scopes. Empty
    # keeps the previous behavior: all non-vision VLM LoRA adapters are trainable.
    mamba_vlm_lora_layers: str = ""
    # Optional path to a WW-guided selection JSON produced by
    # scripts/select_vlm_lora_layers_ww.py. Used when mamba_vlm_lora_layers is empty.
    mamba_vlm_lora_layers_json: str | None = None
    # Optional staged schedule:
    # "0:action_vision_lora:2e-4::5e-5,10000:fullvision_lora:5e-5:5e-5:2e-5:all"
    # Fields are start_step:scope:action_lr:lora_lr:vision_lr[:vlm_lora_layers].
    # Empty LR fields fall back to the config/group LR. Stage-local VLM LoRA layer ids use "|"; "all"
    # unfreezes every VLM LoRA layer for that stage.
    mamba_stage_schedule: str = ""
    # How staged optimizer group LRs are updated each step. The default preserves
    # historical behavior where stage LRs are scaled by the global cosine schedule.
    mamba_stage_lr_mode: Literal["global_cosine", "constant"] = "global_cosine"

    # Extra diagnostic checkpoint controls. extra_save_steps is comma-separated to keep CLI usage simple.
    save_initial_checkpoint: bool = False
    extra_save_steps: str = ""

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")



def _icra_real_robot_lr_schedule() -> _optimizer.CosineDecaySchedule:
    return _optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=2.5e-5,
        decay_steps=30_000,
        decay_lr=2.5e-6,
    )


def _icra_real_robot_pi05_config(
    *,
    name: str,
    data: DataConfigFactory,
    depth: int | None,
) -> "TrainConfig":
    pruned_layers = tuple(range(depth, 18)) if depth is not None and depth < 18 else ()
    return TrainConfig(
        name=name,
        project_name="pi-mamba",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=None,
            lora_alpha=None,
            train_vision_encoder=True,
            paligemma_depth=depth,
            action_expert_depth=depth,
            pruned_layers=pruned_layers,
        ),
        data=data,
        batch_size=16,
        gradient_accumulation_steps=1,
        num_workers=8,
        lr_schedule=_icra_real_robot_lr_schedule(),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        seed=42,
        action_train_scope="fullvision_fullvlm",
        mamba_action_lr=2.5e-5,
        mamba_vision_lr=2.5e-5,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=30_000,
        keep_period=30_000,
    )


def _icra_real_robot_pimamba_config(
    *,
    name: str,
    data: DataConfigFactory,
    early_exit_layer: int,
) -> "TrainConfig":
    return TrainConfig(
        name=name,
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=None,
            lora_alpha=None,
            train_vision_encoder=True,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=4,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=early_exit_layer,
            mamba_context_conditioning="gated",
        ),
        data=data,
        batch_size=16,
        gradient_accumulation_steps=1,
        num_workers=8,
        lr_schedule=_icra_real_robot_lr_schedule(),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        seed=42,
        mamba_train_scope="fullvision_fullvlm",
        mamba_action_lr=2.5e-5,
        mamba_vision_lr=2.5e-5,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=30_000,
        keep_period=30_000,
    )


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    _icra_real_robot_pi05_config(
        name="pi05_icra_ur5_tube_nolora_fullvlm_5090",
        data=_tube_force_depth_joint_data_config(),
        depth=None,
    ),
    _icra_real_robot_pi05_config(
        name="pi05_icra_ur5_pour_nolora_fullvlm_5090",
        data=_pour_full_tcp240_noforce_data_config(),
        depth=None,
    ),
    _icra_real_robot_pi05_config(
        name="pi05_l12_icra_ur5_tube_nolora_fullvlm_5090",
        data=_tube_force_depth_joint_data_config(),
        depth=12,
    ),
    _icra_real_robot_pi05_config(
        name="pi05_l12_icra_ur5_pour_nolora_fullvlm_5090",
        data=_pour_full_tcp240_noforce_data_config(),
        depth=12,
    ),
    _icra_real_robot_pimamba_config(
        name="pi05_mamba_tiny_ctxd18_depth4_nolora_fullvlm_ur5_tube_tcp240_5090",
        data=_tube_force_depth_joint_data_config(),
        early_exit_layer=18,
    ),
    _icra_real_robot_pimamba_config(
        name="pi05_mamba_tiny_ctxd18_depth4_nolora_fullvlm_ur5_pour_tcp240_noforce_5090",
        data=_pour_full_tcp240_noforce_data_config(),
        early_exit_layer=18,
    ),
    _icra_real_robot_pimamba_config(
        name="pi05_mamba_tiny_ctxd12_depth4_nolora_fullvlm_ur5_tube_tcp240_5090",
        data=_tube_force_depth_joint_data_config(),
        early_exit_layer=12,
    ),
    _icra_real_robot_pimamba_config(
        name="pi05_mamba_tiny_ctxd12_depth4_nolora_fullvlm_ur5_pour_tcp240_noforce_5090",
        data=_pour_full_tcp240_noforce_data_config(),
        early_exit_layer=12,
    ),
    TrainConfig(
        name="pi05_famba_tiny_ctxd18_depth4_forcehist8_nolora_fullvlm_catheter_joint_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=None,
            lora_alpha=None,
            train_vision_encoder=True,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=4,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=40,
            mamba_context_language_tokens=12,
            mamba_context_history_tokens=8,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=18,
            mamba_context_conditioning="gated",
            mamba_history_length=8,
            mamba_history_depth=1,
            mamba_history_state_dim=13,
            mamba_history_action_dim=0,
        ),
        data=_catheter_force_depth_joint_force_data_config(),
        batch_size=16,
        gradient_accumulation_steps=1,
        num_workers=8,
        lr_schedule=_icra_real_robot_lr_schedule(),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        seed=42,
        mamba_train_scope="fullvision_fullvlm",
        mamba_action_lr=2.5e-5,
        mamba_vision_lr=2.5e-5,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=30_000,
        keep_period=30_000,
    ),
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_libero_lora_5090",
        project_name="pi-mamba",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            pytorch_compile_mode=None,
            lora_rank=32,
            lora_alpha=32.0,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=1000,
        keep_period=5000,
    ),
    TrainConfig(
        name="pi05_libero_lora_vision_5090",
        project_name="pi-mamba",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            pytorch_compile_mode=None,
            lora_rank=32,
            lora_alpha=32.0,
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=4,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_libero_full_action_fullvision_lora_r16_5090",
        project_name="pi-mamba",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            train_vision_encoder=True,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_libero_full_5090",
        project_name="pi-mamba",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            pytorch_compile_mode=None,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=1,
        gradient_accumulation_steps=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_libero_l12_lora_fullvlm_5090",
        project_name="pi-mamba",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            train_vision_encoder=True,
            paligemma_depth=12,
            action_expert_depth=12,
            pruned_layers=(12, 13, 14, 15, 16, 17),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        num_workers=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        seed=42,
        action_train_scope="fullvision_fullvlm",
        mamba_action_lr=5e-5,
        mamba_lora_lr=5e-5,
        mamba_vision_lr=5e-5,
        mamba_vlm_lora_layers="0,1,2,3,4,5,6,7,8,9,10,11",
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_libero_l12_fullvlm_5090",
        project_name="pi-mamba",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=None,
            lora_alpha=None,
            train_vision_encoder=True,
            paligemma_depth=12,
            action_expert_depth=12,
            pruned_layers=(12, 13, 14, 15, 16, 17),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        num_workers=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        seed=42,
        action_train_scope="fullvision_fullvlm",
        mamba_action_lr=5e-5,
        mamba_vision_lr=5e-5,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_libero_l18_lora_fullvlm_5090",
        project_name="pi-mamba",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            train_vision_encoder=True,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        num_workers=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        seed=42,
        action_train_scope="fullvision_fullvlm",
        mamba_action_lr=5e-5,
        mamba_lora_lr=5e-5,
        mamba_vision_lr=5e-5,
        mamba_vlm_lora_layers="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17",
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=32,
            lora_alpha=32.0,
            mamba_d_state=128,
            mamba_expand=2,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=1000,
        keep_period=5000,
    ),
    TrainConfig(
        name="pi05_mamba_small_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=32,
            lora_alpha=32.0,
            mamba_width=768,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=12,
            mamba_mlp_dim=3072,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=4,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=1000,
        keep_period=5000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=4,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=1000,
        keep_period=5000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_early_ctxd12_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=12,
            mamba_context_conditioning="cross_attn",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_early_ctxd12_gated_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=12,
            mamba_context_conditioning="gated",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_early_ctxd12_gated_fullvlm_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=None,
            lora_alpha=None,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=12,
            mamba_context_conditioning="gated",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_pruned_ctxd12_gated_fullvlm_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            paligemma_depth=12,
            pruned_layers=(12, 13, 14, 15, 16, 17),
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=None,
            lora_alpha=None,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=4,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=12,
            mamba_context_conditioning="gated",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_early_ctxd12_gated_gate512_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_gate_mlp_dim=512,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=12,
            mamba_context_conditioning="gated",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_early_ctxd12_gated_xattn_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=12,
            mamba_context_conditioning="gated_cross_attn",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_early_ctxd9_gated_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=9,
            mamba_context_conditioning="gated",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_early_ctxd9_depth1_gated_fullvlm_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=None,
            lora_alpha=None,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=1,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=9,
            mamba_context_conditioning="gated",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=30_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_early_ctxd9_gated_gate512_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_gate_mlp_dim=512,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=9,
            mamba_context_conditioning="gated",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_early_ctxd15_gated_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=15,
            mamba_context_conditioning="gated",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_early_ctxd15_gated_fullvlm_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=None,
            lora_alpha=None,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=15,
            mamba_context_conditioning="gated",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_early_ctxd15_gated_gate512_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_gate_mlp_dim=512,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=15,
            mamba_context_conditioning="gated",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_ctxd18_gated_fullvlm_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=None,
            lora_alpha=None,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            # Layer 18 executes all PaliGemma language layers, so this is the
            # no-skip full-depth ctxD baseline while preserving early-exit logs.
            mamba_vlm_early_exit_layer=18,
            mamba_context_conditioning="gated",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_ctxd18_crossattn_fullvlm_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=None,
            lora_alpha=None,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            # L18/no-skip ctxD ablation without the pooled FiLM gate.
            # Context reaches Mamba only through the plain ctxD cross-attention bridge.
            mamba_vlm_early_exit_layer=18,
            mamba_context_conditioning="cross_attn",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=30_000,
        keep_period=30_000,
    ),
    TrainConfig(
        name="pi05_transformer_ctxd18_crossattn_fullvlm_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=None,
            lora_alpha=None,
            mamba_action_mixer="transformer",
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=18,
            mamba_context_conditioning="cross_attn",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        num_workers=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=30_000,
        keep_period=30_000,
    ),
    TrainConfig(
        name="pi05_transformer_ctxd18_gated_fullvlm_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=None,
            lora_alpha=None,
            mamba_action_mixer="transformer",
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=18,
            mamba_context_conditioning="gated",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        num_workers=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_ctxd18_gated_lora_only_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=18,
            mamba_context_conditioning="gated",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        num_workers=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_ctxd18_gated_lora_fullvlm_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=18,
            mamba_context_conditioning="gated",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        num_workers=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_ctxd18_gated_depth4_lora_fullvlm_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=4,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=18,
            mamba_context_conditioning="gated",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        num_workers=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_ctxd18_gated_lora_fullvlm_ur5_tube_tcp240_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=18,
            mamba_context_conditioning="gated",
        ),
        data=_tube_force_depth_joint_data_config(),
        batch_size=16,
        gradient_accumulation_steps=1,
        num_workers=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        seed=42,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=30_000,
        keep_period=30_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_ctxd18_gated_lora_fullvlm_ur5_pour_tcp240_noforce_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=18,
            mamba_context_conditioning="gated",
        ),
        data=_pour_full_tcp240_noforce_data_config(),
        batch_size=16,
        gradient_accumulation_steps=1,
        num_workers=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        seed=42,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=30_000,
        keep_period=30_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_ctxd12_gated_lora_fullvlm_ur5_tube_tcp240_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=12,
            mamba_context_conditioning="gated",
        ),
        data=_tube_force_depth_joint_data_config(),
        batch_size=16,
        gradient_accumulation_steps=1,
        num_workers=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        seed=42,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=30_000,
        keep_period=30_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_ctxd12_gated_lora_fullvlm_ur5_pour_tcp240_noforce_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=12,
            mamba_context_conditioning="gated",
        ),
        data=_pour_full_tcp240_noforce_data_config(),
        batch_size=16,
        gradient_accumulation_steps=1,
        num_workers=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        seed=42,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=30_000,
        keep_period=30_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_ctxd12_depth4_fullvlm_catheter_force_depth_eef_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=None,
            lora_alpha=None,
            train_vision_encoder=True,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=4,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=12,
            mamba_context_conditioning="gated",
        ),
        data=_catheter_force_depth_eef_data_config(),
        batch_size=16,
        gradient_accumulation_steps=1,
        num_workers=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        seed=42,
        mamba_train_scope="fullvision_fullvlm",
        mamba_action_lr=2.5e-5,
        mamba_vision_lr=2.5e-5,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=30_000,
        keep_period=30_000,
    ),
    TrainConfig(
        name="pi05_transformer_ctxd18_gated_lora_fullvlm_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            mamba_action_mixer="transformer",
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=18,
            mamba_context_conditioning="gated",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        num_workers=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_multilayer_ctxd_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_multilayer_context=True,
            mamba_multilayer_context_indices=(2, 5, 9, 13, 17),
            mamba_multilayer_context_add_layer_embed=True,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_his_mamba_tiny_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            train_vision_encoder=True,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=40,
            mamba_context_language_tokens=12,
            mamba_context_history_tokens=8,
            mamba_context_global_tokens=4,
            mamba_history_length=16,
            mamba_history_depth=1,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_paired_tiny_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            # One Mamba action block per VLM transformer layer. Block i sees
            # the prefix representation at VLM depth i, preserving the
            # baseline pi0.5 paired-depth assumption while replacing only the
            # action expert stream.
            mamba_depth=18,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_layerwise_context=True,
            mamba_layerwise_context_indices=tuple(range(18)),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=8,
        gradient_accumulation_steps=2,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_paired_micro_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            train_vision_encoder=True,
            mamba_width=256,
            mamba_d_state=32,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            # Paired-depth micro variant: one small Mamba action block per
            # VLM transformer layer, preserving the layer-index relationship
            # without making the action stack as large as paired-tiny.
            mamba_depth=18,
            mamba_mlp_dim=1024,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_layerwise_context=True,
            mamba_layerwise_context_indices=tuple(range(18)),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=8,
        gradient_accumulation_steps=2,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_ur5_real_robot_pytorch_baseline",
        project_name="openpi_cam_real_robot",
        exp_name="pi05_ur5_real_robot_pytorch_baseline_tandon_2gpu_b16",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            discrete_state_input=True,
            pytorch_compile_mode=None,
        ),
        data=_tube_pi05_base_data_config(),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-10,
            clip_gradient_norm=1.0,
        ),
        ema_decay=0.99,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=5_000,
        keep_period=5_000,
    ),
    TrainConfig(
        name="pi05_ur5_pour_pytorch_baseline",
        project_name="openpi_cam_real_robot",
        exp_name="pi05_ur5_pour_pytorch_baseline_tandon_2gpu_b16",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            discrete_state_input=False,
            pytorch_compile_mode=None,
        ),
        data=_pour_pi05_base_data_config(),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-10,
            clip_gradient_norm=1.0,
        ),
        ema_decay=0.99,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=5_000,
        keep_period=5_000,
    ),
    TrainConfig(
        name="pi05_ur5_pour_pytorch_cross_attn_fg_distill_gt_stage2_hard",
        project_name="openpi_cam_real_robot",
        exp_name="pi05_ur5_pour_cross_attn_fg_zeroinit_gt_hard_stage2_tandon_2gpu_b16",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            discrete_state_input=True,
            pytorch_compile_mode=None,
        ),
        data=_pour_pi05_base_data_config(),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-10,
            clip_gradient_norm=1.0,
        ),
        ema_decay=0.99,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=5_000,
        keep_period=5_000,
        policy_metadata={
            "architecture_note": (
                "Compatibility config for ColinSkywalker/pour_pi05_gt. The checkpoint metadata "
                "uses PRoPE ray encoding, foreground cross-view fusion, and an auxiliary point "
                "head. This OpenPI checkout must include the matching model modules to load those "
                "GT-specific weights; otherwise the generic PyTorch loader skips them."
            )
        },
    ),
    TrainConfig(
        name="pi05_mamba_tiny_ur5_lab_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            train_vision_encoder=True,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_global_tokens=4,
        ),
        data=LeRobotUR5DataConfig(
            repo_id="arturaah/ur5_lab_test_tube_camera_shifts",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=True,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_ur5_lab_v2_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            train_vision_encoder=True,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_global_tokens=4,
        ),
        data=LeRobotUR5DataConfig(
            repo_id="arturaah/lab_tubes_ur5_v2",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=True,
            image_key="observation.images.wrist_left_rgb",
            wrist_image_key="observation.images.wrist_right_rgb",
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_early_ctxd12_gated_ur5_tcp240_joint_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            train_vision_encoder=True,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_history_tokens=0,
            mamba_context_global_tokens=4,
            mamba_vlm_early_exit_layer=12,
            mamba_context_conditioning="gated",
        ),
        data=LeRobotUR5DataConfig(
            repo_id="giakhuyendihoc/ur5_tube_full_tcp240_converted",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            image_key="observation.images.front",
            wrist_image_key="observation.images.wrist",
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_full_action_fullvision_lora_ur5_tcp240_joint_5090",
        project_name="pi-mamba",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            train_vision_encoder=True,
        ),
        data=LeRobotUR5DataConfig(
            repo_id="giakhuyendihoc/ur5_tube_full_tcp240_converted",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            image_key="observation.images.front",
            wrist_image_key="observation.images.wrist",
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        seed=42,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_full_ur5_tcp240_joint_5090",
        project_name="pi-mamba",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            pytorch_compile_mode=None,
        ),
        data=LeRobotUR5DataConfig(
            repo_id="giakhuyendihoc/ur5_tube_full_tcp240_converted",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=True,
            image_key="observation.images.front",
            wrist_image_key="observation.images.wrist",
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        seed=42,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=30_000,
        keep_period=30_000,
    ),
    TrainConfig(
        name="pi05_full_ur5_pour_tcp240_noforce_5090",
        project_name="pi-mamba",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            pytorch_compile_mode=None,
        ),
        data=_pour_full_tcp240_noforce_data_config(),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        seed=42,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=30_000,
        keep_period=30_000,
    ),
    TrainConfig(
        name="pi05_lora_fullvlm_ur5_tube_tcp240_5090",
        project_name="pi-mamba",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            train_vision_encoder=True,
        ),
        data=_tube_force_depth_joint_data_config(),
        batch_size=16,
        gradient_accumulation_steps=1,
        num_workers=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        seed=42,
        action_train_scope="fullvision_fullvlm",
        mamba_action_lr=5e-5,
        mamba_lora_lr=5e-5,
        mamba_vision_lr=5e-5,
        mamba_vlm_lora_layers="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17",
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=30_000,
        keep_period=30_000,
    ),
    TrainConfig(
        name="pi05_lora_fullvlm_ur5_pour_tcp240_noforce_5090",
        project_name="pi-mamba",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            train_vision_encoder=True,
        ),
        data=_pour_full_tcp240_noforce_data_config(),
        batch_size=16,
        gradient_accumulation_steps=1,
        num_workers=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        seed=42,
        action_train_scope="fullvision_fullvlm",
        mamba_action_lr=5e-5,
        mamba_lora_lr=5e-5,
        mamba_vision_lr=5e-5,
        mamba_vlm_lora_layers="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17",
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=30_000,
        keep_period=30_000,
    ),
    TrainConfig(
        name="pi05_ww_k6_tube_force_depth_joint_full_5090",
        project_name="prunevla",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=None,
            lora_alpha=None,
            paligemma_depth=12,
            action_expert_depth=12,
            pruned_layers=(2, 4, 5, 7, 13, 15),
        ),
        data=_tube_force_depth_joint_data_config(),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        seed=42,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_prunevla_checkpoint_path("prunevla_openpi", "pi05_ww_k6"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=30_000,
        keep_period=30_000,
    ),
    TrainConfig(
        name="pi05_ww_k6_pour_tcp240_noforce_full_5090",
        project_name="prunevla",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=None,
            lora_alpha=None,
            paligemma_depth=12,
            action_expert_depth=12,
            pruned_layers=(2, 4, 5, 7, 13, 15),
        ),
        data=_pour_full_tcp240_noforce_data_config(),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        seed=42,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_prunevla_checkpoint_path("prunevla_openpi", "pi05_ww_k6"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=30_000,
        keep_period=30_000,
    ),
    TrainConfig(
        name="pi05_his_mamba_tiny_ur5_lab_v2_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            train_vision_encoder=True,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=40,
            mamba_context_language_tokens=12,
            mamba_context_history_tokens=8,
            mamba_context_global_tokens=4,
            mamba_history_length=16,
            mamba_history_depth=1,
        ),
        data=LeRobotUR5DataConfig(
            repo_id="arturaah/lab_tubes_ur5_v2",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=True,
            image_key="observation.images.wrist_left_rgb",
            wrist_image_key="observation.images.wrist_right_rgb",
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_siglip_ww_k12_vlm_ww_k6_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            train_vision_encoder=True,
            paligemma_depth=12,
            vision_depth=15,
            pruned_layers=(2, 4, 5, 7, 13, 15),
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_global_tokens=4,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path(
            "checkpoints",
            "prunevla_siglip_pruned",
            "pi05_ww_k6_siglip_ww_k12",
        ),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_vlm_ww_k6_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            train_vision_encoder=True,
            paligemma_depth=12,
            pruned_layers=(2, 4, 5, 7, 13, 15),
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_global_tokens=4,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "prunevla_openpi", "pi05_ww_k6"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_micro_vlm_ww_k6_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            train_vision_encoder=True,
            paligemma_depth=12,
            pruned_layers=(2, 4, 5, 7, 13, 15),
            # MicroMamba with the VLM WW-k6 initializer: same ~9M action-stack
            # budget as pi05_mamba_micro_libero_5090, but with 12/18 VLM blocks.
            mamba_width=320,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=4,
            mamba_mlp_dim=1280,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_global_tokens=4,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "prunevla_openpi", "pi05_ww_k6"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_his_mamba_tiny_vlm_ww_k6_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            train_vision_encoder=True,
            paligemma_depth=12,
            pruned_layers=(2, 4, 5, 7, 13, 15),
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=40,
            mamba_context_language_tokens=12,
            mamba_context_history_tokens=8,
            mamba_context_global_tokens=4,
            mamba_history_length=16,
            mamba_history_depth=1,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "prunevla_openpi", "pi05_ww_k6"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_micro_libero_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=16,
            lora_alpha=16.0,
            # MicroMamba: action stack is ~8.89M params, comparable to the
            # official pi0.5 action-expert LoRA trainable budget (~9.10M).
            mamba_width=320,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=4,
            mamba_mlp_dim=1280,
            mamba_context_compression_tokens=64,
            mamba_context_vision_tokens=44,
            mamba_context_language_tokens=16,
            mamba_context_global_tokens=4,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=4,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=1000,
        keep_period=5000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_libero_vision_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            lora_rank=32,
            lora_alpha=32.0,
            vision_lora_rank=16,
            vision_lora_alpha=16.0,
            mamba_width=448,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=4,
            mamba_mlp_dim=1792,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=4,
        gradient_accumulation_steps=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_mamba_tiny_libero_full_5090",
        project_name="pi-mamba",
        model=pi0_mamba_config.Pi0MambaConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            mamba_width=512,
            mamba_d_state=64,
            mamba_expand=1,
            mamba_headdim=64,
            mamba_chunk_size=64,
            mamba_is_mimo=False,
            mamba_depth=8,
            mamba_mlp_dim=2048,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=1,
        gradient_accumulation_steps=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_pi_mamba_resource_path("checkpoints", "pi05_base_pytorch"),
        assets_base_dir=_pi_mamba_resource_path("assets"),
        checkpoint_base_dir=_pi_mamba_resource_path("checkpoints"),
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
    ),
    _pruned_pi05_config(
        name="pi05_ww_k2_libero_lora_5090",
        method="ww",
        k=2,
        pruned_layers=(4, 13),
    ),
    _pruned_pi05_config(
        name="pi05_cka_k2_libero_lora_5090",
        method="cka",
        k=2,
        pruned_layers=(3, 4),
    ),
    _pruned_pi05_config(
        name="pi05_ww_k4_libero_lora_5090",
        method="ww",
        k=4,
        pruned_layers=(2, 4, 13, 15),
    ),
    _pruned_pi05_config(
        name="pi05_cka_k4_libero_lora_5090",
        method="cka",
        k=4,
        pruned_layers=(3, 4, 5, 6),
    ),
    _pruned_pi05_config(
        name="pi05_ww_k6_libero_lora_5090",
        method="ww",
        k=6,
        pruned_layers=(2, 4, 5, 7, 13, 15),
    ),
    _pruned_pi05_config(
        name="pi05_cka_k6_libero_lora_5090",
        method="cka",
        k=6,
        pruned_layers=(3, 4, 5, 6, 7, 10),
    ),
    _component_pruned_pi05_config(
        name="pi05_ww_free_unhealthy_libero_lora_5090",
        checkpoint_name="pi05_ww_free_unhealthy",
        pruned_vlm_layers=(2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15),
        pruned_action_expert_layers=(2, 3, 4, 5, 7, 8, 9, 10, 11, 15),
    ),
    _component_pruned_pi05_config(
        name="pi05_ww_paired_all_unhealthy_libero_lora_5090",
        checkpoint_name="pi05_ww_paired_all_unhealthy",
        pruned_vlm_layers=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        pruned_action_expert_layers=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    ),
    _component_pruned_pi05_config(
        name="pi05_manual_pair_10_16_libero_lora_5090",
        checkpoint_name="pi05_manual_pair_10_16",
        pruned_vlm_layers=(10, 16),
        pruned_action_expert_layers=(10, 16),
    ),
    _component_pruned_pi05_config(
        name="pi05_ww_random_k6_seed42_libero_lora_5090",
        checkpoint_name="pi05_ww_random_k6_seed42",
        pruned_vlm_layers=(3, 4, 6, 8, 11, 14),
        pruned_action_expert_layers=(3, 4, 6, 8, 11, 14),
    ),
    _component_pruned_pi05_config(
        name="pi05_ww_vlm_topk_k6_pf1_pl2_libero_lora_5090",
        checkpoint_name="pi05_ww_vlm_topk_k6_pf1_pl2",
        pruned_vlm_layers=(2, 3, 4, 7, 13, 15),
        pruned_action_expert_layers=(2, 3, 4, 7, 13, 15),
    ),
    _component_pruned_pi05_config(
        name="pi05_ww_vlm_topk_k9_pf1_pl2_libero_lora_5090",
        checkpoint_name="pi05_ww_vlm_topk_k9_pf1_pl2",
        pruned_vlm_layers=(1, 2, 3, 4, 5, 7, 12, 13, 15),
        pruned_action_expert_layers=(1, 2, 3, 4, 5, 7, 12, 13, 15),
    ),
    _component_pruned_pi05_config(
        name="pi05_ww_action_topk_k9_pf1_pl2_libero_lora_5090",
        checkpoint_name="pi05_ww_action_topk_k9_pf1_pl2",
        pruned_vlm_layers=(2, 3, 4, 7, 8, 9, 10, 11, 15),
        pruned_action_expert_layers=(2, 3, 4, 7, 8, 9, 10, 11, 15),
    ),
    _component_pruned_pi05_config(
        name="pi05_ww_action_topk_k6_pf1_pl2_libero_lora_5090",
        checkpoint_name="pi05_ww_action_topk_k6_pf1_pl2",
        pruned_vlm_layers=(7, 8, 9, 10, 11, 15),
        pruned_action_expert_layers=(7, 8, 9, 10, 11, 15),
    ),
    _component_pruned_pi05_config(
        name="pi05_ww_component_k6_libero_lora_5090",
        checkpoint_name="pi05_ww_component_k6",
        pruned_vlm_layers=(2, 3, 4, 7, 13, 15),
        pruned_action_expert_layers=(7, 8, 9, 10, 11, 15),
    ),
    _pruned_pi05_config(
        name="pi05_ww_k2_libero_full_5090",
        method="ww",
        k=2,
        pruned_layers=(4, 13),
        finetune_mode="full",
    ),
    _pruned_pi05_config(
        name="pi05_cka_k2_libero_full_5090",
        method="cka",
        k=2,
        pruned_layers=(3, 4),
        finetune_mode="full",
    ),
    _pruned_pi05_config(
        name="pi05_ww_k4_libero_full_5090",
        method="ww",
        k=4,
        pruned_layers=(2, 4, 13, 15),
        finetune_mode="full",
    ),
    _pruned_pi05_config(
        name="pi05_cka_k4_libero_full_5090",
        method="cka",
        k=4,
        pruned_layers=(3, 4, 5, 6),
        finetune_mode="full",
    ),
    _pruned_pi05_config(
        name="pi05_ww_k6_libero_full_5090",
        method="ww",
        k=6,
        pruned_layers=(2, 4, 5, 7, 13, 15),
        finetune_mode="full",
    ),
    _pruned_pi05_config(
        name="pi05_cka_k6_libero_full_5090",
        method="cka",
        k=6,
        pruned_layers=(3, 4, 5, 6, 7, 10),
        finetune_mode="full",
    ),
    _component_pruned_pi05_config(
        name="pi05_ww_free_unhealthy_libero_full_5090",
        checkpoint_name="pi05_ww_free_unhealthy",
        pruned_vlm_layers=(2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15),
        pruned_action_expert_layers=(2, 3, 4, 5, 7, 8, 9, 10, 11, 15),
        finetune_mode="full",
    ),
    _component_pruned_pi05_config(
        name="pi05_ww_paired_all_unhealthy_libero_full_5090",
        checkpoint_name="pi05_ww_paired_all_unhealthy",
        pruned_vlm_layers=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        pruned_action_expert_layers=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        finetune_mode="full",
    ),
    _component_pruned_pi05_config(
        name="pi05_manual_pair_10_16_libero_full_5090",
        checkpoint_name="pi05_manual_pair_10_16",
        pruned_vlm_layers=(10, 16),
        pruned_action_expert_layers=(10, 16),
        finetune_mode="full",
    ),
    _component_pruned_pi05_config(
        name="pi05_ww_random_k6_seed42_libero_full_5090",
        checkpoint_name="pi05_ww_random_k6_seed42",
        pruned_vlm_layers=(3, 4, 6, 8, 11, 14),
        pruned_action_expert_layers=(3, 4, 6, 8, 11, 14),
        finetune_mode="full",
    ),
    _component_pruned_pi05_config(
        name="pi05_ww_vlm_topk_k6_pf1_pl2_libero_full_5090",
        checkpoint_name="pi05_ww_vlm_topk_k6_pf1_pl2",
        pruned_vlm_layers=(2, 3, 4, 7, 13, 15),
        pruned_action_expert_layers=(2, 3, 4, 7, 13, 15),
        finetune_mode="full",
    ),
    _component_pruned_pi05_config(
        name="pi05_ww_vlm_topk_k9_pf1_pl2_libero_full_5090",
        checkpoint_name="pi05_ww_vlm_topk_k9_pf1_pl2",
        pruned_vlm_layers=(1, 2, 3, 4, 5, 7, 12, 13, 15),
        pruned_action_expert_layers=(1, 2, 3, 4, 5, 7, 12, 13, 15),
        finetune_mode="full",
    ),
    _component_pruned_pi05_config(
        name="pi05_ww_action_topk_k9_pf1_pl2_libero_full_5090",
        checkpoint_name="pi05_ww_action_topk_k9_pf1_pl2",
        pruned_vlm_layers=(2, 3, 4, 7, 8, 9, 10, 11, 15),
        pruned_action_expert_layers=(2, 3, 4, 7, 8, 9, 10, 11, 15),
        finetune_mode="full",
    ),
    _component_pruned_pi05_config(
        name="pi05_ww_action_topk_k6_pf1_pl2_libero_full_5090",
        checkpoint_name="pi05_ww_action_topk_k6_pf1_pl2",
        pruned_vlm_layers=(7, 8, 9, 10, 11, 15),
        pruned_action_expert_layers=(7, 8, 9, 10, 11, 15),
        finetune_mode="full",
    ),
    _component_pruned_pi05_config(
        name="pi05_ww_component_k6_libero_full_5090",
        checkpoint_name="pi05_ww_component_k6",
        pruned_vlm_layers=(2, 3, 4, 7, 13, 15),
        pruned_action_expert_layers=(7, 8, 9, 10, 11, 15),
        finetune_mode="full",
    ),
    _pruned_pi05_ur5_camera_shifts_config(
        name="pi05_ww_k6_ur5_camera_shifts_lora_5090",
        method="ww",
        k=6,
        pruned_layers=(2, 4, 5, 7, 13, 15),
    ),
    _pruned_pi05_ur5_camera_shifts_config(
        name="pi05_ww_k6_ur5_camera_shifts_full_5090",
        method="ww",
        k=6,
        pruned_layers=(2, 4, 5, 7, 13, 15),
        finetune_mode="full",
    ),
    _pruned_pi05_ur5_camera_shifts_eef_config(
        name="pi05_ww_k6_ur5_camera_shifts_eef_lora_5090",
        method="ww",
        k=6,
        pruned_layers=(2, 4, 5, 7, 13, 15),
    ),
    _pruned_pi05_ur5_camera_shifts_eef_config(
        name="pi05_ww_k6_ur5_camera_shifts_eef_full_5090",
        method="ww",
        k=6,
        pruned_layers=(2, 4, 5, 7, 13, 15),
        finetune_mode="full",
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]


def _paper_release_config(name: str, source_name: str) -> TrainConfig:
    """Create a portable, exact-default alias for a paper experiment."""
    source = next(config for config in _CONFIGS if config.name == source_name)
    model_updates: dict[str, Any] = {"train_vision_encoder": True}
    if hasattr(source.model, "mamba_depth"):
        model_updates["mamba_depth"] = 4

    return dataclasses.replace(
        source,
        name=name,
        project_name="mambaact",
        model=dataclasses.replace(source.model, **model_updates),
        batch_size=16,
        gradient_accumulation_steps=1,
        num_workers=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        seed=42,
        action_train_scope="fullvision_fullvlm",
        mamba_train_scope="fullvision_fullvlm",
        mamba_action_lr=5e-5,
        mamba_vision_lr=5e-5,
        num_train_steps=30_000,
        save_interval=30_000,
        keep_period=30_000,
    )


_PAPER_RELEASE_CONFIGS = {
    # Standard LIBERO and architecture ablations.
    "mambaact_libero": "pi05_mamba_tiny_ctxd18_gated_fullvlm_libero_5090",
    "pi05_libero_baseline": "pi05_libero_full_5090",
    "mambaact_libero_l12": "pi05_mamba_tiny_early_ctxd12_gated_fullvlm_libero_5090",
    "mambaact_libero_cross_attention": "pi05_mamba_tiny_ctxd18_crossattn_fullvlm_libero_5090",
    "transformer_action_libero_gated": "pi05_transformer_ctxd18_gated_fullvlm_libero_5090",
    "transformer_action_libero_cross_attention": "pi05_transformer_ctxd18_crossattn_fullvlm_libero_5090",
    # Real-world tube placing and nut pouring.
    "mambaact_tube": "pi05_mamba_tiny_ctxd18_depth4_nolora_fullvlm_ur5_tube_tcp240_5090",
    "pi05_tube_baseline": "pi05_icra_ur5_tube_nolora_fullvlm_5090",
    "mambaact_tube_l12": "pi05_mamba_tiny_ctxd12_depth4_nolora_fullvlm_ur5_tube_tcp240_5090",
    "mambaact_pour": "pi05_mamba_tiny_ctxd18_depth4_nolora_fullvlm_ur5_pour_tcp240_noforce_5090",
    "pi05_pour_baseline": "pi05_icra_ur5_pour_nolora_fullvlm_5090",
    "mambaact_pour_l12": "pi05_mamba_tiny_ctxd12_depth4_nolora_fullvlm_ur5_pour_tcp240_noforce_5090",
}
_CONFIGS.extend(_paper_release_config(name, source) for name, source in _PAPER_RELEASE_CONFIGS.items())

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
