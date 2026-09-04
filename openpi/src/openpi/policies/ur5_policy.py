import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_ur5_example() -> dict:
    """Creates a random input example for the UR5 policy."""
    return {
        "observation/state": np.random.rand(7),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "place the lab test tube with red contents and place it in the smaller lab test tube holder",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):
    """Inputs for the UR5 lab test tube dataset and robot policy.

    Expected canonical keys:
    - observation/state: [7], six UR5 joints plus gripper.
    - observation/image: left context RGB image.
    - observation/wrist_image: right wrist RGB image.
    - actions: [action_horizon, 7], only during training.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": np.asarray(data["observation/state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        state_history = data.get("state_history", data.get("observation/state_history"))
        action_history = data.get("action_history", data.get("observation/action_history"))
        history_mask = data.get("history_mask", data.get("observation/history_mask"))
        if state_history is not None:
            inputs["state_history"] = np.asarray(state_history)
        if action_history is not None:
            action_history = np.asarray(action_history)
            if state_history is not None and not bool(data.get("action_history_is_delta", False)):
                state_history = np.asarray(state_history)
                dims = min(6, action_history.shape[-1], state_history.shape[-1])
                action_history = action_history.copy()
                action_history[..., :dims] -= state_history[..., :dims]
            inputs["action_history"] = action_history
        if history_mask is not None:
            inputs["history_mask"] = np.asarray(history_mask, dtype=bool)

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):
    """Outputs for UR5 control.

    Returns the first 7 action dimensions: six UR5 joints plus gripper.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class BuildUR5JointForceStateActions(transforms.DataTransformFn):
    """Build joint-space state with force/torque context for Famba.

    Current state is six joints + gripper + six F/T channels. The action target
    remains seven-dimensional joint/gripper control. When the loader supplies a
    temporal state/force window, the latest ``history_length`` state/force
    entries, including the current entry, become ``state_history``. Past actions
    are dropped; Famba uses state/force history only.
    """

    history_length: int = 0

    def _make_single_state_history(self, fused_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state_history = np.zeros((self.history_length, fused_state.shape[-1]), dtype=np.float32)
        history_mask = np.zeros((self.history_length,), dtype=bool)
        if self.history_length > 0:
            state_history[-1] = fused_state
            history_mask[-1] = True
        return state_history, history_mask

    def __call__(self, data: dict) -> dict:
        joint_state = np.asarray(data["observation/state"], dtype=np.float32)
        force_torque = np.asarray(data["observation/force_torque"], dtype=np.float32)
        if joint_state.ndim >= 2:
            if self.history_length <= 0:
                current_index = joint_state.shape[0] - 1
            else:
                if force_torque.ndim < 2:
                    raise ValueError("observation/force_torque must be a sequence when observation/state is a sequence")
                if joint_state.shape[0] != force_torque.shape[0]:
                    raise ValueError(
                        "observation/state and observation/force_torque sequence lengths must match for Famba; "
                        f"got {joint_state.shape[0]} and {force_torque.shape[0]}"
                    )
                current_index = joint_state.shape[0] - 1
                history_start = max(0, current_index + 1 - self.history_length)
                fused_history = np.concatenate(
                    [joint_state[history_start : current_index + 1], force_torque[history_start : current_index + 1]],
                    axis=-1,
                )
                state_history = np.zeros((self.history_length, fused_history.shape[-1]), dtype=np.float32)
                state_history[-fused_history.shape[0] :] = fused_history

                state_pad = np.asarray(
                    data.pop("observation/state_is_pad", np.zeros(joint_state.shape[0], dtype=bool))
                ).astype(bool)
                force_pad = np.asarray(
                    data.pop("observation/force_torque_is_pad", np.zeros(force_torque.shape[0], dtype=bool))
                ).astype(bool)
                fused_mask = ~(state_pad[history_start : current_index + 1] | force_pad[history_start : current_index + 1])
                history_mask = np.zeros((self.history_length,), dtype=bool)
                history_mask[-fused_mask.shape[0] :] = fused_mask
                data["state_history"] = state_history
                data["history_mask"] = history_mask

            data["observation/state"] = np.concatenate([joint_state[current_index], force_torque[current_index]], axis=-1)
            if "actions" in data:
                actions = np.asarray(data["actions"], dtype=np.float32)
                if actions.ndim >= 2 and actions.shape[0] > current_index:
                    data["actions"] = actions[current_index:]
                else:
                    data["actions"] = actions
        else:
            fused_state = np.concatenate([joint_state, force_torque], axis=-1)
            data["observation/state"] = fused_state
            if self.history_length > 0:
                data["state_history"], data["history_mask"] = self._make_single_state_history(fused_state)

        return data


def _normalize_quaternion_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, 1e-8)


def _tcp_correct_pose_xyzw(pose: np.ndarray, tcp_z_offset_m: float) -> np.ndarray:
    """Move an EEF pose along its local +Z axis by tcp_z_offset_m.

    Pose layout is [x, y, z, qx, qy, qz, qw].
    """

    pose = np.asarray(pose, dtype=np.float32).copy()
    quat = _normalize_quaternion_xyzw(pose[..., 3:7])
    qx, qy, qz, qw = np.moveaxis(quat, -1, 0)
    local_z_in_base = np.stack(
        [
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
        axis=-1,
    )
    pose[..., :3] += float(tcp_z_offset_m) * local_z_in_base
    pose[..., 3:7] = quat
    return pose


@dataclasses.dataclass(frozen=True)
class BuildUR5EEFStateActions(transforms.DataTransformFn):
    """Build TCP-corrected EEF state/actions from UR5 joint dataset fields.

    Expected canonical keys before this transform:
    - observation/state: [7], used only for current gripper state.
    - eef_pose_sequence: [H + 1, 7], old TCP pose sequence.
    - actions: [H, 7], joint action sequence, used only for gripper targets.

    Output:
    - observation/state: [8] = corrected EEF pose at t + current gripper.
    - actions: [H, 8] = corrected EEF pose targets at t+1..t+H + target gripper.
    """

    tcp_z_offset_m: float = 0.041

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            if "eef_pose_sequence" in data:
                eef_pose_sequence = np.asarray(data["eef_pose_sequence"], dtype=np.float32)
                corrected_eef = _tcp_correct_pose_xyzw(eef_pose_sequence, self.tcp_z_offset_m)
                joint_state = np.asarray(data["observation/state"], dtype=np.float32)
                data["observation/state"] = np.concatenate([corrected_eef[0], joint_state[..., 6:7]], axis=-1)
            return data

        eef_pose_sequence = np.asarray(data["eef_pose_sequence"], dtype=np.float32)
        if eef_pose_sequence.ndim == 1:
            raise ValueError("eef_pose_sequence must be a sequence with shape [H + 1, 7].")
        corrected_eef = _tcp_correct_pose_xyzw(eef_pose_sequence, self.tcp_z_offset_m)

        joint_state = np.asarray(data["observation/state"], dtype=np.float32)
        joint_actions = np.asarray(data["actions"], dtype=np.float32)
        horizon = joint_actions.shape[0]
        if corrected_eef.shape[0] < horizon + 1:
            raise ValueError(
                "eef_pose_sequence must contain current pose plus future target poses; "
                f"got {corrected_eef.shape[0]} poses for horizon {horizon}."
            )

        current_gripper = joint_state[..., 6:7]
        target_gripper = joint_actions[..., 6:7]
        data["observation/state"] = np.concatenate([corrected_eef[0], current_gripper], axis=-1)
        data["actions"] = np.concatenate([corrected_eef[1 : horizon + 1], target_gripper], axis=-1)
        return data


@dataclasses.dataclass(frozen=True)
class UR5EEFInputs(transforms.DataTransformFn):
    """Inputs for the UR5 EEF-pose policy.

    Expected canonical keys:
    - observation/state: [8], TCP-corrected EEF pose plus gripper.
    - observation/image: primary context RGB image.
    - observation/wrist_image: wrist RGB image.
    - actions: [action_horizon, 8], only during training.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": np.asarray(data["observation/state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class UR5EEFOutputs(transforms.DataTransformFn):
    """Outputs first 8 dimensions: TCP-corrected EEF pose plus gripper."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}
