import collections
import dataclasses
import json
import logging
import math
import pathlib
import time
from typing import Dict, List, Optional, Union

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi.policies.history import RollingHistoryBuffer
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    history_steps: int = 0

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    metrics_out_path: Optional[str] = None  # Optional JSON file for success-rate and inference-latency metrics
    action_log_out_path: Optional[str] = None  # Optional JSON file with action chunks / executed actions.
    save_videos: bool = True  # Disable for very large benchmarks such as LIBERO-plus.
    start_global_episode: int = 0  # Skip this many suite episodes before evaluating; used for interrupted resumes.

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    eval_start_time = time.monotonic()

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    if args.save_videos:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    server_infer_ms: List[float] = []
    client_roundtrip_ms: List[float] = []
    action_chunks = 0
    executed_policy_actions = 0
    per_task_results = []
    episode_policy_steps_all: List[int] = []
    episode_policy_steps_success: List[int] = []
    action_log = None
    if args.action_log_out_path is not None:
        action_log = {
            "task_suite_name": args.task_suite_name,
            "seed": args.seed,
            "num_trials_per_task": args.num_trials_per_task,
            "replan_steps": args.replan_steps,
            "history_steps": args.history_steps,
            "resize_size": args.resize_size,
            "start_global_episode": args.start_global_episode,
            "episodes": [],
        }
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task_start_global_episode = task_id * args.num_trials_per_task
        task_end_global_episode = task_start_global_episode + args.num_trials_per_task
        if task_end_global_episode <= args.start_global_episode:
            continue

        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_policy_steps_all: List[int] = []
        task_policy_steps_success: List[int] = []
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            global_episode_idx = task_start_global_episode + episode_idx
            if global_episode_idx < args.start_global_episode:
                continue

            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()
            history_buffer: Optional[RollingHistoryBuffer] = None

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            done = False
            replay_images = []
            episode_action_chunks = []
            episode_executed_actions = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": state,
                            "prompt": str(task_description),
                        }
                        if args.history_steps > 0:
                            if history_buffer is None:
                                history_buffer = RollingHistoryBuffer(
                                    history_steps=args.history_steps,
                                    state_dim=state.shape[-1],
                                    action_dim=len(LIBERO_DUMMY_ACTION),
                                )
                            history_buffer.update_payload(element)

                        # Query model to get action
                        infer_start = time.monotonic()
                        policy_result = client.infer(element)
                        client_roundtrip_ms.append((time.monotonic() - infer_start) * 1000)
                        timing = policy_result.get("policy_timing", {})
                        if "infer_ms" in timing:
                            server_infer_ms.append(float(timing["infer_ms"]))
                        action_chunk = policy_result["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        if action_log is not None:
                            episode_action_chunks.append(
                                {
                                    "query_step": int(t),
                                    "actions": np.asarray(action_chunk, dtype=np.float32).tolist(),
                                }
                            )
                        action_plan.extend(action_chunk[: args.replan_steps])
                        action_chunks += 1

                    action = action_plan.popleft()
                    executed_policy_actions += 1
                    if action_log is not None:
                        episode_executed_actions.append(
                            {
                                "step": int(t),
                                "action": np.asarray(action, dtype=np.float32).tolist(),
                            }
                        )

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if history_buffer is not None:
                        history_buffer.append(state, action)
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1
            policy_steps = max(0, int(t) - int(args.num_steps_wait))
            task_policy_steps_all.append(policy_steps)
            episode_policy_steps_all.append(policy_steps)
            if done:
                task_policy_steps_success.append(policy_steps)
                episode_policy_steps_success.append(policy_steps)

            if args.save_videos:
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            if action_log is not None:
                action_log["episodes"].append(
                    {
                        "task_id": int(task_id),
                        "episode_idx": int(episode_idx),
                        "task_name": getattr(task, "name", None),
                        "task_bddl_file": getattr(task, "bddl_file", None),
                        "task_problem_folder": getattr(task, "problem_folder", None),
                        "task_description": task_description,
                        "success": bool(done),
                        "steps": int(t),
                        "policy_steps": int(policy_steps),
                        "max_steps": int(max_steps),
                        "num_steps_wait": int(args.num_steps_wait),
                        "action_chunks": episode_action_chunks,
                        "executed_actions": episode_executed_actions,
                    }
                )

        if task_episodes == 0:
            continue

        task_success_rate = float(task_successes) / float(task_episodes)
        total_success_rate = float(total_successes) / float(total_episodes)
        per_task_results.append(
            {
                "task_id": task_id,
                "task_name": getattr(task, "name", None),
                "task_bddl_file": getattr(task, "bddl_file", None),
                "task_problem_folder": getattr(task, "problem_folder", None),
                "task_description": task_description,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": task_success_rate,
                "policy_steps": _summarize_numeric(task_policy_steps_all),
                "successful_policy_steps": _summarize_numeric(task_policy_steps_success),
            }
        )

        # Log final results
        logging.info(f"Current task success rate: {task_success_rate}")
        logging.info(f"Current total success rate: {total_success_rate}")

    eval_wall_time_seconds = time.monotonic() - eval_start_time
    latency_summary = _summarize_latency(server_infer_ms, client_roundtrip_ms, args.replan_steps)
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0
    metrics = {
        "task_suite_name": args.task_suite_name,
        "seed": args.seed,
        "num_trials_per_task": args.num_trials_per_task,
        "replan_steps": args.replan_steps,
        "history_steps": args.history_steps,
        "start_global_episode": args.start_global_episode,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": total_success_rate,
        "action_chunks": action_chunks,
        "executed_policy_actions": executed_policy_actions,
        "eval_wall_time_seconds": eval_wall_time_seconds,
        "eval_wall_time_minutes": eval_wall_time_seconds / 60.0,
        "episodes_per_minute": total_episodes / (eval_wall_time_seconds / 60.0) if eval_wall_time_seconds > 0 else None,
        "policy_steps": _summarize_numeric(episode_policy_steps_all),
        "successful_policy_steps": _summarize_numeric(episode_policy_steps_success),
        "latency": latency_summary,
        "per_task": per_task_results,
    }

    logging.info(f"Total success rate: {total_success_rate}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Action chunks queried: {action_chunks}")
    logging.info(f"Executed policy actions: {executed_policy_actions}")
    logging.info(f"Eval wall time: {eval_wall_time_seconds / 60.0:.2f} min ({eval_wall_time_seconds:.1f} sec)")
    logging.info(f"Episodes per minute: {total_episodes / (eval_wall_time_seconds / 60.0) if eval_wall_time_seconds > 0 else float('nan'):.3f}")
    logging.info(f"History steps: {args.history_steps}")
    _log_latency_summary(latency_summary)

    if args.metrics_out_path is not None:
        metrics_path = pathlib.Path(args.metrics_out_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logging.info(f"Wrote metrics: {metrics_path}")
    if args.action_log_out_path is not None and action_log is not None:
        action_log["total_episodes"] = total_episodes
        action_log["total_successes"] = total_successes
        action_log["total_success_rate"] = total_success_rate
        action_log["action_chunks"] = action_chunks
        action_log["executed_policy_actions"] = executed_policy_actions
        action_log["eval_wall_time_seconds"] = eval_wall_time_seconds
        action_log_path = pathlib.Path(args.action_log_out_path)
        action_log_path.parent.mkdir(parents=True, exist_ok=True)
        action_log_path.write_text(json.dumps(action_log, indent=2), encoding="utf-8")
        logging.info(f"Wrote action log: {action_log_path}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _summarize_numeric(values: List[Union[int, float]]) -> Dict[str, Optional[Union[float, int]]]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _summarize(values: List[float]) -> Dict[str, Optional[Union[float, int]]]:
    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def _summarize_latency(
    server_infer_ms: List[float], client_roundtrip_ms: List[float], replan_steps: int
) -> Dict[str, Dict[str, Optional[Union[float, int]]]]:
    server_summary = _summarize(server_infer_ms)
    client_summary = _summarize(client_roundtrip_ms)
    server_per_action = [value / replan_steps for value in server_infer_ms]
    client_per_action = [value / replan_steps for value in client_roundtrip_ms]
    return {
        "server_model_infer": server_summary,
        "client_roundtrip": client_summary,
        "server_model_infer_per_executed_action": _summarize(server_per_action),
        "client_roundtrip_per_executed_action": _summarize(client_per_action),
    }


def _log_latency_summary(latency_summary: Dict[str, Dict[str, Optional[Union[float, int]]]]) -> None:
    for name, summary in latency_summary.items():
        if summary["count"] == 0:
            logging.info(f"{name}: no samples")
            continue
        logging.info(
            "%s latency: count=%d mean=%.2fms median=%.2fms p95=%.2fms min=%.2fms max=%.2fms",
            name,
            summary["count"],
            summary["mean_ms"],
            summary["median_ms"],
            summary["p95_ms"],
            summary["min_ms"],
            summary["max_ms"],
        )


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
