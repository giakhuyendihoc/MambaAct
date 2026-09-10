#!/usr/bin/env python3
"""Validate and aggregate the main LIBERO training-seed comparison."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SEEDS = (42, 17, 27)
MODELS = ("pi05", "mambaact")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_run(directory, training_seed):
    suites = {}
    for path in sorted(Path(directory).glob("*.json")):
        content = path.read_bytes()
        raw = json.loads(content)
        suite = raw.get("task_suite_name")
        if suite not in SUITES:
            continue
        require(suite not in suites, f"Duplicate suite report: {suite}")
        require(raw["seed"] == 7, "Evaluation seed must remain 7")
        require(raw["num_trials_per_task"] == 50, "Expected 50 trials/task")
        require(raw["replan_steps"] == 5, "Expected execute5")
        require(raw["history_steps"] == 0, "Expected no history input")
        require("start_global_episode" not in raw or raw["start_global_episode"] == 0,
                "Expected a complete suite rather than a partial resume")
        tasks = raw["per_task"]
        require(len(tasks) == 10 and {t["task_id"] for t in tasks} == set(range(10)),
                "Expected all ten unique tasks")
        clean_tasks = []
        for task in tasks:
            require(task["episodes"] == 50, "Each task must contain 50 trials")
            successes = task["successes"]
            require(isinstance(successes, int) and 0 <= successes <= 50, "Invalid success count")
            require(math.isclose(task["success_rate"], successes / 50), "Task success rate mismatch")
            clean_tasks.append({"task_id": task["task_id"], "successes": successes, "trials": 50})
        successes = sum(t["successes"] for t in clean_tasks)
        require(raw["total_episodes"] == 500 and raw["total_successes"] == successes,
                "Suite totals do not match task counts")
        require(math.isclose(raw["total_success_rate"], successes / 500), "Suite success rate mismatch")
        suites[suite] = {"successes": successes, "trials": 500,
                         "success_percent": successes / 5,
                         "per_task": sorted(clean_tasks, key=lambda t: t["task_id"]),
                         "source_sha256": hashlib.sha256(content).hexdigest()}
    require(set(suites) == set(SUITES), "Missing one or more suite reports")
    successes = sum(suites[name]["successes"] for name in SUITES)
    return {"training_seed": training_seed, "evaluation_seed": 7, "trials": 2000,
            "successes": successes, "average_success_percent": successes / 20,
            "suites": {name: suites[name] for name in SUITES}}


def stats(values):
    return {"mean": statistics.mean(values), "std": statistics.pstdev(values), "n": len(values)}


def aggregate(runs):
    require(len(runs) == 6, "Expected exactly six model/seed runs")
    require(set(runs) == {(model, seed) for model in MODELS for seed in SEEDS},
            "Expected both main models at training seeds 42, 17, and 27")
    models = {}
    for model in MODELS:
        ordered = [runs[model, seed] for seed in SEEDS]
        models[model] = {
            "runs": ordered,
            "aggregate": {
                "suites": {name: stats([r["suites"][name]["success_percent"] for r in ordered])
                           for name in SUITES},
                "average_success_percent": stats([r["average_success_percent"] for r in ordered]),
            },
        }
    gains = [runs["mambaact", seed]["average_success_percent"] -
             runs["pi05", seed]["average_success_percent"] for seed in SEEDS]
    return {
        "schema_version": 1,
        "training_seeds": list(SEEDS),
        "evaluation": {"seed": 7, "trials_per_task": 50, "suites": list(SUITES),
                       "trials_per_model_per_seed": 2000, "replan_steps": 5},
        "statistic": {"name": "mean and population standard deviation across training seeds",
                      "ddof": 0, "unit": "percentage points", "not_ci_or_sem": True},
        "models": models,
        "paired_average_gain": {"training_seeds": list(SEEDS), "values": gains, **stats(gains)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=3, action="append", required=True,
                        metavar=("MODEL", "TRAIN_SEED", "METRICS_DIR"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for model, seed, directory in args.run:
        key = (model, int(seed))
        require(key not in runs, "Duplicate model/training-seed input")
        runs[key] = load_run(directory, key[1])
    result = aggregate(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for model, value in result["models"].items():
        overall = value["aggregate"]["average_success_percent"]
        print(f"{model}: {overall['mean']:.2f} +/- {overall['std']:.2f}% (population SD, n=3)")


if __name__ == "__main__":
    main()
