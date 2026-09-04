#!/usr/bin/env python3
"""Summarize LIBERO-plus metrics in the official leaderboard category format."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import time
from typing import Any

DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CATEGORY_ORDER = ("Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout")
CATEGORY_LABELS = {
    "Camera Viewpoints": "Camera",
    "Robot Initial States": "Robot",
    "Language Instructions": "Language",
    "Light Conditions": "Light",
    "Background Textures": "Background",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}
SUITE_LABELS = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "LIBERO-10",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=pathlib.Path, required=True)
    parser.add_argument("--classification-json", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--suites", default=",".join(DEFAULT_SUITES))
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    suites = _parse_suites(args.suites)
    classification = _load_classification(args.classification_json)
    suite_reports = [_summarize_suite(args.metrics_dir, suite, args.trials, classification) for suite in suites]
    report = _build_report(args.model_label, args.trials, suite_reports)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or f"{_slug(args.model_label)}_libero_plus_{args.trials}trials_{time.strftime('%Y%m%d_%H%M%S')}"
    json_path = args.output_dir / f"{prefix}.json"
    md_path = args.output_dir / f"{prefix}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_format_markdown(report), encoding="utf-8")

    print(f"libero_plus_summary_json={json_path}")
    print(f"libero_plus_summary_markdown={md_path}")
    print(_format_leaderboard_table(report))


def _parse_suites(value: str) -> list[str]:
    return [suite.strip() for suite in value.split(",") if suite.strip()]


def _load_classification(path: pathlib.Path) -> dict[str, list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, list[dict[str, Any]]] = {}
    for suite, items in raw.items():
        result[suite] = []
        for idx, item in enumerate(items):
            category = CATEGORY_LABELS.get(str(item.get("category")), str(item.get("category")))
            result[suite].append(
                {
                    "index": idx,
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "category": category,
                    "raw_category": item.get("category"),
                    "difficulty_level": item.get("difficulty_level"),
                }
            )
    return result


def _summarize_suite(
    metrics_dir: pathlib.Path,
    suite: str,
    trials: int,
    classification: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    metrics_path = _latest_metrics_path(metrics_dir, suite, trials)
    if metrics_path is None:
        return {"suite": suite, "label": SUITE_LABELS.get(suite, suite), "status": "missing", "metrics_path": None}

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    category_totals = _empty_category_totals()
    name_to_meta = {str(item.get("name")): item for item in classification.get(suite, []) if item.get("name") is not None}
    class_items = classification.get(suite, [])
    matched = 0
    unmatched: list[dict[str, Any]] = []

    for task in data.get("per_task") or []:
        task_id = int(task.get("task_id") or 0)
        task_name = task.get("task_name")
        meta = name_to_meta.get(str(task_name)) if task_name else None
        if meta is None and 0 <= task_id < len(class_items):
            meta = class_items[task_id]
        if meta is None:
            unmatched.append(task)
            continue
        matched += 1
        category = meta["category"]
        if category not in category_totals:
            category_totals[category] = {"episodes": 0, "successes": 0, "tasks": 0}
        category_totals[category]["episodes"] += int(task.get("episodes") or 0)
        category_totals[category]["successes"] += int(task.get("successes") or 0)
        category_totals[category]["tasks"] += 1

    for totals in category_totals.values():
        episodes = totals["episodes"]
        totals["success_rate"] = None if episodes == 0 else totals["successes"] / episodes
        totals["success_percent"] = None if episodes == 0 else totals["success_rate"] * 100

    total_episodes = int(data.get("total_episodes") or 0)
    total_successes = int(data.get("total_successes") or 0)
    total_rate = None if total_episodes == 0 else total_successes / total_episodes

    return {
        "suite": suite,
        "label": SUITE_LABELS.get(suite, suite),
        "status": "complete",
        "metrics_path": str(metrics_path),
        "num_trials_per_task": data.get("num_trials_per_task"),
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": total_rate,
        "total_success_percent": None if total_rate is None else total_rate * 100,
        "matched_tasks": matched,
        "unmatched_tasks": len(unmatched),
        "category_summaries": category_totals,
    }


def _empty_category_totals() -> dict[str, dict[str, Any]]:
    return {category: {"episodes": 0, "successes": 0, "tasks": 0} for category in CATEGORY_ORDER}


def _latest_metrics_path(metrics_dir: pathlib.Path, suite: str, trials: int) -> pathlib.Path | None:
    direct_path = metrics_dir / f"{suite}.json"
    if direct_path.is_file():
        return direct_path
    candidates = [path for path in metrics_dir.glob(f"{suite}_{trials}trials_*.json") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _build_report(model_label: str, trials: int, suite_reports: list[dict[str, Any]]) -> dict[str, Any]:
    category_totals = _empty_category_totals()
    total_successes = 0
    total_episodes = 0
    complete = [suite for suite in suite_reports if suite.get("status") == "complete"]

    for suite in complete:
        total_successes += int(suite.get("total_successes") or 0)
        total_episodes += int(suite.get("total_episodes") or 0)
        for category, values in (suite.get("category_summaries") or {}).items():
            if category not in category_totals:
                category_totals[category] = {"episodes": 0, "successes": 0, "tasks": 0}
            category_totals[category]["episodes"] += int(values.get("episodes") or 0)
            category_totals[category]["successes"] += int(values.get("successes") or 0)
            category_totals[category]["tasks"] += int(values.get("tasks") or 0)

    for values in category_totals.values():
        episodes = values["episodes"]
        values["success_rate"] = None if episodes == 0 else values["successes"] / episodes
        values["success_percent"] = None if episodes == 0 else values["success_rate"] * 100

    total_rate = None if total_episodes == 0 else total_successes / total_episodes
    return {
        "model_label": model_label,
        "benchmark": "LIBERO-plus",
        "trials": trials,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "category_order": list(CATEGORY_ORDER),
        "leaderboard": category_totals,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "total_success_rate": total_rate,
        "total_success_percent": None if total_rate is None else total_rate * 100,
        "suite_summaries": suite_reports,
        "complete_suites": len(complete),
        "missing_suites": [suite["suite"] for suite in suite_reports if suite.get("status") != "complete"],
    }


def _format_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# LIBERO-plus Evaluation Report: {report['model_label']}",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Trials per task: {report['trials']}",
        f"- Complete suites: {report['complete_suites']}/{len(report['suite_summaries'])}",
        f"- Total success: {_percent(report.get('total_success_rate'))} ({report['total_successes']} / {report['total_episodes']})",
        "",
        "## LIBERO-Plus Benchmark Leaderboard Format",
        "",
        _format_leaderboard_table(report),
        "",
        "## Category Counts",
        "",
        "| Category | Success | Successes / Episodes | Tasks |",
        "| --- | --- | --- | --- |",
    ]
    for category in CATEGORY_ORDER:
        values = report["leaderboard"].get(category) or {}
        lines.append(
            f"| {category} | {_percent(values.get('success_rate'))} | {values.get('successes', 0)} / {values.get('episodes', 0)} | {values.get('tasks', 0)} |"
        )
    lines.extend(["", "## Suite Breakdown", "", "| Suite | Success | Successes / Episodes | Matched Tasks |", "| --- | --- | --- | --- |"])
    for suite in report["suite_summaries"]:
        if suite.get("status") != "complete":
            lines.append(f"| {suite['label']} | missing | missing | missing |")
            continue
        lines.append(
            f"| {suite['label']} | {_percent(suite.get('total_success_rate'))} | {suite.get('total_successes')} / {suite.get('total_episodes')} | {suite.get('matched_tasks')} |"
        )
    return "\n".join(lines) + "\n"


def _format_leaderboard_table(report: dict[str, Any]) -> str:
    headers = ["Model", *CATEGORY_ORDER, "Total"]
    row = [report["model_label"]]
    for category in CATEGORY_ORDER:
        row.append(_percent((report["leaderboard"].get(category) or {}).get("success_rate")))
    row.append(_percent(report.get("total_success_rate")))
    return _markdown_row(headers) + "\n" + _markdown_row(["---"] * len(headers)) + "\n" + _markdown_row(row)


def _markdown_row(values: list[Any]) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def _percent(value: Any) -> str:
    if value is None:
        return "pending"
    return f"{float(value) * 100:.1f}"


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "libero_plus_report"


if __name__ == "__main__":
    main()
