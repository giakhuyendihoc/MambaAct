#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


SUITES = (
    ("libero_spatial", "Spatial"),
    ("libero_object", "Object"),
    ("libero_goal", "Goal"),
    ("libero_10", "LIBERO-10"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    rows = []
    for slug, label in SUITES:
        metrics_path = args.report_dir / f"{slug}.json"
        metrics = json.loads(metrics_path.read_text())
        success = 100.0 * float(metrics["total_success_rate"])
        rows.append((label, success, int(metrics["total_successes"]), int(metrics["total_episodes"])))

    average = sum(row[1] for row in rows) / len(rows)
    lines = [
        f"# LIBERO evaluation: {args.model}",
        "",
        f"{args.trials} trials per task, seed {args.seed}.",
        "",
        "| Suite | Success (%) | Successful episodes | Episodes |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(f"| {label} | {success:.1f} | {successes} | {episodes} |" for label, success, successes, episodes in rows)
    lines.extend(["", f"Unweighted suite average: **{average:.1f}%**", ""])
    (args.report_dir / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
