"""CPU-only checks for main LIBERO training-seed statistics and input validation."""

import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("training_seeds", ROOT / "scripts/summarize_libero_training_seeds.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TrainingSeedTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.raw = {"seed": 7, "num_trials_per_task": 50, "replan_steps": 5,
                    "history_steps": 0, "total_episodes": 500, "total_successes": 500,
                    "total_success_rate": 1.0,
                    "per_task": [{"task_id": i, "episodes": 50, "successes": 50,
                                  "success_rate": 1.0} for i in range(10)]}
        for suite in MODULE.SUITES:
            self.write(suite, self.raw)

    def write(self, suite, raw):
        (self.directory / (suite + ".json")).write_text(json.dumps(dict(raw, task_suite_name=suite)))

    def test_historical_schema_and_complete_counts(self):
        run = MODULE.load_run(self.directory, 42)
        self.assertEqual(run["trials"], 2000)
        self.assertEqual(run["successes"], 2000)
        self.assertEqual(run["average_success_percent"], 100)
        self.assertNotIn(str(self.directory), json.dumps(run))

    def test_evaluation_seed_must_stay_fixed(self):
        self.write(MODULE.SUITES[0], dict(self.raw, seed=17))
        with self.assertRaisesRegex(ValueError, "Evaluation seed"):
            MODULE.load_run(self.directory, 42)

    def test_incomplete_task_rejected(self):
        raw = copy.deepcopy(self.raw)
        raw["per_task"][0]["episodes"] = 49
        self.write(MODULE.SUITES[0], raw)
        with self.assertRaisesRegex(ValueError, "50 trials"):
            MODULE.load_run(self.directory, 42)

    def test_duplicate_suite_rejected(self):
        source = self.directory / (MODULE.SUITES[0] + ".json")
        (self.directory / "duplicate.json").write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, "Duplicate suite"):
            MODULE.load_run(self.directory, 42)

    def test_population_standard_deviation(self):
        value = MODULE.stats([1, 2, 3])
        self.assertEqual(value["mean"], 2)
        self.assertAlmostEqual(value["std"], math.sqrt(2 / 3))
        self.assertEqual(value["n"], 3)

    def test_published_statistics_and_docs(self):
        bundle = json.loads((ROOT / "results/libero_training_seeds.json").read_text())
        runs = {(model, run["training_seed"]): run
                for model, value in bundle["models"].items() for run in value["runs"]}
        self.assertEqual(MODULE.aggregate(runs), bundle)
        self.assertEqual(sum(run["trials"] for run in runs.values()), 12000)
        self.assertEqual(bundle["statistic"]["ddof"], 0)
        self.assertAlmostEqual(bundle["models"]["pi05"]["aggregate"]["average_success_percent"]["mean"], 96.7166666667)
        self.assertAlmostEqual(bundle["models"]["mambaact"]["aggregate"]["average_success_percent"]["mean"], 97.8)
        for model in bundle["models"].values():
            values = list(model["aggregate"]["suites"].values()) + [model["aggregate"]["average_success_percent"]]
            for value in values:
                formatted = f"{value['mean']:.2f} +/- {value['std']:.2f}"
                for doc in ("README.md", "docs/RESULTS.md"):
                    self.assertIn(formatted, (ROOT / doc).read_text())


if __name__ == "__main__":
    unittest.main()
