"""Validate published SmolVLA results and launcher arguments without a GPU."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ResultsTest(unittest.TestCase):
    def test_complete_results(self):
        bundle = json.loads((ROOT / "results/smolvla_25k.json").read_text())
        self.assertEqual(bundle["training"], {"steps": 25000, "batch_size": 32, "seed": 1000})
        self.assertEqual(len(bundle["models"]), 2)
        for row, count in zip(bundle["models"], (1626, 1699)):
            tasks = row["per_task"]
            self.assertEqual(len({(t["suite"], t["task_id"]) for t in tasks}), 40)
            self.assertEqual(sum(t["successes"] for t in tasks), count)
            self.assertEqual(row["average_success_percent"], count / 20)
            for task in tasks:
                self.assertEqual(len(task["outcomes"]), 50)
                self.assertEqual(sum(task["outcomes"]), task["successes"])
            timing = row["timing"]
            self.assertEqual(len(timing["image_keys"]), 2)
            self.assertEqual(timing["chunk_size"], 50)
            self.assertEqual(timing["executed_actions"], 10)
            self.assertEqual(timing["num_denoise_steps"], 10)
            self.assertEqual(len(timing["raw_runs"]), 10)
            self.assertNotIn("policy_path", timing)


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        fake = self.root / "python"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, pathlib, sys\n"
            "args = sys.argv[1:]\n"
            "with open(os.environ['CALL_LOG'], 'a') as log: log.write(json.dumps(args) + '\\n')\n"
            "if 'lerobot.scripts.lerobot_train' in args:\n"
            "    out = next(x.split('=', 1)[1] for x in args if x.startswith('--output_dir='))\n"
            "    p = pathlib.Path(out) / 'checkpoints/025000/pretrained_model/model.safetensors'\n"
            "    p.parent.mkdir(parents=True); p.write_bytes(b'test')\n"
        )
        fake.chmod(0o755)
        self.log = self.root / "calls.jsonl"
        self.env = dict(os.environ, PATH=str(self.root) + os.pathsep + os.environ["PATH"],
                        CALL_LOG=str(self.log), MAMBAACT_OUTPUT_ROOT=str(self.root / "artifacts"),
                        MAMBAACT_LEROBOT_ROOT=str(self.root / "lerobot"),
                        MAMBAACT_LIBERO_ROOT=str(self.root / "libero"))
        for key in ("OUTPUT_DIR", "EVAL_OUTPUT_DIR", "TIMING_OUTPUT_DIR"):
            self.env.pop(key, None)

    def run_script(self, name, *args):
        return subprocess.run(["bash", str(ROOT / "scripts" / name), *args],
                              env=self.env, capture_output=True, text=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_training_protocol(self):
        result = self.run_script("train_smolvla.sh", "mambaact")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        train = next(c for c in calls if "lerobot.scripts.lerobot_train" in c)
        for flag in ("--steps=25000", "--batch_size=32", "--seed=1000", "--save_freq=25000",
                     "--policy.scheduler_decay_steps=25000", "--policy.use_amp=false",
                     "--policy.mamba_depth=4", "--policy.mamba_width=768"):
            self.assertIn(flag, train)
        self.check_eval_and_profile(calls)

    def check_eval_and_profile(self, calls):
        evaluation = next(c for c in calls if "lerobot.scripts.lerobot_eval" in c)
        for flag in ("--eval.n_episodes=50", "--seed=7", "--policy.chunk_size=50",
                     "--policy.n_action_steps=10", "--policy.num_steps=10", "--env.max_parallel_tasks=1"):
            self.assertIn(flag, evaluation)
        profile = next(c for c in calls if c[0].endswith("profile_smolvla_component_timing.py"))
        self.assertIn("observation.images.camera1", profile)
        self.assertIn("observation.images.camera2", profile)
        self.assertNotIn("observation.images.camera3", profile)

    def test_released_baseline_is_not_trained(self):
        model = self.root / "released"
        model.mkdir()
        (model / "model.safetensors").write_bytes(b"test")
        result = self.run_script("eval_smolvla.sh", "released", str(model))
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertFalse(any("lerobot.scripts.lerobot_train" in c for c in calls))
        self.check_eval_and_profile(calls)

    def test_baseline_training_rejected(self):
        result = self.run_script("train_smolvla.sh", "baseline")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.log.exists())

    def test_existing_training_not_overwritten(self):
        target = Path(self.env["MAMBAACT_OUTPUT_ROOT"]) / "smolvla/mambaact_25k"
        target.mkdir(parents=True)
        result = self.run_script("train_smolvla.sh", "mambaact")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
