# Reproducibility Guide

## Scope

This release covers the experiments directly produced by the code in the
paper: PI0.5 and MambaAct on LIBERO, LIBERO-Plus, the action-mixer and
conditioning ablations, VLM-depth composition, the two real-world tasks, and
the SmolVLA transfer study. Cross-paper numbers in the related-work comparison
must be reproduced from their respective projects.

## Environment

The PI0.5 experiments were run with Python 3.11, PyTorch 2.7.1, CUDA 12.8,
bfloat16, and one NVIDIA H100. Exact Python dependencies are pinned in
`openpi/uv.lock`; the evaluated Mamba3 implementation is vendored in `mamba/`.

```bash
bash scripts/setup.sh
export MAMBAACT_OUTPUT_ROOT="$PWD/artifacts"
export MAMBAACT_PI05_CHECKPOINT="$MAMBAACT_OUTPUT_ROOT/checkpoints/pi05_base_pytorch"
```

The artifact directory is ignored by Git. It may be replaced with any writable
location.

## PI0.5 Initializer

All PI0.5-based rows start from the same public PI0.5 base checkpoint. Convert
it to the PyTorch layout used by this repository:

```bash
cd openpi
uv run python examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir gs://openpi-assets/checkpoints/pi05_base \
  --config-name pi05_libero_baseline \
  --output-path "$MAMBAACT_PI05_CHECKPOINT" \
  --precision bfloat16
cd ..
```

The resulting directory must contain `model.safetensors`.

## Exact Training Contract

The public config aliases have the paper defaults, and the launchers repeat the
critical overrides explicitly:

| Setting | Value |
| --- | --- |
| Optimizer steps | 30,000 |
| Batch size / accumulation | 16 / 1 |
| Data-loader workers | 8 |
| Optimizer | AdamW |
| Gradient clipping | 1.0 |
| Weight decay | `1e-10` |
| Learning rate | cosine `5e-5` to `5e-6` |
| Warmup | 0 |
| Precision | bfloat16 |
| Gradient checkpointing | enabled |
| Training seed | 42 by default; main LIBERO repeats also use 17 and 27 |
| LoRA / EMA | disabled / disabled |
| Trainable modules | vision encoder, executed VLM layers, compressor, gates, and action expert |
| Checkpoint policy | final step only; model weights only |

The main MambaAct expert has four SISO Mamba3 blocks with width 512, state size
64, expansion factor 1, head dimension 64, chunk size 64, and FFN width 2048.
Its Context Compressor emits 44 visual, 16 language, and four learned global
tokens.

## Standard LIBERO

Install the [official LIBERO benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO)
in a separate environment. The policy server uses the OpenPI environment while
the simulator client uses the LIBERO environment.

```bash
export MAMBAACT_LIBERO_ROOT="$PWD/third_party/LIBERO"
export MAMBAACT_LIBERO_PYTHON="$PWD/third_party/libero-env/bin/python"
```

Train the controlled rows:

```bash
bash scripts/train_libero.sh mambaact
bash scripts/train_libero.sh pi05
bash scripts/train_libero.sh mambaact_l12
bash scripts/train_libero.sh mamba_cross_attention
bash scripts/train_libero.sh transformer_gate
bash scripts/train_libero.sh transformer_cross_attention
```

Evaluate a final checkpoint:

```bash
bash scripts/eval_libero.sh \
  mambaact \
  "$MAMBAACT_OUTPUT_ROOT/checkpoints/mambaact_libero/mambaact_l18/30000"
```

The evaluator runs Spatial, Object, Goal, and LIBERO-10 with 50 trials per
task, seed 7, 224x224 images, horizon 10, ten Euler denoising steps, and five
executed environment actions per policy query. It writes one metrics JSON per
suite and a Markdown summary.

## Training-Seed Robustness

The main PI0.5/MambaAct comparison varies only the training seed across
42, 17, and 27. All six evaluations use seed 7 and 50 trials/task, with the
standard LIBERO settings above. This is not the separate experiment that
changes evaluation seeds for one trained checkpoint.

Use unique names to keep the checkpoints and reports separate:

```bash
for MODEL in pi05 mambaact; do
  case "$MODEL" in
    pi05) CONFIG=pi05_libero_baseline ;;
    mambaact) CONFIG=mambaact_libero ;;
  esac
  for SEED in 42 17 27; do
    RUN="${MODEL}_trainseed${SEED}"
    TRAIN_SEED="$SEED" RUN_NAME="$RUN" STEPS=30000 BATCH_SIZE=16 GRAD_ACCUM=1 \
      bash scripts/train_libero.sh "$MODEL"
    RUN_ID="${RUN}_eval50_seed7" EVAL_SEED=7 TRIALS=50 \
      bash scripts/eval_libero.sh "$MODEL" \
      "$MAMBAACT_OUTPUT_ROOT/checkpoints/$CONFIG/$RUN/30000"
  done
done
```

Aggregate the completed evaluations without including local paths in the output:

```bash
REPORTS="$MAMBAACT_OUTPUT_ROOT/reports/libero"
python scripts/summarize_libero_training_seeds.py \
  --run pi05 42 "$REPORTS/pi05_trainseed42_eval50_seed7" \
  --run pi05 17 "$REPORTS/pi05_trainseed17_eval50_seed7" \
  --run pi05 27 "$REPORTS/pi05_trainseed27_eval50_seed7" \
  --run mambaact 42 "$REPORTS/mambaact_trainseed42_eval50_seed7" \
  --run mambaact 17 "$REPORTS/mambaact_trainseed17_eval50_seed7" \
  --run mambaact 27 "$REPORTS/mambaact_trainseed27_eval50_seed7" \
  --output "$MAMBAACT_OUTPUT_ROOT/reports/libero_training_seeds.json"
```

The exporter validates all 40 tasks and 2,000 trials per checkpoint, the fixed
evaluation seed, and the five-action replanning interval. It reports the mean
and population standard deviation (`ddof=0`) across training seeds, including
the standard deviation of per-run suite averages. The recorded scores are
available in [the public result bundle](../results/libero_training_seeds.json).
Latency remains a separately labeled seed-42 reference, not a multi-seed mean.

## LIBERO-Plus

Use the [official LIBERO-Plus repository](https://github.com/sylvestf/LIBERO-plus)
and follow its asset-generation instructions. Point `MAMBAACT_LIBERO_ROOT` and
`MAMBAACT_LIBERO_PYTHON` to that installation, then run one fixed-seed trial
for each listed variant:

```bash
bash scripts/eval_libero_plus.sh \
  mambaact \
  "$MAMBAACT_OUTPUT_ROOT/checkpoints/mambaact_libero/mambaact_l18/30000" \
  "$MAMBAACT_LIBERO_ROOT/libero/libero/benchmark/task_classification.json"
```

Run the same command with `pi05` and its matched checkpoint. The summarizer
reports Camera, Robot, Language, Light, Background, Noise, Layout, and Total in
the paper's leaderboard format.

## Latency Profiling

Component timing uses three warm-up queries and ten timed queries on one
canonical LIBERO observation. Prefix includes image/language encoding and
context or KV-cache construction; Denoise includes all ten flow-matching
steps.

```bash
bash scripts/profile_checkpoint.sh \
  mambaact \
  "$MAMBAACT_OUTPUT_ROOT/checkpoints/mambaact_libero/mambaact_l18/30000"
```

Latency depends on hardware, software versions, clocking, and concurrent load.
Use the same accelerator and settings when comparing against the paper.

## Real-World Training

The paper uses two RGB views, seven-dimensional joint/gripper state, and
six delta-joint targets plus an unchanged gripper target. The released dataset
identifiers are:

| Task | Dataset |
| --- | --- |
| Tube placing | `giakhuyendihoc/ur5_tube_full_tcp240_converted` |
| Nut pouring | `giakhuyendihoc/ur5_pour_full_tcp240_noforce` |

Train each PI0.5, MambaAct, or MambaAct-L12 policy with the same 30,000-step
recipe:

```bash
bash scripts/train_real_world.sh mambaact tube
bash scripts/train_real_world.sh pi05 tube
bash scripts/train_real_world.sh mambaact_l12 tube

bash scripts/train_real_world.sh mambaact pour
bash scripts/train_real_world.sh pi05 pour
bash scripts/train_real_world.sh mambaact_l12 pour
```

Real-world inference uses delta-joint action chunks with horizon 50. The first
30 actions are executed at 30 Hz before replanning. The reported latency was
measured on one NVIDIA RTX 5090. `scripts/serve_checkpoint.sh` starts the policy
server; robot-side safety checks, workspace limits, and emergency-stop logic
remain deployment-specific and must be validated before execution.

## SmolVLA Transfer

The transfer experiment is isolated from the PI0.5 environment and is pinned
to the evaluated LeRobot revision. Follow
[`integrations/smolvla/README.md`](../integrations/smolvla/README.md). Its
baseline is the released `lerobot/smolvla_libero` checkpoint, with no additional
baseline training. SmolVLA-MambaAct uses 25,000 steps, batch size 32, training
seed 1000, 1,000 warmup steps, a cosine schedule from `1e-4` to `2.5e-6`, and
final-only checkpointing. Both models are evaluated with 50 trials/task,
evaluation seed 7, horizon 50, execute 10, ten denoising steps, and two cameras.
The linked protocol pins the baseline, initialization, dataset, and evaluation
implementation and explains the limits of matching a released checkpoint.
