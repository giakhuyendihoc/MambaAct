# SmolVLA Released-Checkpoint 25k Comparison

The baseline is the released SmolVLA LIBERO policy, evaluated without additional
training. MambaAct replaces its expert with four gated Mamba-3 blocks of width
768, trained with the same documented budget. The optional integration leaves
the default SmolVLA Transformer path unchanged.

| Model | Spatial | Object | Goal | L-10 | Average | Model-core query (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Released SmolVLA 25k | 76.6 | 93.0 | 85.4 | 70.2 | 81.30 | 188.69 |
| SmolVLA-MambaAct 25k | 82.6 | 92.4 | 87.0 | 77.8 | 84.95 | 90.61 |

See [results and timing definitions](../../docs/RESULTS.md#smolvla-transfer)
and [task outcomes and timing samples](../../results/smolvla_25k.json).

## Pinned Inputs

| Input | Repository | Revision |
| --- | --- | --- |
| Released baseline | `lerobot/smolvla_libero` | `31d453f7edd78c839a8bbc39744a292686daf0de` |
| MambaAct initialization | `lerobot/smolvla_base` | `c83c3163b8ca9b7e67c509fffd9121e66cb96205` |
| MambaAct training data | `lerobot/libero` | `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` |
| LeRobot | `huggingface/lerobot` | `e40b58a8dfa9e7b86918c374791599d070518d11` |

The released `train_config.json` records 25,000 steps, batch 32, training seed
1000, full VLM fine-tuning, and initialization from `lerobot/smolvla_base`.
We match these documented settings. The release does not pin its dataset or
implementation revision, so this is not an identical-code retraining pair or
a reproduction of the paper's separate 87.3% reference.

## Setup

Use a separate LeRobot environment with its LIBERO and SmolVLA dependencies,
including torchcodec/FFmpeg for the AV1 dataset. Install the vendored Mamba-3
package and CUDA dependencies from the main installation guide. Commands below
use `python` from the active LeRobot environment.

```bash
git clone https://github.com/huggingface/lerobot.git third_party/lerobot
git -C third_party/lerobot checkout e40b58a8dfa9e7b86918c374791599d070518d11
export MAMBAACT_LEROBOT_ROOT="$PWD/third_party/lerobot"
bash scripts/apply_smolvla_patch.sh
export MAMBAACT_LIBERO_ROOT="$PWD/third_party/LIBERO"
export MAMBAACT_OUTPUT_ROOT="$PWD/artifacts"
```

Install and configure LIBERO's assets and initial states using the pinned
LeRobot LIBERO installation instructions before running the commands below.

## Released Baseline

```bash
bash scripts/eval_smolvla.sh released
```

This downloads the pinned official checkpoint, retains its normalization
processors, evaluates all four suites, and runs two-camera profiling. It does
not train the baseline.

## MambaAct Training

```bash
bash scripts/train_smolvla.sh mambaact
```

This downloads the pinned base and dataset, trains MambaAct, evaluates the final
checkpoint, and profiles it. Existing training outputs are never overwritten.
Only the final `checkpoints/025000/pretrained_model` and final training state
are saved; there are no intermediate checkpoints.

| Setting | Value |
| --- | --- |
| Steps / batch / accumulation | 25,000 / 32 / 1 |
| Training seed / data workers | 1000 / 8 |
| Optimizer | AdamW, betas (0.9, 0.95), epsilon 1e-8 |
| LR schedule | 1e-4 to 2.5e-6; 1,000 warmup steps; cosine decay setting 25,000 |
| Weight decay / gradient clipping | 1e-10 / 10 |
| Training scope | Full VLM and vision encoder; no LoRA; `use_amp=false` |
| Image augmentation | Disabled |
| Expert | Mamba-3, depth 4, width 768, state 64, expansion 1 |
| Head dimension / chunk size / context tokens | 64 / 64 / 64 |

Inputs are two 256x256 training images, state8, and action7. The processor
renames `image` to `camera1` and `image2` to `camera2`, and resizes with padding
to 512x512. MambaAct normalization comes from the pinned dataset. Optional
camera3 metadata inherited from the base is unused; never synthesize a third
camera when profiling these models.

## Evaluation and Timing

Both models use 50 trials/task (2,000 total), seed 7, one task/environment at a
time, horizon 50, execute 10 actions before replanning, and ten denoising steps.
Simulator images are 360x360 and resized with padding to 512x512; `use_amp=false`.

```bash
# Re-evaluate a retained or downloaded MambaAct checkpoint.
bash scripts/eval_smolvla.sh mambaact "$CHECKPOINT_DIR"
# Timing only, without rerunning rollouts.
bash scripts/profile_smolvla.sh "$CHECKPOINT_DIR" smolvla_mambaact_25k
```

Timing uses one H100, 3 warmups, and 10 measured synthetic-batch queries with
exactly camera1, camera2, and state8. Model-core query latency includes prefix
encoding/context setup and all denoising steps, but excludes preprocessing,
input transfer, noise sampling, networking, rendering, and environment steps.

Private backups under `giakhuyendihoc` are `smolvla-libero-released-25k` and
`smolvla-mambaact-libero-25k`; access requires authorization. The baseline
remains public upstream, and MambaAct can be reproduced with the training
script. Backup model and normalization tensors are unchanged; only portable
repository metadata is sanitized.
