# MambaAct

Official implementation and reproducibility release for **MambaAct: Efficient
Mamba Action Experts for Flow-Matching Vision-Language-Action Models**.

MambaAct retains the pretrained PI0.5 visual-language backbone and replaces
its repeatedly evaluated Transformer action expert with a compact Mamba3
expert. A Context Compressor maps the VLM prefix to 64 tokens, and blockwise
FiLM gates condition four Mamba action blocks without repeating token-level
cross-attention at every denoising step.

The action-expert core has **29.7M parameters**, compared with **311.5M** in
the original PI0.5 expert. Under the matched LIBERO protocol, MambaAct improves
average success from 96.5% to 98.4% while reducing ten-step denoising latency
by 64.8%.

## Architecture

```text
wrist image + third-person image + language
                    |
                    v
        PI0.5 SigLIP + Gemma prefix
                    |
             VLM prefix tokens
                    |
                    v
             Context Compressor
       44 visual + 16 language + 4 global
                    |
             64 context tokens
                    |
             masked mean pool
                    |
             condition vector z
                    |
                    v
state + noisy action chunk + flow time
                    |
          4 gated Mamba3 blocks
                    |
          flow velocity prediction
                    |
           Euler denoising update
```

The visual-language context and its per-block FiLM parameters are computed
once per policy query and reused across all ten denoising steps. Mamba3 is
applied in its SISO form along the action-token sequence.

## Main Results

LIBERO results use 50 trials per task and evaluation seed 7. Latency was
measured on one NVIDIA H100 with action horizon 10 and ten denoising steps.

| Model | Spatial | Object | Goal | L-10 | Avg. | Model size (B) | Prefix (ms) | Denoise (ms) | E2E (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PI0.5 | 97.6 | 98.8 | 96.6 | 93.2 | 96.5 | 3.62 | 40.76 | 160.26 | 228.6 |
| **MambaAct** | **99.2** | **99.8** | **98.8** | **95.8** | **98.4** | **2.96** | 46.61 | **56.40** | **107.6** |

MambaAct also reaches 79.7% on LIBERO-Plus versus 78.7% for the matched PI0.5
baseline. Complete paper tables, including real-world results and ablations,
are recorded in [docs/RESULTS.md](docs/RESULTS.md).

## Installation

The PI0.5 experiments require Linux, Python 3.11, a CUDA-capable GPU, and
[`uv`](https://docs.astral.sh/uv/). The exact dependency lockfile and the
evaluated Mamba3 source are included.

```bash
git clone git@github.com:giakhuyendihoc/MambaAct.git
cd MambaAct
bash scripts/setup.sh
```

Keep model weights, datasets, logs, and reports outside version control:

```bash
export MAMBAACT_OUTPUT_ROOT="$PWD/artifacts"
export MAMBAACT_PI05_CHECKPOINT="$MAMBAACT_OUTPUT_ROOT/checkpoints/pi05_base_pytorch"
```

Initializer conversion and benchmark setup are described in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Reproducing LIBERO

The paper recipe uses full-model fine-tuning without LoRA or EMA: 30,000
optimizer steps, batch size 16, gradient accumulation 1, eight data workers,
training seed 42, AdamW with gradient clipping at 1.0, and a cosine learning
rate from `5e-5` to `5e-6` without warmup. Only the final checkpoint is saved.

```bash
# Main MambaAct model
bash scripts/train_libero.sh mambaact

# Matched PI0.5 baseline
bash scripts/train_libero.sh pi05
```

The same launcher covers the paper ablations:

```bash
bash scripts/train_libero.sh mambaact_l12
bash scripts/train_libero.sh mamba_cross_attention
bash scripts/train_libero.sh transformer_gate
bash scripts/train_libero.sh transformer_cross_attention
```

For standard evaluation, provide a LIBERO checkout and its Python environment:

```bash
export MAMBAACT_LIBERO_ROOT="$PWD/third_party/LIBERO"
export MAMBAACT_LIBERO_PYTHON="$PWD/third_party/libero-env/bin/python"
bash scripts/eval_libero.sh mambaact artifacts/checkpoints/mambaact_libero/mambaact_l18/30000
```

This evaluates Spatial, Object, Goal, and LIBERO-10 with 50 trials per task,
seed 7, image size 224, action horizon 10, ten denoising steps, and replanning
every five environment steps.

## Other Experiments

- [Real-world tube and pouring tasks](docs/REPRODUCIBILITY.md#real-world-training)
- [SmolVLA transfer experiment](integrations/smolvla/README.md)
- [Component latency profiling](docs/REPRODUCIBILITY.md#latency-profiling)
- [LIBERO-Plus evaluation](docs/REPRODUCIBILITY.md#libero-plus)

## Code Map

| Component | Location |
| --- | --- |
| MambaAct, Context Compressor, gates, and ablations | `openpi/src/openpi/models_pytorch/pi0_mamba_pytorch.py` |
| MambaAct model configuration | `openpi/src/openpi/models/pi0_mamba_config.py` |
| Paper training configs | `openpi/src/openpi/training/config.py` |
| PI0.5 baseline | `openpi/src/openpi/models_pytorch/pi0_pytorch.py` |
| PyTorch trainer | `openpi/scripts/train_pytorch.py` |
| LIBERO evaluator | `openpi/examples/libero/main.py` |
| Mamba3 implementation | `mamba/mamba_ssm/modules/mamba3.py` |
| SmolVLA transfer patch | `integrations/smolvla/mambaact.patch` |

## License

This repository is released under Apache 2.0. See [NOTICE.md](NOTICE.md) for
upstream attribution.
