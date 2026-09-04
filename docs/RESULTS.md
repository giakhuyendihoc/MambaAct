# Paper Results

This document mirrors the values reported in the paper. Success rates are in
percent. Prefix and denoising values are component timings; E2E is the complete
policy-query latency under the definition given for each table.

## LIBERO

Fifty trials per task, evaluation seed 7, one NVIDIA H100, action horizon 10,
and ten flow-matching denoising steps.

| Model | Spatial | Object | Goal | L-10 | Avg. | Model size (B) | Prefix (ms) | Denoise (ms) | E2E (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PI0.5 | 97.6 | 98.8 | 96.6 | 93.2 | 96.5 | 3.62 | 40.76 | 160.26 | 228.6 |
| MambaAct | 99.2 | 99.8 | 98.8 | 95.8 | 98.4 | 2.96 | 46.61 | 56.40 | 107.6 |

The action-expert core decreases from 311.5M to 29.7M parameters. Denoising
latency decreases by 64.8%, and E2E latency decreases by 52.9%.

## LIBERO-Plus

Models are trained only on standard LIBERO and evaluated without
LIBERO-Plus fine-tuning. The two rows below use the matched training and
evaluation protocol from the paper.

| Model | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PI0.5 | 61.6 | 70.9 | 76.4 | 94.9 | 92.7 | 77.4 | 86.4 | 78.7 |
| MambaAct | 61.5 | 77.6 | 79.8 | 95.2 | 93.2 | 73.5 | 86.4 | 79.7 |

## Real-World Tasks

Training uses 112 demonstrations per task. Evaluation contains 28
in-distribution and 52 out-of-distribution trials. Inference runs on one
NVIDIA RTX 5090 with horizon 50; the first 30 actions are executed at 30 Hz
before replanning.

| Task | Model | Size (B) | Train (h) | GFLOPs/query | Prefix (ms) | Denoise (ms) | E2E (ms) | ID | OOD | Final |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Nut pouring | PI0.5 | 3.62 | 14.3 | 5001.2 | 39.45 | 74.82 | 125.49 | 100.0 | 86.5 | 91.3 |
| Nut pouring | MambaAct | 2.96 | 14.0 | 4488.8 | 41.03 | 30.84 | 82.97 | 100.0 | 95.2 | 96.9 |
| Tube placing | PI0.5 | 3.62 | 14.3 | 5001.2 | 39.12 | 75.42 | 124.84 | 100.0 | 76.9 | 85.0 |
| Tube placing | MambaAct | 2.96 | 14.0 | 4488.8 | 41.15 | 31.11 | 83.26 | 100.0 | 82.7 | 88.8 |

## Mixer and Conditioning Ablation

All models use the same Context Compressor. Size is the action-expert core.

| Mixer | Conditioning | Spatial | Object | Goal | L-10 | Avg. | Size (B) | Train (h) | E2E (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mamba | Gate | 99.2 | 99.8 | 98.8 | 95.8 | 98.4 | 0.030 | 13.78 | 107.61 |
| Mamba | Cross-Attn | 98.6 | 99.4 | 94.6 | 94.2 | 96.7 | 0.017 | 13.83 | 119.66 |
| Transformer | Gate | 82.8 | 90.6 | 86.8 | 82.6 | 85.7 | 0.030 | 13.69 | 74.22 |
| Transformer | Cross-Attn | 75.2 | 82.8 | 75.8 | 68.6 | 75.6 | 0.018 | 13.72 | 85.60 |

## VLM-Depth Composition

### LIBERO

| Model | Spatial | Object | Goal | L-10 | Avg. | Train (h) | Prefix (ms) | Denoise (ms) | E2E (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MambaAct | 99.2 | 99.8 | 98.8 | 95.8 | 98.4 | 13.78 | 46.61 | 56.40 | 107.6 |
| MambaAct L12 | 98.0 | 99.8 | 98.0 | 94.0 | 97.5 | 10.81 | 41.15 | 56.06 | 102.91 |

### Real world

| Model | Params (B) | Train (h) | GFLOPs/query | Prefix (ms) | Denoise (ms) | E2E (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MambaAct | 2.96 | 14.00 | 4488.84 | 41.15 | 31.11 | 83.26 |
| MambaAct L12 | 2.30 | 13.55 | 3209.70 | 32.05 | 31.03 | 74.19 |

| Task | Model | ID | OOD | Final |
| --- | --- | ---: | ---: | ---: |
| Nut pouring | MambaAct | 100.0 | 95.20 | 96.90 |
| Nut pouring | MambaAct L12 | 100.0 | 92.31 | 95.00 |
| Tube placing | MambaAct | 100.0 | 82.70 | 88.80 |
| Tube placing | MambaAct L12 | 100.0 | 84.62 | 90.00 |

## SmolVLA Transfer

All measurements use the same H100 setup, horizon 10, and ten denoising steps.
These E2E values exclude simulator rendering, networking, and environment
stepping.

| Model | Spatial | Object | Goal | L-10 | Avg. | Prefix (ms) | Denoise (ms) | E2E/query (ms) | Action expert (M) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SmolVLA baseline | 74.0 | 94.4 | 86.6 | 55.6 | 77.7 | 25.44 | 162.02 | 187.47 | 98.25 |
| SmolVLA-Mamba | 87.4 | 92.6 | 81.8 | 75.2 | 84.3 | 31.00 | 59.53 | 90.59 | 64.31 |

The SmolVLA baseline is the matched reimplementation used by the transfer
ablation and differs from the published SmolVLA reference row discussed in
the paper's cross-method comparison.
