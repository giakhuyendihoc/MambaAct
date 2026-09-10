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

The baseline is the released `lerobot/smolvla_libero` checkpoint at revision
`31d453f7edd78c839a8bbc39744a292686daf0de`, evaluated without further training.
SmolVLA-MambaAct starts from `lerobot/smolvla_base` and matches the release's
documented budget and recipe: 25,000 steps, batch 32, and training seed 1000.
Both evaluations use 50 trials/task, seed 7, four suites (2,000 trials),
two cameras, horizon 50, execute 10 actions before replanning, and ten denoising
steps. Simulator images are 360x360 and resized with padding to 512x512.

| Model | Spatial | Object | Goal | L-10 | Avg. | Prefix (ms) | Denoise (ms) | Model-core query (ms) | Expert core (M) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Released SmolVLA 25k | 76.6 | 93.0 | 85.4 | 70.2 | 81.30 | 25.53 | 163.15 | 188.69 | 98.25 |
| SmolVLA-MambaAct 25k | 82.6 | 92.4 | 87.0 | 77.8 | 84.95 | 30.69 | 59.89 | 90.61 | 64.31 |

Success totals are 1,626/2,000 and 1,699/2,000: +3.65 percentage points overall
and +7.6 points on LIBERO-10. Model-core query latency decreases by 52.0%.
Expert core counts the native `lm_expert` (98,245,840 parameters) or the four
Mamba action blocks including gates and FFNs (64,306,784). The complete Mamba
replacement, including context processing, input/output projections, and final
normalization, contains 70,876,208 parameters. Shared policy wrappers are excluded.

Latency uses the same H100 setup and the two-camera profiler, with 3 warmups and
10 measured synthetic-batch queries. Prefix/context includes image/language
embedding, the VLM, and context or KV-cache setup. Denoising includes all ten
action-expert updates. The complete model-core timer excludes image resizing,
tokenization, input transfer, noise sampling, networking, simulator rendering,
and environment stepping. Component medians need not sum to the total median.

This is a released-checkpoint comparison, not two models retrained in the same
codebase: the release does not pin its dataset or implementation revision.
It is also not a reproduction of the separate published 87.3% reference.
See [the exact protocol](../integrations/smolvla/README.md) and
[machine-readable task outcomes and timing samples](../results/smolvla_25k.json).
