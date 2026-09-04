# Attribution

MambaAct builds on the following Apache-2.0 projects:

- [OpenPI](https://github.com/Physical-Intelligence/openpi), used as the
  PI0.5 training, policy, and evaluation framework.
- [Mamba](https://github.com/state-spaces/mamba), including the Mamba3
  implementation vendored in `mamba/`.
- [LeRobot](https://github.com/huggingface/lerobot), used for the SmolVLA
  transfer experiment through the patch in `integrations/smolvla/`.

The `openpi/` and `mamba/` directories retain their upstream license and
copyright files. Modifications implement MambaAct, its context compressor,
conditioning interfaces, training configurations, and experiment tooling.
