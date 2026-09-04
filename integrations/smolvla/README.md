# SmolVLA transfer experiment

Table VIII uses LeRobot commit
`e40b58a8dfa9e7b86918c374791599d070518d11` and the official
`lerobot/smolvla_base` initializer. The patch adds the optional four-block,
width-768 MambaAct expert without changing the default SmolVLA path.

```bash
git clone https://github.com/huggingface/lerobot.git third_party/lerobot
git -C third_party/lerobot checkout e40b58a8dfa9e7b86918c374791599d070518d11
export MAMBAACT_LEROBOT_ROOT="$PWD/third_party/lerobot"
bash scripts/apply_smolvla_patch.sh
```

Install LeRobot following its instructions for the pinned revision, then run:

```bash
bash scripts/train_smolvla.sh baseline
bash scripts/train_smolvla.sh mambaact
```

Both runs train for 30,000 steps with batch size 16, training seed 42, and
save only the final checkpoint. Evaluation uses 50 trials per task and seed 7.
