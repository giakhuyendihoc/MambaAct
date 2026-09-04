#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${1:?Usage: scripts/eval_libero.sh MODEL CHECKPOINT_DIR}"
CHECKPOINT_DIR="${2:?Usage: scripts/eval_libero.sh MODEL CHECKPOINT_DIR}"
IFS=$'\t' read -r CONFIG DEFAULT_RUN MODEL_KIND < <(resolve_pi05_model "$MODEL")

LIBERO_ROOT="${MAMBAACT_LIBERO_ROOT:?Set MAMBAACT_LIBERO_ROOT to a LIBERO checkout}"
LIBERO_PYTHON="${MAMBAACT_LIBERO_PYTHON:?Set MAMBAACT_LIBERO_PYTHON to the LIBERO environment's Python executable}"
PORT="${PORT:-8000}"
TRIALS="${TRIALS:-50}"
EVAL_SEED="${EVAL_SEED:-7}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
NUM_DENOISE_STEPS="${NUM_DENOISE_STEPS:-10}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
RUN_ID="${RUN_ID:-${DEFAULT_RUN}_eval}"
REPORT_DIR="${REPORT_DIR:-$MAMBAACT_OUTPUT_ROOT/reports/libero/$RUN_ID}"
SERVER_LOG="$REPORT_DIR/server.log"

if [[ ! -s "$CHECKPOINT_DIR/model.safetensors" ]]; then
  echo "Checkpoint does not contain model.safetensors: $CHECKPOINT_DIR" >&2
  exit 1
fi
if [[ ! -x "$LIBERO_PYTHON" ]]; then
  echo "LIBERO Python is not executable: $LIBERO_PYTHON" >&2
  exit 1
fi

mkdir -p "$REPORT_DIR"

cd "$MAMBAACT_ROOT/openpi"
uv run python scripts/serve_policy_from_checkpoint.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --config-name "$CONFIG" \
  --port "$PORT" \
  --num-denoise-steps "$NUM_DENOISE_STEPS" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

SERVER_READY=0
for _ in $(seq 1 180); do
  if "$LIBERO_PYTHON" - "$PORT" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(0.2)
    raise SystemExit(sock.connect_ex(("127.0.0.1", int(sys.argv[1]))))
PY
  then
    SERVER_READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Policy server exited before becoming ready. See $SERVER_LOG" >&2
    exit 1
  fi
  sleep 1
done
if [[ "$SERVER_READY" != "1" ]]; then
  echo "Policy server did not become ready. See $SERVER_LOG" >&2
  exit 1
fi

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$LIBERO_ROOT/libero/libero}"
export PYTHONPATH="$MAMBAACT_ROOT/openpi/src:$MAMBAACT_ROOT/openpi/packages/openpi-client/src:$LIBERO_ROOT:${PYTHONPATH:-}"

for SUITE in $SUITES; do
  "$LIBERO_PYTHON" "$MAMBAACT_ROOT/openpi/examples/libero/main.py" \
    --args.host 127.0.0.1 \
    --args.port "$PORT" \
    --args.resize-size "$RESIZE_SIZE" \
    --args.replan-steps "$REPLAN_STEPS" \
    --args.history-steps 0 \
    --args.task-suite-name "$SUITE" \
    --args.num-trials-per-task "$TRIALS" \
    --args.seed "$EVAL_SEED" \
    --args.video-out-path "$REPORT_DIR/videos/$SUITE" \
    --args.metrics-out-path "$REPORT_DIR/$SUITE.json" \
    --args.no-save-videos
done

"$LIBERO_PYTHON" "$MAMBAACT_ROOT/scripts/summarize_libero.py" \
  --report-dir "$REPORT_DIR" \
  --model "$MODEL" \
  --trials "$TRIALS" \
  --seed "$EVAL_SEED"

echo "LIBERO report: $REPORT_DIR/summary.md"
