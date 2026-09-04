#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${MAMBAACT_LEROBOT_ROOT:?Set MAMBAACT_LEROBOT_ROOT to a LeRobot checkout}"
EXPECTED_COMMIT="e40b58a8dfa9e7b86918c374791599d070518d11"

ACTUAL_COMMIT="$(git -C "$LEROBOT_ROOT" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "LeRobot must be checked out at $EXPECTED_COMMIT; found $ACTUAL_COMMIT" >&2
  exit 1
fi

git -C "$LEROBOT_ROOT" apply --check "$ROOT/integrations/smolvla/mambaact.patch"
git -C "$LEROBOT_ROOT" apply "$ROOT/integrations/smolvla/mambaact.patch"
echo "Applied the MambaAct SmolVLA integration."
