#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

cd "$ROOT/openpi"
uv sync --frozen
uv pip install -e "$ROOT/mamba" --no-build-isolation --no-deps

SITE_PACKAGES="$(uv run python -c 'import site; print(site.getsitepackages()[0])')"
cp -R src/openpi/models_pytorch/transformers_replace/. "$SITE_PACKAGES/transformers/"

echo "MambaAct environment is ready."
