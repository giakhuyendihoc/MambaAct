#!/usr/bin/env python3
"""Download pinned inputs for the SmolVLA released-checkpoint comparison."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

RELEASE_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
BASE_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
DATASET_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("released", "training"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    specs = [("released", "lerobot/smolvla_libero", RELEASE_REVISION, "model")]
    if args.mode == "training":
        specs = [("base", "lerobot/smolvla_base", BASE_REVISION, "model"),
                 ("dataset", "lerobot/libero", DATASET_REVISION, "dataset")]
    for label, repo, revision, kind in specs:
        destination = args.output_root / label / revision
        snapshot_download(repo_id=repo, repo_type=kind, revision=revision,
                          local_dir=destination, max_workers=4)
        required = "model.safetensors" if kind == "model" else "meta/info.json"
        if not (destination / required).is_file():
            raise RuntimeError(f"Incomplete download: {repo}@{revision}")
        print(f"Ready: {repo}@{revision}")


if __name__ == "__main__":
    main()
