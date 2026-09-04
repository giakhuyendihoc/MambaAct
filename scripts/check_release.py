#!/usr/bin/env python3

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PATTERNS = (
    re.compile(r"/dtu/"),
    re.compile(r"/zhome/"),
    re.compile(r"/mnt/"),
    re.compile(r"/home/[A-Za-z]"),
    re.compile(r"/Users/"),
    re.compile(r"hpclogin", re.IGNORECASE),
    re.compile(r"hpccluster", re.IGNORECASE),
    re.compile(r"pi-mamba_project"),
    re.compile(r"LSB_JOB"),
)
REQUIRED_RESULT_ROWS = (
    "| PI0.5 | 97.6 | 98.8 | 96.6 | 93.2 | 96.5 | 3.62 | 40.76 | 160.26 | 228.6 |",
    "| **MambaAct** | **99.2** | **99.8** | **98.8** | **95.8** | **98.4** | **2.96** | 46.61 | **56.40** | **107.6** |",
)


def text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            yield path, path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue


def main() -> None:
    failures = []
    for path, content in text_files():
        for pattern in PRIVATE_PATTERNS:
            if pattern.search(content):
                failures.append(f"private pattern {pattern.pattern!r} in {path.relative_to(ROOT)}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for row in REQUIRED_RESULT_ROWS:
        if row not in readme:
            failures.append(f"README is missing paper result row: {row}")

    if failures:
        raise SystemExit("\n".join(failures))
    print("Release metadata and privacy checks passed.")


if __name__ == "__main__":
    main()
