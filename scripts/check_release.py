#!/usr/bin/env python3
"""Check that the repository holds nothing a run reads or produces. Run it before pushing.

    python scripts/check_release.py

A file of the repository (tracked, or new and not ignored) is reported when it
  - lies in a directory of data, weights or outputs (data/, models/, runs/, ...),
  - has the type of a data, image, weight or log file,
  - is larger than 1 MB, or is not a text file,
  - contains the absolute path of a home or scratch directory,
  - contains a match of a pattern in `.release_blocklist`. That file is optional and not tracked. It holds one regular
    expression per line, for the user names, hosts and institutions that must not appear in a release.

Only file names, line numbers and patterns are printed, never the matching text.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = ("data", "models", "runs", "evals", "logs", "raw", "gen", "probes")  # at the top of the repository
SUFFIXES = (".jsonl", ".safetensors", ".bin", ".pt", ".ckpt", ".parquet", ".jpg", ".jpeg", ".png", ".dcm", ".log")
MAX_BYTES = 1_000_000
PRIVATE_PATH = r"/(?:home|Users|scratch)/\w"


def repository_files(root: Path) -> list[str]:
    cmd = ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [name for name in out.split("\0") if name and (root / name).is_file()]


def blocklist(root: Path) -> list[str]:
    path = root / ".release_blocklist"
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text().splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def problems(root: Path, files: list[str], patterns: list[str]) -> list[str]:
    patterns = [re.compile(p) for p in [PRIVATE_PATH, *patterns]]
    found = []
    for name in files:
        path = root / name
        if name.split("/")[0] in DIRS:
            found.append(f"{name}: in a directory of data or outputs")
        if path.suffix.lower() in SUFFIXES:
            found.append(f"{name}: a {path.suffix} file")
        if path.stat().st_size > MAX_BYTES:
            found.append(f"{name}: larger than {MAX_BYTES // 1_000_000} MB")
        try:
            lines = path.read_text().splitlines()
        except UnicodeDecodeError:
            found.append(f"{name}: not a text file")
            continue
        for number, line in enumerate(lines, 1):
            for pattern in patterns:
                if pattern.search(line):
                    found.append(f"{name}:{number}: matches {pattern.pattern!r}")
    return found


def main() -> int:
    found = problems(ROOT, repository_files(ROOT), blocklist(ROOT))
    for line in found:
        print(line)
    if found:
        print(f"{len(found)} problem(s): none of this belongs in a public repository")
        return 1
    print("release check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
