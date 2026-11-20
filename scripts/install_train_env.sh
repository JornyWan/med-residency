#!/usr/bin/env bash
# The training environment: ms-swift (trainers, teacher servers) + vLLM (rollouts, generation) + MedResidency.
#
#   bash scripts/install_train_env.sh /path/to/envs/medres        # creates the conda environment there
#   bash scripts/install_train_env.sh --patch-only                # only patch the ms-swift of the active environment
#
# The versions are the ones every number in benchmarks/README.md was produced with. Two small patches are applied to
# the installed ms-swift (each is idempotent and marked with "medresidency patch" in the patched file):
#
#   1. swift/pipelines/infer/deploy.py   On-policy distillation asks the teacher server to score the student's response,
#      which is sent as raw token ids. The server's media pre-processing assumes every content item is a dict and
#      answers HTTP 500; the patch skips items that are not.
#   2. swift/infer_engine/vllm_engine.py  At temperature 1 a student occasionally samples a vision placeholder token
#      (<|image_pad|> ...), and the next forward pass fails with more image placeholders than image features. The
#      patch passes the tokens listed in $MEDRES_BAD_WORDS (set by the launcher from `train.banned_tokens`) to vLLM as
#      `bad_words`.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "${1:-}" = "--patch-only" ]; then
    PY=python
else
    ENV="${1:?usage: install_train_env.sh <environment directory> | --patch-only}"
    conda create -y -p "$ENV" python=3.12
    PY="$ENV/bin/python"
    "$PY" -m pip install -U pip
    "$PY" -m pip install "vllm==0.23.0" "ms-swift==4.6.0" deepspeed qwen-vl-utils
    "$PY" -m pip install -e "$REPO"
fi

SWIFT="$("$PY" -c 'import os, swift; print(os.path.dirname(swift.__file__))')"

"$PY" - "$SWIFT" <<'EOF'
import sys
from pathlib import Path

MARK = "medresidency patch"


def insert_before(path, anchor, lines):
    """Insert `lines` before the one line of `path` that reads `anchor`, at that line's indentation."""
    src = path.read_text()
    if MARK in src:
        return
    hits = [line for line in src.splitlines(keepends=True) if line.strip() == anchor]
    if len(hits) != 1:
        sys.exit(f"the patch did not apply to {path} (another ms-swift version?)")
    indent = hits[0][: -len(hits[0].lstrip())]
    path.write_text(src.replace(hits[0], "".join(indent + line + "\n" for line in lines) + hits[0]))


swift = Path(sys.argv[1])
insert_before(
    swift / "pipelines/infer/deploy.py",
    "key = item.get('type', '')",
    [
        f"if not isinstance(item, dict):  # {MARK}: assistant content may be raw token ids (teacher scoring)",
        "    continue",
    ],
)
insert_before(
    swift / "infer_engine/vllm_engine.py",
    "res = SamplingParams(**kwargs)",
    [
        f"if os.environ.get('MEDRES_BAD_WORDS'):  # {MARK}: never sample these tokens",
        "    kwargs['bad_words'] = os.environ['MEDRES_BAD_WORDS'].split(',')",
    ],
)
EOF

"$PY" - <<'EOF'
from importlib.metadata import PackageNotFoundError, version
for name in ("torch", "transformers", "vllm", "ms-swift", "trl", "deepspeed", "peft", "accelerate", "qwen-vl-utils",
             "datasets", "medresidency"):
    try:
        print(f"{name:14s} {version(name)}")
    except PackageNotFoundError:
        print(f"{name:14s} not installed")
EOF
echo "ms-swift patched in $SWIFT"
