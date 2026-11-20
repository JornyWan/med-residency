"""JSON-lines files: every data, generation and score file of the framework has one JSON object per line."""
from __future__ import annotations

import json
import os


def read(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path, rows) -> int:
    """write through a temporary file, so that a reader never sees half a file; returns the number of rows"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(f"{path}.tmp", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    os.replace(f"{path}.tmp", path)
    return n


def user_row(text: str, image: str, **extra) -> dict:
    """one prompt in ms-swift's format: a user turn with `<image>` where the image goes, and the image path"""
    return {"messages": [{"role": "user", "content": text}], "images": [image], **extra}


def answered_row(row: dict, answer: str) -> dict:
    """the prompt of `row` with an assistant turn: a supervised example"""
    return {"messages": [row["messages"][0], {"role": "assistant", "content": answer}], "images": row["images"]}
