"""Mix the training prompts of several tasks -> the files of a multi-teacher task such as `cxr_report_vqa`.

Each source directory holds `train.jsonl` (prompts tagged with their teacher) and `train_teacher.jsonl` (that
teacher's outputs). Only prompts with a teacher output are drawn, so that every method (OPD, GRPO, offline KD)
trains on the same prompts: --n per source, shuffled together.

  train.jsonl          the mixed prompts, each with the `teacher_tag` of its source
  train_teacher.jsonl  the teachers' outputs for the same prompts, in the same order

A mix that contains MIMIC-CXR rows is credentialed data, like its source.
"""
from __future__ import annotations

import argparse
import random
from collections import Counter

from medresidency import config, jsonl


def key(row: dict):
    return row["messages"][0]["content"], tuple(row["images"])


def pick(prompts: list[dict], outputs: list[dict], n: int, seed: int = 0):
    """`n` prompts that have a teacher output, with those outputs"""
    out = {key(r): r for r in outputs}
    rows = [r for r in prompts if key(r) in out]
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    if len(rows) < n:
        raise SystemExit(f"error: only {len(rows)} prompts with a teacher output, {n} asked for")
    return rows, [out[key(r)] for r in rows]


def mix(sources: list[tuple[list[dict], list[dict]]], n: int):
    prompts, outputs = [], []
    for p, o in sources:
        rows, outs = pick(p, o, n)
        prompts += rows
        outputs += outs
    order = list(range(len(prompts)))
    random.Random(1).shuffle(order)
    return [prompts[i] for i in order], [outputs[i] for i in order]


def main():
    p = argparse.ArgumentParser(prog="medres data mixed", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", default=[config.data_path("med_vqa"), config.data_path("cxr_report")],
                   help="task data directories, each with train.jsonl and train_teacher.jsonl")
    p.add_argument("--n", type=int, default=10000, help="prompts per source")
    p.add_argument("--out", default=config.data_path("cxr_report_vqa"))
    a = p.parse_args()
    sources = [(jsonl.read(f"{d}/train.jsonl"), jsonl.read(f"{d}/train_teacher.jsonl")) for d in a.sources]
    prompts, outputs = mix(sources, a.n)
    print("train.jsonl", jsonl.write(f"{a.out}/train.jsonl", prompts),
          "| train_teacher.jsonl", jsonl.write(f"{a.out}/train_teacher.jsonl", outputs),
          "| teachers", dict(Counter(r["teacher_tag"] for r in prompts)))


if __name__ == "__main__":
    main()
