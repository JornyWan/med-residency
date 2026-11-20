"""OmniMedVQA with the Med-R1 modality splits -> the files of the `modality_experts` task.

  train.jsonl   every training question, tagged with the expert of its modality (teacher_tag) and with its solution
  test.jsonl    every test question, with its modality as `domain` and the letter of its answer
  img/          the referenced images at 384 x 384 (Med-R1's protocol), same relative paths as in OmniMedVQA

The prompts are Med-R1's own: its GRPO training template for train.jsonl and its evaluation template for test.jsonl.
Sources (docs/data.md): the `Splits/modality` directory of the Med-R1 repository and the open-access part of
OmniMedVQA. The experts' outputs for offline KD come from `medres eval generate --tag <modality>` and
`medres data teacher-outputs`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor

from medresidency import config, jsonl

SIZE = 384
TRAIN_TEMPLATE = (
    "{Question} Your task: 1. Think through the question step by step, enclose your reasoning process in "
    "<think>...</think> tags. 2. Then provide the correct single-letter choice (A, B, C, D,...) inside "
    "<answer>...</answer> tags. 3. No extra information or text outside of these tags."
)
TEST_TEMPLATE = (
    "{Question} Think through the question step by step in <think>...</think> tags. "
    "Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags."
)
ANSWER = re.compile(r"<answer>\s*(\w+)\s*</answer>")
# file name of the split (as spelled in the Med-R1 repository) -> routing tag of the expert
TAG = {
    "CT(Computed Tomography)": "CT",
    "Dermoscopy": "Dermo",
    "Fundus Photography": "Fundus",
    "MR (Mag-netic Resonance Imaging)": "MRI",
    "Microscopy Images": "Micro",
    "OCT (Optical Coherence Tomography": "OCT",
    "X-Ray": "X-Ray",
    "ultrasound": "US",
}


def split_files(directory: str) -> list[tuple[str, str]]:
    """The (tag, file) pairs of the modality splits in `directory`."""
    files = [(TAG.get(os.path.basename(f).rsplit("_", 1)[0]), f) for f in sorted(glob.glob(f"{directory}/*.json"))]
    return [(tag, f) for tag, f in files if tag]


def train_row(x: dict, tag: str, image: str) -> dict:
    text = "<image>" + TRAIN_TEMPLATE.format(Question=x["problem"])
    return jsonl.user_row(text, image, solution=x["solution"], teacher_tag=tag)


def test_row(x: dict, tag: str, i: int, image: str) -> dict:
    text = "<image>" + TEST_TEMPLATE.format(Question=x["problem"])
    answer = ANSWER.search(x["solution"]).group(1)
    return jsonl.user_row(text, image, id=f"{tag}/{i}", domain=tag, answer=answer, parser="answer_tag")


def resize(job) -> bool:
    src, dst = job
    if os.path.exists(dst):
        return True
    if not os.path.exists(src):
        return False
    from PIL import Image

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    Image.open(src).convert("RGB").resize((SIZE, SIZE)).save(dst)
    return True


def main():
    p = argparse.ArgumentParser(
        prog="medres data modality-experts", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--splits", required=True, help="<Med-R1 repository>/Splits/modality (holds train/ and test/)")
    p.add_argument(
        "--images", required=True, help="OmniMedVQA root; the `image` paths of the splits are relative to it"
    )
    p.add_argument("--out", default=config.data_path("modality_experts"))
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    img = os.path.abspath(f"{a.out}/img")

    splits = {}
    for part in ("train", "test"):
        splits[part] = []
        for tag, f in split_files(f"{a.splits}/{part}"):
            with open(f) as fh:
                splits[part].append((tag, json.load(fh)))
    rels = sorted({x["image"] for part in splits.values() for _, xs in part for x in xs})
    with ProcessPoolExecutor(a.workers) as ex:
        ok = list(ex.map(resize, [(os.path.join(a.images, r), os.path.join(img, r)) for r in rels], chunksize=256))
    missing = {r for r, k in zip(rels, ok) if not k}
    print(f"{len(rels)} images, {len(missing)} missing from {a.images}")

    train = [
        train_row(x, tag, os.path.join(img, x["image"]))
        for tag, xs in splits["train"]
        for x in xs
        if x["image"] not in missing
    ]
    random.Random(0).shuffle(train)
    test = [
        test_row(x, tag, i, os.path.join(img, x["image"]))
        for tag, xs in splits["test"]
        for i, x in enumerate(xs)
        if x["image"] not in missing
    ]
    print("train.jsonl", jsonl.write(f"{a.out}/train.jsonl", train))
    print("test.jsonl", jsonl.write(f"{a.out}/test.jsonl", test))
    for name, rows, k in (("train", train, "teacher_tag"), ("test", test, "domain")):
        print(f"  {name}:", {tag: sum(r[k] == tag for r in rows) for tag in TAG.values()})


if __name__ == "__main__":
    main()
