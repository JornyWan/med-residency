"""PMC-VQA, MedXpertQA and MMMU -> the files of the `med_vqa` task (multiple-choice questions answered with reasoning).

  train.jsonl                 --n-train questions of PMC-VQA train_2.csv: prompt, image, teacher_tag, solution
  test_pmc_vqa.jsonl          --n-test questions of PMC-VQA test_2.csv
  test_medxpertqa_mm.jsonl    MedXpertQA-MM test, the questions with a single image
  test_mmmu_health.jsonl      MMMU validation, the five health subjects, single-image multiple choice
  img/                        the images, longest side at most 896 pixels

Every prompt asks for step-by-step reasoning and the letter of the answer inside <answer></answer>. The teacher's
answers to train.jsonl (offline KD, replay) come from `medres eval generate` + `medres data teacher-outputs`.
All three sources are public (docs/data.md has the download commands).
"""
from __future__ import annotations

import argparse
import ast
import csv
import glob
import io
import os
import random

from medresidency import config, jsonl

INSTRUCTION = "{q}\nReason step by step, then give the letter of the correct option inside <answer></answer> tags."
MAX_SIDE = 896
MMMU_HEALTH = ("Basic_Medical_Science", "Clinical_Medicine", "Diagnostics_and_Laboratory_Medicine", "Pharmacy",
               "Public_Health")


def user_turn(question: str) -> str:
    return "<image>" + INSTRUCTION.format(q=question)


def pmc_question(r: dict) -> str:
    """PMC-VQA writes its options as "A: text"; shown to the model as "A) text" """
    options = "\n".join(r[f"Choice {k}"].strip().replace(f"{k}:", f"{k})", 1) for k in "ABCD")
    return f"{r['Question'].strip()}\n{options}"


def mmmu_question(question: str, options: list[str]) -> str:
    return (question.replace("<image 1>", "the image") + "\nAnswer Choices:\n"
            + "\n".join(f"({k}) {o}" for k, o in zip("ABCDE", options)))


def mmmu_usable(r: dict, options: list) -> bool:
    return (r["question_type"] == "multiple-choice" and r["image_1"] is not None and r["image_2"] is None
            and len(options) <= 5)


class Images:
    def __init__(self, out: str):
        self.dir = f"{out}/img"
        os.makedirs(self.dir, exist_ok=True)

    def save(self, src, name: str) -> str:
        """`src` is a path or the bytes of an image; returns the absolute path of the down-sized copy"""
        from PIL import Image
        dst = os.path.abspath(f"{self.dir}/{name}.jpg")
        if not os.path.exists(dst):
            img = Image.open(io.BytesIO(src) if isinstance(src, bytes) else src).convert("RGB")
            img.thumbnail((MAX_SIDE, MAX_SIDE))
            img.save(dst, quality=95)
        return dst


def pmc_items(root: str, file: str, n: int, seed: int, images: Images) -> list[dict]:
    with open(f"{root}/{file}") as f:
        rows = list(csv.DictReader(f))
    random.Random(seed).shuffle(rows)
    return [dict(image=images.save(f"{root}/figures/{r['Figure_path']}", "pmc_" + os.path.splitext(r["Figure_path"])[0]),
                 question=pmc_question(r), answer=r["Answer"].strip()) for r in rows[:n]]


def medxpert_items(root: str, images: Images) -> list[dict]:
    img_dir = next(d for d in (f"{root}/images/images", f"{root}/images") if glob.glob(f"{d}/*.*"))
    items = []
    for r in jsonl.read(f"{root}/MM/test.jsonl"):
        if len(r["images"]) == 1:
            name = r["images"][0]
            items.append(dict(image=images.save(f"{img_dir}/{name}", "medxpert_" + os.path.splitext(name)[0]),
                              question=r["question"], answer=r["label"]))
    return items


def mmmu_items(root: str, images: Images) -> list[dict]:
    import pandas as pd
    items = []
    for subject in MMMU_HEALTH:
        for path in sorted(glob.glob(f"{root}/{subject}/validation-*.parquet")):
            for _, r in pd.read_parquet(path).iterrows():
                options = ast.literal_eval(r["options"])
                if mmmu_usable(r, options):
                    items.append(dict(image=images.save(r["image_1"]["bytes"], "mmmu_" + r["id"]),
                                      question=mmmu_question(r["question"], options), answer=r["answer"]))
    return items


def test_rows(domain: str, items: list[dict]) -> list[dict]:
    return [jsonl.user_row(user_turn(x["question"]), x["image"], id=f"{domain}/{i}", domain=domain, answer=x["answer"],
                           parser="letter") for i, x in enumerate(items)]


def main():
    p = argparse.ArgumentParser(prog="medres data med-vqa", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pmc-vqa", help="directory with train_2.csv, test_2.csv and figures/")
    p.add_argument("--medxpertqa", help="directory with MM/test.jsonl and images/")
    p.add_argument("--mmmu", help="directory with <subject>/validation-*.parquet")
    p.add_argument("--out", default=config.data_path("med_vqa"))
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--teacher-tag", default="medix", help="the routing tag of the task's teacher")
    a = p.parse_args()
    if not (a.pmc_vqa or a.medxpertqa or a.mmmu):
        p.error("give at least one source")
    images = Images(a.out)
    if a.pmc_vqa:
        train = pmc_items(a.pmc_vqa, "train_2.csv", a.n_train, 0, images)
        rows = [jsonl.user_row(user_turn(x["question"]), x["image"], teacher_tag=a.teacher_tag,
                               solution=f"<answer>{x['answer']}</answer>") for x in train]
        print("train.jsonl", jsonl.write(f"{a.out}/train.jsonl", rows))
        test = pmc_items(a.pmc_vqa, "test_2.csv", a.n_test, 1, images)
        print("test_pmc_vqa.jsonl", jsonl.write(f"{a.out}/test_pmc_vqa.jsonl", test_rows("pmc_vqa", test)))
    if a.medxpertqa:
        rows = test_rows("medxpertqa_mm", medxpert_items(a.medxpertqa, images))
        print("test_medxpertqa_mm.jsonl", jsonl.write(f"{a.out}/test_medxpertqa_mm.jsonl", rows))
    if a.mmmu:
        rows = test_rows("mmmu_health", mmmu_items(a.mmmu, images))
        print("test_mmmu_health.jsonl", jsonl.write(f"{a.out}/test_mmmu_health.jsonl", rows))


if __name__ == "__main__":
    main()
