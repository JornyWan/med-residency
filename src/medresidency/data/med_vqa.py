"""PMC-VQA, MedXpertQA and MMMU -> the files of the `med_vqa` task (multiple-choice questions answered with reasoning).

  train.jsonl                 --n-train questions of PMC-VQA train_2.csv: prompt, image, teacher_tag, solution
  val.jsonl                   --n-val further questions of train_2.csv, on figures no training question uses
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
MMMU_HEALTH = (
    "Basic_Medical_Science",
    "Clinical_Medicine",
    "Diagnostics_and_Laboratory_Medicine",
    "Pharmacy",
    "Public_Health",
)


def user_turn(question: str) -> str:
    return "<image>" + INSTRUCTION.format(q=question)


def pmc_question(r: dict) -> str:
    """The question with its options. PMC-VQA writes them as "A: text"; the model sees "A) text"."""
    options = "\n".join(r[f"Choice {k}"].strip().replace(f"{k}:", f"{k})", 1) for k in "ABCD")
    return f"{r['Question'].strip()}\n{options}"


def mmmu_question(question: str, options: list[str]) -> str:
    choices = "\n".join(f"({k}) {o}" for k, o in zip("ABCDE", options))
    return question.replace("<image 1>", "the image") + "\nAnswer Choices:\n" + choices


def mmmu_usable(r: dict, options: list) -> bool:
    single_image = r["image_1"] is not None and r["image_2"] is None
    return r["question_type"] == "multiple-choice" and single_image and len(options) <= 5


class Images:
    def __init__(self, out: str):
        self.dir = f"{out}/img"
        os.makedirs(self.dir, exist_ok=True)

    def save(self, src, name: str) -> str:
        """Save a down-sized copy of an image (a path or its bytes) and return the absolute path of the copy."""
        from PIL import Image

        dst = os.path.abspath(f"{self.dir}/{name}.jpg")
        if not os.path.exists(dst):
            img = Image.open(io.BytesIO(src) if isinstance(src, bytes) else src).convert("RGB")
            img.thumbnail((MAX_SIDE, MAX_SIDE))
            img.save(dst, quality=95)
        return dst


def pmc_rows(root: str, file: str, seed: int) -> list[dict]:
    """The questions of a PMC-VQA file in the (fixed) order they are drawn in."""
    with open(f"{root}/{file}") as f:
        rows = list(csv.DictReader(f))
    random.Random(seed).shuffle(rows)
    return rows


def held_out(rows: list[dict], n_train: int, n_val: int) -> list[dict]:
    """Validation questions, drawn from the rows after the `n_train` training questions.

    No training question uses their figure, and each figure gives one question."""
    used = {r["Figure_path"] for r in rows[:n_train]}
    val = []
    for r in rows[n_train:]:
        if len(val) < n_val and r["Figure_path"] not in used:
            used.add(r["Figure_path"])
            val.append(r)
    return val


def pmc_items(root: str, rows: list[dict], images: Images) -> list[dict]:
    items = []
    for r in rows:
        figure = r["Figure_path"]
        image = images.save(f"{root}/figures/{figure}", "pmc_" + os.path.splitext(figure)[0])
        items.append({"image": image, "question": pmc_question(r), "answer": r["Answer"].strip()})
    return items


def medxpert_items(root: str, images: Images) -> list[dict]:
    img_dir = next(d for d in (f"{root}/images/images", f"{root}/images") if glob.glob(f"{d}/*.*"))
    items = []
    for r in jsonl.read(f"{root}/MM/test.jsonl"):
        if len(r["images"]) != 1:
            continue
        name = r["images"][0]
        image = images.save(f"{img_dir}/{name}", "medxpert_" + os.path.splitext(name)[0])
        items.append({"image": image, "question": r["question"], "answer": r["label"]})
    return items


def mmmu_items(root: str, images: Images) -> list[dict]:
    import pandas as pd

    items = []
    for subject in MMMU_HEALTH:
        for path in sorted(glob.glob(f"{root}/{subject}/validation-*.parquet")):
            for _, r in pd.read_parquet(path).iterrows():
                options = ast.literal_eval(r["options"])
                if not mmmu_usable(r, options):
                    continue
                image = images.save(r["image_1"]["bytes"], "mmmu_" + r["id"])
                question = mmmu_question(r["question"], options)
                items.append({"image": image, "question": question, "answer": r["answer"]})
    return items


def test_rows(domain: str, items: list[dict]) -> list[dict]:
    rows = []
    for i, x in enumerate(items):
        fields = {"id": f"{domain}/{i}", "domain": domain, "answer": x["answer"], "parser": "letter"}
        rows.append(jsonl.user_row(user_turn(x["question"]), x["image"], **fields))
    return rows


def main():
    p = argparse.ArgumentParser(
        prog="medres data med-vqa", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pmc-vqa", help="directory with train_2.csv, test_2.csv and figures/")
    p.add_argument("--medxpertqa", help="directory with MM/test.jsonl and images/")
    p.add_argument("--mmmu", help="directory with <subject>/validation-*.parquet")
    p.add_argument("--out", default=config.data_path("med_vqa"))
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--teacher-tag", default="medix", help="the routing tag of the task's teacher")
    a = p.parse_args()
    if not (a.pmc_vqa or a.medxpertqa or a.mmmu):
        p.error("give at least one source")
    images = Images(a.out)
    if a.pmc_vqa:
        pool = pmc_rows(a.pmc_vqa, "train_2.csv", 0)
        train = pmc_items(a.pmc_vqa, pool[: a.n_train], images)
        rows = []
        for x in train:
            fields = {"teacher_tag": a.teacher_tag, "solution": f"<answer>{x['answer']}</answer>"}
            rows.append(jsonl.user_row(user_turn(x["question"]), x["image"], **fields))
        print("train.jsonl", jsonl.write(f"{a.out}/train.jsonl", rows))
        val = pmc_items(a.pmc_vqa, held_out(pool, a.n_train, a.n_val), images)
        print("val.jsonl", jsonl.write(f"{a.out}/val.jsonl", test_rows("pmc_val", val)))
        test = pmc_items(a.pmc_vqa, pmc_rows(a.pmc_vqa, "test_2.csv", 1)[: a.n_test], images)
        print("test_pmc_vqa.jsonl", jsonl.write(f"{a.out}/test_pmc_vqa.jsonl", test_rows("pmc_vqa", test)))
    if a.medxpertqa:
        rows = test_rows("medxpertqa_mm", medxpert_items(a.medxpertqa, images))
        print("test_medxpertqa_mm.jsonl", jsonl.write(f"{a.out}/test_medxpertqa_mm.jsonl", rows))
    if a.mmmu:
        rows = test_rows("mmmu_health", mmmu_items(a.mmmu, images))
        print("test_mmmu_health.jsonl", jsonl.write(f"{a.out}/test_mmmu_health.jsonl", rows))


if __name__ == "__main__":
    main()
