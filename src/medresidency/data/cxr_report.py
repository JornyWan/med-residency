"""MIMIC-CXR -> the files of the `cxr_report` task.

One frontal image per study (PA before AP), resized once to 512 x 512, so that the student, the teacher and the
evaluation see the same pixels. The prompt is the findings + impression template of UniRG-CXR with the study's
indication and comparison as context; the reference is "Findings: ... Impression: ..." (ReXrank's format).

  train.jsonl         --n-train studies of the official train split: prompt, image, teacher_tag, solution (reference)
  train_labels.jsonl  the same studies with the reference report as the answer (label SFT)
  val.jsonl           --n-val studies of the official validate split
  test_mimic.jsonl    ReXrank's MIMIC-CXR test studies (their key image, context and reference sections)
  img/                the resized images

Studies of the ReXrank test set never enter train or validation. Sources, obtained under your own PhysioNet
credentialed access (docs/data.md): MIMIC-CXR 2.0.0 (mimic-cxr-reports.zip), MIMIC-CXR-JPG 2.0.0 (files/, the split
and metadata tables) and ReXrank's ReXRank_MIMICCXR_test.json.

MIMIC-CXR is credentialed data: everything this tool writes stays under your data-use agreement, and it prints
counts and length statistics only, never report text.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import re
import statistics
import zipfile
from concurrent.futures import ProcessPoolExecutor

from medresidency import config, jsonl
from medresidency.data.third_party import mimic_cxr_section_parser as section_parser

SIZE = 512  # UniRG-CXR's input size
PRE = "This is a radiology report generation task. Here is the context:"
POST = (
    "Given the image and the context, directly provide the report in the following format:\n"
    "Findings: [write the findings] Impression: [write the impression]\nNow write the report in the format above."
)
INDICATION = (
    "indication",
    "history",
    "reason for exam",
    "reason for examination",
    "clinical information",
    "clinical history",
    "reason for study",
)


def clean(s) -> str:
    s = re.sub(r"\s+", " ", s if isinstance(s, str) else "").strip()  # ReXrank stores missing sections as NaN
    return "" if s.lower() in ("", "nan", "none") else s


def context(indication: str, comparison: str) -> str:
    lines = [f"Indication: {indication}"] if indication else []
    lines += [f"Comparison: {comparison}"] if comparison else []
    return "\n".join(lines)


def rexrank_context(s) -> str:
    """Indication and comparison of ReXrank's context "Age: .. Gender: .. Indication: .. History: .. Comparison: ..".

    Age and gender are not in the MIMIC-CXR reports, so the training prompts cannot have them. History joins
    Indication, as in `parse`."""
    parts = re.split(r"\b(Age|Gender|Indication|History|Comparison):", s if isinstance(s, str) else "")
    d = {}
    for k, v in zip(parts[1::2], parts[2::2]):
        k = "Indication" if k == "History" else k
        if clean(v):
            d[k] = (d[k] + " " + clean(v)) if k in d else clean(v)
    return context(d.get("Indication", ""), d.get("Comparison", ""))


def user_turn(ctx: str) -> str:
    """The prompt, with `<image>` where UniRG-CXR's template puts the image: after the first sentence."""
    return PRE + "<image>" + (ctx + "\n" if ctx else "") + POST


def reference(findings: str, impression: str) -> str:
    return f"Findings: {findings} Impression: {impression}"


def parse(text: str) -> dict:
    """Sections of a MIMIC-CXR report: findings, impression, indication, comparison (those that are present)."""
    sections, names, _ = section_parser.section_text(text)
    d = {}
    for s, n in zip(sections, names):
        n = "indication" if n in INDICATION else n
        if n in ("findings", "impression", "indication", "comparison") and clean(s):
            d[n] = (d[n] + " " + clean(s)) if n in d else clean(s)
    return d


def resize(job) -> bool:
    src, dst = job
    if os.path.exists(dst):
        return True
    from PIL import Image

    try:
        Image.open(src).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC).save(dst, quality=95)
        return True
    except Exception:
        return False


def eval_row(x: dict) -> dict:
    """An evaluation row: the prompt, plus the reference sections the report metrics need."""
    extra = {"patient": x["patient"]} if x.get("patient") else {}
    return jsonl.user_row(
        user_turn(x["context"]),
        os.path.abspath(x["img"]),
        id=x["id"],
        findings=x["findings"],
        impression=x["impression"],
        reference=reference(x["findings"], x["impression"]),
        **extra,
    )


def section_counts(rows: list[dict]) -> str:
    """How many rows have findings, an impression and context, for the progress output."""
    return ", ".join(f"{sum(bool(x[k]) for x in rows)} with {k}" for k in ("findings", "impression", "context"))


def _stats(name, xs):
    xs = sorted(xs)
    p50, p90 = xs[len(xs) // 2], xs[int(len(xs) * 0.9)]
    print(f"  {name}: n={len(xs)} mean={statistics.mean(xs):.1f} p50={p50} p90={p90} max={xs[-1]}")


def main():
    p = argparse.ArgumentParser(
        prog="medres data cxr-report", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mimic-cxr", required=True, help=".../mimic-cxr/2.0.0 (holds mimic-cxr-reports.zip)")
    p.add_argument("--mimic-cxr-jpg", required=True, help=".../mimic-cxr-jpg/2.0.0 (holds files/ and the tables)")
    p.add_argument("--rexrank", required=True, help="ReXRank_MIMICCXR_test.json")
    p.add_argument("--out", default=config.data_path("cxr_report"))
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-val", type=int, default=500)
    p.add_argument("--teacher-tag", default="unirg", help="the routing tag of the task's teacher")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    jpg, out = a.mimic_cxr_jpg, a.out
    os.makedirs(f"{out}/img", exist_ok=True)

    # official split and view position of every image
    with gzip.open(f"{jpg}/mimic-cxr-2.0.0-split.csv.gz", "rt") as f:
        split = {r["dicom_id"]: r for r in csv.DictReader(f)}
    with gzip.open(f"{jpg}/mimic-cxr-2.0.0-metadata.csv.gz", "rt") as f:
        view = {r["dicom_id"]: r["ViewPosition"] for r in csv.DictReader(f)}
    studies = {}  # study -> [subject, split, frontal images]
    for d, r in split.items():
        s = studies.setdefault(r["study_id"], [r["subject_id"], r["split"], []])
        if view.get(d) in ("PA", "AP"):
            s[2].append((0 if view[d] == "PA" else 1, d))
    print("studies", len(studies), {k: sum(v[1] == k for v in studies.values()) for k in ("train", "validate", "test")})

    # ReXrank test set: study, key frontal image, context, reference sections
    with open(a.rexrank) as f:
        rex = json.load(f)
    test_ids, test, missing = set(), [], 0
    for sid, r in rex.items():
        study = str(sid)
        test_ids.add(study)
        src = f"{jpg}/files/{r['key_image_path'].lstrip('/')}"
        if f"/s{study}/" not in src or not os.path.exists(src):
            missing += 1
            continue
        patient = re.search(r"/p(\d+)/s\d+/", src)
        row = {
            "id": study,
            "src": src,
            "img": f"{out}/img/{study}.jpg",
            "context": rexrank_context(r.get("context")),
            "findings": clean(r.get("section_findings")),
            "impression": clean(r.get("section_impression")),
            "patient": patient.group(1) if patient else None,
        }
        test.append(row)
    print(f"ReXrank test: {len(rex)} studies, {missing} without their image; {section_counts(test)}")

    # the report of every train / validate study with a frontal image
    zf = zipfile.ZipFile(f"{a.mimic_cxr}/mimic-cxr-reports.zip")
    names = {re.search(r"s(\d+)\.txt$", n).group(1): n for n in zf.namelist() if re.search(r"s\d+\.txt$", n)}
    print("reports", len(names))
    pools = {"train": [], "validate": []}
    for study, (subject, spl, frontal) in studies.items():
        if spl not in pools or not frontal or study not in names or study in test_ids:
            continue
        d = parse(zf.read(names[study]).decode("utf-8", errors="ignore"))
        if "findings" not in d or "impression" not in d:
            continue
        dicom = sorted(frontal)[0][1]  # PA before AP, then the image id
        row = {
            "id": study,
            "src": f"{jpg}/files/p{subject[:2]}/p{subject}/s{study}/{dicom}.jpg",
            "img": f"{out}/img/{study}.jpg",
            "findings": d["findings"],
            "impression": d["impression"],
            "context": context(d.get("indication", ""), d.get("comparison", "")),
            "patient": subject,
        }
        pools[spl].append(row)
    print("eligible (frontal image, findings and impression):", {k: len(v) for k, v in pools.items()})
    for v in pools.values():
        v.sort(key=lambda x: x["id"])
    random.Random(0).shuffle(pools["train"])
    random.Random(1).shuffle(pools["validate"])
    train, val = pools["train"][: a.n_train], pools["validate"][: a.n_val]

    with ProcessPoolExecutor(a.workers) as ex:
        ok = list(ex.map(resize, [(x["src"], x["img"]) for x in train + val + test], chunksize=64))
    print("images resized", sum(ok), "of", len(ok))
    bad = {x["img"] for x, k in zip(train + val + test, ok) if not k}
    train, val, test = ([x for x in xs if x["img"] not in bad] for xs in (train, val, test))

    prompts, labels = [], []
    for x in train:
        row = jsonl.user_row(
            user_turn(x["context"]),
            os.path.abspath(x["img"]),
            teacher_tag=a.teacher_tag,
            solution=reference(x["findings"], x["impression"]),
        )
        prompts.append(row)
        labels.append(jsonl.answered_row(row, row["solution"]))
    files = {"train": prompts, "train_labels": labels, "val": map(eval_row, val), "test_mimic": map(eval_row, test)}
    for name, rows in files.items():
        print(f"{name}.jsonl", jsonl.write(f"{out}/{name}.jsonl", rows))
    for name, xs in (("train", train), ("test", test)):
        print(name, "reference words / context words:")
        _stats("reference", [len(reference(x["findings"], x["impression"]).split()) for x in xs])
        _stats("context", [len(x["context"].split()) for x in xs])
        n_indication = sum("Indication:" in x["context"] for x in xs)
        n_comparison = sum("Comparison:" in x["context"] for x in xs)
        print(f"  with indication {n_indication}, with comparison {n_comparison}")


if __name__ == "__main__":
    main()
