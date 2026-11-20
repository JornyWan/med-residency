"""The data builders, on synthetic rows: prompt formats, row schemas, joins. No dataset is needed."""
import csv
import sys

import pytest

from medresidency import jsonl
from medresidency.answers import is_correct
from medresidency.data import cxr_report, med_vqa, mixed, modality_experts, teacher_outputs


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "new_dir" / "x.jsonl"
    rows = [jsonl.user_row("<image>Describe.", "/img/1.jpg", teacher_tag="unirg"), {"text": "café"}]
    assert jsonl.write(path, rows) == 2
    assert jsonl.read(path) == rows
    assert [p.name for p in path.parent.iterdir()] == ["x.jsonl"]  # the temporary file is gone
    assert jsonl.answered_row(rows[0], "Clear.") == {
        "messages": [{"role": "user", "content": "<image>Describe."}, {"role": "assistant", "content": "Clear."}],
        "images": ["/img/1.jpg"]}


def test_cxr_report_prompt():
    assert cxr_report.clean("  a \n b ") == "a b"
    assert [cxr_report.clean(x) for x in (float("nan"), None, "NaN", " none ")] == ["", "", "", ""]
    assert cxr_report.context("cough", "") == "Indication: cough"
    assert cxr_report.context("", "") == ""
    ctx = cxr_report.rexrank_context("Age: 60 Gender: F Indication: cough History: smoker Comparison: prior film")
    assert ctx == "Indication: cough smoker\nComparison: prior film"  # no age or gender: MIMIC-CXR reports have none
    assert cxr_report.rexrank_context(float("nan")) == ""
    turn = cxr_report.user_turn(ctx)
    assert turn == cxr_report.PRE + "<image>" + ctx + "\n" + cxr_report.POST
    assert cxr_report.user_turn("") == cxr_report.PRE + "<image>" + cxr_report.POST
    assert turn.count("<image>") == 1


def test_cxr_report_sections():
    # a made-up report in the layout of MIMIC-CXR
    text = ("                                 FINAL REPORT\n EXAMINATION:  CHEST (PA AND LAT)\n\n INDICATION:  Cough.\n\n"
            " COMPARISON:  None.\n\n FINDINGS: \n\n The lungs are\n clear.\n\n IMPRESSION: \n\n No acute process.\n")
    assert cxr_report.parse(text) == {"indication": "Cough.", "comparison": "None.", "findings": "The lungs are clear.",
                                      "impression": "No acute process."}
    # history and the reason for the examination join the indication
    history = cxr_report.parse(" FINAL REPORT\n CLINICAL HISTORY:  Fever.\n\n REASON FOR EXAM:  Pain.\n\n FINDINGS:  Clear.\n")
    assert history == {"indication": "Fever. Pain.", "findings": "Clear."}


def test_cxr_report_eval_row():
    x = {"context": "Indication: cough", "img": "/d/img/s1.jpg", "id": "s1", "findings": "Clear.", "impression": "Normal."}
    row = cxr_report.eval_row({**x, "patient": "p1"})
    assert row["reference"] == "Findings: Clear. Impression: Normal."
    assert (row["id"], row["patient"], row["images"]) == ("s1", "p1", ["/d/img/s1.jpg"])
    assert row["messages"] == [{"role": "user", "content": cxr_report.user_turn("Indication: cough")}]
    assert "patient" not in cxr_report.eval_row(x)


def test_med_vqa_prompts():
    r = {"Question": " What is shown? ", "Choice A": " A: lung ", "Choice B": "B: heart", "Choice C": "C: A: liver",
         "Choice D": "D: bone"}
    q = med_vqa.pmc_question(r)
    assert q == "What is shown?\nA) lung\nB) heart\nC) A: liver\nD) bone"
    assert med_vqa.user_turn(q) == f"<image>{q}\nReason step by step, then give the letter of the correct option inside <answer></answer> tags."
    assert med_vqa.mmmu_question("In <image 1>, what?", ["x", "y"]) == "In the image, what?\nAnswer Choices:\n(A) x\n(B) y"
    usable = {"question_type": "multiple-choice", "image_1": {"bytes": b""}, "image_2": None}
    assert med_vqa.mmmu_usable(usable, ["x"] * 5)
    assert not med_vqa.mmmu_usable(usable, ["x"] * 6)
    assert not med_vqa.mmmu_usable({**usable, "image_2": {"bytes": b""}}, ["x"])
    assert not med_vqa.mmmu_usable({**usable, "question_type": "open"}, ["x"])
    assert med_vqa.test_rows("pmc_vqa", [{"image": "/i/a.jpg", "question": "Q?", "answer": "B"}]) == [
        {"messages": [{"role": "user", "content": med_vqa.user_turn("Q?")}], "images": ["/i/a.jpg"], "id": "pmc_vqa/0",
         "domain": "pmc_vqa", "answer": "B", "parser": "letter"}]


def test_med_vqa_builder(tmp_path, monkeypatch):
    image = pytest.importorskip("PIL.Image")
    src, out = tmp_path / "pmc", tmp_path / "out"
    (src / "figures").mkdir(parents=True)
    for k in range(3):
        image.new("RGB", (1792, 896)).save(src / "figures" / f"fig{k}.jpg")
    for name in ("train_2.csv", "test_2.csv"):
        with open(src / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Figure_path", "Question", "Choice A", "Choice B", "Choice C", "Choice D", "Answer"])
            for k in range(3):
                w.writerow([f"fig{k}.jpg", f"Question {k}?", "A: a", "B: b", "C: c", "D: d", "B"])
    monkeypatch.setattr(sys, "argv", ["medres data med-vqa", "--pmc-vqa", str(src), "--out", str(out), "--n-train", "2",
                                      "--n-test", "1"])
    med_vqa.main()
    train, test = jsonl.read(out / "train.jsonl"), jsonl.read(out / "test_pmc_vqa.jsonl")
    assert len(train) == 2 and set(train[0]) == {"messages", "images", "teacher_tag", "solution"}
    assert (train[0]["teacher_tag"], train[0]["solution"]) == ("medix", "<answer>B</answer>")
    assert image.open(train[0]["images"][0]).size == (896, 448)  # longer side 896
    assert len(test) == 1 and (test[0]["id"], test[0]["answer"], test[0]["parser"]) == ("pmc_vqa/0", "B", "letter")
    assert is_correct(test[0], "the figure shows b. <answer>B</answer>")


def test_mixed():
    def source(tag, n, answered):
        prompts = [jsonl.user_row(f"<image>{tag} {i}", f"/{tag}/{i}.jpg", teacher_tag=tag) for i in range(n)]
        return prompts, [jsonl.answered_row(p, f"output {tag} {i}") for i, p in enumerate(prompts) if i in answered]

    a, b = source("unirg", 6, {0, 1, 2, 3}), source("medix", 5, {1, 2, 3})
    prompts, outputs = mixed.mix([a, b], 3)
    assert sorted(p["teacher_tag"] for p in prompts) == ["medix"] * 3 + ["unirg"] * 3
    assert [mixed.key(p) for p in prompts] == [mixed.key(o) for o in outputs]  # row i of both files is one prompt
    assert {mixed.key(p) for p in prompts} <= {mixed.key(o) for o in a[1] + b[1]}  # only prompts with a teacher output
    assert len({p["teacher_tag"] for p in prompts[:3]}) == 2 or len({p["teacher_tag"] for p in prompts[3:]}) == 2  # shuffled
    assert mixed.mix([a, b], 3) == (prompts, outputs)  # the same rows every time
    with pytest.raises(SystemExit, match="only 3 prompts"):
        mixed.mix([a, b], 4)


def test_modality_experts_rows(tmp_path):
    for name in ("CT(Computed Tomography)_train.json", "X-Ray_train.json", "notes_train.json"):
        (tmp_path / name).write_text("[]")
    assert modality_experts.split_files(str(tmp_path)) == [
        ("CT", str(tmp_path / "CT(Computed Tomography)_train.json")), ("X-Ray", str(tmp_path / "X-Ray_train.json"))]
    x = {"problem": "Which organ? A) liver B) lung", "solution": "<answer> B </answer>"}
    row = modality_experts.train_row(x, "CT", "/i/1.png")
    assert (row["teacher_tag"], row["solution"], row["images"]) == ("CT", x["solution"], ["/i/1.png"])
    assert row["messages"][0]["content"].startswith("<image>Which organ? A) liver B) lung Your task: 1. Think")
    t = modality_experts.test_row(x, "CT", 7, "/i/1.png")
    assert (t["id"], t["domain"], t["answer"], t["parser"]) == ("CT/7", "CT", "B", "answer_tag")
    assert is_correct(t, "<think>the lung</think><answer>B</answer>") and not is_correct(t, "<answer>A</answer>")
    assert set(modality_experts.TAG.values()) == {"CT", "MRI", "X-Ray", "Fundus", "Dermo", "Micro", "OCT", "US"}


def test_teacher_outputs_join():
    prompts = [jsonl.user_row(f"<image>q{i}", f"/i/{i}.jpg", teacher_tag="unirg", solution="reference") for i in range(4)]
    gens = [dict(id=i, text=f" answer {i}\n", n_tokens=10 * (i + 1), truncated=int(i == 1)) for i in range(4)]
    gens[2]["malformed"] = 1
    rows, stats = teacher_outputs.join(prompts, gens[::-1])
    assert [r["messages"][1]["content"] for r in rows] == ["answer 0", "answer 3"]  # prompt order, complete outputs only
    assert all(set(r) == {"messages", "images"} for r in rows)  # no reference, no tag
    assert stats == {"prompts": 4, "kept": 2, "truncated": 1, "malformed": 1, "tokens_mean": 25.0, "tokens_p50": 30,
                     "tokens_p90": 40}
    for i, tag in enumerate(("CT", "X-Ray", "CT", "MRI")):
        prompts[i]["teacher_tag"] = tag
    far, far_stats = teacher_outputs.join(prompts, gens, tags=["CT", "MRI"])
    assert [r["messages"][1]["content"] for r in far] == ["answer 0", "answer 3"] and far_stats["kept"] == 2
    assert teacher_outputs.join(prompts, gens, tags=["X-Ray"])[0] == []  # its only output is truncated
    with pytest.raises(SystemExit, match="3 of 4"):
        teacher_outputs.join(prompts, gens[:3])
    with pytest.raises(SystemExit, match="two generations"):
        teacher_outputs.join(prompts, gens + gens[:1])
