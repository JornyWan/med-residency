"""The evaluation tools without a GPU: selection and settings of generation, VQA scoring, the pure parts of report
scoring, the driver on generations that exist, and the protocol of the reward server."""

import argparse
import json
import re
import socket
import sys
import threading
import time
import urllib.request

import numpy as np
import pytest

from medresidency import config, jsonl
from medresidency.eval import generate, report_score, run, vqa

TASKS = ["cxr_report", "cxr_report_vqa", "med_vqa", "modality_experts"]


def test_chat_content():
    assert generate.chat_content("<image>What?") == [{"type": "image"}, {"type": "text", "text": "What?"}]
    assert generate.chat_content("Context:<image>Write.") == [
        {"type": "text", "text": "Context:"},
        {"type": "image"},
        {"type": "text", "text": "Write."},
    ]
    for bad in ("no image", "<image>a<image>b"):
        with pytest.raises(ValueError):
            generate.chat_content(bad)


def test_select():
    rows = [{"teacher_tag": "CT" if i % 2 else "MRI"} for i in range(10)]
    ids = lambda **kw: [r["id"] for r in generate.select(rows, **kw)]
    assert ids() == list(range(10))  # the id of a row without one is its row number in the whole file
    assert ids(tag="CT") == [1, 3, 5, 7, 9]
    assert ids(shard="1/4") == [1, 5, 9]
    assert sorted(i for k in range(4) for i in ids(shard=f"{k}/4")) == list(range(10))  # the shards are the file
    assert ids(tag="CT", shard="1/2", limit=1) == [3]
    assert generate.select([{"id": "pmc_vqa/0"}])[0]["id"] == "pmc_vqa/0"


def test_generation_settings():
    def ns(**kw):
        return argparse.Namespace(**{**dict.fromkeys(generate.OPTIONS), "task": None, "profile": "eval", **kw})

    assert generate.resolve(ns(task="cxr_report")) == {
        "max_tokens": 1024,
        "max_model_len": 2048,
        "max_num_seqs": 128,
        "temperature": 0.0,
        "well_formed": None,
        "retries": 4,
    }
    vqa_opt = generate.resolve(ns(task="med_vqa"))
    assert (vqa_opt["max_tokens"], vqa_opt["max_model_len"], vqa_opt["max_num_seqs"]) == (4096, 6144, 64)
    assert generate.resolve(ns(task="med_vqa", profile="teacher_outputs"))["max_tokens"] == 2048
    expert = generate.resolve(ns(task="modality_experts", profile="teacher_outputs"))
    assert (expert["temperature"], expert["max_tokens"], expert["retries"]) == (1.0, 512, 4)
    well_formed = re.compile(expert["well_formed"], re.DOTALL)
    assert well_formed.fullmatch("<think>a\nb</think>\n<answer>C</answer>")
    assert not well_formed.fullmatch("<answer>C</answer>")
    assert not well_formed.fullmatch("<think>a</think><answer>C</answer> so C")
    # an explicit option wins over the task; the others stay
    assert generate.resolve(ns(task="cxr_report", max_tokens=64, temperature=0.7)) == {
        "max_tokens": 64,
        "max_model_len": 2048,
        "max_num_seqs": 128,
        "temperature": 0.7,
        "well_formed": None,
        "retries": 4,
    }
    assert generate.resolve(ns(max_tokens=100))["max_model_len"] == 2148
    with pytest.raises(config.ConfigError, match="--task or --max-tokens"):
        generate.resolve(ns())
    with pytest.raises(config.ConfigError, match="no generation profile"):
        generate.resolve(ns(task="cxr_report_vqa"))  # its test sets are those of the two source tasks


@pytest.mark.parametrize("task", TASKS)
def test_task_generation_settings_are_options_of_generate(task):
    profiles = list((config.task(task).get("generation") or {}).values())
    profiles += [s["generation"] for s in config.test_sets(task)]
    assert profiles
    for profile in profiles:
        assert set(profile) <= set(generate.OPTIONS)


def _vqa_example():
    data = [
        dict(id=f"{d}/{i}", domain=d, answer="B", parser="letter")
        for d, n in (("pmc_vqa", 3), ("mmmu_health", 1))
        for i in range(n)
    ]
    texts = ["The answer is B", "<answer>C</answer>", "I cannot tell", "\\boxed{B}"]
    gens = [dict(id=r["id"], text=t, n_tokens=10, truncated=int(k == 2)) for k, (r, t) in enumerate(zip(data, texts))]
    return data, gens


def test_vqa_scoring(tmp_path):
    data, gens = _vqa_example()
    items = vqa.score(data, gens[::-1])
    assert [x["correct"] for x in items] == [1, 0, 0, 1] and [x["answered"] for x in items] == [1, 1, 0, 1]
    assert all("text" not in x for x in items)
    s = vqa.summarize(items)
    assert (s["n"], s["micro_acc"], s["macro_acc"]) == (4, 50.0, 66.67)  # macro: mean of 33.33 and 100
    assert s["domains"]["pmc_vqa"] == {
        "n": 3,
        "acc": 33.33,
        "no_answer": 33.33,
        "truncated": 33.33,
        "mean_tokens": 10.0,
    }
    assert list(s["domains"]) == ["pmc_vqa", "mmmu_health"]
    for incomplete in (gens[:3], gens + gens[:1]):
        with pytest.raises(SystemExit):
            vqa.score(data, incomplete)
    jsonl.write(tmp_path / "model.jsonl", gens)
    assert vqa.score_file(data, str(tmp_path / "model.jsonl")) == s
    assert jsonl.read(tmp_path / "model.items.jsonl") == items
    assert json.loads((tmp_path / "model.summary.json").read_text()) == s


def test_report_views():
    assert report_score.norm(" a\n b ") == "a b"
    assert report_score.findings_part("Findings: clear lungs. Impression: normal.") == "clear lungs."
    assert report_score.findings_part("clear lungs. Impression: normal.") == "clear lungs."  # no heading
    assert report_score.findings_part("FINDINGS: x") == "x"
    assert report_score.FI.search("Findings: a Impression: b") and not report_score.FI.search("a Impression: b")
    data = {
        1: {"findings": "f1", "impression": "i1", "reference": "Findings: f1 Impression: i1"},
        2: {"findings": "f2", "impression": "", "reference": "Findings: f2 Impression: "},
        3: {"findings": "", "impression": "i3", "reference": "Findings:  Impression: i3"},
    }
    gen = {1: {"text": "Findings: a  b Impression: c"}, 2: {"text": ""}, 3: {"text": "Impression: z"}}
    v = report_score.views(data, gen)
    assert v["fi"] == ([1], ["Findings: f1 Impression: i1"], ["Findings: a b Impression: c"])
    assert v["f"] == ([1, 2], ["f1", "f2"], ["a b", "."])  # an empty hypothesis becomes "."


def test_run_groups_generation_by_settings(tmp_path):
    jsonl.write(tmp_path / "r.jsonl", [{"id": 1, "reference": "Findings: a Impression: b"}])
    jsonl.write(tmp_path / "v.jsonl", [{"id": "pmc_vqa/0", "answer": "B"}])
    assert (run.kind(str(tmp_path / "r.jsonl")), run.kind(str(tmp_path / "v.jsonl"))) == ("report", "vqa")
    sets = [
        dict(file="a.jsonl", out="o/a.jsonl", generation={"max_tokens": 1024, "max_model_len": 2048}),
        dict(file="b.jsonl", out="o/b.jsonl", generation={"max_model_len": 2048, "max_tokens": 1024}),
        dict(file="c.jsonl", out="o/c.jsonl", generation={"max_tokens": 4096, "max_model_len": 6144}),
    ]
    first, second = run.generate_commands(sets, "ckpt", ["--seed", "0"])
    assert first == [
        sys.executable,
        "-m",
        "medresidency",
        "eval",
        "generate",
        "--model",
        "ckpt",
        "--data",
        "a.jsonl",
        "b.jsonl",
        "--out",
        "o/a.jsonl",
        "o/b.jsonl",
        "--seed",
        "0",
        "--max-model-len",
        "2048",
        "--max-tokens",
        "1024",
    ]
    assert second[7:] == [
        "--data",
        "c.jsonl",
        "--out",
        "o/c.jsonl",
        "--seed",
        "0",
        "--max-model-len",
        "6144",
        "--max-tokens",
        "4096",
    ]
    assert run.generate_commands([], "ckpt", []) == []
    assert run.headline("vqa", {"micro_acc": 50.0, "macro_acc": 66.67, "n": 4}) == "accuracy 50.00 (macro 66.67), n=4"
    report = {"fi_inv_radcliq": 0.9312, "fi_radgraph": 0.251, "truncated": 0.0, "n_fi": 10}
    assert run.headline("report", report) == "1/RadCliQ-v1 0.931, RadGraph-F1 25.1, truncated % 0.0, n=10"


def test_run_scores_generations_that_exist(tmp_path, monkeypatch, capsys):
    data, gens = _vqa_example()
    monkeypatch.setenv("MEDRES_DATA", str(tmp_path / "data"))
    jsonl.write(tmp_path / "data/med_vqa/test_pmc_vqa.jsonl", data)
    out = tmp_path / "evals"
    jsonl.write(out / "pmc_vqa.jsonl", gens)
    argv = ["medres eval run", "--task", "med_vqa", "--model", "runs/x/checkpoint-300", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", [*argv, "--sets", "pmc_vqa"])
    run.main()  # nothing to generate: no GPU needed
    assert json.loads((out / "summary.json").read_text())["pmc_vqa"]["micro_acc"] == 50.0
    assert "accuracy 50.00 (macro 66.67), n=4" in capsys.readouterr().out
    jsonl.write(tmp_path / "data/med_vqa/test_mmmu_health.jsonl", data)
    jsonl.write(out / "mmmu_health.jsonl", gens)
    monkeypatch.setattr(sys, "argv", [*argv, "--sets", "mmmu_health"])
    run.main()  # a second call adds its test set to the summary
    assert set(json.loads((out / "summary.json").read_text())) == {"pmc_vqa", "mmmu_health"}
    capsys.readouterr()
    for bad, message in (("--sets nope", "no test set nope"), ("", "test files not found")):
        monkeypatch.setattr(sys, "argv", [*argv, *bad.split()])
        with pytest.raises(SystemExit):
            run.main()
        assert message in capsys.readouterr().err


def test_run_needs_the_metrics_environment_for_reports(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MEDRES_DATA", str(tmp_path))
    for name in ("test_mimic", "test_iu"):
        jsonl.write(tmp_path / f"cxr_report/{name}.jsonl", [{"id": "s1", "reference": "Findings: a Impression: b"}])
    argv = ["medres eval run", "--task", "cxr_report", "--model", "m", "--out", str(tmp_path / "e")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        run.main()
    assert "MEDRES_METRICS_PYTHON" in capsys.readouterr().err
    assert not (tmp_path / "e").exists()  # refused before anything is generated


def test_reward_server_protocol(monkeypatch):
    """The reward server answers the launcher's request (/health) and the reward plugin's (/score)."""
    from medresidency.eval import reward_server

    monkeypatch.setattr(reward_server.report_score, "radcliq_metric", lambda: None)
    monkeypatch.setattr(
        reward_server.report_score,
        "composite",
        lambda metric, refs, hyps: {"radcliq": np.array([float(len(h)) for h in hyps])},
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setattr(sys, "argv", ["reward_server.py", "--port", str(port)])
    threading.Thread(target=reward_server.main, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            with urllib.request.urlopen(url + "/health", timeout=1) as r:
                assert json.load(r) == {"ok": True, "calls": 0}
            break
        except OSError:
            time.sleep(0.05)
    else:
        pytest.fail("the reward server did not start")
    monkeypatch.setenv("MEDRES_REWARD_URL", url)
    body = json.dumps({"refs": ["Findings: a Impression: b"] * 2, "hyps": ["abc  d", ""]}).encode()
    req = urllib.request.Request(url + "/score", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        assert json.load(r) == {"rewards": [-5.0, -1.0]}  # -RadCliQ of the normalised text; "" is scored as "."
