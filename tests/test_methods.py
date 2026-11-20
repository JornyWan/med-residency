import json
import os
from pathlib import Path

import pytest

from medresidency import config, methods
from medresidency.config import ConfigError

GOLDEN = Path(__file__).parent / "golden_args.json"
TASKS, METHODS = config.available("tasks"), config.available("methods")


def build(task, method, *overrides):
    return methods.build(config.load(task, method, overrides=list(overrides)))


def pairs():
    return [(t, m) for t in TASKS for m in METHODS if not (m == "sft" and t != "cxr_report")]  # only it has labels


def test_every_method_has_a_builder_and_a_config():
    assert sorted(methods.METHODS) == METHODS


def test_golden_arguments():
    """The ms-swift arguments of every task x method. The file was checked, key by key, against the arguments recorded
    by the runs of the AdaOPD paper; a change here changes the benchmark protocol and must be deliberate:
    MEDRES_UPDATE_GOLDEN=1 pytest tests/test_methods.py, then review the diff."""
    now = {}
    for task, method in pairs():
        for stage in build(task, method):
            args = dict(stage.args)
            if "external_plugins" in args:
                args["external_plugins"] = [Path(p).name for p in args["external_plugins"]]
            now[f"{task}/{method}/{stage.name}"] = {
                "command": stage.command,
                "student_gpus": stage.n_gpus,
                "env": stage.env,
                "teacher_logprobs": stage.teacher_logprobs if stage.teachers else None,
                "reward_server": stage.reward_server,
                "args": args,
            }
    if os.environ.get("MEDRES_UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(now, indent=1, sort_keys=True) + "\n")
    golden = json.loads(GOLDEN.read_text())
    assert sorted(now) == sorted(golden)
    for key in now:
        assert now[key] == golden[key], key


def test_sequences_per_step_is_the_same_for_every_method():
    for task in TASKS:
        want = config.load(task, "opd")["train"]["sequences_per_step"]
        for method in METHODS:
            if (task, method) not in pairs():
                continue
            for stage in build(task, method):
                a = stage.args
                assert stage.n_gpus * a["per_device_train_batch_size"] * a["gradient_accumulation_steps"] == want, (
                    task,
                    method,
                )


def test_teacher_methods_keep_two_cards_for_teachers():
    assert [s.n_gpus for s in build("cxr_report", "opd")] == [2]
    assert [s.n_gpus for s in build("cxr_report", "sft")] == [4]
    assert [(s.name, s.command, s.n_gpus, s.teachers, s.steps) for s in build("cxr_report", "two_stage")] == [
        ("stage1", "sft", 4, False, 150),
        ("stage2", "rlhf", 2, True, 150),
    ]
    with pytest.raises(ConfigError, match="no GPU left"):
        build("cxr_report", "opd", "hardware.gpus=2")
    with pytest.raises(ConfigError, match="not a multiple"):
        build("cxr_report", "opd", "hardware.gpus=5")  # 64 sequences on 3 student cards x 4


def test_two_stage():
    s1, s2 = build("cxr_report", "two_stage")
    assert s1.output_dir == "runs/cxr_report_two_stage_s42/stage1" and s1.args["save_steps"] == 150
    assert (
        s2.args["model"] == "runs/cxr_report_two_stage_s42/stage1/checkpoint-150"
        and s2.output_dir == "runs/cxr_report_two_stage_s42"
    )
    (only,) = build("cxr_report", "two_stage", "method.stage1_checkpoint=/ckpt/kd-150")
    assert only.args["model"] == "/ckpt/kd-150" and only.steps == 150
    with pytest.raises(ConfigError, match="sft_steps"):
        build("cxr_report", "two_stage", "method.sft_steps=300")


def test_adaopd_environment():
    (stage,) = build("cxr_report_vqa", "adaopd", "method.tau=0.5", "method.fresh_tokens=false")
    assert stage.env == {
        "MEDRES_ADAOPD_MODE": "control",
        "MEDRES_ADAOPD_TAU": "0.5",
        "MEDRES_ADAOPD_EMA": "0.9",
        "MEDRES_ADAOPD_TAGS": "unirg",
        "MEDRES_ADAOPD_FRESH": "0",
        "MEDRES_ADAOPD_SIGNAL": "lang",
    }
    assert stage.args["chord_sft_dataset"] == "data/cxr_report/train_teacher.jsonl"  # the report teacher's outputs only
    assert Path(stage.args["external_plugins"][0]).is_file()
    with pytest.raises(ConfigError, match="measure_tags"):
        build("cxr_report", "adaopd", "method.measure_tags=[medix]")
    with pytest.raises(ConfigError, match="signal"):
        build("cxr_report", "adaopd", "method.signal=entropy")


def test_method_requirements():
    with pytest.raises(ConfigError, match="needs data.labels"):
        build("med_vqa", "sft")
    with pytest.raises(ConfigError, match="needs teachers"):
        build("cxr_report", "opd", "teachers={}")
    with pytest.raises(ConfigError, match="method.reward"):
        build("cxr_report", "grpo", "method.reward=bleu")
    with pytest.raises(ConfigError, match="not a multiple of save_steps"):
        build("cxr_report", "opd", "train.steps=120")


def test_user_swift_arguments_win():
    (stage,) = build("cxr_report", "opd", "train.swift_args={weight_decay: 0.1, logging_steps: 10}")
    assert stage.args["weight_decay"] == 0.1 and stage.args["logging_steps"] == 10


def test_gkd_keeps_the_routing_column_with_several_teachers():
    assert "remove_unused_columns" not in build("cxr_report", "gkd")[0].args
    assert build("cxr_report_vqa", "gkd")[0].args["remove_unused_columns"] is False
    assert build("cxr_report", "gkd")[0].teacher_logprobs == 20


def test_reward_plugin_registers_the_listed_rewards():
    source = (methods.PLUGINS / "reward_plugin.py").read_text()  # importing it needs ms-swift
    import re

    assert sorted(re.findall(r'^orms\["(\w+)"\]', source, re.M)) == sorted(methods.REWARDS)
