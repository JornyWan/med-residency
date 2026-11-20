import json

import pytest

from medresidency import cli, config, launch, methods


def test_to_argv():
    assert launch.to_argv({"a": 1, "b": True, "c": False, "d": None, "e": ["x", "y"], "f": {"k": 1}, "g": 5e-06}) == [
        "--a",
        "1",
        "--b",
        "true",
        "--c",
        "false",
        "--e",
        "x",
        "y",
        "--f",
        '{"k": 1}',
        "--g",
        "5e-06",
    ]


def test_gpu_ids(monkeypatch):
    cfg = config.load("cxr_report", "opd")
    assert launch.gpu_ids(cfg) == ["0", "1", "2", "3"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7,8")
    assert launch.gpu_ids(cfg) == ["4", "5", "6", "7"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(launch.LaunchError, match="lists 2 cards"):
        launch.gpu_ids(cfg)


def test_teacher_plan():
    gpus = ["0", "1", "2", "3"]
    one = launch.teacher_plan(config.load("cxr_report", "opd"), gpus)
    assert [(t["tag"], t["gpus"]) for t in one] == [("unirg", ["2", "3"])]  # one teacher: both cards, tensor parallel
    two = launch.teacher_plan(config.load("cxr_report_vqa", "opd"), gpus)
    assert [(t["tag"], t["gpus"]) for t in two] == [("medix", ["2"]), ("unirg", ["3"])]
    eight = launch.teacher_plan(config.load("modality_experts", "opd"), gpus)
    assert [t["gpus"] for t in eight] == [["2"]] * 4 + [["3"]] * 4  # four experts per card


def test_teacher_command():
    cfg = config.load("cxr_report", "gkd")
    (stage,) = methods.build(cfg)
    (teacher,) = launch.teacher_plan(cfg, ["0", "1", "2", "3"])
    cmd = launch.teacher_command(cfg, stage, teacher, 8001)
    assert cmd[1:4] == ["-m", "swift.cli.main", "deploy"]
    opt = dict(zip(cmd[4::2], cmd[5::2]))
    assert opt["--model"] == "models/UniRG-CXR" and opt["--port"] == "8001"
    assert opt["--max_logprobs"] == "20" and opt["--vllm_tensor_parallel_size"] == "2"  # GKD reads the top-20
    assert opt["--vllm_max_model_len"] == "2560"
    assert json.loads(opt["--vllm_engine_kwargs"]) == {"max_num_batched_tokens": 2048}


def test_last_checkpoint(tmp_path):
    assert launch.last_checkpoint(tmp_path / "missing") == 0
    for name in ("checkpoint-50", "checkpoint-100", "checkpoint-last", "medres"):
        (tmp_path / name).mkdir()
    assert launch.last_checkpoint(tmp_path) == 100


def test_launch_record_on_a_node_without_git(monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(launch.subprocess, "run", no_git)
    (stage,) = methods.build(config.load("med_vqa", "opd"))
    record = launch._record(stage, ["0", "1"], 0)
    assert record["code"] is None
    assert record["stage"] == stage.name and record["gpus"] == ["0", "1"]


def test_environment_keeps_what_the_caller_set(monkeypatch):
    cfg = config.load("cxr_report", "opd")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("MEDRES_BAD_WORDS", raising=False)
    env = launch.environment(cfg, "/cache")
    assert env["HF_HUB_OFFLINE"] == "1" and env["MODELSCOPE_CACHE"] == "/cache/modelscope"
    assert env["MEDRES_BAD_WORDS"].split(",")[0] == "<|vision_start|>"
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    assert "HF_HUB_OFFLINE" not in launch.environment(cfg, "/cache")


@pytest.mark.parametrize(
    "task, method, expect",
    [
        (
            "cxr_report",
            "adaopd",
            [
                "# teacher unirg",
                "CUDA_VISIBLE_DEVICES=2,3",
                "MEDRES_ADAOPD_TAU=1.0",
                "NPROC_PER_NODE=2",
                "--chord_sft_dataset data/cxr_report/train_teacher.jsonl",
                "swift.cli.main rlhf",
            ],
        ),
        ("cxr_report", "sft", ["CUDA_VISIBLE_DEVICES=0,1,2,3", "NPROC_PER_NODE=4", "swift.cli.main sft"]),
        ("cxr_report_vqa", "opd", ["# teacher medix", "# teacher unirg", '"tags": ["unirg"]']),
        ("med_vqa", "two_stage", ["# stage stage1: 150 steps", "# stage stage2: 150 steps", "stage1/checkpoint-150"]),
    ],
)
def test_dry_run_prints_the_commands(capsys, task, method, expect):
    cli.main(["train", "--task", task, "--method", method, "--dry-run"])
    out = capsys.readouterr().out
    for text in expect:
        assert text in out, text


def test_reward_server_needs_the_metrics_environment(capsys, monkeypatch):
    with pytest.raises(SystemExit, match="MEDRES_METRICS_PYTHON"):
        cli.main(["train", "--task", "cxr_report", "--method", "grpo", "--dry-run"])
    monkeypatch.setenv("MEDRES_METRICS_PYTHON", "/envs/metrics/bin/python")
    cli.main(["train", "--task", "cxr_report", "--method", "grpo", "--dry-run"])
    out = capsys.readouterr().out
    assert (
        "/envs/metrics/bin/python -I" in out
        and "eval/reward_server.py" in out
        and "MEDRES_REWARD_URL='http://127.0.0.1:" in out
    )
    cli.main(["train", "--task", "med_vqa", "--method", "grpo", "--dry-run"])  # letter accuracy: no server
    assert "reward_server" not in capsys.readouterr().out


def test_cli_list_and_errors(capsys):
    cli.main(["list"])
    out = capsys.readouterr().out
    for word in ("cxr_report", "adaopd", "qwen3-vl-2b", "report_radcliq"):
        assert word in out
    with pytest.raises(SystemExit, match="unknown configuration keys: train.learning_rate"):
        cli.main(["train", "--task", "cxr_report", "--method", "opd", "train.learning_rate=1"])
    cli.main(["train", "--method", "opd", "--task", "med_vqa", "--seed", "3", "--print-config", "train.lr=2e-6"])
    assert "name: med_vqa_opd_s3" in capsys.readouterr().out


def test_refuses_to_overwrite_a_finished_run(tmp_path):
    cfg = config.load("cxr_report", "opd", overrides=[f"run.output_root={tmp_path}"])
    (tmp_path / "cxr_report_opd_s42" / "checkpoint-300").mkdir(parents=True)
    with pytest.raises(launch.LaunchError, match="already holds checkpoint-300"):
        launch.run(cfg)
    cfg["run"]["resume"] = True
    launch.run(cfg)  # complete: nothing is started
