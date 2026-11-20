import shutil

import pytest
import yaml

from medresidency import config
from medresidency.config import ConfigError

TASKS = ["cxr_report", "cxr_report_vqa", "med_vqa", "modality_experts"]
METHODS = ["adaopd", "chord", "chord_phi", "gkd", "grpo", "offline_kd", "opd", "sft", "two_stage"]


def test_available():
    assert config.available("tasks") == TASKS
    assert config.available("methods") == METHODS


def test_layers_in_order(tmp_path):
    cfg = config.load("med_vqa", "opd")
    assert cfg["train"]["lr"] == 1e-6 and cfg["train"]["steps"] == 300  # task over defaults; defaults kept
    assert cfg["run"]["name"] == "med_vqa_opd_s42"
    # the task's block for a method
    assert config.load("cxr_report_vqa", "adaopd")["method"]["measure_tags"] == ["unirg"]
    assert config.load("cxr_report_vqa", "grpo")["train"]["per_device_batch"] == 2
    # a recipe over the task, key=value over the recipe
    recipe = tmp_path / "mine.yaml"
    recipe.write_text("task: med_vqa\nmethod: {name: adaopd, tau: 0.5}\ntrain: {lr: 2e-6}\n")
    cfg = config.load(recipe=str(recipe))
    assert (cfg["method"]["tau"], cfg["train"]["lr"], cfg["run"]["name"]) == (0.5, 2e-6, "mine_s42")
    cfg = config.load(recipe=str(recipe), overrides=["method.tau=0", "run.seed=3", "train.lr=1e-5"])
    assert (cfg["method"]["tau"], cfg["train"]["lr"], cfg["run"]["name"]) == (0, 1e-5, "mine_s3")


def test_recipe_without_method_is_named_after_the_method(tmp_path):
    recipe = tmp_path / "setting.yaml"
    recipe.write_text("task: cxr_report\nstudent: {model: qwen3-vl-4b}\n")
    assert config.load(method="gkd", recipe=str(recipe))["run"]["name"] == "setting_gkd_s42"


def test_scientific_notation_is_a_number():
    # YAML 1.1 reads "1e-6" as a string
    assert config.load("cxr_report", "opd", overrides=["train.lr=1e-6"])["train"]["lr"] == pytest.approx(1e-6)
    assert isinstance(config.load("cxr_report", "opd")["train"]["lr"], float)


@pytest.mark.parametrize("override", ["train.learning_rate=1e-6", "method.tua=0.5", "trian.lr=1", "run.sed=2"])
def test_unknown_keys_are_errors(override):
    with pytest.raises(ConfigError, match="unknown configuration keys"):
        config.load("cxr_report", "adaopd", overrides=[override])


def test_errors():
    with pytest.raises(ConfigError, match="needs a task and a method"):
        config.load("cxr_report")
    with pytest.raises(ConfigError, match="unknown task"):
        config.load("nope", "opd")
    with pytest.raises(ConfigError, match="unknown method"):
        config.load("cxr_report", "nope")
    with pytest.raises(ConfigError, match="unknown model"):
        config.load("cxr_report", "opd", overrides=["student.model=gpt-17"])
    with pytest.raises(ConfigError, match="not key=value"):
        config.load("cxr_report", "opd", overrides=["train.lr"])


def test_teachers_are_replaced_as_a_whole():
    cfg = config.load("cxr_report_vqa", "opd", overrides=["teachers={qoq: qoq-med3-8b}"])
    assert cfg["teachers"] == {"qoq": "qoq-med3-8b"}


def test_models_and_data_paths(monkeypatch):
    cfg = config.load("modality_experts", "opd")
    assert config.model(cfg, "med-r1-ct") == {
        "name": "med-r1-ct",
        "dir": "models/Med-R1/VQA_CT",
        "hub": "yuxianglai117/Med-R1",
        "model_type": "qwen2_vl",
        "template": "qwen2_vl",
    }
    monkeypatch.setenv("MEDRES_MODELS", "/weights")
    monkeypatch.setenv("MEDRES_DATA", "/corpora")
    assert config.model(cfg, "qwen3-vl-2b")["dir"] == "/weights/Qwen3-VL-2B-Instruct"
    assert config.data_path("cxr_report/train.jsonl") == "/corpora/cxr_report/train.jsonl"
    assert config.data_path("/abs/x.jsonl") == "/abs/x.jsonl"
    # a run may bring its own model
    overrides = ["models={mine: {path: /m/mine, model_type: qwen3_vl}}", "student.model=mine"]
    cfg = config.load("cxr_report", "opd", overrides=overrides)
    assert config.model(cfg, "mine")["dir"] == "/m/mine"


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("method", METHODS)
def test_every_pair_resolves(task, method):
    cfg = config.load(task, method)
    assert cfg["task"] == task and cfg["method"]["name"] == method
    yaml.safe_dump(cfg)  # plain data: what the launcher writes to <run>/medres/config.yaml


def test_recipes_resolve():
    recipes = sorted((config.config_dir() / "recipes").rglob("*.yaml"))
    assert recipes
    for r in recipes:
        named = (yaml.safe_load(r.read_text()).get("method") or {}).get("name")
        config.load(method=None if named else "opd", recipe=str(r))


def test_test_sets():
    sets = config.test_sets("cxr_report")
    assert [(s["name"], s["file"]) for s in sets] == [
        ("mimic_cxr", "data/cxr_report/test_mimic.jsonl"),
        ("iu_xray", "data/cxr_report/test_iu.jsonl"),
    ]
    assert sets[0]["generation"] == {"max_tokens": 1024, "max_model_len": 2048, "max_num_seqs": 128}
    # the two-teacher task is tested on sets of its source tasks, each with the settings of the task it comes from
    mixed = {s["name"]: s for s in config.test_sets("cxr_report_vqa")}
    assert list(mixed) == ["mimic_cxr", "iu_xray", "pmc_vqa", "medxpertqa_mm", "mmmu_health"]
    assert (mixed["mimic_cxr"]["generation"]["max_tokens"], mixed["pmc_vqa"]["generation"]["max_tokens"]) == (
        1024,
        4096,
    )
    assert mixed["pmc_vqa"]["file"] == "data/med_vqa/test_pmc_vqa.jsonl"
    with pytest.raises(ConfigError, match="unknown task"):
        config.test_sets("nope")
    # the validation set of a task takes the place of its test sets
    (val,) = config.test_sets("med_vqa", "validation")
    assert (val["name"], val["file"]) == ("validation", "data/med_vqa/val.jsonl")
    assert val["generation"]["max_tokens"] == 4096
    with pytest.raises(ConfigError, match="no validation set"):
        config.test_sets("modality_experts", "validation")


def test_evaluation_blocks_belong_to_the_task(tmp_path, monkeypatch):
    cfg = config.load("cxr_report", "opd")
    assert "eval" not in cfg and "generation" not in cfg  # not part of a run's configuration
    recipe = tmp_path / "mine.yaml"
    recipe.write_text("task: cxr_report\nmethod: {name: opd}\neval: {test: {mine: x.jsonl}}\n")
    with pytest.raises(ConfigError, match="eval"):
        config.load(recipe=str(recipe))
    # a task directory of one's own
    shutil.copytree(config.config_dir(), tmp_path / "configs")
    monkeypatch.setenv("MEDRES_CONFIGS", str(tmp_path / "configs"))
    base = (tmp_path / "configs/tasks/med_vqa.yaml").read_text()
    task = yaml.safe_load(base)
    for name, change, message in (
        ("no_settings", {"generation": None}, "generation.eval"),
        ("borrows", {"eval": {"test_from": {"cxr_report": ["chexpert"]}}}, "no test set chexpert"),
    ):
        (tmp_path / f"configs/tasks/{name}.yaml").write_text(yaml.safe_dump({**task, **change}))
        with pytest.raises(ConfigError, match=message):
            config.test_sets(name)
        assert config.load(name, "opd")["task"] == name  # training does not depend on the evaluation blocks
