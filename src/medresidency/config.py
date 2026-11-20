"""Run configuration.

A run is one nested dictionary, built from YAML layers (later layers win):

    configs/defaults.yaml <- configs/tasks/<task>.yaml <- configs/methods/<method>.yaml
        <- the task's `overrides.<method>` block <- a recipe file <- key=value arguments

`configs/defaults.yaml` documents every key. Keys that no layer defines are errors, so a typo cannot silently train
with a default.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path

import yaml

FIXED = ("run", "student", "data", "train", "teacher_server", "hardware")  # their keys are those of defaults.yaml
TOP = FIXED + ("teachers", "method", "models", "task")
# blocks of a task file that the evaluation tools read (`task`, `test_sets`), not a run
TASK_ONLY = ("eval", "generation")
_SCI = re.compile(r"[+-]?\d+(\.\d*)?[eE][+-]?\d+")  # "1e-6": a string for YAML 1.1, a number for everybody else


class ConfigError(ValueError):
    pass


def config_dir() -> Path:
    d = Path(os.environ.get("MEDRES_CONFIGS") or Path(__file__).resolve().parents[2] / "configs")
    if not (d / "defaults.yaml").is_file():
        raise ConfigError(
            f"no configuration files in {d}: install from a clone (`pip install -e .`) or set MEDRES_CONFIGS"
        )
    return d


def _read(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def available(kind: str) -> list[str]:
    """Names of the `tasks` or `methods`."""
    return sorted(p.stem for p in (config_dir() / kind).glob("*.yaml"))


def describe(kind: str, name: str, full: bool = False) -> str:
    text = _read(config_dir() / kind / f"{name}.yaml").get("description", "")
    return text if full else text.split(". ")[0].rstrip(".")


def task(name: str) -> dict:
    """The YAML of a task, for the tools that need its `eval` and `generation` blocks."""
    if name not in available("tasks"):
        raise ConfigError(f"unknown task {name!r}; available: {', '.join(available('tasks'))}")
    return _read(config_dir() / "tasks" / f"{name}.yaml")


def test_sets(name: str, split: str = "test") -> list[dict]:
    """The test sets of a task: name, file, and the generation settings of the task the set belongs to.

    A task lists its own sets under `eval.test` and borrows sets of other tasks with
    `eval.test_from: {task: [set, ...]}`. With split="validation" the result is the one set under `eval.validation`,
    the set to select hyper-parameters on."""
    t = task(name)
    ev = t.get("eval") or {}
    profile = (t.get("generation") or {}).get("eval")
    if split == "validation":
        if not ev.get("validation"):
            raise ConfigError(f"task {name} has no validation set (`eval.validation` in its task file)")
        return [{"name": "validation", "file": data_path(ev["validation"]), "generation": profile}]
    if ev.get("test") and not profile:
        raise ConfigError(f"task {name}: the test sets under `eval.test` need the settings `generation.eval`")
    sets = [{"name": k, "file": data_path(v), "generation": profile} for k, v in (ev.get("test") or {}).items()]
    for other, names in (ev.get("test_from") or {}).items():
        theirs = {s["name"]: s for s in test_sets(other)}
        missing = [n for n in names if n not in theirs]
        if missing:
            raise ConfigError(f"task {name}: task {other} has no test set {', '.join(missing)}")
        sets += [theirs[n] for n in names]
    return sets


def _merge(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict) and k != "teachers":  # a layer replaces all teachers
            _merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _numbers(node):
    for k, v in node.items() if isinstance(node, dict) else enumerate(node):
        if isinstance(v, (dict, list)):
            _numbers(v)
        elif isinstance(v, str) and _SCI.fullmatch(v):
            node[k] = float(v)


def _set(cfg: dict, item: str) -> None:
    key, eq, raw = item.partition("=")
    if not eq or not key:
        raise ConfigError(f"override {item!r} is not key=value")
    *path, last = key.split(".")
    node = cfg
    for p in path:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ConfigError(f"override {item!r}: {p!r} is not a section")
    node[last] = yaml.safe_load(raw)


def load(task: str | None = None, method: str | None = None, recipe: str | None = None, overrides=()) -> dict:
    """The resolved configuration of a run."""
    d = config_dir()
    rec = _read(recipe) if recipe else {}
    task = task or rec.get("task")
    method = method or (rec.get("method") or {}).get("name")
    if not task or not method:
        raise ConfigError("a run needs a task and a method: --task and --method, or a recipe that names them")
    for kind, name in (("tasks", task), ("methods", method)):
        if name not in available(kind):
            raise ConfigError(f"unknown {kind[:-1]} {name!r}; available: {', '.join(available(kind))}")
    defaults = _read(d / "defaults.yaml")
    t, m = _read(d / "tasks" / f"{task}.yaml"), _read(d / "methods" / f"{method}.yaml")
    task_overrides = t.pop("overrides", None) or {}
    for k in TASK_ONLY:
        t.pop(k, None)
    cfg = copy.deepcopy(defaults)
    for layer in (t, m, task_overrides.get(method) or {}, rec):
        _merge(cfg, {k: v for k, v in layer.items() if k not in ("description", "task")})
    cfg["task"] = task
    cfg["method"]["name"] = method
    for item in overrides:
        _set(cfg, item)
    _numbers(cfg)

    unknown = [k for k in cfg if k not in TOP]
    unknown += [f"{s}.{k}" for s in FIXED for k in cfg[s] if k not in defaults[s]]
    unknown += [f"method.{k}" for k in cfg["method"] if k not in m["method"]]
    if unknown:
        where = f"configs/defaults.yaml and configs/methods/{method}.yaml"
        raise ConfigError(f"unknown configuration keys: {', '.join(unknown)} (see {where})")
    if cfg["run"]["name"] is None:
        stem = f"{task}_{method}"
        if recipe:  # a recipe names its runs; one that leaves the method open gets the method appended
            stem = Path(recipe).stem
            if not (rec.get("method") or {}).get("name"):
                stem += f"_{method}"
        cfg["run"]["name"] = f"{stem}_s{cfg['run']['seed']}"
    for name in [cfg["student"]["model"], *cfg["teachers"].values()]:
        model(cfg, name)
    return cfg


def model(cfg: dict, name: str) -> dict:
    """A model of the zoo (configs/models.yaml, plus the run's own `models` block): directory, model type, template."""
    zoo = _read(config_dir() / "models.yaml")
    for k, v in (cfg.get("models") or {}).items():
        zoo[k] = {**zoo.get(k, {}), **v}
    if name not in zoo:
        raise ConfigError(
            f"unknown model {name!r}; available: {', '.join(sorted(zoo))} (add yours to configs/models.yaml)"
        )
    e = zoo[name]
    path = Path(e.get("path") or e["hub"].split("/")[-1])
    if not path.is_absolute():
        path = Path(os.environ.get("MEDRES_MODELS", "models")) / path
    return {
        "name": name,
        "dir": str(path),
        "hub": e.get("hub"),
        "model_type": e["model_type"],
        "template": e.get("template", e["model_type"]),
    }


def data_path(path: str) -> str:
    """Data files are given relative to $MEDRES_DATA (default ./data) or as absolute paths."""
    p = Path(path)
    return str(p if p.is_absolute() else Path(os.environ.get("MEDRES_DATA", "data")) / p)
