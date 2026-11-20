"""The `medres` command.

    medres list [tasks|methods|models|rewards]
    medres train --task cxr_report --method adaopd [--seed 2] [--name my_run] [key=value ...] [--dry-run]
    medres train configs/recipes/adaopd/cxr_report_4b_adaopd.yaml --seed 2
    medres data ...  |  medres eval ...  |  medres probe ...  |  medres stats ...     (see `medres <group> --help`)
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml

from medresidency import __version__, config, launch, methods

# name -> (module, runs in the report-metric environment, one-line help). Every tool has its own arguments
# (`medres <group> <tool> --help`); the ones of the metrics environment are plain files without package imports,
# executed with $MEDRES_METRICS_PYTHON, because the metric packages cannot be installed next to ms-swift.
TOOLS = {
    "data": {
        "cxr-report": ("data.cxr_report", False, "MIMIC-CXR -> prompts and reference reports for training and validation"),
        "cxr-test": ("data.cxr_test", False, "ReXrank test sets (MIMIC-CXR, IU X-ray) -> evaluation files"),
        "med-vqa": ("data.med_vqa", False, "PMC-VQA / MedXpertQA / MMMU -> reasoning-VQA prompts and test files"),
        "mixed": ("data.mixed", False, "mix report and VQA rows, each tagged with its teacher"),
        "modality-experts": ("data.modality_experts", False, "OmniMedVQA -> prompts tagged with the expert of their modality"),
        "teacher-outputs": ("data.teacher_outputs", False, "responses of a teacher to the training prompts (offline KD, replay)"),
    },
    "eval": {
        "generate": ("eval.generate", False, "greedy responses of a checkpoint on an evaluation file"),
        "vqa": ("eval.vqa", False, "accuracy of generated answers"),
        "report-score": ("eval.report_score", True, "RadCliQ-v1, RadGraph-F1, RaTEScore, CheXbert, BERTScore, BLEU"),
        "report-green": ("eval.report_green", True, "GREEN with a length cap"),
    },
    "probe": {
        "gap": ("probe.gap", False, "language gap and visual evidence of a teacher-student pair before training"),
    },
    "stats": {
        "compare": ("stats", False, "paired bootstrap between two sets of per-item scores"),
    },
}


def _list(what: str | None) -> None:
    def table(rows):
        width = max(len(r[0]) for r in rows)
        for name, text in rows:
            print(f"  {name:<{width}}  {text}")

    for kind in ("tasks", "methods"):
        if what == kind:  # asked for by name: the whole description
            for n in config.available(kind):
                print(f"{n}\n{textwrap.fill(config.describe(kind, n, full=True), 100, initial_indent='    ', subsequent_indent='    ')}\n")
        elif what is None:
            print(f"{kind}:")
            table([(n, config.describe(kind, n)) for n in config.available(kind)])
    if what in (None, "models"):
        zoo = yaml.safe_load((config.config_dir() / "models.yaml").read_text())
        print("models:")
        table([(n, f"{e.get('hub', '')}  [{e['model_type']}]") for n, e in zoo.items()])
    if what in (None, "rewards"):
        print("rewards (method.reward of grpo):")
        table([(n, "needs the report-metric server" if server else "") for n, server in methods.REWARDS.items()])


def _train(a: argparse.Namespace) -> None:
    recipes = [x for x in a.items if "=" not in x]
    if len(recipes) > 1:
        raise config.ConfigError(f"one recipe file at most, got {recipes}")
    overrides = [x for x in a.items if "=" in x]
    overrides += [f"run.{k}={v}" for k, v in (("seed", a.seed), ("name", a.name)) if v is not None]
    cfg = config.load(a.task, a.method, recipes[0] if recipes else None, overrides)
    if a.print_config:
        print(yaml.safe_dump(cfg, sort_keys=False), end="")
        methods.build(cfg)  # still report what the method rejects
        return
    launch.run(cfg, dry_run=a.dry_run)


def _tool(group: str, argv: list[str]) -> None:
    tools = TOOLS[group]
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in tools:
        print(f"usage: medres {group} <tool> [options]     (medres {group} <tool> --help)\n")
        for name, (_, _, text) in tools.items():
            print(f"  {name:<18}{text}")
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 2)
    module, metrics_env, _ = tools[argv[0]]
    if metrics_env:
        python = os.environ.get("MEDRES_METRICS_PYTHON")
        if not python:
            sys.exit("error: set MEDRES_METRICS_PYTHON to the python of the metrics environment (scripts/install_metrics_env.sh)")
        file = Path(__file__).parent / (module.replace(".", "/") + ".py")
        sys.exit(subprocess.run([python, "-I", str(file), *argv[1:]]).returncode)
    sys.argv = [f"medres {group} {argv[0]}", *argv[1:]]
    importlib.import_module(f"medresidency.{module}").main()


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in TOOLS:
        return _tool(argv[0], argv[1:])
    p = argparse.ArgumentParser(prog="medres", description="MedResidency: open post-training of medical vision-language models.",
                                epilog="tool groups with their own help: " + ", ".join(f"medres {g}" for g in TOOLS))
    p.add_argument("--version", action="version", version=f"medresidency {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    ls = sub.add_parser("list", help="tasks, methods, models and rewards")
    ls.add_argument("what", nargs="?", choices=["tasks", "methods", "models", "rewards"])
    tr = sub.add_parser("train", help="train a student with a method on a task",
                        description="Every key of configs/defaults.yaml and of the method's YAML can be set as key=value.")
    tr.add_argument("items", nargs="*", metavar="recipe.yaml | key=value", help="a recipe file and/or overrides such as train.lr=1e-6")
    tr.add_argument("--task", help="see `medres list tasks`")
    tr.add_argument("--method", help="see `medres list methods`")
    tr.add_argument("--seed", type=int, help="run.seed")
    tr.add_argument("--name", help="run.name (default <task>_<method>_s<seed>)")
    tr.add_argument("--dry-run", action="store_true", help="print the commands of the run and start nothing")
    tr.add_argument("--print-config", action="store_true", help="print the resolved configuration and exit")
    if argv and argv[0] == "train":  # its own parser: key=value overrides may come before, between and after options
        a = tr.parse_intermixed_args(argv[1:])
        a.command = "train"
    else:
        a = p.parse_args(argv)
    try:
        if a.command == "list":
            _list(a.what)
        else:
            _train(a)
    except (config.ConfigError, launch.LaunchError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
