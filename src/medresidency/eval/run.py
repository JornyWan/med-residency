"""Evaluate a model or checkpoint on the test sets of a task: generate, score, summarize.

    medres eval run --task cxr_report --model runs/cxr_report_adaopd_s42/checkpoint-300 [--green]
    medres eval run --task cxr_report --model runs/cxr_report_adaopd_s42/checkpoint-300 --split validation

1. `medres eval generate` on every test set, with the generation settings of the task (configs/tasks/<task>.yaml).
   Generation runs in a process of its own, so that the GPU is free again for the metrics; test sets with the same
   settings share one process. An output file that exists is kept.
2. `medres eval report-score` for report test sets (in the metrics environment: set MEDRES_METRICS_PYTHON) and
   `medres eval vqa` for VQA test sets.
3. <out>/summary.json: the corpus-level numbers of every test set. A later call for the same checkpoint with other
   `--sets` or the other `--split` adds its sets to the file.

Output directory: evals/<run>/<checkpoint> (the last two parts of --model) unless --out is given. Every step is a
tool of its own (`medres eval --help`), for shards over several GPUs or several checkpoints in one scoring pass.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from medresidency import config, jsonl
from medresidency.eval import vqa

# summary key -> column, scale: the numbers printed for a report test set (the summary file has all of them)
REPORT_COLUMNS = (
    ("fi_inv_radcliq", "1/RadCliQ-v1", 1),
    ("fi_radgraph", "RadGraph-F1", 100),
    ("fi_ratescore", "RaTEScore", 100),
    ("fi_f1chexbert_5_micro_f1", "CheXbert-5", 100),
    ("fi_f1chexbert_all_micro_f1", "CheXbert-14", 100),
    ("fi_green", "GREEN", 100),
    ("truncated", "truncated %", 100),
)


def kind(test_file: str) -> str:
    """The kind of a test file: `report` (rows with a reference report) or `vqa` (rows with an answer)."""
    with open(test_file) as f:
        return "report" if "reference" in json.loads(f.readline()) else "vqa"


def generate_commands(sets: list[dict], model: str, extra: list[str]) -> list[list[str]]:
    """One `medres eval generate` command per group of test sets with the same generation settings."""
    groups = {}
    for s in sets:
        groups.setdefault(json.dumps(s["generation"], sort_keys=True), []).append(s)
    commands = []
    for key, group in groups.items():
        files, outs = [s["file"] for s in group], [s["out"] for s in group]
        cmd = [sys.executable, "-m", "medresidency", "eval", "generate", "--model", model]
        cmd += ["--data", *files, "--out", *outs, *extra]
        for k, v in json.loads(key).items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        commands.append(cmd)
    return commands


def headline(kind_: str, summary: dict) -> str:
    if kind_ == "vqa":
        return f"accuracy {summary['micro_acc']:.2f} (macro {summary['macro_acc']:.2f}), n={summary['n']}"
    parts = []
    for k, name, scale in REPORT_COLUMNS:
        if k in summary:
            digits = 3 if scale == 1 else 1
            parts.append(f"{name} {scale * summary[k]:.{digits}f}")
    return ", ".join(parts) + f", n={summary['n_fi']}"


def main():
    p = argparse.ArgumentParser(
        prog="medres eval run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--task", required=True, help="see `medres list tasks`")
    p.add_argument("--model", required=True, help="model or checkpoint directory")
    p.add_argument("--out", help="output directory (default: evals/<run>/<checkpoint>)")
    p.add_argument("--sets", nargs="+", help="only these test sets of the task")
    p.add_argument(
        "--split",
        choices=("test", "validation"),
        default="test",
        help="validation: the task's validation set instead of its test sets (hyper-parameter selection)",
    )
    p.add_argument("--green", action="store_true", help="also GREEN for report test sets (a 7B judge)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel size: GPUs of a model too large for one")
    p.add_argument("--gpu-memory", type=float, default=0.85)
    a = p.parse_args()
    try:
        sets = config.test_sets(a.task, a.split)
    except config.ConfigError as e:
        p.error(str(e))
    unknown = set(a.sets or ()) - {s["name"] for s in sets}
    if unknown:
        names = ", ".join(s["name"] for s in sets)
        p.error(f"task {a.task} has no test set {', '.join(sorted(unknown))}; it has {names}")
    sets = [s for s in sets if not a.sets or s["name"] in a.sets]
    missing = [s["file"] for s in sets if not os.path.isfile(s["file"])]
    if missing:
        p.error(f"test files not found: {', '.join(missing)} (build them with `medres data`, see docs/data.md)")
    model = Path(a.model).resolve()
    out = Path(a.out) if a.out else Path("evals") / model.parent.name / model.name
    for s in sets:
        s["kind"], s["out"] = kind(s["file"]), str(out / f"{s['name']}.jsonl")
    metrics_python = os.environ.get("MEDRES_METRICS_PYTHON")
    if not metrics_python and any(s["kind"] == "report" for s in sets):
        p.error(
            "report test sets are scored in the metrics environment: set MEDRES_METRICS_PYTHON "
            "(scripts/install_metrics_env.sh)"
        )

    todo = [s for s in sets if not os.path.exists(s["out"])]
    for s in sets:
        if s not in todo:
            print(f"{s['out']} exists: kept", flush=True)
    extra = ["--seed", str(a.seed), "--tp", str(a.tp), "--gpu-memory", str(a.gpu_memory)]
    for cmd in generate_commands(todo, a.model, extra):
        subprocess.run(cmd, check=True)

    summary_file = out / "summary.json"
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}  # sets of earlier calls stay
    for s in sets:
        if s["kind"] == "vqa":
            summary[s["name"]] = vqa.score_file(jsonl.read(s["file"]), s["out"])
        else:
            cmd = [metrics_python, "-I", str(Path(__file__).parent / "report_score.py")]
            cmd += ["--data", s["file"], "--generations", s["out"]]
            if a.green:
                cmd.append("--green")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            with open(s["out"].removesuffix(".jsonl") + ".summary.json") as f:
                summary[s["name"]] = json.load(f)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\n{a.model} on {a.task}")
    for s in sets:
        print(f"  {s['name']:<16}{headline(s['kind'], summary[s['name']])}")
    print(f"-> {summary_file}")


if __name__ == "__main__":
    main()
