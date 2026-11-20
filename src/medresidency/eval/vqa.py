"""Accuracy of generated answers (`medres eval generate`) on a VQA test file.

Every test row carries its answer, its `domain` (the test set, or the imaging modality) and the `parser` that reads
an answer from a response (medresidency/answers.py, shared with the rewards). For every generations file, writes

  <prefix>.items.jsonl    per item: id, domain, correct, answered, n_tokens, truncated (no text), for paired tests
  <prefix>.summary.json   per domain: n, accuracy, no answer, truncated, mean tokens; micro and macro accuracy

    medres eval vqa --data data/med_vqa/test_pmc_vqa.jsonl --generations evals/my_run/pmc_vqa.jsonl

Several generations files are several models on the same test file. The shards of one model (`generate --shard`)
are concatenated into one file first.
"""

from __future__ import annotations

import argparse
import json

from medresidency import jsonl
from medresidency.answers import has_answer, is_correct


def score(data: list[dict], generations: list[dict]) -> list[dict]:
    gens = {g["id"]: g for g in generations}
    missing = [r["id"] for r in data if r["id"] not in gens]
    if missing or len(gens) != len(data) or len(generations) != len(data):
        raise SystemExit(
            f"error: responses for {len(data) - len(missing)} of {len(data)} test items "
            f"({len(generations)} responses given): generate for the whole file (all shards) before scoring"
        )
    items = []
    for r in data:
        g = gens[r["id"]]
        item = {
            "id": r["id"],
            "domain": r["domain"],
            "correct": int(is_correct(r, g["text"])),
            "answered": int(has_answer(r, g["text"])),
            "n_tokens": g["n_tokens"],
            "truncated": g["truncated"],
        }
        items.append(item)
    return items


def summarize(items: list[dict]) -> dict:
    """Percentages; the macro accuracy is the mean of the domain accuracies."""
    domains, accs = {}, []
    for d in dict.fromkeys(x["domain"] for x in items):
        xs = [x for x in items if x["domain"] == d]
        n = len(xs)
        accs.append(100 * sum(x["correct"] for x in xs) / n)
        domains[d] = {
            "n": n,
            "acc": round(accs[-1], 2),
            "no_answer": round(100 * sum(1 - x["answered"] for x in xs) / n, 2),
            "truncated": round(100 * sum(x["truncated"] for x in xs) / n, 2),
            "mean_tokens": round(sum(x["n_tokens"] for x in xs) / n, 1),
        }
    return {
        "n": len(items),
        "micro_acc": round(100 * sum(x["correct"] for x in items) / len(items), 2),
        "macro_acc": round(sum(accs) / len(accs), 2),
        "domains": domains,
    }


def score_file(data: list[dict], gen_path: str) -> dict:
    prefix = gen_path.removesuffix(".jsonl")
    items = score(data, jsonl.read(gen_path))
    summary = summarize(items)
    jsonl.write(f"{prefix}.items.jsonl", items)
    with open(f"{prefix}.summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def main():
    p = argparse.ArgumentParser(
        prog="medres eval vqa", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", required=True, help="test file")
    p.add_argument("--generations", required=True, nargs="+", help="output file(s) of `medres eval generate`")
    a = p.parse_args()
    data = jsonl.read(a.data)
    for path in a.generations:
        summary = score_file(data, path)
        print(path)
        for d, s in summary["domains"].items():
            rates = f"acc={s['acc']:6.2f}  no answer={s['no_answer']:5.2f}  truncated={s['truncated']:5.2f}"
            print(f"  {d:<16} n={s['n']:<5} {rates}  tokens={s['mean_tokens']}")
        print(f"  micro {summary['micro_acc']:.2f}  macro {summary['macro_acc']:.2f}  (n={summary['n']})")


if __name__ == "__main__":
    main()
