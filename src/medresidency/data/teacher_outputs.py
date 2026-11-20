"""Teacher generations + training prompts -> `train_teacher.jsonl`, the supervised file of offline KD and of the
SFT term of two-stage, CHORD, CHORD-phi and AdaOPD.

The generations come from `medres eval generate --model <teacher> --data <task>/train.jsonl` (one file, or one per
shard or per teacher tag); their `id` is the row number of the prompt. Truncated responses and responses that failed
the format check are dropped (sequence-level KD keeps complete outputs only); the rest is written in prompt order,
stripped of surrounding white space.

    medres eval generate --task cxr_report --model models/UniRG-CXR --data data/cxr_report/train.jsonl --out gen.jsonl
    medres data teacher-outputs --prompts data/cxr_report/train.jsonl --generations gen.jsonl \\
        --out data/cxr_report/train_teacher.jsonl

Outputs of a teacher on credentialed data (MIMIC-CXR) are credentialed data. Prints counts only.
"""
from __future__ import annotations

import argparse

from medresidency import jsonl


def join(prompts: list[dict], generations: list[dict]) -> tuple[list[dict], dict]:
    gens = {}
    for g in generations:
        if g["id"] in gens:
            raise SystemExit(f"error: two generations for prompt {g['id']}")
        gens[g["id"]] = g
    if set(gens) != set(range(len(prompts))):
        n = len(set(gens) & set(range(len(prompts))))
        raise SystemExit(f"error: generations for {n} of {len(prompts)} prompts ({len(gens) - n} with an unknown id): "
                         "generate for every row of --prompts (all shards, all teacher tags) before joining")
    rows, dropped = [], {"truncated": 0, "malformed": 0}
    for i, p in enumerate(prompts):
        g = gens[i]
        reason = "truncated" if g["truncated"] else "malformed" if g.get("malformed") else None
        if reason:
            dropped[reason] += 1
        else:
            rows.append(jsonl.answered_row(p, g["text"].strip()))
    tokens = sorted(g["n_tokens"] for g in gens.values())
    return rows, {"prompts": len(prompts), "kept": len(rows), **dropped,
                  "tokens_mean": round(sum(tokens) / len(tokens), 1), "tokens_p50": tokens[len(tokens) // 2],
                  "tokens_p90": tokens[int(len(tokens) * .9)]}


def main():
    p = argparse.ArgumentParser(prog="medres data teacher-outputs", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts", required=True, help="the task's train.jsonl")
    p.add_argument("--generations", required=True, nargs="+", help="output file(s) of `medres eval generate`")
    p.add_argument("--out", required=True, help="the task's train_teacher.jsonl")
    a = p.parse_args()
    rows, stats = join(jsonl.read(a.prompts), [g for f in a.generations for g in jsonl.read(f)])
    jsonl.write(a.out, rows)
    print(a.out, stats)


if __name__ == "__main__":
    main()
