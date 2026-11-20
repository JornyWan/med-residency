"""Responses of a model to the prompts of one or several data files, with vLLM. One process per GPU.

The same tool produces the test-set responses of a checkpoint (greedy; scored by `medres eval vqa` or
`medres eval report-score`) and the outputs of a teacher on the training prompts (`medres data teacher-outputs`
turns them into the supervised file of offline KD). The prompt is the user turn of the row, with the image where
`<image>` stands, in the chat template of the model: the format the model sees in training.

    medres eval generate --task cxr_report --model runs/my_run/checkpoint-300 \\
        --data data/cxr_report/test_mimic.jsonl data/cxr_report/test_iu.jsonl \\
        --out evals/my_run/mimic_cxr.jsonl evals/my_run/iu_xray.jsonl
    medres eval generate --task modality_experts --profile teacher_outputs --tag CT --model models/Med-R1/VQA_CT \\
        --data data/modality_experts/train.jsonl --out gen/CT.jsonl

--task takes the response limit, context length and decoding of the task's `generation` block (profile `eval`, or
`teacher_outputs` for the outputs the task's offline data were made with); explicit options win. Without --task,
--max-tokens is required.

Output, one line per prompt: {id, text, n_tokens, truncated[, malformed]}; `id` is the row's own or its row number.
Prints counts only: the responses to credentialed data (MIMIC-CXR) are credentialed data.
"""

from __future__ import annotations

import argparse
import os
import re

from medresidency import config, jsonl
from medresidency.launch import ENV_DEFAULTS

CHUNK = 2000  # prompts whose images are held in memory at a time
OPTIONS = {
    "max_tokens": None,
    "max_model_len": None,
    "max_num_seqs": 64,
    "temperature": 0.0,
    "well_formed": None,
    "retries": 4,
}


def chat_content(user_turn: str) -> list[dict]:
    """The user turn as chat-template content: text before the image (if any), the image, text after it (if any)."""
    pre, tag, post = user_turn.partition("<image>")
    if not tag or "<image>" in post:
        raise ValueError("a prompt needs exactly one <image>")
    content = [{"type": "image"}]
    if pre:
        content.insert(0, {"type": "text", "text": pre})
    if post:
        content.append({"type": "text", "text": post})
    return content


def select(rows: list[dict], tag: str | None = None, shard: str = "0/1", limit: int = 0) -> list[dict]:
    """Rows with their `id` (own or row number), restricted to a teacher tag, a shard i/n and the first `limit`."""
    rows = [{**r, "id": r.get("id", i)} for i, r in enumerate(rows)]
    if tag:
        rows = [r for r in rows if r.get("teacher_tag") == tag]
    i, n = map(int, shard.split("/"))
    rows = rows[i::n]
    return rows[:limit] if limit else rows


def resolve(args: argparse.Namespace) -> dict:
    """Generation settings: explicit options, then the task's profile, then the defaults."""
    profile = {}
    if args.task:
        profiles = config.task(args.task).get("generation") or {}
        if args.profile not in profiles:
            raise config.ConfigError(f"task {args.task} has no generation profile {args.profile!r}")
        profile = profiles[args.profile]
    opt = {}
    for k, default in OPTIONS.items():
        given = getattr(args, k)
        opt[k] = given if given is not None else profile.get(k, default)
    if opt["max_tokens"] is None:
        raise config.ConfigError("give --task or --max-tokens")
    if opt["max_model_len"] is None:
        opt["max_model_len"] = opt["max_tokens"] + 2048
    return opt


def generate(llm, proc, rows: list[dict], opt: dict, seed: int, repetition_penalty: float) -> list[dict]:
    from PIL import Image
    from vllm import SamplingParams

    well_formed = re.compile(opt["well_formed"], re.DOTALL) if opt["well_formed"] else None
    out = [None] * len(rows)
    todo = list(range(len(rows)))
    # Format rejection sampling: a sampled response that is not well formed is drawn again (new seed, same
    # temperature), so that the kept responses are well-formed samples of the model. Greedy responses are not retried.
    retries = opt["retries"] if well_formed and opt["temperature"] > 0 else 0
    for attempt in range(1 + retries):
        sampling = SamplingParams(
            temperature=opt["temperature"],
            max_tokens=opt["max_tokens"],
            seed=seed + attempt,
            repetition_penalty=repetition_penalty,
        )
        for k in range(0, len(todo), CHUNK):
            part = todo[k : k + CHUNK]
            prompts = []
            for i in part:
                msg = [{"role": "user", "content": chat_content(rows[i]["messages"][0]["content"])}]
                text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                image = Image.open(rows[i]["images"][0]).convert("RGB")
                prompts.append({"prompt": text, "multi_modal_data": {"image": image}})
            for i, o in zip(part, llm.generate(prompts, sampling)):
                c = o.outputs[0]
                truncated = int(c.finish_reason == "length")
                out[i] = {"id": rows[i]["id"], "text": c.text, "n_tokens": len(c.token_ids), "truncated": truncated}
        if not well_formed:
            break
        todo = [i for i in todo if not well_formed.fullmatch(out[i]["text"].strip())]
        print(f"attempt {attempt}: {len(todo)} responses not well formed", flush=True)
        if not todo:
            break
    if well_formed:
        bad = set(todo)
        for i, o in enumerate(out):
            o["malformed"] = int(i in bad)
    return out


def main():
    p = argparse.ArgumentParser(
        prog="medres eval generate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="model or checkpoint directory")
    p.add_argument("--data", required=True, nargs="+", help="prompt file(s)")
    p.add_argument("--out", required=True, nargs="+", help="one output file per prompt file")
    p.add_argument("--task", help="take the generation settings of this task")
    p.add_argument("--profile", default="eval", choices=["eval", "teacher_outputs"])
    p.add_argument("--max-tokens", type=int, help="response limit")
    p.add_argument("--max-model-len", type=int, help="context length (default: --max-tokens + 2048)")
    p.add_argument("--max-num-seqs", type=int, help="concurrent sequences (default 64)")
    p.add_argument("--temperature", type=float, help="0 = greedy (default)")
    p.add_argument("--well-formed", help="regular expression a sampled response must match in full")
    p.add_argument("--retries", type=int, help="new samples for a response that is not well formed (default 4)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--tag", help="only the rows of this teacher_tag")
    p.add_argument("--shard", default="0/1", help="i/n: every n-th row starting at row i (one process per GPU)")
    p.add_argument("--limit", type=int, default=0, help="first rows only, for a quick check")
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel size: GPUs of a model too large for one")
    p.add_argument("--gpu-memory", type=float, default=0.85)
    a = p.parse_args()
    if len(a.data) != len(a.out):
        p.error("--out needs one file per --data file")
    if not os.path.isdir(a.model):
        p.error(f"model directory not found: {a.model}")
    try:
        opt = resolve(a)
    except config.ConfigError as e:
        p.error(str(e))
    files = [(select(jsonl.read(data), a.tag, a.shard, a.limit), out) for data, out in zip(a.data, a.out)]
    if not all(rows for rows, _ in files):
        p.error("no rows selected from one of the --data files")
    for name, value in ENV_DEFAULTS.items():  # vLLM is configured as in training, unless the caller decided
        os.environ.setdefault(name, value)
    from transformers import AutoProcessor
    from vllm import LLM

    proc = AutoProcessor.from_pretrained(a.model)
    llm = LLM(
        model=a.model,
        dtype="bfloat16",
        max_model_len=opt["max_model_len"],
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=a.gpu_memory,
        seed=a.seed,
        max_num_seqs=opt["max_num_seqs"],
        tensor_parallel_size=a.tp,
    )
    for rows, out in files:
        gens = generate(llm, proc, rows, opt, a.seed, a.repetition_penalty)
        jsonl.write(out, gens)
        tokens = sorted(g["n_tokens"] for g in gens)
        mean, p50 = sum(tokens) / len(tokens), tokens[len(tokens) // 2]
        summary = f"{len(gens)} responses, tokens mean {mean:.1f} p50 {p50} max {tokens[-1]}"
        summary += f", truncated {sum(g['truncated'] for g in gens)}"
        if opt["well_formed"]:
            summary += f", malformed {sum(g['malformed'] for g in gens)}"
        print(f"{out}: {summary}", flush=True)


if __name__ == "__main__":
    main()
