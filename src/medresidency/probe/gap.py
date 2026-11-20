"""Gap probe: measure a teacher-student pair before training.

On-policy distillation pushes each token y the student samples by the advantage log p_T(y | image) - log p_S(y | image).
The probe splits that advantage into the two quantities AdaOPD is built on,

    visual evidence   log p_T(y | image) - log p_T(y | no image)    what the image adds to the teacher's judgement
    language gap      log p_S(y | image) - log p_T(y | no image)    how far the student's language is from the teacher's

(advantage = visual evidence - language gap). The untrained student writes one sampled (T = 1) and one greedy
response to N training prompts; student and teacher score every response with and without the image.

Reported per decoding mode, as means over the *fresh* tokens: `gap`, `visual_evidence`, `advantage`. Token classes
(disjoint): eos = the final end-of-sequence token; overrun = another token at a position where the teacher would have
stopped (its p(EOS) > 0.5); repeat = a token that completes a 4-gram already in the response; fresh = the rest.
`gap_trainer` is the quantity the AdaOPD trainer logs as `lang_gap` at its first step: the same mean over every
token that is not a repeat. A gap above tau (1 nat by default) means the student's samples are far from text the
teacher would write; AdaOPD then starts by imitating teacher outputs (mu > 0). The probe measures the starting point
only: in the benchmark, pure on-policy distillation collapsed into repetition with the teacher whose gap stayed above
tau during training, not with teachers whose large initial gap closed within 100 steps (`lang_gap` in the training
log shows which case a run is in; `--method opd method.monitor_gap=true` logs it without changing the training).

    medres probe gap --data data/cxr_report/train.jsonl --student models/Qwen3-VL-2B-Instruct \\
        --teacher models/UniRG-CXR --out probes/unirg_qwen3vl2b
    (reasoning VQA: --max-new-tokens 2048)

One GPU that holds both models. Both models read the prompt in the student's chat format, so the pair must share a
tokenizer (the requirement of per-token on-policy distillation). Writes <out>.json. Prints and writes aggregates
only: the responses to credentialed prompts (MIMIC-CXR) are credentialed data.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

from medresidency import jsonl
from medresidency.adaopd import GapController, fresh_flags
from medresidency.eval.generate import chat_content

CLASSES = ("eos", "overrun", "repeat", "fresh")


def token_classes(n: int, finished: bool, teacher_stops, repeat) -> dict:
    """Boolean masks over the n tokens of one response; `teacher_stops`: teacher's p(EOS) > 0.5 at the position."""
    body = np.ones(n, bool)
    if finished:
        body[-1] = False
    over = body & np.asarray(teacher_stops, bool)
    rep = body & ~over & np.asarray(repeat, bool)
    return {"eos": ~body, "overrun": over, "repeat": rep, "fresh": body & ~over & ~rep}


def summarize(recs: list[dict]) -> dict:
    """Aggregate the scored responses of one decoding mode.

    A record holds the per-token arrays `A` (advantage), `vT` (visual evidence), `pT` (the teacher's p(EOS)) and
    `rep` (4-gram repeats) of one response, and whether it `finished`."""
    cls = [token_classes(len(r["A"]), r["finished"], r["pT"] > 0.5, r["rep"]) for r in recs]

    def cat(key, masks):
        return np.concatenate([r[key][m] for r, m in zip(recs, masks)])

    fresh = [c["fresh"] for c in cls]
    not_repeat = [~r["rep"] for r in recs]
    s = {
        "n": len(recs),
        "finished": float(np.mean([r["finished"] for r in recs])),
        "mean_len": float(np.mean([len(r["A"]) for r in recs])),
        "gap": float((cat("vT", fresh) - cat("A", fresh)).mean()),
        "visual_evidence": float(cat("vT", fresh).mean()),
        "advantage": float(cat("A", fresh).mean()),
        "gap_trainer": float((cat("vT", not_repeat) - cat("A", not_repeat)).mean()),
    }
    s.update({f"n_{c}": int(sum(m[c].sum() for m in cls)) for c in CLASSES})
    return s


def prompt_inputs(proc, rows: list[dict], image: bool = True):
    """Processor inputs of the prompts, with the image or as text only."""
    from PIL import Image

    texts = []
    for r in rows:
        content = chat_content(r["messages"][0]["content"])
        if not image:
            text = "".join(c["text"] for c in content if c["type"] == "text").strip()
            content = [{"type": "text", "text": text}]
        messages = [{"role": "user", "content": content}]
        texts.append(proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    if not image:
        return proc(text=texts, return_tensors="pt", padding=True)
    images = [Image.open(r["images"][0]).convert("RGB") for r in rows]
    return proc(text=texts, images=images, return_tensors="pt", padding=True)


def main():
    p = argparse.ArgumentParser(
        prog="medres probe gap", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", required=True, help="training prompts of the task (train.jsonl)")
    p.add_argument("--student", required=True, help="model directory of the untrained student")
    p.add_argument("--teacher", required=True, help="model directory of the teacher")
    p.add_argument("--tag", help="only the prompts of this teacher_tag (tasks with several teachers)")
    p.add_argument("--n", type=int, default=128, help="prompts")
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=512, help="the response limit of training")
    p.add_argument("--tau", type=float, default=1.0, help="AdaOPD's threshold, for the last line of the report")
    p.add_argument("--out", required=True, help="output prefix")
    a = p.parse_args()
    rows = [r for r in jsonl.read(a.data) if not a.tag or r.get("teacher_tag") == a.tag]
    if len(rows) < a.n:
        p.error(f"{len(rows)} prompts in {a.data}, --n {a.n}")
    rows = random.Random(a.seed).sample(rows, a.n)

    import torch
    import transformers

    def load(path):
        """Load a model with the class named in its config.json."""
        with open(os.path.join(path, "config.json")) as f:
            cls = getattr(transformers, json.load(f)["architectures"][0])
        model = cls.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
        return model.eval()

    torch.manual_seed(a.seed)
    proc = transformers.AutoProcessor.from_pretrained(a.student)
    proc.tokenizer.padding_side = "left"
    student = load(a.student)
    eos = student.generation_config.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    pad = student.generation_config.pad_token_id if student.generation_config.pad_token_id is not None else eos[-1]

    @torch.no_grad()
    def generate(model, sample: bool) -> list[list[int]]:
        """Token ids of one response per prompt, up to and including the first EOS."""
        if sample:
            decoding = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0)
        else:
            decoding = dict(do_sample=False, temperature=None, top_p=None, top_k=None)
        limits = dict(max_new_tokens=a.max_new_tokens, eos_token_id=eos, pad_token_id=pad, repetition_penalty=1.0)
        outs = []
        for i in range(0, len(rows), a.bs):
            x = prompt_inputs(proc, rows[i : i + a.bs]).to(model.device)
            g = model.generate(**x, **limits, **decoding)
            for row in g[:, x["input_ids"].shape[1] :].tolist():
                stop = next((t for t, y in enumerate(row) if y in eos), len(row) - 1)
                outs.append(row[: stop + 1])
        return outs

    @torch.no_grad()
    def score(model, x, comp: list[int]):
        """Per token of the response: log p(token), p(EOS), under the prompt x (one prompt, with or without image)."""
        ids = torch.cat([x["input_ids"][0], torch.tensor(comp, device=model.device)])[None]
        vision = {k: v for k, v in x.items() if k not in ("input_ids", "attention_mask")}
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), **vision).logits
        start = x["input_ids"].shape[1] - 1
        lg = logits[0, start : start + len(comp)].float().log_softmax(-1)
        lp = lg.gather(1, torch.tensor(comp, device=lg.device)[:, None])[:, 0]
        return lp.cpu().numpy(), lg[:, eos].exp().sum(-1).cpu().numpy()

    rolls = {"sampled": generate(student, True), "greedy": generate(student, False)}
    with_image = [prompt_inputs(proc, [r]).to("cuda") for r in rows]
    without_image = [prompt_inputs(proc, [r], image=False).to("cuda") for r in rows]
    sc = {mode: [{} for _ in rows] for mode in rolls}
    for name in ("S", "T"):
        model = student if name == "S" else load(a.teacher)
        for mode, comps in rolls.items():
            for i, comp in enumerate(comps):
                sc[mode][i]["lp" + name], sc[mode][i]["p" + name] = score(model, with_image[i], comp)
                sc[mode][i]["ln" + name] = score(model, without_image[i], comp)[0]
        if name == "S":  # free the student before the teacher is loaded
            del student, model
            torch.cuda.empty_cache()

    res = {
        "data": a.data,
        "student": a.student,
        "teacher": a.teacher,
        "tag": a.tag,
        "seed": a.seed,
        "max_new_tokens": a.max_new_tokens,
        "tau": a.tau,
    }
    for mode, comps in rolls.items():
        recs = [
            {
                "A": d["lpT"] - d["lpS"],
                "vT": d["lpT"] - d["lnT"],
                "pT": d["pT"],
                "rep": ~np.array(fresh_flags(comp)),
                "finished": comp[-1] in eos,
            }
            for comp, d in zip(comps, sc[mode])
        ]
        res[mode] = summarize(recs)
        rounded = {k: round(v, 3) if isinstance(v, float) else v for k, v in res[mode].items()}
        print(mode, json.dumps(rounded), flush=True)

    sampled, greedy = res["sampled"], res["greedy"]
    mu = GapController(tau=a.tau).update(sampled["gap_trainer"])
    res["first_step_mu"] = mu
    if mu >= 1:
        start = "imitation of teacher outputs first"
    elif mu > 0:
        start = "a mixture of imitation and on-policy distillation"
    else:
        start = "on-policy distillation from the first step"
    print(f"language gap {sampled['gap']:.2f} nats on sampled responses (greedy {greedy['gap']:.2f})")
    print(f"visual evidence {sampled['visual_evidence']:+.2f} nats")
    print(f"tau = {a.tau}: AdaOPD would start with mu = {mu:.2f} ({start})")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
