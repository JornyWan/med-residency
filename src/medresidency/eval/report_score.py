"""Report metrics for generated chest X-ray reports (`medres eval generate`), with RadEval.

This file runs in the metrics environment (scripts/install_metrics_env.sh; `medres eval report-score` starts it with
$MEDRES_METRICS_PYTHON), because RadEval cannot be installed next to ms-swift. It imports nothing from medresidency.

Two views of every response:
  fi  the whole response against "Findings: ... Impression: ...", on the studies whose reference has both sections
      (the headline numbers)
  f   the Findings section of the response against the reference findings, on every study with findings
Per study: RadCliQ-v1 and its four parts (RadGraph-F1, BERTScore, SembScore, BLEU-2), RaTEScore, and GREEN with
--green (view fi). Corpus level: 1/RadCliQ-v1 (the reciprocal of the mean), CheXbert micro and macro F1 over 14 and
over 5 findings, share of responses in the Findings/Impression format, length, truncation.

  <prefix>.scores.jsonl    per study: id and numbers (no text), for paired tests (`medres stats compare`)
  <prefix>.summary.json    corpus-level numbers

BERTScore inside RadCliQ-v1 follows the metric's reference code (rajpurkarlab/CXR-Report-Metric: distilroberta-base,
rescaled, no IDF), on which the RadCliQ-v1 weights were fitted; RadEval's adapter adds IDF weights.
GREEN is computed with a length cap (see `green_capped`).

    medres eval report-score --data data/cxr_report/test_mimic.jsonl \\
        --generations evals/my_run/mimic_cxr.jsonl [--green]

Prints numbers only: reports generated from MIMIC-CXR are credentialed data.
"""

import argparse
import json
import os
import re

import numpy as np

FI = re.compile(r"^\s*findings\s*:.*impression\s*:", re.I | re.S)
FIND = re.compile(r"findings\s*:(.*?)(?:impression\s*:|$)", re.I | re.S)
CHEXBERT = ("f1chexbert_all_micro_f1", "f1chexbert_all_macro_f1", "f1chexbert_5_micro_f1", "f1chexbert_5_macro_f1")


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def findings_part(text):
    """The Findings section of a response; a response without the heading counts up to its impression."""
    m = FIND.search(text)
    return norm(m.group(1) if m else re.split(r"impression\s*:", text, flags=re.I)[0])


def views(data, gen):
    """View -> (ids, references, hypotheses); an empty hypothesis becomes "." (the metrics need some text)."""
    ids = sorted(data)
    fi = [i for i in ids if data[i]["findings"] and data[i]["impression"]]
    f = [i for i in ids if data[i]["findings"]]
    return {
        "fi": (fi, [data[i]["reference"] for i in fi], [norm(gen[i]["text"]) or "." for i in fi]),
        "f": (f, [data[i]["findings"] for i in f], [findings_part(gen[i]["text"]) or "." for i in f]),
    }


def official_bertscore(metric):
    """Give RadCliQ-v1 the BERTScore of its reference code (USE_IDF = False) instead of RadEval's IDF-weighted one."""
    from radeval.metrics.bertscore._vendor import BERTScorer

    scorer = BERTScorer(
        model_type="distilroberta-base", num_layers=5, batch_size=64, idf=False, lang="en", rescale_with_baseline=True
    )
    metric._get_bert_scorer = lambda refs: scorer
    return metric


def radcliq_metric():
    from radeval.metrics.RadCliQv1.radcliq import CompositeMetric

    return official_bertscore(CompositeMetric())


def composite(metric, refs, hyps):
    """RadCliQ-v1 (lower is better) and its parts, per study."""
    sub = metric._compute_sub_metrics(refs, hyps)
    X = metric._build_matrix(sub)
    Xn = np.hstack([metric.scaler.transform(X), np.ones((X.shape[0], 1))])
    out = {
        "radcliq": Xn @ metric.coefs,
        "radgraph": sub["radgraph"],
        "bertscore": sub["bertscore"],
        "semb": sub["semb_score"],
        "bleu2": np.asarray(sub["bleu_score"], dtype=float),
    }
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def green_capped(scorer, refs, hyps, max_words=300, max_tokens=512):
    """GREEN per study, with a length cap.

    GREEN pads each batch and gives prompt + judge response 2,048 tokens in all, so one looping hypothesis shrinks
    the response budget of its batch-mates (a judge response cut before "[Matched Findings]" scores 0) or raises an
    error. Each hypothesis is therefore cut to its first `max_words` words and then to `max_tokens` judge tokens
    (words alone do not bound the tokens of looping text), and the studies go through in batches sorted by length,
    so that a report is never batched with longer ones. Only degenerate outputs are cut.

    Returns the scores in input order (None where the judge's response could not be parsed) and the number of
    hypotheses cut."""
    tok = scorer.tokenizer
    capped, lens, cut = [], [], 0
    for r, h in zip(refs, hyps):
        w = h.split()
        h = " ".join(w[:max_words]) or "."
        ids = tok(h, add_special_tokens=False).input_ids
        if len(ids) > max_tokens:
            h = tok.decode(ids[:max_tokens], skip_special_tokens=True).strip() or "."
        cut += len(w) > max_words or len(ids) > max_tokens
        capped.append(h)
        lens.append(len(tok(r, add_special_tokens=False).input_ids) + min(len(ids), max_tokens))
    order = sorted(range(len(refs)), key=lambda i: -lens[i])
    _, _, scores, _ = scorer([refs[i] for i in order], [capped[i] for i in order])
    out = [None] * len(refs)
    for k, i in enumerate(order):
        out[i] = None if scores[k] is None else float(scores[k])
    return out, cut


class Metrics:
    """The metric models, loaded once for all generation files."""

    def __init__(self, text_metrics: bool, green: bool, green_batch: int):
        self.radcliq = self.rate = self.chexbert = self.green = None
        if text_metrics:
            from radeval.metrics.f1chexbert.adapter import F1CheXbertMetric
            from radeval.metrics.RaTEScore.adapter import RaTEScoreMetric

            self.radcliq, self.rate, self.chexbert = radcliq_metric(), RaTEScoreMetric(), F1CheXbertMetric()
        if green:
            from radeval.metrics.green_score.adapter import GreenMetric

            self.green = GreenMetric()._scorer
            self.green.batch_size = green_batch


def score_file(m: Metrics, data: dict, gen_path: str, add_green: bool) -> dict:
    prefix = gen_path[: -len(".jsonl")] if gen_path.endswith(".jsonl") else gen_path
    gen = {r["id"]: r for r in read(gen_path)}
    if set(gen) != set(data):
        raise SystemExit(
            f"error: {gen_path} has responses for {len(set(gen) & set(data))} of {len(data)} studies: "
            "generate for the whole test file (all shards) before scoring"
        )
    ids, vs = sorted(data), views(data, gen)
    if add_green:  # keep the numbers of an earlier scoring of the same responses
        old = {r["id"]: r for r in read(prefix + ".scores.jsonl")}
        if set(old) != set(ids):
            raise SystemExit(f"error: {prefix}.scores.jsonl covers other studies")
        per = {i: {"fi": {k: v for k, v in old[i]["fi"].items() if k != "green"}, "f": old[i]["f"]} for i in ids}
        with open(prefix + ".summary.json") as f:
            summ = {k: v for k, v in json.load(f).items() if k != "fi_green"}
    else:
        per = {i: {"fi": {}, "f": {}} for i in ids}
        summ = {"n_fi": len(vs["fi"][0]), "n_f": len(vs["f"][0])}
        for view, (vids, refs, hyps) in vs.items():
            sc = composite(m.radcliq, refs, hyps)
            sc["ratescore"] = np.asarray(m.rate.compute(refs, hyps, per_sample=True)["ratescore"], dtype=float)
            for k, v in sc.items():
                for i, x in zip(vids, v):
                    per[i][view][k] = float(x)
                summ[f"{view}_{k}"] = float(v.mean())
            summ[f"{view}_inv_radcliq"] = float(1 / sc["radcliq"].mean())
            chex = m.chexbert.compute(refs, hyps)
            summ.update({f"{view}_{k}": float(chex[k]) for k in CHEXBERT})
    if m.green:
        fi, refs, hyps = vs["fi"]
        scores, cut = green_capped(m.green, refs, hyps)
        for i, x in zip(fi, scores):
            per[i]["fi"]["green"] = x
        parsed = [x for x in scores if x is not None]
        summ.update(fi_green=float(np.mean(parsed)), green_unparsed=len(scores) - len(parsed), green_cut=int(cut))
    summ.update(
        n=len(ids),
        format_ok=float(np.mean([bool(FI.search(gen[i]["text"])) for i in ids])),
        tokens_mean=float(np.mean([gen[i]["n_tokens"] for i in ids])),
        truncated=float(np.mean([gen[i]["truncated"] for i in ids])),
    )
    with open(prefix + ".scores.jsonl.tmp", "w") as f:
        for i in ids:
            f.write(json.dumps(dict(id=i, **per[i])) + "\n")
    os.replace(prefix + ".scores.jsonl.tmp", prefix + ".scores.jsonl")
    with open(prefix + ".summary.json", "w") as f:
        json.dump(summ, f, indent=1)
    return summ


def main():
    p = argparse.ArgumentParser(
        prog="medres eval report-score", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", required=True, help="test file with findings, impression and reference of every study")
    p.add_argument("--generations", required=True, nargs="+", help="output file(s) of `medres eval generate`")
    p.add_argument("--green", action="store_true", help="also GREEN (a 7B judge, ~20 GPU-minutes per 2,000 reports)")
    p.add_argument("--add-green", action="store_true", help="only GREEN, added to an earlier scoring of the same files")
    p.add_argument("--green-batch", type=int, default=16)
    a = p.parse_args()
    data = {r["id"]: r for r in read(a.data)}
    m = Metrics(text_metrics=not a.add_green, green=a.green or a.add_green, green_batch=a.green_batch)
    for path in a.generations:
        summ = score_file(m, data, path, a.add_green)
        rounded = {k: round(v, 4) if isinstance(v, float) else v for k, v in summ.items()}
        print(path, json.dumps(rounded), flush=True)


if __name__ == "__main__":
    main()
