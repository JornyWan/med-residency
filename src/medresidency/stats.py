"""Paired comparison of two models on the same test set, from the per-item files of the scoring tools.

    medres stats compare A.scores.jsonl B.scores.jsonl --metric inv_radcliq \\
        [--cluster-data data/cxr_report/test_mimic.jsonl]
    medres stats compare A.items.jsonl B.items.jsonl --metric accuracy

Reports A - B with a 95% bootstrap interval (1,000 resamples, seed 0). A single run of a method is one draw of its
training randomness, which these intervals do not cover: compare seed by seed, and report the mean over seeds.

Report metrics (`medres eval report-score`): the mean over studies x 100 (radgraph, bertscore, semb, bleu2,
ratescore, green), or `inv_radcliq` = 1 / mean RadCliQ-v1, with the reciprocal taken on every resample. Studies are
resampled, or whole clusters with --cluster-data (the MIMIC-CXR test studies come from far fewer patients, and the
studies of a patient are not independent).

Accuracy (`medres eval vqa`): per domain the difference with an exact McNemar test, and the macro and micro
differences with items resampled inside each domain.
"""

from __future__ import annotations

import argparse
import random
from math import comb

import numpy as np

from medresidency import jsonl

REPORT_METRICS = ("inv_radcliq", "radgraph", "bertscore", "semb", "bleu2", "ratescore", "green")


def paired_scores(a: dict, b: dict, metric: str, n_boot: int = 1000, clusters: dict | None = None) -> dict:
    """Difference of a report metric between two models, with a bootstrap interval.

    `a` and `b` map a study id to the per-study scores of one view. With `clusters` (study id -> cluster, e.g. the
    patient), whole clusters are resampled instead of studies."""
    key = "radcliq" if metric == "inv_radcliq" else metric
    ids = sorted(i for i in a if a[i].get(key) is not None and i in b and b[i].get(key) is not None)
    if not ids:
        raise ValueError(f"no study has {key!r} in both files")
    xa, xb = np.array([a[i][key] for i in ids]), np.array([b[i][key] for i in ids])
    if clusters is None:
        g, n_g = np.arange(len(ids)), len(ids)
    else:
        _, g = np.unique([clusters[i] for i in ids], return_inverse=True)
        n_g = int(g.max()) + 1
    cnt, sa, sb = (np.bincount(g, weights=w, minlength=n_g) for w in (np.ones(len(ids)), xa, xb))

    def stat(m):  # m: [..., n_g] multiplicities of the groups
        ma, mb = (m @ sa) / (m @ cnt), (m @ sb) / (m @ cnt)
        return 1 / ma - 1 / mb if metric == "inv_radcliq" else 100 * (ma - mb)

    draws = np.random.default_rng(0).integers(0, n_g, size=(n_boot, n_g))
    boots = np.sort(stat(np.stack([np.bincount(r, minlength=n_g) for r in draws]).astype(float)))
    nd = 3 if metric == "inv_radcliq" else 2
    return {
        "d": round(float(stat(np.ones(n_g))), nd),
        "lo": round(float(boots[int(0.025 * n_boot)]), nd),
        "hi": round(float(boots[int(0.975 * n_boot) - 1]), nd),
        "n": len(ids),
        "groups": n_g,
    }


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def paired_accuracy(a: list[dict], b: list[dict], n_boot: int = 1000) -> dict:
    """Difference in accuracy between two models; `a` and `b` are their items on the same test file."""
    ca, cb = ({x["id"]: x["correct"] for x in items} for items in (a, b))
    if ca.keys() != cb.keys():
        raise ValueError("the two files hold different test items")
    by = {}
    for x in a:
        by.setdefault(x["domain"], []).append((ca[x["id"]], cb[x["id"]]))

    def macro_micro(sample):
        diffs = {d: sum(x - y for x, y in pairs) for d, pairs in sample.items()}
        macro = 100 * sum(diffs[d] / len(sample[d]) for d in sample) / len(sample)
        micro = 100 * sum(diffs.values()) / sum(len(pairs) for pairs in sample.values())
        return macro, micro

    domains = {}
    for d, pairs in by.items():
        n, ab, ba = len(pairs), sum(x > y for x, y in pairs), sum(y > x for x, y in pairs)
        acc_a, acc_b = 100 * sum(x for x, _ in pairs) / n, 100 * sum(y for _, y in pairs) / n
        domains[d] = {
            "n": n,
            "a": round(acc_a, 2),
            "b": round(acc_b, 2),
            "d": round(acc_a - acc_b, 2),
            "a_only": ab,
            "b_only": ba,
            "p": mcnemar_p(ab, ba),
        }
    rng = random.Random(0)
    boots = []
    for _ in range(n_boot):
        resampled = {d: [pairs[rng.randrange(len(pairs))] for _ in pairs] for d, pairs in by.items()}
        boots.append(macro_micro(resampled))
    out = {"domains": domains, "wins": sum(v["a_only"] > v["b_only"] for v in domains.values()), "n_domains": len(by)}
    for k, name in enumerate(("macro", "micro")):
        s = sorted(x[k] for x in boots)
        out[name] = {
            "d": round(macro_micro(by)[k], 2),
            "lo": round(s[int(0.025 * n_boot)], 2),
            "hi": round(s[int(0.975 * n_boot) - 1], 2),
        }
    return out


def main():
    p = argparse.ArgumentParser(
        prog="medres stats compare", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("a", help="<prefix>.scores.jsonl (reports) or <prefix>.items.jsonl (VQA) of model A")
    p.add_argument("b", help="the same file of model B")
    p.add_argument("--metric", required=True, choices=["accuracy", *REPORT_METRICS])
    p.add_argument("--view", default="fi", choices=["fi", "f"], help="reports: findings and impression, or findings")
    p.add_argument("--cluster-data", help="reports: test file whose rows name the cluster of every study")
    p.add_argument("--cluster-key", default="patient")
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    if a.metric == "accuracy":
        r = paired_accuracy(jsonl.read(a.a), jsonl.read(a.b), a.n_boot)
        print(f"{'domain':<16} {'n':>6} {'A':>7} {'B':>7} {'A-B':>7} {'A>B':>6} {'B>A':>6} {'p':>9}")
        for d, v in r["domains"].items():
            accs = f"{v['a']:7.2f} {v['b']:7.2f} {v['d']:+7.2f}"
            print(f"{d[:16]:<16} {v['n']:6d} {accs} {v['a_only']:6d} {v['b_only']:6d} {v['p']:9.2e}")
        for k in ("macro", "micro"):
            print(f"{k} A-B {r[k]['d']:+.2f} [95% CI {r[k]['lo']:+.2f}, {r[k]['hi']:+.2f}]")
        print(f"A ahead in {r['wins']}/{r['n_domains']} domains")
        return
    clusters = None
    if a.cluster_data:
        clusters = {x["id"]: x[a.cluster_key] for x in jsonl.read(a.cluster_data)}
    scores_a, scores_b = ({x["id"]: x[a.view] for x in jsonl.read(f)} for f in (a.a, a.b))
    r = paired_scores(scores_a, scores_b, a.metric, a.n_boot, clusters)
    unit = f"{r['groups']} {a.cluster_key}s of {r['n']} studies" if clusters else f"{r['n']} studies"
    print(f"{a.metric} ({a.view}) A-B {r['d']:+} [95% CI {r['lo']:+}, {r['hi']:+}]   resampled: {unit}")


if __name__ == "__main__":
    main()
