import sys

import numpy as np
import pytest

from medresidency import jsonl, stats


def _scores(n=200):
    base = np.random.default_rng(1).normal(size=n)
    a = {i: {"radgraph": 0.55 + 0.1 * base[i], "radcliq": 1.0 + 0.1 * base[i], "green": None if i < 10 else 0.5}
         for i in range(n)}
    b = {i: {"radgraph": 0.50 + 0.1 * base[i], "radcliq": 1.25 + 0.2 * base[i], "green": 0.4} for i in range(n)}
    return a, b


def test_paired_scores():
    a, b = _scores()
    r = stats.paired_scores(a, b, "radgraph")
    assert r == {"d": 5.0, "lo": 5.0, "hi": 5.0, "n": 200, "groups": 200}  # every pair differs by the same amount
    r = stats.paired_scores(a, b, "inv_radcliq")
    mean = lambda x: np.mean([v["radcliq"] for v in x.values()])
    assert r["d"] == round(1 / mean(a) - 1 / mean(b), 3)  # the reciprocal of the mean, not the mean of reciprocals
    assert r["lo"] < r["d"] < r["hi"] and r["lo"] > 0
    assert stats.paired_scores(b, a, "inv_radcliq")["d"] == -r["d"]
    green = stats.paired_scores(a, b, "green")
    assert (green["n"], green["d"]) == (190, 10.0)  # studies whose judge response could not be parsed are left out
    with pytest.raises(ValueError):
        stats.paired_scores(a, {"other": b[0]}, "radgraph")


def test_clusters_are_resampled_whole():
    a, b = _scores()
    clusters = {i: i // 4 for i in a}
    r = stats.paired_scores(a, b, "inv_radcliq", clusters=clusters)
    assert (r["n"], r["groups"]) == (200, 50)
    assert r["d"] == stats.paired_scores(a, b, "inv_radcliq")["d"]  # the estimate is the same; the interval is not
    # studies of a patient that agree with each other: the clustered interval is wider
    a2 = {i: {"radcliq": 1.0 + 0.3 * ((i // 4) % 5)} for i in range(200)}
    b2 = {i: {"radcliq": 1.2} for i in range(200)}
    wide, narrow = stats.paired_scores(a2, b2, "inv_radcliq", clusters=clusters), stats.paired_scores(a2, b2, "inv_radcliq")
    assert wide["hi"] - wide["lo"] > narrow["hi"] - narrow["lo"]


def test_mcnemar():
    assert stats.mcnemar_p(0, 0) == 1.0 and stats.mcnemar_p(5, 5) == 1.0
    assert stats.mcnemar_p(0, 10) == stats.mcnemar_p(10, 0) == pytest.approx(2 / 1024)
    assert stats.mcnemar_p(3, 0) == 0.25


def _items():
    a = [dict(id=f"x/{i}", domain="x", correct=int(i < 8)) for i in range(10)]
    b = [dict(id=f"x/{i}", domain="x", correct=int(i < 5)) for i in range(10)]
    same = [dict(id=f"y/{i}", domain="y", correct=int(i < 2)) for i in range(4)]
    return a + same, b + same


def test_paired_accuracy():
    a, b = _items()
    r = stats.paired_accuracy(a, b)
    assert r["domains"]["x"] == {"n": 10, "a": 80.0, "b": 50.0, "d": 30.0, "a_only": 3, "b_only": 0, "p": 0.25}
    assert r["domains"]["y"]["d"] == 0.0 and (r["wins"], r["n_domains"]) == (1, 2)
    assert r["macro"]["d"] == 15.0 and r["micro"]["d"] == 21.43
    for k in ("macro", "micro"):
        assert 0 <= r[k]["lo"] <= r[k]["d"] <= r[k]["hi"]
    assert stats.paired_accuracy(a, b) == r  # seeded
    with pytest.raises(ValueError):
        stats.paired_accuracy(a, b[:-1])


def test_compare_command(tmp_path, monkeypatch, capsys):
    a, b = _items()
    jsonl.write(tmp_path / "a.items.jsonl", a)
    jsonl.write(tmp_path / "b.items.jsonl", b)
    monkeypatch.setattr(sys, "argv", ["medres stats compare", str(tmp_path / "a.items.jsonl"), str(tmp_path / "b.items.jsonl"),
                                      "--metric", "accuracy"])
    stats.main()
    out = capsys.readouterr().out
    assert "macro A-B +15.00" in out and "A ahead in 1/2 domains" in out

    sa, sb = _scores(40)
    for name, s in (("a", sa), ("b", sb)):
        jsonl.write(tmp_path / f"{name}.scores.jsonl", [{"id": i, "fi": v, "f": v} for i, v in s.items()])
    jsonl.write(tmp_path / "test.jsonl", [{"id": i, "patient": f"p{i // 4}"} for i in sa])
    monkeypatch.setattr(sys, "argv", ["medres stats compare", str(tmp_path / "a.scores.jsonl"), str(tmp_path / "b.scores.jsonl"),
                                      "--metric", "radgraph", "--cluster-data", str(tmp_path / "test.jsonl")])
    stats.main()
    assert capsys.readouterr().out.strip() == "radgraph (fi) A-B +5.0 [95% CI +5.0, +5.0]   resampled: 10 patients of 40 studies"
