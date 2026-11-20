import numpy as np
import pytest

from medresidency.adaopd import GapController
from medresidency.probe import gap


def test_token_classes_are_a_partition():
    c = gap.token_classes(6, True, teacher_stops=[0, 1, 0, 0, 1, 1], repeat=[0, 1, 1, 0, 0, 1])
    assert {k: v.astype(int).tolist() for k, v in c.items()} == {
        "eos": [0, 0, 0, 0, 0, 1],       # the last token of a finished response
        "overrun": [0, 1, 0, 0, 1, 0],   # the teacher would have stopped (before any other class)
        "repeat": [0, 0, 1, 0, 0, 0],
        "fresh": [1, 0, 0, 1, 0, 0]}
    assert sum(v.astype(int) for v in c.values()).tolist() == [1] * 6
    cut = gap.token_classes(3, False, [0, 0, 0], [0, 0, 0])  # a response cut at the limit has no end token
    assert cut["fresh"].all() and not cut["eos"].any()


def test_summary():
    recs = [{"A": np.array([1.0, 2.0, 3.0, 4.0]), "vT": np.array([0.5, 0.5, 0.5, 0.5]), "pT": np.array([0.0, 0.9, 0.0, 0.0]),
             "rep": np.array([False, False, True, False]), "finished": True},
            {"A": np.array([-1.0, 0.0]), "vT": np.array([1.5, 0.0]), "pT": np.array([0.0, 0.0]),
             "rep": np.array([False, False]), "finished": False}]
    s = gap.summarize(recs)
    assert (s["n"], s["finished"], s["mean_len"]) == (2, 0.5, 3.0)
    assert (s["n_eos"], s["n_overrun"], s["n_repeat"], s["n_fresh"]) == (1, 1, 1, 3)
    # fresh tokens: the first of response 1 and both of response 2
    assert s["advantage"] == pytest.approx(0.0) and s["visual_evidence"] == pytest.approx(2 / 3)
    assert s["gap"] == pytest.approx(s["visual_evidence"] - s["advantage"])  # advantage = visual evidence - gap
    # the trainer's gap: every token that is not a repeat (overrun and end tokens included)
    assert s["gap_trainer"] == pytest.approx(np.mean([0.5 - 1, 0.5 - 2, 0.5 - 4, 1.5 + 1, 0.0]))


def test_first_step_mu_follows_the_controller():
    # what the probe prints as "AdaOPD would start with mu": the controller's first update
    assert GapController(tau=1.0).update(3.05) == 1.0  # far from the teacher: imitation first
    assert GapController(tau=1.0).update(0.64) == 0.0  # close to the teacher: on-policy distillation from step 1
