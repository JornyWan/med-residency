import json
import random

import pytest

from medresidency.adaopd import GapController, fresh_flags


def reference_fresh(comp, n=4):
    """The loop of the plugin the paper's runs were trained with."""
    out, seen = [], set()
    for t in range(len(comp)):
        g = tuple(comp[max(0, t - n + 1) : t + 1])
        out.append(not (len(g) == n and g in seen))
        if len(g) == n:
            seen.add(g)
    return out


def test_fresh_flags():
    assert fresh_flags([]) == []
    assert fresh_flags([1, 2, 3]) == [True, True, True]
    # the second "1 2 3 4" completes a 4-gram that is already there; so does every token of the loop after it
    assert fresh_flags([1, 2, 3, 4, 1, 2, 3, 4, 1, 2]) == [True] * 7 + [False] * 3
    rng = random.Random(0)
    for _ in range(200):
        ids = [rng.randrange(3) for _ in range(rng.randrange(40))]
        assert fresh_flags(ids) == reference_fresh(ids)


def test_mu_rule():
    c = GapController(tau=1.0, ema=0.9)
    assert c.mu == 1.0  # before the first measurement the step is SFT only
    assert c.update(3.0) == 1.0 and c.gap0 == 3.0  # first step: EMA = gap0, mu = 1
    assert c.update(1.0) == pytest.approx((0.9 * 3.0 + 0.1 * 1.0 - 1.0) / 2.0)
    for _ in range(100):
        c.update(0.2)
    assert c.mu == 0.0  # below tau: pure OPD


def test_mu_is_zero_for_a_close_teacher():
    c = GapController(tau=1.0)
    assert [c.update(g) for g in (0.64, 0.7, 0.5)] == [0.0, 0.0, 0.0]  # the denominator is at least 1


def test_no_smoothing_and_fixed_gap0():
    c = GapController(tau=1.0, ema=0.0, gap0=5.0)
    assert c.update(3.0) == pytest.approx(0.5) and c.update(1.5) == pytest.approx(0.125)


def test_restore_replays_the_log(tmp_path):
    gaps = [3.0, 2.8, 2.9, 2.0, 1.6, 1.2]
    full = GapController()
    for g in gaps:
        full.update(g)
    rows = [
        {"lang_gap": 3.0, "global_step/max_steps": "1/300"},
        {"lang_gap": 2.8, "global_step/max_steps": "2/300"},
        {"eval_loss": 1.0, "global_step/max_steps": "2/300"},  # a row without a gap
        {"lang_gap": 2.9, "global_step/max_steps": "3/300"},
        {"lang_gap": 7.7, "global_step/max_steps": "4/300"},  # an attempt that was interrupted and later repeated
        {"lang_gap": 2.0, "global_step/max_steps": "4/300"},  # from checkpoint-3: the later value of step 4 counts
        {"lang_gap": 9.9, "global_step/max_steps": "5/300"},  # after the checkpoint we resume from: not replayed
    ]
    log = tmp_path / "logging.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    resumed = GapController()
    assert resumed.restore(str(log), step=4) == 4
    for g in gaps[4:]:
        resumed.update(g)
    assert (resumed.mu, resumed.ema, resumed.gap0) == pytest.approx((full.mu, full.ema, full.gap0))
