"""The task rewards (plugins/reward_plugin.py), imported with a stub in place of ms-swift's reward registry."""

import importlib
import sys
import types

import pytest

from medresidency import methods

PLUGIN = "medresidency.plugins.reward_plugin"


@pytest.fixture
def rewards(monkeypatch):
    orm = types.ModuleType("swift.rewards.orm")
    orm.ORM, orm.orms = object, {}
    for name in ("swift", "swift.rewards"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "swift.rewards.orm", orm)
    monkeypatch.delitem(sys.modules, PLUGIN, raising=False)
    yield importlib.import_module(PLUGIN)
    sys.modules.pop(PLUGIN, None)


def test_registers_the_listed_rewards(rewards):
    assert set(rewards.orms) == set(methods.REWARDS)


def test_mcq_accuracy(rewards):
    completions = ["<think>B fits.</think><answer>B</answer>", "<answer>C</answer>", "The lung is clear."]
    assert rewards.McqAccuracy()(completions, ["<answer>B</answer>"] * 3) == [1.0, 0.0, 0.0]


def test_answer_exact(rewards):
    completions = ["<think>x</think>\n<answer> B </answer>", "B", "<answer>C</answer>"]
    solutions = ["<answer>B</answer>", "B", "<answer>B</answer>"]
    assert rewards.AnswerExact()(completions, solutions) == [1.0, 1.0, 0.0]


def test_mixed_sends_only_reports_to_the_metric_server(rewards, monkeypatch):
    sent = {}

    def radcliq(refs, hyps):
        sent["refs"], sent["hyps"] = refs, hyps
        return [-1.5] * len(refs)

    monkeypatch.setattr(rewards, "_radcliq", radcliq)
    looping_report = " ".join(["opacity"] * 400)
    completions = ["<answer>A</answer>", looping_report, "<answer>D</answer>"]
    solutions = ["<answer>A</answer>", "Findings: clear lungs.", "<answer>B</answer>"]
    assert rewards.MixedReportVqa()(completions, solutions) == [1.0, -1.5, 0.0]
    assert sent["refs"] == ["Findings: clear lungs."]
    assert len(sent["hyps"][0].split()) == rewards.MAX_REPORT_WORDS
