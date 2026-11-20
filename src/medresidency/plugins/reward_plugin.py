"""Task rewards for GRPO (loaded into ms-swift with --external_plugins; chosen with `method.reward`).

  report_radcliq    -RadCliQ-v1 of a report against the reference report of its study, scored by the report-metric
                    server (eval/reward_server.py) that the launcher starts and names in $MEDRES_REWARD_URL
  mcq_accuracy      1 if the option letter parsed from the response is the solution's, else 0
  answer_exact      1 if the content of <answer></answer> equals the solution's (Med-R1's accuracy reward)
  mixed_report_vqa  per row: mcq_accuracy if the solution is "<answer>X</answer>", else report_radcliq. GRPO normalizes
                    rewards within the samples of one prompt, so the two scales never meet.

A reward receives the sampled responses and the `solution` field of their rows. To add one, write a class with
`__call__(self, completions, solution, **kwargs) -> list[float]`, register it in `orms`, and list it in
medresidency/methods.py (REWARDS).
"""

import json
import os
import re
import urllib.request

from swift.rewards.orm import ORM, orms

from medresidency.answers import ANSWER, final_answer, to_letter

ONLY_ANSWER = re.compile(r"^\s*<answer>(.*?)</answer>\s*$", re.DOTALL)
ANSWER_RAW = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# RadGraph's memory grows quadratically with length: one looping 2,048-token report exhausts a GPU.
MAX_REPORT_WORDS = 300


def _radcliq(refs, hyps) -> list[float]:
    req = urllib.request.Request(
        os.environ["MEDRES_REWARD_URL"] + "/score",
        data=json.dumps({"refs": list(refs), "hyps": list(hyps)}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["rewards"]


class ReportRadCliQ(ORM):
    def __call__(self, completions, solution, **kwargs) -> list[float]:
        return _radcliq(solution, completions)


class McqAccuracy(ORM):
    def __call__(self, completions, solution, **kwargs) -> list[float]:
        return [float(to_letter(final_answer(c)) == ANSWER.search(s).group(1)) for c, s in zip(completions, solution)]


def _answer(text: str) -> str:
    """The content of <answer></answer>, or the whole text when the tags are missing."""
    m = ANSWER_RAW.search(text)
    return (m.group(1) if m else text).strip()


class AnswerExact(ORM):
    def __call__(self, completions, solution, **kwargs) -> list[float]:
        return [float(_answer(c) == _answer(s)) for c, s in zip(completions, solution)]


class MixedReportVqa(ORM):
    def __call__(self, completions, solution, **kwargs) -> list[float]:
        out, reports = [0.0] * len(completions), []
        for i, (c, s) in enumerate(zip(completions, solution)):
            m = ONLY_ANSWER.match(s)
            if m:
                out[i] = float(to_letter(final_answer(c)) == m.group(1).strip())
            else:
                reports.append(i)
        if reports:
            hyps = [" ".join(completions[i].split()[:MAX_REPORT_WORDS]) for i in reports]
            for i, v in zip(reports, _radcliq([solution[i] for i in reports], hyps)):
                out[i] = v
        return out


orms["report_radcliq"] = ReportRadCliQ
orms["mcq_accuracy"] = McqAccuracy
orms["answer_exact"] = AnswerExact
orms["mixed_report_vqa"] = MixedReportVqa
