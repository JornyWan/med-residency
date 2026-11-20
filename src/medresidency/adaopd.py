"""AdaOPD's controller, without any deep-learning dependency (the ms-swift glue is plugins/adaopd_plugin.py).

AdaOPD trains with  (1 - mu) * L_OPD + mu * L_SFT(teacher outputs).  At every step it measures the language gap

    g = mean over fresh tokens y of the student's samples of  log p_student(y) - log p_teacher(y | no image),

smooths it over steps (EMA), and sets  mu = clip((EMA(g) - tau) / max(g_0 - tau, 1), 0, 1),  where g_0 is the gap of
the first step. A student whose language is far from the teacher's (large gap) imitates the teacher's outputs; once
the gap is below tau the step is pure on-policy distillation.
"""

from __future__ import annotations

import json
import math


def fresh_flags(ids, n: int = 4) -> list[bool]:
    """For each token of one response: False if it completes an n-gram that already occurred in the response.

    Only fresh tokens measure the gap: a student that loops repeats n-grams the teacher also finds likely once they
    have been written, so its repeated tokens would make it look close to the teacher."""
    seen, flags = set(), []
    for t in range(len(ids)):
        gram = tuple(ids[max(0, t - n + 1) : t + 1])
        full = len(gram) == n
        flags.append(not (full and gram in seen))
        if full:
            seen.add(gram)
    return flags


class GapController:
    """Computes mu from the measured language gap; `update` is called once per optimizer step."""

    def __init__(self, tau: float = 1.0, ema: float = 0.9, gap0: float | None = None):
        self.tau, self.decay, self.gap0 = tau, ema, gap0
        self.ema = None
        self.mu = 1.0

    def update(self, gap: float) -> float:
        self.ema = gap if self.ema is None else self.decay * self.ema + (1 - self.decay) * gap
        if self.gap0 is None:
            self.gap0 = gap
        self.mu = min(1.0, max(0.0, (self.ema - self.tau) / max(self.gap0 - self.tau, 1.0)))
        return self.mu

    def restore(self, logging_jsonl: str, step: int) -> int:
        """Replay the gaps a run logged up to `step` (ms-swift's logging.jsonl).

        A resumed run then continues with the EMA and the first-step gap of the original run. Returns the number of
        steps replayed."""
        gaps = {}
        with open(logging_jsonl) as f:
            for line in f:
                r = json.loads(line) if line.lstrip().startswith("{") else {}
                g = r.get("lang_gap")
                s = int(str(r.get("global_step/max_steps", "0/0")).split("/")[0])
                if g is not None and math.isfinite(g) and s <= step:
                    gaps[s] = g  # a step logged twice (an earlier resume) counts once, with its last value
        for s in sorted(gaps):
            self.update(gaps[s])
        return len(gaps)
