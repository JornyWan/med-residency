"""Answer parsing shared by the VQA evaluation and the VQA rewards, so that training and evaluation score an answer
in the same way."""
from __future__ import annotations

import re

BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
LETTER = [re.compile(r"^\W*\(?([A-E])\)?\W*$"), re.compile(r"answer is[:\s]*\(?([A-E])\b", re.I),
          re.compile(r"(?i:answer)\s*[:\uff1a]\s*\(?([A-E])\b"), re.compile(r"(?<![A-Za-z])\(?([A-E])\)")]


def final_answer(text: str) -> str:
    """content of the last \\boxed{} or <answer></answer>; otherwise the end of the text after </think>"""
    b = BOXED.findall(text)
    if b:
        return b[-1].strip()
    a = ANSWER.findall(text)
    if a:
        return a[-1].strip()
    return re.split(r"</think(?:ing)?>", text)[-1].strip()[-200:]


def to_letter(answer: str) -> str | None:
    """the option letter (A-E) an answer names, or None"""
    for rx in LETTER:
        m = rx.search(answer)
        if m:
            return m.group(1).upper()
    return None
