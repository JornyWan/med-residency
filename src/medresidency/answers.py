"""Answer parsing shared by the VQA evaluation and the VQA rewards, so that training and evaluation score an answer
in the same way."""
from __future__ import annotations

import re

BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
ANSWER_WORD = re.compile(r"<answer>\s*(\w+)\s*</answer>")  # Med-R1's evaluation: the first tagged answer, one word
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


def is_correct(item: dict, text: str) -> bool:
    """is `text` a correct response to an evaluation item? `item["parser"]` names how its answer is read:
    `letter` (the option letter of the final answer) or `answer_tag` (the word inside <answer></answer>, exact)"""
    if item["parser"] == "letter":
        return to_letter(final_answer(text)) == item["answer"]
    if item["parser"] == "answer_tag":
        m = ANSWER_WORD.search(text)
        return bool(m) and m.group(1) == item["answer"]
    raise ValueError(f"unknown answer parser {item['parser']!r}")


def has_answer(item: dict, text: str) -> bool:
    """did the response give an answer in the expected form (right or wrong)?"""
    if item["parser"] == "letter":
        return to_letter(final_answer(text)) is not None
    return bool(ANSWER_WORD.search(text))
