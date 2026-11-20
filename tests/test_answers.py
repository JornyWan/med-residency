import pytest

from medresidency.answers import final_answer, has_answer, is_correct, to_letter


@pytest.mark.parametrize("text, answer, letter", [
    ("<think>it is B</think> The answer is \\boxed{C}", "C", "C"),
    ("first \\boxed{A}, corrected to \\boxed{D}", "D", "D"),
    ("<think>x</think><answer> B </answer>", "B", "B"),
    ("<think>long reasoning</think>\nAnswer: (E)", "Answer: (E)", "E"),
    ("So the answer is (a) pneumonia", "So the answer is (a) pneumonia", "A"),
    ("no option here", "no option here", None),
])
def test_answers(text, answer, letter):
    assert final_answer(text) == answer
    assert to_letter(final_answer(text)) == letter


def test_evaluation_items():
    letter = {"parser": "letter", "answer": "B"}
    assert is_correct(letter, "<think>A? no</think>\\boxed{B}") and not is_correct(letter, "The answer is (C)")
    assert has_answer(letter, "The answer is (C)") and not has_answer(letter, "I cannot tell")
    # Med-R1's evaluation: the first tagged word, compared exactly
    tag = {"parser": "answer_tag", "answer": "B"}
    assert is_correct(tag, "<think>x</think><answer> B </answer>")
    assert not is_correct(tag, "<answer>b</answer>") and not is_correct(tag, "The answer is B")
    assert not is_correct(tag, "<answer>A</answer> or <answer>B</answer>")
    assert has_answer(tag, "<answer>A</answer>") and not has_answer(tag, "B")
    with pytest.raises(ValueError):
        is_correct({"parser": "regex", "answer": "B"}, "B")
