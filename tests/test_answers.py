import pytest

from medresidency.answers import final_answer, to_letter


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
