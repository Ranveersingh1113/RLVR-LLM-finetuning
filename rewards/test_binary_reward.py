"""Tests for the Phase 2 binary reward function."""

from __future__ import annotations

import unittest

from rewards.binary_reward import binary_reward


class _FakeTokenizer:
    """Tokenizer stub whose token count equals the number of whitespace-split words."""

    def __init__(self, n_tokens: int = 10) -> None:
        self._n_tokens = n_tokens

    def encode(self, text: str) -> list[int]:
        return list(range(self._n_tokens))


_SHORT_TOKENIZER = _FakeTokenizer(n_tokens=10)   # well under 200-token threshold
_LONG_TOKENIZER = _FakeTokenizer(n_tokens=500)   # 300 tokens over threshold


class BinaryRewardCorrectnessTests(unittest.TestCase):
    """Verify that the verifier signal is correctly propagated."""

    def test_correct_answer_gives_full_reward(self) -> None:
        # verify_with_timeout returns 1.0, plus 0.1 format bonus for \boxed{}
        rewards = binary_reward(
            completions=[r"\boxed{4}"],
            answers=["4"],
            tokenizer=_SHORT_TOKENIZER,
            current_step=0,
        )
        self.assertAlmostEqual(rewards[0], 1.1)

    def test_wrong_answer_with_boxed_gives_format_bonus_only(self) -> None:
        # verify_with_timeout returns 0.0, but \boxed{} earns the 0.1 format bonus
        rewards = binary_reward(
            completions=[r"\boxed{99}"],
            answers=["4"],
            tokenizer=_SHORT_TOKENIZER,
            current_step=0,
        )
        self.assertAlmostEqual(rewards[0], 0.1)

    def test_wrong_answer_without_boxed_gives_zero(self) -> None:
        rewards = binary_reward(
            completions=["The answer is definitely 99."],
            answers=["4"],
            tokenizer=_SHORT_TOKENIZER,
            current_step=0,
        )
        self.assertAlmostEqual(rewards[0], 0.0)

    def test_empty_completion_gives_zero(self) -> None:
        rewards = binary_reward(
            completions=[""],
            answers=["4"],
            tokenizer=_SHORT_TOKENIZER,
            current_step=0,
        )
        self.assertAlmostEqual(rewards[0], 0.0)

    def test_multiple_completions_independent(self) -> None:
        rewards = binary_reward(
            completions=[r"\boxed{4}", r"\boxed{99}"],
            answers=["4", "4"],
            tokenizer=_SHORT_TOKENIZER,
            current_step=0,
        )
        self.assertAlmostEqual(rewards[0], 1.1)
        self.assertAlmostEqual(rewards[1], 0.1)


class BinaryRewardLengthPenaltyTests(unittest.TestCase):
    """Length penalty activates at step >= 300 and penalises tokens > 200."""

    def test_no_length_penalty_before_step_300(self) -> None:
        # Even with a long completion, step < 300 means no penalty
        rewards = binary_reward(
            completions=[r"\boxed{4}"],
            answers=["4"],
            tokenizer=_LONG_TOKENIZER,
            current_step=299,
        )
        self.assertAlmostEqual(rewards[0], 1.1)

    def test_length_penalty_applied_at_step_300(self) -> None:
        # n_tokens=500, free threshold=200 → excess=300 → penalty=0.0008*300=0.24
        # Base score for correct + boxed = 1.1, after penalty = 0.86
        rewards = binary_reward(
            completions=[r"\boxed{4}"],
            answers=["4"],
            tokenizer=_LONG_TOKENIZER,
            current_step=300,
        )
        expected = 1.1 - 0.0008 * (500 - 200)
        self.assertAlmostEqual(rewards[0], expected, places=5)

    def test_no_length_penalty_when_under_threshold(self) -> None:
        # n_tokens=10, free threshold=200 → no excess → no penalty
        rewards = binary_reward(
            completions=[r"\boxed{4}"],
            answers=["4"],
            tokenizer=_SHORT_TOKENIZER,
            current_step=300,
        )
        self.assertAlmostEqual(rewards[0], 1.1)

    def test_length_penalty_on_wrong_answer(self) -> None:
        # Wrong answer (score=0.0), has \boxed{} (+0.1), long completion at step 300
        rewards = binary_reward(
            completions=[r"\boxed{99}"],
            answers=["4"],
            tokenizer=_LONG_TOKENIZER,
            current_step=300,
        )
        expected = 0.1 - 0.0008 * (500 - 200)
        self.assertAlmostEqual(rewards[0], expected, places=5)


class BinaryRewardFractionEquivalenceTests(unittest.TestCase):
    """Ensure the verifier correctly recognises mathematically equivalent answers."""

    def test_fraction_and_decimal_equivalent(self) -> None:
        rewards = binary_reward(
            completions=[r"The answer is \boxed{\frac{1}{2}}"],
            answers=["0.5"],
            tokenizer=_SHORT_TOKENIZER,
            current_step=0,
        )
        self.assertAlmostEqual(rewards[0], 1.1)

    def test_sqrt_simplification(self) -> None:
        rewards = binary_reward(
            completions=[r"\boxed{\sqrt{9}}"],
            answers=["3"],
            tokenizer=_SHORT_TOKENIZER,
            current_step=0,
        )
        self.assertAlmostEqual(rewards[0], 1.1)

    def test_percentage_and_decimal_equivalent(self) -> None:
        rewards = binary_reward(
            completions=[r"\boxed{50\%}"],
            answers=["0.5"],
            tokenizer=_SHORT_TOKENIZER,
            current_step=0,
        )
        self.assertAlmostEqual(rewards[0], 1.1)


if __name__ == "__main__":
    unittest.main()
