"""Tests for the Phase 3 ternary reward function."""

from __future__ import annotations

import unittest

from rewards.ternary_reward import make_ternary_reward_fn


class _FakeTokenizer:
    def __init__(self, n_tokens: int = 10) -> None:
        self._n_tokens = n_tokens

    def encode(self, text: str) -> list[int]:
        return list(range(self._n_tokens))


_SHORT_TOKENIZER = _FakeTokenizer(n_tokens=10)   # well under 200-token threshold
_LONG_TOKENIZER = _FakeTokenizer(n_tokens=500)   # 300 tokens over threshold

_PHASE3_START = 1000
_WARMUP = 50

# Convenience: reward fn whose alpha is fully ramped (step = start + warmup)
_FULL_ALPHA_STEP = _PHASE3_START + _WARMUP

# Convenience: reward fn called before phase 3 begins
_PRE_PHASE3_STEP = _PHASE3_START - 1

_reward_fn = make_ternary_reward_fn(
    phase3_start_step=_PHASE3_START,
    warmup_steps=_WARMUP,
)


class TernaryRewardCorrectAnswerTests(unittest.TestCase):
    def test_correct_answer_always_gives_1_0(self) -> None:
        for step in (_PRE_PHASE3_STEP, _PHASE3_START, _FULL_ALPHA_STEP):
            with self.subTest(step=step):
                rewards = _reward_fn(
                    completions=[r"\boxed{4}"],
                    answers=["4"],
                    tokenizer=_SHORT_TOKENIZER,
                    current_step=step,
                )
                self.assertAlmostEqual(rewards[0], 1.0, msg=f"step={step}")

    def test_correct_answer_has_no_length_penalty(self) -> None:
        """Correct completions are never length-penalised."""
        rewards = _reward_fn(
            completions=[r"\boxed{4}"],
            answers=["4"],
            tokenizer=_LONG_TOKENIZER,
            current_step=_FULL_ALPHA_STEP,
        )
        self.assertAlmostEqual(rewards[0], 1.0)


class TernaryRewardHallucinationTests(unittest.TestCase):
    """Wrong answer with a \boxed{} is treated as a hallucination."""

    def test_hallucination_before_phase3_gives_zero(self) -> None:
        # alpha=0 → hallucination_penalty = -1.5 * 0 = 0.0
        rewards = _reward_fn(
            completions=[r"\boxed{99}"],
            answers=["4"],
            tokenizer=_SHORT_TOKENIZER,
            current_step=_PRE_PHASE3_STEP,
        )
        self.assertAlmostEqual(rewards[0], 0.0)

    def test_hallucination_fully_ramped_gives_penalty(self) -> None:
        # alpha=1 → hallucination_penalty = -1.5
        rewards = _reward_fn(
            completions=[r"\boxed{99}"],
            answers=["4"],
            tokenizer=_SHORT_TOKENIZER,
            current_step=_FULL_ALPHA_STEP,
        )
        self.assertAlmostEqual(rewards[0], -1.5)

    def test_hallucination_penalty_scaled_by_alpha(self) -> None:
        # at step = start + warmup/2 → alpha = 0.5 → penalty = -0.75
        mid_step = _PHASE3_START + _WARMUP // 2
        rewards = _reward_fn(
            completions=[r"\boxed{99}"],
            answers=["4"],
            tokenizer=_SHORT_TOKENIZER,
            current_step=mid_step,
        )
        self.assertAlmostEqual(rewards[0], -1.5 * 0.5)


class TernaryRewardAbstentionTests(unittest.TestCase):
    """Explicit abstentions earn a small positive reward once phase 3 is active."""

    def test_abstention_before_phase3_gives_zero(self) -> None:
        rewards = _reward_fn(
            completions=["I don't know the answer."],
            answers=["4"],
            tokenizer=_SHORT_TOKENIZER,
            current_step=_PRE_PHASE3_STEP,
        )
        self.assertAlmostEqual(rewards[0], 0.0)

    def test_abstention_fully_ramped_gives_positive_reward(self) -> None:
        rewards = _reward_fn(
            completions=["I don't know the answer."],
            answers=["4"],
            tokenizer=_SHORT_TOKENIZER,
            current_step=_FULL_ALPHA_STEP,
        )
        self.assertAlmostEqual(rewards[0], 0.15)

    def test_abstention_has_no_length_penalty(self) -> None:
        """Abstentions skip the length-penalty block (early continue)."""
        rewards = _reward_fn(
            completions=["I don't know the answer."],
            answers=["4"],
            tokenizer=_LONG_TOKENIZER,
            current_step=_FULL_ALPHA_STEP,
        )
        self.assertAlmostEqual(rewards[0], 0.15)


class TernaryRewardNoBoxedWrongTests(unittest.TestCase):
    """Wrong answer with no boxed gets a mild negative reward scaled by alpha."""

    def test_no_boxed_wrong_before_phase3_gives_zero(self) -> None:
        # alpha=0 → -0.3 * 0 = 0.0
        rewards = _reward_fn(
            completions=["The answer is certainly 99."],
            answers=["4"],
            tokenizer=_SHORT_TOKENIZER,
            current_step=_PRE_PHASE3_STEP,
        )
        self.assertAlmostEqual(rewards[0], 0.0)

    def test_no_boxed_wrong_fully_ramped_gives_mild_penalty(self) -> None:
        # alpha=1 → -0.3 * 1 = -0.3
        rewards = _reward_fn(
            completions=["The answer is certainly 99."],
            answers=["4"],
            tokenizer=_SHORT_TOKENIZER,
            current_step=_FULL_ALPHA_STEP,
        )
        self.assertAlmostEqual(rewards[0], -0.3)


class TernaryRewardMultipleCompletionsTests(unittest.TestCase):
    def test_mixed_batch(self) -> None:
        rewards = _reward_fn(
            completions=[
                r"\boxed{4}",                   # correct
                r"\boxed{99}",                  # hallucination
                "I don't know the answer.",     # abstention
            ],
            answers=["4", "4", "4"],
            tokenizer=_SHORT_TOKENIZER,
            current_step=_FULL_ALPHA_STEP,
        )
        self.assertAlmostEqual(rewards[0], 1.0)    # correct
        self.assertAlmostEqual(rewards[1], -1.5)   # hallucination penalty
        self.assertAlmostEqual(rewards[2], 0.15)   # abstention reward


if __name__ == "__main__":
    unittest.main()
