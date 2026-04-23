"""Unit tests for training helpers."""

from __future__ import annotations

import unittest
import warnings

from data.difficulty_sampler import AdaptiveDifficultySampler
from training.common import _extract_levels, make_phase2_reward


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(500))


class TrainingCommonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sampler = AdaptiveDifficultySampler(
            {
                1: [{"problem": "easy", "answer": "4"}],
                2: [{"problem": "hard", "answer": "5"}],
            },
            window=10,
        )

    def test_extract_levels_warns_when_missing(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            levels = _extract_levels({})
        self.assertEqual(levels, [])
        self.assertTrue(any("No level data found" in str(item.message) for item in caught))

    def test_phase2_sampler_uses_verifier_correctness_not_reward_threshold(self) -> None:
        reward_fn = make_phase2_reward(tokenizer=_FakeTokenizer(), sampler=self.sampler)
        rewards = reward_fn(
            completions=[r"\boxed{4} " + "word " * 250],
            answers=["4"],
            _sampled_level=[1],
            current_step=300,
        )
        self.assertLess(rewards[0], 1.0)
        self.assertEqual(self.sampler.acc[1][-1], 1.0)


if __name__ == "__main__":
    unittest.main()
