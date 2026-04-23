"""Unit tests for the adaptive curriculum sampler."""

from __future__ import annotations

import unittest

from data.difficulty_sampler import AdaptiveDifficultySampler


class AdaptiveDifficultySamplerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = {
            1: [{"problem": "easy", "answer": "1"}],
            2: [{"problem": "medium", "answer": "2"}],
            5: [{"problem": "hard", "answer": "5"}],
        }
        self.sampler = AdaptiveDifficultySampler(self.dataset, window=10)

    def test_initial_weights_are_balanced(self) -> None:
        stats = self.sampler.get_stats()
        self.assertEqual(stats["curriculum/weight_level_1"], 1.0)
        self.assertEqual(stats["curriculum/weight_level_2"], 1.0)
        self.assertEqual(stats["curriculum/weight_level_5"], 1.0)

    def test_weight_has_floor(self) -> None:
        for _ in range(10):
            self.sampler.update(5, correct=False)
        self.assertEqual(self.sampler._weight(5), 0.05)

    def test_sample_batch_tags_level(self) -> None:
        batch = self.sampler.sample_batch(8)
        self.assertEqual(len(batch), 8)
        for item in batch:
            self.assertIn("_sampled_level", item)
            self.assertIn(item["_sampled_level"], self.dataset)


if __name__ == "__main__":
    unittest.main()
