"""Unit tests for dataset normalization helpers."""

from __future__ import annotations

import unittest

from data.prepare_dataset import (
    _normalize_gsm8k_item,
    _normalize_math_item,
    _parse_level,
    group_by_level,
)


class PrepareDatasetTests(unittest.TestCase):
    def test_parse_level_from_string(self) -> None:
        self.assertEqual(_parse_level("Level 4"), 4)

    def test_normalize_math_item(self) -> None:
        item = {
            "problem": "What is 1/2 as a decimal?",
            "solution": r"We compute it and get \boxed{0.5}.",
            "level": "Level 2",
            "type": "algebra",
        }
        normalized = _normalize_math_item(item)
        self.assertEqual(normalized["answer"], "0.5")
        self.assertEqual(normalized["level"], 2)
        self.assertEqual(normalized["subject"], "algebra")
        self.assertEqual(normalized["source"], "math")

    def test_normalize_gsm8k_item(self) -> None:
        item = {
            "question": "If Tom has 40 apples and gives away 1, how many remain?",
            "answer": "Tom starts with 40 and gives away 1. #### 39",
        }
        normalized = _normalize_gsm8k_item(item)
        self.assertEqual(normalized["problem"], item["question"])
        self.assertEqual(normalized["answer"], "39")
        self.assertEqual(normalized["level"], 1)
        self.assertEqual(normalized["source"], "gsm8k")

    def test_group_by_level(self) -> None:
        grouped = group_by_level(
            [
                {"problem": "a", "answer": "1", "level": 1, "subject": "x", "source": "math"},
                {"problem": "b", "answer": "2", "level": 2, "subject": "y", "source": "math"},
                {"problem": "c", "answer": "3", "level": 1, "subject": "z", "source": "gsm8k"},
            ]
        )
        self.assertEqual(len(grouped[1]), 2)
        self.assertEqual(len(grouped[2]), 1)


if __name__ == "__main__":
    unittest.main()
