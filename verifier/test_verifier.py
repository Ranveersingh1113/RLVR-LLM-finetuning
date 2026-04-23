"""Acceptance tests for the math verifier."""

from __future__ import annotations

import time
import unittest

from verifier.math_verifier import extract_boxed, is_abstention, verify_with_timeout


class ExtractBoxedTests(unittest.TestCase):
    def test_extracts_nested_expression(self) -> None:
        value = extract_boxed(r"Result is \boxed{\frac{1}{2} + \sqrt{4}}")
        self.assertEqual(value, r"\frac{1}{2} + \sqrt{4}")

    def test_returns_none_for_empty_box(self) -> None:
        self.assertIsNone(extract_boxed(r"\boxed{}"))


class AbstentionTests(unittest.TestCase):
    def test_detects_plain_uncertainty(self) -> None:
        self.assertTrue(is_abstention("I don't know the answer."))

    def test_boxed_answer_is_not_abstention(self) -> None:
        self.assertFalse(is_abstention(r"I am not sure, but \boxed{5}"))


class VerifyWithTimeoutTests(unittest.TestCase):
    def test_acceptance_cases(self) -> None:
        self.assertEqual(
            verify_with_timeout(r"The answer is \boxed{\frac{1}{2}}", "0.5"),
            1.0,
        )
        self.assertEqual(verify_with_timeout(r"\boxed{-3}", "{-3}"), 1.0)
        self.assertEqual(verify_with_timeout(r"\boxed{\sqrt{4}}", "2"), 1.0)
        self.assertEqual(verify_with_timeout(r"\boxed{2x + 1}", "2x+1"), 1.0)
        self.assertEqual(verify_with_timeout(r"\boxed{0.333}", "1/3"), 1.0)
        self.assertEqual(verify_with_timeout(r"\boxed{50\%}", "0.5"), 1.0)
        self.assertEqual(verify_with_timeout(r"\boxed{25\%}", "1/4"), 1.0)
        self.assertEqual(verify_with_timeout(r"I think it might be 5", "5"), 0.0)
        self.assertEqual(verify_with_timeout(r"\boxed{6}", "5"), 0.0)
        self.assertEqual(verify_with_timeout(r"\boxed{}", "5"), 0.0)

    def test_timeout_guard_returns_promptly(self) -> None:
        start = time.time()
        reward = verify_with_timeout(r"\boxed{e^{e^{e^{e^{x}}}}}", "1", timeout=2.0)
        elapsed = time.time() - start
        self.assertEqual(reward, 0.0)
        self.assertLess(elapsed, 3.0)


if __name__ == "__main__":
    unittest.main()
