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

    def test_returns_none_when_no_boxed(self) -> None:
        self.assertIsNone(extract_boxed("The answer is 42."))

    def test_extracts_integer(self) -> None:
        self.assertEqual(extract_boxed(r"\boxed{7}"), "7")

    def test_extracts_first_box_when_multiple(self) -> None:
        self.assertEqual(extract_boxed(r"\boxed{3} and \boxed{5}"), "3")

    def test_unbalanced_braces_returns_none(self) -> None:
        self.assertIsNone(extract_boxed(r"\boxed{unclosed"))


class AbstentionTests(unittest.TestCase):
    def test_detects_plain_uncertainty(self) -> None:
        self.assertTrue(is_abstention("I don't know the answer."))

    def test_boxed_answer_is_not_abstention(self) -> None:
        self.assertFalse(is_abstention(r"I am not sure, but \boxed{5}"))

    def test_cannot_determine_is_abstention(self) -> None:
        self.assertTrue(is_abstention("I cannot determine the answer from the given information."))

    def test_insufficient_information_is_abstention(self) -> None:
        self.assertTrue(is_abstention("There is insufficient information to solve this problem."))

    def test_normal_answer_is_not_abstention(self) -> None:
        self.assertFalse(is_abstention("The answer is clearly 42."))


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

    def test_integer_float_equivalence(self) -> None:
        """4.0 and 4 are mathematically the same value."""
        self.assertEqual(verify_with_timeout(r"\boxed{4.0}", "4"), 1.0)

    def test_negative_fraction(self) -> None:
        self.assertEqual(verify_with_timeout(r"\boxed{-\frac{1}{2}}", "-0.5"), 1.0)

    def test_sqrt_simplification(self) -> None:
        self.assertEqual(verify_with_timeout(r"\boxed{\sqrt{9}}", "3"), 1.0)

    def test_latex_in_ground_truth(self) -> None:
        """Ground truth may also be in LaTeX form."""
        self.assertEqual(verify_with_timeout(r"\boxed{\frac{1}{4}}", r"\frac{1}{4}"), 1.0)

    def test_string_equality_fast_path(self) -> None:
        """Exact string match should short-circuit SymPy."""
        self.assertEqual(verify_with_timeout(r"\boxed{42}", "42"), 1.0)

    def test_wrong_answer_returns_zero(self) -> None:
        self.assertEqual(verify_with_timeout(r"\boxed{7}", "8"), 0.0)

    def test_no_boxed_returns_zero(self) -> None:
        self.assertEqual(verify_with_timeout("The answer is 42.", "42"), 0.0)

    def test_timeout_guard_returns_promptly(self) -> None:
        start = time.time()
        reward = verify_with_timeout(r"\boxed{e^{e^{e^{e^{x}}}}}", "1", timeout=2.0)
        elapsed = time.time() - start
        self.assertEqual(reward, 0.0)
        self.assertLess(elapsed, 3.0)

    def test_percentage_equivalence_to_fraction(self) -> None:
        self.assertEqual(verify_with_timeout(r"\boxed{20\%}", "1/5"), 1.0)

    def test_large_integer_correct(self) -> None:
        self.assertEqual(verify_with_timeout(r"\boxed{1000}", "1000"), 1.0)

    def test_large_integer_wrong(self) -> None:
        self.assertEqual(verify_with_timeout(r"\boxed{1001}", "1000"), 0.0)


if __name__ == "__main__":
    unittest.main()
