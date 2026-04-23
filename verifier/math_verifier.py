"""Math answer verification with hard timeout protection."""

from __future__ import annotations

import concurrent.futures
import re
from typing import Optional

from sympy import N, simplify
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

try:
    from latex2sympy2 import latex2sympy  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    latex2sympy = None

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

IDK_PATTERNS = [
    r"i('m| am) not sure",
    r"i don'?t know",
    r"cannot (determine|solve|compute)",
    r"insufficient information",
    r"\\boxed\{\\text\{(idk|unknown|unclear)\}\}",
    r"no (unique |definitive )?solution",
]


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.strip().split())


def _normalize_percentage(value: str) -> str:
    """Convert percentage strings to decimal strings."""
    stripped = value.strip()
    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*(?:\\%)|([+-]?\d+(?:\.\d+)?)\s*%", stripped)
    if not match:
        return stripped
    number = match.group(1) or match.group(2)
    return str(float(number) / 100.0)


def _strip_outer_braces(value: str) -> str:
    text = value.strip()
    while text.startswith("{") and text.endswith("}"):
        depth = 0
        balanced = True
        for index, char in enumerate(text):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and index != len(text) - 1:
                    balanced = False
                    break
        if balanced and depth == 0:
            text = text[1:-1].strip()
        else:
            break
    return text


def _extract_braced_value(text: str, start: int) -> tuple[str, int]:
    if start >= len(text) or text[start] != "{":
        raise ValueError("Expected opening brace")

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    raise ValueError("Unbalanced braces")


def extract_boxed(text: str) -> Optional[str]:
    """Extract the first non-empty ``\\boxed{...}`` payload."""
    match = re.search(r"\\boxed\{", text)
    if not match:
        return None

    try:
        content, _ = _extract_braced_value(text, match.end() - 1)
    except ValueError:
        return None

    stripped = content.strip()
    return stripped or None


def is_abstention(completion: str) -> bool:
    """Return True when the completion abstains and omits a boxed answer."""
    if extract_boxed(completion) is not None:
        return False

    lowered = completion.lower()
    return any(re.search(pattern, lowered) for pattern in IDK_PATTERNS)


def _replace_latex_commands(expr: str) -> str:
    text = expr
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\cdot", "*").replace("\\times", "*")
    text = text.replace("^", "**")

    while "\\frac" in text:
        frac_index = text.find("\\frac")
        left_start = frac_index + len("\\frac")
        numerator, after_num = _extract_braced_value(text, left_start)
        denominator, after_den = _extract_braced_value(text, after_num)
        replacement = f"(({numerator})/({denominator}))"
        text = text[:frac_index] + replacement + text[after_den:]

    while "\\sqrt" in text:
        sqrt_index = text.find("\\sqrt")
        radicand, after = _extract_braced_value(text, sqrt_index + len("\\sqrt"))
        replacement = f"sqrt({radicand})"
        text = text[:sqrt_index] + replacement + text[after:]

    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:,|!|\;|:)", "", text)
    text = text.replace("{", "(").replace("}", ")")
    return text


def _parse_expression(expr: str):
    cleaned = _strip_outer_braces(_normalize_percentage(_normalize_whitespace(expr)))

    if latex2sympy is not None:
        try:
            return latex2sympy(cleaned)
        except Exception:
            pass

    normalized = _replace_latex_commands(cleaned)
    return parse_expr(normalized, transformations=_TRANSFORMS, evaluate=True)


def _verify_internal(prediction: str, ground_truth: str) -> float:
    pred_expr = _parse_expression(prediction)
    truth_expr = _parse_expression(ground_truth)

    if simplify(pred_expr - truth_expr) == 0:
        return 1.0

    pred_num = complex(N(pred_expr))
    truth_num = complex(N(truth_expr))
    if abs(pred_num - truth_num) < 1e-3:
        return 1.0

    return 0.0


def verify_with_timeout(completion: str, ground_truth: str, timeout: float = 2.0) -> float:
    """
    Verify a completion against a ground-truth answer.

    Fast-path string equality is handled in-process. Anything requiring symbolic
    parsing is isolated in a subprocess so pathological expressions cannot hang
    the caller.
    """
    prediction = extract_boxed(completion)
    if prediction is None:
        return 0.0

    normalized_prediction = _strip_outer_braces(
        _normalize_percentage(_normalize_whitespace(prediction))
    )
    normalized_truth = _strip_outer_braces(
        _normalize_percentage(_normalize_whitespace(ground_truth))
    )

    if not normalized_prediction:
        return 0.0

    if normalized_prediction == normalized_truth:
        return 1.0

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_verify_internal, normalized_prediction, normalized_truth)
        try:
            return float(future.result(timeout=timeout))
        except (concurrent.futures.TimeoutError, Exception):
            return 0.0
