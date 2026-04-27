"""Prompt builders shared across training and evaluation."""

from __future__ import annotations

import random

_ABSTENTION_SYSTEM = (
    "Please reason step by step, and put your final answer within \\boxed{}. "
    "If you are genuinely uncertain, you may write 'I don't know' instead of guessing."
)


def build_prompt(problem: str, *, allow_abstention: bool = False, tokenizer=None) -> str:
    """Build a prompt for the given problem.

    When *tokenizer* is provided the model's native chat template is applied,
    which is required for correct behaviour with Qwen2.5-Math-7B-Instruct.
    Falls back to plain text when tokenizer is None (used by unit tests).
    """
    if tokenizer is not None:
        if allow_abstention:
            messages = [
                {"role": "system", "content": _ABSTENTION_SYSTEM},
                {"role": "user", "content": problem},
            ]
        else:
            messages = [{"role": "user", "content": problem}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Plain-text fallback for tests / environments without a loaded tokenizer.
    if allow_abstention:
        instruction = (
            "Solve the following math problem step by step. "
            "If you are genuinely uncertain, you may respond with "
            "'I don't know' instead of guessing. "
            "Put your final answer in \\boxed{}.\n\n"
        )
    else:
        instruction = (
            "Solve the following math problem step by step. "
            "Put your final answer in \\boxed{}.\n\n"
        )
    return instruction + f"Problem: {problem}\n\nSolution:"


def sample_prompt(problem: str, phase: int, tokenizer=None) -> str:
    if phase == 3 and random.random() < 0.30:
        return build_prompt(problem, allow_abstention=True, tokenizer=tokenizer)
    return build_prompt(problem, allow_abstention=False, tokenizer=tokenizer)

