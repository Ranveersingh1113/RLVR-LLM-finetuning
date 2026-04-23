"""Prompt builders shared across training and evaluation."""

from __future__ import annotations

import random


def build_prompt(problem: str, allow_abstention: bool = False) -> str:
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


def sample_prompt(problem: str, phase: int) -> str:
    if phase == 3 and random.random() < 0.30:
        return build_prompt(problem, allow_abstention=True)
    return build_prompt(problem, allow_abstention=False)

