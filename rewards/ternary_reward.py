"""Phase 3 reward and prompt helpers for calibrated abstention."""

from __future__ import annotations

import functools
import random

from verifier.math_verifier import extract_boxed, is_abstention, verify_with_timeout


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


def _ternary_reward_core(
    completions: list[str],
    answers: list[str],
    tokenizer,
    current_step: int,
    phase3_start_step: int,
    warmup_steps: int = 50,
    **_: object,
) -> list[float]:
    steps_into_phase3 = current_step - phase3_start_step
    alpha = min(1.0, max(0.0, steps_into_phase3 / warmup_steps))

    hallucination_penalty = -1.5 * alpha
    abstention_reward = 0.15 * alpha

    rewards: list[float] = []
    for completion, answer in zip(completions, answers):
        if is_abstention(completion):
            rewards.append(abstention_reward)
            continue

        score = verify_with_timeout(completion, answer)
        if score == 1.0:
            rewards.append(1.0)
        elif extract_boxed(completion) is not None:
            rewards.append(hallucination_penalty)
        else:
            rewards.append(-0.3 * alpha)

        if rewards[-1] != 1.0:
            n_tokens = len(tokenizer.encode(completion))
            rewards[-1] -= 0.0008 * max(0, n_tokens - 200)

    return rewards


def make_ternary_reward_fn(phase3_start_step: int, warmup_steps: int = 50):
    return functools.partial(
        _ternary_reward_core,
        phase3_start_step=phase3_start_step,
        warmup_steps=warmup_steps,
    )

