"""Phase 2 binary reward with delayed length penalty."""

from __future__ import annotations

from verifier.math_verifier import extract_boxed, verify_with_timeout


def binary_reward(
    completions: list[str],
    answers: list[str],
    tokenizer,
    current_step: int = 0,
    **_: object,
) -> list[float]:
    rewards: list[float] = []
    for completion, answer in zip(completions, answers):
        score = verify_with_timeout(completion, answer)

        if extract_boxed(completion) is not None:
            score += 0.1

        if current_step >= 300:
            n_tokens = len(tokenizer.encode(completion))
            score -= 0.0008 * max(0, n_tokens - 200)

        rewards.append(score)
    return rewards

