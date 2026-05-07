"""Phase 2 binary reward.

Length penalty was removed in v3 after diagnostic analysis showed it was actively
counterproductive: a truncated hard-problem rollout (512 tokens, no boxed answer)
received -0.25, while a fast wrong guess (100 tokens with boxed) received +0.10.
This taught the model to guess quickly on hard problems rather than reason longer,
and was the single largest contributor to flat L4/L5 performance.
"""

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
        rewards.append(score)
    return rewards

