"""Uniform static sampler — required for the Novelty B ablation.

Drop-in replacement for AdaptiveDifficultySampler that samples each level with
equal probability regardless of rolling rollout accuracy. The .update() method
is a no-op so reward functions can call it identically. Used to isolate the
contribution of adaptive curriculum vs. RLVR alone in the paper's ablation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class StaticUniformSampler:
    """Sample levels with equal probability; ignore rollout feedback."""

    def __init__(self, dataset_by_level: dict[int, list[dict[str, Any]]]) -> None:
        if not dataset_by_level:
            raise ValueError("dataset_by_level must not be empty")
        for level, items in dataset_by_level.items():
            if not items:
                raise ValueError(f"dataset for level {level} must not be empty")
        self.data = {int(level): list(items) for level, items in dataset_by_level.items()}
        self.levels = sorted(self.data.keys())

    def sample_batch(self, batch_size: int) -> list[dict[str, Any]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        batch: list[dict[str, Any]] = []
        for _ in range(batch_size):
            level = int(np.random.choice(self.levels))
            index = int(np.random.randint(0, len(self.data[level])))
            batch.append({**self.data[level][index], "_sampled_level": level})
        return batch

    def update(self, level: int, correct: bool) -> None:
        # Static sampler ignores rollout feedback by design.
        return None

    def get_stats(self) -> dict[str, float]:
        uniform = 1.0 / len(self.levels)
        return {f"curriculum/weight_level_{level}": uniform for level in self.levels}
