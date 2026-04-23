"""Adaptive difficulty sampling based on rolling rollout accuracy."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class AdaptiveDifficultySampler:
    """Keep each difficulty tier near a 50% success regime."""

    def __init__(self, dataset_by_level: dict[int, list[dict[str, Any]]], window: int = 100):
        if not dataset_by_level:
            raise ValueError("dataset_by_level must not be empty")
        if window <= 0:
            raise ValueError("window must be positive")

        normalized: dict[int, list[dict[str, Any]]] = {}
        for level, items in dataset_by_level.items():
            if not items:
                raise ValueError(f"dataset for level {level} must not be empty")
            normalized[int(level)] = list(items)

        self.data = normalized
        self.levels = sorted(normalized.keys())
        self.acc = {
            level: deque([0.5] * min(20, window), maxlen=window) for level in self.levels
        }

    def _weight(self, level: int) -> float:
        probability = float(np.mean(self.acc[level]))
        return max(0.05, 1.0 - abs(probability - 0.5) * 2.0)

    def sample_batch(self, batch_size: int) -> list[dict[str, Any]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        raw_weights = np.array(
            [self._weight(level) * len(self.data[level]) for level in self.levels],
            dtype=float,
        )
        probabilities = raw_weights / raw_weights.sum()

        batch: list[dict[str, Any]] = []
        for _ in range(batch_size):
            level = int(np.random.choice(self.levels, p=probabilities))
            index = int(np.random.randint(0, len(self.data[level])))
            batch.append({**self.data[level][index], "_sampled_level": level})
        return batch

    def update(self, level: int, correct: bool) -> None:
        if level not in self.acc:
            raise KeyError(f"unknown level: {level}")
        self.acc[level].append(float(correct))

    def get_stats(self) -> dict[str, float]:
        stats: dict[str, float] = {}
        for level in self.levels:
            probability = float(np.mean(self.acc[level]))
            stats[f"curriculum/acc_level_{level}"] = probability
            stats[f"curriculum/weight_level_{level}"] = self._weight(level)
        return stats

