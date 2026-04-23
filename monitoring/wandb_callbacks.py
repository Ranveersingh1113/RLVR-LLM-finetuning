"""Monitoring hooks for training metrics."""

from __future__ import annotations

from typing import Any


def merge_metric_dicts(*metric_sets: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for metric_set in metric_sets:
        merged.update(metric_set)
    return merged

