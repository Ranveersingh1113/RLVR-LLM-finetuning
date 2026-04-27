"""ECE evaluation for MATH-500."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from datasets import load_dataset

from utils.prompts import build_prompt
from verifier.math_verifier import verify_with_timeout


def generate_n(
    model,
    tokenizer,
    problem: str,
    *,
    n: int,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
) -> list[str]:
    prompt = build_prompt(problem, allow_abstention=False, tokenizer=tokenizer)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    encoded = {
        key: value.repeat(n, 1) if value.ndim == 2 else value
        for key, value in encoded.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = encoded["input_ids"].shape[1]
    return [
        tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True)
        for sequence in output_ids
    ]


def eval_calibration(model, tokenizer, K: int = 8, n_bins: int = 10) -> dict[str, Any]:
    """Estimate Expected Calibration Error per difficulty level."""
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    bins_by_level = defaultdict(lambda: [[] for _ in range(n_bins)])

    model.eval()
    for item in dataset:
        level = int(str(item["level"]).split()[-1])
        completions = generate_n(
            model,
            tokenizer,
            item["problem"],
            n=K,
            temperature=0.7,
        )
        n_correct = sum(
            verify_with_timeout(completion, item["answer"]) == 1.0
            for completion in completions
        )
        confidence = n_correct / K
        accuracy = float(n_correct > 0)

        bin_index = min(int(confidence * n_bins), n_bins - 1)
        bins_by_level[level][bin_index].append((confidence, accuracy))

    ece_by_level: dict[str, Any] = {}
    for level, bins in bins_by_level.items():
        total = sum(len(bin_items) for bin_items in bins)
        if total == 0:
            ece_by_level[f"eval/ece_level_{level}"] = 0.0
            continue
        ece = sum(
            (len(bin_items) / total)
            * abs(
                np.mean([value[1] for value in bin_items])
                - np.mean([value[0] for value in bin_items])
            )
            for bin_items in bins
            if bin_items
        )
        ece_by_level[f"eval/ece_level_{level}"] = float(ece)

    model.train()
    return ece_by_level
