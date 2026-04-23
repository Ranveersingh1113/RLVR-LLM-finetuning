"""Fast Pass@1 evaluation for MATH-500."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from datasets import load_dataset

from utils.prompts import build_prompt
from verifier.math_verifier import is_abstention, verify_with_timeout


def generate_one(
    model,
    tokenizer,
    prompt: str,
    *,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    do_sample = temperature > 0.0
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5) if do_sample else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completion_ids = output_ids[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def eval_pass1(model, tokenizer) -> dict[str, Any]:
    """Greedy Pass@1 evaluation on MATH-500."""
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    results = {
        "overall": [],
        "by_level": {1: [], 2: [], 3: [], 4: [], 5: []},
        "abstention_by_level": {1: [], 2: [], 3: [], 4: [], 5: []},
    }

    model.eval()
    for item in dataset:
        level = int(str(item["level"]).split()[-1])
        prompt = build_prompt(item["problem"], allow_abstention=False)
        completion = generate_one(model, tokenizer, prompt, temperature=0.0)
        correct = verify_with_timeout(completion, item["answer"]) == 1.0
        abstained = is_abstention(completion)

        results["overall"].append(correct)
        results["by_level"][level].append(correct)
        results["abstention_by_level"][level].append(abstained)

    model.train()
    return {
        "eval/pass@1_overall": float(np.mean(results["overall"])),
        **{
            f"eval/pass@1_level_{level}": float(np.mean(values)) if values else 0.0
            for level, values in results["by_level"].items()
        },
        **{
            f"eval/abstention_rate_level_{level}": float(np.mean(values)) if values else 0.0
            for level, values in results["abstention_by_level"].items()
        },
    }
