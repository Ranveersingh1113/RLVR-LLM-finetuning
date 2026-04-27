"""Fast Pass@1 evaluation for MATH-500."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
        prompt = build_prompt(item["problem"], allow_abstention=False, tokenizer=tokenizer)
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


if __name__ == "__main__":
    import argparse
    import json

    from training.runtime_compat import (
        ensure_accelerate_batch_compat,
        ensure_torch_argsort_bool_cuda_compat,
        ensure_torch_inductor_config_compat,
    )

    ensure_torch_inductor_config_compat()
    ensure_accelerate_batch_compat()
    ensure_torch_argsort_bool_cuda_compat()

    from transformers import AutoTokenizer
    from unsloth import FastLanguageModel

    parser = argparse.ArgumentParser(description="Standalone Pass@1 eval on MATH-500")
    parser.add_argument("--checkpoint", default="./checkpoints/phase2_best")
    parser.add_argument("--base-model", default="unsloth/qwen2.5-math-7b-instruct-bnb-4bit")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Evaluate the base model only (zero-shot baseline, no LoRA adapter loaded)",
    )
    cli_args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cli_args.base_model,
        max_seq_length=cli_args.max_seq_length,
        load_in_4bit=True,
    )
    if not cli_args.no_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cli_args.checkpoint)
    FastLanguageModel.for_inference(model)

    metrics = eval_pass1(model, tokenizer)
    print(json.dumps(metrics, indent=2))
