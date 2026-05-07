"""Run the L4/L5 zero-pass prefilter and cache the filtered dataset.

This implements the spec step (RLVR_MATH_PROJECT.md §"Pre-filtering — scope and
compute cost") that was skipped in the v1/v2 runs. With n=4 samples per problem
on the base model, ~20% of L4 and ~40% of L5 problems are zero-pass and produce
exclusively zero-reward training rollouts; removing them recovers gradient signal
on hard tiers.

Runtime: ~2 hours on an A4000. Output cached at the path given via --cache-path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from data.prepare_dataset import group_by_level, load_and_normalize_datasets, prefilter_hard_problems
from training.runtime_compat import (
    ensure_accelerate_batch_compat,
    ensure_torch_argsort_bool_cuda_compat,
    ensure_torch_inductor_config_compat,
    ensure_torch_load_safe_compat,
)
from utils.prompts import build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", default="./data/train_filtered_v3.hf")
    parser.add_argument("--base-model", default="unsloth/qwen2.5-math-7b-instruct-bnb-4bit")
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cache_path = Path(args.cache_path)
    if cache_path.exists():
        print(f"Cache already exists at {cache_path}; nothing to do.")
        return

    ensure_torch_inductor_config_compat()
    ensure_accelerate_batch_compat()
    ensure_torch_argsort_bool_cuda_compat()
    ensure_torch_load_safe_compat()
    from unsloth import FastLanguageModel

    print(f"Loading base model {args.base_model} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    def generate_n(problem, *, n, temperature, max_new_tokens, tokenizer):
        prompt = build_prompt(problem, allow_abstention=False, tokenizer=tokenizer)
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        encoded = {k: (v.repeat(n, 1) if v.ndim == 2 else v) for k, v in encoded.items()}
        with torch.no_grad():
            out = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = encoded["input_ids"].shape[1]
        return [tokenizer.decode(seq[prompt_len:], skip_special_tokens=True) for seq in out]

    print("Loading and normalizing datasets ...")
    items = load_and_normalize_datasets()
    grouped = group_by_level(items)

    for level in (4, 5):
        if level not in grouped:
            continue
        before = len(grouped[level])
        print(f"\nPrefiltering Level {level}: {before} problems")
        grouped[level] = prefilter_hard_problems(
            grouped[level],
            model=generate_n,
            tokenizer=tokenizer,
            n_samples=args.n_samples,
        )
        after = len(grouped[level])
        print(f"Level {level}: kept {after}/{before} ({100*after/before:.1f}%)")

    from datasets import Dataset, DatasetDict

    print(f"\nSaving filtered dataset to {cache_path} ...")
    dataset_dict = DatasetDict({str(level): Dataset.from_list(items) for level, items in grouped.items()})
    dataset_dict.save_to_disk(str(cache_path))
    print("Done.")


if __name__ == "__main__":
    main()
