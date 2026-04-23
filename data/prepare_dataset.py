"""Dataset loading, normalization, and caching for RLVR math training."""

from __future__ import annotations

from pathlib import Path
import re
import warnings
from typing import Any, Callable, Iterable

from verifier.math_verifier import extract_boxed, verify_with_timeout


def _parse_level(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        digits = "".join(char for char in value if char.isdigit())
        if digits:
            return int(digits)
    raise ValueError(f"Could not parse difficulty level from {value!r}")


def _extract_math_answer(item: dict[str, Any]) -> str:
    if "answer" in item and item["answer"]:
        return str(item["answer"]).strip()

    solution = str(item.get("solution", "")).strip()
    boxed = extract_boxed(solution)
    if boxed is not None:
        return boxed

    shorthand_boxed = re.search(r"\\boxed\s+([^\s$.,;]+)", solution)
    if shorthand_boxed:
        return shorthand_boxed.group(1).strip()

    raise ValueError("Could not extract MATH answer from item")


def _extract_gsm8k_answer(answer: str) -> str:
    text = answer.strip()
    if "####" in text:
        return text.rsplit("####", 1)[-1].strip()
    return text


def _normalize_math_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem": str(item["problem"]).strip(),
        "answer": _extract_math_answer(item),
        "level": _parse_level(item["level"]),
        "subject": str(item.get("type") or item.get("subject") or "unknown").strip(),
        "source": "math",
    }


def _normalize_gsm8k_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem": str(item["question"]).strip(),
        "answer": _extract_gsm8k_answer(str(item["answer"])),
        "level": 1,
        "subject": "arithmetic",
        "source": "gsm8k",
    }


def load_and_normalize_datasets() -> list[dict[str, Any]]:
    """Load training data from Hugging Face and convert to the common schema."""
    from datasets import load_dataset

    math_train = None
    last_error: Exception | None = None
    for dataset_name in ("lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"):
        try:
            math_train = load_dataset(dataset_name, split="train")
            break
        except Exception as exc:
            last_error = exc
    if math_train is None:
        raise RuntimeError("Failed to load any MATH training dataset.") from last_error

    gsm8k_train = load_dataset("gsm8k", "main", split="train")

    normalized_math = []
    skipped_math = 0
    for item in math_train:
        try:
            normalized_math.append(_normalize_math_item(item))
        except ValueError as exc:
            skipped_math += 1
            warnings.warn(f"Skipping MATH item during normalization: {exc}", stacklevel=2)

    normalized_gsm8k = [_normalize_gsm8k_item(item) for item in gsm8k_train]
    if skipped_math:
        warnings.warn(
            f"Skipped {skipped_math} MATH training items during normalization.",
            stacklevel=2,
        )
    return normalized_math + normalized_gsm8k


def group_by_level(items: Iterable[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(int(item["level"]), []).append(item)
    return grouped


def _resolve_generate_n(model) -> Callable[..., list[str]]:
    if callable(model):
        return model
    generate_n = getattr(model, "generate_n", None)
    if callable(generate_n):
        return generate_n
    raise TypeError(
        "model must be a callable or expose generate_n(problem, n, temperature, max_new_tokens, tokenizer=...)"
    )


def prefilter_hard_problems(dataset_levels_4_5, model, tokenizer, n_samples: int = 4) -> list:
    """Remove Level 4/5 problems where the base model gets 0-for-n correct."""
    from tqdm import tqdm

    generate_n = _resolve_generate_n(model)

    keep = []
    for item in tqdm(dataset_levels_4_5, desc="Pre-filtering L4/L5"):
        completions = generate_n(
            item["problem"],
            n=n_samples,
            temperature=0.7,
            max_new_tokens=512,
            tokenizer=tokenizer,
        )
        if any(verify_with_timeout(completion, item["answer"]) == 1.0 for completion in completions):
            keep.append(item)
    return keep


def prepare_dataset(
    cache_path: str = "./data/train_filtered.hf",
    *,
    prefilter_model=None,
    tokenizer=None,
    n_samples: int = 4,
) -> dict[int, list[dict[str, Any]]]:
    """Prepare, optionally prefilter, and cache the train split grouped by level."""
    from datasets import Dataset, DatasetDict, load_from_disk

    cache_dir = Path(cache_path)
    if cache_dir.exists():
        cached = load_from_disk(str(cache_dir))
        return {int(level): list(dataset) for level, dataset in cached.items()}

    grouped = group_by_level(load_and_normalize_datasets())

    if prefilter_model is not None:
        for level in (4, 5):
            if level in grouped:
                grouped[level] = prefilter_hard_problems(
                    grouped[level],
                    model=prefilter_model,
                    tokenizer=tokenizer,
                    n_samples=n_samples,
                )

    dataset_dict = DatasetDict(
        {str(level): Dataset.from_list(items) for level, items in grouped.items()}
    )
    dataset_dict.save_to_disk(str(cache_dir))
    return grouped
