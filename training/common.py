"""Shared training helpers for RLVR math runs."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from torch.utils.data import IterableDataset

from data.difficulty_sampler import AdaptiveDifficultySampler
from data.prepare_dataset import prepare_dataset
from rewards.binary_reward import binary_reward
from rewards.ternary_reward import make_ternary_reward_fn
from training.runtime_compat import (
    ensure_accelerate_batch_compat,
    ensure_torch_argsort_bool_cuda_compat,
    ensure_torch_inductor_config_compat,
)
from utils.prompts import sample_prompt
from verifier.math_verifier import verify_with_timeout


@dataclass
class TrainingArtifacts:
    model: Any
    tokenizer: Any
    config: dict[str, Any]


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_model_and_tokenizer(config: dict[str, Any]):
    ensure_torch_inductor_config_compat()
    ensure_accelerate_batch_compat()
    ensure_torch_argsort_bool_cuda_compat()
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"]["name"],
        max_seq_length=config["model"]["max_seq_length"],
        load_in_4bit=config["model"]["load_in_4bit"],
        use_gradient_checkpointing="unsloth"
        if config["model"].get("gradient_checkpointing", True)
        else False,
        max_lora_rank=config["lora"]["r"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias=config["lora"]["bias"],
        use_gradient_checkpointing="unsloth"
        if config["model"].get("gradient_checkpointing", True)
        else False,
        max_seq_length=config["model"]["max_seq_length"],
    )
    return model, tokenizer


def resolve_training_precision() -> dict[str, bool]:
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return {
        "bf16": bf16_supported,
        "fp16": not bf16_supported,
    }


def resolve_grpo_batch_settings(config: dict[str, Any]) -> dict[str, int]:
    per_device_batch_size = int(config["grpo"]["per_device_train_batch_size"])
    grad_accum = int(config["grpo"]["gradient_accumulation_steps"])
    num_generations = int(config["grpo"]["num_generations"])

    if per_device_batch_size % num_generations == 0:
        return {
            "per_device_train_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": grad_accum,
        }

    effective_batch = per_device_batch_size * grad_accum
    adjusted_per_device = num_generations
    adjusted_grad_accum = max(1, effective_batch // adjusted_per_device)
    return {
        "per_device_train_batch_size": adjusted_per_device,
        "gradient_accumulation_steps": adjusted_grad_accum,
    }


class AdaptiveCurriculumDataset(IterableDataset):
    """Infinite iterable dataset backed by the adaptive difficulty sampler."""

    def __init__(
        self,
        sampler: AdaptiveDifficultySampler,
        *,
        phase: int,
        prompt_builder: Callable[[str, int], str] | None = None,
    ) -> None:
        super().__init__()
        self.sampler = sampler
        self.phase = phase
        self.prompt_builder = prompt_builder or sample_prompt

    def __iter__(self):
        while True:
            item = self.sampler.sample_batch(batch_size=1)[0]
            level = int(item.get("_sampled_level", item["level"]))
            yield {
                "prompt": self.prompt_builder(item["problem"], self.phase),
                "answer": item["answer"],
                "problem": item["problem"],
                "level": int(item["level"]),
                "_sampled_level": level,
                "subject": item["subject"],
                "source": item["source"],
            }


def build_sampler(cache_path: str = "./data/train_filtered.hf") -> AdaptiveDifficultySampler:
    dataset_by_level = prepare_dataset(cache_path=cache_path)
    return AdaptiveDifficultySampler(dataset_by_level)


def _extract_step(kwargs: dict[str, Any]) -> int:
    trainer_state = kwargs.get("trainer_state")
    if trainer_state is not None:
        return int(getattr(trainer_state, "global_step", 0))
    return int(kwargs.get("current_step", 0))


def _extract_levels(kwargs: dict[str, Any]) -> list[int]:
    levels = kwargs.get("_sampled_level") or kwargs.get("level") or []
    if not levels:
        warnings.warn(
            "No level data found in reward kwargs; sampler not updating. "
            "Check TRL batch key names and smoke-test curriculum metrics.",
            stacklevel=2,
        )
        return []
    return [int(level) for level in levels]


def _extract_answers(kwargs: dict[str, Any], answers: list[str] | None) -> list[str]:
    if answers is not None:
        return answers
    extracted = kwargs.get("answers") or kwargs.get("answer") or []
    return list(extracted)


def make_phase2_reward(tokenizer, sampler: AdaptiveDifficultySampler | None = None):
    def reward_fn(completions: list[str], answers: list[str] | None = None, **kwargs) -> list[float]:
        normalized_answers = _extract_answers(kwargs, answers)
        current_step = _extract_step(kwargs)
        rewards = binary_reward(
            completions=completions,
            answers=normalized_answers,
            tokenizer=tokenizer,
            current_step=current_step,
        )
        if sampler is not None:
            levels = _extract_levels(kwargs)
            for completion, answer, level in zip(completions, normalized_answers, levels):
                correct = verify_with_timeout(completion, answer) == 1.0
                sampler.update(level, correct)
        return rewards

    return reward_fn


def make_phase3_reward(
    tokenizer,
    phase3_start_step: int,
    sampler: AdaptiveDifficultySampler | None = None,
    warmup_steps: int = 50,
):
    reward_core = make_ternary_reward_fn(
        phase3_start_step=phase3_start_step,
        warmup_steps=warmup_steps,
    )

    def reward_fn(completions: list[str], answers: list[str] | None = None, **kwargs) -> list[float]:
        normalized_answers = _extract_answers(kwargs, answers)
        current_step = _extract_step(kwargs)
        rewards = reward_core(
            completions=completions,
            answers=normalized_answers,
            tokenizer=tokenizer,
            current_step=current_step,
        )
        if sampler is not None:
            levels = _extract_levels(kwargs)
            for completion, answer, level in zip(completions, normalized_answers, levels):
                correct = verify_with_timeout(completion, answer) == 1.0
                sampler.update(level, correct)
        return rewards

    return reward_fn
