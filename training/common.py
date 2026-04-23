"""Shared training helpers for RLVR math runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml
from torch.utils.data import IterableDataset

from data.difficulty_sampler import AdaptiveDifficultySampler
from data.prepare_dataset import prepare_dataset
from rewards.binary_reward import binary_reward
from rewards.ternary_reward import make_ternary_reward_fn, sample_prompt
from training.runtime_compat import ensure_torch_inductor_config_compat


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


def make_phase2_reward(tokenizer, sampler: AdaptiveDifficultySampler | None = None):
    def reward_fn(completions: list[str], answers: list[str], **kwargs) -> list[float]:
        current_step = _extract_step(kwargs)
        rewards = binary_reward(
            completions=completions,
            answers=answers,
            tokenizer=tokenizer,
            current_step=current_step,
        )
        if sampler is not None:
            levels = kwargs.get("_sampled_level") or kwargs.get("level") or []
            for level, reward in zip(levels, rewards):
                sampler.update(int(level), reward >= 1.0)
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

    def reward_fn(completions: list[str], answers: list[str], **kwargs) -> list[float]:
        current_step = _extract_step(kwargs)
        rewards = reward_core(
            completions=completions,
            answers=answers,
            tokenizer=tokenizer,
            current_step=current_step,
        )
        if sampler is not None:
            levels = kwargs.get("_sampled_level") or kwargs.get("level") or []
            for level, reward in zip(levels, rewards):
                sampler.update(int(level), reward >= 1.0)
        return rewards

    return reward_fn
