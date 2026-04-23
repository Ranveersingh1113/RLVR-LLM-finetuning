"""Phase 2 GRPO training entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.common import (
    AdaptiveCurriculumDataset,
    build_sampler,
    load_config,
    load_model_and_tokenizer,
    make_phase2_reward,
    resolve_grpo_batch_settings,
    resolve_training_precision,
)
from training.runtime_compat import ensure_torch_inductor_config_compat
from training.runtime_compat import ensure_unsloth_compiled_grpo_cache_patched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo_a4000.yaml")
    parser.add_argument("--cache-path", default="./data/train_filtered.hf")
    parser.add_argument("--max-steps", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    sampler = build_sampler(cache_path=args.cache_path)
    train_dataset = AdaptiveCurriculumDataset(sampler, phase=2)
    model, tokenizer = load_model_and_tokenizer(config)
    ensure_unsloth_compiled_grpo_cache_patched()
    reward_fn = make_phase2_reward(tokenizer=tokenizer, sampler=sampler)

    ensure_torch_inductor_config_compat()
    from trl import GRPOConfig, GRPOTrainer
    from transformers import TrainerCallback

    batch_settings = resolve_grpo_batch_settings(config)
    precision = resolve_training_precision()

    class CurriculumMetricsCallback(TrainerCallback):
        def __init__(self, sampler) -> None:
            self.sampler = sampler

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                logs.update(self.sampler.get_stats())
            return control

    training_args = GRPOConfig(
        output_dir=config["output_dir"],
        learning_rate=config["grpo"]["learning_rate"],
        per_device_train_batch_size=batch_settings["per_device_train_batch_size"],
        gradient_accumulation_steps=batch_settings["gradient_accumulation_steps"],
        max_prompt_length=config["model"]["max_seq_length"]
        - config["grpo"]["max_completion_length"],
        max_completion_length=config["grpo"]["max_completion_length"],
        num_generations=config["grpo"]["num_generations"],
        temperature=config["grpo"]["temperature"],
        max_grad_norm=config["grpo"]["max_grad_norm"],
        warmup_steps=config["grpo"]["warmup_steps"],
        logging_steps=config["grpo"]["logging_steps"],
        save_steps=config["grpo"]["save_steps"],
        max_steps=args.max_steps,
        report_to=config["monitoring"]["report_to"],
        run_name="phase2_grpo_adaptive",
        bf16=precision["bf16"],
        fp16=precision["fp16"],
        gradient_checkpointing=config["model"]["gradient_checkpointing"],
        log_completions=False,
        beta=config["grpo"]["kl_coeff"],
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[CurriculumMetricsCallback(sampler)],
    )
    trainer.train()
    trainer.save_model(f"{config['output_dir']}/phase2_best")


if __name__ == "__main__":
    main()
