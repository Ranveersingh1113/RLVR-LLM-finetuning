"""Phase 2 GRPO training entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring.wandb_callbacks import (
    CurriculumMetricsCallback,
    PeriodicEvalCallback,
    configure_wandb_project,
)
from data.prepare_dataset import prepare_dataset
from data.static_sampler import StaticUniformSampler
from training.common import (
    AdaptiveCurriculumDataset,
    build_sampler,
    load_config,
    load_model_and_tokenizer,
    make_phase2_reward,
    resolve_grpo_batch_settings,
    resolve_training_precision,
)
from training.runtime_compat import (
    ensure_torch_inductor_config_compat,
    ensure_torch_load_safe_compat,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo_a4000.yaml")
    parser.add_argument("--cache-path", default="./data/train_filtered.hf")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const=True,
        default=False,
        help="Resume training; pass a path or omit value to auto-detect latest checkpoint in output_dir",
    )
    parser.add_argument(
        "--static-curriculum",
        action="store_true",
        help="Run with uniform static sampler instead of adaptive (for Novelty B ablation).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Override the W&B run name (useful to distinguish ablation runs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_wandb_project(config.get("monitoring", {}).get("project"))
    model, tokenizer = load_model_and_tokenizer(config)
    if args.static_curriculum:
        dataset_by_level = prepare_dataset(cache_path=args.cache_path)
        sampler = StaticUniformSampler(dataset_by_level)
    else:
        sampler = build_sampler(cache_path=args.cache_path)
    train_dataset = AdaptiveCurriculumDataset(sampler, phase=2, tokenizer=tokenizer)
    reward_fn = make_phase2_reward(tokenizer=tokenizer, sampler=sampler)

    ensure_torch_inductor_config_compat()
    from trl import GRPOConfig, GRPOTrainer

    batch_settings = resolve_grpo_batch_settings(config)
    precision = resolve_training_precision()

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
        run_name=args.run_name or ("phase2_grpo_static" if args.static_curriculum else "phase2_grpo_adaptive"),
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
        callbacks=[
            CurriculumMetricsCallback(sampler),
            PeriodicEvalCallback(
                tokenizer,
                fast_eval_every_steps=config.get("monitoring", {}).get("fast_eval_every_steps"),
                calibration_eval_every_steps=config.get("monitoring", {}).get(
                    "calibration_eval_every_steps"
                ),
            ),
        ],
    )
    ensure_torch_load_safe_compat()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(f"{config['output_dir']}/phase2_best")


if __name__ == "__main__":
    main()
