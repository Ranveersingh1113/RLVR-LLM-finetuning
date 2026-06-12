"""Phase 3 ternary-reward calibration training entry point."""

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
from training.common import (
    AdaptiveCurriculumDataset,
    build_sampler,
    load_config,
    load_model_and_tokenizer,
    load_model_with_adapter,
    make_phase3_reward,
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
    parser.add_argument(
        "--phase2-checkpoint",
        default="./checkpoints_v3/phase2_best",
        help="Path to Phase 2 LoRA adapter to continue training from. "
        "Set to 'none' to start Phase 3 from the base model (not recommended).",
    )
    parser.add_argument(
        "--phase3-start-step",
        type=int,
        default=0,
        help="Global step at which Phase 3 begins (controls ternary reward warmup). "
        "Phase 3 is its own training run starting at global_step=0, so default is 0.",
    )
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const=True,
        default=False,
        help="Resume Phase 3 itself from a checkpoint in output_dir.",
    )
    parser.add_argument("--run-name", default="phase3_calibration_adaptive")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_wandb_project(config.get("monitoring", {}).get("project"))
    if args.phase2_checkpoint and args.phase2_checkpoint.lower() != "none":
        print(f"[phase3] Loading base model + Phase 2 adapter from {args.phase2_checkpoint}")
        model, tokenizer = load_model_with_adapter(config, args.phase2_checkpoint)
    else:
        print("[phase3] Starting from base model (no Phase 2 checkpoint)")
        model, tokenizer = load_model_and_tokenizer(config)
    sampler = build_sampler(cache_path=args.cache_path)
    train_dataset = AdaptiveCurriculumDataset(sampler, phase=3, tokenizer=tokenizer)
    reward_fn = make_phase3_reward(
        tokenizer=tokenizer,
        phase3_start_step=args.phase3_start_step,
        sampler=sampler,
        warmup_steps=config["phase3"]["reward_warmup_steps"],
    )

    ensure_torch_inductor_config_compat()
    from trl import GRPOConfig, GRPOTrainer

    batch_settings = resolve_grpo_batch_settings(config)
    precision = resolve_training_precision()

    training_args = GRPOConfig(
        output_dir=config["output_dir"],
        learning_rate=config["phase3"]["learning_rate"],
        per_device_train_batch_size=batch_settings["per_device_train_batch_size"],
        gradient_accumulation_steps=batch_settings["gradient_accumulation_steps"],
        max_prompt_length=config["model"]["max_seq_length"]
        - config["phase3"]["max_completion_length"],
        max_completion_length=config["phase3"]["max_completion_length"],
        num_generations=config["grpo"]["num_generations"],
        temperature=config["grpo"]["temperature"],
        max_grad_norm=config["grpo"]["max_grad_norm"],
        warmup_steps=config["grpo"]["warmup_steps"],
        logging_steps=config["grpo"]["logging_steps"],
        save_steps=config["grpo"]["save_steps"],
        max_steps=args.max_steps,
        report_to=config["monitoring"]["report_to"],
        run_name=args.run_name,
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
    trainer.save_model(f"{config['output_dir']}/phase3_final")


if __name__ == "__main__":
    main()
