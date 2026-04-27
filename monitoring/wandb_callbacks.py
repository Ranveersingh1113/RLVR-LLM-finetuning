"""Monitoring hooks for training metrics."""

from __future__ import annotations

import os
from typing import Any

import torch
from transformers import TrainerCallback

from eval.eval_calibration import eval_calibration
from eval.eval_fast import eval_pass1


def merge_metric_dicts(*metric_sets: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for metric_set in metric_sets:
        merged.update(metric_set)
    return merged


def configure_wandb_project(project: str | None) -> None:
    """Respect the configured WandB project unless the shell already set one."""
    if project:
        os.environ.setdefault("WANDB_PROJECT", project)


class _WandbMetricLogger:
    def __init__(self) -> None:
        self._defined = False

    def log(self, metrics: dict[str, Any], global_step: int) -> None:
        try:
            import wandb

            if wandb.run is None:
                return
            if not self._defined:
                wandb.define_metric("train/global_step")
                wandb.define_metric("curriculum/*", step_metric="train/global_step")
                wandb.define_metric("eval/*", step_metric="train/global_step")
                self._defined = True
            wandb.log({"train/global_step": global_step, **metrics})
        except Exception:
            pass


class CurriculumMetricsCallback(TrainerCallback):
    """Log adaptive curriculum stats during training."""

    def __init__(self, sampler) -> None:
        self.sampler = sampler
        self._wandb = _WandbMetricLogger()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step <= 0:
            return control
        stats = self.sampler.get_stats()
        if logs is not None:
            logs.update(stats)
        self._wandb.log(stats, state.global_step)
        return control


class PeriodicEvalCallback(TrainerCallback):
    """Run scheduled evals from the project spec without interrupting training on errors."""

    def __init__(
        self,
        tokenizer,
        *,
        fast_eval_every_steps: int | None = None,
        calibration_eval_every_steps: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.fast_eval_every_steps = fast_eval_every_steps
        self.calibration_eval_every_steps = calibration_eval_every_steps
        self._last_fast_eval_step = -1
        self._last_calibration_eval_step = -1
        self._wandb = _WandbMetricLogger()

    def _should_run(self, interval: int | None, global_step: int, last_step: int) -> bool:
        return bool(interval and global_step > 0 and global_step % interval == 0 and global_step != last_step)

    def _log_metrics(self, metrics: dict[str, Any], global_step: int) -> None:
        self._wandb.log(metrics, global_step)

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        if self._should_run(self.fast_eval_every_steps, state.global_step, self._last_fast_eval_step):
            try:
                metrics = eval_pass1(model, self.tokenizer)
                self._log_metrics(metrics, state.global_step)
                self._last_fast_eval_step = state.global_step
            except Exception as exc:
                print(f"[monitoring] fast eval failed at step {state.global_step}: {exc}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if self._should_run(
            self.calibration_eval_every_steps,
            state.global_step,
            self._last_calibration_eval_step,
        ):
            try:
                metrics = eval_calibration(model, self.tokenizer)
                self._log_metrics(metrics, state.global_step)
                self._last_calibration_eval_step = state.global_step
            except Exception as exc:
                print(f"[monitoring] calibration eval failed at step {state.global_step}: {exc}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return control
