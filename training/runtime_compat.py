"""Runtime compatibility helpers for the local training environment."""

from __future__ import annotations

import importlib.util
import importlib
import sys
from collections.abc import Mapping
from pathlib import Path


def ensure_torch_inductor_config_compat() -> None:
    """
    Provide a minimal ``torch._inductor.config`` shim for older Torch builds.

    Newer Unsloth releases assume this module exists, but the local Torch 2.4.1
    build exposes ``torch._inductor`` without a ``config`` submodule.
    """
    import torch

    inductor = getattr(torch, "_inductor", None)
    if inductor is None or hasattr(inductor, "config"):
        return

    try:
        config_module = importlib.import_module("torch._inductor.config")
    except ModuleNotFoundError:
        shim_path = Path(__file__).with_name("_torch_inductor_config_shim.py")
        spec = importlib.util.spec_from_file_location("torch._inductor.config", shim_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not build import spec for {shim_path}")
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        sys.modules["torch._inductor.config"] = config_module

    setattr(inductor, "config", config_module)


def ensure_accelerate_batch_compat() -> None:
    """Patch Accelerate batch-size detection for GRPO's identity-collated text batches."""
    import accelerate.data_loader as data_loader
    from accelerate.utils import operations

    original_find_batch_size = operations.find_batch_size
    if getattr(original_find_batch_size, "_rlvr_patched", False):
        return

    def patched_find_batch_size(data):
        if isinstance(data, list) and data and isinstance(data[0], Mapping):
            return len(data)
        return original_find_batch_size(data)

    patched_find_batch_size._rlvr_patched = True  # type: ignore[attr-defined]
    operations.find_batch_size = patched_find_batch_size
    data_loader.find_batch_size = patched_find_batch_size


def ensure_torch_argsort_bool_cuda_compat() -> None:
    """Patch torch.argsort so CUDA bool tensors are cast before sorting."""
    import torch

    original_argsort = torch.argsort
    if getattr(original_argsort, "_rlvr_patched", False):
        return

    def patched_argsort(input, *args, **kwargs):
        if isinstance(input, torch.Tensor) and input.is_cuda and input.dtype == torch.bool:
            input = input.to(dtype=torch.int32)
        return original_argsort(input, *args, **kwargs)

    patched_argsort._rlvr_patched = True  # type: ignore[attr-defined]
    torch.argsort = patched_argsort
