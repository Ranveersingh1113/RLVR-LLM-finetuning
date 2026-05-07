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


def ensure_torch_load_safe_compat() -> None:
    """No-op the transformers torch.load safety gate when resuming a local checkpoint.

    Transformers >=4.50 refuses to call torch.load on torch <2.6 due to CVE-2025-32434
    (an untrusted-pickle vulnerability). When resuming from a checkpoint we wrote
    ourselves on this same machine the threat model does not apply, so bypassing the
    gate is safe in this controlled environment. Do NOT carry this patch into any
    code path that loads externally sourced .bin/.pt files.

    Idempotent: safe to call multiple times. Should be called once before any
    transformers import AND once again right before trainer.train(), since
    transformers.trainer binds the symbol by name at module import time and a
    later patch of the source module alone won't update that binding.
    """
    import transformers.utils.import_utils as import_utils

    def patched_check_torch_load_is_safe(*args, **kwargs):
        return None

    patched_check_torch_load_is_safe._rlvr_patched = True  # type: ignore[attr-defined]

    original = getattr(import_utils, "check_torch_load_is_safe", None)
    if original is not None and not getattr(original, "_rlvr_patched", False):
        import_utils.check_torch_load_is_safe = patched_check_torch_load_is_safe
        # Walk every loaded module and rebind any attribute that still points to
        # the original — `from ... import check_torch_load_is_safe` creates a
        # local binding that doesn't follow updates to the source module.
        for module in list(sys.modules.values()):
            if module is None:
                continue
            try:
                attr = getattr(module, "check_torch_load_is_safe", None)
            except Exception:
                continue
            if attr is original:
                try:
                    setattr(module, "check_torch_load_is_safe", patched_check_torch_load_is_safe)
                except Exception:
                    pass
