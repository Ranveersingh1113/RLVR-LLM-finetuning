"""Runtime compatibility helpers for the local training environment."""

from __future__ import annotations

import importlib.util
import importlib
import sys
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
