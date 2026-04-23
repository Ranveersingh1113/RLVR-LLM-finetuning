"""File-backed shim for ``torch._inductor.config`` on older Torch builds."""

from __future__ import annotations


class _ConfigNamespace:
    """Namespace that auto-creates nested attributes on demand."""

    def __getattr__(self, name: str):
        value = _ConfigNamespace()
        setattr(self, name, value)
        return value


epilogue_fusion = True
max_autotune = False
shape_padding = True
debug = False
dce = True
memory_planning = False
coordinate_descent_tuning = False
compile_threads = 8
group_fusion = False
disable_progress = True
verbose_progress = False
freezing = False
combo_kernels = False
benchmark_combo_kernel = True
combo_kernel_foreach_dynamic_shapes = True

trace = _ConfigNamespace()
trace.enabled = False
trace.graph_diagram = False

triton = _ConfigNamespace()
triton.cudagraphs = False
triton.multi_kernel = 0
triton.use_block_ptr = False
triton.enable_persistent_tma_matmul = True
triton.autotune_at_compile_time = False
triton.cooperative_reductions = False

cuda = _ConfigNamespace()
cuda.compile_opt_level = "-O2"
cuda.enable_cuda_lto = True
cuda.use_fast_math = False
