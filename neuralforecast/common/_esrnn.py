"""Batched ES kernels; optional fused CPU recurrence with exact first-order VJP."""

import os
from pathlib import Path
import sys
import tempfile
import threading
import warnings

import torch
from torch.autograd.function import once_differentiable

_NATIVE = None
_NATIVE_ERROR = None
_BUILD_LOCK = threading.Lock()


def _native_module(required=False):
    global _NATIVE, _NATIVE_ERROR
    if _NATIVE is None and _NATIVE_ERROR is None:
        with _BUILD_LOCK:
            if _NATIVE is None and _NATIVE_ERROR is None:
                try:
                    from torch.utils.cpp_extension import load

                    cache = Path(
                        os.environ.get(
                            "TORCH_EXTENSIONS_DIR",
                            Path(tempfile.gettempdir()) / "neuralforecast_extensions",
                        )
                    )
                    build = cache / (
                        "esrnn_"
                        + torch.__version__.split("+")[0]
                        + "_"
                        + sys.implementation.cache_tag
                    )
                    build.mkdir(parents=True, exist_ok=True)
                    flags = (
                        ["/O2"]
                        if sys.platform == "win32"
                        else ["-O3", "-ffp-contract=off"]
                    )
                    link_flags = (
                        ["-framework", "Accelerate"] if sys.platform == "darwin" else []
                    )
                    if (
                        sys.platform == "darwin"
                        and (Path(torch.__file__).parent / "lib/libomp.dylib").exists()
                    ):
                        flags += ["-Xpreprocessor", "-fopenmp"]
                        link_flags += ["-lomp"]
                    _NATIVE = load(
                        name="neuralforecast_esrnn_scan",
                        sources=[str(Path(__file__).with_name("_esrnn_native.cpp"))],
                        extra_cflags=flags,
                        extra_ldflags=link_flags,
                        build_directory=str(build),
                        verbose=False,
                    )
                except (RuntimeError, OSError, ImportError) as error:
                    _NATIVE_ERROR = error
                    if not required:
                        warnings.warn(
                            "ESRNN CPU compilation unavailable; using batched PyTorch. "
                            'Set es_backend="torch" to disable compilation.',
                            RuntimeWarning,
                        )
    if required and _NATIVE is None:
        raise RuntimeError(
            "ESRNN native backend could not compile. A C++ compiler and ninja "
            'are required; use es_backend="torch" for the portable backend.'
        ) from _NATIVE_ERROR
    return _NATIVE


def torch_scan(y, coefficients, initial1, initial2):
    """Portable differentiable batched recurrence, including higher-order grads."""
    seasons = [list(v.unbind(-1)) for v in (initial1, initial2) if v.shape[1]]
    product = y.new_ones(y.shape[0])
    for factors in seasons:
        product = product * factors[0]
        factors.append(factors[0])
    levels = [y[:, 0] / product]
    for t in range(1, y.shape[1]):
        product = y.new_ones(y.shape[0])
        for factors in seasons:
            product = product * factors[t]
        level = (
            coefficients[:, 0] * y[:, t] / product
            + (1 - coefficients[:, 0]) * levels[-1]
        )
        levels.append(level)
        for j, factors in enumerate(seasons):
            other = y.new_ones(y.shape[0])
            for k, values in enumerate(seasons):
                if k != j:
                    other = other * values[t]
            factors.append(
                coefficients[:, j + 1] * y[:, t] / (level * other)
                + (1 - coefficients[:, j + 1]) * factors[t]
            )
    output = [torch.stack(levels, dim=1), *[torch.stack(v, dim=1) for v in seasons]]
    while len(output) < 3:
        output.append(y.new_empty((len(y), 0)))
    return tuple(output)


class _NativeScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, coefficients, initial1, initial2):
        module = _native_module(required=True)
        output = tuple(module.forward(y, coefficients, initial1, initial2))
        ctx.save_for_backward(y, coefficients, *output)
        ctx.periods = (initial1.shape[1], initial2.shape[1])
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gl, gs, gr):
        y, coefficients, levels, s1, s2 = ctx.saved_tensors
        return tuple(
            _native_module(required=True).backward(
                y,
                coefficients,
                levels,
                s1,
                s2,
                gl.contiguous(),
                gs.contiguous(),
                gr.contiguous(),
                *ctx.periods,
            )
        )


def native_scan(y, coefficients, initial1, initial2):
    """Fused CPU scan. First-order autograd; use torch_scan for higher orders."""
    return _NativeScan.apply(
        *(v.contiguous() for v in (y, coefficients, initial1, initial2))
    )


def scan(y, coefficients, initial1, initial2, backend):
    supported = y.device.type == "cpu" and y.dtype in (torch.float32, torch.float64)
    if backend == "native" and not supported:
        raise ValueError(
            "ESRNN native backend requires CPU float32 or float64 tensors."
        )
    if (
        backend != "torch"
        and supported
        and _native_module(required=backend == "native") is not None
    ):
        return native_scan(y, coefficients, initial1, initial2)
    return torch_scan(y, coefficients, initial1, initial2)


class _NativeLSTM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wi, wh, bi, bh):
        out, states, gates = _native_module(required=True).lstm_forward(
            x, wi, wh, bi, bh
        )
        ctx.save_for_backward(x, wi, wh, out, states, gates)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return tuple(
            _native_module(required=True).lstm_backward(
                *ctx.saved_tensors, gradient.contiguous()
            )
        )


def native_lstm(x, wi, wh, bi, bh):
    """Fused zero-initialized one-layer CPU LSTM, with exact first-order VJP."""
    return _NativeLSTM.apply(*(v.contiguous() for v in (x, wi, wh, bi, bh)))


def use_native_lstm(x, backend):
    # Keep mixed precision on PyTorch's autocast-aware native implementation.
    supported = x.device.type == "cpu" and x.dtype in (torch.float32, torch.float64)
    if supported and torch.is_autocast_enabled("cpu"):
        return False
    if (
        backend == "auto"
        and torch.backends.mkldnn.enabled
        and torch.backends.mkldnn.is_available()
    ):
        return False
    return (
        backend != "torch"
        and supported
        and _native_module(required=backend == "native") is not None
    )
