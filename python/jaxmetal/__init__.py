"""jaxmetal — run JAX workloads on the Apple GPU through hand-written Metal.

A learning-focused, compiler-level backend that executes matmul and a full MLP
training step (forward + backprop + SGD) on Apple Silicon via hand-written Metal
Shading Language kernels plus MPS, with all state kept GPU-resident.

Public surface
--------------
    import jaxmetal
    jaxmetal.device_name()                       # "Apple M4 Pro"
    jaxmetal.matmul(a, b, device="mps")          # jnp.matmul-style, explicit backend
    m = jaxmetal.Mlp(784, 1024, 10, max_batch=512)   # resident GPU MLP trainer
    loss = m.train_step(lr=0.5)

Submodules
----------
    jaxmetal.ffi          native jax.jit integration (XLA FFI custom call)
    jaxmetal.data         MNIST download / cache / parse
    jaxmetal.reference    NumPy golden reference (matches jax.grad)
    jaxmetal.plugin       PJRT-device scaffold (roadmap; see docs/PJRT_PLUGIN.md)

`jaxmetal.matmul(device=...)` is an eager convenience bridge (JAX ops have no
`device=` kwarg); `jaxmetal.ffi.matmul` is the jit-native path; a true `metal`
*device* is the PJRT roadmap.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from . import _capi
from ._capi import DeviceBuffer, Mlp, device_name, library_path

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "matmul", "device_name", "devices", "library_path",
    "DeviceBuffer", "Mlp",
]

DEVICES = ("mps", "metal", "cpu", "auto")

_BACKENDS = {
    "mps": _capi.matmul_mps,
    "metal": _capi.matmul,
    "cpu": _capi.matmul_cpu,
    "auto": _capi.matmul_auto,
}


def devices() -> tuple:
    """Backend names accepted by `matmul(device=...)`."""
    return DEVICES


def matmul(a, b, device: str = "auto"):
    """C = A @ B on the requested backend, with full `jnp.matmul` semantics.

    Follows NumPy/JAX matmul rules: 2-D @ 2-D is a plain matmul; higher ranks are
    stacks of matrices in the last two axes, broadcast over the leading batch
    dims; a 1-D operand is promoted (and its dummy axis removed from the result).

    device: one of "mps" | "metal" | "cpu" | "auto". Returns a jax.numpy f32 array.
    """
    import math

    if device not in _BACKENDS:
        raise ValueError(f"unknown device {device!r}; expected one of {DEVICES}")
    backend = _BACKENDS[device]

    an = np.ascontiguousarray(np.asarray(a), dtype=np.float32)
    bn = np.ascontiguousarray(np.asarray(b), dtype=np.float32)
    if an.ndim == 0 or bn.ndim == 0:
        raise ValueError("matmul operands must be at least 1-D")

    a_was_1d = an.ndim == 1
    b_was_1d = bn.ndim == 1
    if a_was_1d:
        an = an[np.newaxis, :]
    if b_was_1d:
        bn = bn[:, np.newaxis]

    M, K = an.shape[-2:]
    K2, N = bn.shape[-2:]
    if K != K2:
        raise ValueError(f"inner dims mismatch: {an.shape} @ {bn.shape}")

    batch = np.broadcast_shapes(an.shape[:-2], bn.shape[:-2])
    a_b = np.ascontiguousarray(np.broadcast_to(an, batch + (M, K)))
    b_b = np.ascontiguousarray(np.broadcast_to(bn, batch + (K2, N)))
    nb = math.prod(batch) if batch else 1
    a_f = a_b.reshape(nb, M, K)
    b_f = b_b.reshape(nb, K2, N)

    out = np.empty((nb, M, N), dtype=np.float32)
    for i in range(nb):
        out[i] = backend(a_f[i], b_f[i])
    out = out.reshape(batch + (M, N))

    drop = []
    if a_was_1d:
        drop.append(-2)
    if b_was_1d:
        drop.append(-1)
    if drop:
        out = np.squeeze(out, axis=tuple(drop))
    return jnp.asarray(out)
