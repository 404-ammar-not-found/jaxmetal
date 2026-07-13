"""Native jax.jit integration via XLA FFI.

`jaxmetal.ffi.matmul(a, b)` is a **jittable** matmul: it lowers to an XLA FFI
custom call that dispatches to our Metal (MPS) matmul, so it composes inside
`jax.jit` with the rest of a JAX program. This is the supported "call native code
from JAX" mechanism (distinct from a PJRT *device*; see docs/PJRT_PLUGIN.md).

Requires libmetal_capi built with the FFI handler:
    cmake -S . -B build -G Ninja \\
        -DJAX_FFI_INCLUDE_DIR=$(python -c "import jax.ffi; print(jax.ffi.include_dir())")
"""

from __future__ import annotations

import ctypes

import jax
import jax.numpy as jnp

from . import _capi

_registered = False


def _ensure_registered() -> None:
    global _registered
    if _registered:
        return
    lib = ctypes.CDLL(_capi.library_path())
    if not hasattr(lib, "metal_ffi_matmul_handler"):
        raise RuntimeError(
            "libmetal_capi was built without the FFI handler. Reconfigure with:\n"
            "  cmake -S . -B build -G Ninja "
            "-DJAX_FFI_INCLUDE_DIR=$(python -c 'import jax.ffi; print(jax.ffi.include_dir())')"
        )
    lib.metal_ffi_matmul_handler.restype = ctypes.c_void_p
    capsule = jax.ffi.pycapsule(lib.metal_ffi_matmul_handler())
    jax.ffi.register_ffi_target("metal_matmul", capsule, platform="cpu")
    _registered = True


def matmul(a, b):
    """Jittable C = A @ B (rank-2 f32) via an XLA FFI call into Metal/MPS."""
    _ensure_registered()
    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    m, k = a.shape
    n = b.shape[1]
    out_type = jax.ShapeDtypeStruct((m, n), jnp.float32)
    return jax.ffi.ffi_call("metal_matmul", out_type, vmap_method="sequential")(a, b)
