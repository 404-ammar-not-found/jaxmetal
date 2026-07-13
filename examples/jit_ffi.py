"""Example 2 — run Metal *inside* jax.jit via the XLA FFI custom call.

    python examples/02_jit_ffi.py

`jaxmetal.ffi.matmul` lowers to an XLA FFI custom call, so it composes with
native JAX ops under `jax.jit` (and traces like any primitive). This is the
supported "call native code from JAX" path — distinct from a PJRT device.

Build requirement: libmetal_capi must be built with the FFI handler:
    cmake -S . -B build -G Ninja \\
      -DJAX_FFI_INCLUDE_DIR=$(python -c "import jax.ffi; print(jax.ffi.include_dir())")
    cmake --build build
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
import jax
import jax.numpy as jnp

import jaxmetal.ffi as jf

rng = np.random.default_rng(0)
a = jnp.asarray(rng.standard_normal((256, 384), dtype=np.float32))
b = jnp.asarray(rng.standard_normal((384, 128), dtype=np.float32))


@jax.jit
def model(a, b):
    c = jf.matmul(a, b)          # <- Metal (MPS) matmul, as an XLA custom call
    return jnp.tanh(c) + 1.0     # <- native JAX ops fuse around it


out = np.asarray(model(a, b))
ref = np.asarray(jnp.tanh(jnp.matmul(a, b)) + 1.0)
print("jitted model(a,b) vs jnp:  max|Δ| =", f"{np.max(np.abs(out - ref)):.1e}")

# The custom call is baked into the compiled program:
print("\njaxpr (note the ffi/custom call):")
print(jax.make_jaxpr(model)(a, b))
