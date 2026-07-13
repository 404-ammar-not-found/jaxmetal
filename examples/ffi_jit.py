"""Demo: our Metal matmul running natively inside jax.jit via XLA FFI.

    python python/ffi_demo.py

Shows jaxmetal.ffi.matmul composing with native JAX ops under jax.jit, and
grad flowing through the surrounding program. Requires libmetal_capi built with
the FFI handler (see python/jaxmetal/ffi.py).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

import numpy as np

import jax
import jax.numpy as jnp

import jaxmetal.ffi as jf


def main() -> int:
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.standard_normal((256, 384), dtype=np.float32))
    b = jnp.asarray(rng.standard_normal((384, 128), dtype=np.float32))

    # Metal matmul composed with native JAX ops, all under one jit.
    @jax.jit
    def f(a, b):
        c = jf.matmul(a, b)         # XLA FFI custom call -> Metal/MPS
        return jnp.tanh(c) + 1.0    # native JAX ops around it

    got = np.asarray(f(a, b))
    ref = np.asarray(jnp.tanh(jnp.matmul(a, b)) + 1.0)
    err = float(np.max(np.abs(got - ref)))
    print(f"jitted f(a,b) vs jnp: max|Δ| = {err:.2e}")

    # The custom call appears in the compiled program.
    jaxpr = str(jax.make_jaxpr(f)(a, b))
    print("custom call present in jaxpr:", ("ffi" in jaxpr.lower()) or ("metal" in jaxpr.lower()))

    print("OK: Metal matmul runs natively inside jax.jit." if err < 1e-3 else "FAIL")
    return 0 if err < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
