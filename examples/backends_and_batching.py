"""Example 1 — pick a backend, and batched/N-D matmul.

    python examples/01_backends_and_batching.py

`jaxmetal.matmul(a, b, device=...)` takes/returns JAX arrays and runs on the
backend you name. It follows full `jnp.matmul` shape semantics (1-D, 2-D,
batched, broadcast), so it's a drop-in for `jnp.matmul` with an explicit device.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
import jax.numpy as jnp

import jaxmetal

rng = np.random.default_rng(0)
print("Metal GPU:", jaxmetal.device_name())
print("backends :", jaxmetal.devices(), "\n")

# --- 1. Same matmul on each backend --------------------------------------------
a = jnp.asarray(rng.standard_normal((256, 384), dtype=np.float32))
b = jnp.asarray(rng.standard_normal((384, 128), dtype=np.float32))
ref = jnp.matmul(a, b)

for device in ("mps", "metal", "cpu", "auto"):
    c = jaxmetal.matmul(a, b, device=device)   # -> jax array
    err = float(np.max(np.abs(np.asarray(c) - np.asarray(ref))))
    print(f"  device={device:<5} shape={c.shape}  max|Δ| vs jnp.matmul = {err:.1e}")

# --- 2. Batched / broadcast (full jnp.matmul semantics) ------------------------
print("\nbatched & broadcasting (device='mps'):")
cases = [
    ("(8,5) @ (5,9)      ", (8, 5), (5, 9)),          # plain 2-D
    ("(4,8,5) @ (4,5,9)  ", (4, 8, 5), (4, 5, 9)),     # batched
    ("(4,8,5) @ (5,9)    ", (4, 8, 5), (5, 9)),        # broadcast batch
    ("(7,) @ (7,3)       ", (7,), (7, 3)),             # 1-D promotion
]
for label, ashape, bshape in cases:
    a = jnp.asarray(rng.standard_normal(ashape, dtype=np.float32))
    b = jnp.asarray(rng.standard_normal(bshape, dtype=np.float32))
    got = jaxmetal.matmul(a, b, device="mps")
    ref = jnp.matmul(a, b)
    ok = got.shape == ref.shape and np.allclose(np.asarray(got), np.asarray(ref), atol=1e-3)
    print(f"  {label} -> {str(got.shape):<10} matches jnp.matmul: {ok}")
