"""Showcase: jaxmetal.matmul(a, b, device=...) validated against JAX.

    python python/showcase.py

Requires a Python env with jax + numpy (see python/README.md) and the built
libmetal_capi.dylib (cmake --build build).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

import numpy as np

import jax
import jax.numpy as jnp

import jaxmetal


def check(device: str, m: int, k: int, n: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    a = jnp.asarray(rng.standard_normal((m, k), dtype=np.float32))  # jax arrays in
    b = jnp.asarray(rng.standard_normal((k, n), dtype=np.float32))

    out = jaxmetal.matmul(a, b, device=device)      # our front-end (returns jax array)
    ref = jnp.matmul(a, b)                           # native JAX (CPU) reference

    err = float(np.max(np.abs(np.asarray(out) - np.asarray(ref))))
    rel = err / (float(np.max(np.abs(np.asarray(ref)))) + 1e-12)
    status = "OK " if rel < 1e-3 else "BAD"
    print(f"  [{status}] device={device:<5} {m}x{k} @ {k}x{n}  max|Δ|={err:.2e}  rel={rel:.2e}")
    return rel


def main() -> int:
    print(f"Metal GPU  : {jaxmetal.device_name()}")
    print(f"JAX backend: {jax.default_backend()}  (reference: jnp.matmul on CPU)")
    print(f"devices    : {jaxmetal.devices()}\n")

    print("jaxmetal.matmul(a, b, device=...) vs jnp.matmul:")
    worst = 0.0
    for device in ("mps", "metal", "cpu", "auto"):
        worst = max(worst, check(device, 256, 384, 128, 0))
        worst = max(worst, check(device, 512, 512, 512, 1))

    print(f"\nworst relative error: {worst:.2e}")
    if worst < 1e-3:
        print("SUCCESS: every backend matches JAX.")
        return 0
    print("FAILURE: a backend diverged from JAX.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
