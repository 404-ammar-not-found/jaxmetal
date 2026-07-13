"""Verify jaxmetal.matmul matches jnp.matmul across shapes, ranks, and backends.

    python python/test_jaxmetal.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))

import numpy as np
import jax.numpy as jnp

import jaxmetal


CASES = [
    # (a_shape, b_shape) — covering 1-D promotion, 2-D, batched, and broadcasting.
    ((5,), (5,)),                 # 1-D . 1-D -> scalar
    ((7,), (7, 3)),               # 1-D @ 2-D
    ((4, 6), (6,)),               # 2-D @ 1-D
    ((8, 5), (5, 9)),             # 2-D @ 2-D
    ((3, 8, 5), (3, 5, 9)),       # batched
    ((3, 8, 5), (5, 9)),          # batch @ 2-D (broadcast)
    ((8, 5), (3, 5, 9)),          # 2-D @ batch (broadcast)
    ((2, 1, 8, 5), (1, 4, 5, 9)), # multi-dim broadcast
    ((1, 7), (3, 7, 2)),          # 1-D-ish rows broadcast
]


def main() -> int:
    rng = np.random.default_rng(0)
    fails = 0
    for device in ("mps", "metal", "cpu", "auto"):
        for ashape, bshape in CASES:
            a = rng.standard_normal(ashape).astype(np.float32)
            b = rng.standard_normal(bshape).astype(np.float32)
            got = np.asarray(jaxmetal.matmul(a, b, device=device))
            ref = np.asarray(jnp.matmul(jnp.asarray(a), jnp.asarray(b)))
            ok = got.shape == ref.shape and np.allclose(got, ref, rtol=1e-3, atol=1e-3)
            if not ok:
                fails += 1
                print(f"  [FAIL] device={device} {ashape} @ {bshape}: "
                      f"got {got.shape} vs ref {ref.shape}, "
                      f"max|Δ|={np.max(np.abs(got - ref)) if got.shape == ref.shape else 'shape'}")
    total = len(CASES) * 4
    print(f"\n{total - fails}/{total} passed")
    if fails == 0:
        print("SUCCESS: jaxmetal.matmul matches jnp.matmul (all shapes, all backends).")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
