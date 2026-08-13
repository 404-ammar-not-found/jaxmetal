"""Batched tiny-system solve: GPU thread-per-system vs honest CPU baselines.

BASELINE CHOICE MATTERS MORE THAN USUAL HERE. Looping `numpy.linalg.solve` over a
batch measures numpy's per-call overhead, not linear algebra, and would flatter the
GPU by one to two orders of magnitude. Two fairer baselines are used instead:

  * a plain scalar C loop (`jaxmetal.batched_solve_cpu`) -- the same LU with partial
    pivoting, single-threaded. This is what a competent C programmer would write, and
    it is several times faster than looping numpy.
  * `numpy.linalg.solve` on the STACKED array, which numpy genuinely batches inside
    one call. This is what a numpy user would actually write.

Both are reported. The GPU number includes the host round trip (allocate, upload,
solve, download), because that is what `jaxmetal.batched_solve(numpy_array)` costs --
quoting a resident-only figure would not be what anyone actually gets.

Run: .venv/bin/python benchmarks/bench_batched_solve.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import jaxmetal


def best(fn, reps: int = 5) -> float:
    fn()
    t = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t = min(t, time.perf_counter() - t0)
    return t


def make(batch: int, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Diagonal boost keeps the conditioning sane so the accuracy column measures the
    # solver rather than the draw; pivoting still fires on the off-diagonal noise.
    A = rng.standard_normal((batch, n, n)).astype(np.float32)
    A += (n * np.eye(n, dtype=np.float32))[None]
    rhs = rng.standard_normal((batch, n)).astype(np.float32)
    return np.ascontiguousarray(A), np.ascontiguousarray(rhs)


def throughput(batch: int) -> None:
    print(f"\n=== throughput at batch = {batch:,} (systems/second) ===\n")
    hdr = (f"{'n':>3} {'GPU':>13} {'C loop':>13} {'numpy':>13} "
           f"{'vs C':>7} {'vs numpy':>9} {'resid':>9}")
    print(hdr)
    print("-" * len(hdr))

    for n in range(2, 9):
        A, rhs = make(batch, n)
        rhs_col = rhs[..., None]

        t_gpu = best(lambda: jaxmetal.batched_solve(A, rhs))
        t_c = best(lambda: jaxmetal.batched_solve_cpu(A, rhs))
        t_np = best(lambda: np.linalg.solve(A, rhs_col))

        x = jaxmetal.batched_solve(A, rhs)
        r = np.einsum("bij,bj->bi", A.astype(np.float64), x.astype(np.float64)) - rhs
        anorm = np.abs(A).sum(axis=2).max(axis=1)
        xnorm = np.abs(x).max(axis=1)
        resid = float((np.abs(r).max(axis=1) / (anorm * np.maximum(xnorm, 1e-30))).max())

        print(f"{n:>3} {batch/t_gpu:>13.3e} {batch/t_c:>13.3e} {batch/t_np:>13.3e} "
              f"{t_c/t_gpu:>6.1f}x {t_np/t_gpu:>8.1f}x {resid:>9.1e}")


def crossover(n: int = 6) -> None:
    """Below some batch the command-buffer round trip costs more than the whole solve."""
    print(f"\n=== crossover in batch size (n = {n}) ===\n")
    hdr = f"{'batch':>9} {'GPU ms':>9} {'C loop ms':>11} {'numpy ms':>10} {'winner':>8}"
    print(hdr)
    print("-" * len(hdr))
    for batch in (64, 256, 1024, 4096, 16384, 65536, 262144):
        A, rhs = make(batch, n)
        rhs_col = rhs[..., None]
        t_gpu = best(lambda: jaxmetal.batched_solve(A, rhs))
        t_c = best(lambda: jaxmetal.batched_solve_cpu(A, rhs))
        t_np = best(lambda: np.linalg.solve(A, rhs_col))
        cpu_best = min(t_c, t_np)
        print(f"{batch:>9,} {t_gpu*1e3:>8.3f} {t_c*1e3:>10.3f} {t_np*1e3:>9.3f} "
              f"{'GPU' if t_gpu < cpu_best else 'CPU':>8}")


def main() -> None:
    print("device:", jaxmetal.device_name())
    throughput(1 << 16)
    crossover()
    print("\nThe GPU column includes the host round trip, which is what a caller passing")
    print("numpy arrays actually pays. These kernels do well under one FLOP per byte")
    print("moved, so residency matters more here than it does for matmul: keeping the")
    print("data resident across several solves would move the crossover down sharply.")
    print("\nAbove n=8 this op raises. Apple's batched MPSMatrixDecompositionLU wins")
    print("there (measured 2.4x at n=12, 54x at n=32), so use numpy or MPS instead.")


if __name__ == "__main__":
    main()
