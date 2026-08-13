"""Measured GPU-vs-CPU crossovers for the scientific ops, and `device="auto"`.

Every op in this package has a size below which the CPU wins. Handing a user a GPU
routine that is 3x slower on their problem size is worse than not shipping it, so each
op that has a measured crossover takes `device="auto" | "gpu" | "cpu"`.

RESIDENCY MOVES THE CROSSOVER, AND IT MOVES IT A LOT. These kernels do well under one
FLOP per byte moved, so copying host arrays in and out can cost more than the compute:

    batched_solve (n=6)   host operands: GPU wins from ~5,000 systems
                          resident:      GPU wins from ~2,500 systems
    df64 elementwise      host operands: CPU ALWAYS wins (copies are 41x the kernel)
                          resident:      GPU wins from ~4M elements

So the routers below take `resident` explicitly rather than assuming. A host-operand
call and a resident call are different operations with different answers.

All constants are Apple M4 Pro measurements, reproducible with the scripts in
`benchmarks/`. Any other machine moves them; override with `JAXMETAL_DEVICE=gpu|cpu`.
"""
from __future__ import annotations

import os

DEVICES = ("auto", "gpu", "cpu")


def _forced() -> str | None:
    """Global override. Returns "gpu", "cpu", or None."""
    v = os.environ.get("JAXMETAL_DEVICE")
    return v if v in ("gpu", "cpu") else None


def resolve(device: str, prefer_gpu_fn) -> str:
    """Turn a user-supplied device string into "gpu" or "cpu"."""
    if device not in DEVICES:
        raise ValueError(f"unknown device {device!r}; expected one of {DEVICES}")
    forced = _forced()
    if forced:
        return forced
    if device != "auto":
        return device
    return "gpu" if prefer_gpu_fn() else "cpu"


# ---------------------------------------------------------------------------
# batched_solve: thousands of tiny systems.
#
# Measured at n=6 against the scalar C loop (benchmarks/bench_batched_solve.py):
#   resident  1,024 -> 0.131 ms vs 0.053  (CPU)     4,096 -> 0.163 vs 0.205  (GPU)
#   host      4,096 -> 0.218 ms vs 0.205  (CPU)    16,384 -> 0.366 vs 0.807  (GPU)
# The ratio is fairly flat in n (1.8-2.4x at batch 65,536 across n=2..8), so the
# threshold is on batch alone rather than on batch x n^3.
kBatchedSolveMinBatchResident = 2500
kBatchedSolveMinBatchHost = 5000


def prefer_gpu_batched_solve(batch: int, n: int, resident: bool = False) -> bool:
    if n < 2 or n > 8:
        return False   # outside the kernel's range; caller must use another path
    floor = kBatchedSolveMinBatchResident if resident else kBatchedSolveMinBatchHost
    return batch >= floor


# ---------------------------------------------------------------------------
# df64 elementwise.
#
# Measured (benchmarks/bench_df64.py), resident vs CPU float64:
#   1.0M 0.65x   4.2M 1.07x   16.8M 1.76x   67.1M 1.93x
# Host operands never win: 288 ms against the CPU's 13.5 ms at 67M, because the
# copies are 41x the kernel. There is no size at which the host path is the right
# choice for speed, so `auto` always routes it to the CPU.
kDF64MinElemsResident = 4_000_000


def prefer_gpu_df64(n: int, resident: bool = False) -> bool:
    return bool(resident) and n >= kDF64MinElemsResident


# ---------------------------------------------------------------------------
# Dense Cholesky.
#
# Measured (benchmarks/bench_cholesky.py) against np.linalg.cholesky, which is what a
# Python caller actually invokes:
#   N=512 0.23x   N=1024 0.66x   N=2048 2.01x   N=4096 3.77x
# so the crossover against numpy is ~N=1300.
#
# CAVEAT, and it is why this threshold is not lower: numpy is NOT the fastest CPU
# option. A direct `spotrf` on a Fortran-ordered array is ~4x faster than numpy
# (it skips the reorder), and against that the GPU only reaches parity at N=4096.
# If your CPU path is hand-tuned LAPACK rather than numpy, set device="cpu".
kCholeskyMinNVsNumpy = 1300


def prefer_gpu_cholesky(n: int) -> bool:
    return n >= kCholeskyMinNVsNumpy
