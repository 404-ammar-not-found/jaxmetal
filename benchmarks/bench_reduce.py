"""Compensated f32 summation on the GPU: accuracy and throughput.

Two questions, because either one alone is misleading:

  1. How much accuracy does compensation actually buy, on inputs that matter?
  2. What does it cost, against an uncompensated tree sum with identical memory
     traffic and launch shape?

The baseline choices are deliberate. `numpy.sum` on f32 is *not* a naive loop --
it is pairwise, so it is already fairly accurate; beating a naive Python loop would
prove nothing. The GPU tree sum is the honest baseline for cost, since it differs
from the compensated kernel only in the arithmetic, not the parallel structure.

Ground truth throughout is float64 accumulation on the CPU. Apple GPUs have no f64
at all (Metal has no `double`), which is the whole reason compensation matters here
rather than "just use float64".

Run: .venv/bin/python benchmarks/bench_reduce.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import jaxmetal
from jaxmetal import _capi

GRID_STRIDE = 512 * 256   # kReduceGroups * kReduceTG in src/ops/reduce.mm


def cases(n: int) -> dict:
    """Input distributions, from benign to adversarial."""
    rng = np.random.default_rng(0)
    out = {}

    # Benign: mean zero, similar magnitudes. Everything handles this well.
    out["uniform [-1,1)"] = rng.uniform(-1, 1, n).astype(np.float32)

    # All-positive: the running total grows without bound, so late addends fall
    # further and further below its ulp. The classic large-sum failure.
    out["uniform [0,1)"] = rng.uniform(0, 1, n).astype(np.float32)

    # One large value per thread's grid-stride slot, then values far below its ulp
    # (ulp(1e8) in f32 is 8, so every +1.0 rounds away). The loss happens inside a
    # single thread's sequential run, where the tree structure cannot help.
    v = np.ones(n, np.float32)
    v[:GRID_STRIDE] = 1e8
    out["1e8 prefix + ones"] = v

    # Heavy dynamic range: magnitudes spanning ~10 decades in random order.
    e = rng.uniform(-5, 5, n)
    out["log-uniform 1e-5..1e5"] = (10.0 ** e).astype(np.float32)

    return out


def accuracy(n: int) -> None:
    print(f"\n=== accuracy (n = {n:,}) — relative error vs float64 ===\n")
    hdr = f"{'input':<24} {'GPU compensated':>16} {'GPU tree':>12} {'numpy f32':>12}"
    print(hdr)
    print("-" * len(hdr))
    for name, x in cases(n).items():
        want = float(np.sum(x, dtype=np.float64))

        def rel(got: float) -> float:
            return abs(got - want) / abs(want) if want != 0 else abs(got)

        e_comp = rel(jaxmetal.reduce_sum(x, compensated=True))
        e_tree = rel(jaxmetal.reduce_sum(x, compensated=False))
        e_np = rel(float(np.sum(x, dtype=np.float32)))
        print(f"{name:<24} {e_comp:>16.2e} {e_tree:>12.2e} {e_np:>12.2e}")

    print("\nf32 machine epsilon is 1.19e-07 — a relative error at or below that is")
    print("as good as an f32 result can be. Note numpy's f32 sum is pairwise, not")
    print("naive, so it is a genuinely strong baseline rather than a straw man.")


def throughput(sizes) -> None:
    print(f"\n=== throughput (GPU-resident, no host copy) ===\n")
    hdr = (f"{'n':>12} {'bytes':>9} {'compensated':>14} {'tree':>14} "
           f"{'overhead':>9} {'numpy f64':>12}")
    print(hdr)
    print("-" * len(hdr))
    # Below ~16M elements the two-pass dispatch (two command buffers, ~140 us of
    # driver round trip each) dominates, so the GB/s columns there measure
    # submission latency rather than bandwidth. Read the large rows.

    for n in sizes:
        x = np.random.default_rng(1).uniform(0, 1, n).astype(np.float32)
        buf = _capi.DeviceBuffer.from_numpy(x)
        nbytes = n * 4

        # Min over trials, not mean: the fastest run is the one least perturbed by
        # scheduling noise, and these kernels have no warm-up state to amortise.
        def bench(compensated: bool) -> float:
            _capi.reduce_sum_resident(buf, n, compensated)          # warm
            best = float("inf")
            for _ in range(10):
                t0 = time.perf_counter()
                for _ in range(10):
                    _capi.reduce_sum_resident(buf, n, compensated)
                best = min(best, (time.perf_counter() - t0) / 10)
            return best

        t_comp = bench(True)
        t_tree = bench(False)

        xd = x.astype(np.float64)   # what you would otherwise do for accuracy
        np.sum(xd)
        t_np = float("inf")
        for _ in range(10):
            t0 = time.perf_counter()
            for _ in range(10):
                np.sum(xd)
            t_np = min(t_np, (time.perf_counter() - t0) / 10)

        print(f"{n:>12,} {nbytes/1e6:>7.0f}MB "
              f"{nbytes/t_comp/1e9:>10.1f} GB/s {nbytes/t_tree/1e9:>10.1f} GB/s "
              f"{t_comp/t_tree:>8.2f}x {xd.nbytes/t_np/1e9:>7.1f} GB/s")
        del buf

    print("\nM4 Pro peak memory bandwidth is ~273 GB/s; both GPU kernels are")
    print("bandwidth-bound, which is why compensation is close to free. The numpy")
    print("f64 column moves twice the bytes for the same element count — the cost")
    print("of the usual 'promote to float64' workaround, which on Apple Silicon")
    print("cannot run on the GPU at all.")


def main() -> None:
    print("device:", jaxmetal.device_name())
    accuracy(1 << 24)
    throughput([1 << 20, 1 << 22, 1 << 24, 1 << 26])


if __name__ == "__main__":
    main()
