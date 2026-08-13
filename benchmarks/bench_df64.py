"""Double-single (df64) extended precision on the GPU: what it buys, and what it costs.

READ THE COST SECTION BEFORE USING THIS. df64 is a PRECISION feature. It is slower
than doing the same work in float64 on the CPU, at every size. The original rationale
-- that the GPU's higher memory bandwidth would pay for the emulation on
bandwidth-bound kernels -- was WRONG: Apple Silicon is unified memory, so the CPU and
GPU share one memory controller and there is no bandwidth advantage to exploit.

It exists because Metal has no `double` type at all, so a pipeline already resident on
the GPU otherwise has no way to exceed f32 without round-tripping to the host.

Run: .venv/bin/python benchmarks/bench_df64.py
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


def accuracy() -> None:
    print("\n=== accuracy: relative error vs float64 ===\n")
    hdr = f"{'case':<34} {'GPU df64':>11} {'GPU f32':>11} {'improvement':>12}"
    print(hdr)
    print("-" * len(hdr))
    rng = np.random.default_rng(0)

    # Cancellation: a + b where b ~= -a. The result is 1e-5 of the operands, so the
    # relative error of both precisions is amplified ~1e5x -- see the tolerance note
    # in tests/cpp/df64_test.cpp. df64 has 24 more bits to give away.
    a = rng.uniform(1.0, 2.0, 1 << 20)
    b = -(a - 1e-5 * a)
    want = a + b
    got = jaxmetal.df64_binop(a, b, "add")
    f32 = (a.astype(np.float32) + b.astype(np.float32)).astype(np.float64)
    e_df = np.abs(got - want).max() / np.abs(want).max()
    e_32 = np.abs(f32 - want).max() / np.abs(want).max()
    print(f"{'a + b with b ~= -a (cancel 1e5)':<34} {e_df:>11.2e} {e_32:>11.2e} "
          f"{e_32/max(e_df,1e-300):>11.0f}x")

    # Products needing more than 24 bits.
    x = rng.uniform(0.5, 2.0, 1 << 20)
    y = rng.uniform(0.5, 2.0, 1 << 20)
    want = x * y
    got = jaxmetal.df64_binop(x, y, "mul")
    f32 = (x.astype(np.float32) * y.astype(np.float32)).astype(np.float64)
    e_df = np.abs(got - want).max() / np.abs(want).max()
    e_32 = np.abs(f32 - want).max() / np.abs(want).max()
    print(f"{'a * b':<34} {e_df:>11.2e} {e_32:>11.2e} {e_32/max(e_df,1e-300):>11.0f}x")

    # Second difference of a smooth field: the PDE case. Normwise error, because the
    # second difference of a sinusoid legitimately passes through zero.
    n = 1 << 20
    fld = 1.0 + np.sin(0.01 * np.arange(n))
    want = np.zeros(n)
    want[1:-1] = fld[:-2] - 2.0 * fld[1:-1] + fld[2:]
    g_df = jaxmetal.df64_stencil3(fld, use_df64=True)
    g_32 = jaxmetal.df64_stencil3(fld, use_df64=False)
    scale = np.abs(want[1:-1]).max()
    e_df = np.abs(g_df[1:-1] - want[1:-1]).max() / scale
    e_32 = np.abs(g_32[1:-1] - want[1:-1]).max() / scale
    print(f"{'3-pt stencil (1,-2,1), h=0.01':<34} {e_df:>11.2e} {e_32:>11.2e} "
          f"{e_32/max(e_df,1e-300):>11.0f}x")

    print("\nf32 eps is 1.19e-07; df64 targets ~48 bits (~3.6e-15). NOT IEEE float64:")
    print("f32's exponent range, no guaranteed correct rounding.")


def cost() -> None:
    print("\n=== cost: the same work in float64 on the CPU ===\n")
    hdr = f"{'n':>10} {'GPU df64':>11} {'CPU f64':>11} {'GPU f32':>11} {'df64 vs CPU':>12}"
    print(hdr)
    print("-" * len(hdr))
    for n in (1 << 20, 1 << 22, 1 << 24):
        rng = np.random.default_rng(1)
        a = rng.random(n) + 1.0
        b = rng.random(n) + 1.0
        af, bf = a.astype(np.float32), b.astype(np.float32)

        t_df = best(lambda: jaxmetal.df64_binop(a, b, "add"))
        t_cpu = best(lambda: a + b)
        t_f32 = best(lambda: af + bf)
        print(f"{n:>10,} {t_df*1e3:>9.2f}ms {t_cpu*1e3:>9.2f}ms {t_f32*1e3:>9.2f}ms "
              f"{t_cpu/t_df:>11.2f}x")

    print("\nBelow 1.0x means the CPU wins, and it does. Two reasons, both structural:")
    print("  * Unified memory: CPU and GPU share one memory controller, so there is no")
    print("    GPU bandwidth advantage to pay for the ~10-20 ops per df64 operation.")
    print("  * df64 moves 2x the bytes of f32 for the same element count, and these")
    print("    kernels are bandwidth-bound.")
    print("Use df64 when you need the digits on data that is ALREADY resident, not to")
    print("go faster. If the data is on the host, numpy float64 is the better answer.")


def main() -> None:
    print("device:", jaxmetal.device_name())
    accuracy()
    cost()


if __name__ == "__main__":
    main()
