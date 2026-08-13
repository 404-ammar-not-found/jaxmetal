"""Double-single (df64) extended precision on the GPU: what it buys, and what it costs.

RESIDENCY DECIDES WHETHER THIS IS FAST. Resident, df64 is 1.65-1.88x FASTER than CPU
float64 above ~4M elements while carrying ~48 bits of significand. Through the
host-operand path it is ~0.04x, because copying 2*n floats in and out costs 36x the
kernel. Both are measured below; do not quote one for the other.

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
    hdr = (f"{'n':>10} {'df64 resident':>14} {'df64 host':>11} {'CPU f64':>10} "
           f"{'GPU f32':>10} {'res vs CPU':>11}")
    print(hdr)
    print("-" * len(hdr))
    from jaxmetal import _capi
    for n in (1 << 20, 1 << 22, 1 << 24, 1 << 26):
        rng = np.random.default_rng(1)
        a = rng.random(n) + 1.0
        b = rng.random(n) + 1.0
        af, bf = a.astype(np.float32), b.astype(np.float32)

        bA = _capi.DeviceBuffer.from_numpy(jaxmetal.to_df64(a).ravel())
        bB = _capi.DeviceBuffer.from_numpy(jaxmetal.to_df64(b).ravel())
        bO = _capi.DeviceBuffer.from_numpy(np.zeros(2 * n, np.float32))
        t_res = best(lambda: _capi.df64_binop_resident(bA, bB, bO, n, "add"))
        t_df = best(lambda: jaxmetal.df64_binop(a, b, "add"))
        t_cpu = best(lambda: a + b)
        t_f32 = best(lambda: af + bf)
        print(f"{n:>10,} {t_res*1e3:>12.2f}ms {t_df*1e3:>9.2f}ms {t_cpu*1e3:>8.2f}ms "
              f"{t_f32*1e3:>8.2f}ms {t_cpu/t_res:>10.2f}x")
        del bA, bB, bO

    print("\nThe crossover is ~4M elements: below it the command-buffer round trip")
    print("dominates, above it df64 beats CPU float64 outright while carrying ~48 bits.")
    print("\nCompare the two df64 columns. They run the SAME kernel; the host one just")
    print("copies 2*n floats in and out, and that is 36x the cost at 16.7M elements.")
    print("\nNote GPU f32: df64 costs only ~12% more than plain f32 on the GPU despite")
    print("moving twice the bytes. Larger accesses use the memory system better, so the")
    print("emulation is close to free once you are bandwidth-bound. Unified memory does")
    print("mean there is no raw bandwidth RATIO to exploit -- but the GPU still achieves")
    print("higher streaming bandwidth on this access pattern than the CPU does.")


def main() -> None:
    print("device:", jaxmetal.device_name())
    accuracy()
    cost()


if __name__ == "__main__":
    main()
