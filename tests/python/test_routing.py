"""Gates for `device="auto"` on the scientific ops.

A cost model that is never checked against a clock is a hardcoded guess. These tests
time both arms at points either side of each measured crossover and assert the router
picked the one that actually won.

They also pin the two properties that make a router trustworthy rather than merely
present: it must be monotonic in size (never flip back to CPU as the problem grows),
and both arms must produce the same answer, so routing is a performance decision and
never a numerical one.

Run: .venv/bin/python tests/python/test_routing.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import jaxmetal
from jaxmetal import routing


def best(fn, reps: int = 5) -> float:
    fn()
    t = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t = min(t, time.perf_counter() - t0)
    return t


def test_arms_agree_numerically():
    """Routing must never change the answer, only the runtime."""
    rng = np.random.default_rng(0)

    A = np.ascontiguousarray(rng.standard_normal((4096, 5, 5)).astype(np.float32)
                             + 5 * np.eye(5, dtype=np.float32))
    r = np.ascontiguousarray(rng.standard_normal((4096, 5)).astype(np.float32))
    g = jaxmetal.batched_solve(A, r, device="gpu")
    c = jaxmetal.batched_solve(A, r, device="cpu")
    d = float(np.abs(g - c).max())
    assert d < 1e-4, f"batched_solve arms disagree by {d:.3e}"

    n = 512
    b = rng.standard_normal((n, n)).astype(np.float32)
    spd = np.ascontiguousarray(b @ b.T + n * np.eye(n, dtype=np.float32))
    g = jaxmetal.cholesky(spd, device="gpu")
    c = jaxmetal.cholesky(spd, device="cpu")
    # Both are valid factorisations; compare via the residual rather than elementwise,
    # since they are different algorithms.
    for name, L in (("gpu", g), ("cpu", c)):
        res = float(np.abs(L @ L.T - spd).max() / np.abs(spd).max())
        assert res < 1e-5, f"cholesky {name} residual {res:.3e}"

    x = rng.random(1 << 16) + 1.0
    y = rng.random(1 << 16) + 1.0
    g = jaxmetal.df64_binop(x, y, "mul", device="gpu")
    c = jaxmetal.df64_binop(x, y, "mul", device="cpu")
    d = float(np.abs(g - c).max() / np.abs(c).max())
    assert d < 1e-12, f"df64 arms disagree by {d:.3e}"
    print("[ok] gpu and cpu arms agree for batched_solve, cholesky and df64")


def test_router_is_monotonic():
    """Bigger problems must never flip the decision back to the CPU."""
    for resident in (False, True):
        prev = False
        for batch in (64, 256, 1024, 4096, 16384, 65536, 262144):
            got = routing.prefer_gpu_batched_solve(batch, 6, resident=resident)
            assert not (prev and not got), f"batched_solve flipped back at {batch}"
            prev = got

    prev = False
    for n in (1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26):
        got = routing.prefer_gpu_df64(n, resident=True)
        assert not (prev and not got), f"df64 flipped back at {n}"
        prev = got

    prev = False
    for n in (128, 512, 1024, 2048, 4096):
        got = routing.prefer_gpu_cholesky(n)
        assert not (prev and not got), f"cholesky flipped back at {n}"
        prev = got

    # Residency must never make the GPU look WORSE: copies can only cost time, so the
    # resident crossover has to be at or below the host one.
    assert (routing.kBatchedSolveMinBatchResident
            <= routing.kBatchedSolveMinBatchHost)
    print("[ok] routers are monotonic in size, and residency only lowers thresholds")


def test_env_override():
    for forced in ("gpu", "cpu"):
        os.environ["JAXMETAL_DEVICE"] = forced
        try:
            assert routing.resolve("auto", lambda: forced == "cpu") == forced
            assert routing.resolve("gpu", lambda: True) == forced
        finally:
            del os.environ["JAXMETAL_DEVICE"]
    print("[ok] JAXMETAL_DEVICE overrides both auto and explicit choices")


def test_router_picks_the_faster_arm():
    """The decision must agree with the clock on both sides of each crossover."""
    rng = np.random.default_rng(1)
    worst = 1.0

    for batch in (512, 65536):
        n = 6
        A = np.ascontiguousarray(rng.standard_normal((batch, n, n)).astype(np.float32)
                                 + n * np.eye(n, dtype=np.float32))
        r = np.ascontiguousarray(rng.standard_normal((batch, n)).astype(np.float32))
        t_g = best(lambda: jaxmetal.batched_solve(A, r, device="gpu"))
        t_c = best(lambda: jaxmetal.batched_solve(A, r, device="cpu"))
        faster = "gpu" if t_g < t_c else "cpu"
        chose = "gpu" if routing.prefer_gpu_batched_solve(batch, n) else "cpu"
        ratio = (t_g if chose == "gpu" else t_c) / min(t_g, t_c)
        worst = max(worst, ratio)
        print(f"[route] batched_solve batch={batch:<7} gpu={t_g*1e3:7.3f}ms "
              f"cpu={t_c*1e3:7.3f}ms faster={faster} chose={chose} ({ratio:.2f}x)")

    for n in (256, 3072):
        b = rng.standard_normal((n, n)).astype(np.float32)
        spd = np.ascontiguousarray(b @ b.T + n * np.eye(n, dtype=np.float32))
        t_g = best(lambda: jaxmetal.cholesky(spd, device="gpu"), reps=3)
        t_c = best(lambda: jaxmetal.cholesky(spd, device="cpu"), reps=3)
        faster = "gpu" if t_g < t_c else "cpu"
        chose = "gpu" if routing.prefer_gpu_cholesky(n) else "cpu"
        ratio = (t_g if chose == "gpu" else t_c) / min(t_g, t_c)
        worst = max(worst, ratio)
        print(f"[route] cholesky      N={n:<9} gpu={t_g*1e3:7.3f}ms "
              f"cpu={t_c*1e3:7.3f}ms faster={faster} chose={chose} ({ratio:.2f}x)")

    # Near a crossover the arms are within noise, so choosing "wrong" costs nothing;
    # the band is what makes this a stable gate rather than a flaky one.
    assert worst < 1.35, f"router landed {worst:.2f}x off the faster arm"
    print(f"[ok] router never landed more than {worst:.2f}x off the faster arm")


if __name__ == "__main__":
    test_arms_agree_numerically()
    test_router_is_monotonic()
    test_env_override()
    test_router_picks_the_faster_arm()
    print("\nall routing gates passed")
