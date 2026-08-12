"""Gates for the chunked trainer and the device="auto" router.

Two things that are easy to get silently wrong:
  1. train_steps(n) must produce exactly the same parameters as n separate
     train_step() calls — the whole point is that batching steps into one command
     buffer changes only *when* work is submitted, never what it computes.
  2. The router must pick the arm that is actually faster. A cost model that is
     never checked against a clock is just a hardcoded guess.

Run: .venv/bin/python tests/python/test_mlp_auto.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import jaxmetal
import jaxmetal.reference as ref
from jaxmetal.mlp import crossover_work, prefer_gpu, step_work

IN_DIM, OUT_DIM = 784, 10


def test_chunked_matches_per_step():
    """train_steps(n) == n x train_step(), parameter for parameter."""
    rng = np.random.default_rng(0)
    hidden, batch, n = 64, 32, 8
    p = ref.init_params(0, IN_DIM, hidden, OUT_DIM)
    X = rng.standard_normal((n * batch, IN_DIM)).astype(np.float32)
    y = rng.integers(0, OUT_DIM, n * batch).astype(np.int32)

    chunked = jaxmetal.Mlp(IN_DIM, hidden, OUT_DIM, batch, chunk_steps=n, device="gpu")
    chunked.set_params(p["W1"], p["b1"], p["W2"], p["b2"])
    chunked.upload_chunk(X, y)
    chunked.train_steps(n, batch, 0.1)

    stepped = jaxmetal.Mlp(IN_DIM, hidden, OUT_DIM, batch, device="gpu")
    stepped.set_params(p["W1"], p["b1"], p["W2"], p["b2"])
    for s in range(n):
        stepped.upload_batch(X[s * batch:(s + 1) * batch], y[s * batch:(s + 1) * batch])
        stepped.train_step(0.1)

    for name, a, b in zip(("W1", "b1", "W2", "b2"),
                          chunked.get_params(), stepped.get_params()):
        d = float(np.max(np.abs(a - b)))
        assert d == 0.0, f"chunked vs per-step {name} differ by {d:.3e}"
    print("[ok] train_steps(n) is bit-identical to n x train_step()")


def test_cpu_arm_matches_gpu_arm():
    """Both arms train the same model, so they must agree to f32 tolerance."""
    rng = np.random.default_rng(1)
    hidden, batch = 64, 32
    p = ref.init_params(0, IN_DIM, hidden, OUT_DIM)
    X = rng.standard_normal((batch, IN_DIM)).astype(np.float32)
    y = rng.integers(0, OUT_DIM, batch).astype(np.int32)

    out = {}
    for dev in ("gpu", "cpu"):
        m = jaxmetal.Mlp(IN_DIM, hidden, OUT_DIM, batch, device=dev)
        assert m.device == dev
        m.set_params(p["W1"], p["b1"], p["W2"], p["b2"])
        m.upload_batch(X, y)
        loss = m.train_step(0.1)
        out[dev] = (loss, m.get_params())

    assert abs(out["gpu"][0] - out["cpu"][0]) < 1e-5, (out["gpu"][0], out["cpu"][0])
    for name, a, b in zip(("W1", "b1", "W2", "b2"), out["gpu"][1], out["cpu"][1]):
        d = float(np.max(np.abs(a - b)))
        assert d < 1e-5, f"gpu vs cpu {name} max|d|={d:.3e}"
    print("[ok] cpu arm matches gpu arm (loss + all four parameters)")


def test_router_is_monotonic_and_forceable():
    """Bigger work must never flip the decision back to CPU, and env forces win."""
    prev = False
    for batch in (1, 8, 32, 128, 512, 2048, 8192):
        got = prefer_gpu(IN_DIM, 128, OUT_DIM, batch, chunk_steps=128)
        assert not (prev and not got), f"router flipped back to cpu at batch={batch}"
        prev = got
    assert prefer_gpu(IN_DIM, 128, OUT_DIM, 8192, chunk_steps=128), "never picks gpu"
    assert not prefer_gpu(IN_DIM, 128, OUT_DIM, 1, chunk_steps=128), "never picks cpu"

    # A small chunk leaves the round trip unamortised, so the crossover must move right.
    assert crossover_work(1) > crossover_work(128)

    for forced in ("gpu", "cpu"):
        os.environ["JAXMETAL_MLP_DEVICE"] = forced
        try:
            assert prefer_gpu(IN_DIM, 128, OUT_DIM, 1) == (forced == "gpu")
        finally:
            del os.environ["JAXMETAL_MLP_DEVICE"]
    print("[ok] router is monotonic in work, and JAXMETAL_MLP_DEVICE forces an arm")


def test_router_picks_the_faster_arm():
    """The cost model must agree with the clock, near and away from the crossover."""
    rng = np.random.default_rng(2)
    chunk = 64
    cases = [(128, 32), (128, 512), (1024, 256)]
    for hidden, batch in cases:
        p = ref.init_params(0, IN_DIM, hidden, OUT_DIM)
        X = rng.standard_normal((chunk * batch, IN_DIM)).astype(np.float32)
        y = rng.integers(0, OUT_DIM, chunk * batch).astype(np.int32)

        timed = {}
        for dev in ("gpu", "cpu"):
            m = jaxmetal.Mlp(IN_DIM, hidden, OUT_DIM, batch, chunk_steps=chunk,
                             device=dev)
            m.set_params(p["W1"], p["b1"], p["W2"], p["b2"])
            m.upload_chunk(X, y)
            m.train_steps(chunk, batch, 0.01)          # warm up / build plans
            t0 = time.perf_counter()
            m.train_steps(chunk, batch, 0.01)
            timed[dev] = (time.perf_counter() - t0) / chunk

        faster = "gpu" if timed["gpu"] < timed["cpu"] else "cpu"
        chosen = "gpu" if prefer_gpu(IN_DIM, hidden, OUT_DIM, batch, chunk) else "cpu"
        ratio = timed[chosen] / timed[faster]
        work = step_work(IN_DIM, hidden, OUT_DIM, batch)
        print(f"[router] hidden={hidden:<5} batch={batch:<5} work={work:.2e}  "
              f"gpu={timed['gpu']*1e3:.3f}ms cpu={timed['cpu']*1e3:.3f}ms  "
              f"faster={faster} chose={chosen} ({ratio:.2f}x)")
        # Near the crossover the arms are within noise of each other, so require the
        # choice to be no worse than 15% off the better arm rather than always exact.
        assert ratio < 1.15, (f"router chose {chosen} but {faster} was {1/ratio:.2f}x "
                              f"faster at hidden={hidden} batch={batch}")
    print("[ok] router never lands more than 15% off the faster arm")


if __name__ == "__main__":
    test_chunked_matches_per_step()
    test_cpu_arm_matches_gpu_arm()
    test_router_is_monotonic_and_forceable()
    test_router_picks_the_faster_arm()
    print("\nall mlp auto/chunk gates passed")
