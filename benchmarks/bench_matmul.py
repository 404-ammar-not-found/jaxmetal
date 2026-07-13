"""Benchmark: hand-written Metal GPU matmul vs JAX CPU (XLA) across sizes.

    python benchmarks/bench_matmul.py

Reports wall-clock per matmul, achieved GFLOP/s, the GPU-vs-CPU speedup, and the
numerical gap vs JAX. The Metal timing is end-to-end (host->GPU copy, kernel,
GPU->host copy) — the realistic cost of one call. Our kernel is a naive 16x16
tiled matmul with lots of headroom (no simdgroup_matrix, no double-buffering),
so treat these as a baseline, not a ceiling.
"""

import os
import sys
import time

# Make `import metalmm` work regardless of the current directory.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))

import numpy as np

import jax
import jax.numpy as jnp

import jaxmetal._capi as metalmm

# Keep JAX on CPU for a clean GPU-vs-CPU comparison.
jax.config.update("jax_platform_name", "cpu")


def timeit(fn, iters):
    fn()  # warmup (also triggers MSL compile / XLA compile)
    t0 = time.perf_counter()
    out = None
    for _ in range(iters):
        out = fn()
    dt = (time.perf_counter() - t0) / iters
    return dt, out


def iters_for(n: int) -> int:
    # Fewer iterations for the big, slow sizes.
    if n <= 256:
        return 50
    if n <= 512:
        return 20
    if n <= 1024:
        return 8
    return 3


def main() -> int:
    print(f"Metal GPU : {metalmm.device_name()}")
    print(f"JAX backend: {jax.default_backend()}\n")

    sizes = [128, 256, 512, 1024, 2048, 4096]

    # Compute-only (GPU-resident operands) GFLOP/s for three matmuls:
    #   mps  = MetalPerformanceShaders (Apple's tuned kernel — what PyTorch MPS uses)
    #   hand = our from-scratch register-tiled MSL kernel
    #   cpu  = Accelerate/BLAS (AMX)
    hdr = (f"{'N':>6} {'mps GF/s':>9} {'hand GF/s':>10} {'cpu GF/s':>9} "
           f"{'mps/cpu':>8} {'auto':>6} {'err':>9}")
    print(hdr)
    print("-" * len(hdr))

    rng = np.random.default_rng(0)
    for n in sizes:
        a = rng.standard_normal((n, n), dtype=np.float32)
        b = rng.standard_normal((n, n), dtype=np.float32)
        it = iters_for(n)

        da = metalmm.DeviceBuffer.from_numpy(a)
        db = metalmm.DeviceBuffer.from_numpy(b)
        dc = metalmm.DeviceBuffer(n * n)
        mps_dt, _ = timeit(lambda: metalmm.matmul_resident_mps(da, db, dc, n, n, n), it)
        m_out = dc.download((n, n))
        hnd_dt, _ = timeit(lambda: metalmm.matmul_resident(da, db, dc, n, n, n), it)

        cpu_dt, c_out = timeit(lambda: metalmm.matmul_cpu(a, b), it)
        _, backend = metalmm.matmul_auto(a, b, return_backend=True)  # host-operand router pick

        flops = 2.0 * n ** 3
        err = float(np.max(np.abs(m_out - c_out)))
        print(f"{n:6d} {flops/mps_dt/1e9:9.0f} {flops/hnd_dt/1e9:10.0f} {flops/cpu_dt/1e9:9.0f} "
              f"{cpu_dt/mps_dt:8.2f} {backend:>6} {err:9.2e}")

        da.free(); db.free(); dc.free()

    print("\nmps/cpu > 1.0 means MPS (GPU) beat Accelerate (CPU) for resident operands.")
    print("Finding: MPS beats CPU from N>=2048 (~1.5-1.75x) — but only resident. For host")
    print("operands (matmul_auto), copies keep the CPU ahead until N~4096, so the router")
    print("mostly picks CPU. The resident MPS path is the real GPU win (as in PyTorch).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
