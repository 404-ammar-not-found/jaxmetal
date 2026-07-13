"""Example 3 — the GPU win is real, but only for *resident* operands.

    python examples/03_resident_speed.py

Compares MPS on GPU-resident buffers vs Accelerate (CPU BLAS). MPS wins at large
N — but only because the operands already live on the GPU (no per-call copies).
This is the whole reason device placement matters (and why the PJRT plugin, which
keeps arrays resident, is the endgame). See docs/PJRT_PLUGIN.md.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np

import jaxmetal._capi as metalmm


def timed(fn, iters):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


print("Metal GPU:", metalmm.device_name())
print(f"{'N':>6} {'MPS (resident)':>16} {'CPU (Accelerate)':>18} {'MPS/CPU':>8}")
for n in (512, 1024, 2048, 4096):
    a = np.random.randn(n, n).astype(np.float32)
    b = np.random.randn(n, n).astype(np.float32)

    # Upload once; time compute only (operands stay on the GPU).
    da = metalmm.DeviceBuffer.from_numpy(a)
    db = metalmm.DeviceBuffer.from_numpy(b)
    dc = metalmm.DeviceBuffer(n * n)
    it = 5 if n <= 2048 else 3
    mps = timed(lambda: metalmm.matmul_resident_mps(da, db, dc, n, n, n), it)
    cpu = timed(lambda: metalmm.matmul_cpu(a, b), it)

    f = 2 * n ** 3
    winner = "GPU wins" if mps < cpu else "cpu wins"
    print(f"{n:6d} {f/mps/1e9:12.0f} GF/s {f/cpu/1e9:14.0f} GF/s {cpu/mps:7.2f}x  {winner}")
    da.free(); db.free(); dc.free()
