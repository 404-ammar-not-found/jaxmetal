"""Backend routing for the MLP trainer: `Mlp(..., device="auto"|"gpu"|"cpu")`.

WHY THIS EXISTS. The GPU does not win at every size, and the crossover is not where
a total-FLOP threshold would put it. A train step costs

    fixed  +  marginal * work,        work = batch * (in_dim*hidden + hidden*out_dim)

and the GPU loses on `fixed`, not on `marginal`. Per-step time measured on an M4 Pro
with chunk_steps=128 (large enough to amortise the command-buffer round trip):

    hidden  batch    GPU ms   CPU ms   speedup
       128     32     0.150    0.097     0.65x
       128    128     0.137    0.186     1.36x
       128    512     0.200    0.446     2.23x
       128   2048     0.489    1.098     2.24x
      1024     32     0.201    0.308     1.53x
      1024   2048     2.004    6.421     3.20x

The GPU's marginal rate is ~3x better than the CPU's, but it carries a ~0.16 ms
per-step floor (CPU-side encoding of ~18 dispatches plus their GPU dispatch cost)
against the CPU's ~0.02 ms. So it wins from batch ~75 at hidden=128 and loses below
that — while at hidden=1024 it already wins at batch 32. A total-FLOP threshold
cannot express this, which is why the matmul router's kAutoGpuFlopThreshold is the
wrong shape here.

THE MODEL IS A CROSSOVER, NOT TWO COST LINES. Fitting `fixed + marginal*work`
separately per arm is ill-conditioned here: across a 500x range of `work` the fit is
dominated by the large end and comes back with kCpuFixed > kGpuFixed, which inverts
reality (the CPU is the one with the small fixed cost) and predicts "GPU always
wins". What is actually well behaved is the *speedup ratio*, which is close to a
power law in `work` near the boundary:

    log(t_cpu / t_gpu) = kSpeedupA + kSpeedupB * log(work)

so the decision collapses to a single number: the `work` at which that crosses 1.

The constants below are M4 Pro measurements. Re-fit on other hardware with
`examples/train_mnist.py --calibrate`, or force an arm with
JAXMETAL_MLP_DEVICE=gpu|cpu.
"""
from __future__ import annotations

import math
import os

import numpy as np

from . import reference as _ref

DEVICES = ("auto", "gpu", "cpu")

# log(speedup) = kSpeedupA + kSpeedupB * log(work), fitted near the crossover.
kSpeedupA = -4.954
kSpeedupB = 0.313
# GPU per-step floor (ms) excluding the command-buffer round trip, and the round trip
# itself. Only their ratio is used, to shift the crossover when a chunk is small.
kGpuFixed = 0.16
kRoundTrip = 0.14


def step_work(in_dim: int, hidden: int, out_dim: int, batch: int) -> float:
    """Multiply-accumulates in one train step, up to a constant factor."""
    return float(batch) * (in_dim * hidden + hidden * out_dim)


def crossover_work(chunk_steps: int = 1) -> float:
    """`work` above which the GPU is predicted to win, for a given chunk size.

    The fit is taken at a large chunk, where the round trip is amortised away. A
    smaller chunk inflates the GPU's fixed cost by kRoundTrip/chunk_steps and pushes
    the crossover proportionally to the right.
    """
    base = math.exp(-kSpeedupA / kSpeedupB)
    inflate = (kGpuFixed + kRoundTrip / max(1, chunk_steps)) / kGpuFixed
    return base * inflate


def prefer_gpu(in_dim: int, hidden: int, out_dim: int, batch: int,
               chunk_steps: int = 1) -> bool:
    """Whether the GPU arm is predicted to be faster for this shape."""
    env = os.environ.get("JAXMETAL_MLP_DEVICE")
    if env in ("gpu", "cpu"):
        return env == "gpu"
    return step_work(in_dim, hidden, out_dim, batch) >= crossover_work(chunk_steps)


class CpuMlp:
    """CPU arm, mirroring the resident-GPU `Mlp` surface.

    Deliberately no new numerics: this delegates to `jaxmetal.reference`, the NumPy
    golden reference already gate-verified against `jax.grad` to ~1e-8. Its matmuls
    are plain `@`, so they run on Accelerate/BLAS — this is a fast CPU path, not a
    teaching-only one.
    """

    def __init__(self, in_dim, hidden, out_dim, max_batch, chunk_steps=1):
        self.in_dim, self.hidden, self.out_dim = in_dim, hidden, out_dim
        self.max_batch, self.chunk_steps = max_batch, chunk_steps
        self._p = None
        self._X = self._y = None
        self._loss = 0.0

    def set_params(self, W1, b1, W2, b2):
        self._p = dict(W1=np.array(W1, np.float32).reshape(self.in_dim, self.hidden),
                       b1=np.array(b1, np.float32).ravel(),
                       W2=np.array(W2, np.float32).reshape(self.hidden, self.out_dim),
                       b2=np.array(b2, np.float32).ravel())

    def get_params(self):
        p = self._p
        return p["W1"], p["b1"], p["W2"], p["b2"]

    def upload_batch(self, X, y=None):
        self._X = np.ascontiguousarray(X, np.float32)
        self._y = None if y is None else np.ascontiguousarray(y, np.int32)

    def upload_chunk(self, X, y):
        self.upload_batch(X, y)
        return self._X.shape[0]

    def forward(self, batch=None):
        b = batch or self._X.shape[0]
        _, _, logits = _ref.forward(self._p, self._X[:b])
        return logits

    def train_step(self, lr: float, batch=None) -> float:
        b = batch or self._X.shape[0]
        loss, g = _ref.loss_and_grads(self._p, self._X[:b], self._y[:b])
        self._p = _ref.sgd_step(self._p, g, lr)
        self._loss = float(loss)
        return self._loss

    def train_steps(self, n_steps: int, batch: int, lr: float) -> float:
        total = 0.0
        for s in range(n_steps):
            sl = slice(s * batch, (s + 1) * batch)
            loss, g = _ref.loss_and_grads(self._p, self._X[sl], self._y[sl])
            self._p = _ref.sgd_step(self._p, g, lr)
            total += float(loss)
        self._loss = total / n_steps
        return self._loss

    def last_loss(self) -> float:
        return self._loss


def make_mlp(in_dim, hidden, out_dim, max_batch, chunk_steps=1, device="auto",
             batch=None):
    """Build the GPU or CPU MLP trainer.

    device: "auto" picks with `prefer_gpu`, "gpu" and "cpu" force an arm. `batch` is
    the batch size the router should predict for (defaults to `max_batch`); it only
    affects "auto". Returns (trainer, chosen_device).
    """
    if device not in DEVICES:
        raise ValueError(f"unknown device {device!r}; expected one of {DEVICES}")

    if device == "auto":
        b = batch or max_batch
        device = "gpu" if prefer_gpu(in_dim, hidden, out_dim, b, chunk_steps) else "cpu"

    if device == "cpu":
        return CpuMlp(in_dim, hidden, out_dim, max_batch, chunk_steps), "cpu"

    from ._capi import Mlp as _GpuMlp
    return _GpuMlp(in_dim, hidden, out_dim, max_batch, chunk_steps), "gpu"
