"""Network-free correctness gate: the resident GPU MLP's forward + backward + SGD
matches the NumPy golden reference (which itself matches jax.grad) across seeds,
batch sizes, and hidden widths. Uses synthetic data so it runs anywhere (CI).

    .venv/bin/python tests/python/test_mlp_gate.py      # or: pytest tests/python
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))

import numpy as np
import jaxmetal
import jaxmetal.reference as ref

IN_DIM, OUT_DIM = 784, 10


def _gate(seed: int, batch: int, hidden: int, lr: float = 0.1) -> None:
    params = ref.init_params(seed, IN_DIM, hidden, OUT_DIM)
    rng = np.random.default_rng(seed + 100)
    X = (rng.standard_normal((batch, IN_DIM)) * 0.1).astype(np.float32)
    y = rng.integers(0, OUT_DIM, batch).astype(np.int32)

    m = jaxmetal.Mlp(IN_DIM, hidden, OUT_DIM, batch)
    m.set_params(params["W1"], params["b1"], params["W2"], params["b2"])
    m.upload_batch(X, y)

    ref_loss, ref_g = ref.loss_and_grads(params, X, y)
    theta0 = {k: v.copy() for k, v in params.items()}
    gpu_loss = m.train_step(lr, batch)
    W1, b1, W2, b2 = m.get_params()
    gpu_g = {
        "W1": (theta0["W1"] - W1.reshape(theta0["W1"].shape)) / lr,
        "b1": (theta0["b1"] - b1) / lr,
        "W2": (theta0["W2"] - W2.reshape(theta0["W2"].shape)) / lr,
        "b2": (theta0["b2"] - b2) / lr,
    }
    assert abs(gpu_loss - float(ref_loss)) < 1e-3, f"loss {gpu_loss} vs {ref_loss}"
    for k in ("W1", "b1", "W2", "b2"):
        d = float(np.max(np.abs(gpu_g[k] - ref_g[k])))
        thr = 2e-4 + 1e-3 * float(np.max(np.abs(ref_g[k])))
        assert d <= thr, f"grad {k}: max|d|={d:.2e} > thr={thr:.2e}"


def test_gate():
    for seed in (0, 1, 2):
        for batch, hidden in ((64, 128), (256, 256), (512, 1024)):
            _gate(seed, batch, hidden)


if __name__ == "__main__":
    test_gate()
    print("gate OK — resident GPU MLP matches the NumPy/jax.grad golden reference "
          "across seeds x batch x hidden.")
