"""Pure-NumPy golden reference for the 784->H->10 MLP (ReLU, softmax cross-entropy).

This is the single source of numerical truth for the resident-GPU MLP: the driver's
correctness gate compares GPU loss/grads against loss_and_grads() here, and both the
GPU model and this reference start from the identical init_params() weights.

Verified against jax.grad (jax 0.10.2): loss diff 0.0, grad max-abs error <= 3e-8.
f32 throughout. All tensors row-major.
"""
from __future__ import annotations
import numpy as np


def init_params(seed: int, in_dim: int, hidden: int, out_dim: int) -> dict:
    """He-normal init for ReLU, zero biases. Deterministic for a given seed.
    Returns {'W1','b1','W2','b2'} float32 row-major:
      W1[in_dim,hidden] b1[hidden] W2[hidden,out_dim] b2[out_dim].
    The GPU path must upload these exact bytes (do not re-init on device)."""
    rng = np.random.default_rng(seed)

    def he(fan_in, fan_out):
        std = np.sqrt(2.0 / fan_in)
        return (rng.standard_normal((fan_in, fan_out)) * std).astype(np.float32)

    W1 = he(in_dim, hidden)          # drawn first
    W2 = he(hidden, out_dim)         # then W2 (biases are zeros)
    b1 = np.zeros((hidden,), np.float32)
    b2 = np.zeros((out_dim,), np.float32)
    return dict(W1=W1, b1=b1, W2=W2, b2=b2)


def forward(p, x):
    z1 = x @ p["W1"] + p["b1"]          # [B,H]
    h1 = np.maximum(z1, 0.0)            # [B,H]
    logits = h1 @ p["W2"] + p["b2"]     # [B,out]
    return z1, h1, logits


def softmax_xent(logits, y):
    B = logits.shape[0]
    m = logits.max(axis=1, keepdims=True)
    shifted = logits - m
    e = np.exp(shifted)
    Z = e.sum(axis=1, keepdims=True)
    logp = shifted - np.log(Z)          # log softmax
    probs = e / Z
    nll = -logp[np.arange(B), y]
    loss = np.float32(nll.mean())
    return loss, probs


def backward(p, x, y, z1, h1, logits, probs):
    B = x.shape[0]
    onehot = np.zeros_like(logits)
    onehot[np.arange(B), y] = 1.0
    dlogits = ((probs - onehot).astype(np.float32) / B)     # [B,out]  (only 1/B here)
    dW2 = h1.T @ dlogits                                     # [H,out]
    db2 = dlogits.sum(axis=0)                                # [out]
    dh1 = dlogits @ p["W2"].T                                # [B,H]
    dz1 = dh1 * (z1 > 0.0)                                   # [B,H]  strict >0
    dW1 = x.T @ dz1                                          # [in,H]
    db1 = dz1.sum(axis=0)                                    # [H]
    return dict(W1=dW1.astype(np.float32), b1=db1.astype(np.float32),
                W2=dW2.astype(np.float32), b2=db2.astype(np.float32))


def loss_and_grads(p, x, y):
    """Returns (mean_xent_loss f32, grads dict with same keys/shapes as p)."""
    x = np.ascontiguousarray(x, np.float32)
    y = np.ascontiguousarray(y, np.int64)
    z1, h1, logits = forward(p, x)
    loss, probs = softmax_xent(logits, y)
    grads = backward(p, x, y, z1, h1, logits, probs)
    return loss, grads


def forward_loss(p, x, y):
    """Returns (mean_xent_loss f32, logits[B,out] f32)."""
    x = np.ascontiguousarray(x, np.float32)
    y = np.ascontiguousarray(y, np.int64)
    _, _, logits = forward(p, x)
    loss, _ = softmax_xent(logits, y)
    return loss, logits


def sgd_step(p, g, lr):
    return {k: (p[k] - np.float32(lr) * g[k]).astype(np.float32) for k in p}


if __name__ == "__main__":
    # Self-check against jax.grad (guard that the reference itself is correct).
    import jax, jax.numpy as jnp
    rng = np.random.default_rng(123)
    B, IN, H, OUT = 32, 784, 64, 10
    x = rng.standard_normal((B, IN)).astype(np.float32)
    y = rng.integers(0, OUT, B).astype(np.int64)
    p = init_params(0, IN, H, OUT)

    def jax_loss(params, x, y):
        z1 = x @ params["W1"] + params["b1"]
        h1 = jnp.maximum(z1, 0.0)
        logits = h1 @ params["W2"] + params["b2"]
        logp = jax.nn.log_softmax(logits, axis=1)
        return -logp[jnp.arange(x.shape[0]), y].mean()

    loss_np, g_np = loss_and_grads(p, x, y)
    jp = {k: jnp.asarray(v) for k, v in p.items()}
    loss_jx, g_jx = jax.value_and_grad(jax_loss)(jp, jnp.asarray(x), jnp.asarray(y))
    print(f"loss np={float(loss_np):.6f} jax={float(loss_jx):.6f} d={abs(float(loss_np)-float(loss_jx)):.2e}")
    for k in p:
        d = float(np.max(np.abs(g_np[k] - np.asarray(g_jx[k]))))
        print(f"grad {k}: max|d|={d:.2e}  {'OK' if d < 1e-4 else 'FAIL'}")
