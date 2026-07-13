"""End-to-end: train a resident-GPU MLP on MNIST, benchmark vs JAX-CPU, and gate
the GPU forward/backward against the NumPy golden reference.

Usage:
  .venv/bin/python examples/train_mnist.py                      # gate + benchmark + train
  .venv/bin/python examples/train_mnist.py --epochs 40 --batch 1024 --hidden 1024 --lr 0.5
  .venv/bin/python examples/train_mnist.py --gate-only          # correctness gate only
  .venv/bin/python examples/train_mnist.py --bench-only         # benchmark only
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import jaxmetal
from jaxmetal.data import load_mnist
import jaxmetal.reference as ref

IN_DIM, OUT_DIM = 784, 10
DEFAULTS = dict(batch=1024, hidden=1024, lr=0.5, epochs=40, seed=0)


def build_model(params, hidden, max_batch):
    m = jaxmetal.Mlp(IN_DIM, hidden, OUT_DIM, max_batch)
    m.set_params(params["W1"], params["b1"], params["W2"], params["b2"])
    return m


def accuracy(model, X, y, batch):
    correct = 0
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]; yb = y[i:i + batch]
        model.upload_batch(xb)
        logits = model.forward(len(xb))
        correct += int((logits.argmax(1) == yb).sum())
    return correct / len(X)


# ---- correctness gate: GPU loss/grads vs numpy reference on one fixed minibatch ----
def gate(params, X, y, hidden, lr):
    m = build_model(params, hidden, len(X))
    m.upload_batch(X, y)
    ref_loss, ref_grads = ref.loss_and_grads(params, X, y)
    theta0 = {k: v.copy() for k, v in params.items()}
    gpu_loss = m.train_step(lr, len(X))
    W1, b1, W2, b2 = m.get_params()
    gpu_grads = {
        "W1": (theta0["W1"] - W1.reshape(theta0["W1"].shape)) / lr,
        "b1": (theta0["b1"] - b1) / lr,
        "W2": (theta0["W2"] - W2.reshape(theta0["W2"].shape)) / lr,
        "b2": (theta0["b2"] - b2) / lr,
    }
    tol = {"W1": (2e-4, 1e-3), "b1": (1e-4, 1e-3),
           "W2": (2e-4, 1e-3), "b2": (1e-4, 1e-3)}
    ok = abs(gpu_loss - float(ref_loss)) < 1e-3
    print(f"[gate] loss gpu={gpu_loss:.6f} ref={float(ref_loss):.6f} d={abs(gpu_loss-float(ref_loss)):.2e}")
    for k in ("W1", "b1", "W2", "b2"):
        atol, rtol = tol[k]
        d = float(np.max(np.abs(gpu_grads[k] - ref_grads[k])))
        thr = atol + rtol * float(np.max(np.abs(ref_grads[k])))
        pk = d <= thr
        ok &= pk
        print(f"[gate] grad {k:>2} max|d|={d:.2e} thr={thr:.2e} {'OK' if pk else 'FAIL'}")
    print("[gate]", "PASS" if ok else "FAIL")
    return ok


# ---- JAX-CPU reference step (the baseline to beat) ----
def make_jax_step(hidden):
    import jax, jax.numpy as jnp
    jax.config.update("jax_platform_name", "cpu")

    def loss_fn(p, X, y):
        h = jnp.maximum(X @ p["W1"] + p["b1"], 0.0)
        logits = h @ p["W2"] + p["b2"]
        logp = logits - jax.scipy.special.logsumexp(logits, axis=1, keepdims=True)
        return -jnp.mean(logp[jnp.arange(y.shape[0]), y])

    @jax.jit
    def step(p, X, y, lr):
        loss, grads = jax.value_and_grad(loss_fn)(p, X, y)
        p = {k: p[k] - lr * grads[k] for k in p}
        return p, loss
    return jax, step


def benchmark(params, X, y, hidden, batch, lr, iters=50):
    xb = np.ascontiguousarray(X[:batch]); yb = np.ascontiguousarray(y[:batch])
    flops = 6.0 * batch * (IN_DIM * hidden + hidden * OUT_DIM)

    # GPU resident (params + activations stay on-device; only the loss is read back)
    m = build_model(params, hidden, batch)
    m.upload_batch(xb, yb)
    m.train_step(lr)  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        m.train_step(lr)
    gpu_s = (time.perf_counter() - t0) / iters

    # JAX CPU
    import jax.numpy as jnp
    jaxmod, step = make_jax_step(hidden)
    p = {k: jnp.asarray(v) for k, v in params.items()}
    Xj, yj = jnp.asarray(xb), jnp.asarray(yb.astype(np.int32))
    p, l = step(p, Xj, yj, lr); jaxmod.block_until_ready(l)  # warm + compile
    t0 = time.perf_counter()
    for _ in range(iters):
        p, l = step(p, Xj, yj, lr)
    jaxmod.block_until_ready(l)
    cpu_s = (time.perf_counter() - t0) / iters

    print(f"\n[bench] batch={batch} hidden={hidden}  step FLOPs={flops/1e9:.2f} G")
    print(f"[bench] GPU-resident : {gpu_s*1e3:8.3f} ms  {flops/gpu_s/1e9:7.1f} GFLOP/s")
    print(f"[bench] JAX-CPU jit  : {cpu_s*1e3:8.3f} ms  {flops/cpu_s/1e9:7.1f} GFLOP/s")
    print(f"[bench] speedup GPU/CPU = {cpu_s/gpu_s:.2f}x  "
          f"({'GPU FASTER' if cpu_s > gpu_s else 'CPU faster'})")
    return cpu_s / gpu_s


def train(params, Xtr, ytr, Xte, yte, hidden, batch, lr, epochs, seed):
    m = build_model(params, hidden, batch)
    rng = np.random.default_rng(seed)
    n = len(Xtr)
    best = 0.0
    for ep in range(epochs):
        perm = rng.permutation(n)
        running = 0.0; steps = 0
        for i in range(0, n - batch + 1, batch):  # drop last partial -> fixed matmul shape
            idx = perm[i:i + batch]
            m.upload_batch(Xtr[idx], ytr[idx])
            running += m.train_step(lr); steps += 1
        acc = accuracy(m, Xte, yte, batch)
        best = max(best, acc)
        print(f"epoch {ep:2d}  train_loss={running/steps:.4f}  test_acc={acc*100:.2f}%  best={best*100:.2f}%")
    final = accuracy(m, Xte, yte, batch)
    print(f"\nFINAL test accuracy: {final*100:.2f}%  best={best*100:.2f}%  "
          f"({'PASS >=97%' if best >= 0.97 else 'BELOW TARGET'})")
    return final, best


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--bench-only", action="store_true")
    a = ap.parse_args()

    print("device:", jaxmetal.device_name())
    Xtr, ytr, Xte, yte = load_mnist()
    params = ref.init_params(a.seed, IN_DIM, a.hidden, OUT_DIM)  # shared golden init

    if a.gate_only:
        gate(params, Xtr[:a.batch], ytr[:a.batch], a.hidden, a.lr); return
    if a.bench_only:
        benchmark(params, Xtr, ytr, a.hidden, a.batch, a.lr); return

    print("\n== correctness gate ==")
    if not gate(params, Xtr[:a.batch], ytr[:a.batch], a.hidden, a.lr):
        print("gate failed; aborting"); sys.exit(1)
    print("\n== benchmark ==")
    benchmark(params, Xtr, ytr, a.hidden, a.batch, a.lr)
    print("\n== training ==")
    train(params, Xtr, ytr, Xte, yte, a.hidden, a.batch, a.lr, a.epochs, a.seed)


if __name__ == "__main__":
    main()
