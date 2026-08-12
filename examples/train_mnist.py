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


# Steps encoded into one command buffer. The ~0.14 ms driver round trip is paid
# once per chunk, so this is the single biggest lever on per-step cost at small and
# moderate batch sizes; 128 is where the curve flattens on an M4 Pro.
CHUNK_STEPS = 128


def build_model(params, hidden, max_batch, device="gpu", chunk_steps=1):
    m = jaxmetal.Mlp(IN_DIM, hidden, OUT_DIM, max_batch, chunk_steps=chunk_steps,
                     device=device, batch=max_batch)
    m.set_params(params["W1"], params["b1"], params["W2"], params["b2"])
    return m


def calibrate(params, X, y, hiddens=(128, 1024), batches=(32, 128, 512, 2048)):
    """Sweep both arms and re-fit the jaxmetal.mlp cost-model constants.

    The constants in jaxmetal/mlp.py are M4 Pro measurements; any other Mac moves
    them. Prints a least-squares fit of  time_ms = fixed + marginal * work  for each
    arm, plus the batch size where they cross, so the numbers can be pasted back.
    """
    import jax.numpy as jnp
    from jaxmetal.mlp import step_work

    rows = []
    print(f"{'hidden':>7} {'batch':>6} {'GPU ms':>9} {'CPU ms':>9} {'speedup':>8}")
    for hidden in hiddens:
        p = ref.init_params(0, IN_DIM, hidden, OUT_DIM)
        jaxmod, step = make_jax_step(hidden)
        for batch in batches:
            k = min(CHUNK_STEPS, len(X) // batch)   # dataset must cover the chunk
            xb = np.ascontiguousarray(X[:k * batch])
            yb = np.ascontiguousarray(y[:k * batch].astype(np.int32))
            m = build_model(p, hidden, batch, device="gpu", chunk_steps=k)
            m.upload_chunk(xb, yb)
            m.train_steps(k, batch, 0.01)                      # warm + build plans
            t0 = time.perf_counter()
            for _ in range(3):
                m.train_steps(k, batch, 0.01)
            gpu_ms = (time.perf_counter() - t0) / (3 * k) * 1e3
            del m

            jp = {kk: jnp.asarray(v) for kk, v in p.items()}
            Xj, yj = jnp.asarray(xb[:batch]), jnp.asarray(yb[:batch])
            jp, l = step(jp, Xj, yj, 0.1); jaxmod.block_until_ready(l)
            t0 = time.perf_counter()
            for _ in range(30):
                jp, l = step(jp, Xj, yj, 0.1)
            jaxmod.block_until_ready(l)
            cpu_ms = (time.perf_counter() - t0) / 30 * 1e3

            rows.append((step_work(IN_DIM, hidden, OUT_DIM, batch), gpu_ms, cpu_ms))
            print(f"{hidden:>7} {batch:>6} {gpu_ms:>8.3f} {cpu_ms:>8.3f} "
                  f"{cpu_ms / gpu_ms:>7.2f}x")

    # Fit the SPEEDUP ratio, not each arm's cost line: across this work range a
    # per-arm linear fit is dominated by the large end and mispredicts the crossover
    # (see the module docstring in jaxmetal/mlp.py).
    w = np.log(np.array([r[0] for r in rows]))
    sp = np.log(np.array([r[2] / r[1] for r in rows]))
    near = np.abs(sp) < np.log(2.5)          # weight the region around speedup == 1
    if near.sum() < 2:
        near = np.ones_like(sp, dtype=bool)
    b, a = np.polyfit(w[near], sp[near], 1)
    print("\n[calibrate] paste into python/jaxmetal/mlp.py:")
    print(f"kSpeedupA = {a:.3f}\nkSpeedupB = {b:.3f}")
    cross = np.exp(-a / b)
    per_sample = IN_DIM * hiddens[0] + hiddens[0] * OUT_DIM
    print(f"[calibrate] crossover work={cross:.3e} "
          f"(batch {cross / per_sample:.0f} at hidden={hiddens[0]}, "
          f"chunk_steps={CHUNK_STEPS})")


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

    # GPU resident, chunked: params + activations stay on-device and CHUNK_STEPS
    # steps share one command buffer, so the driver round trip is amortised. Timing
    # a lone train_step() instead measures mostly that round trip, not the model.
    k = min(CHUNK_STEPS, len(X) // batch)   # dataset must cover the whole chunk
    xc = np.ascontiguousarray(X[:k * batch]); yc = np.ascontiguousarray(y[:k * batch])
    m = build_model(params, hidden, batch, device="gpu", chunk_steps=k)
    m.upload_chunk(xc, yc)
    m.train_steps(k, batch, lr)  # warmup + build the per-slot plans
    t0 = time.perf_counter()
    reps = max(1, iters // k)
    for _ in range(reps):
        m.train_steps(k, batch, lr)
    gpu_s = (time.perf_counter() - t0) / (reps * k)

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


def train(params, Xtr, ytr, Xte, yte, hidden, batch, lr, epochs, seed, device="auto"):
    n = len(Xtr)
    steps_per_epoch = n // batch          # drop last partial -> fixed matmul shape
    chunk = max(1, min(CHUNK_STEPS, steps_per_epoch))
    m = build_model(params, hidden, batch, device=device, chunk_steps=chunk)
    print(f"[train] device={m.device} batch={batch} hidden={hidden} "
          f"chunk_steps={chunk} ({steps_per_epoch} steps/epoch)")
    rng = np.random.default_rng(seed)
    best = 0.0
    for ep in range(epochs):
        perm = rng.permutation(n)
        running = 0.0; steps = 0
        # Submit `chunk` shuffled minibatches at a time: one command buffer, one host
        # sync. The shuffle is still per-epoch, so this is the same SGD as before.
        for c in range(0, steps_per_epoch - chunk + 1, chunk):
            idx = perm[c * batch:(c + chunk) * batch]
            m.upload_chunk(Xtr[idx], ytr[idx])
            running += m.train_steps(chunk, batch, lr) * chunk
            steps += chunk
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
    ap.add_argument("--device", default="auto", choices=jaxmetal.MLP_DEVICES,
                    help="MLP backend: auto (cost model), gpu, or cpu")
    ap.add_argument("--calibrate", action="store_true",
                    help="sweep GPU vs CPU and re-fit the device=auto cost model")
    a = ap.parse_args()

    print("device:", jaxmetal.device_name())
    Xtr, ytr, Xte, yte = load_mnist()
    params = ref.init_params(a.seed, IN_DIM, a.hidden, OUT_DIM)  # shared golden init

    if a.gate_only:
        gate(params, Xtr[:a.batch], ytr[:a.batch], a.hidden, a.lr); return
    if a.calibrate:
        calibrate(params, Xtr, ytr); return
    if a.bench_only:
        benchmark(params, Xtr, ytr, a.hidden, a.batch, a.lr); return

    print("\n== correctness gate ==")
    if not gate(params, Xtr[:a.batch], ytr[:a.batch], a.hidden, a.lr):
        print("gate failed; aborting"); sys.exit(1)
    print("\n== benchmark ==")
    benchmark(params, Xtr, ytr, a.hidden, a.batch, a.lr)
    print("\n== training ==")
    train(params, Xtr, ytr, Xte, yte, a.hidden, a.batch, a.lr, a.epochs, a.seed,
          device=a.device)


if __name__ == "__main__":
    main()
