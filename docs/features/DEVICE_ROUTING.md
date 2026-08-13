# `device="auto"` routing

**Since v0.2.0 (MLP), v0.7.0 (scientific ops)** · `python/jaxmetal/mlp.py` ·
`python/jaxmetal/routing.py` · `tests/python/test_mlp_auto.py` ·
`tests/python/test_routing.py`

## The problem

The GPU is not faster at every size. After [chunked training](CHUNKED_TRAINING.md)
the crossover sits near batch 60 at `hidden=128`, and moves with `hidden` — at
`hidden=1024` the GPU already wins at batch 32. Handing users a GPU trainer that is
3× *slower* than NumPy on their problem size is worse than not having one.

So `jaxmetal.Mlp` routes:

```python
m = jaxmetal.Mlp(784, 128, 10, max_batch=512)   # device="auto" by default
m.device                                         # -> "gpu" or "cpu"

jaxmetal.Mlp(..., device="gpu")                  # force an arm
JAXMETAL_MLP_DEVICE=cpu python ...               # or force from the environment
```

## The CPU arm introduces no new numerics

It delegates to `jaxmetal.reference` — the pure-NumPy golden reference that is
already gate-verified against `jax.grad` to ~1e-8. Its matmuls are plain `@`, so they
run on Accelerate/BLAS: this is a genuinely fast CPU path, not a teaching
implementation. `test_cpu_arm_matches_gpu_arm` asserts both arms agree on the loss
and all four parameter arrays after a step.

Writing a second CPU trainer would have meant a second implementation of the
gradients to keep correct. There was already a verified one in the repo.

## The model is a crossover, not two cost lines

The obvious approach — fit `time = fixed + marginal × work` per arm and compare — was
tried and **does not work here**.

Across the ~500× range of `work` the sweep covers, a per-arm linear fit is dominated
by the large end. The least-squares result was:

```
kGpuFixed = 0.1597    kCpuFixed = 0.2154
```

`kCpuFixed > kGpuFixed` inverts reality — the CPU is the arm with the *small* fixed
cost — and the model that follows predicts "GPU always wins", which is wrong at
`hidden=128, batch=32` where the CPU is 1.3× faster.

What *is* well behaved is the **speedup ratio**, which is close to a power law in
`work` near the boundary:

```
log(t_cpu / t_gpu) = kSpeedupA + kSpeedupB · log(work)
work = batch × (in_dim·hidden + hidden·out_dim)
```

The decision then collapses to a single number — the `work` at which that crosses 1.
Fitted on an M4 Pro: `kSpeedupA = -5.234`, `kSpeedupB = 0.335`, giving a crossover at
`work ≈ 6.2e6` (batch ≈ 60 at `hidden=128`).

`chunk_steps` shifts it: a small chunk leaves the ~140 µs command-buffer round trip
unamortised, inflating the GPU's fixed cost and pushing the crossover right.
`crossover_work()` applies that correction, so `chunk_steps=1` lands at batch ~138
rather than ~60.

## Why not the matmul router's FLOP threshold

`metal_matmul_auto_f32` uses `kAutoGpuFlopThreshold` — a single total-FLOP number.
That works for one GEMM, whose cost is nearly all marginal. It **cannot express this
crossover**: identical FLOP counts fall on opposite sides depending on how they split
into batch versus hidden. `hidden=128, batch=2048` and `hidden=1024, batch=256` are
within ~4% on FLOPs and land at 2.79× and ~2.4× respectively — but `hidden=128,
batch=32` and `hidden=1024, batch=32` differ by 8× in FLOPs and sit on *opposite*
sides of the boundary.

## Calibration

The constants are M4 Pro measurements. Any other Mac moves them:

```bash
.venv/bin/python examples/train_mnist.py --calibrate
```

sweeps both arms over `hidden × batch`, refits, and prints constants to paste into
`python/jaxmetal/mlp.py`. Sample output:

```
 hidden  batch    GPU ms    CPU ms  speedup
    128     32     0.124     0.097     0.79x
    128    128     0.137     0.179     1.31x
    128    512     0.206     0.446     2.17x
   1024     32     0.200     0.319     1.60x
   1024   2048     2.023     6.400     3.16x

kSpeedupA = -5.234
kSpeedupB = 0.335
crossover work=6.234e+06 (batch 61 at hidden=128, chunk_steps=128)
```

## Load-bearing invariants

> **A cost model that is never checked against a clock is a hardcoded guess.**
> `test_router_picks_the_faster_arm` times both arms at three points and asserts the
> router's choice is never more than 15% slower than the better one. The 15% band
> exists because near the crossover the arms are within noise of each other, where
> choosing "wrong" costs nothing.

- `test_router_is_monotonic_and_forceable` asserts the decision never flips back to
  CPU as `work` grows, that both outcomes are reachable, that a smaller
  `chunk_steps` moves the crossover right, and that `JAXMETAL_MLP_DEVICE` overrides.
- **`tests/python/test_mlp_gate.py` pins `device="gpu"` explicitly.** That gate exists
  to check the Metal path against the NumPy reference; under `device="auto"` at its
  small batch size it would be handed the reference itself and compare it to itself.

## The scientific ops

`batched_solve`, `cholesky` and `df64_binop` all take `device="auto" | "gpu" | "cpu"`
too, with thresholds in `python/jaxmetal/routing.py`. `JAXMETAL_DEVICE=gpu|cpu`
overrides everything.

> **Residency moves the crossover, and it moves it a lot.** These kernels do well under
> one FLOP per byte moved, so copying host arrays in and out can cost more than the
> compute. The routers take `resident` explicitly rather than assuming — a host-operand
> call and a resident call are genuinely different operations with different answers.

| op | CPU arm | host operands | resident |
|---|---|---|---|
| `batched_solve` (n=6) | scalar C loop | GPU from ~5,000 systems | GPU from ~2,500 |
| `df64_binop` | numpy `float64` | **CPU always** — copies are 41× the kernel | GPU from ~4M elements |
| `cholesky` | `np.linalg.cholesky` | GPU from N≈1300 | — |

The df64 row is the sharpest illustration: the *same kernel* is 1.93× faster than the
CPU when resident and ~0.04× through the copy path, so a router that ignored residency
would be wrong by ~50× in one direction or the other.

**The `cholesky` threshold is set against numpy, and numpy is not the fastest CPU
option.** A direct `spotrf` on a Fortran-ordered array is ~4× quicker (it skips the
reorder numpy does for a column-major routine), and against *that* the GPU only reaches
parity at N=4096. The threshold routes against what a Python caller actually invokes;
if your CPU path is hand-tuned LAPACK, pass `device="cpu"`.

### Gates

`tests/python/test_routing.py` asserts four things, because a cost model that is never
checked against a clock is a hardcoded guess:

- **Both arms agree numerically** — routing is a performance decision, never a
  numerical one.
- **The router picks the faster arm**, timed either side of each crossover, within a
  35% band (near a crossover the arms are within noise, so being "wrong" costs nothing).
- **Monotonicity** — a bigger problem never flips the decision back to the CPU.
- **Residency only lowers thresholds** — copies can only cost time, so the resident
  crossover must sit at or below the host one.

## Limits and things left out

- **The model is fitted, not derived.** It interpolates the measured grid; well
  outside it (very large `hidden`, tiny `in_dim`) it is extrapolating.
- **The router lives in Python, not the C ABI.** The original plan put it beside
  `auto_prefer_gpu` in `metal_capi.mm`, but the CPU arm is NumPy and there is no C++
  consumer, so a C-side policy would have had no callers.
- **It routes at construction, on one batch size.** A trainer that varied batch size
  across its lifetime would want per-call routing; nothing needs that yet.
- **`matmul` still has its own separate FLOP-threshold router** and is not covered by
  `routing.py`.
- **`reduce_sum` is not routed.** Its argument is accuracy rather than speed —
  compensated summation is 127× more accurate at no cost — so "which is faster" is the
  wrong question for it.
- **Thresholds are constants, not a fitted model** (unlike the MLP's). Each op has one
  or two measured crossover points rather than a swept grid; that is enough to place a
  threshold but not to interpolate.
