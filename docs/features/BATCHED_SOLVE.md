# Batched tiny-system solve

**Since v0.5.0** · `kernels/batched_solve.metal` · `src/ops/batched_solve.mm` ·
`tests/cpp/batched_solve_test.cpp` · `benchmarks/bench_batched_solve.py`

## The problem

Scientific code is full of workloads that are millions of *independent* 3×3 or 6×6
solves: per-element constitutive updates, per-particle frames, per-voxel fits, small
Kalman updates. LAPACK is built for one large matrix, so a batch of them becomes
millions of library calls whose per-call overhead dwarfs the ~50 FLOPs of actual work.
`numpy.linalg.solve` does batch internally, but still moves each system through
generic machinery.

One GPU thread per system inverts the problem: the batch axis *is* the parallelism,
and each matrix never leaves registers.

## How it works

LU with partial pivoting, one thread per system, `A` and `rhs` held in registers for
the whole factorisation. `n` is a template parameter with one kernel instantiated per
size (2–8).

**Every index into the register arrays must be compile-time constant.** A single
dynamic index — a pivot row chosen at runtime, say — forces the whole array to
thread-local memory and costs roughly 10×. This is why the row interchange is written
as a sequence of *conditional* swaps with static indices:

```metal
for (uint i = k + 1; i < N; ++i) {          // unrolled
    const bool sw = (i == piv);
    for (uint j = 0; j < N; ++j) {          // unrolled
        const float t = a[k * N + j];
        if (sw) { a[k * N + j] = a[i * N + j]; a[i * N + j] = t; }
    }
}
```

rather than the natural `a[k*N+j] = a[piv*N+j]`.

### Why only n ≤ 8

`MPSMatrixDecompositionLU` and `MPSMatrixSolveLU` inherit `batchStart`/`batchSize`, so
**Apple already ships a batched LU** — and measured, it beats a thread-per-system
kernel from about n=12 (2.4× at n=12, 54× at n=32). Hand-writing those sizes would be
reimplementing something slower. `batched_solve` raises above n=8 rather than
silently doing something bad.

### `pivmin` is a singularity flag, not a condition estimate

Each system reports `min|U_kk| / max|U_kk|`, or **exactly 0.0** when it is singular or
its input held NaN/Inf.

It is deliberately *not* built from `min()`/`max()`: those are `fmin`/`fmax`, which
**drop** a NaN operand, so a NaN input would sail through reporting the healthiest
possible score while its answer is entirely NaN. `!(u > 0.0f)` is true for NaN, zero
and negative alike, and is the only form that catches all three.

Threshold it for *"did this fail"*, never for *"is this accurate"*. A well-scaled but
ill-conditioned system — every `|U_kk|` equal, cond ~1e5 — scores a perfect 1.0 while
its answer is wrong in the fourth digit.

## Load-bearing invariants

> **Diagonally-dominant test matrices perform no row interchanges.** A swap bug would
> pass a "random matrices" sweep completely. `BatchedSolveExercisesPivoting` uses
> anti-diagonal systems where `a[0][0]` is zero, so every size *requires* a swap.

> **Assertions are on the scaled residual, not the forward error.** Backward-stable LU
> guarantees a small residual regardless of conditioning; forward error over random
> Gaussian draws is heavy-tailed and would make the suite flaky. (Measured: forward
> error hits 0.08 on random Gaussians at n=6 while the residual stays at 1e-5 — that
> is conditioning, not a bug.)

- `BatchedSolveMatchesCpuPath` cross-checks GPU against the scalar CPU implementation
  of the *same* algorithm — this is what catches a register spill silently changing
  what the GPU computes.
- `BatchedSolveFlagsSingularAndNaN` asserts `pivmin == 0.0f` exactly for the zero
  matrix, a NaN input and an inf input.

## Measurements

`.venv/bin/python benchmarks/bench_batched_solve.py` — M4 Pro, batch = 65,536,
systems/second. Two GPU columns, because they answer different questions: **host**
includes the round trip (what `jaxmetal.batched_solve(numpy_array)` costs), **resident**
runs the same kernel on buffers already on the device.

| n | GPU host | GPU resident | scalar C loop | `np.linalg.solve` | resident vs C | residual |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1.97e8 | 3.59e8 | 1.11e8 | 7.43e6 | 3.2× | 1.3e-07 |
| 3 | 1.33e8 | 2.96e8 | 5.64e7 | 5.29e6 | 5.2× | 1.6e-07 |
| 4 | 8.72e7 | 2.00e8 | 4.38e7 | 3.96e6 | 4.6× | 1.6e-07 |
| 6 | 4.34e7 | 1.23e8 | 1.94e7 | 2.53e6 | 6.3× | 1.7e-07 |
| 8 | 2.51e7 | 5.17e7 | 1.05e7 | 1.91e6 | 4.9× | 2.1e-07 |

**Read against the C loop.** Looping `numpy.linalg.solve` per system measures numpy's
per-call overhead rather than linear algebra and would have shown two orders of
magnitude; even numpy's genuinely-batched call is 15–24× off, which is mostly generic
machinery. A plain scalar C loop is the honest bar.

**Residency is worth ~2.4×** — these kernels do well under one FLOP per byte moved, so
the host copies cost about as much as the solve.

### `spd=True`: Cholesky for symmetric positive definite systems

Strictly less work than LU — `n³/6` against `n³/3` — and, more importantly here, **no
pivot search and no row interchange at all**. The conditional-swap sequence is a large
part of the LU kernel's instruction stream, so removing it gains more than the FLOP
count suggests. Resident, batch = 65,536:

| n | 2 | 3 | 4 | 6 | 8 |
|---|---:|---:|---:|---:|---:|
| Cholesky vs LU | 0.97× | 1.27× | 1.69× | **2.69×** | **2.63×** |

Residual improves too (4.0e-08 against LU's ~1.7e-07). At n=2 there is nothing to save
and the extra triangular masking costs 3%.

Only the **lower triangle** of `A` is read, so the upper may hold anything —
`BatchedCholeskyIgnoresUpperTriangle` fills it with `1e30` and asserts the result is
bit-identical, which a symmetric test matrix could never detect. `pivmin` keeps the
same contract: exactly `0.0` for a non-SPD, singular, or NaN input.

### Crossover (n = 6)

| batch | GPU host | GPU resident | C loop | numpy | host wins | resident wins |
|---:|---:|---:|---:|---:|:--|:--|
| 1,024 | 0.283 ms | 0.131 ms | 0.053 ms | 0.382 ms | no | no |
| 4,096 | 0.218 ms | 0.163 ms | 0.205 ms | 1.524 ms | no | **yes** |
| 16,384 | 0.366 ms | 0.243 ms | 0.807 ms | 6.329 ms | **yes** | **yes** |
| 262,144 | 5.361 ms | **0.915 ms** | 13.121 ms | 104.513 ms | **yes** | **yes** |

Residency moves the crossover from ~16,000 systems to ~4,000, and at batch 262,144 it
is **14.3× the C loop**. Below the crossover the command-buffer round trip costs more
than the entire solve.

## Limits and things left out

- **`nrhs` is 1.** Multiple right-hand sides would need tail masking on both load and
  store; getting that wrong writes past the end of the output buffer, which Metal does
  not bounds-check. Not worth the hazard until a caller needs it.
- **`batched_solve_resident` takes `DeviceBuffer`s directly**; the numpy-facing
  `batched_solve` copies in and out and is ~2.4× slower. Use the resident path when
  data is already on the device or reused across calls.
- **`spd=True` selects Cholesky** for symmetric positive definite systems.
- **f32 only.** Apple GPUs have no f64. Forward error tracks `cond · eps_f32`, so
  systems with cond ≳ 1e4 want the CPU regardless of throughput.
- **No `auto` router.** [DEVICE_ROUTING.md](DEVICE_ROUTING.md) has the machinery and
  the crossover table above is the input it needs, but nothing calls this from a
  size-varying path yet.
