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
systems/second. The GPU column **includes the host round trip**, which is what a
caller passing numpy arrays actually pays:

| n | GPU | scalar C loop | `np.linalg.solve` | vs C | vs numpy | residual |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1.61e8 | 1.02e8 | 6.97e6 | 1.6× | 23.1× | 1.3e-07 |
| 3 | 1.21e8 | 5.31e7 | 4.97e6 | 2.3× | 24.2× | 1.6e-07 |
| 4 | 8.58e7 | 4.12e7 | 3.71e6 | 2.1× | 23.1× | 1.6e-07 |
| 6 | 4.78e7 | 1.89e7 | 2.36e6 | 2.5× | 20.3× | 1.7e-07 |
| 8 | 2.77e7 | 1.01e7 | 1.82e6 | 2.8× | 15.3× | 2.1e-07 |

**Read the `vs C` column.** Looping `numpy.linalg.solve` per system measures numpy's
per-call overhead rather than linear algebra and would have shown two orders of
magnitude; even numpy's genuinely-batched call is 15–24× off, which is mostly generic
machinery. A plain scalar C loop is the honest bar, and against it this is **2–3×**.

### Crossover (n = 6)

| batch | GPU | C loop | numpy | winner |
|---:|---:|---:|---:|:--|
| 1,024 | 0.221 ms | 0.052 ms | 0.421 ms | CPU |
| 4,096 | 0.331 ms | 0.203 ms | 1.619 ms | CPU |
| 16,384 | 0.509 ms | 0.886 ms | 6.608 ms | **GPU** |
| 262,144 | 6.268 ms | 14.104 ms | 111.844 ms | **GPU** |

Below ~8,000 systems the command-buffer round trip costs more than the entire solve.

## Limits and things left out

- **`nrhs` is 1.** Multiple right-hand sides would need tail masking on both load and
  store; getting that wrong writes past the end of the output buffer, which Metal does
  not bounds-check. Not worth the hazard until a caller needs it.
- **Host operands only.** These kernels do well under one FLOP per byte moved, so
  residency matters more here than for matmul — the transfer costs about as much as
  the solve. A resident entry point would move the crossover down sharply and is the
  obvious next step.
- **No batched Cholesky.** The SPD case would save roughly half the work; it is a
  mechanical variant of the same kernel.
- **f32 only.** Apple GPUs have no f64. Forward error tracks `cond · eps_f32`, so
  systems with cond ≳ 1e4 want the CPU regardless of throughput.
- **No `auto` router.** [DEVICE_ROUTING.md](DEVICE_ROUTING.md) has the machinery and
  the crossover table above is the input it needs, but nothing calls this from a
  size-varying path yet.
