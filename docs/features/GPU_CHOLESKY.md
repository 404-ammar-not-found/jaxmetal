# Blocked Cholesky on the GPU

**Since v0.4.0** · `kernels/cholesky.metal` · `src/ops/cholesky.mm` ·
`tests/cpp/cholesky_test.cpp` · `benchmarks/bench_cholesky.py`

## The problem

Dense factorisations are the gap in every Apple-Silicon JAX backend.
[jax-mps](https://github.com/tillahoffmann/jax-mps) routes `cholesky`, `qr`, `svd`,
`eigh` and `triangular_solve` to `mlx::core::Device::cpu` — every decomposition runs
on the CPU even in the GPU backend.

The obvious question is why not call Apple's own GPU implementation, since
`MPSMatrixDecompositionCholesky` exists in the SDK. **Because it is unusable.**
Measured on an M4 Pro:

| N | MPS Cholesky | LAPACK `spotrf` | MPS vs LAPACK |
|---:|---:|---:|---:|
| 512 | 10.0 ms | 0.07 ms | 0.01× |
| 1024 | 36.1 ms | 0.36 ms | 0.01× |
| 2048 | 141.5 ms | 2.63 ms | 0.02× |
| 4096 | 619.7 ms | 45.4 ms | **0.07×** |

620 ms against LAPACK's 45 ms, scaling like one dispatch per column. **jax-mps was
right to punt to the CPU** — that was a correct engineering call, not a missed
opportunity. A hand-written blocked factorisation is the only viable GPU path.

## How it works

Right-looking blocked factorisation. Only the O(NB³) panel work is hand-written; the
O(N³) trailing update is an MPS GEMM, which carries essentially all the FLOPs.

```
for k = 0, NB, 2NB, ... < N:
    nb = min(NB, N-k);  m = N-k-nb
    (1) chol_panel       L11 = chol(A[k:k+nb, k:k+nb])   one threadgroup
    (2) chol_trsm_right  L21 = A21 · L11⁻ᵀ               one thread per row
    (3) MPS GEMM         A22 -= L21 · L21ᵀ               alpha=-1, beta=1
```

**The whole factorisation is ONE command buffer** — one commit, one wait, so the
~140 µs driver round trip is paid once regardless of N rather than once per block
step (64 steps at N=4096). Metal hazard-tracks within a command buffer, so the
panel → trsm → GEMM read-after-write chain needs no explicit barriers. Same lesson as
[CHUNKED_TRAINING.md](CHUNKED_TRAINING.md).

**Why `chol_trsm_right` is hand-written rather than `MPSMatrixSolveTriangular`.**
That API exists, but it is serial in its order: measured **150 ms at order=4096,
nrhs=1**, scaling as O(N²). Our panel solve has order nb=64 with *m independent
rows*, one thread each — the parallelism is m-wide, not order-wide.

**NB=64 is a measured optimum, and the panel phases are the bottleneck** — see
Refuted below. Both `JAXMETAL_CHOL_NB` and `JAXMETAL_CHOL_STRIPS` exist to reproduce
those sweeps.

## Load-bearing invariants

> **`CHOL_TG` in `kernels/cholesky.metal` must equal `kCholTG` in
> `src/ops/cholesky.mm`.** The panel kernel strides its column loop by exactly that
> width, so a mismatch silently skips rows rather than failing loudly. NB is no longer
> a kernel-side constant — it is passed per dispatch in `CholDims.nb`, which is what
> lets `JAXMETAL_CHOL_NB` sweep it.

> **Non-positive-definite must be reported, not silently returned.** The panel kernel
> tests `!(s > 0.0f)` rather than `s <= 0.0f` — the latter is *false* for NaN and
> would let a NaN input through as a successful factorisation. On failure it records
> the first failing column (LAPACK `info`, 1-based) and substitutes a benign pivot so
> the trailing GEMM cannot fill the matrix with NaN and destroy the evidence.
> `CholeskyReportsNonPositiveDefinite` and `CholeskyRejectsNaNInput` cover both.

- **MPS GEMM aliasing between disjoint windows of one buffer is undocumented.** Steps
  (2) and (3) read and write different windows of the same `MTLBuffer`; Apple documents
  in-place behaviour only for the *decomposition* kernels and is silent about GEMM.
  `CholeskyMatchesLapack` is what establishes empirically that this is safe. If it ever
  starts failing at large N, a scratch panel buffer is the fix.
- Test sizes deliberately straddle the block boundary (63, 64, 65, 130, 200): a
  blocking bug confined to the ragged final block survives a power-of-two-only sweep.

## Measurements

`.venv/bin/python benchmarks/bench_cholesky.py` — Apple M4 Pro, f32, `n³/3` FLOPs:

| N | jaxmetal | GF/s | `spotrf` | GF/s | vs spotrf | `np.linalg.cholesky` | vs numpy | residual |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 6.12 ms | 7 | 0.36 ms | 123 | 0.06× | 1.34 ms | 0.22× | 4.3e-07 |
| 1024 | 8.24 ms | 43 | 1.92 ms | 186 | 0.23× | 6.32 ms | 0.77× | 5.4e-07 |
| 2048 | 16.79 ms | 171 | 8.88 ms | 322 | 0.53× | 37.40 ms | 2.23× | 6.8e-07 |
| 4096 | 46.49 ms | 493 | **54.82 ms** | 418 | **1.18×** | 191.79 ms | 4.13× | 8.0e-07 |

Run-to-run variance on this machine is roughly ±15% under sustained benchmarking
(thermal), so treat 1.1–1.2× as "parity, slightly ahead" rather than a precise figure.
The A/B comparisons in Refuted below were each taken within a single run, where that
drift cancels.

**Read the `vs spotrf` column.** It is a bare Accelerate call on a Fortran-ordered
array with no copy — the true CPU floor. The result is *parity at N=4096 and a loss
below it*, not a win.

The `vs numpy` column is 4× more flattering and is the number it would be tempting to
quote, but `np.linalg.cholesky` spends most of its time reordering a C-contiguous
array for a column-major routine. Three different "LAPACK" numbers exist for N=4096 —
numpy 178 ms, scipy `spotrf` with copies 102 ms, direct `spotrf_` 45–51 ms — and only
the last is the honest baseline.

Residual is `max|L·Lᵀ − A| / max|A|`, at 4–8e-07 throughout, i.e. a few f32 eps.

## Where the time actually goes

Measured by phase at N=4096 (`JAXMETAL_CHOL_PHASES` selects `p`anel/`t`rsm/`g`emm;
skipping a phase gives wrong results and exists only for attribution):

| phase | time | share |
|---|---:|---:|
| **TRSM** (`chol_trsm_right`) | ~29 ms | **~65%** |
| panel (`chol_panel`) | ~16 ms | ~35% |
| trailing MPS GEMM | ~0.25 ms | **~0.5%** |

**The trailing GEMM is essentially free.** That is the opposite of the assumption this
feature was designed on — "only the panel is hand-written, the GEMM carries all the
FLOPs" — and it explains every failed optimisation below. The panel *phases* are the
factorisation.

## Refuted: four optimisations that measured worse or flat

All three were predicted to be wins. None are, and together they relocate the
bottleneck.

**1. Approximating SYRK with block-column strips.** `A22` is symmetric, so only its
lower triangle is needed, but MPS has no SYRK and one square GEMM computes both —
half the FLOPs are waste. Issuing the update as P column strips walks the triangle in
a staircase, cutting the computed area from `m²` to `m²(P+1)/2P`: 1.25× waste at P=4
instead of 2.00×. Predicted ~1.6× faster. Measured at N=4096:

| P | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| ms | 46.1 | 46.1 | 48.7 | 57.6 | 71.1 |

Halving the FLOPs does not help because **the trailing GEMM is not compute-bound at
rank 64** — it runs at 502 GFLOP/s at M=2048, against 3297 at rank 256. Narrower
strips land in an even less efficient regime and add an MPS encode each. Reverted to
P=1.

**2. Raising NB.** Rank-256 GEMM is 1.66× more efficient than rank-64, so a larger
block should be a clear win. The panel kernels originally staged the NB×NB block in
threadgroup memory, capping NB at 64 (16 KB against Apple's 32 KB); that staging was
removed so NB could be swept. Measured at N=4096:

| NB | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|
| ms | **46.1** | 64.3 | 110.8 | 227.6 |

**That inversion is the real diagnosis.** If the trailing GEMM dominated, bigger NB
would help. It does not, because `chol_panel` is one threadgroup doing O(NB³) work and
`chol_trsm_right` is one thread per row doing O(NB²) *sequential* work — both grow
faster than the GEMM saving. The bottleneck is the panel phases, not the update, and
not the SYRK waste.

**3. Breaking the TRSM's dependent FMA chain.** `s -= r[p] * L11[j][p]` is a chain up
to 64 long, and under safe math the compiler may not reassociate it into independent
partial sums — so a hand-split four-way accumulator should expose ILP. Measured: TRSM
29.13 ms → 29.47 ms. **No improvement**, so chain latency is not what binds it either.
Reverted rather than kept as unearned complexity.

**4. `MPSMatrixSolveTriangular` for the panel solve.** This API was first dismissed on
a measurement at order=4096, where it is serial in the order and takes 150 ms — but
the panel solve is order=**64** with m right-hand sides, an entirely different regime,
so the dismissal was against the wrong shape. A standalone probe at order=64 looked
strong: ~0.26 ms for m=4096, against a hand kernel costing ~29 ms across all 64 steps.
Wired in and measured end to end: **62.2 ms against 47.3 ms**, i.e. clearly worse.

The probe misled because its ~0.25 ms is *real GPU work*, not the command-buffer round
trip it was assumed to include — so it does not amortise inside a single command
buffer, and 64 invocations cost more than the hand kernel does in total. Note also
that MPS TRSM's cost is nearly flat in m (0.405 ms at m=512, 0.264 ms at m=4096),
which is the signature of a fixed overhead rather than useful work. Reverted;
reproduce with `JAXMETAL_CHOL_MPS_TRSM=1`.

What *did* help, modestly: holding each thread's row in registers instead of
re-reading `row[p]` from device memory inside the inner loop (46.6 → 44.7 ms overall,
TRSM 31.6 → 29.1 ms, ~8% on the phase). At 64 floats per thread `r` is likely
spilling, which is the next thing to attack.

The real fix is **a blocked TRSM that expresses the panel solve as GEMMs too** —
genuine two-level blocking, so the 65% phase becomes GEMM work like the 0.5% phase
already is. That is a different algorithm, not a tuning constant, which is why it is
not in this version.

## Limits and things left out

- **Below N≈2048 the CPU wins outright**, and below N≈1024 by 4–10×. The fixed
  command-buffer cost plus the sequential panel chain dominate.
- **Two-level blocking is the outstanding work**, per the refutations above.
- **LU is not implemented.** Unpivoted LU is not shippable: measured growth factor
  `max|U|/max|A|` is 4.0e3 at N=512 and `inf` on a zero leading pivot. Partial
  pivoting forces a data-dependent host decision per panel, so LU wants a MAGMA-style
  CPU panel (Accelerate factoring in place in the same unified-memory buffer) and
  N/NB command buffers rather than one. That is a different enough shape to be its own
  feature.
- **`solve`, `det` and `inv` are not GPU ops.** They need triangular solves at
  order=N, which is inherently sequential and measured 150 ms at N=4096 via
  `MPSMatrixSolveTriangular` — roughly 1000× slower than the CPU. Factor on the GPU,
  solve on the CPU.
- **f32 only.** Apple GPUs have no f64, so ill-conditioned systems (cond ≳ 1e5) want
  the CPU regardless of speed.
