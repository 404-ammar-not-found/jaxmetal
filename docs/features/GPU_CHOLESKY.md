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

**NB is capped by threadgroup memory, not by GEMM efficiency.** The panel kernels
hold an NB×NB block in threadgroup memory: NB=64 is 16 KB against the 32 KB Apple
GPUs provide. NB=128 would need 64 KB and does not fit. This is the main thing
holding performance back — see Limits.

## Load-bearing invariants

> **`RED`-style constant matching:** `CHOL_NB` and `CHOL_TG` in
> `kernels/cholesky.metal` must equal `kCholNB` / `kCholTG` in `src/ops/cholesky.mm`.
> The kernels size their threadgroup arrays to exactly `CHOL_NB`.

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
| 512 | 3.56 ms | 13 | 0.36 ms | 123 | 0.10× | 1.31 ms | 0.37× | 4.3e-07 |
| 1024 | 7.36 ms | 49 | 1.90 ms | 189 | 0.26× | 6.22 ms | 0.85× | 5.4e-07 |
| 2048 | 16.29 ms | 176 | 8.40 ms | 341 | 0.52× | 34.97 ms | 2.15× | 6.8e-07 |
| 4096 | 46.70 ms | 491 | **50.59 ms** | 453 | **1.08×** | 178.46 ms | 3.82× | 8.0e-07 |

**Read the `vs spotrf` column.** It is a bare Accelerate call on a Fortran-ordered
array with no copy — the true CPU floor. The result is *parity at N=4096 and a loss
below it*, not a win.

The `vs numpy` column is 4× more flattering and is the number it would be tempting to
quote, but `np.linalg.cholesky` spends most of its time reordering a C-contiguous
array for a column-major routine. Three different "LAPACK" numbers exist for N=4096 —
numpy 178 ms, scipy `spotrf` with copies 102 ms, direct `spotrf_` 45–51 ms — and only
the last is the honest baseline.

Residual is `max|L·Lᵀ − A| / max|A|`, at 4–8e-07 throughout, i.e. a few f32 eps.

## Limits and things left out

- **Half the GPU FLOPs are wasted.** The trailing update computes the full `m×m`
  rectangle where only the symmetric half is needed, because **MPS has no SYRK**.
  Fixing this is the identified path to actually beating `spotrf` — predicted ~2×,
  which would put N=4096 at ~25 ms against LAPACK's 50 ms. The fix is to issue the
  update as a few block-column strips instead of one square GEMM; the tradeoff is more
  MPS encodes per step.
- **NB=64 is too small for GEMM efficiency.** A rank-64 update at M=2048 measured only
  502 GFLOP/s against 3297 GFLOP/s for rank-256. Raising NB needs a two-level scheme
  (outer NB=256 whose diagonal block is itself factored by an inner NB=64 pass),
  because the panel cannot exceed 32 KB of threadgroup memory.
- **Below N≈2048 the CPU wins outright**, and below N≈1024 by 4–10×. The fixed
  command-buffer cost plus the sequential panel chain dominate. This is a real regime
  limit, not unfinished work — though the two fixes above would move the crossover.
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
