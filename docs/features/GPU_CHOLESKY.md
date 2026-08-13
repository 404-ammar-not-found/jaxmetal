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
| 512 | 5.56 ms | 8 | 0.38 ms | 119 | 0.07× | 1.29 ms | 0.23× | 4.3e-07 |
| 1024 | 9.48 ms | 38 | 1.90 ms | 189 | 0.20× | 6.22 ms | 0.66× | 5.4e-07 |
| 2048 | 17.45 ms | 164 | 8.43 ms | 340 | 0.48× | 35.15 ms | 2.01× | 6.2e-07 |
| 4096 | 48.03 ms | 477 | **51.14 ms** | 448 | **1.06×** | 181.04 ms | 3.77× | 8.0e-07 |

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

Use `JAXMETAL_CHOL_PROFILE=1`, which puts each phase in its own command buffer and
sums the GPU's own clock (`MTLCommandBuffer.GPUStartTime`/`GPUEndTime`). At N=4096 the
three phases are **roughly balanced**:

| phase | GPU time | share |
|---|---:|---:|
| panel (`chol_panel`) | ~19–27 ms | ~40% |
| trailing MPS GEMM | ~24–26 ms | ~38% |
| TRSM (`chol_trsm_right`) | ~15–18 ms | ~22% |

> **An earlier attribution here was wrong, and six optimisations were attempted
> against it.** `JAXMETAL_CHOL_PHASES` (skip a phase, time the rest) reported "TRSM
> 65%, GEMM 0.5%". That method is invalid: its numbers are not additive — the isolated
> phases summed to *more* than the whole run — because skipping a phase leaves the
> matrix holding different values, which changes what the remaining phases cost. The
> flag is kept only as a rough cross-check; trust `JAXMETAL_CHOL_PROFILE`.

No phase dominates, which is why every single-phase optimisation below moved the total
so little. The panel is the largest and does only ~2.8 MMAC over the whole
factorisation (~0.2 GFLOP/s), so it is pure latency: 64 columns × 64 blocks = 4096
sequential steps, each with a serial diagonal computation and a barrier. That chain is
Cholesky's inherent data dependency, not an implementation artefact.

## Refuted: nine optimisations that measured worse or flat

Recorded in full because the aggregate finding is more useful than any of them
individually: **this factorisation has no single hot spot to attack.** Its three phases
are within 2× of each other, and its largest is a latency chain that is intrinsic to
the algorithm.

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

**5. One SIMD group for the panel** instead of a 256-thread threadgroup, so the
per-column barrier becomes `simdgroup_barrier` (lanes already run in lockstep) rather
than a full `threadgroup_barrier` — 4096 of them across a factorisation. Measured:
panel 27.1 ms → **32.1 ms**. The lost parallelism in the column update costs more than
the cheaper barrier saves.

**6. Threadgroup staging for the panel**, on the theory its dependent inner loop was
device-memory-latency bound. Measured: panel 27.1 ms → **28.9 ms**, no better, and it
caps NB at 64 (16 KB against Apple's 32 KB). Left operating on device memory, which is
simpler and keeps `JAXMETAL_CHOL_NB` able to sweep.

**7. Inverting the diagonal block so the panel solve becomes a GEMM.** The textbook fix
and what MAGMA does: form `L11⁻¹` once per step (O(nb³/6), ~2.8 MMAC total) and the
solve becomes `L21 = A21 · (L11⁻¹)ᵀ`, a pure GEMM. Measured end to end: **49.0 ms
against 44.9 ms**. The GEMM is `m×64 @ 64×64` — too narrow to be efficient — and the
inverse and copy-back kernels add two more dispatches per step. Residual was unharmed
(6.2e-07 vs 6.8e-07), so the stability trade was fine; it just was not faster. Kept
behind `JAXMETAL_CHOL_TRSM=gemm`.

**8. Chunking the TRSM row into 16 registers** instead of 64, to raise occupancy.
Measured: TRSM 29.8 ms → **30.8 ms**. Register pressure is not the limit either.

**9. A recursive (divide-and-conquer) formulation.** The best-motivated attempt of the
nine, and the only one with direct supporting evidence beforehand. Per-phase GPU time
against block rank at N=4096:

| NB | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|
| GEMM | 43.0 ms | 13.3 ms | 8.9 ms | **7.7 ms** |
| panel + TRSM | 68.5 ms | 57.6 ms | 109.3 ms | 220.0 ms |

The trailing GEMM improves **5.6×** with rank while the panel degrades, because panel
work is O(N·NB²). A right-looking loop must pick one NB and accept both curves;
recursion should not have to — `chol(A11); A21 = A21·A11⁻ᵀ; A22 -= A21·A21ᵀ; chol(A22)`
keeps a 64-wide base-case panel while its GEMMs are rank n/2.

Implemented (host-side recursion emitting into the same command buffer, so no extra
synchronisation) and measured: **59.3 ms against 48.8 ms**. Not shipped.

The recursive TRSM is what costs it. Splitting down to a 64-wide base case turns one
large substitution kernel into ~31 internal GEMMs plus 32 leaf TRSMs *per level*, and
the per-dispatch overhead of that swarm exceeds what the better GEMM rank returns.
Meanwhile the base-case panel count is unchanged, so the largest phase does not move at
all. The 5.6× GEMM figure was real; it just was not reachable at this dispatch cost.

(Implementation note for anyone retrying it: `n1 = ((n/2)/NB)*NB` floors to **0** for
`NB < n < 2NB`, which is an infinite recursion. It segfaults exactly there.)

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
- **There is no known next step that would help.** Nine optimisations have been
  implemented and measured; one returned 4%. The three phases are balanced, so no
  single-phase win moves the total much, and the largest phase is a sequential chain
  intrinsic to Cholesky. Both structural alternatives have now been tried: two-level
  blocking in miniature (refutation 7) and full recursion (refutation 9), and both lost
  to per-dispatch overhead. Anything further should start by questioning the
  one-command-buffer, MPS-GEMM architecture rather than tuning inside it.
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
