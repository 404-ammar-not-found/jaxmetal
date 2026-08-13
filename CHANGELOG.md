# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Every performance figure below is measured on an Apple M4 Pro and reproducible with
the scripts in [`benchmarks/`](benchmarks/). Where a change was *reverted* because it
measured worse, it is recorded rather than dropped — see
[`docs/README.md`](docs/README.md) for why.

## [0.7.0] — 2026-08-13

### Added
- `device="auto" | "gpu" | "cpu"` for `batched_solve`, `cholesky` and `df64_binop`,
  with thresholds in `python/jaxmetal/routing.py` and a `JAXMETAL_DEVICE` override.
  Routing accounts for **residency**, which moves the df64 threshold by ~50×: the same
  kernel is 1.93× faster than the CPU when resident and ~0.04× through the copy path.
- `tests/python/test_routing.py`: asserts both arms agree numerically, that the router
  picks the faster arm when timed either side of each crossover, that decisions are
  monotonic in size, and that residency can only *lower* a threshold.

### Packaging
- Version single-sourced from `jaxmetal.__version__`; `py.typed` marker (PEP 561);
  corrected project URLs; expanded classifiers; `bench` extra for `scipy`.
- All Python tests are now pytest-discoverable (`pytest` from the repo root).

## [0.6.0] — 2026-08-13

### Added
- **Double-single (`df64`) extended precision** — ~48 bits of significand on a GPU
  with no `double` type at all. Elementwise add/mul/div and a 3-point stencil.
  1e7× more accurate than f32 on cancelling work.
- `df64_binop_resident`, which is **36× faster than the host path** and **1.65–1.88×
  faster than CPU `float64`** above ~4M elements.
- **SPD Cholesky path** for `batched_solve` (`spd=True`): up to **2.7×** the LU path
  at n=6–8, with better residuals, by skipping the pivot search and row interchange.
- `batched_solve_resident`: up to **14.3×** a scalar C loop, and moves the crossover
  from ~16,000 systems to ~4,000.

### Changed
- **Corrected the df64 headline claim.** It previously read "slower than the CPU at
  every size measured", which was true of the host path and wrong about the feature.
  The copies, not the arithmetic, were the cost.

## [0.5.0] — 2026-08-13

### Added
- **Batched tiny-system solve** — thousands of independent 2×2…8×8 systems, one GPU
  thread each, LU with partial pivoting, everything in registers. 2–3× a scalar C loop
  through the host path, 15–24× numpy's batched `solve`.
- `pivmin` per system: a **singularity flag** (exactly 0 for singular/NaN/Inf input),
  explicitly not a condition estimate.

### Notes
- Capped at n ≤ 8 deliberately: `MPSMatrixDecompositionLU` inherits `batchStart`/
  `batchSize`, so Apple already ships a batched LU that wins from ~n=12 (54× at n=32).

## [0.4.0] — 2026-08-13

### Added
- **Blocked right-looking Cholesky** on the GPU — hand-written panel and TRSM, MPS
  GEMM for the trailing update, the entire factorisation in one command buffer.
  Parity with direct LAPACK `spotrf` at N=4096, 3.8× `np.linalg.cholesky`, and **13×
  Apple's own `MPSMatrixDecompositionCholesky`** (which measures 620 ms against
  LAPACK's 45 ms and is unusable).

### Notes — nine refuted optimisations
Recorded in [`docs/features/GPU_CHOLESKY.md`](docs/features/GPU_CHOLESKY.md). SYRK
strips, larger `NB`, a split accumulator, one-SIMD-group and threadgroup-staged panels,
`MPSMatrixSolveTriangular`, inverting the diagonal block, TRSM register chunking, and
a full recursive formulation all measured worse or flat; only register-resident TRSM
rows helped, by 4%.

An earlier phase attribution ("TRSM 65%, GEMM 0.5%") was **wrong** — it came from a
skip-a-phase method whose numbers were not additive. GPU timestamps show the three
phases are *balanced*, which is why no single-phase optimisation moved the total.

## [0.3.0] — 2026-08-12

### Added
- **Compensated (Neumaier) f32 summation** — 127× more accurate than a tree sum on
  adversarial input, better than numpy's pairwise f32 on every case tested, at **no
  measurable cost** (0.99× at 268 MB; both kernels are bandwidth-bound).
- Per-feature documentation structure under [`docs/`](docs/).

## [0.2.0] — 2026-08-12

### Added
- **Chunked training** — `train_steps(n)` encodes many SGD steps into one Metal
  command buffer, so the ~140 µs driver round trip is paid once per chunk. Cut MLP
  per-step cost **~4.4×** and moved the GPU-beats-CPU crossover from batch >2048 to
  **~60**. Gated as bit-identical to n separate `train_step()` calls.
- `Mlp(device="auto")`, routing on a measured crossover model.

### Fixed
- `nn_reduce_sum_axis0` dispatched **10 threads** for `db2` (C=10 classes), each
  walking the entire batch serially. Rewritten as one threadgroup per column with a
  tree reduction — batch 2048 went 1.31 → 0.79 ms on this change alone.
- Documentation claimed the GPU beat JAX-CPU "at batch ≥512 (1.3–2.7×)". Measured, it
  was **0.53×** there at `hidden=128`; the claim only held at `hidden ≥ 1024`.

## [0.1.0]

Initial prototype: Metal runtime (context, buffers, kernel library, dispatcher),
hand-written MSL elementwise and matmul kernels, MPS and Accelerate matmul backends,
a resident MLP trainer reaching 98% on MNIST, and an XLA FFI custom call for
`jax.jit` integration.

[0.7.0]: https://github.com/404-ammar-not-found/jaxmetal/releases/tag/v0.7.0
[0.6.0]: https://github.com/404-ammar-not-found/jaxmetal/releases/tag/v0.6.0
[0.5.0]: https://github.com/404-ammar-not-found/jaxmetal/releases/tag/v0.5.0
[0.4.0]: https://github.com/404-ammar-not-found/jaxmetal/releases/tag/v0.4.0
[0.3.0]: https://github.com/404-ammar-not-found/jaxmetal/releases/tag/v0.3.0
[0.2.0]: https://github.com/404-ammar-not-found/jaxmetal/releases/tag/v0.2.0
