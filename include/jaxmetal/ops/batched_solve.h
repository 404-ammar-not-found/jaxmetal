#pragma once
#include <cstdint>

namespace jaxmetal {

class MetalContext;
class KernelLibrary;
class Dispatcher;
class MetalBuffer;

// Compile + register kernels/batched_solve.metal into `lib`. Call once.
void register_batched_solve_kernels(KernelLibrary& lib);

// Largest system order this op supports. Above it, Apple's own batched
// MPSMatrixDecompositionLU wins (measured 2.4x at n=12, 54x at n=32), so
// hand-writing larger sizes would be reimplementing something slower.
constexpr int64_t kBatchedSolveMaxN = 8;

// Solve `batch` independent systems A[b] x[b] = rhs[b], one GPU thread per system.
//   A      [batch, n, n] row-major f32
//   rhs    [batch, n]    f32
//   x      [batch, n]    f32   (output; may not alias rhs)
//   pivmin [batch]       f32   (output; see below)
//
// LU with partial pivoting. `pivmin[b]` is min|U_kk|/max|U_kk| for system b, or
// exactly 0.0f when that system is singular or its input contained NaN/Inf. It is a
// SINGULARITY flag, not a condition estimate: a well-scaled but ill-conditioned
// system scores 1.0 while its answer is inaccurate.
//
// Throws if n < 2 or n > kBatchedSolveMaxN. Does not wait; the caller synchronises.
void batched_solve_f32(KernelLibrary& lib, Dispatcher& disp,
                       MetalBuffer& A, MetalBuffer& rhs, MetalBuffer& x,
                       MetalBuffer& pivmin, int64_t batch, int64_t n);

// Single-threaded CPU reference: the same algorithm, scalar, one system at a time.
// This is the honest baseline for the benchmark -- a plain C loop is 3-26x faster
// than looping numpy.linalg.solve, whose per-call overhead dominates at these sizes.
void batched_solve_cpu_f32(const float* A, const float* rhs, float* x,
                           int64_t batch, int64_t n);

}  // namespace jaxmetal
