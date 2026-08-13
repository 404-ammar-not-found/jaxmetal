// Batched tiny-system solve vs double-precision CPU references.
//
// Two things these tests are shaped around, both from failure modes that are easy to
// ship by accident:
//
//   * The row interchange is written as conditional swaps with static indices, which
//     is the single most delicate part of the kernel. Diagonally-dominant test
//     matrices perform NO interchanges, so a swap bug would pass a "random matrices"
//     sweep completely. BatchedSolveExercisesPivoting uses matrices that REQUIRE a
//     swap at every size.
//   * Assertions are on the scaled RESIDUAL, not the forward error. Backward-stable
//     LU guarantees a small residual regardless of conditioning; forward error is
//     heavy-tailed over random draws and would make the suite flaky.

#include "test_framework.h"
#include "gpu_test_util.h"

#include "jaxmetal/metal/metal_context.h"
#include "jaxmetal/metal/metal_buffer.h"
#include "jaxmetal/metal/kernel_library.h"
#include "jaxmetal/metal/dtype.h"
#include "jaxmetal/runtime/dispatcher.h"
#include "jaxmetal/ops/batched_solve.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace jaxmetal;

namespace {

KernelLibrary& bs_lib() {
  KernelLibrary& l = testutil::lib();
  static int once = (register_batched_solve_kernels(l), 0);
  (void)once;
  return l;
}

struct Result { std::vector<float> x, pivmin; };

Result run(const std::vector<float>& A, const std::vector<float>& rhs,
           int64_t batch, int64_t n) {
  MetalContext& ctx = testutil::ctx();
  Dispatcher disp(ctx);
  auto dA = ctx.from_host(A.data(), {batch, n, n}, DType::F32);
  auto dR = ctx.from_host(rhs.data(), {batch, n}, DType::F32);
  auto dX = ctx.alloc({batch, n}, DType::F32);
  auto dP = ctx.alloc({batch}, DType::F32);
  batched_solve_f32(bs_lib(), disp, *dA, *dR, *dX, *dP, batch, n);
  disp.wait();
  const float* xp = static_cast<const float*>(dX->contents());
  const float* pp = static_cast<const float*>(dP->contents());
  return {std::vector<float>(xp, xp + (size_t)(batch * n)),
          std::vector<float>(pp, pp + (size_t)batch)};
}

// max_b ||A_b x_b - rhs_b||_inf / (||A_b||_inf ||x_b||_inf), in double.
double worst_residual(const std::vector<float>& A, const std::vector<float>& rhs,
                      const std::vector<float>& x, int64_t batch, int64_t n) {
  double worst = 0.0;
  for (int64_t s = 0; s < batch; ++s) {
    double anorm = 0.0, xnorm = 0.0, rnorm = 0.0;
    for (int64_t i = 0; i < n; ++i) {
      double rowsum = 0.0, r = -(double)rhs[(size_t)(s * n + i)];
      for (int64_t j = 0; j < n; ++j) {
        const double aij = A[(size_t)(s * n * n + i * n + j)];
        rowsum += std::abs(aij);
        r += aij * (double)x[(size_t)(s * n + j)];
      }
      anorm = std::max(anorm, rowsum);
      xnorm = std::max(xnorm, std::abs((double)x[(size_t)(s * n + i)]));
      rnorm = std::max(rnorm, std::abs(r));
    }
    if (anorm * xnorm > 0.0) worst = std::max(worst, rnorm / (anorm * xnorm));
  }
  return worst;
}

std::vector<float> randv(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& e : v) e = d(rng);
  return v;
}

}  // namespace

// Random Gaussian systems -- deliberately NOT diagonally dominant, so partial
// pivoting actually fires. Every supported size, over a batch large enough that a
// per-size indexing bug cannot hide.
TEST(BatchedSolveMatchesReference) {
  const int64_t batch = 4096;
  for (int64_t n = 2; n <= kBatchedSolveMaxN; ++n) {
    auto A = randv((size_t)(batch * n * n), (uint32_t)(100 + n));
    auto rhs = randv((size_t)(batch * n), (uint32_t)(200 + n));
    auto r = run(A, rhs, batch, n);
    CHECK(worst_residual(A, rhs, r.x, batch, n) < 1e-5);
  }
}

// The GPU and the scalar CPU path run the same algorithm, so they must agree closely.
// This is what would catch the register-array spill / dynamic-index class of bug,
// where the GPU silently computes something else.
TEST(BatchedSolveMatchesCpuPath) {
  const int64_t batch = 512;
  for (int64_t n = 2; n <= kBatchedSolveMaxN; ++n) {
    auto A = randv((size_t)(batch * n * n), (uint32_t)(300 + n));
    auto rhs = randv((size_t)(batch * n), (uint32_t)(400 + n));
    auto gpu = run(A, rhs, batch, n);
    std::vector<float> cpu((size_t)(batch * n));
    batched_solve_cpu_f32(A.data(), rhs.data(), cpu.data(), batch, n);
    double worst = 0.0;
    for (size_t i = 0; i < cpu.size(); ++i)
      worst = std::max(worst, (double)std::abs(gpu.x[i] - cpu[i]));
    CHECK(worst < 1e-3);
  }
}

// Systems that CANNOT be solved without a row interchange: the leading entry is zero,
// so an implementation that skips pivoting divides by zero and returns inf/NaN.
// Without this, the conditional-swap logic is effectively untested.
TEST(BatchedSolveExercisesPivoting) {
  for (int64_t n = 2; n <= kBatchedSolveMaxN; ++n) {
    const int64_t batch = 64;
    std::vector<float> A((size_t)(batch * n * n), 0.0f), rhs((size_t)(batch * n), 1.0f);
    for (int64_t s = 0; s < batch; ++s)
      for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j)
          // Anti-diagonal: a[i][j] = 1 when i+j == n-1. a[0][0] is zero for n >= 2,
          // so row 0 must be swapped before elimination can proceed.
          A[(size_t)(s * n * n + i * n + j)] = (i + j == n - 1) ? 1.0f : 0.0f;
    auto r = run(A, rhs, batch, n);
    CHECK(worst_residual(A, rhs, r.x, batch, n) < 1e-5);
    for (int64_t s = 0; s < batch; ++s) CHECK(r.pivmin[(size_t)s] > 0.0f);
  }
}

// pivmin must be EXACTLY zero for a singular system, for a NaN input, and for an inf
// input. The NaN case is the one that is easy to get wrong: built from min()/max()
// (which are fmin/fmax) the NaN would be dropped and the system would report the
// healthiest possible score while its answer is entirely NaN.
TEST(BatchedSolveFlagsSingularAndNaN) {
  const int64_t n = 4, batch = 4;
  std::vector<float> A((size_t)(batch * n * n), 0.0f), rhs((size_t)(batch * n), 1.0f);
  auto setI = [&](int64_t s) {
    for (int64_t i = 0; i < n; ++i) A[(size_t)(s * n * n + i * n + i)] = 1.0f;
  };
  setI(0);                                                   // 0: well conditioned
  /* 1: all zeros -> singular */                             // (left at 0)
  setI(2); A[(size_t)(2 * n * n)] = std::nanf("");           // 2: NaN on the diagonal
  setI(3); A[(size_t)(3 * n * n)] = INFINITY;                // 3: inf on the diagonal

  auto r = run(A, rhs, batch, n);
  CHECK(r.pivmin[0] > 0.0f);
  CHECK(r.pivmin[1] == 0.0f);
  CHECK(r.pivmin[2] == 0.0f);
  CHECK(r.pivmin[3] == 0.0f);
}

// Batch sizes that are not multiples of the threadgroup width, including 1.
TEST(BatchedSolveHandlesRaggedBatch) {
  const int64_t n = 6;
  for (int64_t batch : {1, 2, 31, 33, 255, 257, 1023}) {
    auto A = randv((size_t)(batch * n * n), (uint32_t)batch);
    auto rhs = randv((size_t)(batch * n), (uint32_t)(batch + 7));
    auto r = run(A, rhs, batch, n);
    CHECK(worst_residual(A, rhs, r.x, batch, n) < 1e-5);
  }
}

// Sizes outside the supported range must fail loudly rather than silently doing
// something slow or wrong -- above n=8 Apple's batched MPS LU is the right answer.
TEST(BatchedSolveRejectsUnsupportedSizes) {
  MetalContext& ctx = testutil::ctx();
  Dispatcher disp(ctx);
  auto A = ctx.alloc({1, 9, 9}, DType::F32);
  auto rhs = ctx.alloc({1, 9}, DType::F32);
  auto x = ctx.alloc({1, 9}, DType::F32);
  auto p = ctx.alloc({1}, DType::F32);
  CHECK_THROWS(batched_solve_f32(bs_lib(), disp, *A, *rhs, *x, *p, 1, 9));
  CHECK_THROWS(batched_solve_f32(bs_lib(), disp, *A, *rhs, *x, *p, 1, 1));
}
