// Blocked GPU Cholesky vs double-precision CPU references.
//
// The decisive test is CholeskyMatchesLapack: it reconstructs L·Lᵀ and compares
// against the original A, which is a residual check rather than an elementwise
// comparison against another Cholesky. That matters because it is backward-stable
// regardless of conditioning, and because it is what catches the two failure modes
// this design is most exposed to: a wrong submatrix window in the MPS trailing update
// (which would corrupt a block far from the diagonal) and MPS GEMM aliasing between
// disjoint windows of one buffer, which Apple does not document.

#include "test_framework.h"
#include "gpu_test_util.h"

#include "jaxmetal/metal/metal_context.h"
#include "jaxmetal/metal/metal_buffer.h"
#include "jaxmetal/metal/kernel_library.h"
#include "jaxmetal/metal/dtype.h"
#include "jaxmetal/ops/cholesky.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace jaxmetal;

namespace {

KernelLibrary& chol_lib() {
  KernelLibrary& l = testutil::lib();
  static int once = (register_cholesky_kernels(l), 0);
  (void)once;
  return l;
}

// Symmetric positive definite: A = B·Bᵀ + n·I. The +n·I keeps the condition number
// modest so a residual failure means a bug, not f32 running out of digits.
std::vector<float> make_spd(int64_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  std::vector<float> b((size_t)(n * n)), a((size_t)(n * n), 0.0f);
  for (auto& x : b) x = d(rng);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j <= i; ++j) {
      double s = 0.0;
      for (int64_t k = 0; k < n; ++k)
        s += (double)b[(size_t)(i * n + k)] * (double)b[(size_t)(j * n + k)];
      a[(size_t)(i * n + j)] = a[(size_t)(j * n + i)] = (float)s;
    }
  for (int64_t i = 0; i < n; ++i) a[(size_t)(i * n + i)] += (float)n;
  return a;
}

// max |L·Lᵀ - A| / max|A|, accumulated in double.
double residual(const std::vector<float>& L, const std::vector<float>& A, int64_t n) {
  double worst = 0.0, scale = 0.0;
  for (double v : A) scale = std::max(scale, std::abs(v));
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j <= i; ++j) {
      double s = 0.0;
      for (int64_t k = 0; k <= j; ++k)
        s += (double)L[(size_t)(i * n + k)] * (double)L[(size_t)(j * n + k)];
      worst = std::max(worst, std::abs(s - (double)A[(size_t)(i * n + j)]));
    }
  return worst / scale;
}

// Factor in place on the GPU; returns (info, L).
std::pair<int, std::vector<float>> gpu_chol(const std::vector<float>& a, int64_t n) {
  MetalContext& ctx = testutil::ctx();
  auto buf = ctx.from_host(a.data(), {n, n}, DType::F32);
  int info = cholesky_f32(ctx, chol_lib(), *buf, n);
  const float* p = static_cast<const float*>(buf->contents());
  return {info, std::vector<float>(p, p + (size_t)(n * n))};
}

}  // namespace

// Sizes deliberately straddle the NB=64 block boundary: an exact multiple, one under,
// one over, and sizes smaller than a single block. A blocking bug that only shows up
// in the ragged final block would survive a power-of-two-only sweep.
TEST(CholeskyMatchesLapack) {
  for (int64_t n : {1, 7, 63, 64, 65, 128, 200, 256}) {
    auto a = make_spd(n, (uint32_t)n);
    auto [info, L] = gpu_chol(a, n);
    CHECK(info == 0);
    const double r = residual(L, a, n);
    CHECK(r < 1e-5);
  }
}

// The strict upper triangle must be zero, not whatever the caller passed in. A
// reconstruction test alone would not notice, because it only reads the lower part.
TEST(CholeskyZeroesUpperTriangle) {
  const int64_t n = 100;
  auto a = make_spd(n, 3);
  auto [info, L] = gpu_chol(a, n);
  CHECK(info == 0);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = i + 1; j < n; ++j)
      CHECK(L[(size_t)(i * n + j)] == 0.0f);
}

// Not positive definite must be REPORTED, not silently returned as garbage. The
// failing column is past the first block so this also checks that `info` survives
// later panels rather than being overwritten by them.
TEST(CholeskyReportsNonPositiveDefinite) {
  const int64_t n = 128;
  auto a = make_spd(n, 5);
  const int64_t bad = 70;
  // Make the leading minor at `bad` indefinite by flipping its diagonal negative.
  a[(size_t)(bad * n + bad)] = -1.0f;
  auto [info, L] = gpu_chol(a, n);
  (void)L;
  CHECK(info != 0);
  CHECK(info <= (int)n);
}

// NaN must be caught too. The kernel tests !(s > 0) rather than s <= 0 precisely so
// that a NaN pivot fails the test; s <= 0 would be false for NaN and let it through,
// producing an all-NaN factor reported as success.
TEST(CholeskyRejectsNaNInput) {
  const int64_t n = 96;
  auto a = make_spd(n, 9);
  a[(size_t)(80 * n + 80)] = std::nanf("");
  auto [info, L] = gpu_chol(a, n);
  (void)L;
  CHECK(info != 0);
}

// A diagonal matrix has an exactly representable factor (sqrt of each entry), so any
// discrepancy here is an indexing bug rather than rounding. Uses a size that is not a
// block multiple to exercise the ragged tail at the same time.
TEST(CholeskyDiagonalIsExact) {
  const int64_t n = 130;
  std::vector<float> a((size_t)(n * n), 0.0f);
  for (int64_t i = 0; i < n; ++i) a[(size_t)(i * n + i)] = 4.0f;
  auto [info, L] = gpu_chol(a, n);
  CHECK(info == 0);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < n; ++j)
      CHECK_NEAR(L[(size_t)(i * n + j)], i == j ? 2.0f : 0.0f, 1e-6f);
}
