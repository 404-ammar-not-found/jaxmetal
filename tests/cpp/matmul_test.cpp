// Tests for the tiled GPU matmul against a CPU reference. Includes exact
// small-integer cases (f32-exact) and random tolerance checks across sizes that
// straddle the 16x16 tile boundary.

#include "test_framework.h"
#include "gpu_test_util.h"

#include "jaxmetal/metal/dtype.h"
#include "jaxmetal/metal/kernel_library.h"
#include "jaxmetal/metal/metal_buffer.h"
#include "jaxmetal/metal/metal_context.h"
#include "jaxmetal/runtime/dispatcher.h"
#include "jaxmetal/ops/matmul.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace jaxmetal;

namespace {

// Register the matmul kernel exactly once for this test binary.
KernelLibrary& mm_lib() {
  KernelLibrary& l = testutil::lib();
  static int once = (register_matmul_kernel(l), 0);
  (void)once;
  return l;
}

std::vector<float> cpu_matmul(const std::vector<float>& a, const std::vector<float>& b,
                              int64_t M, int64_t K, int64_t N) {
  std::vector<float> c(static_cast<size_t>(M * N), 0.0f);
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j) {
      double acc = 0.0;  // double accumulation as the "truth"
      for (int64_t k = 0; k < K; ++k)
        acc += static_cast<double>(a[i * K + k]) * static_cast<double>(b[k * N + j]);
      c[i * N + j] = static_cast<float>(acc);
    }
  return c;
}

// Run GPU matmul and return max abs error vs the CPU reference.
double run_matmul(const std::vector<float>& a, const std::vector<float>& b,
                  int64_t M, int64_t K, int64_t N) {
  auto& c = testutil::ctx();
  Dispatcher disp(c);
  auto da = c.from_host(a.data(), {M, K}, DType::F32);
  auto db = c.from_host(b.data(), {K, N}, DType::F32);
  auto out = matmul(c, mm_lib(), disp, *da, *db);
  disp.wait();

  CHECK(out->shape().size() == 2);
  CHECK(out->shape()[0] == M && out->shape()[1] == N);

  auto ref = cpu_matmul(a, b, M, K, N);
  const float* ho = static_cast<const float*>(out->contents());
  double e = 0.0;
  for (int64_t i = 0; i < M * N; ++i)
    e = std::fmax(e, std::fabs(static_cast<double>(ho[i]) - static_cast<double>(ref[i])));
  return e;
}

std::vector<float> random_mat(int64_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (int64_t i = 0; i < n; ++i) v[i] = dist(rng);
  return v;
}

}  // namespace

TEST(MatmulIdentity) {
  const int64_t d = 5;
  std::vector<float> id(d * d, 0.0f), x(d * d);
  for (int64_t i = 0; i < d; ++i) id[i * d + i] = 1.0f;
  for (int64_t i = 0; i < d * d; ++i) x[i] = static_cast<float>(i) - 7.0f;
  // I @ X == X, exactly.
  CHECK_NEAR(run_matmul(id, x, d, d, d), 0.0, 0.0);
}

TEST(MatmulSmallExact) {
  // A[2x3] @ B[3x2], small integers -> exact in f32.
  std::vector<float> a = {1, 2, 3, 4, 5, 6};      // 2x3
  std::vector<float> b = {7, 8, 9, 10, 11, 12};   // 3x2
  // Expected: [[58,64],[139,154]]
  CHECK_NEAR(run_matmul(a, b, 2, 3, 2), 0.0, 0.0);
}

TEST(MatmulNonSquare) {
  const int64_t M = 17, K = 20, N = 9;  // all straddle the 16-wide tile
  auto a = random_mat(M * K, -1, 1, 1);
  auto b = random_mat(K * N, -1, 1, 2);
  CHECK_NEAR(run_matmul(a, b, M, K, N), 0.0, 1e-3);
}

TEST(MatmulExactTileMultiple) {
  const int64_t M = 32, K = 32, N = 32;  // exact tile multiples
  auto a = random_mat(M * K, -1, 1, 3);
  auto b = random_mat(K * N, -1, 1, 4);
  CHECK_NEAR(run_matmul(a, b, M, K, N), 0.0, 1e-3);
}

TEST(MatmulLarge) {
  const int64_t M = 128, K = 256, N = 64;
  auto a = random_mat(M * K, -1, 1, 5);
  auto b = random_mat(K * N, -1, 1, 6);
  // Larger K accumulates more f32 error; tolerance scales accordingly.
  CHECK_NEAR(run_matmul(a, b, M, K, N), 0.0, 5e-3);
}

TEST(MatmulRowVector) {
  const int64_t M = 1, K = 64, N = 48;  // [1,K] @ [K,N] -> [1,N]
  auto a = random_mat(M * K, -1, 1, 7);
  auto b = random_mat(K * N, -1, 1, 8);
  CHECK_NEAR(run_matmul(a, b, M, K, N), 0.0, 1e-3);
}

TEST(MatmulRankMismatchThrows) {
  auto& c = testutil::ctx();
  Dispatcher disp(c);
  auto a = c.alloc({4}, DType::F32);      // rank-1
  auto b = c.alloc({4, 4}, DType::F32);
  CHECK_THROWS(matmul(c, mm_lib(), disp, *a, *b));
}

TEST(MatmulInnerDimMismatchThrows) {
  auto& c = testutil::ctx();
  Dispatcher disp(c);
  auto a = c.alloc({4, 3}, DType::F32);
  auto b = c.alloc({5, 2}, DType::F32);  // 3 != 5
  CHECK_THROWS(matmul(c, mm_lib(), disp, *a, *b));
}
