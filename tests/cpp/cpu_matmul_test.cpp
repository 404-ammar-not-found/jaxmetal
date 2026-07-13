// Tests for the CPU (Accelerate) matmul path used by the auto router.

#include "test_framework.h"

#include "jaxmetal/cpu/cpu_matmul.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace jaxmetal;

namespace {

std::vector<float> ref_matmul(const std::vector<float>& a, const std::vector<float>& b,
                              int64_t M, int64_t K, int64_t N) {
  std::vector<float> c(static_cast<size_t>(M * N), 0.0f);
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int64_t k = 0; k < K; ++k)
        acc += static_cast<double>(a[i * K + k]) * static_cast<double>(b[k * N + j]);
      c[i * N + j] = static_cast<float>(acc);
    }
  return c;
}

std::vector<float> rnd(int64_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

}  // namespace

TEST(CpuMatmulSmallExact) {
  std::vector<float> a = {1, 2, 3, 4, 5, 6};     // 2x3
  std::vector<float> b = {7, 8, 9, 10, 11, 12};  // 3x2
  std::vector<float> c(4, 0.0f);
  cpu_matmul_f32(a.data(), b.data(), c.data(), 2, 3, 2);
  // [[58,64],[139,154]]
  CHECK_NEAR(c[0], 58.0f, 1e-4);
  CHECK_NEAR(c[1], 64.0f, 1e-4);
  CHECK_NEAR(c[2], 139.0f, 1e-4);
  CHECK_NEAR(c[3], 154.0f, 1e-4);
}

TEST(CpuMatmulRandomMatchesReference) {
  const int64_t M = 40, K = 55, N = 33;
  auto a = rnd(M * K, 1), b = rnd(K * N, 2);
  std::vector<float> c(static_cast<size_t>(M * N), 0.0f);
  cpu_matmul_f32(a.data(), b.data(), c.data(), M, K, N);
  auto ref = ref_matmul(a, b, M, K, N);
  double e = 0.0;
  for (int64_t i = 0; i < M * N; ++i)
    e = std::fmax(e, std::fabs(static_cast<double>(c[i]) - static_cast<double>(ref[i])));
  CHECK_NEAR(e, 0.0, 1e-3);
}
