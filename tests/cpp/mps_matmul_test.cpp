// Tests for the MetalPerformanceShaders (MPS) matmul path.

#include "test_framework.h"
#include "gpu_test_util.h"

#include "jaxmetal/metal/dtype.h"
#include "jaxmetal/metal/metal_buffer.h"
#include "jaxmetal/metal/metal_context.h"
#include "jaxmetal/ops/mps_matmul.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace jaxmetal;

namespace {

double run_mps(const std::vector<float>& a, const std::vector<float>& b,
               int64_t M, int64_t K, int64_t N) {
  auto& c = testutil::ctx();
  auto da = c.from_host(a.data(), {M, K}, DType::F32);
  auto db = c.from_host(b.data(), {K, N}, DType::F32);
  auto out = c.alloc({M, N}, DType::F32);
  mps_matmul_into(c, *da, *db, *out, M, K, N);

  // double-accumulated reference
  std::vector<float> ref(static_cast<size_t>(M * N), 0.0f);
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int64_t k = 0; k < K; ++k)
        acc += static_cast<double>(a[i * K + k]) * static_cast<double>(b[k * N + j]);
      ref[i * N + j] = static_cast<float>(acc);
    }
  const float* ho = static_cast<const float*>(out->contents());
  double e = 0.0;
  for (int64_t i = 0; i < M * N; ++i)
    e = std::fmax(e, std::fabs(static_cast<double>(ho[i]) - static_cast<double>(ref[i])));
  return e;
}

std::vector<float> rnd(int64_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

}  // namespace

TEST(MpsMatmulSquare) {
  const int64_t N = 128;
  CHECK_NEAR(run_mps(rnd(N * N, 1), rnd(N * N, 2), N, N, N), 0.0, 1e-3);
}

TEST(MpsMatmulNonSquare) {
  const int64_t M = 100, K = 77, N = 33;  // non-tile-aligned
  CHECK_NEAR(run_mps(rnd(M * K, 3), rnd(K * N, 4), M, K, N), 0.0, 1e-3);
}

TEST(MpsMatmulLarge) {
  const int64_t M = 512, K = 384, N = 256;
  CHECK_NEAR(run_mps(rnd(M * K, 5), rnd(K * N, 6), M, K, N), 0.0, 2e-3);
}
