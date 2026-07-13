// Tests for the elementwise arithmetic ops running on the GPU, checked against a
// CPU reference. Covers numeric correctness, non-round dispatch sizes, scalar
// shapes, and the error guards.

#include "test_framework.h"
#include "gpu_test_util.h"

#include "jaxmetal/metal/metal_context.h"
#include "jaxmetal/metal/metal_buffer.h"
#include "jaxmetal/metal/kernel_library.h"
#include "jaxmetal/runtime/dispatcher.h"
#include "jaxmetal/ops/elementwise.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

using namespace jaxmetal;

namespace {

std::vector<float> random_vec(int64_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (int64_t i = 0; i < n; ++i) v[i] = dist(rng);
  return v;
}

// Max abs error between GPU out=op(a,b) and a CPU float reference.
double max_err_binary(BinaryOp op, const std::vector<float>& a, const std::vector<float>& b,
                      const std::function<float(float, float)>& ref) {
  auto& c = testutil::ctx();
  Dispatcher disp(c);
  const int64_t n = static_cast<int64_t>(a.size());
  auto da = c.from_host(a.data(), {n}, DType::F32);
  auto db = c.from_host(b.data(), {n}, DType::F32);
  auto out = elementwise_binary(c, testutil::lib(), disp, op, *da, *db);
  disp.wait();
  const float* ho = static_cast<const float*>(out->contents());
  double e = 0.0;
  for (int64_t i = 0; i < n; ++i)
    e = std::fmax(e, std::fabs(static_cast<double>(ho[i]) - static_cast<double>(ref(a[i], b[i]))));
  return e;
}

double max_err_unary(UnaryOp op, const std::vector<float>& a,
                     const std::function<float(float)>& ref) {
  auto& c = testutil::ctx();
  Dispatcher disp(c);
  const int64_t n = static_cast<int64_t>(a.size());
  auto da = c.from_host(a.data(), {n}, DType::F32);
  auto out = elementwise_unary(c, testutil::lib(), disp, op, *da);
  disp.wait();
  const float* ho = static_cast<const float*>(out->contents());
  double e = 0.0;
  for (int64_t i = 0; i < n; ++i)
    e = std::fmax(e, std::fabs(static_cast<double>(ho[i]) - static_cast<double>(ref(a[i]))));
  return e;
}

constexpr int64_t kN = 4096;

}  // namespace

// ---- binary ops (safe-math => bit-exact vs CPU float) ----
TEST(BinaryAdd) {
  auto a = random_vec(kN, -50, 50, 1), b = random_vec(kN, -50, 50, 2);
  CHECK_NEAR(max_err_binary(BinaryOp::Add, a, b, [](float x, float y) { return x + y; }), 0.0, 1e-6);
}
TEST(BinarySub) {
  auto a = random_vec(kN, -50, 50, 3), b = random_vec(kN, -50, 50, 4);
  CHECK_NEAR(max_err_binary(BinaryOp::Sub, a, b, [](float x, float y) { return x - y; }), 0.0, 1e-6);
}
TEST(BinaryMul) {
  auto a = random_vec(kN, -20, 20, 5), b = random_vec(kN, -20, 20, 6);
  CHECK_NEAR(max_err_binary(BinaryOp::Mul, a, b, [](float x, float y) { return x * y; }), 0.0, 1e-6);
}
TEST(BinaryDiv) {
  auto a = random_vec(kN, -50, 50, 7), b = random_vec(kN, 1, 4, 8);  // b > 0
  CHECK_NEAR(max_err_binary(BinaryOp::Div, a, b, [](float x, float y) { return x / y; }), 0.0, 1e-6);
}
TEST(BinaryMax) {
  auto a = random_vec(kN, -50, 50, 9), b = random_vec(kN, -50, 50, 10);
  CHECK_NEAR(max_err_binary(BinaryOp::Max, a, b, [](float x, float y) { return std::fmax(x, y); }),
             0.0, 1e-6);
}
TEST(BinaryMin) {
  auto a = random_vec(kN, -50, 50, 11), b = random_vec(kN, -50, 50, 12);
  CHECK_NEAR(max_err_binary(BinaryOp::Min, a, b, [](float x, float y) { return std::fmin(x, y); }),
             0.0, 1e-6);
}

// ---- unary ops ----
TEST(UnaryNeg) {
  auto a = random_vec(kN, -50, 50, 13);
  CHECK_NEAR(max_err_unary(UnaryOp::Neg, a, [](float x) { return -x; }), 0.0, 1e-6);
}
TEST(UnaryAbs) {
  auto a = random_vec(kN, -50, 50, 14);
  CHECK_NEAR(max_err_unary(UnaryOp::Abs, a, [](float x) { return std::fabs(x); }), 0.0, 1e-6);
}
TEST(UnaryExp) {
  auto a = random_vec(kN, -3, 3, 15);  // bounded to avoid overflow
  CHECK_NEAR(max_err_unary(UnaryOp::Exp, a, [](float x) { return std::exp(x); }), 0.0, 1e-4);
}

// ---- dispatch edge cases ----
TEST(NonRoundSizes) {
  // Sizes that are not multiples of a threadgroup width exercise the bounds guard.
  for (int64_t n : {1, 7, 33, 1000, 1023, 1025, 4097}) {
    auto a = random_vec(n, -10, 10, 100 + static_cast<uint32_t>(n));
    auto b = random_vec(n, -10, 10, 200 + static_cast<uint32_t>(n));
    double e = max_err_binary(BinaryOp::Add, a, b, [](float x, float y) { return x + y; });
    CHECK_NEAR(e, 0.0, 1e-6);
  }
}

TEST(LargeSize) {
  const int64_t n = 1 << 20;  // 1M elements
  auto a = random_vec(n, -1, 1, 42), b = random_vec(n, -1, 1, 43);
  CHECK_NEAR(max_err_binary(BinaryOp::Mul, a, b, [](float x, float y) { return x * y; }), 0.0, 1e-6);
}

TEST(ScalarElementwise) {
  auto& c = testutil::ctx();
  Dispatcher disp(c);
  float xa = 3.0f, xb = 4.0f;
  auto a = c.from_host(&xa, {}, DType::F32);
  auto b = c.from_host(&xb, {}, DType::F32);
  auto out = elementwise_binary(c, testutil::lib(), disp, BinaryOp::Add, *a, *b);
  disp.wait();
  CHECK_NEAR(static_cast<const float*>(out->contents())[0], 7.0f, 1e-6);
}

// ---- error guards ----
TEST(SizeMismatchThrows) {
  auto& c = testutil::ctx();
  Dispatcher disp(c);
  auto a = c.alloc({4}, DType::F32);
  auto b = c.alloc({5}, DType::F32);
  CHECK_THROWS(elementwise_binary(c, testutil::lib(), disp, BinaryOp::Add, *a, *b));
}

TEST(NonF32DtypeThrows) {
  auto& c = testutil::ctx();
  Dispatcher disp(c);
  auto a = c.alloc({4}, DType::I32);
  auto b = c.alloc({4}, DType::I32);
  CHECK_THROWS(elementwise_binary(c, testutil::lib(), disp, BinaryOp::Add, *a, *b));
}

TEST(MissingKernelThrows) {
  CHECK_THROWS(testutil::lib().pipeline("does_not_exist"));
}
