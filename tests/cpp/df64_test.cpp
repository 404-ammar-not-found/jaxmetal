// Double-single (df64) arithmetic vs double-precision CPU references.
//
// The whole point of this feature is precision, so every test is an accuracy test,
// and each is constructed so that plain f32 DEMONSTRABLY fails it. A df64 test that
// f32 would also pass proves nothing — and would not notice if a fast-math build
// folded away every error-free transformation and silently turned df64 back into f32.

#include "test_framework.h"
#include "gpu_test_util.h"

#include "jaxmetal/metal/metal_context.h"
#include "jaxmetal/metal/metal_buffer.h"
#include "jaxmetal/metal/kernel_library.h"
#include "jaxmetal/metal/dtype.h"
#include "jaxmetal/runtime/dispatcher.h"
#include "jaxmetal/ops/df64.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace jaxmetal;

namespace {

KernelLibrary& df_lib() {
  KernelLibrary& l = testutil::lib();
  static int once = (register_df64_kernels(l), 0);
  (void)once;
  return l;
}

// double -> (hi, lo) limbs. hi is the f32 nearest to x; lo is the remainder, which is
// itself exactly representable because it is more than 24 bits below hi.
std::vector<float> to_df64(const std::vector<double>& v) {
  std::vector<float> out(v.size() * 2);
  for (size_t i = 0; i < v.size(); ++i) {
    const float hi = static_cast<float>(v[i]);
    out[2 * i] = hi;
    out[2 * i + 1] = static_cast<float>(v[i] - static_cast<double>(hi));
  }
  return out;
}

std::vector<double> from_df64(const std::vector<float>& v) {
  std::vector<double> out(v.size() / 2);
  for (size_t i = 0; i < out.size(); ++i)
    out[i] = static_cast<double>(v[2 * i]) + static_cast<double>(v[2 * i + 1]);
  return out;
}

std::vector<double> run_ew(DF64Op op, const std::vector<double>& a,
                           const std::vector<double>& b) {
  MetalContext& ctx = testutil::ctx();
  Dispatcher disp(ctx);
  const int64_t n = static_cast<int64_t>(a.size());
  auto av = to_df64(a), bv = to_df64(b);
  auto da = ctx.from_host(av.data(), {n * 2}, DType::F32);
  auto db = ctx.from_host(bv.data(), {n * 2}, DType::F32);
  auto dout = ctx.alloc({n * 2}, DType::F32);
  df64_elementwise(df_lib(), disp, op, *da, *db, *dout, n);
  disp.wait();
  const float* p = static_cast<const float*>(dout->contents());
  return from_df64(std::vector<float>(p, p + (size_t)(n * 2)));
}

double max_rel(const std::vector<double>& got, const std::vector<double>& want) {
  double worst = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = std::abs(got[i] - want[i]);
    const double s = std::abs(want[i]);
    worst = std::max(worst, s > 0 ? d / s : d);
  }
  return worst;
}

// f32 epsilon is 1.19e-07; df64 targets ~48 bits, i.e. ~3.6e-15.
constexpr double kF32Eps = 1.19e-07;
constexpr double kDF64Tol = 1e-13;

}  // namespace

// Values whose sum cancels in f32: the operands agree to well within f32 precision,
// so f32 loses most significant digits of the difference.
//
// TOLERANCE NOTE, and it is not a fudge. Cancellation amplifies RELATIVE error for any
// precision: if the operands are stored to relative accuracy eps and the sum is a
// factor R smaller than them, the result carries relative error ~eps*R. Here R = 1e5,
// so df64 (eps ~ 3.6e-15 at 48 bits) can only reach ~3.6e-10 and f32 (eps 1.2e-07)
// can only reach ~1.2e-02. Asserting 1e-13 here would be asserting something no
// finite precision can deliver, not testing df64.
TEST(DF64AddBeatsFloat32) {
  const size_t n = 4096;
  std::vector<double> a(n), b(n), want(n);
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> d(1.0, 2.0);
  for (size_t i = 0; i < n; ++i) {
    a[i] = d(rng);
    b[i] = -(a[i] - 1e-5 * a[i]);   // b ~= -a, so a+b is ~1e-5 of a
    want[i] = a[i] + b[i];
  }
  const double e_df = max_rel(run_ew(DF64Op::Add, a, b), want);

  // What plain f32 would have produced on the same inputs.
  double e_f32 = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double got = (double)((float)a[i] + (float)b[i]);
    e_f32 = std::max(e_f32, std::abs(got - want[i]) / std::abs(want[i]));
  }

  CHECK(e_f32 > 1e-3);        // f32 really does fall apart here
  CHECK(e_df < 1e-8);         // df64 does not (limit is ~3.6e-10; see the note above)
  CHECK(e_df * 1e4 < e_f32);  // and the gap is decisive, not marginal
}

// Products of values that need more than 24 bits to represent exactly.
TEST(DF64MulIsExtendedPrecision) {
  const size_t n = 4096;
  std::vector<double> a(n), b(n), want(n);
  std::mt19937 rng(2);
  std::uniform_real_distribution<double> d(0.5, 2.0);
  for (size_t i = 0; i < n; ++i) {
    a[i] = d(rng);
    b[i] = d(rng);
    want[i] = a[i] * b[i];
  }
  CHECK(max_rel(run_ew(DF64Op::Mul, a, b), want) < kDF64Tol);
}

// Division goes through a Newton step from an f32 reciprocal, so it is the operation
// most likely to be left at f32 accuracy by a broken implementation.
TEST(DF64DivIsExtendedPrecision) {
  const size_t n = 2048;
  std::vector<double> a(n), b(n), want(n);
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> d(0.5, 2.0);
  for (size_t i = 0; i < n; ++i) {
    a[i] = d(rng);
    b[i] = d(rng);
    want[i] = a[i] / b[i];
  }
  const double e = max_rel(run_ew(DF64Op::Div, a, b), want);
  CHECK(e < 1e-12);
  CHECK(e < kF32Eps * 1e-3);   // decisively better than f32, not marginally
}

// Round-tripping a value through the limb representation must be lossless to ~48 bits
// even when the value needs far more than 24. Catches a conversion that truncates.
TEST(DF64RoundTripsHostValues) {
  std::vector<double> v{1.0 / 3.0, M_PI, 1e-20, 1e20, 123456.789012345,
                        -2.718281828459045, 0.0, 1.0};
  auto back = from_df64(to_df64(v));
  for (size_t i = 0; i < v.size(); ++i) {
    if (v[i] == 0.0) { CHECK(back[i] == 0.0); continue; }
    CHECK(std::abs(back[i] - v[i]) / std::abs(v[i]) < kDF64Tol);
  }
}

// The resident path must be bit-identical to the host path: same kernel, same data,
// only buffer ownership differs. It is 36x faster at 16.7M elements purely by not
// copying, so any numerical difference would mean the host wrapper is converting
// something it should not.
TEST(DF64ResidentMatchesHostPath) {
  const int64_t n = 1 << 16;
  std::vector<double> a((size_t)n), b((size_t)n);
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> d(0.5, 2.0);
  for (int64_t i = 0; i < n; ++i) { a[(size_t)i] = d(rng); b[(size_t)i] = d(rng); }

  auto host = run_ew(DF64Op::Add, a, b);

  MetalContext& ctx = testutil::ctx();
  Dispatcher disp(ctx);
  auto av = to_df64(a), bv = to_df64(b);
  auto da = ctx.from_host(av.data(), {n * 2}, DType::F32);
  auto db = ctx.from_host(bv.data(), {n * 2}, DType::F32);
  auto dout = ctx.alloc({n * 2}, DType::F32);
  df64_elementwise(df_lib(), disp, DF64Op::Add, *da, *db, *dout, n);
  disp.wait();
  const float* p = static_cast<const float*>(dout->contents());
  auto res = from_df64(std::vector<float>(p, p + (size_t)(n * 2)));

  for (size_t i = 0; i < res.size(); ++i) CHECK(res[i] == host[i]);
}

// The stencil is the representative bandwidth-bound PDE kernel. Coefficients
// (1, -2, 1) on a smooth field is a second difference, which cancels: with grid step
// h the result is ~h^2 of the operands, so both precisions lose ~1/h^2 of their
// relative accuracy (see the tolerance note on DF64AddBeatsFloat32). h = 0.01 gives an
// amplification of 1e4, which f32 cannot survive and df64 comfortably can.
TEST(DF64StencilBeatsFloat32) {
  const int64_t n = 8192;
  MetalContext& ctx = testutil::ctx();
  Dispatcher disp(ctx);

  std::vector<double> x((size_t)n), want((size_t)n);
  for (int64_t i = 0; i < n; ++i) x[(size_t)i] = 1.0 + std::sin(0.01 * (double)i);
  for (int64_t i = 0; i < n; ++i) {
    const double l = i > 0 ? x[(size_t)(i - 1)] : 0.0;
    const double r = i + 1 < n ? x[(size_t)(i + 1)] : 0.0;
    want[(size_t)i] = l - 2.0 * x[(size_t)i] + r;
  }

  auto xv = to_df64(x);
  std::vector<double> cd{1.0, -2.0, 1.0};
  auto cv = to_df64(cd);
  auto dx = ctx.from_host(xv.data(), {n * 2}, DType::F32);
  auto dc = ctx.from_host(cv.data(), {6}, DType::F32);
  auto dout = ctx.alloc({n * 2}, DType::F32);
  df64_stencil3(df_lib(), disp, *dx, *dout, *dc, n, /*use_df64=*/true);
  disp.wait();
  const float* p = static_cast<const float*>(dout->contents());
  auto got = from_df64(std::vector<float>(p, p + (size_t)(n * 2)));

  // NORMWISE error (max|err| / max|want|), not pointwise relative: the second
  // difference of a sinusoid passes through zero, and dividing by a value that is
  // legitimately ~0 measures the zero crossing rather than the arithmetic.
  // Interior only -- the boundary rows use a different formula in the reference.
  double scale = 0.0;
  for (int64_t i = 1; i + 1 < n; ++i) scale = std::max(scale, std::abs(want[(size_t)i]));

  double e_df = 0.0, e_f32 = 0.0;
  for (int64_t i = 1; i + 1 < n; ++i) {
    const double w = want[(size_t)i];
    e_df = std::max(e_df, std::abs(got[(size_t)i] - w) / scale);
    const float lf = (float)x[(size_t)(i - 1)], mf = (float)x[(size_t)i],
                rf = (float)x[(size_t)(i + 1)];
    const double gf = (double)(lf - 2.0f * mf + rf);
    e_f32 = std::max(e_f32, std::abs(gf - w) / scale);
  }
  CHECK(e_f32 > 1e-4);        // f32 second differences are badly degraded
  CHECK(e_df < 1e-9);         // df64 keeps the digits
  CHECK(e_df * 1e4 < e_f32);  // decisively, not marginally
}
