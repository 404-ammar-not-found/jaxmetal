// Tests for compensated (Neumaier) f32 summation vs double-precision references.
//
// The interesting property is not "the sum is roughly right" -- an uncompensated
// tree sum is already roughly right. It is that the compensated version stays
// accurate on inputs where the tree sum measurably loses digits, and the test that
// pins that down (ReduceCompensatedBeatsNaive) is deliberately adversarial.
//
// It is also the regression test for the safe-math dependency: every compensation
// term in kernels/reduce.metal is algebraically zero, so a fast-math build would
// fold it away and silently turn the compensated kernel back into a tree sum. That
// failure is invisible except as a loss of accuracy, which is exactly what these
// tests measure.

#include "test_framework.h"
#include "gpu_test_util.h"

#include "jaxmetal/metal/metal_context.h"
#include "jaxmetal/metal/metal_buffer.h"
#include "jaxmetal/metal/kernel_library.h"
#include "jaxmetal/metal/dtype.h"
#include "jaxmetal/runtime/dispatcher.h"
#include "jaxmetal/ops/reduce.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace jaxmetal;

namespace {

KernelLibrary& reduce_lib() {
  KernelLibrary& l = testutil::lib();
  static int once = (register_reduce_kernels(l), 0);
  (void)once;
  return l;
}

// Ground truth: double accumulation. For the magnitudes used here every partial
// sum is exactly representable in f64, so this is the exact answer.
double exact_sum(const std::vector<float>& v) {
  double s = 0.0;
  for (float x : v) s += static_cast<double>(x);
  return s;
}

double rel_err(double got, double want) {
  if (want == 0.0) return std::abs(got);
  return std::abs(got - want) / std::abs(want);
}

float gpu_sum(const std::vector<float>& v, bool compensated) {
  MetalContext& ctx = testutil::ctx();
  Dispatcher disp(ctx);
  auto buf = ctx.from_host(v.data(), {static_cast<int64_t>(v.size())}, DType::F32);
  return reduce_sum_f32(ctx, reduce_lib(), disp, *buf,
                        static_cast<int64_t>(v.size()), compensated);
}

}  // namespace

// Well-conditioned input: both kernels should be fine. Catches gross errors in the
// grid-stride/tree plumbing (dropped elements, double-counted partials) that an
// accuracy-only test could mask.
TEST(ReduceSumMatchesReference) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  std::vector<float> v(1 << 20);
  for (auto& x : v) x = d(rng);

  const double want = exact_sum(v);
  CHECK(rel_err(gpu_sum(v, /*compensated=*/true), want) < 1e-5);
  CHECK(rel_err(gpu_sum(v, /*compensated=*/false), want) < 1e-4);
}

// Every element counted exactly once, with a value whose sum is exactly
// representable: any indexing bug shows up as an integer-sized discrepancy rather
// than a rounding difference.
TEST(ReduceSumCountsEveryElement) {
  std::vector<float> v(1000000, 1.0f);
  CHECK_NEAR(gpu_sum(v, true), 1000000.0f, 0.0f);   // exact
  CHECK_NEAR(gpu_sum(v, false), 1000000.0f, 0.0f);
}

// Sizes that are not multiples of the threadgroup width or the grid stride, plus
// the degenerate ones. The grid-stride loop and the `groups` clamp both have to
// handle a tail here.
TEST(ReduceSumHandlesRaggedSizes) {
  for (int64_t n : {1, 2, 3, 255, 257, 1023, 1025, 131073}) {
    std::vector<float> v(static_cast<size_t>(n), 1.0f);
    CHECK_NEAR(gpu_sum(v, true), static_cast<float>(n), 0.0f);
    CHECK_NEAR(gpu_sum(v, false), static_cast<float>(n), 0.0f);
  }
}

// THE test. A large value in every thread's first grid-stride slot, then many
// values far below its ulp. ulp(1e8) in f32 is 8, so each subsequent +1.0 rounds
// away completely and an uncompensated sum loses the entire tail -- and the tree
// structure cannot help, because the loss happens inside one thread's sequential
// accumulation, before any tree combining.
//
// Compensation must recover the tail; if it does not, either the algorithm is wrong
// or fast math folded the compensation terms away.
TEST(ReduceCompensatedBeatsNaive) {
  // Must match kReduceGroups * kReduceTG in src/ops/reduce.mm: this is the grid
  // stride, so the first `gsz` elements are exactly one per thread.
  constexpr int64_t gsz = 512 * 256;
  constexpr int64_t n = 1 << 24;
  static_assert(n > gsz, "need many small values after each large one");

  std::vector<float> v(static_cast<size_t>(n), 1.0f);
  for (int64_t i = 0; i < gsz; ++i) v[static_cast<size_t>(i)] = 1e8f;

  const double want = exact_sum(v);
  const double e_comp = rel_err(gpu_sum(v, /*compensated=*/true), want);
  const double e_tree = rel_err(gpu_sum(v, /*compensated=*/false), want);

  // The tree sum really does lose the tail here; if it did not, the test would be
  // proving nothing and the thresholds below would be vacuous.
  CHECK(e_tree > 1e-7);
  // Compensation recovers it to near the representable limit of the f32 result.
  CHECK(e_comp < 1e-7);
  CHECK(e_comp * 10.0 < e_tree);
}

// Neumaier's branch (as opposed to classic Kahan) is what makes a large value
// arriving *after* a small running sum recoverable. Reversing the layout above puts
// the large value last in each thread's run, which classic Kahan gets wrong.
TEST(ReduceCompensatedHandlesLateLargeValues) {
  constexpr int64_t gsz = 512 * 256;
  constexpr int64_t n = 1 << 24;

  std::vector<float> v(static_cast<size_t>(n), 1.0f);
  for (int64_t i = n - gsz; i < n; ++i) v[static_cast<size_t>(i)] = 1e8f;

  const double want = exact_sum(v);
  CHECK(rel_err(gpu_sum(v, /*compensated=*/true), want) < 1e-7);
}

// Compensated axis-0 reduction vs per-column double sums, on the same adversarial
// magnitude spread. This is the accuracy-preserving counterpart of
// nn_reduce_sum_axis0, so it must agree column for column.
TEST(ReduceSumAxis0Compensated) {
  constexpr int64_t M = 4096, N = 7;   // N deliberately not a nice power of two
  std::vector<float> a(static_cast<size_t>(M * N), 1.0f);
  for (int64_t j = 0; j < N; ++j) a[static_cast<size_t>(j)] = 1e8f;  // row 0 large

  MetalContext& ctx = testutil::ctx();
  Dispatcher disp(ctx);
  auto da = ctx.from_host(a.data(), {M, N}, DType::F32);
  auto dout = ctx.alloc({N}, DType::F32);
  reduce_sum_axis0_comp_into(reduce_lib(), disp, *da, *dout, M, N);
  disp.wait();

  const float* got = static_cast<const float*>(dout->contents());
  for (int64_t j = 0; j < N; ++j) {
    double want = 0.0;
    for (int64_t i = 0; i < M; ++i) want += static_cast<double>(a[static_cast<size_t>(i * N + j)]);
    CHECK(rel_err(got[j], want) < 1e-7);
  }
}
