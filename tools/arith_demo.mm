// Stage 0/1 smoke test: send arithmetic instructions to the GPU and verify the
// results against a CPU reference. Exits non-zero on any mismatch.

#include "jaxmetal/metal/metal_context.h"
#include "jaxmetal/metal/metal_buffer.h"
#include "jaxmetal/metal/kernel_library.h"
#include "jaxmetal/runtime/dispatcher.h"
#include "jaxmetal/ops/elementwise.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace jaxmetal;

int main() {
  MetalContext& ctx = MetalContext::instance();
  std::printf("Metal device: %s\n\n", ctx.device_name().c_str());

  KernelLibrary lib(ctx);
  register_elementwise_kernels(lib);
  Dispatcher disp(ctx);

  const int64_t n = 4096;
  std::vector<float> ha(n), hb(n), hc(n);
  for (int64_t i = 0; i < n; ++i) {
    ha[i] = static_cast<float>(i) * 0.5f - 100.0f;
    hb[i] = static_cast<float>(n - i) * 0.25f + 1.0f;  // never 0 -> safe div
    hc[i] = static_cast<float>(i % 200) / 100.0f - 1.0f;  // [-1, 1) -> safe exp
  }
  auto a = ctx.from_host(ha.data(), {n}, DType::F32);
  auto b = ctx.from_host(hb.data(), {n}, DType::F32);
  auto c = ctx.from_host(hc.data(), {n}, DType::F32);

  int failures = 0;
  const float tol = 1e-5f;

  auto check = [&](const char* name, const std::shared_ptr<MetalBuffer>& out,
                   auto ref) {
    disp.wait();
    const float* ho = static_cast<const float*>(out->contents());
    double max_err = 0.0;
    for (int64_t i = 0; i < n; ++i) {
      double e = std::fabs(static_cast<double>(ho[i]) - static_cast<double>(ref(i)));
      if (e > max_err) max_err = e;
    }
    bool ok = max_err <= tol;
    std::printf("  %-5s max_err=%.3e  %s\n", name, max_err, ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
  };

  // Binary arithmetic instructions.
  check("add", elementwise_binary(ctx, lib, disp, BinaryOp::Add, *a, *b),
        [&](int64_t i) { return ha[i] + hb[i]; });
  check("sub", elementwise_binary(ctx, lib, disp, BinaryOp::Sub, *a, *b),
        [&](int64_t i) { return ha[i] - hb[i]; });
  check("mul", elementwise_binary(ctx, lib, disp, BinaryOp::Mul, *a, *b),
        [&](int64_t i) { return ha[i] * hb[i]; });
  check("div", elementwise_binary(ctx, lib, disp, BinaryOp::Div, *a, *b),
        [&](int64_t i) { return ha[i] / hb[i]; });
  check("max", elementwise_binary(ctx, lib, disp, BinaryOp::Max, *a, *b),
        [&](int64_t i) { return std::fmax(ha[i], hb[i]); });
  check("min", elementwise_binary(ctx, lib, disp, BinaryOp::Min, *a, *b),
        [&](int64_t i) { return std::fmin(ha[i], hb[i]); });

  // Unary arithmetic instructions.
  check("neg", elementwise_unary(ctx, lib, disp, UnaryOp::Neg, *a),
        [&](int64_t i) { return -ha[i]; });
  check("abs", elementwise_unary(ctx, lib, disp, UnaryOp::Abs, *a),
        [&](int64_t i) { return std::fabs(ha[i]); });
  check("exp", elementwise_unary(ctx, lib, disp, UnaryOp::Exp, *c),
        [&](int64_t i) { return std::exp(hc[i]); });

  std::printf("\n%s (%d failure%s)\n", failures ? "RESULT: FAIL" : "RESULT: PASS",
              failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
