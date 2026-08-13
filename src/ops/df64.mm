#include "jaxmetal/ops/df64.h"

#include "jaxmetal/metal/kernel_library.h"
#include "jaxmetal/metal/metal_buffer.h"
#include "jaxmetal/runtime/dispatcher.h"

#include "kernels_df64.h"   // generated: jaxmetal::kernels::df64_msl

#include <cstdint>
#include <vector>

namespace jaxmetal {
namespace {
struct DFDims { uint32_t n; };
}  // namespace

void register_df64_kernels(KernelLibrary& lib) {
  lib.add_source("df64", kernels::df64_msl);
}

void df64_elementwise(KernelLibrary& lib, Dispatcher& disp, DF64Op op,
                      MetalBuffer& a, MetalBuffer& b, MetalBuffer& out, int64_t n) {
  if (n <= 0) return;
  const char* name = op == DF64Op::Add ? "df64_add"
                   : op == DF64Op::Mul ? "df64_mul"
                                       : "df64_div";
  DFDims d{static_cast<uint32_t>(n)};
  std::vector<MetalBuffer*> bufs = {&a, &b, &out};
  disp.dispatch_1d(lib.pipeline(name), bufs, n, &d, sizeof(d));
}

void df64_stencil3(KernelLibrary& lib, Dispatcher& disp, MetalBuffer& x,
                   MetalBuffer& out, MetalBuffer& coef, int64_t n, bool use_df64) {
  if (n <= 0) return;
  DFDims d{static_cast<uint32_t>(n)};
  std::vector<MetalBuffer*> bufs = {&x, &out, &coef};
  disp.dispatch_1d(lib.pipeline(use_df64 ? "df64_stencil3" : "f32_stencil3"),
                   bufs, n, &d, sizeof(d));
}

}  // namespace jaxmetal
