#include "jaxmetal/ops/reduce.h"

#include "jaxmetal/metal/dtype.h"
#include "jaxmetal/metal/kernel_library.h"
#include "jaxmetal/metal/metal_buffer.h"
#include "jaxmetal/metal/metal_context.h"
#include "jaxmetal/runtime/dispatcher.h"

#include "kernels_reduce.h"   // generated: jaxmetal::kernels::reduce_msl

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace jaxmetal {
namespace {
// Byte-match ReduceDims in kernels/reduce.metal.
struct ReduceDims { uint32_t n; uint32_t stride; };
}  // namespace

// Threadgroup width; must match RED_TG in kernels/reduce.metal, which sizes its
// threadgroup array and its tree reduction to exactly this.
constexpr int64_t kReduceTG = 256;

// Threadgroups in pass 1. Each grid-strides the input and emits one (sum,
// compensation) pair, so this is both the parallel width and the length of the
// array pass 2 reduces -- it must stay <= kReduceTG * (elements one threadgroup can
// stride), and small enough that a single threadgroup can finish it.
constexpr int64_t kReduceGroups = 512;

void register_reduce_kernels(KernelLibrary& lib) {
  lib.add_source("reduce", kernels::reduce_msl);
}

float reduce_sum_f32(MetalContext& ctx, KernelLibrary& lib, Dispatcher& disp,
                     MetalBuffer& x, int64_t n, bool compensated) {
  if (n <= 0) return 0.0f;

  // Never launch more threadgroups than there is work for; a threadgroup whose
  // grid-stride loop never executes would still write an (uninitialised-looking)
  // zero partial, which is harmless but wasteful.
  const int64_t groups =
      std::min<int64_t>(kReduceGroups, (n + kReduceTG - 1) / kReduceTG);

  // Pass 1 emits one partial per threadgroup: float2 (sum, compensation) when
  // compensated, a bare float otherwise.
  auto partials = ctx.alloc({groups * (compensated ? 2 : 1)}, DType::F32);
  auto out = ctx.alloc({1}, DType::F32);

  ReduceDims d1{static_cast<uint32_t>(n), 0};
  std::vector<MetalBuffer*> b1 = {&x, partials.get()};
  disp.dispatch_threadgroups(
      lib.pipeline(compensated ? "reduce_sum_comp_partials"
                               : "reduce_sum_tree_partials"),
      b1, /*groups_x=*/groups, 1, 1, /*tg_x=*/kReduceTG, 1, 1, &d1, sizeof(d1));

  // Pass 2: one threadgroup folds the `groups` partials into a scalar. Same queue
  // as pass 1, so it is ordered after it.
  ReduceDims d2{static_cast<uint32_t>(groups), 0};
  std::vector<MetalBuffer*> b2 = {partials.get(), out.get()};
  disp.dispatch_threadgroups(
      lib.pipeline(compensated ? "reduce_sum_comp_finish"
                               : "reduce_sum_tree_finish"),
      b2, /*groups_x=*/1, 1, 1, /*tg_x=*/kReduceTG, 1, 1, &d2, sizeof(d2));

  disp.wait();
  return *static_cast<const float*>(out->contents());
}

void reduce_sum_axis0_comp_into(KernelLibrary& lib, Dispatcher& disp,
                                MetalBuffer& a, MetalBuffer& out,
                                int64_t M, int64_t N) {
  if (M <= 0 || N <= 0) return;
  ReduceDims d{static_cast<uint32_t>(M), static_cast<uint32_t>(N)};
  std::vector<MetalBuffer*> bufs = {&a, &out};
  // One threadgroup of exactly kReduceTG threads per output column.
  disp.dispatch_threadgroups(lib.pipeline("reduce_sum_axis0_comp"), bufs,
                             /*groups_x=*/N, 1, 1,
                             /*tg_x=*/kReduceTG, 1, 1, &d, sizeof(d));
}

}  // namespace jaxmetal
