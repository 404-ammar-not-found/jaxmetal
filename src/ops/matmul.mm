#include "jaxmetal/ops/matmul.h"

#include "jaxmetal/metal/dtype.h"
#include "jaxmetal/metal/kernel_library.h"
#include "jaxmetal/metal/metal_buffer.h"
#include "jaxmetal/metal/metal_context.h"
#include "jaxmetal/runtime/dispatcher.h"

#include "kernels_matmul.h"  // generated: jaxmetal::kernels::matmul_msl

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace jaxmetal {

// Must match matmul_blocked's BM/BN/TX/TY in kernels/matmul.metal.
static constexpr int64_t kBlockM = 64;
static constexpr int64_t kBlockN = 64;
static constexpr int64_t kThreadsX = 16;  // = BN/TN
static constexpr int64_t kThreadsY = 16;  // = BM/TM

namespace {
struct MatMulDims {
  uint32_t M;
  uint32_t N;
  uint32_t K;
};
int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }
}  // namespace

void register_matmul_kernel(KernelLibrary& lib) {
  lib.add_source("matmul", kernels::matmul_msl);
}

void matmul_into(KernelLibrary& lib, Dispatcher& disp, MetalBuffer& a, MetalBuffer& b,
                 MetalBuffer& out, int64_t M, int64_t K, int64_t N) {
  void* pso = lib.pipeline("matmul_blocked");
  MatMulDims dims{static_cast<uint32_t>(M), static_cast<uint32_t>(N), static_cast<uint32_t>(K)};
  std::vector<MetalBuffer*> bufs = {&a, &b, &out};
  // One BM x BN output tile per threadgroup, laid out as a TX x TY thread grid.
  disp.dispatch_threadgroups(pso, bufs,
                             /*groups_x=*/ceil_div(N, kBlockN),
                             /*groups_y=*/ceil_div(M, kBlockM),
                             /*groups_z=*/1,
                             /*tg_x=*/kThreadsX, /*tg_y=*/kThreadsY, /*tg_z=*/1,
                             &dims, sizeof(dims));
}

std::shared_ptr<MetalBuffer> matmul(MetalContext& ctx, KernelLibrary& lib, Dispatcher& disp,
                                    MetalBuffer& a, MetalBuffer& b) {
  if (a.shape().size() != 2 || b.shape().size() != 2)
    throw std::runtime_error("matmul: both operands must be rank-2");
  if (a.dtype() != DType::F32 || b.dtype() != DType::F32)
    throw std::runtime_error("matmul: only f32 is supported");

  const int64_t M = a.shape()[0];
  const int64_t K = a.shape()[1];
  const int64_t N = b.shape()[1];
  if (b.shape()[0] != K)
    throw std::runtime_error("matmul: inner dimensions do not match (A[M,K] @ B[K,N])");

  auto out = ctx.alloc({M, N}, DType::F32);
  matmul_into(lib, disp, a, b, *out, M, K, N);
  return out;
}

}  // namespace jaxmetal
