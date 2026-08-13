#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "jaxmetal/ops/cholesky.h"

#include "jaxmetal/metal/dtype.h"
#include "jaxmetal/metal/kernel_library.h"
#include "jaxmetal/metal/metal_buffer.h"
#include "jaxmetal/metal/metal_context.h"

#include "kernels_cholesky.h"   // generated: jaxmetal::kernels::cholesky_msl

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace jaxmetal {

// Block size. Capped by THREADGROUP MEMORY, not by GEMM efficiency: the panel kernels
// hold an NB x NB block in threadgroup memory, so NB=64 is 16 KB against the 32 KB
// Apple GPUs provide. NB=128 would need 64 KB and does not fit. Must match CHOL_NB in
// kernels/cholesky.metal.
constexpr int64_t kCholNB = 64;
constexpr NSUInteger kCholTG = 256;   // must match CHOL_TG

namespace {
struct CholDims { uint32_t n, k, nb, m; };

// A row-major [rows, cols] f32 window into a larger matrix of leading dimension `ld`.
// rowBytes stays ld*4 so the view is a genuine submatrix, not a copy.
MPSMatrix* window(id<MTLBuffer> buf, int64_t row, int64_t col, int64_t rows,
                  int64_t cols, int64_t ld) {
  MPSMatrixDescriptor* d =
      [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)rows
                                            columns:(NSUInteger)cols
                                           rowBytes:(NSUInteger)ld * sizeof(float)
                                           dataType:MPSDataTypeFloat32];
  return [[MPSMatrix alloc] initWithBuffer:buf
                                    offset:(NSUInteger)((row * ld + col) * (int64_t)sizeof(float))
                                descriptor:d];
}
}  // namespace

void register_cholesky_kernels(KernelLibrary& lib) {
  lib.add_source("cholesky", kernels::cholesky_msl);
}

int cholesky_f32(MetalContext& ctx, KernelLibrary& lib, MetalBuffer& A, int64_t N) {
  if (N <= 0) return 0;

  id<MTLDevice> dev = (__bridge id<MTLDevice>)ctx.device_handle();
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ctx.queue_handle();
  id<MTLBuffer> a = (__bridge id<MTLBuffer>)A.mtl_handle();

  auto pso = [&](const char* name) {
    return (__bridge id<MTLComputePipelineState>)lib.pipeline(name);
  };
  id<MTLComputePipelineState> pso_panel = pso("chol_panel");
  id<MTLComputePipelineState> pso_trsm = pso("chol_trsm_right");
  id<MTLComputePipelineState> pso_zero = pso("chol_zero_upper");

  auto status = ctx.alloc({1}, DType::I32);
  *static_cast<uint32_t*>(status->contents()) = 0u;   // nothing in flight yet
  id<MTLBuffer> st = (__bridge id<MTLBuffer>)status->mtl_handle();

  @autoreleasepool {
    // ONE command buffer for every block step. Metal hazard-tracks within a command
    // buffer, so the panel -> trsm -> GEMM read-after-write chain is ordered without
    // explicit barriers, and the driver round trip is paid once for the whole
    // factorisation instead of once per block.
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = nil;
    auto begin = [&]() { if (!enc) enc = [cmd computeCommandEncoder]; };
    auto flush = [&]() { if (enc) { [enc endEncoding]; enc = nil; } };

    for (int64_t k = 0; k < N; k += kCholNB) {
      const int64_t nb = std::min<int64_t>(kCholNB, N - k);
      const int64_t m = N - k - nb;
      CholDims d{(uint32_t)N, (uint32_t)k, (uint32_t)nb, (uint32_t)m};

      // (1) factor the diagonal block, one threadgroup.
      begin();
      [enc setComputePipelineState:pso_panel];
      [enc setBuffer:a offset:0 atIndex:0];
      [enc setBytes:&d length:sizeof(d) atIndex:1];
      [enc setBuffer:st offset:0 atIndex:2];
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kCholTG, 1, 1)];

      if (m > 0) {
        // (2) off-diagonal panel: L21 = A21 · L11⁻ᵀ. One thread per row, so the m
        // rows are independent — this is where the panel-phase parallelism lives.
        // Deliberately NOT MPSMatrixSolveTriangular: that is serial in its order and
        // measured 150 ms at order=4096, but here the order is only nb=64 with m
        // independent right-hand sides.
        [enc setComputePipelineState:pso_trsm];
        [enc setBuffer:a offset:0 atIndex:0];
        [enc setBytes:&d length:sizeof(d) atIndex:1];
        const NSUInteger tg = std::min<NSUInteger>(kCholTG, (NSUInteger)m);
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((m + kCholTG - 1) / kCholTG), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kCholTG, 1, 1)];
        (void)tg;

        // (3) trailing update A22 -= L21 · L21ᵀ, via MPS. This is where all the FLOPs
        // are. Both operands and the result are disjoint windows of the SAME buffer;
        // MPS documents in-place only for the decomposition kernels and is silent
        // about GEMM aliasing, so CholeskyMatchesLapack is what establishes that
        // disjoint windows are safe here.
        flush();
        MPSMatrix* l21 = window(a, k + nb, k, m, nb, N);
        MPSMatrix* a22 = window(a, k + nb, k + nb, m, m, N);
        MPSMatrixMultiplication* mm =
            [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                              transposeLeft:NO
                                             transposeRight:YES
                                                 resultRows:(NSUInteger)m
                                              resultColumns:(NSUInteger)m
                                            interiorColumns:(NSUInteger)nb
                                                      alpha:-1.0
                                                       beta:1.0];
        [mm encodeToCommandBuffer:cmd leftMatrix:l21 rightMatrix:l21 resultMatrix:a22];
      }
    }

    // Leave a clean L rather than L over the caller's original upper triangle.
    CholDims dz{(uint32_t)N, 0, 0, 0};
    begin();
    [enc setComputePipelineState:pso_zero];
    [enc setBuffer:a offset:0 atIndex:0];
    [enc setBytes:&dz length:sizeof(dz) atIndex:1];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)(N * N), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kCholTG, 1, 1)];
    flush();

    [cmd commit];
    [cmd waitUntilCompleted];
  }

  return (int)*static_cast<const uint32_t*>(status->contents());
}

}  // namespace jaxmetal
