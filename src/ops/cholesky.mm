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
#include <cstdlib>
#include <stdexcept>

namespace jaxmetal {

// Block size. Capped by THREADGROUP MEMORY, not by GEMM efficiency: the panel kernels
// hold an NB x NB block in threadgroup memory, so NB=64 is 16 KB against the 32 KB
// Apple GPUs provide. NB=128 would need 64 KB and does not fit. Must match CHOL_NB in
// kernels/cholesky.metal.
constexpr int64_t kCholNB = 64;   // default; JAXMETAL_CHOL_NB overrides for sweeps
constexpr NSUInteger kCholTG = 256;   // must match CHOL_TG

// Block-column strips per trailing update. A22 is symmetric but MPS has no SYRK, so
// one square GEMM computes both triangles and wastes half the FLOPs; P strips would
// cut the computed area from m^2 to m^2(P+1)/2P. MEASURED AND REFUTED: at N=4096,
// P=1 -> 46.1 ms, P=2 -> 46.1, P=4 -> 48.7, P=8 -> 57.6, P=16 -> 71.1. Halving the
// FLOPs does not help because the trailing GEMM is not compute-bound at rank 64 (502
// GFLOP/s at M=2048), so narrower strips land in an even less efficient regime and add
// an MPS encode each. Kept at 1, with JAXMETAL_CHOL_STRIPS to reproduce the sweep.
constexpr int64_t kCholStrips = 1;

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

    static const int64_t kNB = [] {
      if (const char* e = getenv("JAXMETAL_CHOL_NB")) return (int64_t)atoi(e);
      return kCholNB;
    }();
    for (int64_t k = 0; k < N; k += kNB) {
      const int64_t nb = std::min<int64_t>(kNB, N - k);
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
        //
        // A22 is symmetric, so only its lower triangle is needed — but MPS has no
        // SYRK, and one square GEMM computes both triangles, wasting half the FLOPs.
        // Instead the update is issued as kCholStrips block-COLUMN strips, each
        // covering only the rows at or below its first column:
        //
        //     A22[c0:m, c0:c1] -= L21[c0:m, :] · L21[c0:c1, :]ᵀ
        //
        // which walks the lower triangle in a staircase. With P strips the computed
        // area is m²(P+1)/2P against the m²/2 actually needed, so waste falls from
        // 2.00x (P=1) to 1.25x (P=4) — at the cost of P MPS encodes per block step
        // instead of one. The strict upper triangle of A22 is left stale, which is
        // safe because every later step reads only the lower triangle, and
        // chol_zero_upper clears it at the end.
        flush();
        static const int64_t kStrips = [] {
          if (const char* e = getenv("JAXMETAL_CHOL_STRIPS")) return (int64_t)atoi(e);
          return kCholStrips;
        }();
        const int64_t strips = std::min<int64_t>(kStrips, std::max<int64_t>(1, m / nb));
        const int64_t w = (m + strips - 1) / strips;
        for (int64_t c0 = 0; c0 < m; c0 += w) {
          const int64_t cw = std::min<int64_t>(w, m - c0);
          const int64_t rows = m - c0;
          MPSMatrix* lhs = window(a, k + nb + c0, k, rows, nb, N);
          MPSMatrix* rhs = window(a, k + nb + c0, k, cw, nb, N);
          MPSMatrix* dst = window(a, k + nb + c0, k + nb + c0, rows, cw, N);
          MPSMatrixMultiplication* mm =
              [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                                transposeLeft:NO
                                               transposeRight:YES
                                                   resultRows:(NSUInteger)rows
                                                resultColumns:(NSUInteger)cw
                                              interiorColumns:(NSUInteger)nb
                                                        alpha:-1.0
                                                         beta:1.0];
          [mm encodeToCommandBuffer:cmd leftMatrix:lhs rightMatrix:rhs resultMatrix:dst];
        }
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
