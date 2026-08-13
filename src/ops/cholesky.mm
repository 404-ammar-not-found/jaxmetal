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
#include <string>
#include <stdexcept>

namespace jaxmetal {

// Block size. Capped by THREADGROUP MEMORY, not by GEMM efficiency: the panel kernels
// hold an NB x NB block in threadgroup memory, so NB=64 is 16 KB against the 32 KB
// Apple GPUs provide. NB=128 would need 64 KB and does not fit. Must match CHOL_NB in
// kernels/cholesky.metal.
constexpr int64_t kCholNB = 64;   // default; JAXMETAL_CHOL_NB overrides for sweeps
constexpr NSUInteger kCholTG = 256;         // must match CHOL_TG
constexpr NSUInteger kCholPanelTG = 32;     // must match CHOL_PANEL_TG: exactly one
                                            // SIMD group, so the panel's per-column
                                            // barrier is a simdgroup_barrier

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
  static const bool kChunkedTrsm = [] {
    if (const char* e = getenv("JAXMETAL_CHOL_TRSM_CHUNKED")) return atoi(e) != 0;
    return false;
  }();
  id<MTLComputePipelineState> pso_trsm =
      pso(kChunkedTrsm ? "chol_trsm_right_chunked" : "chol_trsm_right");
  id<MTLComputePipelineState> pso_zero = pso("chol_zero_upper");

  // TRSM strategy: "gemm" inverts the diagonal block and turns the panel solve into
  // an MPS GEMM; "kernel" uses the hand-written per-row substitution.
  static const bool kGemmTrsm = [] {
    if (const char* e = getenv("JAXMETAL_CHOL_TRSM")) return std::string(e) == "gemm";
    return true;
  }();
  id<MTLComputePipelineState> pso_inv = pso("chol_invert_panel");
  id<MTLComputePipelineState> pso_copy = pso("chol_copy_panel");

  // Scratch: the nb x nb inverse, and the m x nb GEMM result (which cannot alias its
  // own input, so it is written here and copied back).
  auto linv = ctx.alloc({kCholNB, kCholNB}, DType::F32);
  auto panel = ctx.alloc({N, kCholNB}, DType::F32);
  id<MTLBuffer> bl = (__bridge id<MTLBuffer>)linv->mtl_handle();
  id<MTLBuffer> bp = (__bridge id<MTLBuffer>)panel->mtl_handle();

  auto status = ctx.alloc({1}, DType::I32);
  *static_cast<uint32_t*>(status->contents()) = 0u;   // nothing in flight yet
  id<MTLBuffer> st = (__bridge id<MTLBuffer>)status->mtl_handle();

  // Per-phase GPU-time attribution. Puts each phase in its own command buffer and
  // sums MTLCommandBuffer GPUStartTime/GPUEndTime, which is the GPU's own clock. This
  // exists because the cheaper JAXMETAL_CHOL_PHASES approach (skip a phase, time the
  // rest) is NOT additive -- its numbers summed to more than the whole -- and six
  // optimisations were attempted against the wrong diagnosis as a result. Slower than
  // the real path because it adds a round trip per phase; only the ratios are useful.
  static const bool kProfilePhases = [] {
    const char* e = getenv("JAXMETAL_CHOL_PROFILE");
    return e && atoi(e) != 0;
  }();
  double g_panel = 0.0, g_trsm = 0.0, g_gemm = 0.0;

  @autoreleasepool {
    // ONE command buffer for every block step. Metal hazard-tracks within a command
    // buffer, so the panel -> trsm -> GEMM read-after-write chain is ordered without
    // explicit barriers, and the driver round trip is paid once for the whole
    // factorisation instead of once per block.
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = nil;
    auto begin = [&]() { if (!enc) enc = [cmd computeCommandEncoder]; };
    auto flush = [&]() { if (enc) { [enc endEncoding]; enc = nil; } };

    // Phase mask for attribution only: "ptg" = panel/trsm/gemm. Dropping a phase
    // produces WRONG results; it exists solely to time where the runtime goes, after
    // two rounds of reasoning about the bottleneck turned out to be wrong.
    static const std::string kPhases = [] {
      const char* e = getenv("JAXMETAL_CHOL_PHASES");
      return std::string(e ? e : "ptg");
    }();
    const bool do_panel = kPhases.find('p') != std::string::npos;
    const bool do_trsm  = kPhases.find('t') != std::string::npos;
    const bool do_gemm  = kPhases.find('g') != std::string::npos;

    // Hand-written panel TRSM vs MPSMatrixSolveTriangular. MEASURED AND REFUTED:
    // at N=4096 the MPS path is 62.2 ms against 47.3 ms for the hand kernel. The
    // earlier dismissal of this API (150 ms at order=4096, serial in the order) was
    // for the wrong regime, and a standalone probe at order=64 looked promising at
    // ~0.26 ms per call -- but that fixed ~0.25 ms is real GPU work, not command
    // buffer round trip, so 64 invocations cost more than the hand kernel total.
    // Kept behind JAXMETAL_CHOL_MPS_TRSM=1 so the comparison is reproducible.
    static const bool kMpsTrsm = [] {
      if (const char* e = getenv("JAXMETAL_CHOL_MPS_TRSM")) return atoi(e) != 0;
      return false;
    }();

    static const int64_t kNB = [] {
      if (const char* e = getenv("JAXMETAL_CHOL_NB")) return (int64_t)atoi(e);
      return kCholNB;
    }();
    auto phase_end = [&](double* acc) {
      if (!kProfilePhases) return;
      flush();
      [cmd commit];
      [cmd waitUntilCompleted];
      *acc += cmd.GPUEndTime - cmd.GPUStartTime;
      cmd = [queue commandBuffer];
    };

    for (int64_t k = 0; k < N; k += kNB) {
      const int64_t nb = std::min<int64_t>(kNB, N - k);
      const int64_t m = N - k - nb;
      CholDims d{(uint32_t)N, (uint32_t)k, (uint32_t)nb, (uint32_t)m};

      // (1) factor the diagonal block, one threadgroup.
      if (do_panel) {
      begin();
      [enc setComputePipelineState:pso_panel];
      [enc setBuffer:a offset:0 atIndex:0];
      [enc setBytes:&d length:sizeof(d) atIndex:1];
      [enc setBuffer:st offset:0 atIndex:2];
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kCholTG, 1, 1)];
      }
      phase_end(&g_panel);

      if (m > 0) {
        if (do_trsm) {
        if (kGemmTrsm && nb == kCholNB) {
          // Panel solve as a GEMM: invert L11 once, then L21 = A21 * (L11^-1)^T.
          // The solve was 65% of the factorisation at ~19 GFLOP/s; the GEMM phase
          // measured ~0.5%. Five attempts to speed up the solve directly all measured
          // flat or worse, so this removes it instead of optimising it.
          begin();
          [enc setComputePipelineState:pso_inv];
          [enc setBuffer:a offset:0 atIndex:0];
          [enc setBuffer:bl offset:0 atIndex:1];
          [enc setBytes:&d length:sizeof(d) atIndex:2];
          [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(kCholTG, 1, 1)];
          flush();

          MPSMatrix* a21 = window(a, k + nb, k, m, nb, N);
          MPSMatrix* mli = window(bl, 0, 0, nb, nb, nb);
          MPSMatrix* mpanel = window(bp, 0, 0, m, nb, nb);
          MPSMatrixMultiplication* mm =
              [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                                transposeLeft:NO
                                               transposeRight:YES
                                                   resultRows:(NSUInteger)m
                                                resultColumns:(NSUInteger)nb
                                              interiorColumns:(NSUInteger)nb
                                                        alpha:1.0
                                                         beta:0.0];
          [mm encodeToCommandBuffer:cmd leftMatrix:a21 rightMatrix:mli resultMatrix:mpanel];

          begin();
          [enc setComputePipelineState:pso_copy];
          [enc setBuffer:bp offset:0 atIndex:0];
          [enc setBuffer:a offset:0 atIndex:1];
          [enc setBytes:&d length:sizeof(d) atIndex:2];
          [enc dispatchThreads:MTLSizeMake((NSUInteger)(m * nb), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(kCholTG, 1, 1)];
        } else if (kMpsTrsm) {
          // MPSMatrixSolveTriangular at ORDER=nb (64) with m right-hand sides.
          // This API was dismissed earlier on a measurement at order=4096, where it
          // is serial in the order and takes 150 ms. That was the wrong regime: the
          // panel solve is order=64, and measured there it is ~0.1 ms for m=4096,
          // roughly 4x faster than the hand-written kernel on the phase that is 65%
          // of the whole factorisation.
          flush();
          MPSMatrix* l11 = window(a, k, k, nb, nb, N);
          MPSMatrix* a21 = window(a, k + nb, k, m, nb, N);
          MPSMatrixSolveTriangular* ts =
              [[MPSMatrixSolveTriangular alloc] initWithDevice:dev
                                                         right:YES
                                                         upper:NO
                                                     transpose:YES
                                                          unit:NO
                                                         order:(NSUInteger)nb
                                        numberOfRightHandSides:(NSUInteger)m
                                                         alpha:1.0];
          // Solves in place: rightHandSide and solution are the same window. Apple
          // does not document whether that is legal, exactly as with the GEMM
          // aliasing below -- CholeskyMatchesLapack is what establishes it.
          [ts encodeToCommandBuffer:cmd sourceMatrix:l11 rightHandSideMatrix:a21
                     solutionMatrix:a21];
        } else {
        begin();
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
        }
        }

        // (3) trailing update A22 -= L21 · L21ᵀ, via MPS. This is where all the FLOPs
        // are. Both operands and the result are disjoint windows of the SAME buffer;
        // MPS documents in-place only for the decomposition kernels and is silent
        // about GEMM aliasing, so CholeskyMatchesLapack is what establishes that
        // disjoint windows are safe here.
        //
        phase_end(&g_trsm);

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
        phase_end(&g_gemm);
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

  if (kProfilePhases) {
    fprintf(stderr, "[chol N=%lld] panel=%.1fms trsm=%.1fms gemm=%.1fms\n",
            (long long)N, g_panel * 1e3, g_trsm * 1e3, g_gemm * 1e3);
  }
  return (int)*static_cast<const uint32_t*>(status->contents());
}

}  // namespace jaxmetal
