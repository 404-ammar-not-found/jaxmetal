#pragma once
#include <cstdint>

namespace jaxmetal {

class MetalContext;
class MetalBuffer;

// C[M,N] = A[M,K] @ B[K,N] via Apple's MetalPerformanceShaders
// (MPSMatrixMultiplication) — the same professionally-tuned GPU matmul PyTorch's
// MPS backend uses. Much faster than our hand-written matmul_blocked (which we
// keep as the from-scratch learning kernel). Operands are GPU-resident
// MetalBuffers; this encodes, commits, and waits for completion.
void mps_matmul_into(MetalContext& ctx, MetalBuffer& a, MetalBuffer& b, MetalBuffer& out,
                     int64_t M, int64_t K, int64_t N);

}  // namespace jaxmetal
