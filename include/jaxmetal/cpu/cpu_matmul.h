#pragma once
#include <cstdint>

namespace jaxmetal {

// C[M,N] = A[M,K] @ B[K,N], row-major f32, on the CPU via Apple's Accelerate
// (cblas_sgemm — AMX-accelerated BLAS). This is the CPU arm of the auto router.
void cpu_matmul_f32(const float* A, const float* B, float* C,
                    int64_t M, int64_t K, int64_t N);

}  // namespace jaxmetal
