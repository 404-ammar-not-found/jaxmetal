#include "jaxmetal/cpu/cpu_matmul.h"

#include <Accelerate/Accelerate.h>

namespace jaxmetal {

void cpu_matmul_f32(const float* A, const float* B, float* C,
                    int64_t M, int64_t K, int64_t N) {
  // C = 1.0 * A * B + 0.0 * C, all row-major, no transpose.
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
              1.0f, A, static_cast<int>(K),
              B, static_cast<int>(N),
              0.0f, C, static_cast<int>(N));
}

}  // namespace jaxmetal
