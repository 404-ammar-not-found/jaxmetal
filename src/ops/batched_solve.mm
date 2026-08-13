#include "jaxmetal/ops/batched_solve.h"

#include "jaxmetal/metal/dtype.h"
#include "jaxmetal/metal/kernel_library.h"
#include "jaxmetal/metal/metal_buffer.h"
#include "jaxmetal/runtime/dispatcher.h"

#include "kernels_batched_solve.h"   // generated: jaxmetal::kernels::batched_solve_msl

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace jaxmetal {
namespace {
struct BSolveDims { uint32_t batch; };
}  // namespace

void register_batched_solve_kernels(KernelLibrary& lib) {
  lib.add_source("batched_solve", kernels::batched_solve_msl);
}

void batched_solve_f32(KernelLibrary& lib, Dispatcher& disp,
                       MetalBuffer& A, MetalBuffer& rhs, MetalBuffer& x,
                       MetalBuffer& pivmin, int64_t batch, int64_t n) {
  if (n < 2 || n > kBatchedSolveMaxN)
    throw std::runtime_error("batched_solve: n must be in [2, " +
                             std::to_string(kBatchedSolveMaxN) + "]");
  if (batch <= 0) return;

  // One kernel per size so every index into the register-resident matrix is a
  // compile-time constant; see the header comment in kernels/batched_solve.metal.
  void* pso = lib.pipeline(("batched_solve_" + std::to_string(n)).c_str());
  BSolveDims d{static_cast<uint32_t>(batch)};
  std::vector<MetalBuffer*> bufs = {&A, &rhs, &x, &pivmin};
  disp.dispatch_1d(pso, bufs, batch, &d, sizeof(d));
}

void batched_solve_cpu_f32(const float* A, const float* rhs, float* x,
                           int64_t batch, int64_t n) {
  std::vector<float> a((size_t)(n * n)), b((size_t)n);
  for (int64_t s = 0; s < batch; ++s) {
    for (int64_t i = 0; i < n * n; ++i) a[(size_t)i] = A[s * n * n + i];
    for (int64_t i = 0; i < n; ++i) b[(size_t)i] = rhs[s * n + i];

    for (int64_t k = 0; k < n; ++k) {
      int64_t piv = k;
      float best = std::fabs(a[(size_t)(k * n + k)]);
      for (int64_t i = k + 1; i < n; ++i) {
        const float v = std::fabs(a[(size_t)(i * n + k)]);
        if (v > best) { best = v; piv = i; }
      }
      if (piv != k) {
        for (int64_t j = 0; j < n; ++j)
          std::swap(a[(size_t)(k * n + j)], a[(size_t)(piv * n + j)]);
        std::swap(b[(size_t)k], b[(size_t)piv]);
      }
      const float akk = a[(size_t)(k * n + k)];
      for (int64_t i = k + 1; i < n; ++i) {
        const float f = a[(size_t)(i * n + k)] / akk;
        for (int64_t j = k + 1; j < n; ++j)
          a[(size_t)(i * n + j)] -= f * a[(size_t)(k * n + j)];
        b[(size_t)i] -= f * b[(size_t)k];
      }
    }
    for (int64_t i = n - 1; i >= 0; --i) {
      float s2 = b[(size_t)i];
      for (int64_t j = i + 1; j < n; ++j) s2 -= a[(size_t)(i * n + j)] * b[(size_t)j];
      b[(size_t)i] = s2 / a[(size_t)(i * n + i)];
    }
    for (int64_t i = 0; i < n; ++i) x[s * n + i] = b[(size_t)i];
  }
}

}  // namespace jaxmetal
