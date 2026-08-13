#pragma once
#include <cstdint>

namespace jaxmetal {

class MetalContext;
class KernelLibrary;
class MetalBuffer;

// Compile + register kernels/cholesky.metal into `lib`. Call once.
void register_cholesky_kernels(KernelLibrary& lib);

// Blocked right-looking Cholesky: A = L·Lᵀ, in place, f32, row-major [N,N] resident.
// On success the lower triangle holds L and the strict upper triangle is zeroed.
//
// Returns LAPACK `info`: 0 on success, or the 1-based index of the first column at
// which the leading minor stopped being positive definite (which also catches NaN in
// the input). On failure the contents of A are undefined.
//
// The whole factorisation is ONE command buffer with one commit and one wait, so the
// ~140 us driver round trip is paid once regardless of N. Only the O(NB³) panel work
// is hand-written; the O(N³) trailing update is an MPS GEMM. See
// docs/features/GPU_CHOLESKY.md for why MPS's own MPSMatrixDecompositionCholesky is
// not used (measured 14x slower than LAPACK).
int cholesky_f32(MetalContext& ctx, KernelLibrary& lib, MetalBuffer& A, int64_t N);

}  // namespace jaxmetal
