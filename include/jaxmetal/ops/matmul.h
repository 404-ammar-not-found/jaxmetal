#pragma once
#include <memory>

namespace jaxmetal {

class MetalContext;
class KernelLibrary;
class Dispatcher;
class MetalBuffer;

// Compile + register the matmul kernel module into `lib`. Call once.
void register_matmul_kernel(KernelLibrary& lib);

// C = A @ B on the GPU (f32). A is [M, K], B is [K, N] (both rank-2, row-major);
// returns a freshly allocated [M, N] buffer. Submits the dispatch; call
// Dispatcher::wait() before reading results. Throws on rank/shape/dtype misuse.
//
// This is the building block for StableHLO dot_general (the general batched /
// arbitrary-contraction form will normalize to this via transposes later).
std::shared_ptr<MetalBuffer> matmul(MetalContext& ctx, KernelLibrary& lib, Dispatcher& disp,
                                    MetalBuffer& a, MetalBuffer& b);

// Lower-level: dispatch C[M,N] = A[M,K] @ B[K,N] into a pre-allocated `out`,
// using explicit dims (buffer shapes are not consulted). Used by the resident
// C-ABI path where operands stay on the GPU across calls. Does not wait.
void matmul_into(KernelLibrary& lib, Dispatcher& disp, MetalBuffer& a, MetalBuffer& b,
                 MetalBuffer& out, int64_t M, int64_t K, int64_t N);

}  // namespace jaxmetal
