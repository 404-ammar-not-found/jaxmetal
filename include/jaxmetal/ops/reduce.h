#pragma once
#include <cstdint>

namespace jaxmetal {

class MetalContext;
class KernelLibrary;
class Dispatcher;
class MetalBuffer;

// Compile + register kernels/reduce.metal into `lib`. Call once.
//
// Compensated (Neumaier) f32 summation. A plain f32 sum of n values accumulates
// O(n) rounding error as the running total outgrows the addends; a pairwise/tree
// sum reduces that to O(log n); compensated summation tracks the rounding error
// explicitly and stays accurate to ~1 ulp of the exact result almost independently
// of n. On Apple Silicon this is the only route to a trustworthy large sum, because
// the GPU has no f64 at all -- see the header comment in kernels/reduce.metal,
// including its hard dependency on safe math.
void register_reduce_kernels(KernelLibrary& lib);

// Sum of x[0..n) as a single f32.
//
// `compensated=false` selects an uncompensated tree sum with identical memory
// traffic and launch shape, so the two can be benchmarked against each other to
// isolate the cost of compensation. Allocates small scratch buffers per call
// (kReduceGroups float2 + one float); negligible beside the array scan at the
// sizes this is for, but it is per call.
float reduce_sum_f32(MetalContext& ctx, KernelLibrary& lib, Dispatcher& disp,
                     MetalBuffer& x, int64_t n, bool compensated = true);

// out[j] = sum_i a[i,j] for a[M,N] -> out[N], compensated over the M axis.
// The accuracy-preserving counterpart of nn_reduce_sum_axis0 (which is an
// uncompensated tree sum). Does not wait; caller synchronises.
void reduce_sum_axis0_comp_into(KernelLibrary& lib, Dispatcher& disp,
                                MetalBuffer& a, MetalBuffer& out,
                                int64_t M, int64_t N);

}  // namespace jaxmetal
