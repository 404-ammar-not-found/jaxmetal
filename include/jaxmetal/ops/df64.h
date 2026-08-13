#pragma once
#include <cstdint>

namespace jaxmetal {

class KernelLibrary;
class Dispatcher;
class MetalBuffer;

// Compile + register kernels/df64.metal into `lib`. Call once.
//
// Double-single ("df64") extended precision: a value is an unevaluated sum of two f32
// limbs, giving ~48 bits of significand against f32's 24. Metal has no `double` type
// at all, so this is the only way to exceed f32 on an Apple GPU.
//
// THIS IS A PRECISION FEATURE, NOT A PERFORMANCE ONE -- it is slower than the same
// work in f64 on the CPU at every size measured, because Apple Silicon's unified
// memory gives the GPU no bandwidth advantage to pay for the emulation. See
// docs/features/DF64_PRECISION.md. NOT IEEE float64: f32's exponent range, no
// guaranteed correct rounding.
void register_df64_kernels(KernelLibrary& lib);

enum class DF64Op { Add, Mul, Div };

// Elementwise op over n df64 values. Buffers hold interleaved (hi, lo) f32 pairs,
// i.e. 2*n floats. Does not wait; the caller synchronises.
void df64_elementwise(KernelLibrary& lib, Dispatcher& disp, DF64Op op,
                      MetalBuffer& a, MetalBuffer& b, MetalBuffer& out, int64_t n);

// 3-point stencil out[i] = c[0]*x[i-1] + c[1]*x[i] + c[2]*x[i+1], zero boundaries.
// `coef` holds 3 df64 values. `df64=false` runs the plain-f32 kernel instead, for
// accuracy and cost comparison, in which case buffers hold n (not 2n) floats.
void df64_stencil3(KernelLibrary& lib, Dispatcher& disp, MetalBuffer& x,
                   MetalBuffer& out, MetalBuffer& coef, int64_t n, bool use_df64);

}  // namespace jaxmetal
