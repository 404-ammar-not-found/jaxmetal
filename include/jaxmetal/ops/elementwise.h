#pragma once
#include <memory>

namespace jaxmetal {

class MetalContext;
class KernelLibrary;
class Dispatcher;
class MetalBuffer;

// The arithmetic "instruction set" currently supported on the GPU. Each maps to
// one hand-written MSL kernel in kernels/elementwise.metal.
enum class BinaryOp { Add, Sub, Mul, Div, Max, Min };
enum class UnaryOp { Neg, Abs, Exp };

// Compile + register the elementwise kernel module into `lib`. Call once.
void register_elementwise_kernels(KernelLibrary& lib);

// MSL kernel function name for an op (matches kernels/elementwise.metal).
const char* kernel_name(BinaryOp op);
const char* kernel_name(UnaryOp op);

// Run out = op(a, b) elementwise on the GPU (f32). Allocates and returns `out`.
// Submits the dispatch; call Dispatcher::wait() before reading results.
std::shared_ptr<MetalBuffer> elementwise_binary(MetalContext& ctx, KernelLibrary& lib,
                                                Dispatcher& disp, BinaryOp op,
                                                MetalBuffer& a, MetalBuffer& b);

// Run out = op(a) elementwise on the GPU (f32).
std::shared_ptr<MetalBuffer> elementwise_unary(MetalContext& ctx, KernelLibrary& lib,
                                               Dispatcher& disp, UnaryOp op, MetalBuffer& a);

}  // namespace jaxmetal
