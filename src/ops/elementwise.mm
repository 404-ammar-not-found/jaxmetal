#include "jaxmetal/ops/elementwise.h"

#include "jaxmetal/metal/dtype.h"
#include "jaxmetal/metal/kernel_library.h"
#include "jaxmetal/metal/metal_buffer.h"
#include "jaxmetal/metal/metal_context.h"
#include "jaxmetal/runtime/dispatcher.h"

#include "kernels_elementwise.h"  // generated: jaxmetal::kernels::elementwise_msl

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace jaxmetal {

void register_elementwise_kernels(KernelLibrary& lib) {
  lib.add_source("elementwise", kernels::elementwise_msl);
}

const char* kernel_name(BinaryOp op) {
  switch (op) {
    case BinaryOp::Add: return "ew_add";
    case BinaryOp::Sub: return "ew_sub";
    case BinaryOp::Mul: return "ew_mul";
    case BinaryOp::Div: return "ew_div";
    case BinaryOp::Max: return "ew_max";
    case BinaryOp::Min: return "ew_min";
  }
  throw std::runtime_error("unknown BinaryOp");
}

const char* kernel_name(UnaryOp op) {
  switch (op) {
    case UnaryOp::Neg: return "ew_neg";
    case UnaryOp::Abs: return "ew_abs";
    case UnaryOp::Exp: return "ew_exp";
  }
  throw std::runtime_error("unknown UnaryOp");
}

std::shared_ptr<MetalBuffer> elementwise_binary(MetalContext& ctx, KernelLibrary& lib,
                                                Dispatcher& disp, BinaryOp op,
                                                MetalBuffer& a, MetalBuffer& b) {
  if (a.num_elements() != b.num_elements())
    throw std::runtime_error("elementwise_binary: element count mismatch");
  if (a.dtype() != DType::F32 || b.dtype() != DType::F32)
    throw std::runtime_error("elementwise_binary: only f32 is supported");

  auto out = ctx.alloc(a.shape(), DType::F32);
  void* pso = lib.pipeline(kernel_name(op));
  uint32_t n = static_cast<uint32_t>(a.num_elements());
  std::vector<MetalBuffer*> bufs = {&a, &b, out.get()};
  disp.dispatch_1d(pso, bufs, a.num_elements(), &n, sizeof(n));
  return out;
}

std::shared_ptr<MetalBuffer> elementwise_unary(MetalContext& ctx, KernelLibrary& lib,
                                               Dispatcher& disp, UnaryOp op, MetalBuffer& a) {
  if (a.dtype() != DType::F32)
    throw std::runtime_error("elementwise_unary: only f32 is supported");

  auto out = ctx.alloc(a.shape(), DType::F32);
  void* pso = lib.pipeline(kernel_name(op));
  uint32_t n = static_cast<uint32_t>(a.num_elements());
  std::vector<MetalBuffer*> bufs = {&a, out.get()};
  disp.dispatch_1d(pso, bufs, a.num_elements(), &n, sizeof(n));
  return out;
}

}  // namespace jaxmetal
