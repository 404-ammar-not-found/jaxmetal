#include "jaxmetal/ops/nn.h"

#include "jaxmetal/metal/dtype.h"
#include "jaxmetal/metal/kernel_library.h"
#include "jaxmetal/metal/metal_buffer.h"
#include "jaxmetal/runtime/dispatcher.h"

#include "kernels_nn.h"   // generated: jaxmetal::kernels::nn_msl

#include <cstdint>
#include <vector>

namespace jaxmetal {
namespace {
// Byte-match the push-constant structs in kernels/nn.metal.
struct NNDims2 { uint32_t M; uint32_t N; };
struct NNAxpy  { float lr; uint32_t n; };
struct SCEDims { uint32_t B; uint32_t C; };
}  // namespace

void register_nn_kernels(KernelLibrary& lib) {
  lib.add_source("nn", kernels::nn_msl);
}

void bias_add_into(KernelLibrary& lib, Dispatcher& disp,
                   MetalBuffer& a, MetalBuffer& bias, MetalBuffer& out,
                   int64_t M, int64_t N) {
  void* pso = lib.pipeline("nn_bias_add");
  NNDims2 d{static_cast<uint32_t>(M), static_cast<uint32_t>(N)};
  std::vector<MetalBuffer*> bufs = {&a, &bias, &out};
  disp.dispatch_1d(pso, bufs, M * N, &d, sizeof(d));
}

void relu_into(KernelLibrary& lib, Dispatcher& disp,
               MetalBuffer& x, MetalBuffer& out, int64_t n) {
  void* pso = lib.pipeline("nn_relu");
  uint32_t nn = static_cast<uint32_t>(n);
  std::vector<MetalBuffer*> bufs = {&x, &out};
  disp.dispatch_1d(pso, bufs, n, &nn, sizeof(nn));
}

void relu_grad_into(KernelLibrary& lib, Dispatcher& disp,
                    MetalBuffer& pre, MetalBuffer& g, MetalBuffer& out, int64_t n) {
  void* pso = lib.pipeline("nn_relu_grad");
  uint32_t nn = static_cast<uint32_t>(n);
  std::vector<MetalBuffer*> bufs = {&pre, &g, &out};
  disp.dispatch_1d(pso, bufs, n, &nn, sizeof(nn));
}

void reduce_sum_axis0_into(KernelLibrary& lib, Dispatcher& disp,
                           MetalBuffer& a, MetalBuffer& out, int64_t M, int64_t N) {
  void* pso = lib.pipeline("nn_reduce_sum_axis0");
  NNDims2 d{static_cast<uint32_t>(M), static_cast<uint32_t>(N)};
  std::vector<MetalBuffer*> bufs = {&a, &out};
  disp.dispatch_1d(pso, bufs, N, &d, sizeof(d));   // one thread per output column
}

void transpose2d_into(KernelLibrary& lib, Dispatcher& disp,
                      MetalBuffer& a, MetalBuffer& out, int64_t M, int64_t N) {
  void* pso = lib.pipeline("nn_transpose2d");
  NNDims2 d{static_cast<uint32_t>(M), static_cast<uint32_t>(N)};
  std::vector<MetalBuffer*> bufs = {&a, &out};
  disp.dispatch_1d(pso, bufs, M * N, &d, sizeof(d));
}

void sgd_update_into(KernelLibrary& lib, Dispatcher& disp,
                     MetalBuffer& w, MetalBuffer& dw, float lr, int64_t n) {
  void* pso = lib.pipeline("nn_sgd_update");
  NNAxpy p{lr, static_cast<uint32_t>(n)};
  std::vector<MetalBuffer*> bufs = {&w, &dw};
  disp.dispatch_1d(pso, bufs, n, &p, sizeof(p));
}

void softmax_xent_into(KernelLibrary& lib, Dispatcher& disp,
                       MetalBuffer& logits, MetalBuffer& labels,
                       MetalBuffer& loss, MetalBuffer& dlogits, MetalBuffer& probs,
                       int64_t B, int64_t C) {
  void* pso = lib.pipeline("nn_softmax_xent");
  SCEDims d{static_cast<uint32_t>(B), static_cast<uint32_t>(C)};
  std::vector<MetalBuffer*> bufs = {&logits, &labels, &loss, &dlogits, &probs};
  disp.dispatch_1d(pso, bufs, B, &d, sizeof(d));   // one thread per row
}

void argmax_axis1_into(KernelLibrary& lib, Dispatcher& disp,
                       MetalBuffer& a, MetalBuffer& out_i32, int64_t B, int64_t C) {
  void* pso = lib.pipeline("nn_argmax_axis1");
  NNDims2 d{static_cast<uint32_t>(B), static_cast<uint32_t>(C)};
  std::vector<MetalBuffer*> bufs = {&a, &out_i32};
  disp.dispatch_1d(pso, bufs, B, &d, sizeof(d));
}

}  // namespace jaxmetal
