#pragma once
#include <cstdint>

namespace jaxmetal {

class KernelLibrary;
class Dispatcher;
class MetalBuffer;

// Compile + register kernels/nn.metal into `lib`. Call once. Provides the neural-
// net op set the MLP training path needs beyond matmul. These are resident `_into`
// ops (write into a pre-allocated `out`, take explicit dims, do NOT wait) so a
// scheduler can chain a whole train step on one queue and wait once, like
// matmul_into. They deliberately trust the dims (no shape()/dtype() consulting).
void register_nn_kernels(KernelLibrary& lib);

// out[i,j] = a[i,j] + bias[j] over [M,N]. `out` may alias `a`.
void bias_add_into(KernelLibrary& lib, Dispatcher& disp,
                   MetalBuffer& a, MetalBuffer& bias, MetalBuffer& out,
                   int64_t M, int64_t N);

// out = max(x, 0), n elements.
void relu_into(KernelLibrary& lib, Dispatcher& disp,
               MetalBuffer& x, MetalBuffer& out, int64_t n);

// out = (pre > 0) ? g : 0, n elements. `pre` is the pre-activation.
void relu_grad_into(KernelLibrary& lib, Dispatcher& disp,
                    MetalBuffer& pre, MetalBuffer& g, MetalBuffer& out, int64_t n);

// out[j] = sum_i a[i,j]  ([M,N] -> [N]); bias gradient over the batch axis.
void reduce_sum_axis0_into(KernelLibrary& lib, Dispatcher& disp,
                           MetalBuffer& a, MetalBuffer& out, int64_t M, int64_t N);

// out[N,M] = (a[M,N])^T.
void transpose2d_into(KernelLibrary& lib, Dispatcher& disp,
                      MetalBuffer& a, MetalBuffer& out, int64_t M, int64_t N);

// In-place: w[i] -= lr * dw[i], n elements.
void sgd_update_into(KernelLibrary& lib, Dispatcher& disp,
                     MetalBuffer& w, MetalBuffer& dw, float lr, int64_t n);

// Stable softmax cross-entropy. logits[B,C] f32, labels[B] i32.
// Outputs: loss[B] (per-example NLL), dlogits[B,C] (= (softmax-onehot)/B),
// probs[B,C]. All out buffers pre-allocated by the caller.
void softmax_xent_into(KernelLibrary& lib, Dispatcher& disp,
                       MetalBuffer& logits, MetalBuffer& labels,
                       MetalBuffer& loss, MetalBuffer& dlogits, MetalBuffer& probs,
                       int64_t B, int64_t C);

// out_i32[b] = argmax_j a[b,j], a is [B,C], out is i32[B].
void argmax_axis1_into(KernelLibrary& lib, Dispatcher& disp,
                       MetalBuffer& a, MetalBuffer& out_i32, int64_t B, int64_t C);

}  // namespace jaxmetal
