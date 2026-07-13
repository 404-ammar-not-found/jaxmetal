// Hand-written MSL kernels for the MLP training path (784->H->10, ReLU,
// softmax cross-entropy, SGD). Compiled at runtime by KernelLibrary and embedded
// as a string via cmake/EmbedMetal.cmake. All compute is f32 except the label
// buffer, which is int32. Buffer-binding convention matches elementwise.metal:
// data buffers at indices 0..k-1, push constants at index k (setBytes), and every
// kernel guards `if (gid >= n) return;`. Kernels compile with MTLMathModeSafe so
// +,-,*,/ are IEEE-exact vs the CPU reference and exp/log are within ~1e-4.

#include <metal_stdlib>
using namespace metal;

struct NNDims2 { uint M; uint N; };     // logical [M,N] matrix
struct NNAxpy  { float lr; uint n; };   // sgd_update: {lr, n}
struct SCEDims { uint B; uint C; };     // softmax cross-entropy: batch, classes

// ---- bias_add: out[i,j] = a[i,j] + bias[j]  over [M,N] (row-vector broadcast).
// `out` may alias `a`.
kernel void nn_bias_add(device const float* a    [[buffer(0)]],
                        device const float* bias [[buffer(1)]],
                        device float*       out  [[buffer(2)]],
                        constant NNDims2&   d     [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid >= d.M * d.N) return;
  out[gid] = a[gid] + bias[gid % d.N];
}

// ---- relu: out = max(x, 0). Strict: exactly 0 -> 0 (matches jnp.maximum).
kernel void nn_relu(device const float* x   [[buffer(0)]],
                    device float*       out [[buffer(1)]],
                    constant uint&      n    [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  out[gid] = max(x[gid], 0.0f);
}

// ---- relu_grad: out = (pre > 0) ? g : 0. `pre` is the pre-activation.
// Subgradient at 0 is 0 (strict >), matching jax.nn.relu.
kernel void nn_relu_grad(device const float* pre [[buffer(0)]],
                         device const float* g   [[buffer(1)]],
                         device float*       out [[buffer(2)]],
                         constant uint&      n    [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  out[gid] = (pre[gid] > 0.0f) ? g[gid] : 0.0f;
}

// ---- reduce_sum_axis0: out[j] = sum_i a[i,j]  ([M,N] -> [N]); bias gradients.
// One thread per output column j; sequential accumulate over the M rows.
kernel void nn_reduce_sum_axis0(device const float* a   [[buffer(0)]],
                                device float*       out [[buffer(1)]],
                                constant NNDims2&   d    [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid >= d.N) return;
  float s = 0.0f;
  for (uint i = 0; i < d.M; ++i) s += a[i * d.N + gid];
  out[gid] = s;
}

// ---- transpose2d: out[N,M] = (a[M,N])^T ; forms A^T and B^T for grads.
// One thread per input element.
kernel void nn_transpose2d(device const float* a   [[buffer(0)]],
                           device float*       out [[buffer(1)]],
                           constant NNDims2&   d    [[buffer(2)]],  // a is [M,N]
                           uint gid [[thread_position_in_grid]]) {
  if (gid >= d.M * d.N) return;
  uint i = gid / d.N, j = gid % d.N;
  out[j * d.M + i] = a[i * d.N + j];
}

// ---- sgd_update (in-place axpy): w[i] -= lr * dw[i].
kernel void nn_sgd_update(device float*       w  [[buffer(0)]],
                          device const float* dw [[buffer(1)]],
                          constant NNAxpy&    p   [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) return;
  w[gid] -= p.lr * dw[gid];
}

// ---- softmax_cross_entropy (numerically stable, fused).
// logits[B,C], int32 labels[B] -> per-example loss[B], dlogits[B,C], probs[B,C].
// dlogits is already divided by B (ready for the backward matmuls). loss[b] is the
// raw per-example NLL; the host sums and divides by B for the mean to report.
// One thread per row; inner loop over C.
kernel void nn_softmax_xent(device const float* logits  [[buffer(0)]],
                            device const int*   labels  [[buffer(1)]],
                            device float*       loss    [[buffer(2)]],
                            device float*       dlogits [[buffer(3)]],
                            device float*       probs   [[buffer(4)]],
                            constant SCEDims&   d        [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]) {
  if (gid >= d.B) return;
  const uint row = gid * d.C;

  float m = -INFINITY;                                   // 1) row max (stability)
  for (uint j = 0; j < d.C; ++j) m = max(m, logits[row + j]);

  float s = 0.0f;                                        // 2) sum exp(x - m)
  for (uint j = 0; j < d.C; ++j) s += exp(logits[row + j] - m);

  const float inv_s = 1.0f / s;
  const float invB  = 1.0f / float(d.B);
  const int   label = labels[gid];

  for (uint j = 0; j < d.C; ++j) {                       // 3) probs + gradient
    float p = exp(logits[row + j] - m) * inv_s;
    probs[row + j]   = p;
    float onehot     = (int(j) == label) ? 1.0f : 0.0f;
    dlogits[row + j] = (p - onehot) * invB;
  }
  loss[gid] = -(logits[row + label] - m) + log(s);       // per-example NLL
}

// ---- argmax over axis=1 for on-GPU accuracy. a is [B,C], out is int32[B].
kernel void nn_argmax_axis1(device const float* a   [[buffer(0)]],
                            device int*         out [[buffer(1)]],
                            constant NNDims2&   d    [[buffer(2)]],  // a is [B,C]
                            uint gid [[thread_position_in_grid]]) {
  if (gid >= d.M) return;
  const uint row = gid * d.N;
  float best = a[row]; int bi = 0;
  for (uint j = 1; j < d.N; ++j) { float v = a[row + j]; if (v > best) { best = v; bi = int(j); } }
  out[gid] = bi;
}
