#pragma once
#include <cstdint>
#include <memory>

namespace jaxmetal {

class MetalContext;
class KernelLibrary;
class MetalBuffer;

// A D -> H -> C MLP (ReLU hidden, softmax cross-entropy) with ALL parameters,
// gradients, and per-batch activations kept GPU-resident. Parameters are uploaded
// explicitly (set_params) so init matches the JAX/NumPy golden reference exactly.
//
// The whole forward+backward+SGD train step is encoded into a SINGLE Metal command
// buffer (compute encoders + MPS matmuls, automatic intra-command-buffer hazard
// tracking), committed once and waited on once. That single waitUntilCompleted is
// the only host sync per step, and it guarantees every parameter update has landed
// before get_params/next-step reads — the residency condition under which the GPU
// beats the CPU (params never leave the device across the SGD loop).
//
// Buffers are sized to `max_batch` once; every op takes the runtime batch (<= max)
// so eval can run trailing partial batches without reallocation.
class MLP {
 public:
  MLP(int64_t in_dim, int64_t hidden, int64_t out_dim, int64_t max_batch);
  ~MLP();
  MLP(const MLP&) = delete;
  MLP& operator=(const MLP&) = delete;

  int64_t in_dim() const;
  int64_t hidden() const;
  int64_t out_dim() const;
  int64_t max_batch() const;

  // Host -> resident params. Row-major f32. W1[D*H] b1[H] W2[H*C] b2[C].
  void set_params(const float* W1, const float* b1, const float* W2, const float* b2);
  // Resident -> host (checkpoint / golden compare). Same sizes/layout.
  void get_params(float* W1, float* b1, float* W2, float* b2) const;

  // Host -> reused resident batch buffers. x[batch*D] row-major f32; labels[batch]
  // int32 (may be null for eval-only forward). batch <= max_batch.
  void upload_batch(const float* x, const int32_t* labels, int64_t batch);

  // Forward on the uploaded batch; copies logits[batch*C] to logits_out. One sync.
  void forward(int64_t batch, float* logits_out);

  // One SGD step on the uploaded (x, labels): forward -> softmax-xent -> backward
  // -> theta -= lr*grad, all resident in one command buffer. Returns the mean
  // cross-entropy loss over the batch (computed before the update).
  float train_step(int64_t batch, float lr);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace jaxmetal
