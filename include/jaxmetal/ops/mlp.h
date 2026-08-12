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
//
// ONE COMMAND BUFFER PER STEP IS NOT ENOUGH. Measured on an M4 Pro, a single
// commit + waitUntilCompleted round trip costs ~140 us of pure driver latency on
// top of GPU execution, and the ~100 us of CPU-side encoding cannot overlap it
// because the host is blocked. At hidden=128 that fixed ~0.42 ms swamped the step
// until batch ~1000. `upload_chunk` + `train_steps` therefore encode up to
// `chunk_steps` consecutive SGD steps into ONE command buffer: the round trip is
// paid once per chunk instead of once per step, and encoding step i+1 overlaps
// the GPU running step i. Steps stay correctly ordered because Metal hazard-tracks
// within a command buffer, and each step's SGD update is a read-after-write on the
// same resident parameter buffers.
class MLP {
 public:
  // `chunk_steps` sizes the input/label buffers to hold that many consecutive
  // minibatches (activations stay sized to one). 1 keeps the old per-step behaviour.
  MLP(int64_t in_dim, int64_t hidden, int64_t out_dim, int64_t max_batch,
      int64_t chunk_steps = 1);
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
  // Equivalent to train_steps(1, batch, lr) followed by last_loss().
  float train_step(int64_t batch, float lr);

  int64_t chunk_steps() const;

  // Host -> resident buffers for `n_steps` consecutive minibatches laid end to end.
  // X[n_steps*batch*in_dim] f32 row-major, labels[n_steps*batch] i32.
  // n_steps <= chunk_steps() and batch <= max_batch().
  void upload_chunk(const float* X, const int32_t* labels, int64_t n_steps,
                    int64_t batch);

  // `n_steps` SGD steps over the uploaded chunk (step i takes rows [i*batch,
  // (i+1)*batch)), all encoded into ONE command buffer with a single host sync at
  // the end. Read the mean loss over the whole chunk with last_loss().
  void train_steps(int64_t n_steps, int64_t batch, float lr);

  // Mean cross-entropy over the most recent train_step/train_steps call. Cheap:
  // the per-step sums were reduced on the GPU and the sync already happened.
  float last_loss() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace jaxmetal
