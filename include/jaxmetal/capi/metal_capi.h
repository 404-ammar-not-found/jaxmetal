#pragma once
// Flat C ABI over the Metal runtime, so Python (ctypes) — and later other
// frontends — can drive the GPU without touching C++/Objective-C++. Kept
// intentionally tiny; grows alongside the op set.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// C = A[M,K] @ B[K,N], all row-major float32. Caller owns all buffers; C must
// hold M*N floats. Returns 0 on success, non-zero on failure.
int metal_matmul_f32(const float* A, const float* B, float* C,
                     int64_t M, int64_t K, int64_t N);

// Name of the Metal device (e.g. "Apple M4 Pro"). Valid until the next call.
const char* metal_device_name(void);

// --- GPU-resident buffers -----------------------------------------------------
// Keep operands on the GPU across calls to avoid per-call host<->device copies
// (Apple Silicon is unified memory, so copies are memcpy, but they and buffer
// allocation still dominate small/medium matmuls). Handles are opaque.
typedef void* metal_buffer_t;

metal_buffer_t metal_buffer_alloc(int64_t nelem_f32);
void metal_buffer_upload(metal_buffer_t h, const float* src, int64_t nelem_f32);
void metal_buffer_download(metal_buffer_t h, float* dst, int64_t nelem_f32);
void metal_buffer_free(metal_buffer_t h);

// C = A @ B on resident buffers (A:[M,K], B:[K,N], C:[M,N]); waits for
// completion. No host copies. Returns 0 on success.
int metal_matmul_resident(metal_buffer_t A, metal_buffer_t B, metal_buffer_t C,
                          int64_t M, int64_t K, int64_t N);

// C = A @ B via MetalPerformanceShaders (Apple's tuned GPU matmul, what PyTorch
// MPS uses). Host operands (alloc + copy in/out). Returns 0 on success.
int metal_mps_matmul_f32(const float* A, const float* B, float* C,
                         int64_t M, int64_t K, int64_t N);

// MPS matmul on GPU-resident buffers (no host copies). Returns 0 on success.
int metal_mps_matmul_resident(metal_buffer_t A, metal_buffer_t B, metal_buffer_t C,
                              int64_t M, int64_t K, int64_t N);

// --- CPU path + auto router ---------------------------------------------------
// C = A @ B on the CPU (Accelerate/BLAS). Host buffers, row-major f32.
void metal_cpu_matmul_f32(const float* A, const float* B, float* C,
                          int64_t M, int64_t K, int64_t N);

// C = A @ B, routed to whichever backend is faster for this shape (host
// operands: CPU vs end-to-end GPU incl. copies). If used_gpu != NULL, it is set
// to 1 when the GPU path ran, else 0. Returns 0 on success.
int metal_matmul_auto_f32(const float* A, const float* B, float* C,
                          int64_t M, int64_t K, int64_t N, int* used_gpu);

// --- Resident MLP (in_dim -> hidden -> out_dim, ReLU, softmax cross-entropy) ----
// All weights, gradients, activations, and the current minibatch stay GPU-resident
// across calls; the whole SGD step runs on-device in one command buffer. This is
// where the GPU beats the CPU (no per-op host copies). Handle is opaque.
typedef void* metal_mlp_t;

// max_batch sizes the resident activation/scratch buffers once, up front.
// Parameters are UNINITIALIZED — call metal_mlp_set_params. Returns NULL on failure.
metal_mlp_t metal_mlp_create(int64_t in_dim, int64_t hidden, int64_t out_dim,
                             int64_t max_batch);
void        metal_mlp_destroy(metal_mlp_t mlp);

// Host <-> resident params. Row-major f32. W1[in_dim*hidden] b1[hidden]
// W2[hidden*out_dim] b2[out_dim].
void metal_mlp_set_params(metal_mlp_t mlp, const float* W1, const float* b1,
                          const float* W2, const float* b2);
void metal_mlp_get_params(metal_mlp_t mlp, float* W1, float* b1,
                          float* W2, float* b2);

// Copy a minibatch into the reused resident buffers. X[batch*in_dim] f32,
// labels[batch] int32 (labels may be NULL for eval). batch <= max_batch.
void metal_mlp_upload_batch(metal_mlp_t mlp, const float* X, const int32_t* labels,
                            int64_t batch);

// Forward on the uploaded batch; writes logits[batch*out_dim] to host (argmax done
// caller-side). Returns 0 on success.
int metal_mlp_forward(metal_mlp_t mlp, int64_t batch, float* logits_out);

// One SGD step on the uploaded (X, labels): forward, softmax-xent, backward, and
// theta -= lr*grad, all resident. Writes the mean cross-entropy loss (pre-update)
// to *out_loss. Returns 0 on success.
int metal_mlp_train_step(metal_mlp_t mlp, int64_t batch, float lr, float* out_loss);

#ifdef __cplusplus
}  // extern "C"
#endif
