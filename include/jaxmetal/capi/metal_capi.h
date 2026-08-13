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

// --- Compensated reductions ----------------------------------------------------
// Sum x[0..n) in f32. `compensated`: 1 = Neumaier compensated summation (accurate
// to ~1 ulp of the exact sum almost regardless of n), 0 = plain tree sum with
// identical memory traffic and launch shape (the benchmark baseline). Apple GPUs
// have no f64, so compensation is the only route to a trustworthy large f32 sum
// here. Writes the result to *out. Returns 0 on success.
int metal_reduce_sum_f32(const float* x, int64_t n, int compensated, float* out);

// Same, on an already-resident buffer: no host copy, so timings measure the
// reduction rather than the transfer.
int metal_reduce_sum_resident(metal_buffer_t x, int64_t n, int compensated,
                              float* out);

// --- Blocked Cholesky ----------------------------------------------------------
// A = L*L^T in place on a resident [N,N] row-major f32 buffer. Lower triangle
// receives L; the strict upper triangle is zeroed. Returns LAPACK `info`: 0 on
// success, else the 1-based column at which the matrix stopped being positive
// definite (this also catches NaN input). The whole factorisation is one command
// buffer. Apple's own MPSMatrixDecompositionCholesky is NOT used - measured 14x
// slower than LAPACK on an M4 Pro.
int metal_cholesky_resident(metal_buffer_t A, int64_t n);

// Host-operand convenience wrapper: copies A in, factors, copies L back out.
int metal_cholesky_f32(float* A, int64_t n);

// --- Batched tiny-system solve -------------------------------------------------
// Solve `batch` independent A[b] x[b] = rhs[b] with LU + partial pivoting, one GPU
// thread per system. A[batch,n,n], rhs[batch,n], x[batch,n], pivmin[batch], all f32
// row-major. n must be in [2,8]: above that Apple's batched MPSMatrixDecompositionLU
// is faster (2.4x at n=12, 54x at n=32) so hand-writing it would be a pessimisation.
// pivmin[b] is min|U_kk|/max|U_kk|, or exactly 0 if system b is singular or its input
// held NaN/Inf -- a singularity flag, NOT a condition estimate. Returns 0 on success.
int metal_batched_solve_f32(const float* A, const float* rhs, float* x, float* pivmin,
                            int64_t batch, int64_t n);

// Same, on already-resident buffers. These kernels do well under one FLOP per byte
// moved, so the host copies cost about as much as the solve: use this whenever the
// data is already on the GPU or is reused across calls. pivmin may be NULL.
int metal_batched_solve_resident(metal_buffer_t A, metal_buffer_t rhs, metal_buffer_t x,
                                 metal_buffer_t pivmin, int64_t batch, int64_t n);

// Single-threaded CPU reference, same algorithm. The honest benchmark baseline: a
// plain scalar loop is 3-26x faster than looping numpy.linalg.solve at these sizes.
void metal_batched_solve_cpu_f32(const float* A, const float* rhs, float* x,
                                 int64_t batch, int64_t n);

// --- Double-single (df64) extended precision -----------------------------------
// ~48 bits of significand on a GPU with no f64 at all. A PRECISION feature, not a
// performance one: slower than the same work in f64 on the CPU at every size, because
// unified memory gives the GPU no bandwidth advantage to pay for the emulation.
// Buffers hold interleaved (hi, lo) f32 pairs, i.e. 2*n floats. op: 0=add 1=mul 2=div.
int metal_df64_binop(const float* a, const float* b, float* out, int64_t n, int op);

// out[i] = c[0]*x[i-1] + c[1]*x[i] + c[2]*x[i+1], zero boundaries. `coef` holds 3
// df64 values (6 floats). use_df64=0 runs the plain-f32 kernel, in which case x/out
// hold n floats rather than 2*n.
int metal_df64_stencil3(const float* x, float* out, const float* coef, int64_t n,
                        int use_df64);

// --- Resident MLP (in_dim -> hidden -> out_dim, ReLU, softmax cross-entropy) ----
// All weights, gradients, activations, and the current minibatch stay GPU-resident
// across calls; the whole SGD step runs on-device in one command buffer. This is
// where the GPU beats the CPU (no per-op host copies). Handle is opaque.
typedef void* metal_mlp_t;

// max_batch sizes the resident activation/scratch buffers once, up front.
// chunk_steps sizes the input/label buffers to hold that many consecutive
// minibatches, so metal_mlp_train_steps can encode that many SGD steps into one
// command buffer (pass 1 for the plain per-step behaviour).
// Parameters are UNINITIALIZED — call metal_mlp_set_params. Returns NULL on failure.
metal_mlp_t metal_mlp_create(int64_t in_dim, int64_t hidden, int64_t out_dim,
                             int64_t max_batch, int64_t chunk_steps);
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

// Copy n_steps consecutive minibatches (laid end to end) into the resident input
// buffers. X[n_steps*batch*in_dim] f32, labels[n_steps*batch] i32.
void metal_mlp_upload_chunk(metal_mlp_t mlp, const float* X, const int32_t* labels,
                            int64_t n_steps, int64_t batch);

// n_steps SGD steps over the uploaded chunk, encoded into ONE command buffer with a
// single host sync. This is what makes the GPU beat the CPU at moderate batch sizes:
// the ~140us driver round trip is paid once per chunk, not once per step. Read the
// mean loss with metal_mlp_last_loss. Returns 0 on success.
int metal_mlp_train_steps(metal_mlp_t mlp, int64_t n_steps, int64_t batch, float lr);

// Mean cross-entropy over the most recent train_step/train_steps call (no sync).
float metal_mlp_last_loss(metal_mlp_t mlp);

#ifdef __cplusplus
}  // extern "C"
#endif
