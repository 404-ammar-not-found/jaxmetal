#include "jaxmetal/capi/metal_capi.h"

#include "jaxmetal/cpu/cpu_matmul.h"
#include "jaxmetal/metal/dtype.h"
#include "jaxmetal/metal/kernel_library.h"
#include "jaxmetal/metal/metal_buffer.h"
#include "jaxmetal/metal/metal_context.h"
#include "jaxmetal/ops/matmul.h"
#include "jaxmetal/ops/mlp.h"
#include "jaxmetal/ops/mps_matmul.h"
#include "jaxmetal/ops/reduce.h"
#include "jaxmetal/ops/cholesky.h"
#include "jaxmetal/ops/batched_solve.h"
#include "jaxmetal/ops/df64.h"
#include "jaxmetal/runtime/dispatcher.h"

#include <cstring>
#include <memory>
#include <string>

using namespace jaxmetal;

namespace {

// Matmul kernel is compiled once, on first use.
KernelLibrary& matmul_lib() {
  static KernelLibrary lib(MetalContext::instance());
  static int once = (register_matmul_kernel(lib), 0);
  (void)once;
  return lib;
}

// A resident buffer handle owns a shared_ptr<MetalBuffer>.
using BufferHandle = std::shared_ptr<MetalBuffer>;

// FLOP (2*M*N*K) crossover above which the end-to-end GPU path beats CPU BLAS
// for HOST operands. The GPU arm is MPS (MetalPerformanceShaders) — Apple's
// tuned matmul, the same one PyTorch MPS uses.
//
// Calibration on Apple M4 Pro (f32):
//  * MPS *resident* (data already on GPU) beats Accelerate from N>=2048
//    (1.5-1.75x; ~5.5 vs ~3.0 TFLOP/s) — the real GPU win, see
//    metal_mps_matmul_resident.
//  * MPS *host/e2e* (this path, incl. H2D/D2H copies + per-call setup) only
//    reaches break-even around N~4096-5120; below that the CPU wins. So the
//    host-operand router favors the CPU until the work is large, and the robust
//    GPU advantage lives in the resident path (a pipeline / the PJRT backend,
//    where operands stay on-device — which is exactly why PyTorch/JAX keep
//    tensors resident rather than auto-routing per op).
constexpr double kAutoGpuFlopThreshold = 1.4e11;  // ~2*4096^3; measured e2e crossover

}  // namespace

int metal_matmul_f32(const float* A, const float* B, float* C,
                     int64_t M, int64_t K, int64_t N) {
  try {
    MetalContext& ctx = MetalContext::instance();
    Dispatcher disp(ctx);
    auto da = ctx.from_host(A, {M, K}, DType::F32);
    auto db = ctx.from_host(B, {K, N}, DType::F32);
    auto out = matmul(ctx, matmul_lib(), disp, *da, *db);
    disp.wait();
    std::memcpy(C, out->contents(), sizeof(float) * static_cast<size_t>(M * N));
    return 0;
  } catch (...) {
    return 1;
  }
}

const char* metal_device_name(void) {
  static std::string name;
  try {
    name = MetalContext::instance().device_name();
  } catch (...) {
    name = "unknown";
  }
  return name.c_str();
}

// --- GPU-resident buffers -----------------------------------------------------

metal_buffer_t metal_buffer_alloc(int64_t nelem_f32) {
  try {
    auto buf = MetalContext::instance().alloc({nelem_f32}, DType::F32);
    return new BufferHandle(std::move(buf));
  } catch (...) {
    return nullptr;
  }
}

void metal_buffer_upload(metal_buffer_t h, const float* src, int64_t nelem_f32) {
  if (!h) return;
  auto& buf = *static_cast<BufferHandle*>(h);
  std::memcpy(buf->contents(), src, sizeof(float) * static_cast<size_t>(nelem_f32));
}

void metal_buffer_download(metal_buffer_t h, float* dst, int64_t nelem_f32) {
  if (!h) return;
  auto& buf = *static_cast<BufferHandle*>(h);
  std::memcpy(dst, buf->contents(), sizeof(float) * static_cast<size_t>(nelem_f32));
}

void metal_buffer_free(metal_buffer_t h) {
  delete static_cast<BufferHandle*>(h);
}

int metal_matmul_resident(metal_buffer_t A, metal_buffer_t B, metal_buffer_t C,
                          int64_t M, int64_t K, int64_t N) {
  if (!A || !B || !C) return 2;
  try {
    auto& a = *static_cast<BufferHandle*>(A);
    auto& b = *static_cast<BufferHandle*>(B);
    auto& c = *static_cast<BufferHandle*>(C);
    Dispatcher disp(MetalContext::instance());
    matmul_into(matmul_lib(), disp, *a, *b, *c, M, K, N);
    disp.wait();
    return 0;
  } catch (...) {
    return 1;
  }
}

int metal_mps_matmul_f32(const float* A, const float* B, float* C,
                         int64_t M, int64_t K, int64_t N) {
  try {
    MetalContext& ctx = MetalContext::instance();
    auto da = ctx.from_host(A, {M, K}, DType::F32);
    auto db = ctx.from_host(B, {K, N}, DType::F32);
    auto dc = ctx.alloc({M, N}, DType::F32);
    mps_matmul_into(ctx, *da, *db, *dc, M, K, N);
    std::memcpy(C, dc->contents(), sizeof(float) * static_cast<size_t>(M * N));
    return 0;
  } catch (...) {
    return 1;
  }
}

int metal_mps_matmul_resident(metal_buffer_t A, metal_buffer_t B, metal_buffer_t C,
                              int64_t M, int64_t K, int64_t N) {
  if (!A || !B || !C) return 2;
  try {
    auto& a = *static_cast<BufferHandle*>(A);
    auto& b = *static_cast<BufferHandle*>(B);
    auto& c = *static_cast<BufferHandle*>(C);
    mps_matmul_into(MetalContext::instance(), *a, *b, *c, M, K, N);
    return 0;
  } catch (...) {
    return 1;
  }
}

void metal_cpu_matmul_f32(const float* A, const float* B, float* C,
                          int64_t M, int64_t K, int64_t N) {
  cpu_matmul_f32(A, B, C, M, K, N);
}

// Calibrated on Apple M4 Pro (f32): for host operands the GPU only wins once the
// work is large enough to amortize the H2D/D2H copies (~2*M*N*K FLOPs). Below the
// threshold Accelerate/BLAS wins. See benchmarks/bench_matmul.py + calibration.
static bool auto_prefer_gpu(int64_t M, int64_t K, int64_t N) {
  // 2*M*N*K FLOPs; threshold picked from the measured crossover.
  const double work = 2.0 * static_cast<double>(M) * static_cast<double>(N) *
                      static_cast<double>(K);
  return work >= kAutoGpuFlopThreshold;
}

int metal_matmul_auto_f32(const float* A, const float* B, float* C,
                          int64_t M, int64_t K, int64_t N, int* used_gpu) {
  const bool gpu = auto_prefer_gpu(M, K, N);
  if (used_gpu) *used_gpu = gpu ? 1 : 0;
  if (gpu) return metal_mps_matmul_f32(A, B, C, M, K, N);  // MPS: Apple's tuned GPU matmul
  cpu_matmul_f32(A, B, C, M, K, N);
  return 0;
}

// --- Compensated reductions ----------------------------------------------------

namespace {
// Reduce kernels compiled once, on first use.
KernelLibrary& reduce_lib() {
  static KernelLibrary lib(MetalContext::instance());
  static int once = (register_reduce_kernels(lib), 0);
  (void)once;
  return lib;
}
}  // namespace

int metal_reduce_sum_f32(const float* x, int64_t n, int compensated, float* out) {
  if (!x || !out) return 2;
  try {
    MetalContext& ctx = MetalContext::instance();
    Dispatcher disp(ctx);
    auto dx = ctx.from_host(x, {n}, DType::F32);
    *out = reduce_sum_f32(ctx, reduce_lib(), disp, *dx, n, compensated != 0);
    return 0;
  } catch (...) {
    return 1;
  }
}

int metal_reduce_sum_resident(metal_buffer_t x, int64_t n, int compensated,
                              float* out) {
  if (!x || !out) return 2;
  try {
    auto& b = *static_cast<BufferHandle*>(x);
    MetalContext& ctx = MetalContext::instance();
    Dispatcher disp(ctx);
    *out = reduce_sum_f32(ctx, reduce_lib(), disp, *b, n, compensated != 0);
    return 0;
  } catch (...) {
    return 1;
  }
}

// --- Blocked Cholesky ----------------------------------------------------------

namespace {
KernelLibrary& cholesky_lib() {
  static KernelLibrary lib(MetalContext::instance());
  static int once = (register_cholesky_kernels(lib), 0);
  (void)once;
  return lib;
}
}  // namespace

int metal_cholesky_resident(metal_buffer_t A, int64_t n) {
  if (!A) return -1;
  try {
    auto& b = *static_cast<BufferHandle*>(A);
    return cholesky_f32(MetalContext::instance(), cholesky_lib(), *b, n);
  } catch (...) {
    return -1;
  }
}

int metal_cholesky_f32(float* A, int64_t n) {
  if (!A) return -1;
  try {
    MetalContext& ctx = MetalContext::instance();
    auto d = ctx.from_host(A, {n, n}, DType::F32);
    int info = cholesky_f32(ctx, cholesky_lib(), *d, n);
    std::memcpy(A, d->contents(), sizeof(float) * (size_t)(n * n));
    return info;
  } catch (...) {
    return -1;
  }
}

// --- Batched tiny-system solve -------------------------------------------------

namespace {
KernelLibrary& batched_solve_lib() {
  static KernelLibrary lib(MetalContext::instance());
  static int once = (register_batched_solve_kernels(lib), 0);
  (void)once;
  return lib;
}
}  // namespace

int metal_batched_solve_f32(const float* A, const float* rhs, float* x, float* pivmin,
                            int64_t batch, int64_t n, int spd) {
  if (!A || !rhs || !x) return 2;
  try {
    MetalContext& ctx = MetalContext::instance();
    Dispatcher disp(ctx);
    auto dA = ctx.from_host(A, {batch, n, n}, DType::F32);
    auto dR = ctx.from_host(rhs, {batch, n}, DType::F32);
    auto dX = ctx.alloc({batch, n}, DType::F32);
    auto dP = ctx.alloc({batch}, DType::F32);
    batched_solve_f32(batched_solve_lib(), disp, *dA, *dR, *dX, *dP, batch, n, spd != 0);
    disp.wait();
    std::memcpy(x, dX->contents(), sizeof(float) * (size_t)(batch * n));
    if (pivmin) std::memcpy(pivmin, dP->contents(), sizeof(float) * (size_t)batch);
    return 0;
  } catch (...) {
    return 1;
  }
}

int metal_batched_solve_resident(metal_buffer_t A, metal_buffer_t rhs, metal_buffer_t x,
                                 metal_buffer_t pivmin, int64_t batch, int64_t n,
                                 int spd) {
  if (!A || !rhs || !x) return 2;
  try {
    MetalContext& ctx = MetalContext::instance();
    Dispatcher disp(ctx);
    auto& dA = *static_cast<BufferHandle*>(A);
    auto& dR = *static_cast<BufferHandle*>(rhs);
    auto& dX = *static_cast<BufferHandle*>(x);
    // pivmin is optional; the kernel always writes it, so give it somewhere to go.
    static thread_local std::shared_ptr<MetalBuffer> scratch;
    BufferHandle* dP = static_cast<BufferHandle*>(pivmin);
    if (!dP) {
      if (!scratch || scratch->num_elements() < batch)
        scratch = ctx.alloc({batch}, DType::F32);
    }
    batched_solve_f32(batched_solve_lib(), disp, *dA, *dR, *dX,
                      dP ? **dP : *scratch, batch, n, spd != 0);
    disp.wait();
    return 0;
  } catch (...) {
    return 1;
  }
}

void metal_batched_solve_cpu_f32(const float* A, const float* rhs, float* x,
                                 int64_t batch, int64_t n) {
  batched_solve_cpu_f32(A, rhs, x, batch, n);
}

// --- Double-single (df64) extended precision -----------------------------------

namespace {
KernelLibrary& df64_lib() {
  static KernelLibrary lib(MetalContext::instance());
  static int once = (register_df64_kernels(lib), 0);
  (void)once;
  return lib;
}
}  // namespace

int metal_df64_binop(const float* a, const float* b, float* out, int64_t n, int op) {
  if (!a || !b || !out) return 2;
  try {
    MetalContext& ctx = MetalContext::instance();
    Dispatcher disp(ctx);
    auto da = ctx.from_host(a, {n * 2}, DType::F32);
    auto db = ctx.from_host(b, {n * 2}, DType::F32);
    auto dout = ctx.alloc({n * 2}, DType::F32);
    const DF64Op o = op == 1 ? DF64Op::Mul : (op == 2 ? DF64Op::Div : DF64Op::Add);
    df64_elementwise(df64_lib(), disp, o, *da, *db, *dout, n);
    disp.wait();
    std::memcpy(out, dout->contents(), sizeof(float) * (size_t)(n * 2));
    return 0;
  } catch (...) {
    return 1;
  }
}

int metal_df64_binop_resident(metal_buffer_t a, metal_buffer_t b, metal_buffer_t out,
                              int64_t n, int op) {
  if (!a || !b || !out) return 2;
  try {
    Dispatcher disp(MetalContext::instance());
    auto& da = *static_cast<BufferHandle*>(a);
    auto& db = *static_cast<BufferHandle*>(b);
    auto& dout = *static_cast<BufferHandle*>(out);
    const DF64Op o = op == 1 ? DF64Op::Mul : (op == 2 ? DF64Op::Div : DF64Op::Add);
    df64_elementwise(df64_lib(), disp, o, *da, *db, *dout, n);
    disp.wait();
    return 0;
  } catch (...) {
    return 1;
  }
}

int metal_df64_stencil3(const float* x, float* out, const float* coef, int64_t n,
                        int use_df64) {
  if (!x || !out || !coef) return 2;
  try {
    MetalContext& ctx = MetalContext::instance();
    Dispatcher disp(ctx);
    const int64_t w = use_df64 ? 2 : 1;
    auto dx = ctx.from_host(x, {n * w}, DType::F32);
    auto dc = ctx.from_host(coef, {use_df64 ? 6 : 3}, DType::F32);
    auto dout = ctx.alloc({n * w}, DType::F32);
    df64_stencil3(df64_lib(), disp, *dx, *dout, *dc, n, use_df64 != 0);
    disp.wait();
    std::memcpy(out, dout->contents(), sizeof(float) * (size_t)(n * w));
    return 0;
  } catch (...) {
    return 1;
  }
}

// --- Resident MLP -------------------------------------------------------------

metal_mlp_t metal_mlp_create(int64_t in_dim, int64_t hidden, int64_t out_dim,
                             int64_t max_batch, int64_t chunk_steps) {
  try {
    return new jaxmetal::MLP(in_dim, hidden, out_dim, max_batch, chunk_steps);
  } catch (...) {
    return nullptr;
  }
}

void metal_mlp_destroy(metal_mlp_t m) { delete static_cast<jaxmetal::MLP*>(m); }

void metal_mlp_set_params(metal_mlp_t m, const float* W1, const float* b1,
                          const float* W2, const float* b2) {
  if (!m) return;
  static_cast<jaxmetal::MLP*>(m)->set_params(W1, b1, W2, b2);
}

void metal_mlp_get_params(metal_mlp_t m, float* W1, float* b1,
                          float* W2, float* b2) {
  if (!m) return;
  static_cast<jaxmetal::MLP*>(m)->get_params(W1, b1, W2, b2);
}

void metal_mlp_upload_batch(metal_mlp_t m, const float* X, const int32_t* labels,
                            int64_t batch) {
  if (!m) return;
  static_cast<jaxmetal::MLP*>(m)->upload_batch(X, labels, batch);
}

void metal_mlp_upload_chunk(metal_mlp_t m, const float* X, const int32_t* labels,
                            int64_t n_steps, int64_t batch) {
  if (!m) return;
  static_cast<jaxmetal::MLP*>(m)->upload_chunk(X, labels, n_steps, batch);
}

int metal_mlp_train_steps(metal_mlp_t m, int64_t n_steps, int64_t batch, float lr) {
  if (!m) return 2;
  try {
    static_cast<jaxmetal::MLP*>(m)->train_steps(n_steps, batch, lr);
    return 0;
  } catch (...) {
    return 1;
  }
}

float metal_mlp_last_loss(metal_mlp_t m) {
  return m ? static_cast<jaxmetal::MLP*>(m)->last_loss() : 0.0f;
}

int metal_mlp_forward(metal_mlp_t m, int64_t batch, float* logits_out) {
  if (!m) return 2;
  try {
    static_cast<jaxmetal::MLP*>(m)->forward(batch, logits_out);
    return 0;
  } catch (...) {
    return 1;
  }
}

int metal_mlp_train_step(metal_mlp_t m, int64_t batch, float lr, float* out_loss) {
  if (!m) return 2;
  try {
    float l = static_cast<jaxmetal::MLP*>(m)->train_step(batch, lr);
    if (out_loss) *out_loss = l;
    return 0;
  } catch (...) {
    return 1;
  }
}
