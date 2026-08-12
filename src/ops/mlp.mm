#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <CoreFoundation/CoreFoundation.h>

#include "jaxmetal/ops/mlp.h"

#include "jaxmetal/metal/dtype.h"
#include "jaxmetal/metal/kernel_library.h"
#include "jaxmetal/metal/metal_buffer.h"
#include "jaxmetal/metal/metal_context.h"
#include "jaxmetal/ops/matmul.h"  // register_matmul_kernel (fallback / A-vs-B)
#include "jaxmetal/ops/nn.h"      // register_nn_kernels

#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace jaxmetal {

// Threadgroup width of nn_reduce_sum_axis0; must match RED_TG in kernels/nn.metal.
constexpr NSUInteger kReduceTG = 256;

namespace {
// Push-constant structs, byte-matching kernels/nn.metal.
struct NNDims2 { uint32_t M; uint32_t N; };
struct NNAxpy  { float lr; uint32_t n; };
struct SCEDims { uint32_t B; uint32_t C; };

inline id<MTLBuffer> mtl(const std::shared_ptr<MetalBuffer>& b) {
  return (__bridge id<MTLBuffer>)b->mtl_handle();
}

// An MPS GEMM with its three matrix views, built ONCE per shape. Building these
// per step cost ~7 Objective-C allocations x 5 matmuls x every step, which
// measured as the bulk of the ~0.5 ms fixed per-step cost.
struct Gemm {
  MPSMatrixMultiplication* mm = nil;
  MPSMatrix* A = nil;
  MPSMatrix* B = nil;
  MPSMatrix* C = nil;

  void encode(id<MTLCommandBuffer> cmd) const {
    [mm encodeToCommandBuffer:cmd leftMatrix:A rightMatrix:B resultMatrix:C];
  }
};

// A row-major [rows, cols] f32 view of a resident buffer.
MPSMatrix* mps_view(id<MTLBuffer> buf, int64_t rows, int64_t cols) {
  MPSMatrixDescriptor* d =
      [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)rows
                                            columns:(NSUInteger)cols
                                           rowBytes:(NSUInteger)cols * sizeof(float)
                                           dataType:MPSDataTypeFloat32];
  return [[MPSMatrix alloc] initWithBuffer:buf descriptor:d];
}

// C[M,N] = op(A) @ op(B) with K = interior columns AFTER transposition. The MPSMatrix
// views always describe the STORED (untransposed) layout — transposeLeft/Right tell
// MPS to read the transpose, which is why we no longer materialise x^T / h1^T / W2^T.
Gemm make_gemm(id<MTLDevice> dev, bool tA, bool tB,
               id<MTLBuffer> A, int64_t Arows, int64_t Acols,
               id<MTLBuffer> B, int64_t Brows, int64_t Bcols,
               id<MTLBuffer> C, int64_t M, int64_t N, int64_t K) {
  Gemm g;
  g.mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                           transposeLeft:tA
                                          transposeRight:tB
                                              resultRows:(NSUInteger)M
                                           resultColumns:(NSUInteger)N
                                         interiorColumns:(NSUInteger)K
                                                   alpha:1.0
                                                    beta:0.0];
  g.A = mps_view(A, Arows, Acols);
  g.B = mps_view(B, Brows, Bcols);
  g.C = mps_view(C, M, N);
  return g;
}

// A compute encoder held open across consecutive kernel dispatches and closed only
// when an MPS matmul needs the command buffer (MPS opens its own encoder). This is
// PyTorch MPS's endKernelCoalescing(): 11 encoders per step collapse to 4, and each
// encoder boundary we remove is a GPU pipeline flush we no longer pay for.
struct Coalescer {
  id<MTLCommandBuffer> cmd = nil;
  id<MTLComputeCommandEncoder> enc = nil;

  // Serial dispatch: consecutive dispatches are implicitly ordered, so the
  // read-after-write chains inside a group need no explicit barriers.
  void begin() { if (!enc) enc = [cmd computeCommandEncoder]; }
  void flush() { if (enc) { [enc endEncoding]; enc = nil; } }
};
}  // namespace

struct MLP::Impl {
  MetalContext& ctx;
  std::unique_ptr<KernelLibrary> lib;
  int64_t D, H, C, Bmax;
  int64_t Kmax = 1;          // chunk_steps: max steps per command buffer
  int64_t last_steps = 0;    // steps encoded by the last train_steps call
  int64_t last_batch = 0;

  id<MTLDevice> dev;
  id<MTLCommandQueue> queue;

  // Cached pipeline states (compiled once).
  id<MTLComputePipelineState> pso_bias_add;
  id<MTLComputePipelineState> pso_bias_relu;
  id<MTLComputePipelineState> pso_relu_grad;
  id<MTLComputePipelineState> pso_reduce0;
  id<MTLComputePipelineState> pso_sgd;
  id<MTLComputePipelineState> pso_xent;

  // Parameters + gradients.
  std::shared_ptr<MetalBuffer> W1, b1, W2, b2;
  std::shared_ptr<MetalBuffer> dW1, db1, dW2, db2;
  // Inputs (reused across steps).
  std::shared_ptr<MetalBuffer> x, labels;
  // Forward activations.
  std::shared_ptr<MetalBuffer> z1, h1, logits, probs;
  // Backward temporaries. (No h1T/W2T/xT: MPS transposes in place.)
  std::shared_ptr<MetalBuffer> dlogits, dh1, drelu, loss;
  // loss_sums[slot] = sum of the slot's per-example NLL, reduced on the GPU so the
  // host never walks the batch and last_loss() needs no extra sync.
  std::shared_ptr<MetalBuffer> loss_sums;

  // The five GEMMs of a step, cached per (batch, chunk slot). Only the two GEMMs
  // that read `x` depend on the slot — they view the input buffer at that step's row
  // offset — but a Plan is small and caching whole ones keeps the lookup single.
  // Training touches one or two batch sizes x chunk_steps slots, so this stays tiny.
  struct Plan { Gemm z1, logits, dW2, dh1, dW1; };
  std::unordered_map<int64_t, Plan> plans;

  explicit Impl(MetalContext& c) : ctx(c) {}

  id<MTLComputePipelineState> pso(const char* name) {
    return (__bridge id<MTLComputePipelineState>)lib->pipeline(name);
  }

  const Plan& plan_for(int64_t b, int64_t slot) {
    const int64_t key = b * (Kmax + 1) + slot;
    auto it = plans.find(key);
    if (it != plans.end()) return it->second;

    // This step's slice of the input buffer: `b` rows starting at row slot*b.
    MPSMatrixDescriptor* xd =
        [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)b
                                              columns:(NSUInteger)D
                                             rowBytes:(NSUInteger)D * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    MPSMatrix* xv = [[MPSMatrix alloc]
        initWithBuffer:mtl(x)
                offset:(NSUInteger)(slot * b * D) * sizeof(float)
            descriptor:xd];

    Plan p;
    //                                tA     tB    A                  B                       C          M  N  K
    p.z1     = make_gemm(dev, false, false, mtl(x),       b, D, mtl(W1),      D, H, mtl(z1),      b, H, D);
    p.logits = make_gemm(dev, false, false, mtl(h1),      b, H, mtl(W2),      H, C, mtl(logits),  b, C, H);
    // dW2[H,C] = h1^T @ dlogits — left read transposed, so K = b.
    p.dW2    = make_gemm(dev, true,  false, mtl(h1),      b, H, mtl(dlogits), b, C, mtl(dW2),     H, C, b);
    // dh1[b,H] = dlogits @ W2^T — right read transposed, so K = C.
    p.dh1    = make_gemm(dev, false, true,  mtl(dlogits), b, C, mtl(W2),      H, C, mtl(dh1),     b, H, C);
    // dW1[D,H] = x^T @ drelu — left read transposed, so K = b.
    p.dW1    = make_gemm(dev, true,  false, mtl(x),       b, D, mtl(drelu),   b, H, mtl(dW1),     D, H, b);
    // Point the two x-reading GEMMs at this slot's slice.
    p.z1.A = xv;
    p.dW1.A = xv;
    return plans.emplace(key, p).first->second;
  }

  // Encode one full forward + backward + SGD step for chunk slot `slot` into `co`.
  // Reads rows [slot*b, (slot+1)*b) of the input/label buffers; writes that step's
  // batch-summed loss to loss_sums[slot].
  void encode_step(Coalescer& co, int64_t b, int64_t slot, float lr) {
    const Plan& P = plan_for(b, slot);
    NNDims2 dBH{(uint32_t)b, (uint32_t)H};
    NNDims2 dBC{(uint32_t)b, (uint32_t)C};
    SCEDims sce{(uint32_t)b, (uint32_t)C};
    NNAxpy sgdW1{lr, (uint32_t)(D * H)}, sgdb1{lr, (uint32_t)H};
    NNAxpy sgdW2{lr, (uint32_t)(H * C)}, sgdb2{lr, (uint32_t)C};
    uint32_t nBH = (uint32_t)(b * H);

    // --- forward ---
    co.flush();
    P.z1.encode(co.cmd);                                                 // z1 = x@W1  [b,H]
    encode_1d(co, pso_bias_relu,                                         // h1 = relu(z1+b1)
              {mtl(z1), mtl(b1), mtl(z1), mtl(h1)}, b * H, &dBH, sizeof(dBH));
    co.flush();
    P.logits.encode(co.cmd);                                             // logits=h1@W2 [b,C]

    // --- bias, loss + output grad (dlogits = (softmax - onehot)/b), db2 ---
    // db2 only needs dlogits, so it joins this group rather than opening an encoder.
    encode_1d(co, pso_bias_add, {mtl(logits), mtl(b2), mtl(logits)}, b * C, &dBC, sizeof(dBC));
    encode_xent(co, b, slot, sce);
    encode_reduce0(co, mtl(dlogits), mtl(db2), 0, b, C);                 // db2 [C]
    // Sum this step's per-example NLL into loss_sums[slot] ([b,1] -> [1]).
    encode_reduce0(co, mtl(loss), mtl(loss_sums), slot * sizeof(float), b, 1);

    // --- backward. Both GEMMs read dlogits; dh1 must read W2 before its update. ---
    co.flush();
    P.dW2.encode(co.cmd);                                                // dW2 = h1^T@dlogits [H,C]
    P.dh1.encode(co.cmd);                                                // dh1 = dlogits@W2^T [b,H]
    encode_1d(co, pso_relu_grad, {mtl(z1), mtl(dh1), mtl(drelu)}, b * H, &nBH, sizeof(nBH));
    co.flush();
    P.dW1.encode(co.cmd);                                                // dW1 = x^T@drelu [D,H]

    // --- db1 + SGD update ---
    encode_reduce0(co, mtl(drelu), mtl(db1), 0, b, H);                   // db1 [H]
    encode_1d(co, pso_sgd, {mtl(W1), mtl(dW1)}, D * H, &sgdW1, sizeof(sgdW1));
    encode_1d(co, pso_sgd, {mtl(b1), mtl(db1)}, H, &sgdb1, sizeof(sgdb1));
    encode_1d(co, pso_sgd, {mtl(W2), mtl(dW2)}, H * C, &sgdW2, sizeof(sgdW2));
    encode_1d(co, pso_sgd, {mtl(b2), mtl(db2)}, C, &sgdb2, sizeof(sgdb2));
  }

  // Encode a 1-D compute dispatch into the coalescer's open encoder, opening one if
  // needed (buffers at 0.., push constant at k).
  void encode_1d(Coalescer& co, id<MTLComputePipelineState> pso,
                 const std::vector<id<MTLBuffer>>& bufs, int64_t n_threads,
                 const void* push, size_t push_bytes) {
    if (n_threads <= 0) return;
    co.begin();
    id<MTLComputeCommandEncoder> enc = co.enc;
    [enc setComputePipelineState:pso];
    NSUInteger idx = 0;
    for (id<MTLBuffer> b : bufs) [enc setBuffer:b offset:0 atIndex:idx++];
    if (push && push_bytes) [enc setBytes:push length:push_bytes atIndex:idx++];
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    if ((NSUInteger)n_threads < tg) tg = (NSUInteger)n_threads;
    if (tg == 0) tg = 1;
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n_threads, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  }

  // Softmax cross-entropy for chunk slot `slot`: the label buffer holds the whole
  // chunk, so its binding is offset to this step's rows.
  void encode_xent(Coalescer& co, int64_t b, int64_t slot, const SCEDims& d) {
    co.begin();
    id<MTLComputeCommandEncoder> enc = co.enc;
    [enc setComputePipelineState:pso_xent];
    [enc setBuffer:mtl(logits) offset:0 atIndex:0];
    [enc setBuffer:mtl(labels) offset:(NSUInteger)(slot * b) * sizeof(int32_t) atIndex:1];
    [enc setBuffer:mtl(loss) offset:0 atIndex:2];
    [enc setBuffer:mtl(dlogits) offset:0 atIndex:3];
    [enc setBuffer:mtl(probs) offset:0 atIndex:4];
    [enc setBytes:&d length:sizeof(d) atIndex:5];
    NSUInteger tg = pso_xent.maxTotalThreadsPerThreadgroup;
    if ((NSUInteger)b < tg) tg = (NSUInteger)b;
    [enc dispatchThreads:MTLSizeMake((NSUInteger)b, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg ? tg : 1, 1, 1)];
  }

  // nn_reduce_sum_axis0: one threadgroup of kReduceTG threads per output column.
  // `out_offset` is a byte offset into `out` (used to scatter per-step loss sums).
  void encode_reduce0(Coalescer& co, id<MTLBuffer> a, id<MTLBuffer> out,
                      size_t out_offset, int64_t M, int64_t N) {
    NNDims2 d{(uint32_t)M, (uint32_t)N};
    co.begin();
    id<MTLComputeCommandEncoder> enc = co.enc;
    [enc setComputePipelineState:pso_reduce0];
    [enc setBuffer:a offset:0 atIndex:0];
    [enc setBuffer:out offset:out_offset atIndex:1];
    [enc setBytes:&d length:sizeof(d) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kReduceTG, 1, 1)];
  }
};

MLP::MLP(int64_t in_dim, int64_t hidden, int64_t out_dim, int64_t max_batch,
         int64_t chunk_steps)
    : impl_(std::make_unique<Impl>(MetalContext::instance())) {
  auto& I = *impl_;
  if (in_dim <= 0 || hidden <= 0 || out_dim <= 0 || max_batch <= 0 || chunk_steps <= 0)
    throw std::runtime_error("MLP: dims must be positive");
  I.D = in_dim; I.H = hidden; I.C = out_dim; I.Bmax = max_batch;
  I.Kmax = chunk_steps;

  I.dev = (__bridge id<MTLDevice>)I.ctx.device_handle();
  I.queue = (__bridge id<MTLCommandQueue>)I.ctx.queue_handle();

  I.lib = std::make_unique<KernelLibrary>(I.ctx);
  register_matmul_kernel(*I.lib);  // hand kernel available as fallback
  register_nn_kernels(*I.lib);

  I.pso_bias_add = I.pso("nn_bias_add");
  I.pso_bias_relu = I.pso("nn_bias_relu");
  I.pso_relu_grad = I.pso("nn_relu_grad");
  I.pso_reduce0 = I.pso("nn_reduce_sum_axis0");
  I.pso_sgd = I.pso("nn_sgd_update");
  I.pso_xent = I.pso("nn_softmax_xent");

  const int64_t D = I.D, H = I.H, C = I.C, B = I.Bmax;
  auto& ctx = I.ctx;
  I.W1 = ctx.alloc({D, H}, DType::F32);  I.dW1 = ctx.alloc({D, H}, DType::F32);
  I.b1 = ctx.alloc({H}, DType::F32);     I.db1 = ctx.alloc({H}, DType::F32);
  I.W2 = ctx.alloc({H, C}, DType::F32);  I.dW2 = ctx.alloc({H, C}, DType::F32);
  I.b2 = ctx.alloc({C}, DType::F32);     I.db2 = ctx.alloc({C}, DType::F32);
  // Inputs hold a whole chunk of consecutive minibatches; activations hold one.
  const int64_t R = B * I.Kmax;
  I.x = ctx.alloc({R, D}, DType::F32);   I.labels = ctx.alloc({R}, DType::I32);
  I.loss_sums = ctx.alloc({I.Kmax}, DType::F32);
  I.z1 = ctx.alloc({B, H}, DType::F32);  I.h1 = ctx.alloc({B, H}, DType::F32);
  I.logits = ctx.alloc({B, C}, DType::F32);  I.probs = ctx.alloc({B, C}, DType::F32);
  I.dlogits = ctx.alloc({B, C}, DType::F32); I.dh1 = ctx.alloc({B, H}, DType::F32);
  I.drelu = ctx.alloc({B, H}, DType::F32);
  I.loss = ctx.alloc({B}, DType::F32);
}

MLP::~MLP() = default;

int64_t MLP::in_dim() const { return impl_->D; }
int64_t MLP::hidden() const { return impl_->H; }
int64_t MLP::out_dim() const { return impl_->C; }
int64_t MLP::max_batch() const { return impl_->Bmax; }

void MLP::set_params(const float* W1, const float* b1, const float* W2, const float* b2) {
  auto& I = *impl_;
  std::memcpy(I.W1->contents(), W1, sizeof(float) * (size_t)(I.D * I.H));
  std::memcpy(I.b1->contents(), b1, sizeof(float) * (size_t)I.H);
  std::memcpy(I.W2->contents(), W2, sizeof(float) * (size_t)(I.H * I.C));
  std::memcpy(I.b2->contents(), b2, sizeof(float) * (size_t)I.C);
}

void MLP::get_params(float* W1, float* b1, float* W2, float* b2) const {
  auto& I = *impl_;
  std::memcpy(W1, I.W1->contents(), sizeof(float) * (size_t)(I.D * I.H));
  std::memcpy(b1, I.b1->contents(), sizeof(float) * (size_t)I.H);
  std::memcpy(W2, I.W2->contents(), sizeof(float) * (size_t)(I.H * I.C));
  std::memcpy(b2, I.b2->contents(), sizeof(float) * (size_t)I.C);
}

void MLP::upload_batch(const float* x, const int32_t* labels, int64_t batch) {
  auto& I = *impl_;
  if (batch <= 0 || batch > I.Bmax) throw std::runtime_error("MLP::upload_batch: bad batch");
  std::memcpy(I.x->contents(), x, sizeof(float) * (size_t)(batch * I.D));
  if (labels)
    std::memcpy(I.labels->contents(), labels, sizeof(int32_t) * (size_t)batch);
}

void MLP::forward(int64_t b, float* logits_out) {
  auto& I = *impl_;
  if (b <= 0 || b > I.Bmax) throw std::runtime_error("MLP::forward: bad batch");
  const int64_t D = I.D, H = I.H, C = I.C;
  NNDims2 dh{(uint32_t)b, (uint32_t)H};
  NNDims2 dc{(uint32_t)b, (uint32_t)C};

  // @autoreleasepool drains the MPS descriptors / command buffer / encoders each
  // call — there is no run loop to do it when driven from Python (else they leak).
  @autoreleasepool {
    const Impl::Plan& P = I.plan_for(b, 0);
    Coalescer co{[I.queue commandBuffer], nil};

    P.z1.encode(co.cmd);                                                  // z1 = x@W1
    I.encode_1d(co, I.pso_bias_relu,                                      // h1 = relu(z1+b1)
                {mtl(I.z1), mtl(I.b1), mtl(I.z1), mtl(I.h1)}, b * H, &dh, sizeof(dh));
    co.flush();
    P.logits.encode(co.cmd);                                              // logits = h1@W2
    I.encode_1d(co, I.pso_bias_add,
                {mtl(I.logits), mtl(I.b2), mtl(I.logits)}, b * C, &dc, sizeof(dc));
    co.flush();
    [co.cmd commit];
    [co.cmd waitUntilCompleted];
  }

  std::memcpy(logits_out, I.logits->contents(), sizeof(float) * (size_t)(b * C));
}

int64_t MLP::chunk_steps() const { return impl_->Kmax; }

void MLP::upload_chunk(const float* X, const int32_t* labels, int64_t n_steps,
                       int64_t batch) {
  auto& I = *impl_;
  if (batch <= 0 || batch > I.Bmax || n_steps <= 0 || n_steps > I.Kmax)
    throw std::runtime_error("MLP::upload_chunk: bad n_steps/batch");
  const size_t rows = (size_t)(n_steps * batch);
  std::memcpy(I.x->contents(), X, sizeof(float) * rows * (size_t)I.D);
  if (labels) std::memcpy(I.labels->contents(), labels, sizeof(int32_t) * rows);
}

void MLP::train_steps(int64_t n_steps, int64_t b, float lr) {
  auto& I = *impl_;
  if (b <= 0 || b > I.Bmax) throw std::runtime_error("MLP::train_steps: bad batch");
  if (n_steps <= 0 || n_steps > I.Kmax)
    throw std::runtime_error("MLP::train_steps: bad n_steps");

  // ONE command buffer for all `n_steps` steps: the ~140 us driver round trip is
  // paid once instead of n_steps times, and encoding step i+1 overlaps the GPU
  // executing step i. @autoreleasepool drains the per-call command-buffer/encoder
  // objects (no run loop under Python -> would leak); the MPS GEMMs and their matrix
  // views are cached in `plans`, not rebuilt here.
  @autoreleasepool {
    double t_begin = CFAbsoluteTimeGetCurrent();
    Coalescer co{[I.queue commandBuffer], nil};
    for (int64_t s = 0; s < n_steps; ++s) I.encode_step(co, b, s, lr);
    co.flush();

    if (getenv("JAXMETAL_PROFILE")) {
      double t_enc = CFAbsoluteTimeGetCurrent();
      [co.cmd commit];
      [co.cmd waitUntilCompleted];
      double t_done = CFAbsoluteTimeGetCurrent();
      fprintf(stderr, "steps=%lld encode=%.1fus wait=%.1fus gpu=%.1fus\n",
              (long long)n_steps, (t_enc - t_begin) * 1e6, (t_done - t_enc) * 1e6,
              (co.cmd.GPUEndTime - co.cmd.GPUStartTime) * 1e6);
    } else {
      [co.cmd commit];
      [co.cmd waitUntilCompleted];
    }
  }
  I.last_steps = n_steps;
  I.last_batch = b;
}

float MLP::last_loss() const {
  auto& I = *impl_;
  if (I.last_steps <= 0) return 0.0f;
  // loss_sums[s] is the GPU-reduced sum of step s's per-example NLL.
  const float* ls = static_cast<const float*>(I.loss_sums->contents());
  double sum = 0.0;
  for (int64_t s = 0; s < I.last_steps; ++s) sum += ls[s];
  return static_cast<float>(sum / (double)(I.last_batch * I.last_steps));
}

float MLP::train_step(int64_t b, float lr) {
  train_steps(1, b, lr);
  return last_loss();
}

}  // namespace jaxmetal
