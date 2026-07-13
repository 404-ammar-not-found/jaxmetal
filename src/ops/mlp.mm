#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "jaxmetal/ops/mlp.h"

#include "jaxmetal/metal/dtype.h"
#include "jaxmetal/metal/kernel_library.h"
#include "jaxmetal/metal/metal_buffer.h"
#include "jaxmetal/metal/metal_context.h"
#include "jaxmetal/ops/matmul.h"  // register_matmul_kernel (fallback / A-vs-B)
#include "jaxmetal/ops/nn.h"      // register_nn_kernels

#include <cstring>
#include <stdexcept>
#include <vector>

namespace jaxmetal {

namespace {
// Push-constant structs, byte-matching kernels/nn.metal.
struct NNDims2 { uint32_t M; uint32_t N; };
struct NNAxpy  { float lr; uint32_t n; };
struct SCEDims { uint32_t B; uint32_t C; };

inline id<MTLBuffer> mtl(const std::shared_ptr<MetalBuffer>& b) {
  return (__bridge id<MTLBuffer>)b->mtl_handle();
}
}  // namespace

struct MLP::Impl {
  MetalContext& ctx;
  std::unique_ptr<KernelLibrary> lib;
  int64_t D, H, C, Bmax;

  id<MTLDevice> dev;
  id<MTLCommandQueue> queue;

  // Cached pipeline states (compiled once).
  id<MTLComputePipelineState> pso_bias_add;
  id<MTLComputePipelineState> pso_relu;
  id<MTLComputePipelineState> pso_relu_grad;
  id<MTLComputePipelineState> pso_reduce0;
  id<MTLComputePipelineState> pso_transpose;
  id<MTLComputePipelineState> pso_sgd;
  id<MTLComputePipelineState> pso_xent;

  // Parameters + gradients.
  std::shared_ptr<MetalBuffer> W1, b1, W2, b2;
  std::shared_ptr<MetalBuffer> dW1, db1, dW2, db2;
  // Inputs (reused across steps).
  std::shared_ptr<MetalBuffer> x, labels;
  // Forward activations.
  std::shared_ptr<MetalBuffer> z1, h1, logits, probs;
  // Backward temporaries.
  std::shared_ptr<MetalBuffer> dlogits, dh1, drelu, h1T, W2T, xT, loss;

  explicit Impl(MetalContext& c) : ctx(c) {}

  id<MTLComputePipelineState> pso(const char* name) {
    return (__bridge id<MTLComputePipelineState>)lib->pipeline(name);
  }

  // Encode a 1-D compute dispatch into `cmd` (buffers at 0.., push constant at k).
  void encode_1d(id<MTLCommandBuffer> cmd, id<MTLComputePipelineState> pso,
                 const std::vector<id<MTLBuffer>>& bufs, int64_t n_threads,
                 const void* push, size_t push_bytes) {
    if (n_threads <= 0) return;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    NSUInteger idx = 0;
    for (id<MTLBuffer> b : bufs) [enc setBuffer:b offset:0 atIndex:idx++];
    if (push && push_bytes) [enc setBytes:push length:push_bytes atIndex:idx++];
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    if ((NSUInteger)n_threads < tg) tg = (NSUInteger)n_threads;
    if (tg == 0) tg = 1;
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n_threads, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
  }

  // Encode C[M,N] = A[M,K] @ B[K,N] via MPS into `cmd` (no transpose, no wait).
  void encode_matmul(id<MTLCommandBuffer> cmd, id<MTLBuffer> A, id<MTLBuffer> B,
                     id<MTLBuffer> Cc, int64_t M, int64_t K, int64_t N) {
    const NSUInteger fsz = sizeof(float);
    MPSMatrixDescriptor* ad = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K
                                     rowBytes:K * fsz dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* bd = [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N
                                     rowBytes:N * fsz dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* cd = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N
                                     rowBytes:N * fsz dataType:MPSDataTypeFloat32];
    MPSMatrix* am = [[MPSMatrix alloc] initWithBuffer:A descriptor:ad];
    MPSMatrix* bm = [[MPSMatrix alloc] initWithBuffer:B descriptor:bd];
    MPSMatrix* cm = [[MPSMatrix alloc] initWithBuffer:Cc descriptor:cd];
    MPSMatrixMultiplication* mm =
        [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:NO
                                             resultRows:M resultColumns:N interiorColumns:K
                                                  alpha:1.0 beta:0.0];
    [mm encodeToCommandBuffer:cmd leftMatrix:am rightMatrix:bm resultMatrix:cm];
  }
};

MLP::MLP(int64_t in_dim, int64_t hidden, int64_t out_dim, int64_t max_batch)
    : impl_(std::make_unique<Impl>(MetalContext::instance())) {
  auto& I = *impl_;
  if (in_dim <= 0 || hidden <= 0 || out_dim <= 0 || max_batch <= 0)
    throw std::runtime_error("MLP: dims must be positive");
  I.D = in_dim; I.H = hidden; I.C = out_dim; I.Bmax = max_batch;

  I.dev = (__bridge id<MTLDevice>)I.ctx.device_handle();
  I.queue = (__bridge id<MTLCommandQueue>)I.ctx.queue_handle();

  I.lib = std::make_unique<KernelLibrary>(I.ctx);
  register_matmul_kernel(*I.lib);  // hand kernel available as fallback
  register_nn_kernels(*I.lib);

  I.pso_bias_add = I.pso("nn_bias_add");
  I.pso_relu = I.pso("nn_relu");
  I.pso_relu_grad = I.pso("nn_relu_grad");
  I.pso_reduce0 = I.pso("nn_reduce_sum_axis0");
  I.pso_transpose = I.pso("nn_transpose2d");
  I.pso_sgd = I.pso("nn_sgd_update");
  I.pso_xent = I.pso("nn_softmax_xent");

  const int64_t D = I.D, H = I.H, C = I.C, B = I.Bmax;
  auto& ctx = I.ctx;
  I.W1 = ctx.alloc({D, H}, DType::F32);  I.dW1 = ctx.alloc({D, H}, DType::F32);
  I.b1 = ctx.alloc({H}, DType::F32);     I.db1 = ctx.alloc({H}, DType::F32);
  I.W2 = ctx.alloc({H, C}, DType::F32);  I.dW2 = ctx.alloc({H, C}, DType::F32);
  I.b2 = ctx.alloc({C}, DType::F32);     I.db2 = ctx.alloc({C}, DType::F32);
  I.x = ctx.alloc({B, D}, DType::F32);   I.labels = ctx.alloc({B}, DType::I32);
  I.z1 = ctx.alloc({B, H}, DType::F32);  I.h1 = ctx.alloc({B, H}, DType::F32);
  I.logits = ctx.alloc({B, C}, DType::F32);  I.probs = ctx.alloc({B, C}, DType::F32);
  I.dlogits = ctx.alloc({B, C}, DType::F32); I.dh1 = ctx.alloc({B, H}, DType::F32);
  I.drelu = ctx.alloc({B, H}, DType::F32);
  I.h1T = ctx.alloc({H, B}, DType::F32);  I.W2T = ctx.alloc({C, H}, DType::F32);
  I.xT = ctx.alloc({D, B}, DType::F32);   I.loss = ctx.alloc({B}, DType::F32);
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
    id<MTLCommandBuffer> cmd = [I.queue commandBuffer];
    I.encode_matmul(cmd, mtl(I.x), mtl(I.W1), mtl(I.z1), b, D, H);        // z1 = x@W1
    I.encode_1d(cmd, I.pso_bias_add, {mtl(I.z1), mtl(I.b1), mtl(I.z1)}, b * H, &dh, sizeof(dh));
    uint32_t nBH = (uint32_t)(b * H);
    I.encode_1d(cmd, I.pso_relu, {mtl(I.z1), mtl(I.h1)}, b * H, &nBH, sizeof(nBH));
    I.encode_matmul(cmd, mtl(I.h1), mtl(I.W2), mtl(I.logits), b, H, C);   // logits = h1@W2
    I.encode_1d(cmd, I.pso_bias_add, {mtl(I.logits), mtl(I.b2), mtl(I.logits)}, b * C, &dc, sizeof(dc));
    [cmd commit];
    [cmd waitUntilCompleted];
  }

  std::memcpy(logits_out, I.logits->contents(), sizeof(float) * (size_t)(b * C));
}

float MLP::train_step(int64_t b, float lr) {
  auto& I = *impl_;
  if (b <= 0 || b > I.Bmax) throw std::runtime_error("MLP::train_step: bad batch");
  const int64_t D = I.D, H = I.H, C = I.C;

  NNDims2 dBH{(uint32_t)b, (uint32_t)H};
  NNDims2 dBC{(uint32_t)b, (uint32_t)C};
  NNDims2 dHC{(uint32_t)H, (uint32_t)C};
  NNDims2 dBD{(uint32_t)b, (uint32_t)D};
  SCEDims sce{(uint32_t)b, (uint32_t)C};
  NNAxpy sgdW1{lr, (uint32_t)(D * H)}, sgdb1{lr, (uint32_t)H};
  NNAxpy sgdW2{lr, (uint32_t)(H * C)}, sgdb2{lr, (uint32_t)C};
  uint32_t nBH = (uint32_t)(b * H);

  // One command buffer for the whole step; @autoreleasepool drains the per-step
  // MPS/command-buffer/encoder objects (no run loop under Python -> would leak).
  @autoreleasepool {
  id<MTLCommandBuffer> cmd = [I.queue commandBuffer];

  // --- forward ---
  I.encode_matmul(cmd, mtl(I.x), mtl(I.W1), mtl(I.z1), b, D, H);          // z1 = x@W1  [b,H]
  I.encode_1d(cmd, I.pso_bias_add, {mtl(I.z1), mtl(I.b1), mtl(I.z1)}, b * H, &dBH, sizeof(dBH));
  I.encode_1d(cmd, I.pso_relu, {mtl(I.z1), mtl(I.h1)}, b * H, &nBH, sizeof(nBH));  // h1=relu(z1)
  I.encode_matmul(cmd, mtl(I.h1), mtl(I.W2), mtl(I.logits), b, H, C);     // logits=h1@W2 [b,C]
  I.encode_1d(cmd, I.pso_bias_add, {mtl(I.logits), mtl(I.b2), mtl(I.logits)}, b * C, &dBC, sizeof(dBC));

  // --- loss + output grad (dlogits = (softmax - onehot)/b) ---
  I.encode_1d(cmd, I.pso_xent,
              {mtl(I.logits), mtl(I.labels), mtl(I.loss), mtl(I.dlogits), mtl(I.probs)},
              b, &sce, sizeof(sce));

  // --- backward ---
  I.encode_1d(cmd, I.pso_transpose, {mtl(I.h1), mtl(I.h1T)}, b * H, &dBH, sizeof(dBH));  // h1T [H,b]
  I.encode_matmul(cmd, mtl(I.h1T), mtl(I.dlogits), mtl(I.dW2), H, b, C);  // dW2 = h1^T@dlogits [H,C]
  I.encode_1d(cmd, I.pso_reduce0, {mtl(I.dlogits), mtl(I.db2)}, C, &dBC, sizeof(dBC));   // db2 [C]
  I.encode_1d(cmd, I.pso_transpose, {mtl(I.W2), mtl(I.W2T)}, H * C, &dHC, sizeof(dHC));  // W2T [C,H]
  I.encode_matmul(cmd, mtl(I.dlogits), mtl(I.W2T), mtl(I.dh1), b, C, H);  // dh1 = dlogits@W2^T [b,H]
  I.encode_1d(cmd, I.pso_relu_grad, {mtl(I.z1), mtl(I.dh1), mtl(I.drelu)}, b * H, &nBH, sizeof(nBH));
  I.encode_1d(cmd, I.pso_transpose, {mtl(I.x), mtl(I.xT)}, b * D, &dBD, sizeof(dBD));    // xT [D,b]
  I.encode_matmul(cmd, mtl(I.xT), mtl(I.drelu), mtl(I.dW1), D, b, H);     // dW1 = x^T@drelu [D,H]
  I.encode_1d(cmd, I.pso_reduce0, {mtl(I.drelu), mtl(I.db1)}, H, &dBH, sizeof(dBH));     // db1 [H]

  // --- SGD update ---
  I.encode_1d(cmd, I.pso_sgd, {mtl(I.W1), mtl(I.dW1)}, D * H, &sgdW1, sizeof(sgdW1));
  I.encode_1d(cmd, I.pso_sgd, {mtl(I.b1), mtl(I.db1)}, H, &sgdb1, sizeof(sgdb1));
  I.encode_1d(cmd, I.pso_sgd, {mtl(I.W2), mtl(I.dW2)}, H * C, &sgdW2, sizeof(sgdW2));
  I.encode_1d(cmd, I.pso_sgd, {mtl(I.b2), mtl(I.db2)}, C, &sgdb2, sizeof(sgdb2));

  [cmd commit];
  [cmd waitUntilCompleted];
  }  // @autoreleasepool

  // Mean loss over the batch (loss_ holds per-example NLL).
  const float* lh = static_cast<const float*>(I.loss->contents());
  double sum = 0.0;
  for (int64_t i = 0; i < b; ++i) sum += lh[i];
  return static_cast<float>(sum / static_cast<double>(b));
}

}  // namespace jaxmetal
