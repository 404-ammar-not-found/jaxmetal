#import <Metal/Metal.h>

#include "jaxmetal/runtime/dispatcher.h"
#include "jaxmetal/metal/metal_buffer.h"
#include "jaxmetal/metal/metal_context.h"

namespace jaxmetal {

struct Dispatcher::Impl {
  MetalContext* ctx;
  id<MTLCommandBuffer> last;  // most recently submitted, for wait()
};

Dispatcher::Dispatcher(MetalContext& ctx) : impl_(std::make_unique<Impl>()) {
  impl_->ctx = &ctx;
  impl_->last = nil;
}

Dispatcher::~Dispatcher() = default;

void Dispatcher::dispatch_1d(void* pipeline_state,
                             const std::vector<MetalBuffer*>& buffers,
                             int64_t n_threads,
                             const void* push_constants,
                             size_t push_constants_bytes) {
  if (n_threads <= 0) return;

  id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pipeline_state;
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)impl_->ctx->queue_handle();

  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pso];

  NSUInteger idx = 0;
  for (MetalBuffer* b : buffers) {
    id<MTLBuffer> mb = (__bridge id<MTLBuffer>)b->mtl_handle();
    [enc setBuffer:mb offset:0 atIndex:idx++];
  }
  if (push_constants && push_constants_bytes > 0) {
    [enc setBytes:push_constants length:push_constants_bytes atIndex:idx++];
  }

  // One thread per element. dispatchThreads: uses non-uniform threadgroups
  // (supported on Apple GPUs), so it never launches threads past n_threads.
  NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
  if (static_cast<NSUInteger>(n_threads) < tg) tg = static_cast<NSUInteger>(n_threads);
  if (tg == 0) tg = 1;
  MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(n_threads), 1, 1);
  MTLSize threadgroup = MTLSizeMake(tg, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:threadgroup];

  [enc endEncoding];
  [cmd commit];
  impl_->last = cmd;
}

void Dispatcher::dispatch_threadgroups(void* pipeline_state,
                                       const std::vector<MetalBuffer*>& buffers,
                                       int64_t groups_x, int64_t groups_y, int64_t groups_z,
                                       int64_t tg_x, int64_t tg_y, int64_t tg_z,
                                       const void* push_constants,
                                       size_t push_constants_bytes) {
  if (groups_x <= 0 || groups_y <= 0 || groups_z <= 0) return;

  id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pipeline_state;
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)impl_->ctx->queue_handle();

  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pso];

  NSUInteger idx = 0;
  for (MetalBuffer* b : buffers) {
    id<MTLBuffer> mb = (__bridge id<MTLBuffer>)b->mtl_handle();
    [enc setBuffer:mb offset:0 atIndex:idx++];
  }
  if (push_constants && push_constants_bytes > 0) {
    [enc setBytes:push_constants length:push_constants_bytes atIndex:idx++];
  }

  MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(groups_x),
                             static_cast<NSUInteger>(groups_y),
                             static_cast<NSUInteger>(groups_z));
  MTLSize threadgroup = MTLSizeMake(static_cast<NSUInteger>(tg_x),
                                    static_cast<NSUInteger>(tg_y),
                                    static_cast<NSUInteger>(tg_z));
  [enc dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];

  [enc endEncoding];
  [cmd commit];
  impl_->last = cmd;
}

void Dispatcher::wait() {
  if (impl_->last) [impl_->last waitUntilCompleted];
}

}  // namespace jaxmetal
