#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "jaxmetal/ops/mps_matmul.h"

#include "jaxmetal/metal/metal_buffer.h"
#include "jaxmetal/metal/metal_context.h"

namespace jaxmetal {

void mps_matmul_into(MetalContext& ctx, MetalBuffer& a, MetalBuffer& b, MetalBuffer& out,
                     int64_t M, int64_t K, int64_t N) {
  id<MTLDevice> dev = (__bridge id<MTLDevice>)ctx.device_handle();
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ctx.queue_handle();
  id<MTLBuffer> ab = (__bridge id<MTLBuffer>)a.mtl_handle();
  id<MTLBuffer> bb = (__bridge id<MTLBuffer>)b.mtl_handle();
  id<MTLBuffer> cb = (__bridge id<MTLBuffer>)out.mtl_handle();

  const NSUInteger fsz = sizeof(float);
  MPSMatrixDescriptor* ad =
      [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:K * fsz
                                           dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor* bd =
      [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N rowBytes:N * fsz
                                           dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor* cd =
      [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N * fsz
                                           dataType:MPSDataTypeFloat32];

  MPSMatrix* am = [[MPSMatrix alloc] initWithBuffer:ab descriptor:ad];
  MPSMatrix* bm = [[MPSMatrix alloc] initWithBuffer:bb descriptor:bd];
  MPSMatrix* cm = [[MPSMatrix alloc] initWithBuffer:cb descriptor:cd];

  MPSMatrixMultiplication* mm =
      [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                        transposeLeft:NO
                                       transposeRight:NO
                                           resultRows:M
                                        resultColumns:N
                                       interiorColumns:K
                                                alpha:1.0
                                                 beta:0.0];

  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  [mm encodeToCommandBuffer:cmd leftMatrix:am rightMatrix:bm resultMatrix:cm];
  [cmd commit];
  [cmd waitUntilCompleted];
}

}  // namespace jaxmetal
