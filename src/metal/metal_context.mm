#import <Metal/Metal.h>

#include "jaxmetal/metal/metal_context.h"

#include <stdexcept>

namespace jaxmetal {

struct MetalContext::Impl {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
};

MetalContext::MetalContext() : impl_(std::make_unique<Impl>()) {
  impl_->device = MTLCreateSystemDefaultDevice();
  if (!impl_->device) throw std::runtime_error("No Metal device found");
  impl_->queue = [impl_->device newCommandQueue];
  if (!impl_->queue) throw std::runtime_error("Failed to create MTLCommandQueue");
}

MetalContext::~MetalContext() = default;

MetalContext& MetalContext::instance() {
  static MetalContext ctx;
  return ctx;
}

std::string MetalContext::device_name() const {
  return std::string(impl_->device.name.UTF8String);
}

void* MetalContext::device_handle() const { return (__bridge void*)impl_->device; }
void* MetalContext::queue_handle() const { return (__bridge void*)impl_->queue; }

// NOTE: alloc()/from_host() are defined in metal_buffer.mm, which has the
// definition of MetalBuffer::Impl.

}  // namespace jaxmetal
