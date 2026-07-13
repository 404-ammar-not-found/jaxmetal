#import <Metal/Metal.h>

#include "jaxmetal/metal/metal_buffer.h"
#include "jaxmetal/metal/metal_context.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace jaxmetal {

struct MetalBuffer::Impl {
  id<MTLBuffer> buffer;
  std::vector<int64_t> shape;
  DType dtype;
  int64_t num_elements;
  size_t size_bytes;
};

MetalBuffer::MetalBuffer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MetalBuffer::~MetalBuffer() = default;

void* MetalBuffer::contents() { return impl_->buffer.contents; }
const void* MetalBuffer::contents() const { return impl_->buffer.contents; }
size_t MetalBuffer::size_bytes() const { return impl_->size_bytes; }
int64_t MetalBuffer::num_elements() const { return impl_->num_elements; }
DType MetalBuffer::dtype() const { return impl_->dtype; }
const std::vector<int64_t>& MetalBuffer::shape() const { return impl_->shape; }
void* MetalBuffer::mtl_handle() const { return (__bridge void*)impl_->buffer; }

static int64_t elem_count(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;  // empty shape -> scalar (n == 1)
}

// --- MetalContext buffer factories (defined here for access to Impl) ---

std::shared_ptr<MetalBuffer> MetalContext::alloc(std::vector<int64_t> shape, DType dt) {
  auto impl = std::make_unique<MetalBuffer::Impl>();
  impl->shape = std::move(shape);
  impl->dtype = dt;
  impl->num_elements = elem_count(impl->shape);
  impl->size_bytes = static_cast<size_t>(impl->num_elements) * dtype_size(dt);

  id<MTLDevice> dev = (__bridge id<MTLDevice>)device_handle();
  impl->buffer = [dev newBufferWithLength:std::max<size_t>(impl->size_bytes, 1)
                                  options:MTLResourceStorageModeShared];
  if (!impl->buffer) throw std::runtime_error("Failed to allocate MTLBuffer");

  return std::shared_ptr<MetalBuffer>(new MetalBuffer(std::move(impl)));
}

std::shared_ptr<MetalBuffer> MetalContext::from_host(const void* data,
                                                     std::vector<int64_t> shape, DType dt) {
  auto buf = alloc(std::move(shape), dt);
  std::memcpy(buf->contents(), data, buf->size_bytes());
  return buf;
}

}  // namespace jaxmetal
