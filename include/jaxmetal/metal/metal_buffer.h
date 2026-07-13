#pragma once
#include "jaxmetal/metal/dtype.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace jaxmetal {

// A device buffer backed by an id<MTLBuffer> (Shared storage mode → the same
// bytes are visible to CPU and GPU on Apple Silicon's unified memory, so
// host<->device transfer is a plain memcpy on contents()).
//
// Constructed only via MetalContext::alloc / MetalContext::from_host. The
// Objective-C internals live behind a PIMPL so this header stays pure C++ and
// can be included from .cc translation units (the future compiler/ layer).
class MetalBuffer {
 public:
  ~MetalBuffer();
  MetalBuffer(const MetalBuffer&) = delete;
  MetalBuffer& operator=(const MetalBuffer&) = delete;

  void* contents();              // host-visible pointer (Shared storage)
  const void* contents() const;
  size_t size_bytes() const;
  int64_t num_elements() const;
  DType dtype() const;
  const std::vector<int64_t>& shape() const;

  // id<MTLBuffer> bridged to void* for use inside .mm files (Dispatcher).
  void* mtl_handle() const;

 private:
  friend class MetalContext;
  struct Impl;
  explicit MetalBuffer(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace jaxmetal
