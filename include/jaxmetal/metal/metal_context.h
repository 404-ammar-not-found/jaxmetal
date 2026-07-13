#pragma once
#include "jaxmetal/metal/dtype.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace jaxmetal {

class MetalBuffer;

// Owns the single MTLDevice + MTLCommandQueue used by the runtime, and is the
// factory for device buffers. Process-wide singleton for now (one GPU).
class MetalContext {
 public:
  static MetalContext& instance();

  std::string device_name() const;
  void* device_handle() const;  // id<MTLDevice>  bridged to void*
  void* queue_handle() const;   // id<MTLCommandQueue> bridged to void*

  // Allocate an uninitialized device buffer of the given shape/dtype.
  std::shared_ptr<MetalBuffer> alloc(std::vector<int64_t> shape, DType dt);
  // Allocate and copy `data` (size_bytes worth) from host into the buffer.
  std::shared_ptr<MetalBuffer> from_host(const void* data,
                                         std::vector<int64_t> shape, DType dt);

  MetalContext(const MetalContext&) = delete;
  MetalContext& operator=(const MetalContext&) = delete;

 private:
  MetalContext();
  ~MetalContext();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace jaxmetal
