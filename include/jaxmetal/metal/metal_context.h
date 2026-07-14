#pragma once
#include "jaxmetal/metal/dtype.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace jaxmetal {

class MetalBuffer;

// Thrown when no Metal GPU device is available (e.g. a headless/virtualized CI
// runner). Distinct type so callers — notably the test harness — can treat a
// missing device as "skip", separate from a genuine runtime failure.
struct MetalUnavailable : std::runtime_error {
  using std::runtime_error::runtime_error;
};

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
