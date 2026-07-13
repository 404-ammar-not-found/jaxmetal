#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace jaxmetal {

class MetalContext;
class MetalBuffer;

// Encodes and submits compute passes to the GPU. Kept deliberately small: a 1-D
// dispatch that binds buffers in order, appends optional push-constant bytes at
// the next binding index, and launches one thread per element.
class Dispatcher {
 public:
  explicit Dispatcher(MetalContext& ctx);
  ~Dispatcher();

  // Bind `buffers` at indices 0..k-1; if push_constants != null, bind those
  // bytes at index k (via setBytes:). Launch `n_threads` threads (1-D grid).
  // Submits the command buffer but does not block; call wait() to synchronize.
  void dispatch_1d(void* pipeline_state,
                   const std::vector<MetalBuffer*>& buffers,
                   int64_t n_threads,
                   const void* push_constants = nullptr,
                   size_t push_constants_bytes = 0);

  // Launch a grid of `groups_{x,y,z}` threadgroups, each `tg_{x,y,z}` threads
  // (dispatchThreadgroups:). Used by tiled kernels that rely on a fixed,
  // uniform threadgroup shape for threadgroup memory + barriers; the kernel is
  // responsible for guarding global-index bounds. Same buffer/push-constant
  // binding convention as dispatch_1d.
  void dispatch_threadgroups(void* pipeline_state,
                             const std::vector<MetalBuffer*>& buffers,
                             int64_t groups_x, int64_t groups_y, int64_t groups_z,
                             int64_t tg_x, int64_t tg_y, int64_t tg_z,
                             const void* push_constants = nullptr,
                             size_t push_constants_bytes = 0);

  // Block until the most recently submitted command buffer completes.
  void wait();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace jaxmetal
