#pragma once
#include <memory>
#include <string>

namespace jaxmetal {

class MetalContext;

// Compiles MSL source modules at runtime (newLibraryWithSource:) and caches the
// resulting compute pipeline states by function name. Look up a kernel with
// pipeline("fn_name"); the returned void* is an id<MTLComputePipelineState>.
class KernelLibrary {
 public:
  explicit KernelLibrary(MetalContext& ctx);
  ~KernelLibrary();

  // Compile an MSL module. `name` is only used in error messages. All functions
  // in the module become discoverable via pipeline().
  void add_source(const std::string& name, const std::string& msl);

  // Return (and cache) the compute pipeline state for a kernel function.
  // Throws std::runtime_error if the function is not found in any module.
  void* pipeline(const std::string& fn_name);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace jaxmetal
