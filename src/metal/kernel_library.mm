#import <Metal/Metal.h>

#include "jaxmetal/metal/kernel_library.h"
#include "jaxmetal/metal/metal_context.h"

#include <stdexcept>
#include <unordered_map>

namespace jaxmetal {

struct KernelLibrary::Impl {
  MetalContext* ctx;
  NSMutableArray<id<MTLLibrary>>* libs;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
};

KernelLibrary::KernelLibrary(MetalContext& ctx) : impl_(std::make_unique<Impl>()) {
  impl_->ctx = &ctx;
  impl_->libs = [NSMutableArray array];
}

KernelLibrary::~KernelLibrary() = default;

void KernelLibrary::add_source(const std::string& name, const std::string& msl) {
  id<MTLDevice> dev = (__bridge id<MTLDevice>)impl_->ctx->device_handle();
  NSError* err = nil;
  NSString* src = [NSString stringWithUTF8String:msl.c_str()];
  MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
  // Disable fast-math so arithmetic (notably division) is IEEE-correct and
  // matches the JAX CPU reference. `mathMode`/`MTLMathModeSafe` only exist in
  // the macOS 15+ SDK; older SDKs (e.g. Xcode 15.4 CI) fall back to the
  // deprecated-but-equivalent `fastMathEnabled = NO`.
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
  if (@available(macOS 15.0, *)) {
    opts.mathMode = MTLMathModeSafe;
  } else {
    opts.fastMathEnabled = NO;
  }
#else
  opts.fastMathEnabled = NO;
#endif
  id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
  if (!lib) {
    std::string msg = "Failed to compile MSL module '" + name + "': " +
                      (err ? std::string(err.localizedDescription.UTF8String) : "unknown error");
    throw std::runtime_error(msg);
  }
  [impl_->libs addObject:lib];
}

void* KernelLibrary::pipeline(const std::string& fn_name) {
  auto it = impl_->pipelines.find(fn_name);
  if (it != impl_->pipelines.end()) return (__bridge void*)it->second;

  id<MTLDevice> dev = (__bridge id<MTLDevice>)impl_->ctx->device_handle();
  NSString* name = [NSString stringWithUTF8String:fn_name.c_str()];
  for (id<MTLLibrary> lib in impl_->libs) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) continue;
    NSError* err = nil;
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
      std::string msg = "Failed to create pipeline for '" + fn_name + "': " +
                        (err ? std::string(err.localizedDescription.UTF8String) : "unknown error");
      throw std::runtime_error(msg);
    }
    impl_->pipelines.emplace(fn_name, pso);
    return (__bridge void*)pso;
  }
  throw std::runtime_error("Kernel function not found: " + fn_name);
}

}  // namespace jaxmetal
