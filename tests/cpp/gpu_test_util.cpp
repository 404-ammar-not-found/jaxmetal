#include "gpu_test_util.h"

#include "jaxmetal/metal/metal_context.h"
#include "jaxmetal/metal/kernel_library.h"
#include "jaxmetal/ops/elementwise.h"

namespace testutil {

jaxmetal::MetalContext& ctx() { return jaxmetal::MetalContext::instance(); }

jaxmetal::KernelLibrary& lib() {
  static jaxmetal::KernelLibrary l(jaxmetal::MetalContext::instance());
  static int once = (jaxmetal::register_elementwise_kernels(l), 0);
  (void)once;
  return l;
}

}  // namespace testutil
