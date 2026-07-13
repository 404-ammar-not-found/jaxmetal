// XLA FFI custom-call handler: makes our Metal matmul callable from *inside* a
// jitted JAX program via jax.ffi.ffi_call. This is the modern, supported native
// integration (jaxlib ships xla/ffi/api/ffi.h). It runs on the CPU backend's
// custom-call path — JAX hands us host pointers — and we dispatch to MPS on the
// GPU. (A separate `mps` *device* is the PJRT-plugin path; see docs/PJRT_PLUGIN.md.)

#include "xla/ffi/api/ffi.h"

#include "jaxmetal/capi/metal_capi.h"

#include <cstdint>

namespace ffi = xla::ffi;

static ffi::Error MetalMatmulImpl(ffi::BufferR2<ffi::F32> a,
                                  ffi::BufferR2<ffi::F32> b,
                                  ffi::ResultBufferR2<ffi::F32> out) {
  const int64_t M = a.dimensions()[0];
  const int64_t K = a.dimensions()[1];
  const int64_t K2 = b.dimensions()[0];
  const int64_t N = b.dimensions()[1];
  if (K != K2) return ffi::Error::InvalidArgument("matmul inner dimensions do not match");

  // GPU via MPS (Apple's tuned kernel). Host pointers -> device -> host inside.
  const int rc = metal_mps_matmul_f32(a.typed_data(), b.typed_data(),
                                      out->typed_data(), M, K, N);
  if (rc != 0) return ffi::Error::Internal("metal_mps_matmul_f32 failed");
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(MetalMatmul, MetalMatmulImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::BufferR2<ffi::F32>>()   // a [M,K]
                                  .Arg<ffi::BufferR2<ffi::F32>>()   // b [K,N]
                                  .Ret<ffi::BufferR2<ffi::F32>>()); // out [M,N]

// Returns the handler's address so Python can wrap it in a PyCapsule and register
// it with jax.ffi.register_ffi_target.
extern "C" void* metal_ffi_matmul_handler() {
  return reinterpret_cast<void*>(&MetalMatmul);
}
