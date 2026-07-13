#pragma once
// Shared GPU fixtures for tests: a process-wide MetalContext and a KernelLibrary
// with the elementwise kernels compiled once (MSL compilation is not free, so we
// avoid recompiling per test).

namespace jaxmetal {
class MetalContext;
class KernelLibrary;
}  // namespace jaxmetal

namespace testutil {

jaxmetal::MetalContext& ctx();
jaxmetal::KernelLibrary& lib();  // elementwise kernels pre-registered

}  // namespace testutil
