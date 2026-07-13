# Native JAX on Metal — two integration paths

There are two ways to get JAX to run on our Metal backend. They are different in
kind, and it's worth being precise about what each gives you.

## 1. XLA FFI custom call — DONE (works today)

`jaxmetal.ffi.matmul(a, b)` lowers to an **XLA FFI custom call** that dispatches
to our Metal/MPS matmul. It is **jittable** and composes with native JAX ops:

```python
@jax.jit
def f(a, b):
    c = jaxmetal.ffi.matmul(a, b)   # runs on Metal (MPS), inside jit
    return jnp.tanh(c) + 1.0        # native JAX ops around it
```

- Mechanism: `jax.ffi.register_ffi_target` + `jax.ffi.ffi_call` (jaxlib ships the
  headers at `jax.ffi.include_dir()/xla/ffi/api/ffi.h`).
- Handler: `src/ffi/metal_ffi.cpp` (`XLA_FFI_DEFINE_HANDLER_SYMBOL`), built into
  `libmetal_capi.dylib` when configured with `-DJAX_FFI_INCLUDE_DIR=...`.
- Runs on the **CPU backend's** custom-call path: JAX hands the handler *host*
  pointers; we copy to the GPU, run MPS, copy back. So it's native *integration*
  (tracing/jit/composition) but not a distinct device — data lives on CPU and we
  pay a copy per call.

This is the pragmatic, supported "call native code from JAX" path, and it works.

## 2. PJRT plugin — the native `mps` *device* (Stage 3, in progress)

A PJRT (Pluggable JAX RunTime) C-API plugin makes JAX expose a real `metal`
device: `jax.devices("metal")`, `jax.device_put(x, dev)`, and then `jnp.matmul`
(and everything else) runs there. Arrays stay **resident** on the GPU across ops —
which is exactly where the MPS speed win lives (no per-op copies; see the
benchmark, where resident MPS beats CPU 1.5–1.8×).

### Status & why it's larger
- **Registration APIs are present** in this jaxlib (verified):
  `jax._src.xla_bridge.register_plugin`, `jaxlib.xla_client.load_pjrt_plugin_dynamically`,
  `jaxlib.xla_client.make_c_api_client`. Scaffold: `python/jax_metal_plugin/`.
- **Blocker:** jaxlib does **not** bundle `pjrt_c_api.h`. It must be vendored from
  `openxla/xla` (`xla/pjrt/c/pjrt_c_api.h`) at the commit matching this jaxlib, and
  the plugin's `PJRT_Api.pjrt_api_version` must match or JAX refuses to load it.
- **Surface to implement** behind `extern "C" const PJRT_Api* GetPjrtApi()`:
  - Plugin/errors: `PJRT_Plugin_Initialize`, `PJRT_Plugin_Attributes`, `PJRT_Error_*`.
  - Client/device: `PJRT_Client_{Create,Destroy,PlatformName,Devices,AddressableDevices,LookupDevice,Compile}`,
    `PJRT_Device*`, `PJRT_DeviceDescription_*`.
  - Buffers (MTLBuffer-backed): `PJRT_Client_BufferFromHostBuffer`,
    `PJRT_Buffer_{ToHostBuffer,Delete,Destroy,Dimensions,ElementType,...}`.
  - Executable: `PJRT_LoadedExecutable_{Execute,Destroy}`, `PJRT_Executable_*`, `PJRT_Event_*`.
  - **`PJRT_Client_Compile`** receives a `PJRT_Program` = **StableHLO** (MLIR).
    Parsing StableHLO in C++ is the central task (hand-rolled subset parser for the
    op set we support; see CLAUDE.md R1). For matmul the op is `stablehlo.dot_general`.

### Incremental plan
1. Vendor `pjrt_c_api.h` (version-matched); add a `libpjrt_metal.dylib` CMake target.
2. Implement the client/device/buffer surface + a synchronous `Execute`; make
   `jax.devices("metal")` enumerate one device.
3. `Compile`: parse a minimal StableHLO subset (`dot_general`, elementwise) and
   build a schedule over our existing kernels (`ops/`, `ops/mps_matmul`).
4. Enable `python/jax_metal_plugin/initialize()` and ship the `jax_plugins` entry point.

Until step 4, `jaxmetal.matmul(..., device="mps")` and `jaxmetal.ffi.matmul` are
the working entry points.
