# Architecture

`jaxmetal` is a from-scratch backend that runs JAX-style workloads on the Apple
GPU through hand-written **Metal Shading Language (MSL)** kernels. It has two
independent execution surfaces — a **matmul** family and a **resident MLP
trainer** — layered on one small Metal runtime, plus a thin Python front-end.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Python (python/jaxmetal)                                                  │
│    __init__.py   matmul(device=), device_name, Mlp, DeviceBuffer           │
│    _capi.py      ctypes binding to libmetal_capi.dylib                      │
│    ffi.py        jax.jit-native matmul via XLA FFI custom call              │
│    data.py       MNIST download / parse       reference.py  NumPy golden    │
│    plugin/       PJRT device scaffold (roadmap)                             │
└───────────────┬──────────────────────────────────────────────────────────┘
                │ ctypes (flat C ABI)
┌───────────────▼──────────────────────────────────────────────────────────┐
│  C ABI   include/jaxmetal/capi/metal_capi.h   (src/capi/metal_capi.mm)     │
│    metal_matmul_* / metal_*_resident   ·   metal_mlp_*   ·   metal_buffer_* │
└───────────────┬──────────────────────────────────────────────────────────┘
                │ C++ (namespace jaxmetal)
┌───────────────▼──────────────────────────────────────────────────────────┐
│  Ops           src/ops/                                                    │
│    mlp.mm        resident 784→H→10 MLP: whole train step in ONE cmd buffer │
│    matmul.mm     register-tiled MSL GEMM      mps_matmul.mm   MPS GEMM      │
│    nn.mm         bias_add · relu(+grad) · reduce · transpose · softmax-XE   │
│    elementwise.mm                             cpu/cpu_matmul.cpp (BLAS ref) │
├────────────────────────────────────────────────────────────────────────── ┤
│  Runtime       src/metal/ + src/runtime/                                    │
│    MetalContext (device+queue, buffer factory)                             │
│    MetalBuffer  (Shared storage → unified-memory, contents() is host ptr)  │
│    KernelLibrary (runtime MSL compile + pipeline-state cache)              │
│    Dispatcher   (1-D + threadgroup-grid encode/dispatch)                   │
└───────────────┬──────────────────────────────────────────────────────────┘
                │ Metal / MPS
┌───────────────▼──────────────────────────────────────────────────────────┐
│  kernels/*.metal   elementwise · matmul · nn   (embedded as strings at     │
│                    build time, compiled at runtime with MTLMathModeSafe)   │
└────────────────────────────────────────────────────────────────────────── ┘
```

## Runtime primitives (`src/metal`, `src/runtime`)

- **`MetalContext`** — owns the single `MTLDevice` + `MTLCommandQueue`; the factory
  for device buffers. Process-wide singleton.
- **`MetalBuffer`** — an `id<MTLBuffer>` with `MTLResourceStorageModeShared`. On
  Apple Silicon's unified memory this means `contents()` is a valid CPU pointer and
  host↔device transfer is a plain `memcpy` — no blit encoder, no staging.
- **`KernelLibrary`** — compiles MSL modules at runtime (`newLibraryWithSource:`,
  no `metal` CLI / `.metallib` needed) and caches `MTLComputePipelineState` by
  function name.
- **`Dispatcher`** — encodes a compute pass: binds buffers at indices `0..k-1`,
  push constants at index `k`, and dispatches a 1-D or threadgroup grid.

Kernels compile with **`MTLMathModeSafe`**, so `+ − × ÷` are IEEE-exact against the
CPU/NumPy reference and `exp`/`log` are within ~1e-4 — this is what makes bit-level
parity testing possible.

## The resident MLP (`src/ops/mlp.mm`) — the performance core

A `784→H→10` MLP (ReLU hidden, softmax cross-entropy, plain SGD) keeps **every**
tensor — parameters, gradients, activations, and the current minibatch — resident
on the GPU across the whole training loop. Parameters are uploaded once via
`set_params` so init matches the golden reference exactly.

The load-bearing design decision is **one command buffer per training step**:

- The naive approach — a separate command buffer per op, plus MPS's own internal
  `waitUntilCompleted` — incurs ~15 commits and 5 host stalls per step and *loses*
  to the CPU.
- Instead, `train_step` encodes the full forward + backward + SGD sequence
  (MPS `encodeToCommandBuffer:` for the GEMMs, compute encoders for the NN kernels)
  into a **single command buffer**, commits once, and `waitUntilCompleted` once.
  Metal's automatic intra-command-buffer hazard tracking orders the ~19 encoders
  correctly (verified: 30× fixed-input runs are bit-identical). Only the scalar
  loss is read back.

Backward uses the "normalize to `[M,K]×[K,N]`" strategy: three explicit transpose
kernels materialize `h1ᵀ`, `W2ᵀ`, `xᵀ` so every gradient is a plain GEMM. The
`1/B` averaging is baked once into `dlogits` by the fused softmax-XE kernel.

`@autoreleasepool` wraps each step's command-buffer body so the transient MPS /
encoder objects don't accumulate when driven from a Python loop (no run loop to
drain the pool otherwise).

## Two JAX integration paths

- **XLA FFI custom call** (`jaxmetal.ffi`) — `matmul` lowers to an FFI custom call
  that dispatches to Metal/MPS, so it traces and composes inside `jax.jit`. Runs on
  the CPU backend and copies to the GPU per call: native *integration*, not a native
  *device*.
- **PJRT plugin** (`jaxmetal.plugin`, roadmap) — a real `metal` device so
  `jax.device_put(x, metal)` keeps arrays GPU-resident and `jax.jit(f,
  backend='metal')` runs there. This is where residency (and the GPU win) would be
  free for arbitrary JAX programs. See [PJRT_PLUGIN.md](PJRT_PLUGIN.md).

## Numerical verification strategy

1. **Kernel unit tests** (`tests/cpp/nn_test.cpp`, `mlp_test.cpp`) — each kernel and
   the full MLP step vs a double-precision CPU reference.
2. **Golden reference** (`python/jaxmetal/reference.py`) — a pure-NumPy MLP that
   matches `jax.grad` to ~1e-8; the GPU matches it to ~1e-7.
3. **Correctness gate** (`tests/python/test_mlp_gate.py`) — GPU vs golden across
   seeds × batch × hidden, run before every training run and in CI.

## Build

CMake (Ninja, ObjC++). `kernels/*.metal` are embedded as C++ string headers at
configure time by `cmake/EmbedMetal.cmake`, then compiled from source **at runtime**
(Command Line Tools have no `metal` compiler). Targets: `metal_rt` (static runtime),
`libmetal_capi.dylib` (C ABI + optional FFI handler), `kernel_tests`, `arith_demo`.
