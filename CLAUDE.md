# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# jax-metal-prototype — Ground Truth

> **Project:** A learning-focused, compiler-level prototype that runs **JAX on the Apple Metal GPU**
> via hand-written **Metal Shading Language (MSL)** kernels, growing toward a real JAX backend.
> This file holds the goals, verified environment baselines, architecture, and staged roadmap.

> **Status (2026-07-13):** The **matmul foundation is implemented and tested** — three matmul
> backends (hand-written MSL kernel, MPS, Accelerate CPU), a `jaxmetal` Python front-end with full
> `jnp.matmul` semantics, and **native `jax.jit` integration via an XLA FFI custom call**.
>
> **A full resident-GPU MLP trainer is now implemented and verified** (the Stage-4 *learning*
> milestone, reached via a direct **resident C-ABI** path rather than the PJRT device): hand-written
> MSL NN kernels (`kernels/nn.metal`) + a `mtlrt::MLP` class (`src/ops/mlp.mm`) that runs the entire
> `784→H→10` forward + backward + SGD step on-device (MPS matmuls + custom kernels, params never
> leave the GPU). It **trains MNIST to 98.15% test accuracy** (verified 2026-08-13). Numerically it
> matches a NumPy golden reference (which matches `jax.grad`) to ~1e-7. Driver:
> `examples/train_mnist.py` (gate + benchmark + calibrate + train); C-ABI: `metal_mlp_*` in
> `include/jaxmetal/capi/metal_capi.h`; golden ref: `python/jaxmetal/reference.py`.
>
> **Per-step cost was overhead-dominated, and is no longer** (fixed 2026-08-13). One command buffer
> *per step* meant paying a ~140 µs driver round trip plus ~100 µs of CPU encoding on every step —
> a ~0.5 ms floor that swamped the actual compute. `MLP::train_steps` now encodes up to
> `chunk_steps` consecutive SGD steps into **one** command buffer. Combined with cached MPS objects,
> MPS transpose flags (no materialised x^T/h1^T/W2^T), coalesced encoders, and a properly parallel
> `nn_reduce_sum_axis0`, the per-step time at hidden=128 dropped **~4.4×** (batch 128: 0.619 →
> 0.137 ms) and the GPU-beats-CPU crossover moved from **batch >2048 to batch ~60**. See §7.
>
> **`Mlp(device=...)`** — the GPU does *not* always win, so the Python trainer routes:
> `"auto"` (default) picks via a measured cost model, `"gpu"`/`"cpu"` force an arm, and `.device`
> reports the choice. The CPU arm is `jaxmetal.reference` (NumPy on Accelerate BLAS), so it adds no
> new numerics. Re-fit the model on other hardware with `--calibrate`; override with
> `JAXMETAL_MLP_DEVICE=gpu|cpu`.
>
> The **PJRT plugin** (a real `mps` *device* so `jax.jit(f, backend='metal')` runs natively) and the
> **StableHLO→kernel compiler** (Stages 2–3 below) remain the forward roadmap — **not yet built**.
> The resident MLP is the pragmatic route to the training goal; PJRT is the "textbook" integration.
> Where §1/§4 describe those stages as future work, they are the plan, not a description of what
> exists. When editing, keep this status line, §7, and `README.md` in sync.

---

## 0. Operational quickref (read this first)

```bash
# One-time env (system Python 3.14 is too new for jaxlib — use isolated 3.12)
brew install cmake ninja
uv venv --python 3.12 .venv
uv pip install --python .venv numpy "jax[cpu]"

# Configure + build. Add -DJAX_FFI_INCLUDE_DIR=... to also build the XLA FFI handler
# (needed for jaxmetal.ffi / jax.jit integration; omit for kernels/CPU/MPS only).
cmake -S . -B build -G Ninja \
  -DJAX_FFI_INCLUDE_DIR=$(.venv/bin/python -c "import jax.ffi; print(jax.ffi.include_dir())")
cmake --build build            # -> build/libmetal_capi.dylib, arith_demo, kernel_tests

# Tests
ctest --test-dir build --output-on-failure   # 68 C++ unit tests (one CTest case per TEST(...))
ctest --test-dir build -R MatmulTiled -V      # run a single test by its TEST(Name)
.venv/bin/python python/test_jaxmetal.py      # jaxmetal vs jnp.matmul, all shapes × backends

# Smoke / demos
./build/arith_demo                            # Stage 0/1 GPU-vs-CPU smoke demo
.venv/bin/python examples/03_resident_speed.py  # the GPU-vs-CPU matmul crossover

# Resident-GPU MLP on MNIST (98.15% test acc; beats JAX-CPU from batch ~60 at hidden=128)
.venv/bin/python examples/train_mnist.py                        # gate + benchmark + full train
.venv/bin/python examples/train_mnist.py --gate-only            # GPU fwd/bwd vs numpy golden ref
.venv/bin/python examples/train_mnist.py --bench-only --batch 1024 --hidden 1024   # GPU vs JAX-CPU step
.venv/bin/python examples/train_mnist.py --calibrate            # sweep both arms, re-fit device="auto"
.venv/bin/python examples/train_mnist.py --device cpu           # force an arm (auto|gpu|cpu)
.venv/bin/python examples/train_mnist.py --batch 512 --hidden 1024 --lr 0.5 --epochs 25  # 98.15% acc
.venv/bin/python python/jaxmetal/reference.py                  # verify the golden ref vs jax.grad
.venv/bin/python tests/python/test_mlp_gate.py                 # network-free GPU-vs-ref gate
.venv/bin/python tests/python/test_mlp_auto.py                 # chunked == per-step; router vs clock
.venv/bin/python benchmarks/bench_reduce.py                    # compensated f32 sum: accuracy + GB/s
JAXMETAL_PROFILE=1 .venv/bin/python examples/train_mnist.py --bench-only  # encode vs wait vs GPU us
```

> **Layout note (post-restructure):** C++ namespace is `jaxmetal` (was `mtlrt`); public headers
> live in `include/jaxmetal/{metal,runtime,ops,cpu,capi}/`, implementations in `src/`. Python is
> one package `python/jaxmetal/` (`_capi` ctypes, `ffi`, `data`, `reference`, `plugin`); examples
> in `examples/`, python tests in `tests/python/`. `pyproject.toml` enables `uv pip install -e .`.

- **MNIST data** auto-downloads (cached under `data/mnist/`, gitignored). The whole `mlp_numpy_ref`
  golden reference matches `jax.grad` to ~1e-8; the GPU MLP matches that reference to ~1e-7.
- **The GPU-beats-CPU crossover is ~batch 60 at hidden=128, and lower as hidden grows** (it already
  wins at batch 32 for hidden=1024). Below it the per-step fixed cost dominates. **This only holds
  with `train_steps` / a large `chunk_steps`** — with one command buffer per step the crossover is
  ~2.3× further right, because the ~140 µs driver round trip is then paid on every step. Measured
  per-step (M4 Pro, `chunk_steps=128`, 2026-08-13):

  | hidden | batch | GPU ms | JAX-CPU ms | speedup |
  |-------:|------:|-------:|-----------:|--------:|
  | 128 | 32   | 0.124 | 0.097 | 0.79× |
  | 128 | 128  | 0.137 | 0.179 | **1.31×** |
  | 128 | 512  | 0.206 | 0.446 | **2.17×** |
  | 128 | 2048 | 0.496 | 1.382 | **2.79×** |
  | 1024 | 32  | 0.200 | 0.319 | **1.60×** |
  | 1024 | 2048| 2.023 | 6.400 | **3.16×** |

  Same residency/scale lesson as the matmul crossover (§7), but the binding constraint here was
  *submission overhead*, not FLOPs — which is why `device="auto"` uses a crossover model rather
  than the matmul router's `kAutoGpuFlopThreshold` (a FLOP threshold cannot express it: identical
  FLOP counts fall on opposite sides depending on how they split into batch vs hidden).

- **No lint/format config** is checked in — match surrounding style (C++17/ObjC++17, ARC on `.mm`).
- **Single C++ test:** `ctest -R <Name>` where `<Name>` is the `TEST(<Name>)` macro argument; each
  `TEST(...)` in `tests/cpp/*.cpp` auto-registers as its own CTest case (see `CMakeLists.txt:94`).
- **Rebuilding after editing a `kernels/*.metal` file:** the CMake `embed_metal` step re-embeds it as
  a C++ string header on the next `cmake --build build` — no manual step, but a plain rebuild is
  required (kernels are baked in at build time, then compiled from source *at runtime*).
- **Never link/enable MPS**Graph** — MPS (`MPSMatrixMultiplication`) is used, but MPSGraph is
  deliberately unused (we own codegen/dispatch). Kernels compile with **safe math** (`MTLMathModeSafe`)
  so f32 results are bit-exact vs the JAX/CPU reference — do not change this without a numerics reason.

---

## 1. Goal & success criteria

- **Primary goal:** Learn **compilers + GPU programming** deeply by building a real JAX backend. This
  is a portfolio/depth project — *not* a production replacement for Apple's (abandoned) `jax-metal`.
- **Success = end-to-end:** an MLP on MNIST runs **forward and backward** on the M4 Pro GPU through
  real `jax.jit(f, backend='metal')`, and matches the JAX **CPU** backend numerically (f32).
- **Golden reference for every stage:** the JAX **CPU** backend on identical inputs (fixed PRNG seed).

### Locked architectural decisions
| Axis | Decision |
|------|----------|
| Integration | Real **PJRT (Pluggable JAX RunTime) C API plugin**, implemented **directly** (Option A — *not* the C++ `PjRtClient` wrapper, which needs a full Bazel XLA build) |
| Backend | **Hand-written MSL compute kernels**; we own codegen + dispatch. **No MPSGraph.** |
| Language | **Objective-C++ / C++** for plugin + runtime; thin **Python** package for registration |
| Shaders | **Runtime compilation** via `newLibraryWithSource:options:error:` (no `metal` CLI needed) |
| Build | **CMake + Ninja**, ObjCXX enabled |
| Parsing | **Hand-rolled StableHLO subset parser** (see Risk R1) |
| dtype | **f32** for all compute through Stage 4; `convert` handles i32/pred boundaries |

---

## 2. Verified environment baselines (captured 2026-07-11)

- **Hardware:** Apple **M4 Pro**, macOS 15, **Metal 4**. Unified memory (use
  `MTLResourceStorageModeShared` everywhere → `MTLBuffer.contents` is a valid CPU pointer; host↔device
  copies are plain `memcpy`, no blit encoder).
- **Frameworks present:** `Metal`, `MetalKit`, `MetalPerformanceShaders`,
  `MetalPerformanceShadersGraph`, `Accelerate` (MPS/MPSGraph present but deliberately unused).
- **Toolchain gap:** only **Command Line Tools** installed → **no `metal` CLI compiler** (that needs
  full Xcode). ⇒ compile MSL **at runtime** from source strings. Keep a seam to add AOT `.metallib`
  later if full Xcode is installed.
- **`cmake` NOT installed** → `brew install cmake ninja` in Stage 0. `clang++` 21, Homebrew, `uv`,
  `rustc` present (Rust unused).
- **Python:** system Python is **3.14 — too new**; jaxlib **0.6.x** macOS **arm64** wheels exist only
  for **3.10–3.12**. ⇒ create an isolated **Python 3.11/3.12** env via `uv` and **pin** `jax`/`jaxlib`.
- **PJRT version lock (load-bearing):** the plugin's PJRT C API version **must match** the installed
  jaxlib. Pin an exact `jax`/`jaxlib` pair in `pyproject.toml`; **vendor** that release's
  `pjrt_c_api.h` (from `openxla/xla` at `xla/pjrt/c/pjrt_c_api.h`, matching commit) into
  `third_party/pjrt/`; assert `PJRT_API_MAJOR/MINOR` at load. Bumping jaxlib ⇒ re-check the header.
- **Registration API:** `jax._src.xla_bridge.register_plugin(name, priority, library_path)` inside an
  `initialize()`, discovered via a `[project.entry-points.'jax_plugins']` entry point. For dev, use
  explicit `xla_client.load_pjrt_plugin_dynamically('metal', path)` + register; ship the entry point
  once it works.

---

## 3. Key risks & mitigations

- **R1 — Parsing StableHLO in C++ (central risk).** JAX's `PJRT_Client_Compile` receives a
  `PJRT_Program` with `format = "mlir"` (StableHLO), usually as **MLIR bytecode**.
  - *Option A:* link full MLIR + StableHLO C++ libs — correct/future-proof but a massive LLVM/MLIR
    build; overkill for learning.
  - *Option B (chosen):* **hand-rolled recursive-descent parser** over the **textual** StableHLO
    subset our model emits (~15 ops). Maximum learning, zero heavy deps.
  - **Bytecode mitigation:** in Stage 3's first hour, **log `PJRT_Program.format` + first bytes** JAX
    actually sends. If bytecode, link *only* the narrow StableHLO **serialization C API**
    (`deserializePortableArtifact`) to convert bytecode→text, then still parse text by hand.
  - *This is why Stage 2 exists:* first drive lowering from **Python** (jaxlib parses StableHLO for
    us) to nail op→kernel mapping, then port the parse to C++ against a known-good reference.
- **R2 — PJRT ABI drift:** pin jax/jaxlib; vendor matching header; assert version at load.
- **R3 — Registration churn (`xla_bridge` internals move):** keep dev registration in one Python file;
  add the entry point only after it works.
- **R4 — Autodiff is "free":** JAX differentiates **before** lowering, so `grad` emits a *larger*
  StableHLO graph using the *same op vocabulary*. No backward kernels — just widen op coverage.
- **R5 — `dot_general` generality:** grad produces transposed contractions. Normalize by inserting
  explicit `transpose`s so the matmul kernel always sees `[M,K]x[K,N]`; assert-and-fail on batch dims.
- **R6 — `reduce` region decoding:** pattern-match the nested reducer body (single `add`→sum, single
  `maximum`→max); reject anything else loudly.
- **R7 — Async/event contract:** MVP executes **synchronously** and returns an **already-ready**
  `PJRT_Event`. Real async via `MTLCommandBuffer.addCompletedHandler` only if benchmarking demands it.
- **R8 — Numerical mismatch localization:** add a **per-SSA-value buffer dump** mode + a NumPy
  reference interpreter of the same StableHLO graph to pinpoint the offending op.
- **R9 — Storage mode:** centralize allocation in `MetalBuffer`, force `Shared`, assert (a stray
  `Private` buffer breaks the memcpy assumption).

---

## 4. Staged roadmap (each stage independently runnable / demoable)

### Stage 0 — Toolchain + "hello Metal kernel" from C++
`brew install cmake ninja`; `uv venv --python 3.11/3.12` + `uv pip install "jax[cpu]==<pin>"`; confirm
`jax.devices()` and `jax.jit(f).lower(*a).compiler_ir('stablehlo')` print IR. Build one Obj-C++ binary
that compiles a `vector_add`/`saxpy` MSL kernel at runtime, runs it on `MTLBuffer`s, and matches a CPU
loop. **Verify:** GPU output == numpy; prints M4 Pro device name.

### Stage 1 — Standalone MSL kernel library + minimal runtime (no JAX)
Runtime: `MetalContext`(device+queue), `MetalBuffer`(size/shape/dtype), `KernelLibrary`(compile+cache
`MTLComputePipelineState`), `Dispatcher`(encode pass, threadgroup sizing). Kernels: elementwise
unary/binary (`add sub mul div max min exp log tanh negate relu`), `compare`+`select`, tiled
`matmul` (naive → 32×32 threadgroup-tiled), `reduce` (sum/max along axis), shape ops
(`reshape transpose broadcast_in_dim iota`). **Verify:** per-kernel numeric tests vs numpy, f32
`rtol=1e-5 atol=1e-6`. This is the unit-test spine — every new op lands with a test here first.

### Stage 2 — StableHLO → kernel-schedule lowering, driven from Python
Python harness dumps StableHLO text from `jax.jit(mlp_forward).lower(...).compiler_ir('stablehlo')` →
save `.mlir` fixtures. C++ **subset parser** → in-memory `HloOp` IR → **scheduler** (topo-order,
one `MetalBuffer` per SSA value, op→kernel dispatch) → `MetalExecutable.execute(inputs)`. Drive via a
pybind11/ctypes shim or a CLI reading `.mlir`+`.npy`. **Verify:** GPU MLP forward vs `jax.jit` CPU,
`max|Δ| < 1e-4`; dump per-SSA intermediates to localize mismatches.

### Stage 3 — Real PJRT C API plugin
`libjax_metal_plugin.dylib` exporting `extern "C" const PJRT_Api* GetPjrtApi()`. Implement the MVP
surface (§5). `Compile` parses StableHLO (port Stage 2 mapping to C++, validated against Stage 2 JSON)
→ builds an executable holding pipelines + schedule. `Execute` binds `MTLBuffer`-backed
`PJRT_Buffer`s, dispatches, returns outputs. Sync execution, ready events. Python `jax_metal_plugin`
package registers via `register_plugin` / `load_pjrt_plugin_dynamically`. **Verify:**
`jax.jit(mlp_forward, backend='metal')(x)` == CPU backend (atol 1e-4); pytest CPU-vs-Metal parity.

### Stage 4 — Backward pass, broaden ops, benchmark, writeup
`jax.value_and_grad(loss, backend='metal')` + Python-side SGD loop (jit per step — **keep the loop in
Python** to avoid `while`/`scan`). Fill extra grad ops (transposed `dot_general`, batch-axis `reduce`,
`select`+`compare` relu grad, `log`, mean `reduce`, scalar `mul`/`sub`). Optimize matmul (measure
GFLOP/s), benchmark vs CPU, write `docs/`. **Stretch:** elementwise **kernel fusion** (the headline
"compiler" extension), real async events, AOT `.metallib`. **Verify:** grad vs CPU (atol 1e-3); MNIST
training loss tracks CPU; accuracy sanity check.

---

## 5. MVP checklists

**MVP PJRT functions:** `GetPjrtApi`, `PJRT_Plugin_Initialize`, `PJRT_Plugin_Attributes`,
`PJRT_Error_{Destroy,Message,GetCode}`,
`PJRT_Client_{Create,Destroy,PlatformName,PlatformVersion,Devices,AddressableDevices,LookupDevice,Compile}`,
`PJRT_Device_{GetDescription,IsAddressable,LocalHardwareId}`,
`PJRT_DeviceDescription_{Id,Kind,ProcessIndex}`, `PJRT_Client_BufferFromHostBuffer`,
`PJRT_Buffer_{ToHostBuffer,Destroy,Dimensions,UnpaddedDimensions,ElementType,Device,IsDeleted,ReadyEvent}`,
`PJRT_Executable_{Name,NumOutputs,Destroy}`,
`PJRT_LoadedExecutable_{Execute,Destroy,AddressableDevices,GetExecutable}`,
`PJRT_Event_{Destroy,IsReady,Await,OnReady,Error}`. Everything else → stub a proper "unimplemented"
`PJRT_Error`.

**MVP StableHLO ops (forward):** `func.func`, `constant`, `dot_general`, `broadcast_in_dim`, `add`,
`subtract`, `multiply`, `divide`, `maximum`, `exponential`, `reduce` (add/max reducers), `reshape`,
`transpose`, `convert`, `compare`, `select`, `iota`.
**Add for grad/training:** transposed `dot_general`, batch-axis `reduce`, `select`+`compare` (relu
grad), `log`, mean `reduce`, scalar `multiply`/`subtract` (SGD).

**First model:** MLP `784→128→10`, ReLU, softmax-cross-entropy, on MNIST; `jax.value_and_grad` + SGD.
An even smaller Stage-2 starter: `f(x,W) = relu(x @ W)`.

---

## 6. Proposed repo layout
```
jax-metal-prototype/
├── CLAUDE.md                      # this ground-truth doc
├── CMakeLists.txt                 # root: Ninja, ObjCXX; targets: runtime lib, kernel_test, plugin
├── cmake/                         # FindMetal helper, toolchain snippets
├── pyproject.toml                 # uv-managed; pins jax/jaxlib; jax_plugins entry point
├── third_party/pjrt/pjrt_c_api.h  # vendored, version-pinned from openxla/xla
├── kernels/                       # *.metal MSL source (loaded as strings)
│   ├── elementwise.metal  matmul.metal  reduce.metal  shape.metal
├── src/
│   ├── metal/     metal_context.mm  metal_buffer.mm  kernel_library.mm   # Stage 0–1
│   ├── runtime/   dispatcher.mm
│   ├── compiler/  hlo_parser.cc  hlo_ir.h  scheduler.cc  metal_executable.cc   # Stage 2
│   └── pjrt/      pjrt_plugin.cc  pjrt_client.cc  pjrt_buffer.cc  pjrt_executable.cc  pjrt_event.cc # Stage 3
├── python/jax_metal_plugin/       # __init__.py (register_plugin) + harness/dump_hlo.py
├── tests/  cpp/ (ctest kernel tests)  fixtures/*.mlir  python/test_pjrt.py
├── docs/                          # README.md index + ARCHITECTURE.md + PJRT_PLUGIN.md
│   └── features/                  # ONE DOC PER FEATURE — add one whenever a feature lands
```
Build targets: `libmetal_rt` (metal+runtime+compiler, static) → linked into `kernel_test` (Stage 1)
and `libjax_metal_plugin.dylib` (Stage 3; links `-framework Metal -framework Foundation`; exports only
`GetPjrtApi`).

**Critical files (when implementation starts):** `CMakeLists.txt` · `src/pjrt/pjrt_plugin.cc`
(GetPjrtApi + fn-pointer table) · `src/compiler/hlo_parser.cc` · `src/compiler/scheduler.cc` ·
`src/metal/kernel_library.mm`.

---

## 7. Build & test

```
cmake -S . -B build -G Ninja      # configure (embeds kernels/*.metal)
cmake --build build               # metal_rt, libmetal_capi.dylib, arith_demo, kernel_tests
./build/arith_demo                # Stage 0/1 smoke demo (GPU vs CPU)
ctest --test-dir build --output-on-failure   # 33 unit tests, per-TEST cases
```
Python frontend (`python/`): **`jaxmetal`** — JAX-facing front-end; `jaxmetal.matmul(a, b,
device="mps"|"metal"|"cpu"|"auto")` with full `jnp.matmul` semantics (1-D/2-D/batched/broadcast, verified
in `test_jaxmetal.py`). **`metalmm`** — low-level ctypes binding to `build/libmetal_capi.dylib` (incl.
resident `DeviceBuffer`).

**JAX integration — two native paths** (JAX ops have no `device=` kwarg, so `jaxmetal.matmul(device=)` is
an eager bridge):
- **XLA FFI custom call — WORKS:** `jaxmetal.ffi.matmul` is jittable, composes inside `jax.jit`
  (`src/ffi/metal_ffi.cpp` via `XLA_FFI_DEFINE_HANDLER_SYMBOL`, built into `libmetal_capi` when configured
  with `-DJAX_FFI_INCLUDE_DIR=$(python -c 'import jax.ffi;print(jax.ffi.include_dir())')`). Runs on the CPU
  backend, copies to GPU per call. Demo: `python/ffi_demo.py`.
- **PJRT plugin — Stage 3 scaffold:** the real `mps` *device* (`jax.device_put` → GPU-resident, where the
  MPS win is free). Blocked on vendoring version-matched `pjrt_c_api.h` (not bundled) + the StableHLO
  Compile surface. Registration APIs confirmed present (`register_plugin`, `load_pjrt_plugin_dynamically`,
  `make_c_api_client`). Roadmap: `docs/PJRT_PLUGIN.md`; scaffold: `python/jax_metal_plugin/`.

`python/showcase.py` validates all backends vs `jnp.matmul`; `benchmarks/bench_matmul.py` = MPS vs
hand-kernel vs CPU. Env: `uv venv --python 3.12 .venv && uv pip install --python .venv numpy "jax[cpu]"`.
See `python/README.md`. C ABI in `src/capi/metal_capi.{h,mm}`.

**Matmul backends & perf (M4 Pro, f32, compute-only / resident):** three matmuls exist —
`ops/matmul` (our hand-written register-tiled MSL kernel, ~2.2 TFLOP/s), `ops/mps_matmul`
(`MPSMatrixMultiplication`, Apple's tuned kernel = what PyTorch MPS uses, **~5.4 TFLOP/s**), and
`cpu/cpu_matmul` (Accelerate/AMX `cblas_sgemm`, ~3.0 TFLOP/s). **MPS beats the CPU from N≥1024
(1.25×→1.78× at 4096)** — the real GPU win, and it needs **resident** operands. For host operands the
per-call copies erase it until ~N=4096, so the `matmul_auto` router (`metal_matmul_auto_f32`, GPU arm =
MPS) mostly picks CPU; threshold `kAutoGpuFlopThreshold≈1.4e11`. This is the data-locality lesson
(why PyTorch/JAX keep tensors on-device rather than auto-routing per op; the PJRT backend gets
residency for free). Our hand kernel is ~40% of MPS — lessons: `simdgroup_matrix` gave *us* no win
(no f32 matrix unit); **occupancy dominated** (8×8 micro-tiles over-spilled registers, slower than
4×4); float4 vectorization helped. C-ABI exposes `metal_{mps,cpu}_matmul_f32`, `metal_matmul_auto_f32`,
and resident variants. Frameworks linked: Metal, Foundation, Accelerate, MetalPerformanceShaders.
Current `src/` runtime: `metal/` (context, buffer, kernel_library),
`runtime/dispatcher` (1-D `dispatch_1d` + threadgroup-grid `dispatch_threadgroups` for tiled kernels),
`ops/elementwise` (add/sub/mul/div/max/min, neg/abs/exp), `ops/matmul` (tiled shared-memory matmul,
`C[M,N]=A[M,K]@B[K,N]`), `ops/mps_matmul` (MPS), `ops/nn` (bias_add, **bias_relu** (fused),
relu, relu_grad, reduce_sum_axis0, transpose2d, sgd_update, stable softmax_xent, argmax — the MLP
op set), and `ops/mlp` (the resident `MLP` class: forward + backward + SGD for `in→H→out`), and
`ops/reduce` (**compensated f32 summation**, below).

**Compensated reductions (`kernels/reduce.metal`, `ops/reduce`) — the scientific-computing
angle.** Apple GPUs have **no f64 at all** (Metal has no `double`), so the usual "promote to
float64" fix for large-sum error is unavailable on-device. `reduce_sum_f32(..., compensated=true)`
uses **Neumaier** summation: a `(sum, compensation)` accumulator that composes associatively via
`two_sum`, so partial results merge across threads, threadgroups, and the two-pass structure
without discarding the correction. Measured at n=16.7M: on an adversarial input (a large value in
every thread's first grid-stride slot, then values below its ulp) the relative error is **1.0e-8 vs
1.3e-6 for an uncompensated tree sum — 127× better**, and better than numpy's pairwise f32
(7.0e-8). It is **free**: 214 GB/s vs 212 GB/s for the tree sum at 268 MB, because both are
bandwidth-bound against ~273 GB/s peak. Benchmark: `benchmarks/bench_reduce.py`.

> **`kernels/reduce.metal` hard-depends on safe math.** Every compensation term has the form
> `(a - (a + b)) + b`, algebraically zero and legal for a fast-math compiler to fold away. Under
> fast math these kernels silently degrade to a plain tree sum — same speed, no accuracy, no error.
> `TEST(ReduceCompensatedBeatsNaive)` is the regression test for exactly that, and it asserts the
> tree sum *does* lose accuracy so it cannot pass vacuously. Do not enable fast math.

`reduce_sum_axis0_comp` is the accuracy-preserving counterpart of `nn_reduce_sum_axis0`. **The MLP
still uses the uncompensated one** — its batch reductions are short and the golden-reference gate
bounds the error already; switching would change MLP numerics for no measured benefit.

**`ops/mlp` submission rules — the load-bearing part.** (a) MPS `MPSMatrixMultiplication` +
`MPSMatrix` objects are built **once per (batch, chunk slot)** and cached in `Impl::plans`;
rebuilding them per step cost ~28 ObjC allocations/step. (b) The three backward contractions use
MPS's **`transposeLeft`/`transposeRight`** — descriptors stay the *stored* (untransposed) shapes and
`interiorColumns` is K *after* transposition, so `x^T`/`h1^T`/`W2^T` are never materialised.
(c) A `Coalescer` holds one compute encoder open across consecutive kernel dispatches and closes it
only when an MPS matmul needs the buffer (PyTorch MPS's `endKernelCoalescing`); serial dispatch
means the read-after-write chains inside a group need no explicit barriers. (d) `train_steps(n)`
encodes n whole steps into one command buffer — **bit-identical** to n `train_step()` calls (gated
in `tests/python/test_mlp_auto.py`), since Metal hazard-tracks within a command buffer.
(e) `nn_reduce_sum_axis0` dispatches **one threadgroup per output column** with a tree reduction;
the one-thread-per-column version ran 10 threads for `db2` (C=10) and scaled linearly in batch.

Kernels in `kernels/*.metal` (`elementwise`, `matmul`, `nn`), each embedded as a
string via `cmake/EmbedMetal.cmake`. Tests in `tests/cpp/` use a dependency-free framework; each
`TEST(Name)` auto-registers as its own CTest case (**68 tests** currently, incl. `nn_test` and
`mlp_test` parity vs double-precision CPU references, and `reduce_test` for compensated summation). Kernels compile with **safe math**
(`MTLMathModeSafe`) so arithmetic is IEEE-correct and matches the JAX CPU reference.

> **Every feature gets a doc.** `docs/features/<FEATURE>.md`, listed in `docs/README.md`, stating
> the problem it solves as a measurement, how it works, the numbers plus the command that
> reproduces them, the load-bearing invariants, and the limits. **Record refuted hypotheses there
> too** — approaches measured as worse are the most expensive knowledge here and the easiest to
> lose (see the MPS-vs-own-kernel experiment in `features/CHUNKED_TRAINING.md`).

## 8. Verification strategy (cumulative)
1. **Kernel unit tests** (Stage 1): each MSL kernel vs a CPU reference, f32 (safe-math ⇒ bit-exact for
   +,−,×,÷; `exp` within 1e-4). Run via `ctest`.
2. **Schedule diff** (Stage 2): GPU MLP forward vs `jax.jit` CPU, `max|Δ| < 1e-4`; per-SSA dumps.
3. **End-to-end via JAX** (Stage 3): `jax.jit(f, backend='metal')` vs CPU backend.
4. **Training correctness** (Stage 4): value_and_grad + SGD; loss trajectory tracks CPU.

**GPU coverage caveat (load-bearing).** There is **no CI pipeline** — verification runs
**locally only**. Any hosted macOS runner (e.g. GitHub-hosted `macos-14`) would be virtualized and
expose **no Metal GPU** (`MTLCreateSystemDefaultDevice()` returns null), so GPU-dependent C++ tests
would **skip** rather than fail: `MetalContext` throws `jaxmetal::MetalUnavailable`, the test
harness reports `[ SKIP ]` and returns `125`, and each CTest case carries `SKIP_RETURN_CODE 125`
(set in `CMakeLists.txt`). **GPU regressions are only caught locally** (`ctest` on the M4 Pro, where
all 68 run for real). Treat the local `ctest` run as the authoritative GPU gate and the Python
parity gates (`reference.py`, `test_frontend.py`, `test_mlp_gate.py`, `test_mlp_auto.py`) as the
fast correctness check.
To get real GPU coverage in an automated pipeline, add a **self-hosted macOS runner with a GPU**.

## 9. Open decisions to confirm before Stage 0
- Exact `jaxlib` version to pin (latest 0.6.x with a 3.11/3.12 arm64 wheel unless told otherwise).
- First model: MLP/MNIST (default) vs the smaller `relu(x @ W)` starter for Stage 2.
