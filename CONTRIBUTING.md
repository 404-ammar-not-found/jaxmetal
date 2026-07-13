# Contributing

Thanks for your interest. This is a learning-focused, compiler-level project, so
clarity and correctness matter more than feature count.

## Prerequisites

- macOS on **Apple Silicon** with **Xcode Command Line Tools** (`xcode-select
  --install`). A full Xcode is *not* required — MSL is compiled at runtime.
- [Homebrew](https://brew.sh), and [`uv`](https://docs.astral.sh/uv/) for Python.

## Setup

```bash
brew install cmake ninja
uv venv --python 3.12 .venv
uv pip install --python .venv numpy "jax[cpu]"

cmake -S . -B build -G Ninja \
  -DJAX_FFI_INCLUDE_DIR=$(.venv/bin/python -c "import jax.ffi; print(jax.ffi.include_dir())")
cmake --build build
```

## Test before you push

```bash
ctest --test-dir build --output-on-failure     # C++ kernels + runtime + MLP
.venv/bin/python tests/python/test_frontend.py  # matmul front-end vs jnp.matmul
.venv/bin/python tests/python/test_mlp_gate.py  # resident MLP vs golden reference
.venv/bin/python python/jaxmetal/reference.py   # golden reference vs jax.grad
```

CI (`.github/workflows/ci.yml`) runs exactly these on a macOS runner.

## Adding a kernel (the golden path)

Every new op lands with a numeric test **first**. The workflow:

1. Write the MSL in the relevant `kernels/*.metal` (buffers at indices `0..k-1`,
   push constant at index `k`, guard `if (gid >= n) return;`). If it's a new file,
   add an `embed_metal(...)` line in `CMakeLists.txt`.
2. Add a `_into` C++ wrapper in `src/ops/` (header in `include/jaxmetal/ops/`) that
   takes explicit dims and does **not** wait — so a scheduler can chain a whole step
   and sync once. Mirror `src/ops/nn.mm`.
3. Add a `TEST(...)` in `tests/cpp/` comparing the GPU output to a **double-precision
   CPU reference**. Each `TEST(...)` auto-registers as its own CTest case. Add the
   file to `TEST_SRCS` in `CMakeLists.txt` if it's new.
4. Tolerances: `1e-6` for `+ − × ÷` (safe-math is bit-exact), `1e-4` for `exp`/`log`.

## Conventions

- **C++/ObjC++17**, namespace `jaxmetal`, PIMPL for headers that must stay pure C++
  (so they can be included from `.cc` translation units). No exceptions across the C
  ABI — wrap in `try/catch` and return an error code.
- **Buffers** are always `MTLResourceStorageModeShared` (unified memory). Never
  introduce a `Private` buffer without a blit path — it breaks the `memcpy`
  assumption centralized in `MetalBuffer`.
- **Python**: keep the public surface in `python/jaxmetal/__init__.py`; the ctypes
  binding lives in `_capi.py` and must set `argtypes`/`restype` for **every**
  function (an unset `c_void_p` argtype truncates 64-bit handles → segfault).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the layering and the
single-command-buffer design.
