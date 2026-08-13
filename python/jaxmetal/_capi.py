"""jaxmetal._capi — low-level ctypes binding to build/libmetal_capi.dylib.

Wraps the flat C ABI in include/jaxmetal/capi/metal_capi.h: device name,
GPU-resident buffers, the matmul family, and the resident MLP (create/destroy/
set_params/get_params/upload_batch/forward/train_step). No third-party deps beyond
numpy. The public `jaxmetal` package re-exports the user-facing names from here.
"""
from __future__ import annotations

import ctypes
import os
from ctypes import c_void_p, c_int, c_int64, c_int32, c_float, POINTER
import numpy as np

from . import routing as _routing

__all__ = [
    "lib", "dylib_path", "library_path", "device_name",
    "DeviceBuffer",
    "matmul", "matmul_mps", "matmul_cpu", "matmul_auto",
    "matmul_resident", "matmul_resident_mps",
    "reduce_sum", "reduce_sum_resident", "cholesky", "cholesky_resident",
    "batched_solve", "batched_solve_cpu", "batched_solve_resident",
    "df64_binop", "df64_binop_resident", "df64_stencil3", "to_df64", "from_df64",
    "Mlp",
]


def _find_dylib() -> str:
    """Locate libmetal_capi.dylib.

    The wheel is pure Python (`py3-none-any`) and does NOT bundle the native library:
    it is built out of tree with CMake, because it links Metal, MetalPerformanceShaders
    and Accelerate and only builds on macOS with the Command Line Tools. So this looks,
    in order, for an explicit override, a copy sitting beside the package (which is
    where a future binary wheel would put it), and a sibling `build/` from a source
    checkout.
    """
    # JAXMETAL_DYLIB is the current name; METALMM_DYLIB is the pre-rename spelling,
    # still honoured so existing scripts do not break.
    for var in ("JAXMETAL_DYLIB", "METALMM_DYLIB"):
        override = os.environ.get(var)
        if override:
            if not os.path.exists(override):
                raise FileNotFoundError(f"{var}={override!r} does not exist")
            return override

    here = os.path.abspath(os.path.dirname(__file__))            # .../jaxmetal
    repo_root = os.path.dirname(os.path.dirname(here))           # source checkout root
    candidates = [
        os.path.join(here, "libmetal_capi.dylib"),               # bundled beside package
        os.path.join(repo_root, "build", "libmetal_capi.dylib"),
        os.path.join(repo_root, "build", "Release", "libmetal_capi.dylib"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    installed = "site-packages" in here or "dist-packages" in here
    hint = (
        "jaxmetal is installed but its native library was not found. The wheel is pure\n"
        "Python by design; build the library from a checkout of the repository:\n"
        if installed else
        "Build the native library first, from the repository root:\n"
    )
    raise FileNotFoundError(
        "Could not locate libmetal_capi.dylib.\n" + hint +
        "  cmake -S . -B build -G Ninja && cmake --build build\n"
        "then point at it with JAXMETAL_DYLIB=/abs/path/to/libmetal_capi.dylib\n"
        f"Searched: {candidates}")


dylib_path = _find_dylib()
lib = ctypes.CDLL(dylib_path)


def library_path() -> str:
    """Absolute path of the loaded libmetal_capi.dylib (used by jaxmetal.ffi)."""
    return dylib_path

_f32 = POINTER(c_float)
_i32 = POINTER(c_int32)


def _c_f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(_f32)


def _ptr_i32(a: np.ndarray):
    return a.ctypes.data_as(_i32)


# ---- signatures ----
lib.metal_device_name.restype = ctypes.c_char_p
lib.metal_device_name.argtypes = []

for _name in ("metal_matmul_f32", "metal_mps_matmul_f32", "metal_cpu_matmul_f32"):
    _fn = getattr(lib, _name)
    _fn.argtypes = [_f32, _f32, _f32, c_int64, c_int64, c_int64]
    _fn.restype = c_int if _name != "metal_cpu_matmul_f32" else None

lib.metal_matmul_auto_f32.argtypes = [_f32, _f32, _f32, c_int64, c_int64, c_int64,
                                      POINTER(c_int)]
lib.metal_matmul_auto_f32.restype = c_int

lib.metal_buffer_alloc.argtypes = [c_int64]
lib.metal_buffer_alloc.restype = c_void_p
lib.metal_buffer_upload.argtypes = [c_void_p, _f32, c_int64]
lib.metal_buffer_upload.restype = None
lib.metal_buffer_download.argtypes = [c_void_p, _f32, c_int64]
lib.metal_buffer_download.restype = None
lib.metal_buffer_free.argtypes = [c_void_p]
lib.metal_buffer_free.restype = None

# resident matmul on DeviceBuffer handles (all c_void_p -> MUST be typed, else ctypes
# truncates 64-bit handles to int and segfaults).
lib.metal_matmul_resident.argtypes = [c_void_p, c_void_p, c_void_p, c_int64, c_int64, c_int64]
lib.metal_matmul_resident.restype = c_int
lib.metal_mps_matmul_resident.argtypes = [c_void_p, c_void_p, c_void_p, c_int64, c_int64, c_int64]
lib.metal_mps_matmul_resident.restype = c_int

lib.metal_reduce_sum_f32.argtypes = [_f32, c_int64, c_int, POINTER(c_float)]
lib.metal_reduce_sum_f32.restype = c_int
lib.metal_reduce_sum_resident.argtypes = [c_void_p, c_int64, c_int, POINTER(c_float)]
lib.metal_reduce_sum_resident.restype = c_int

lib.metal_cholesky_resident.argtypes = [c_void_p, c_int64]
lib.metal_cholesky_resident.restype = c_int
lib.metal_cholesky_f32.argtypes = [_f32, c_int64]
lib.metal_cholesky_f32.restype = c_int

lib.metal_batched_solve_f32.argtypes = [_f32, _f32, _f32, _f32, c_int64, c_int64, c_int]
lib.metal_batched_solve_f32.restype = c_int
lib.metal_batched_solve_resident.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p,
                                             c_int64, c_int64, c_int]
lib.metal_batched_solve_resident.restype = c_int
lib.metal_batched_solve_cpu_f32.argtypes = [_f32, _f32, _f32, c_int64, c_int64]
lib.metal_batched_solve_cpu_f32.restype = None

lib.metal_df64_binop.argtypes = [_f32, _f32, _f32, c_int64, c_int]
lib.metal_df64_binop.restype = c_int
lib.metal_df64_binop_resident.argtypes = [c_void_p, c_void_p, c_void_p, c_int64, c_int]
lib.metal_df64_binop_resident.restype = c_int
lib.metal_df64_stencil3.argtypes = [_f32, _f32, _f32, c_int64, c_int]
lib.metal_df64_stencil3.restype = c_int

lib.metal_mlp_create.argtypes = [c_int64, c_int64, c_int64, c_int64, c_int64]
lib.metal_mlp_create.restype = c_void_p
lib.metal_mlp_destroy.argtypes = [c_void_p]
lib.metal_mlp_destroy.restype = None
lib.metal_mlp_set_params.argtypes = [c_void_p, _f32, _f32, _f32, _f32]
lib.metal_mlp_set_params.restype = None
lib.metal_mlp_get_params.argtypes = [c_void_p, _f32, _f32, _f32, _f32]
lib.metal_mlp_get_params.restype = None
lib.metal_mlp_upload_batch.argtypes = [c_void_p, _f32, _i32, c_int64]
lib.metal_mlp_upload_batch.restype = None
lib.metal_mlp_forward.argtypes = [c_void_p, c_int64, _f32]
lib.metal_mlp_forward.restype = c_int
lib.metal_mlp_train_step.argtypes = [c_void_p, c_int64, c_float, POINTER(c_float)]
lib.metal_mlp_train_step.restype = c_int
lib.metal_mlp_upload_chunk.argtypes = [c_void_p, _f32, _i32, c_int64, c_int64]
lib.metal_mlp_upload_chunk.restype = None
lib.metal_mlp_train_steps.argtypes = [c_void_p, c_int64, c_int64, c_float]
lib.metal_mlp_train_steps.restype = c_int
lib.metal_mlp_last_loss.argtypes = [c_void_p]
lib.metal_mlp_last_loss.restype = c_float


# ---- thin wrappers ----
def device_name() -> str:
    return lib.metal_device_name().decode()


def _matmul2d(fn, a, b):
    a = _c_f32(a); b = _c_f32(b)
    M, K = a.shape; K2, N = b.shape
    assert K == K2, f"inner dim mismatch {a.shape} @ {b.shape}"
    c = np.empty((M, N), dtype=np.float32)
    fn(_ptr(a), _ptr(b), _ptr(c), c_int64(M), c_int64(K), c_int64(N))
    return c


def matmul(a, b):     return _matmul2d(lib.metal_matmul_f32, a, b)
def matmul_mps(a, b): return _matmul2d(lib.metal_mps_matmul_f32, a, b)
def matmul_cpu(a, b): return _matmul2d(lib.metal_cpu_matmul_f32, a, b)


def matmul_auto(a, b, return_backend: bool = False):
    a = _c_f32(a); b = _c_f32(b)
    M, K = a.shape; K2, N = b.shape
    c = np.empty((M, N), dtype=np.float32)
    used = c_int(0)
    lib.metal_matmul_auto_f32(_ptr(a), _ptr(b), _ptr(c),
                              c_int64(M), c_int64(K), c_int64(N), ctypes.byref(used))
    if return_backend:
        return c, ("gpu" if used.value else "cpu")
    return c


def matmul_resident(A, B, C, M, K, N):
    """C = A @ B on GPU-resident DeviceBuffers (our tiled MSL kernel). No copies."""
    rc = lib.metal_matmul_resident(A.handle, B.handle, C.handle,
                                   c_int64(M), c_int64(K), c_int64(N))
    if rc:
        raise RuntimeError(f"metal_matmul_resident rc={rc}")


def matmul_resident_mps(A, B, C, M, K, N):
    """C = A @ B on GPU-resident DeviceBuffers via MPS (Apple's tuned kernel)."""
    rc = lib.metal_mps_matmul_resident(A.handle, B.handle, C.handle,
                                       c_int64(M), c_int64(K), c_int64(N))
    if rc:
        raise RuntimeError(f"metal_mps_matmul_resident rc={rc}")


def to_df64(x):
    """float64 array -> df64 limbs, shape (..., 2) float32. Exact: `lo` holds the
    part of x that did not fit in `hi`, and is itself representable in f32."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    hi = x.astype(np.float32)
    lo = (x - hi.astype(np.float64)).astype(np.float32)
    return np.ascontiguousarray(np.stack([hi, lo], axis=-1))


def from_df64(v):
    """df64 limbs -> float64 array."""
    v = np.asarray(v, dtype=np.float32)
    return v[..., 0].astype(np.float64) + v[..., 1].astype(np.float64)


_DF64_OPS = {"add": 0, "mul": 1, "div": 2}


def df64_binop(a, b, op: str = "add", device: str = "auto"):
    """Elementwise df64 arithmetic on float64 inputs; returns float64.

    ~48 bits of significand, against f32's 24 and f64's 53. This is a PRECISION
    feature: it is slower than doing the same work in float64 on the CPU. It exists
    because Metal has no `double` type at all, so a GPU-resident pipeline otherwise
    has no way to exceed f32 without round-tripping to the host.
    """
    if op not in _DF64_OPS:
        raise ValueError(f"unknown op {op!r}; expected one of {sorted(_DF64_OPS)}")

    # HOST operands. The copies are ~41x the kernel, so this path never beats numpy
    # float64 on speed at any size -- `auto` routes it to the CPU, and the GPU is
    # worth asking for only when you specifically want the ~48-bit result computed
    # on-device. Use df64_binop_resident to get the speed.
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    chosen = _routing.resolve(
        device, lambda: _routing.prefer_gpu_df64(a.size, resident=False))
    if chosen == "cpu":
        return {"add": np.add, "mul": np.multiply, "div": np.divide}[op](a, b)

    av, bv = to_df64(a), to_df64(b)
    n = av.size // 2
    out = np.empty_like(av)
    rc = lib.metal_df64_binop(_ptr(av), _ptr(bv), _ptr(out), c_int64(n),
                              c_int(_DF64_OPS[op]))
    if rc:
        raise RuntimeError(f"metal_df64_binop rc={rc}")
    return from_df64(out)


def df64_binop_resident(bufA, bufB, bufOut, n: int, op: str = "add"):
    """Elementwise df64 arithmetic on already-resident DeviceBuffers holding
    interleaved (hi, lo) pairs. df64 moves 2x the bytes of f32 for the same element
    count, so the host copies dominate `df64_binop`; use this when data stays put."""
    if op not in _DF64_OPS:
        raise ValueError(f"unknown op {op!r}; expected one of {sorted(_DF64_OPS)}")
    rc = lib.metal_df64_binop_resident(bufA.handle, bufB.handle, bufOut.handle,
                                       c_int64(n), c_int(_DF64_OPS[op]))
    if rc:
        raise RuntimeError(f"metal_df64_binop_resident rc={rc}")


def df64_stencil3(x, coef=(1.0, -2.0, 1.0), use_df64: bool = True):
    """3-point stencil out[i] = c0*x[i-1] + c1*x[i] + c2*x[i+1], zero boundaries."""
    n = int(np.asarray(x).size)
    cd = np.asarray(coef, dtype=np.float64)
    if use_df64:
        # df64 kernel reads 3 (hi, lo) pairs; f32 kernel reads 3 plain floats. Passing
        # the df64 layout to the f32 kernel would silently feed it [hi0, lo0, hi1] --
        # i.e. the wrong coefficients -- and the comparison would flatter df64 by
        # orders of magnitude. Keep the two layouts strictly separate.
        cv = to_df64(cd)
        xv = to_df64(x)
        out = np.empty_like(xv)
    else:
        cv = np.ascontiguousarray(cd, dtype=np.float32)
        xv = np.ascontiguousarray(x, dtype=np.float32)
        out = np.empty_like(xv)
    rc = lib.metal_df64_stencil3(_ptr(xv), _ptr(out), _ptr(cv), c_int64(n),
                                 c_int(1 if use_df64 else 0))
    if rc:
        raise RuntimeError(f"metal_df64_stencil3 rc={rc}")
    return from_df64(out) if use_df64 else out.astype(np.float64)


def batched_solve(A, rhs, return_pivmin: bool = False, spd: bool = False,
                  device: str = "auto"):
    """Solve `batch` independent tiny systems A[b] @ x[b] = rhs[b] on the GPU.

    A is [batch, n, n] and rhs is [batch, n], n in [2, 8]. Uses LU with partial
    pivoting, one GPU thread per system. Above n=8 use numpy or MPS -- Apple's
    batched MPSMatrixDecompositionLU is faster there.

    With return_pivmin=True also returns min|U_kk|/max|U_kk| per system, which is
    exactly 0 for a singular system or one whose input held NaN/Inf. It flags
    SINGULARITY, not ill-conditioning.
    """
    A = _c_f32(A); rhs = _c_f32(rhs)
    assert A.ndim == 3 and A.shape[1] == A.shape[2], "A must be [batch, n, n]"
    batch, n = A.shape[0], A.shape[1]
    assert rhs.shape == (batch, n), f"rhs must be [batch, n], got {rhs.shape}"

    # These are HOST operands, so the crossover is the host one (~5,000 systems):
    # copying A and rhs in and out costs about as much as solving them.
    chosen = _routing.resolve(
        device, lambda: _routing.prefer_gpu_batched_solve(batch, n, resident=False))
    if chosen == "cpu":
        x = batched_solve_cpu(A, rhs)
        return (x, np.ones(batch, np.float32)) if return_pivmin else x

    x = np.empty((batch, n), np.float32)
    piv = np.empty(batch, np.float32)
    rc = lib.metal_batched_solve_f32(_ptr(A), _ptr(rhs), _ptr(x), _ptr(piv),
                                     c_int64(batch), c_int64(n), c_int(1 if spd else 0))
    if rc:
        raise RuntimeError(f"metal_batched_solve_f32 rc={rc} (n must be in [2, 8])")
    return (x, piv) if return_pivmin else x


def batched_solve_resident(bufA, bufR, bufX, batch: int, n: int, bufP=None,
                           spd: bool = False):
    """Batched solve on already-resident DeviceBuffers: no host copies.

    These kernels do well under one FLOP per byte moved, so the copies in
    `batched_solve` cost about as much as the solve. Use this when the data is
    already on the GPU or is reused across calls.
    """
    rc = lib.metal_batched_solve_resident(bufA.handle, bufR.handle, bufX.handle,
                                          bufP.handle if bufP is not None else None,
                                          c_int64(batch), c_int64(n),
                                          c_int(1 if spd else 0))
    if rc:
        raise RuntimeError(f"metal_batched_solve_resident rc={rc}")


def batched_solve_cpu(A, rhs):
    """Single-threaded scalar CPU reference for the same problem."""
    A = _c_f32(A); rhs = _c_f32(rhs)
    batch, n = A.shape[0], A.shape[1]
    x = np.empty((batch, n), np.float32)
    lib.metal_batched_solve_cpu_f32(_ptr(A), _ptr(rhs), _ptr(x), c_int64(batch), c_int64(n))
    return x


def cholesky(a, device: str = "auto"):
    """Blocked GPU Cholesky: returns lower-triangular L with A = L @ L.T.

    Raises numpy.linalg.LinAlgError if A is not positive definite (or contains
    NaN), matching numpy.linalg.cholesky's contract.
    """
    a = _c_f32(a)
    assert a.ndim == 2 and a.shape[0] == a.shape[1], "cholesky needs a square matrix"

    # Routed against np.linalg.cholesky, which is what a Python caller would otherwise
    # invoke. Note numpy is not the fastest CPU option -- direct spotrf on a
    # Fortran-ordered array is ~4x quicker -- so pass device="cpu" if your CPU path is
    # hand-tuned LAPACK. See python/jaxmetal/routing.py.
    chosen = _routing.resolve(device, lambda: _routing.prefer_gpu_cholesky(a.shape[0]))
    if chosen == "cpu":
        return np.linalg.cholesky(a)

    out = a.copy()
    info = lib.metal_cholesky_f32(_ptr(out), c_int64(out.shape[0]))
    if info != 0:
        raise np.linalg.LinAlgError(
            f"matrix is not positive definite (failed at column {info})")
    return out


def cholesky_resident(buf, n: int) -> int:
    """In-place Cholesky on a resident DeviceBuffer. Returns LAPACK `info`."""
    return int(lib.metal_cholesky_resident(buf.handle, c_int64(n)))


def reduce_sum(x, compensated: bool = True) -> float:
    """Sum a 1-D f32 array on the GPU.

    compensated=True uses Neumaier compensated summation, which stays accurate to
    ~1 ulp of the exact sum almost regardless of length; False uses a plain tree
    sum with identical memory traffic (the baseline). Apple GPUs have no f64, so
    compensation is the only way to get a trustworthy large f32 sum here.
    """
    x = _c_f32(x).ravel()
    out = c_float(0.0)
    rc = lib.metal_reduce_sum_f32(_ptr(x), c_int64(x.size), c_int(1 if compensated else 0),
                                  ctypes.byref(out))
    if rc:
        raise RuntimeError(f"metal_reduce_sum_f32 rc={rc}")
    return float(out.value)


def reduce_sum_resident(buf, n: int, compensated: bool = True) -> float:
    """Sum an already-resident DeviceBuffer: no host copy, so timings measure the
    reduction itself rather than the transfer."""
    out = c_float(0.0)
    rc = lib.metal_reduce_sum_resident(buf.handle, c_int64(n),
                                       c_int(1 if compensated else 0), ctypes.byref(out))
    if rc:
        raise RuntimeError(f"metal_reduce_sum_resident rc={rc}")
    return float(out.value)


class DeviceBuffer:
    """RAII handle over a GPU-resident f32 buffer."""
    def __init__(self, nelem: int):
        self._h = lib.metal_buffer_alloc(c_int64(nelem))
        if not self._h:
            raise RuntimeError("metal_buffer_alloc failed")
        self.nelem = int(nelem)

    @classmethod
    def from_numpy(cls, a) -> "DeviceBuffer":
        """Allocate a resident buffer and upload `a` (flattened f32)."""
        a = _c_f32(a).ravel()
        buf = cls(a.size)
        buf.upload(a)
        return buf

    def upload(self, a):
        a = _c_f32(a).ravel()
        assert a.size == self.nelem
        lib.metal_buffer_upload(self._h, _ptr(a), c_int64(self.nelem))

    def download(self, shape=None):
        """Copy the resident buffer to a new host array, optionally reshaped."""
        out = np.empty(self.nelem, dtype=np.float32)
        lib.metal_buffer_download(self._h, _ptr(out), c_int64(self.nelem))
        return out if shape is None else out.reshape(shape)

    @property
    def handle(self):
        return self._h

    def free(self):
        """Release the GPU buffer now (idempotent; also runs on __del__)."""
        h = getattr(self, "_h", None)
        if h:
            lib.metal_buffer_free(h); self._h = None

    def __del__(self):
        self.free()


class Mlp:
    """Resident in_dim->hidden->out_dim MLP over the C-ABI. Weights, activations,
    gradients, and the current minibatch stay GPU-resident across steps."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, max_batch: int,
                 chunk_steps: int = 1):
        self.in_dim, self.hidden, self.out_dim = in_dim, hidden, out_dim
        self.max_batch = max_batch
        self.chunk_steps = chunk_steps
        self._cur_batch = 0
        self._h = lib.metal_mlp_create(c_int64(in_dim), c_int64(hidden),
                                       c_int64(out_dim), c_int64(max_batch),
                                       c_int64(chunk_steps))
        if not self._h:
            raise RuntimeError("metal_mlp_create failed")

    def set_params(self, W1, b1, W2, b2):
        self._buf = [_c_f32(W1).ravel(), _c_f32(b1).ravel(),
                     _c_f32(W2).ravel(), _c_f32(b2).ravel()]  # keep alive
        lib.metal_mlp_set_params(self._h, *[_ptr(x) for x in self._buf])

    def get_params(self):
        W1 = np.empty(self.in_dim * self.hidden, np.float32)
        b1 = np.empty(self.hidden, np.float32)
        W2 = np.empty(self.hidden * self.out_dim, np.float32)
        b2 = np.empty(self.out_dim, np.float32)
        lib.metal_mlp_get_params(self._h, _ptr(W1), _ptr(b1), _ptr(W2), _ptr(b2))
        return (W1.reshape(self.in_dim, self.hidden), b1,
                W2.reshape(self.hidden, self.out_dim), b2)

    def upload_batch(self, X, y=None):
        X = _c_f32(X)
        b = X.shape[0]
        assert b <= self.max_batch and X.shape[1] == self.in_dim
        self._X = X  # keep alive
        if y is None:
            yp = None
        else:
            self._y = np.ascontiguousarray(y, dtype=np.int32)
            assert self._y.shape[0] == b
            yp = _ptr_i32(self._y)
        lib.metal_mlp_upload_batch(self._h, _ptr(X), yp, c_int64(b))
        self._cur_batch = b

    def forward(self, batch=None):
        b = batch or self._cur_batch
        logits = np.empty(b * self.out_dim, np.float32)
        rc = lib.metal_mlp_forward(self._h, c_int64(b), _ptr(logits))
        if rc:
            raise RuntimeError(f"metal_mlp_forward rc={rc}")
        return logits.reshape(b, self.out_dim)

    def train_step(self, lr: float, batch=None) -> float:
        b = batch or self._cur_batch
        loss = c_float(0.0)
        rc = lib.metal_mlp_train_step(self._h, c_int64(b), c_float(lr), ctypes.byref(loss))
        if rc:
            raise RuntimeError(f"metal_mlp_train_step rc={rc}")
        return float(loss.value)

    def upload_chunk(self, X, y):
        """Upload n_steps consecutive minibatches laid end to end. X is
        [n_steps*batch, in_dim] f32, y is [n_steps*batch] int32."""
        X = _c_f32(X)
        assert X.shape[1] == self.in_dim
        self._X = X                                  # keep alive
        self._y = np.ascontiguousarray(y, dtype=np.int32)
        assert self._y.shape[0] == X.shape[0]
        return X.shape[0]

    def train_steps(self, n_steps: int, batch: int, lr: float) -> float:
        """n_steps SGD steps over the uploaded chunk in ONE command buffer (one host
        sync for the whole chunk). Returns the mean loss over the chunk."""
        if n_steps > self.chunk_steps or batch > self.max_batch:
            raise ValueError(f"train_steps({n_steps}, {batch}) exceeds "
                             f"chunk_steps={self.chunk_steps} / max_batch={self.max_batch}")
        # The C side memcpys n_steps*batch rows out of this pointer, so a short
        # upload would read past the end of the host array.
        if self._X is None or self._X.shape[0] < n_steps * batch:
            have = 0 if self._X is None else self._X.shape[0]
            raise ValueError(f"train_steps needs {n_steps * batch} uploaded rows, "
                             f"upload_chunk provided {have}")
        lib.metal_mlp_upload_chunk(self._h, _ptr(self._X), _ptr_i32(self._y),
                                   c_int64(n_steps), c_int64(batch))
        rc = lib.metal_mlp_train_steps(self._h, c_int64(n_steps), c_int64(batch),
                                       c_float(lr))
        if rc:
            raise RuntimeError(f"metal_mlp_train_steps rc={rc}")
        return float(lib.metal_mlp_last_loss(self._h))

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            lib.metal_mlp_destroy(h); self._h = None
