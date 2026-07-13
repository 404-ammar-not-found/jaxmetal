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

__all__ = [
    "lib", "dylib_path", "library_path", "device_name",
    "DeviceBuffer",
    "matmul", "matmul_mps", "matmul_cpu", "matmul_auto",
    "matmul_resident", "matmul_resident_mps",
    "Mlp",
]


def _find_dylib() -> str:
    override = os.environ.get("METALMM_DYLIB")
    if override:
        if not os.path.exists(override):
            raise FileNotFoundError(f"METALMM_DYLIB={override!r} does not exist")
        return override
    here = os.path.abspath(os.path.dirname(__file__))          # python/metalmm
    repo_root = os.path.dirname(os.path.dirname(here))          # repo root
    candidates = [
        os.path.join(repo_root, "build", "libmetal_capi.dylib"),
        os.path.join(repo_root, "build", "Release", "libmetal_capi.dylib"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "Could not locate libmetal_capi.dylib. Build it first:\n"
        "  cmake -S . -B build -G Ninja && cmake --build build\n"
        f"Searched: {candidates}\n"
        "Or set METALMM_DYLIB=/abs/path/to/libmetal_capi.dylib")


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

lib.metal_mlp_create.argtypes = [c_int64, c_int64, c_int64, c_int64]
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
    def __init__(self, in_dim: int, hidden: int, out_dim: int, max_batch: int):
        self.in_dim, self.hidden, self.out_dim = in_dim, hidden, out_dim
        self.max_batch = max_batch
        self._cur_batch = 0
        self._h = lib.metal_mlp_create(c_int64(in_dim), c_int64(hidden),
                                       c_int64(out_dim), c_int64(max_batch))
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

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            lib.metal_mlp_destroy(h); self._h = None
