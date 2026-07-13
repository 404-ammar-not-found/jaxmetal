"""MNIST loader: download + cache the 4 IDX.gz files, parse to normalized f32.

Source (verified reachable): https://storage.googleapis.com/cvdf-datasets/mnist/
Cache dir: $METALMM_DATA_DIR or <repo>/data/mnist/.
"""
from __future__ import annotations
import gzip, os, struct, urllib.request
import numpy as np

_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}
_EXPECT = {"train_images": (60000, 784), "train_labels": (60000,),
           "test_images": (10000, 784),  "test_labels": (10000,)}


def _data_dir() -> str:
    d = os.environ.get("METALMM_DATA_DIR")
    if not d:
        # this file: <repo>/python/jaxmetal/data.py -> three dirnames to <repo>.
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        d = os.path.join(repo, "data", "mnist")
    os.makedirs(d, exist_ok=True)
    return d


def _download(fname: str, dst: str):
    url = _BASE + fname
    tmp = dst + ".part"
    try:
        print(f"  downloading {url}")
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            f.write(r.read())
        os.replace(tmp, dst)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            f"Failed to download {url}: {e}\n"
            f"Check network access, or manually place {fname} in {os.path.dirname(dst)}") from e


def _parse_idx(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        if magic == 2051:                      # images
            n, rows, cols = struct.unpack(">III", f.read(12))
            buf = f.read(n * rows * cols)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols)
        elif magic == 2049:                    # labels
            n, = struct.unpack(">I", f.read(4))
            arr = np.frombuffer(f.read(n), dtype=np.uint8)
        else:
            raise ValueError(f"{path}: bad IDX magic {magic}")
    return arr


def _get(key: str) -> np.ndarray:
    path = os.path.join(_data_dir(), _FILES[key])
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        _download(_FILES[key], path)
    arr = _parse_idx(path)
    exp = _EXPECT[key]
    if arr.shape != exp:
        raise ValueError(f"{key}: expected {exp}, got {arr.shape} (corrupt cache? delete {path})")
    if key.endswith("images"):
        return arr.astype(np.float32) / 255.0     # [N,784] in [0,1]
    return arr.astype(np.int32)                    # [N]


def load_mnist():
    """Returns (X_train[60000,784] f32, y_train[60000] i32,
                X_test[10000,784] f32,  y_test[10000] i32)."""
    Xtr = _get("train_images"); ytr = _get("train_labels")
    Xte = _get("test_images");  yte = _get("test_labels")
    return Xtr, ytr, Xte, yte


if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load_mnist()
    print("train", Xtr.shape, Xtr.dtype, "labels", ytr.shape,
          "range", float(Xtr.min()), float(Xtr.max()),
          "classes", int(ytr.min()), int(ytr.max()), "cache", _data_dir())
