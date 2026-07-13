# `jaxmetal` — Python package

The Python front-end + ctypes binding over the native Metal runtime
(`build/libmetal_capi.dylib`).

```python
import jaxmetal

jaxmetal.device_name()                        # "Apple M4 Pro"
jaxmetal.matmul(a, b, device="mps")           # jnp.matmul-style, explicit backend
m = jaxmetal.Mlp(784, 1024, 10, max_batch=512)  # resident GPU MLP trainer
m.set_params(W1, b1, W2, b2)
m.upload_batch(x, y); loss = m.train_step(lr=0.5)
```

| Module | What it is |
|---|---|
| `jaxmetal` | Public API: `matmul(device=)`, `device_name`, `Mlp`, `DeviceBuffer` |
| `jaxmetal._capi` | Low-level ctypes binding to the flat C ABI |
| `jaxmetal.ffi` | `matmul` as a jittable XLA FFI custom call |
| `jaxmetal.data` | MNIST download / cache / parse |
| `jaxmetal.reference` | Pure-NumPy golden reference (matches `jax.grad`) |
| `jaxmetal.plugin` | PJRT-device scaffold (roadmap) |

Build the dylib first (`cmake --build build`), then `import jaxmetal` works from the
repo, or `uv pip install --python .venv -e .` to install. See the root
[`README.md`](../README.md) and [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).
