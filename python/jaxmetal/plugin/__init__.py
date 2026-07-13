"""Scaffold for the PJRT-plugin registration (Stage 3 — the native `mps` device).

This is the wiring JAX uses to discover a C-API PJRT plugin. It is NOT functional
yet: it needs `libpjrt_metal.dylib` exporting `GetPjrtApi()` (a full PJRT C API
implementation), which requires vendoring the version-matched `pjrt_c_api.h` and
implementing the Compile/Execute/Buffer surface. See docs/PJRT_PLUGIN.md.

Once that dylib exists, `initialize()` registers it and JAX exposes a `metal`
device, so `jax.device_put(x, jax.devices('metal')[0])` + `jnp.matmul` run on the
GPU natively (no per-op host copies — the real win). Until then, use the working
paths: `jaxmetal.matmul(a, b, device='mps')` or `jaxmetal.ffi.matmul` (jittable).
"""

from __future__ import annotations

import os

# Confirmed-present registration APIs in this jaxlib (see docs/PJRT_PLUGIN.md):
#   jax._src.xla_bridge.register_plugin(name, library_path=..., ...)
#   jaxlib.xla_client.load_pjrt_plugin_dynamically(name, path)
#   jaxlib.xla_client.make_c_api_client(name)

_PLUGIN_LIB = os.path.join(os.path.dirname(__file__), "libpjrt_metal.dylib")


def initialize() -> None:
    """Register the Metal PJRT plugin with JAX (packaged via a `jax_plugins`
    entry point). Raises until the PJRT dylib is implemented."""
    if not os.path.exists(_PLUGIN_LIB):
        raise NotImplementedError(
            "The Metal PJRT plugin (libpjrt_metal.dylib) is not built yet — this "
            "is the Stage-3 native-device path. See docs/PJRT_PLUGIN.md. For now "
            "use jaxmetal.matmul(..., device='mps') or jaxmetal.ffi.matmul."
        )
    # Target wiring (enabled once the dylib exists):
    #   import jax._src.xla_bridge as xb
    #   xb.register_plugin("metal", library_path=_PLUGIN_LIB, priority=200)
    raise NotImplementedError("PJRT Compile/Execute surface not yet implemented.")
