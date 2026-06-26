import os
import ctypes
import numpy as np

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import ffi

lib = ctypes.cdll.LoadLibrary("/src/ds_experiment/cpp/build/libds_ffi.so")

ffi.register_ffi_target(
    "ds_from_f64",
    ffi.pycapsule(lib.DsFromF64),
    platform="cpu",
)

def ds_from_f64(x):
    x = jnp.asarray(x, dtype=jnp.float64)
    out_type = jax.ShapeDtypeStruct(shape=x.shape + (2,), dtype=jnp.float32)
    return ffi.ffi_call(
        "ds_from_f64",
        out_type,
        vmap_method="broadcast_all",
    )(x)

x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)

print("devices:", jax.devices())
print("default backend:", jax.default_backend())

y = ds_from_f64(x)

print("input:", x)
print("output:", y)
print("output shape:", y.shape)
print("output dtype:", y.dtype)

reconstructed = y[:, 0].astype(jnp.float64) + y[:, 1].astype(jnp.float64)
print("reconstructed:", reconstructed)

np.testing.assert_allclose(np.asarray(reconstructed), np.asarray(x), rtol=1e-6, atol=1e-6)
print("PASS")
