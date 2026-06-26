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

ffi.register_ffi_target(
    "ds_to_f64",
    ffi.pycapsule(lib.DsToF64),
    platform="cpu",
)

def ds_from_f64(x):
    x = jnp.asarray(x, dtype=jnp.float64)
    out_type = jax.ShapeDtypeStruct(shape=x.shape + (2,), dtype=jnp.float32)
    return ffi.ffi_call("ds_from_f64", out_type, vmap_method="broadcast_all")(x)

def ds_to_f64(x):
    x = jnp.asarray(x, dtype=jnp.float32)
    out_type = jax.ShapeDtypeStruct(shape=x.shape[:-1], dtype=jnp.float64)
    return ffi.ffi_call("ds_to_f64", out_type, vmap_method="broadcast_all")(x)

x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)

ds = ds_from_f64(x)
rt = ds_to_f64(ds)

print("devices:", jax.devices())
print("input:", x)
print("ds:", ds)
print("roundtrip:", rt)

np.testing.assert_allclose(np.asarray(rt), np.asarray(x), rtol=1e-6, atol=1e-6)
print("PASS")
