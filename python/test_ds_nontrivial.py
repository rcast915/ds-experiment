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

ffi.register_ffi_target("ds_from_f64", ffi.pycapsule(lib.DsFromF64), platform="cpu")
ffi.register_ffi_target("ds_to_f64",   ffi.pycapsule(lib.DsToF64),   platform="cpu")
ffi.register_ffi_target("ds_add",      ffi.pycapsule(lib.DsAdd),     platform="cpu")
ffi.register_ffi_target("ds_mul",      ffi.pycapsule(lib.DsMul),     platform="cpu")

def ds_from_f64(x):
    x = jnp.asarray(x, dtype=jnp.float64)
    out_type = jax.ShapeDtypeStruct(shape=x.shape + (2,), dtype=jnp.float32)
    return ffi.ffi_call("ds_from_f64", out_type, vmap_method="broadcast_all")(x)

def ds_to_f64(x):
    x = jnp.asarray(x, dtype=jnp.float32)
    out_type = jax.ShapeDtypeStruct(shape=x.shape[:-1], dtype=jnp.float64)
    return ffi.ffi_call("ds_to_f64", out_type, vmap_method="broadcast_all")(x)

def ds_add(a, b):
    out_type = jax.ShapeDtypeStruct(shape=a.shape, dtype=jnp.float32)
    return ffi.ffi_call("ds_add", out_type, vmap_method="broadcast_all")(a, b)

def ds_mul(a, b):
    out_type = jax.ShapeDtypeStruct(shape=a.shape, dtype=jnp.float32)
    return ffi.ffi_call("ds_mul", out_type, vmap_method="broadcast_all")(a, b)

x = jnp.array([
    1.0000000001,
    1.23456789012345,
    123456789.12345679,
], dtype=jnp.float64)

y = jnp.array([
    2.0000000001,
    9.87654321098765,
    0.000000123456789,
], dtype=jnp.float64)

x_ds = ds_from_f64(x)
y_ds = ds_from_f64(y)

out_ds = ds_mul(ds_add(x_ds, y_ds), y_ds)
out = ds_to_f64(out_ds)

expected = (x + y) * y

print("out_ds:", out_ds)
print("out:", out)
print("expected:", expected)

np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-5, atol=1e-5)
print("PASS")
