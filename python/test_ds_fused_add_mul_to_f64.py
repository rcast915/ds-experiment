import os
import ctypes
import numpy as np

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import ffi

LIB_PATH = "/src/ds_experiment/cpp/build/libds_ffi.so"

lib = ctypes.cdll.LoadLibrary(LIB_PATH)

ffi.register_ffi_target(
    "ds_fused_add_mul_to_f64",
    ffi.pycapsule(lib.DsFusedAddMulToF64),
    platform="cpu",
)

def ds_fused_add_mul_to_f64(x, y):
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    out_type = jax.ShapeDtypeStruct(shape=x.shape, dtype=jnp.float64)
    return ffi.ffi_call(
        "ds_fused_add_mul_to_f64",
        out_type,
        vmap_method="broadcast_all",
    )(x, y)

x = jnp.array([1.23456789012345, 2.5, 123456789.12345679], dtype=jnp.float64)
y = jnp.array([9.87654321098765, 3.0, 1.23456789e-7], dtype=jnp.float64)

out = ds_fused_add_mul_to_f64(x, y)
expected = (x + y) * y

print("out:", out)
print("expected:", expected)
np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-5, atol=1e-5)
print("PASS")
