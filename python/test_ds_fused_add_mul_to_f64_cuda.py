import os
import ctypes
import numpy as np

os.environ["JAX_ENABLE_X64"] = "1"
# Do NOT force cpu now.

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import ffi

LIB_PATH = "/src/ds_experiment/cpp/build/libds_ffi.so"

lib = ctypes.cdll.LoadLibrary(LIB_PATH)

ffi.register_ffi_target(
    "ds_fused_add_mul_to_f64_cuda",
    ffi.pycapsule(lib.DsFusedAddMulToF64Cuda),
    platform="CUDA",
)

@jax.jit
def ds_fused_add_mul_to_f64_cuda(x, y):
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    out_type = jax.ShapeDtypeStruct(shape=x.shape, dtype=jnp.float64)
    return ffi.ffi_call(
        "ds_fused_add_mul_to_f64_cuda",
        out_type,
        vmap_method="broadcast_all",
    )(x, y)

x = jnp.array([1.23456789012345, 2.5, 123456789.12345679], dtype=jnp.float64)
y = jnp.array([9.87654321098765, 3.0, 1.23456789e-7], dtype=jnp.float64)

print("devices:", jax.devices())
out = ds_fused_add_mul_to_f64_cuda(x, y)
expected = (x + y) * y
print("out:", np.asarray(out))
print("expected:", np.asarray(expected))
