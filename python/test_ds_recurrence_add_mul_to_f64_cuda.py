import os
import ctypes
import numpy as np

os.environ["JAX_ENABLE_X64"] = "1"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import ffi
from functools import partial

LIB_PATH = "/src/ds_experiment/cpp/build/libds_ffi.so"

lib = ctypes.cdll.LoadLibrary(LIB_PATH)

ffi.register_ffi_target(
    "ds_recurrence_add_mul_to_f64_cuda",
    ffi.pycapsule(lib.DsRecurrenceAddMulToF64Cuda),
    platform="CUDA",
)

@partial(jax.jit, static_argnames=["k"])
def ds_recurrence(x, y, k: int):
    out_type = jax.ShapeDtypeStruct(shape=x.shape, dtype=jnp.float64)
    return ffi.ffi_call(
        "ds_recurrence_add_mul_to_f64_cuda",
        out_type,
        vmap_method="broadcast_all",
    )(x, y, k=np.int32(k))

def ref_recurrence(x, y, k: int):
    z = x
    for _ in range(k):
        z = (z + y) * y
    return z

x = jnp.array([0.25, 0.5, 0.75], dtype=jnp.float64)
y = jnp.array([0.2, 0.3, 0.4], dtype=jnp.float64)
k = 8

out = ds_recurrence(x, y, k)
ref = ref_recurrence(x, y, k)

print("devices:", jax.devices())
print("out:", np.asarray(out))
print("ref:", np.asarray(ref))
print("max abs err:", np.max(np.abs(np.asarray(out) - np.asarray(ref))))
