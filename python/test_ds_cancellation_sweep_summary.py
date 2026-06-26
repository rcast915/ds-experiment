import os
import ctypes
import numpy as np

# Force CPU because the FFI targets are registered only for CPU.
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import ffi

# -----------------------------------------------------------------------------
# Load and register FFI targets
# -----------------------------------------------------------------------------

LIB_PATH = "/src/ds_experiment/cpp/build/libds_ffi.so"
lib = ctypes.cdll.LoadLibrary(LIB_PATH)

ffi.register_ffi_target("ds_from_f64", ffi.pycapsule(lib.DsFromF64), platform="cpu")
ffi.register_ffi_target("ds_to_f64",   ffi.pycapsule(lib.DsToF64),   platform="cpu")
ffi.register_ffi_target("ds_add",      ffi.pycapsule(lib.DsAdd),     platform="cpu")
ffi.register_ffi_target("ds_sub",      ffi.pycapsule(lib.DsSub),     platform="cpu")
ffi.register_ffi_target("ds_mul",      ffi.pycapsule(lib.DsMul),     platform="cpu")

# -----------------------------------------------------------------------------
# DS wrappers
# -----------------------------------------------------------------------------

def ds_from_f64(x):
    x = jnp.asarray(x, dtype=jnp.float64)
    out_type = jax.ShapeDtypeStruct(shape=x.shape + (2,), dtype=jnp.float32)
    return ffi.ffi_call("ds_from_f64", out_type, vmap_method="broadcast_all")(x)

def ds_to_f64(x):
    x = jnp.asarray(x, dtype=jnp.float32)
    out_type = jax.ShapeDtypeStruct(shape=x.shape[:-1], dtype=jnp.float64)
    return ffi.ffi_call("ds_to_f64", out_type, vmap_method="broadcast_all")(x)

def ds_add(a, b):
    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    out_type = jax.ShapeDtypeStruct(shape=a.shape, dtype=jnp.float32)
    return ffi.ffi_call("ds_add", out_type, vmap_method="broadcast_all")(a, b)

def ds_sub(a, b):
    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    out_type = jax.ShapeDtypeStruct(shape=a.shape, dtype=jnp.float32)
    return ffi.ffi_call("ds_sub", out_type, vmap_method="broadcast_all")(a, b)

def ds_mul(a, b):
    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    out_type = jax.ShapeDtypeStruct(shape=a.shape, dtype=jnp.float32)
    return ffi.ffi_call("ds_mul", out_type, vmap_method="broadcast_all")(a, b)

# -----------------------------------------------------------------------------
# Experiment
# -----------------------------------------------------------------------------

def cancellation_expr_ref(x, y):
    # Reference computed in float64 JAX
    return (x + y) - x

def cancellation_expr_f32(x, y):
    x32 = x.astype(jnp.float32)
    y32 = y.astype(jnp.float32)
    return ((x32 + y32) - x32).astype(jnp.float64)

def cancellation_expr_ds(x, y):
    x_ds = ds_from_f64(x)
    y_ds = ds_from_f64(y)
    return ds_to_f64(ds_sub(ds_add(x_ds, y_ds), x_ds))

def main():
    print("devices:", jax.devices())
    print("default backend:", jax.default_backend())
    print()

    # Sweep x over many magnitudes; keep y fixed and small.
    xs = jnp.array([1e4, 1e6, 1e8, 1e10, 1e12, 1e14], dtype=jnp.float64)
    y_value = 1e-4
    ys = jnp.full_like(xs, y_value)

    ref = cancellation_expr_ref(xs, ys)
    f32 = cancellation_expr_f32(xs, ys)
    ds = cancellation_expr_ds(xs, ys)

    abs_err_f32 = jnp.abs(f32 - ref)
    abs_err_ds = jnp.abs(ds - ref)

    # Avoid divide-by-zero if any reference entry is exactly zero.
    rel_err_f32 = jnp.where(ref != 0, abs_err_f32 / jnp.abs(ref), jnp.nan)
    rel_err_ds = jnp.where(ref != 0, abs_err_ds / jnp.abs(ref), jnp.nan)

    header = (
        f"{'x':>14}  {'y':>12}  {'ref':>14}  {'f32':>14}  {'ds':>14}  "
        f"{'abs_err_f32':>14}  {'abs_err_ds':>14}  "
        f"{'rel_err_f32':>14}  {'rel_err_ds':>14}"
    )
    print(header)
    print("-" * len(header))

    xs_np = np.asarray(xs)
    ys_np = np.asarray(ys)
    ref_np = np.asarray(ref)
    f32_np = np.asarray(f32)
    ds_np = np.asarray(ds)
    abs_err_f32_np = np.asarray(abs_err_f32)
    abs_err_ds_np = np.asarray(abs_err_ds)
    rel_err_f32_np = np.asarray(rel_err_f32)
    rel_err_ds_np = np.asarray(rel_err_ds)

    for i in range(len(xs_np)):
        print(
            f"{xs_np[i]:14.6e}  "
            f"{ys_np[i]:12.6e}  "
            f"{ref_np[i]:14.6e}  "
            f"{f32_np[i]:14.6e}  "
            f"{ds_np[i]:14.6e}  "
            f"{abs_err_f32_np[i]:14.6e}  "
            f"{abs_err_ds_np[i]:14.6e}  "
            f"{rel_err_f32_np[i]:14.6e}  "
            f"{rel_err_ds_np[i]:14.6e}"
        )

    print()
    print("Summary:")
    print("max abs err f32:", np.max(abs_err_f32_np))
    print("max abs err ds: ", np.max(abs_err_ds_np))
    print("max rel err f32:", np.nanmax(rel_err_f32_np))
    print("max rel err ds: ", np.nanmax(rel_err_ds_np))

if __name__ == "__main__":
    main()
