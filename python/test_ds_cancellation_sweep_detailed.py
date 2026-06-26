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


def register_targets():
    lib = ctypes.cdll.LoadLibrary(LIB_PATH)

    ffi.register_ffi_target("ds_from_f64", ffi.pycapsule(lib.DsFromF64), platform="cpu")
    ffi.register_ffi_target("ds_to_f64",   ffi.pycapsule(lib.DsToF64),   platform="cpu")
    ffi.register_ffi_target("ds_add",      ffi.pycapsule(lib.DsAdd),     platform="cpu")
    ffi.register_ffi_target("ds_sub",      ffi.pycapsule(lib.DsSub),     platform="cpu")
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
    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    out_type = jax.ShapeDtypeStruct(shape=a.shape, dtype=jnp.float32)
    return ffi.ffi_call("ds_add", out_type, vmap_method="broadcast_all")(a, b)


def ds_sub(a, b):
    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    out_type = jax.ShapeDtypeStruct(shape=a.shape, dtype=jnp.float32)
    return ffi.ffi_call("ds_sub", out_type, vmap_method="broadcast_all")(a, b)


def run_sweep(y_value=1e-4):
    # You can change or extend this list.
    x_values = jnp.array(
        [1e4, 1e6, 1e8, 1e10, 1e12, 1e14],
        dtype=jnp.float64,
    )
    y_values = jnp.full_like(x_values, y_value)

    # Reference in float64
    ref = (x_values + y_values) - x_values

    # Plain float32 baseline
    x32 = x_values.astype(jnp.float32)
    y32 = y_values.astype(jnp.float32)
    f32 = ((x32 + y32) - x32).astype(jnp.float64)

    # DS path
    x_ds = ds_from_f64(x_values)
    y_ds = ds_from_f64(y_values)
    ds = ds_to_f64(ds_sub(ds_add(x_ds, y_ds), x_ds))

    abs_err_f32 = jnp.abs(f32 - ref)
    abs_err_ds = jnp.abs(ds - ref)

    # Avoid divide-by-zero if ref ever becomes zero.
    rel_err_f32 = jnp.where(ref != 0, abs_err_f32 / jnp.abs(ref), jnp.nan)
    rel_err_ds = jnp.where(ref != 0, abs_err_ds / jnp.abs(ref), jnp.nan)

    print("devices:", jax.devices())
    print("default backend:", jax.default_backend())
    print()
    print(f"Cancellation sweep for y = {y_value:g}")
    print()

    header = (
        f"{'x':>14}  "
        f"{'ref':>14}  "
        f"{'f32':>14}  "
        f"{'ds':>14}  "
        f"{'abs_err_f32':>14}  "
        f"{'abs_err_ds':>14}"
    )
    print(header)
    print("-" * len(header))

    x_np = np.asarray(x_values)
    ref_np = np.asarray(ref)
    f32_np = np.asarray(f32)
    ds_np = np.asarray(ds)
    ae_f32_np = np.asarray(abs_err_f32)
    ae_ds_np = np.asarray(abs_err_ds)
    re_f32_np = np.asarray(rel_err_f32)
    re_ds_np = np.asarray(rel_err_ds)

    for i in range(len(x_np)):
        print(
            f"{x_np[i]:14.6e}  "
            f"{ref_np[i]:14.6e}  "
            f"{f32_np[i]:14.6e}  "
            f"{ds_np[i]:14.6e}  "
            f"{ae_f32_np[i]:14.6e}  "
            f"{ae_ds_np[i]:14.6e}"
        )

    print()
    print("Relative errors:")
    print(f"{'x':>14}  {'rel_err_f32':>14}  {'rel_err_ds':>14}")
    print("-" * 48)
    for i in range(len(x_np)):
        print(
            f"{x_np[i]:14.6e}  "
            f"{re_f32_np[i]:14.6e}  "
            f"{re_ds_np[i]:14.6e}"
        )

    print()
    print("DS low parts for x and y:")
    print("x_ds[:, 1] =", np.asarray(x_ds)[:, 1])
    print("y_ds[:, 1] =", np.asarray(y_ds)[:, 1])

    print()
    print("Recovered DS result low-level representation:")
    out_ds = ds_sub(ds_add(x_ds, y_ds), x_ds)
    print(np.asarray(out_ds))


if __name__ == "__main__":
    register_targets()
    run_sweep(y_value=1e-4)
