import os
import ctypes
import numpy as np
import csv

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
    out_type = jax.ShapeDtypeStruct(shape=a.shape, dtype=jnp.float32)
    return ffi.ffi_call("ds_add", out_type, vmap_method="broadcast_all")(a, b)


def ds_sub(a, b):
    out_type = jax.ShapeDtypeStruct(shape=a.shape, dtype=jnp.float32)
    return ffi.ffi_call("ds_sub", out_type, vmap_method="broadcast_all")(a, b)


def run_case(x_scalar, y_scalar):
    x = jnp.array([x_scalar], dtype=jnp.float64)
    y = jnp.array([y_scalar], dtype=jnp.float64)

    truth = y[0]
    fp64_expr = ((x + y) - x)[0]

    # float32 baseline
    x32 = x.astype(jnp.float32)
    y32 = y.astype(jnp.float32)
    f32 = (((x32 + y32) - x32).astype(jnp.float64))[0]

    # DS path
    x_ds = ds_from_f64(x)
    y_ds = ds_from_f64(y)
    ds_raw = ds_sub(ds_add(x_ds, y_ds), x_ds)
    ds = ds_to_f64(ds_raw)[0]

    abs_err_f32 = jnp.abs(f32 - truth)
    abs_err_ds = jnp.abs(ds - truth)

    rel_err_f32 = jnp.where(truth != 0, abs_err_f32 / jnp.abs(truth), jnp.nan)
    rel_err_ds = jnp.where(truth != 0, abs_err_ds / jnp.abs(truth), jnp.nan)

    return {
        "x": float(x_scalar),
        "y": float(y_scalar),
        "truth": float(truth),
        "fp64_expr": float(fp64_expr),
        "f32": float(f32),
        "ds": float(ds),
        "abs_err_f32": float(abs_err_f32),
        "abs_err_ds": float(abs_err_ds),
        "rel_err_f32": float(rel_err_f32),
        "rel_err_ds": float(rel_err_ds),
    }


def write_full_csv(filename, results):
    keys = list(results[0].keys())
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def write_matrix_csv(filename, results, x_values, y_values, key):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # header row
        writer.writerow(["x \\ y"] + list(y_values))

        for x in x_values:
            row = [x]
            for y in y_values:
                r = next(rr for rr in results if rr["x"] == float(x) and rr["y"] == float(y))
                row.append(r[key])
            writer.writerow(row)


def main():
    register_targets()

    x_values = np.array([1e4, 1e6, 1e8, 1e10, 1e12, 1e14], dtype=np.float64)
    y_values = np.array([1e-2, 1e-4, 1e-6, 1e-8], dtype=np.float64)

    print("devices:", jax.devices())
    print("default backend:", jax.default_backend())

    results = []
    for x in x_values:
        for y in y_values:
            results.append(run_case(x, y))

    # --- Write CSVs ---
    out_dir = "artifacts"
    os.makedirs(out_dir, exist_ok=True)

    write_full_csv(f"{out_dir}/ds_cancellation_full.csv", results)

    write_matrix_csv(f"{out_dir}/ds_rel_err_matrix.csv",
                     results, x_values, y_values, "rel_err_ds")

    write_matrix_csv(f"{out_dir}/f32_rel_err_matrix.csv",
                     results, x_values, y_values, "rel_err_f32")

    write_matrix_csv(f"{out_dir}/ds_values_matrix.csv",
                     results, x_values, y_values, "ds")

    print("\nWrote CSV files to:", out_dir)


if __name__ == "__main__":
    main()
