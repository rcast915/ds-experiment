import os
import ctypes
import csv
import numpy as np

os.environ["JAX_ENABLE_X64"] = "1"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import ffi

LIB_PATH = "/src/ds_experiment/cpp/build/libds_ffi.so"


def register_targets():
    lib = ctypes.cdll.LoadLibrary(LIB_PATH)
    ffi.register_ffi_target(
        "ds_fused_add_sub_to_f64_cuda",
        ffi.pycapsule(lib.DsFusedAddSubToF64Cuda),
        platform="CUDA",
    )


def block(x):
    x.block_until_ready()
    return x


@jax.jit
def f64_cancel(x, y):
    return (x + y) - x


@jax.jit
def f32_cancel(x, y):
    return (x + y) - x


@jax.jit
def ds_cancel_cuda(x, y):
    out_type = jax.ShapeDtypeStruct(shape=x.shape, dtype=jnp.float64)
    return ffi.ffi_call(
        "ds_fused_add_sub_to_f64_cuda",
        out_type,
        vmap_method="broadcast_all",
    )(x, y)


def run_grid():
    x_values = np.array([1e4, 1e6, 1e8, 1e10, 1e12, 1e14], dtype=np.float64)
    y_values = np.array([1e-2, 1e-4, 1e-6, 1e-8], dtype=np.float64)

    rows = []

    for x_scalar in x_values:
        for y_scalar in y_values:
            x64 = jnp.array([x_scalar], dtype=jnp.float64)
            y64 = jnp.array([y_scalar], dtype=jnp.float64)

            x32 = jnp.array([x_scalar], dtype=jnp.float32)
            y32 = jnp.array([y_scalar], dtype=jnp.float32)

            truth = float(y_scalar)

            out_f64 = float(np.asarray(block(f64_cancel(x64, y64)))[0])
            out_f32 = float(np.asarray(block(f32_cancel(x32, y32)))[0])
            out_ds = float(np.asarray(block(ds_cancel_cuda(x64, y64)))[0])

            abs_err_f64 = abs(out_f64 - truth)
            abs_err_f32 = abs(out_f32 - truth)
            abs_err_ds = abs(out_ds - truth)

            rel_err_f64 = abs_err_f64 / abs(truth) if truth != 0 else np.nan
            rel_err_f32 = abs_err_f32 / abs(truth) if truth != 0 else np.nan
            rel_err_ds = abs_err_ds / abs(truth) if truth != 0 else np.nan

            rows.append({
                "x": x_scalar,
                "y": y_scalar,
                "truth": truth,
                "f64": out_f64,
                "f32": out_f32,
                "ds": out_ds,
                "abs_err_f64": abs_err_f64,
                "abs_err_f32": abs_err_f32,
                "abs_err_ds": abs_err_ds,
                "rel_err_f64": rel_err_f64,
                "rel_err_f32": rel_err_f32,
                "rel_err_ds": rel_err_ds,
            })

    return rows, x_values, y_values


def write_full_csv(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(path, rows, x_values, y_values, field):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x \\ y"] + list(y_values))
        for x in x_values:
            row = [x]
            for y in y_values:
                r = next(rr for rr in rows if rr["x"] == float(x) and rr["y"] == float(y))
                row.append(r[field])
            writer.writerow(row)


def write_better_matrix_csv(path, rows, x_values, y_values):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x \\ y"] + list(y_values))
        for x in x_values:
            row = [x]
            for y in y_values:
                r = next(rr for rr in rows if rr["x"] == float(x) and rr["y"] == float(y))
                if np.isnan(r["abs_err_ds"]) or np.isnan(r["abs_err_f32"]):
                    val = "?"
                elif r["abs_err_ds"] < r["abs_err_f32"]:
                    val = "Y"
                elif r["abs_err_ds"] > r["abs_err_f32"]:
                    val = "N"
                else:
                    val = "="
                row.append(val)
            writer.writerow(row)


def main():
    register_targets()

    print("devices:", jax.devices())
    print("default backend:", jax.default_backend())

    rows, x_values, y_values = run_grid()

    out_dir = "artifacts/gpu_cancellation"
    os.makedirs(out_dir, exist_ok=True)

    write_full_csv(os.path.join(out_dir, "gpu_cancellation_full.csv"), rows)

    write_matrix_csv(os.path.join(out_dir, "gpu_rel_err_f64_matrix.csv"),
                     rows, x_values, y_values, "rel_err_f64")
    write_matrix_csv(os.path.join(out_dir, "gpu_rel_err_f32_matrix.csv"),
                     rows, x_values, y_values, "rel_err_f32")
    write_matrix_csv(os.path.join(out_dir, "gpu_rel_err_ds_matrix.csv"),
                     rows, x_values, y_values, "rel_err_ds")

    write_matrix_csv(os.path.join(out_dir, "gpu_values_f64_matrix.csv"),
                     rows, x_values, y_values, "f64")
    write_matrix_csv(os.path.join(out_dir, "gpu_values_f32_matrix.csv"),
                     rows, x_values, y_values, "f32")
    write_matrix_csv(os.path.join(out_dir, "gpu_values_ds_matrix.csv"),
                     rows, x_values, y_values, "ds")

    write_better_matrix_csv(os.path.join(out_dir, "gpu_ds_better_than_f32_matrix.csv"),
                            rows, x_values, y_values)

    print("\nWrote CSV files to:", out_dir)
    for name in sorted(os.listdir(out_dir)):
        print(" -", os.path.join(out_dir, name))


if __name__ == "__main__":
    main()
