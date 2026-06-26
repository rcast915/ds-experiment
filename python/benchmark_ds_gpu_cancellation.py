import os
import ctypes
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

    return rows


def print_table(rows):
    header = (
        f"{'x':>12}  {'y':>12}  {'truth':>12}  "
        f"{'f64':>12}  {'f32':>12}  {'ds':>12}  "
        f"{'rel_f64':>12}  {'rel_f32':>12}  {'rel_ds':>12}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['x']:12.4e}  "
            f"{r['y']:12.4e}  "
            f"{r['truth']:12.4e}  "
            f"{r['f64']:12.4e}  "
            f"{r['f32']:12.4e}  "
            f"{r['ds']:12.4e}  "
            f"{r['rel_err_f64']:12.4e}  "
            f"{r['rel_err_f32']:12.4e}  "
            f"{r['rel_err_ds']:12.4e}"
        )


def print_summary_matrix(rows, field, title):
    x_values = sorted({r["x"] for r in rows})
    y_values = sorted({r["y"] for r in rows}, reverse=True)

    print(f"\n{title}")
    print(" " * 12 + "".join(f"{y:>14.0e}" for y in y_values))

    for x in x_values:
        row = f"{x:>12.0e}"
        for y in y_values:
            r = next(rr for rr in rows if rr["x"] == x and rr["y"] == y)
            row += f"{r[field]:14.3e}"
        print(row)


def main():
    register_targets()

    print("devices:", jax.devices())
    print("default backend:", jax.default_backend())
    print()

    rows = run_grid()
    print_table(rows)

    print_summary_matrix(rows, "rel_err_f64", "Relative error to mathematical truth: float64")
    print_summary_matrix(rows, "rel_err_f32", "Relative error to mathematical truth: float32")
    print_summary_matrix(rows, "rel_err_ds",  "Relative error to mathematical truth: DS")

    best = 0
    total = 0
    for r in rows:
        total += 1
        if r["abs_err_ds"] < r["abs_err_f32"]:
            best += 1

    print()
    print(f"DS better than float32 in {best}/{total} cases")


if __name__ == "__main__":
    main()
