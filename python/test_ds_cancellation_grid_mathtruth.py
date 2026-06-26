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


def run_case(x_scalar, y_scalar):
    x = jnp.array([x_scalar], dtype=jnp.float64)
    y = jnp.array([y_scalar], dtype=jnp.float64)

    # Floating-point reference of the unstable expression.
    fp64_expr = ((x + y) - x)[0]

    # Mathematical truth for this benchmark.
    truth = y[0]

    # Plain float32 baseline.
    x32 = x.astype(jnp.float32)
    y32 = y.astype(jnp.float32)
    f32 = (((x32 + y32) - x32).astype(jnp.float64))[0]

    # DS path.
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
        "x_hi": float(np.asarray(x_ds)[0, 0]),
        "x_lo": float(np.asarray(x_ds)[0, 1]),
        "y_hi": float(np.asarray(y_ds)[0, 0]),
        "y_lo": float(np.asarray(y_ds)[0, 1]),
        "ds_hi": float(np.asarray(ds_raw)[0, 0]),
        "ds_lo": float(np.asarray(ds_raw)[0, 1]),
    }


def print_table(results):
    header = (
        f"{'x':>12}  {'y':>12}  {'truth':>12}  {'fp64_expr':>12}  "
        f"{'f32':>12}  {'ds':>12}  {'abs_f32':>12}  {'abs_ds':>12}  "
        f"{'rel_f32':>12}  {'rel_ds':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['x']:12.4e}  "
            f"{r['y']:12.4e}  "
            f"{r['truth']:12.4e}  "
            f"{r['fp64_expr']:12.4e}  "
            f"{r['f32']:12.4e}  "
            f"{r['ds']:12.4e}  "
            f"{r['abs_err_f32']:12.4e}  "
            f"{r['abs_err_ds']:12.4e}  "
            f"{r['rel_err_f32']:12.4e}  "
            f"{r['rel_err_ds']:12.4e}"
        )


def print_summary_matrix(results, x_values, y_values):
    print("\nDS better than float32 relative to mathematical truth?  (Y/N/=)")
    header = " " * 12 + "".join(f"{y:>12.0e}" for y in y_values)
    print(header)
    for x in x_values:
        row = f"{x:>12.0e}"
        for y in y_values:
            r = next(rr for rr in results if rr["x"] == float(x) and rr["y"] == float(y))
            if np.isnan(r["abs_err_f32"]) or np.isnan(r["abs_err_ds"]):
                mark = "?"
            elif r["abs_err_ds"] < r["abs_err_f32"]:
                mark = "Y"
            elif r["abs_err_ds"] > r["abs_err_f32"]:
                mark = "N"
            else:
                mark = "="
            row += f"{mark:>12}"
        print(row)


def print_recovered_matrix(results, x_values, y_values):
    print("\nRecovered values (DS):")
    header = " " * 12 + "".join(f"{y:>18.0e}" for y in y_values)
    print(header)
    for x in x_values:
        row = f"{x:>12.0e}"
        for y in y_values:
            r = next(rr for rr in results if rr["x"] == float(x) and rr["y"] == float(y))
            row += f"{r['ds']:18.6e}"
        print(row)


def print_fp64_expr_matrix(results, x_values, y_values):
    print("\nFloat64 evaluation of ((x + y) - x):")
    header = " " * 12 + "".join(f"{y:>18.0e}" for y in y_values)
    print(header)
    for x in x_values:
        row = f"{x:>12.0e}"
        for y in y_values:
            r = next(rr for rr in results if rr["x"] == float(x) and rr["y"] == float(y))
            row += f"{r['fp64_expr']:18.6e}"
        print(row)


def print_relerr_matrix(results, x_values, y_values, key, title):
    print(f"\n{title}")
    header = " " * 12 + "".join(f"{y:>18.0e}" for y in y_values)
    print(header)
    for x in x_values:
        row = f"{x:>12.0e}"
        for y in y_values:
            r = next(rr for rr in results if rr["x"] == float(x) and rr["y"] == float(y))
            row += f"{r[key]:18.6e}"
        print(row)


def main():
    register_targets()

    x_values = np.array([1e4, 1e6, 1e8, 1e10, 1e12, 1e14], dtype=np.float64)
    y_values = np.array([1e-2, 1e-4, 1e-6, 1e-8], dtype=np.float64)

    print("devices:", jax.devices())
    print("default backend:", jax.default_backend())
    print()

    results = []
    for x in x_values:
        for y in y_values:
            results.append(run_case(x, y))

    print_table(results)
    print_summary_matrix(results, x_values, y_values)
    print_recovered_matrix(results, x_values, y_values)
    print_fp64_expr_matrix(results, x_values, y_values)
    print_relerr_matrix(results, x_values, y_values, "rel_err_f32",
                        "Relative error to mathematical truth: float32")
    print_relerr_matrix(results, x_values, y_values, "rel_err_ds",
                        "Relative error to mathematical truth: DS")

    max_abs_f32 = max(r["abs_err_f32"] for r in results if not np.isnan(r["abs_err_f32"]))
    max_abs_ds = max(r["abs_err_ds"] for r in results if not np.isnan(r["abs_err_ds"]))
    max_rel_f32 = max(r["rel_err_f32"] for r in results if not np.isnan(r["rel_err_f32"]))
    max_rel_ds = max(r["rel_err_ds"] for r in results if not np.isnan(r["rel_err_ds"]))

    print("\nSummary relative to mathematical truth:")
    print("max abs err f32:", max_abs_f32)
    print("max abs err ds: ", max_abs_ds)
    print("max rel err f32:", max_rel_f32)
    print("max rel err ds: ", max_rel_ds)


if __name__ == "__main__":
    main()
