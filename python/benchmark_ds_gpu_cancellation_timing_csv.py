import os
import ctypes
import time
import csv
import numpy as np

os.environ["JAX_ENABLE_X64"] = "1"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import ffi

LIB_PATH = "/src/ds_experiment/cpp/build/libds_ffi.so"
CSV_PATH = "ds_gpu_cancellation_timing.csv"


# -----------------------------
# FFI registration
# -----------------------------
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


# -----------------------------
# Kernels
# -----------------------------
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


# -----------------------------
# Timing
# -----------------------------
def time_fn(fn, *args, warmup=10, iters=50):
    for _ in range(warmup):
        out = fn(*args)
        block(out)

    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
        block(out)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


# -----------------------------
# Bandwidth helpers
# -----------------------------
def bytes_f64(n):
    return 8 * n * 3


def bytes_f32(n):
    return 4 * n * 3


def bytes_ds(n):
    return 8 * n * 3


def gbps(nbytes, sec):
    return nbytes / sec / 1e9


# -----------------------------
# Benchmark
# -----------------------------
def benchmark_case(n, x_scalar, y_scalar, iters):
    x64 = jnp.full((n,), x_scalar, dtype=jnp.float64)
    y64 = jnp.full((n,), y_scalar, dtype=jnp.float64)

    x32 = jnp.full((n,), x_scalar, dtype=jnp.float32)
    y32 = jnp.full((n,), y_scalar, dtype=jnp.float32)

    t_f64 = time_fn(f64_cancel, x64, y64, iters=iters)
    t_f32 = time_fn(f32_cancel, x32, y32, iters=iters)
    t_ds = time_fn(ds_cancel_cuda, x64, y64, iters=iters)

    # Accuracy check (single element)
    truth = float(y_scalar)

    out_f64 = float(np.asarray(block(f64_cancel(x64[:1], y64[:1])))[0])
    out_f32 = float(np.asarray(block(f32_cancel(x32[:1], y32[:1])))[0])
    out_ds = float(np.asarray(block(ds_cancel_cuda(x64[:1], y64[:1])))[0])

    rel_f64 = abs(out_f64 - truth) / abs(truth) if truth != 0 else np.nan
    rel_f32 = abs(out_f32 - truth) / abs(truth) if truth != 0 else np.nan
    rel_ds = abs(out_ds - truth) / abs(truth) if truth != 0 else np.nan

    return {
        "N": n,
        "x": x_scalar,
        "y": y_scalar,
        "f64_ms": 1e3 * t_f64,
        "f32_ms": 1e3 * t_f32,
        "ds_ms": 1e3 * t_ds,
        "ds_over_f64": t_ds / t_f64,
        "ds_over_f32": t_ds / t_f32,
        "f64_GBs": gbps(bytes_f64(n), t_f64),
        "f32_GBs": gbps(bytes_f32(n), t_f32),
        "ds_GBs": gbps(bytes_ds(n), t_ds),
        "rel_err_f64": rel_f64,
        "rel_err_f32": rel_f32,
        "rel_err_ds": rel_ds,
        "out_f64": out_f64,
        "out_f32": out_f32,
        "out_ds": out_ds,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    register_targets()

    print("devices:", jax.devices())
    print("default backend:", jax.default_backend())
    print()

    Ns = [1_000_000, 10_000_000]
    xs = [1e4, 1e6, 1e8, 1e10, 1e12]
    ys = [1e-2, 1e-4, 1e-6, 1e-8]

    rows = []

    for n in Ns:
        for x in xs:
            for y in ys:
                iters = 50 if n <= 1_000_000 else 20
                print(f"Running N={n}, x={x:.0e}, y={y:.0e}")
                row = benchmark_case(n, x, y, iters)
                rows.append(row)

    # -----------------------------
    # Write CSV
    # -----------------------------
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Wrote results to: {CSV_PATH}")


if __name__ == "__main__":
    main()
