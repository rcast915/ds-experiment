import os
import ctypes
import time
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
        "ds_fused_add_mul_to_f64_cuda",
        ffi.pycapsule(lib.DsFusedAddMulToF64Cuda),
        platform="CUDA",
    )


def block(x):
    x.block_until_ready()
    return x


@jax.jit
def f64_expr(x, y):
    return (x + y) * y


@jax.jit
def f32_expr(x, y):
    return (x + y) * y


@jax.jit
def ds_expr_cuda(x, y):
    out_type = jax.ShapeDtypeStruct(shape=x.shape, dtype=jnp.float64)
    return ffi.ffi_call(
        "ds_fused_add_mul_to_f64_cuda",
        out_type,
        vmap_method="broadcast_all",
    )(x, y)


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


def bytes_f64_expr(n):
    # Reads x and y, writes out. Rough lower-bound traffic estimate.
    return (8 * n) + (8 * n) + (8 * n)


def bytes_f32_expr(n):
    return (4 * n) + (4 * n) + (4 * n)


def bytes_ds_expr(n):
    # DS fused op reads x,y as f64 and writes out as f64.
    # This is only external traffic, not internal register work.
    return (8 * n) + (8 * n) + (8 * n)


def gb_per_s(num_bytes, seconds):
    return num_bytes / seconds / 1e9


def benchmark_one(n, iters):
    rng = np.random.default_rng(0)

    x64 = jnp.array(rng.standard_normal(n), dtype=jnp.float64)
    y64 = jnp.array(rng.standard_normal(n), dtype=jnp.float64)

    x32 = jnp.array(rng.standard_normal(n), dtype=jnp.float32)
    y32 = jnp.array(rng.standard_normal(n), dtype=jnp.float32)

    t_f64 = time_fn(f64_expr, x64, y64, iters=iters)
    t_f32 = time_fn(f32_expr, x32, y32, iters=iters)
    t_ds = time_fn(ds_expr_cuda, x64, y64, iters=iters)

    # Correctness spot-check on a tiny prefix
    ref = np.asarray(f64_expr(x64[:8], y64[:8]))
    ds = np.asarray(ds_expr_cuda(x64[:8], y64[:8]))
    max_abs_err = float(np.max(np.abs(ds - ref)))

    return {
        "N": n,
        "f64_ms": 1e3 * t_f64,
        "f32_ms": 1e3 * t_f32,
        "ds_ms": 1e3 * t_ds,
        "ds_vs_f64": t_ds / t_f64,
        "ds_vs_f32": t_ds / t_f32,
        "f64_GBs": gb_per_s(bytes_f64_expr(n), t_f64),
        "f32_GBs": gb_per_s(bytes_f32_expr(n), t_f32),
        "ds_GBs": gb_per_s(bytes_ds_expr(n), t_ds),
        "max_abs_err_vs_f64": max_abs_err,
    }


def main():
    register_targets()

    print("devices:", jax.devices())
    print("default backend:", jax.default_backend())
    print()

    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    results = []

    for n in sizes:
        if n <= 10_000:
            iters = 200
        elif n <= 100_000:
            iters = 100
        elif n <= 1_000_000:
            iters = 50
        else:
            iters = 20
        results.append(benchmark_one(n, iters))

    header = (
        f"{'N':>12}  "
        f"{'f64 (ms)':>10}  "
        f"{'f32 (ms)':>10}  "
        f"{'ds (ms)':>10}  "
        f"{'ds/f64':>8}  "
        f"{'ds/f32':>8}  "
        f"{'f64 GB/s':>10}  "
        f"{'f32 GB/s':>10}  "
        f"{'ds GB/s':>10}  "
        f"{'max|ds-f64|':>12}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['N']:12d}  "
            f"{r['f64_ms']:10.3f}  "
            f"{r['f32_ms']:10.3f}  "
            f"{r['ds_ms']:10.3f}  "
            f"{r['ds_vs_f64']:8.2f}  "
            f"{r['ds_vs_f32']:8.2f}  "
            f"{r['f64_GBs']:10.2f}  "
            f"{r['f32_GBs']:10.2f}  "
            f"{r['ds_GBs']:10.2f}  "
            f"{r['max_abs_err_vs_f64']:12.3e}"
        )


if __name__ == "__main__":
    main()
