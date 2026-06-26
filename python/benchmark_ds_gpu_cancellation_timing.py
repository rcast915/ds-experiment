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


def bytes_f64(n):
    return 8 * n + 8 * n + 8 * n


def bytes_f32(n):
    return 4 * n + 4 * n + 4 * n


def bytes_ds(n):
    return 8 * n + 8 * n + 8 * n


def gb_per_s(num_bytes, seconds):
    return num_bytes / seconds / 1e9


def benchmark_case(n, x_scalar, y_scalar, iters):
    x64 = jnp.full((n,), x_scalar, dtype=jnp.float64)
    y64 = jnp.full((n,), y_scalar, dtype=jnp.float64)

    x32 = jnp.full((n,), x_scalar, dtype=jnp.float32)
    y32 = jnp.full((n,), y_scalar, dtype=jnp.float32)

    t_f64 = time_fn(f64_cancel, x64, y64, iters=iters)
    t_f32 = time_fn(f32_cancel, x32, y32, iters=iters)
    t_ds = time_fn(ds_cancel_cuda, x64, y64, iters=iters)

    # Accuracy check on one element
    truth = float(y_scalar)
    out_f64 = float(np.asarray(block(f64_cancel(x64[:1], y64[:1])))[0])
    out_f32 = float(np.asarray(block(f32_cancel(x32[:1], y32[:1])))[0])
    out_ds = float(np.asarray(block(ds_cancel_cuda(x64[:1], y64[:1])))[0])

    abs_err_f64 = abs(out_f64 - truth)
    abs_err_f32 = abs(out_f32 - truth)
    abs_err_ds = abs(out_ds - truth)

    rel_err_f64 = abs_err_f64 / abs(truth) if truth != 0 else np.nan
    rel_err_f32 = abs_err_f32 / abs(truth) if truth != 0 else np.nan
    rel_err_ds = abs_err_ds / abs(truth) if truth != 0 else np.nan

    return {
        "N": n,
        "x": x_scalar,
        "y": y_scalar,
        "f64_ms": 1e3 * t_f64,
        "f32_ms": 1e3 * t_f32,
        "ds_ms": 1e3 * t_ds,
        "ds_vs_f64": t_ds / t_f64,
        "ds_vs_f32": t_ds / t_f32,
        "f64_GBs": gb_per_s(bytes_f64(n), t_f64),
        "f32_GBs": gb_per_s(bytes_f32(n), t_f32),
        "ds_GBs": gb_per_s(bytes_ds(n), t_ds),
        "rel_err_f64": rel_err_f64,
        "rel_err_f32": rel_err_f32,
        "rel_err_ds": rel_err_ds,
        "out_f64": out_f64,
        "out_f32": out_f32,
        "out_ds": out_ds,
    }


def main():
    register_targets()

    print("devices:", jax.devices())
    print("default backend:", jax.default_backend())
    print()

    # A few representative cancellation cases:
    cases = [
        (1_000_000, 1e8,  1e-4),
        (1_000_000, 1e10, 1e-6),
        (1_000_000, 1e10, 1e-8),
        (1_000_000, 1e12, 1e-2),
        (10_000_000, 1e8,  1e-4),
        (10_000_000, 1e10, 1e-8),
    ]

    results = []
    for n, x_scalar, y_scalar in cases:
        if n <= 1_000_000:
            iters = 50
        else:
            iters = 20
        results.append(benchmark_case(n, x_scalar, y_scalar, iters))

    header = (
        f"{'N':>10}  "
        f"{'x':>10}  "
        f"{'y':>10}  "
        f"{'f64 ms':>9}  "
        f"{'f32 ms':>9}  "
        f"{'ds ms':>9}  "
        f"{'ds/f64':>8}  "
        f"{'ds/f32':>8}  "
        f"{'rel_f64':>10}  "
        f"{'rel_f32':>10}  "
        f"{'rel_ds':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['N']:10d}  "
            f"{r['x']:10.0e}  "
            f"{r['y']:10.0e}  "
            f"{r['f64_ms']:9.3f}  "
            f"{r['f32_ms']:9.3f}  "
            f"{r['ds_ms']:9.3f}  "
            f"{r['ds_vs_f64']:8.2f}  "
            f"{r['ds_vs_f32']:8.2f}  "
            f"{r['rel_err_f64']:10.3e}  "
            f"{r['rel_err_f32']:10.3e}  "
            f"{r['rel_err_ds']:10.3e}"
        )

    print()
    print("Detailed outputs:")
    for r in results:
        print(
            f"N={r['N']}, x={r['x']:.0e}, y={r['y']:.0e} | "
            f"f64={r['out_f64']:.6e}, f32={r['out_f32']:.6e}, ds={r['out_ds']:.6e}"
        )

    print()
    print("Effective bandwidth estimates (GB/s):")
    header2 = (
        f"{'N':>10}  {'x':>10}  {'y':>10}  "
        f"{'f64 GB/s':>10}  {'f32 GB/s':>10}  {'ds GB/s':>10}"
    )
    print(header2)
    print("-" * len(header2))
    for r in results:
        print(
            f"{r['N']:10d}  "
            f"{r['x']:10.0e}  "
            f"{r['y']:10.0e}  "
            f"{r['f64_GBs']:10.2f}  "
            f"{r['f32_GBs']:10.2f}  "
            f"{r['ds_GBs']:10.2f}"
        )


if __name__ == "__main__":
    main()
