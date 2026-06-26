import os
import ctypes
import time
import numpy as np

os.environ["JAX_ENABLE_X64"] = "1"
# Do not force CPU.

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


def block_until_ready(x):
    x.block_until_ready()
    return x


@jax.jit
def f64_expr(x, y):
    return (x + y) * y


@jax.jit
def f32_expr(x, y):
    x32 = x.astype(jnp.float32)
    y32 = y.astype(jnp.float32)
    return ((x32 + y32) * y32).astype(jnp.float64)


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
        block_until_ready(out)

    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
        block_until_ready(out)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def benchmark_one(n, iters):
    rng = np.random.default_rng(0)
    x = jnp.array(rng.standard_normal(n), dtype=jnp.float64)
    y = jnp.array(rng.standard_normal(n), dtype=jnp.float64)

    t_f64 = time_fn(f64_expr, x, y, iters=iters)
    t_f32 = time_fn(f32_expr, x, y, iters=iters)
    t_ds = time_fn(ds_expr_cuda, x, y, iters=iters)

    return {
        "N": n,
        "f64_ms": 1e3 * t_f64,
        "f32_ms": 1e3 * t_f32,
        "ds_ms": 1e3 * t_ds,
        "ds_vs_f64": t_ds / t_f64,
        "ds_vs_f32": t_ds / t_f32,
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
        f"{'f64 (ms)':>12}  "
        f"{'f32 (ms)':>12}  "
        f"{'ds gpu (ms)':>14}  "
        f"{'ds/f64':>10}  "
        f"{'ds/f32':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['N']:12d}  "
            f"{r['f64_ms']:12.3f}  "
            f"{r['f32_ms']:12.3f}  "
            f"{r['ds_ms']:14.3f}  "
            f"{r['ds_vs_f64']:10.2f}  "
            f"{r['ds_vs_f32']:10.2f}"
        )


if __name__ == "__main__":
    main()
