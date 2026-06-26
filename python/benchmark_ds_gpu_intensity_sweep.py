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
def ds_step(x, y):
    out_type = jax.ShapeDtypeStruct(shape=x.shape, dtype=jnp.float64)
    return ffi.ffi_call(
        "ds_fused_add_mul_to_f64_cuda",
        out_type,
        vmap_method="broadcast_all",
    )(x, y)


def make_f64_kernel(k):
    @jax.jit
    def kernel(x, y):
        z = x
        for _ in range(k):
            z = (z + y) * y
        return z
    return kernel


def make_f32_kernel(k):
    @jax.jit
    def kernel(x, y):
        z = x
        for _ in range(k):
            z = (z + y) * y
        return z
    return kernel


def make_ds_kernel(k):
    @jax.jit
    def kernel(x, y):
        z = x
        for _ in range(k):
            z = ds_step(z, y)
        return z
    return kernel


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


def benchmark_one(n, k, iters):
    rng = np.random.default_rng(0)

    # Keep values bounded so repeated application stays stable.
    x64 = jnp.array(rng.uniform(0.1, 1.0, size=n), dtype=jnp.float64)
    y64 = jnp.array(rng.uniform(0.1, 0.5, size=n), dtype=jnp.float64)

    x32 = x64.astype(jnp.float32)
    y32 = y64.astype(jnp.float32)

    f64_kernel = make_f64_kernel(k)
    f32_kernel = make_f32_kernel(k)
    ds_kernel = make_ds_kernel(k)

    t_f64 = time_fn(f64_kernel, x64, y64, iters=iters)
    t_f32 = time_fn(f32_kernel, x32, y32, iters=iters)
    t_ds = time_fn(ds_kernel, x64, y64, iters=iters)

    # Spot-check accuracy on a small prefix against f64.
    ref = np.asarray(block(f64_kernel(x64[:16], y64[:16])))
    out_ds = np.asarray(block(ds_kernel(x64[:16], y64[:16])))
    out_f32 = np.asarray(block(f32_kernel(x32[:16], y32[:16])).astype(jnp.float64))

    max_abs_err_ds = float(np.max(np.abs(out_ds - ref)))
    max_abs_err_f32 = float(np.max(np.abs(out_f32 - ref)))

    return {
        "N": n,
        "K": k,
        "f64_ms": 1e3 * t_f64,
        "f32_ms": 1e3 * t_f32,
        "ds_ms": 1e3 * t_ds,
        "ds_vs_f64": t_ds / t_f64,
        "ds_vs_f32": t_ds / t_f32,
        "max_abs_err_ds": max_abs_err_ds,
        "max_abs_err_f32": max_abs_err_f32,
    }


def main():
    register_targets()

    print("devices:", jax.devices())
    print("default backend:", jax.default_backend())
    print()

    sizes = [1_000_000, 10_000_000]
    ks = [1, 2, 4, 8, 16, 32, 64]

    results = []
    for n in sizes:
        for k in ks:
            if n <= 1_000_000:
                iters = 50
            else:
                iters = 20
            results.append(benchmark_one(n, k, iters))

    header = (
        f"{'N':>12}  "
        f"{'K':>4}  "
        f"{'f64 (ms)':>10}  "
        f"{'f32 (ms)':>10}  "
        f"{'ds (ms)':>10}  "
        f"{'ds/f64':>8}  "
        f"{'ds/f32':>8}  "
        f"{'max|ds-f64|':>12}  "
        f"{'max|f32-f64|':>12}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['N']:12d}  "
            f"{r['K']:4d}  "
            f"{r['f64_ms']:10.3f}  "
            f"{r['f32_ms']:10.3f}  "
            f"{r['ds_ms']:10.3f}  "
            f"{r['ds_vs_f64']:8.2f}  "
            f"{r['ds_vs_f32']:8.2f}  "
            f"{r['max_abs_err_ds']:12.3e}  "
            f"{r['max_abs_err_f32']:12.3e}"
        )


if __name__ == "__main__":
    main()
