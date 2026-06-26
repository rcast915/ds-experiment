import os
import ctypes
import time
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
    out_type = jax.ShapeDtypeStruct(shape=x.shape + (2,), dtype=jnp.float32)
    return ffi.ffi_call("ds_from_f64", out_type, vmap_method="broadcast_all")(x)


def ds_to_f64(x):
    out_type = jax.ShapeDtypeStruct(shape=x.shape[:-1], dtype=jnp.float64)
    return ffi.ffi_call("ds_to_f64", out_type, vmap_method="broadcast_all")(x)


def ds_add(a, b):
    out_type = jax.ShapeDtypeStruct(shape=a.shape, dtype=jnp.float32)
    return ffi.ffi_call("ds_add", out_type, vmap_method="broadcast_all")(a, b)


def ds_mul(a, b):
    out_type = jax.ShapeDtypeStruct(shape=a.shape, dtype=jnp.float32)
    return ffi.ffi_call("ds_mul", out_type, vmap_method="broadcast_all")(a, b)


@jax.jit
def f64_expr(x, y):
    return (x + y) * y


@jax.jit
def f32_expr(x, y):
    x32 = x.astype(jnp.float32)
    y32 = y.astype(jnp.float32)
    return ((x32 + y32) * y32).astype(jnp.float64)


@jax.jit
def ds_roundtrip(x):
    return ds_to_f64(ds_from_f64(x))


@jax.jit
def ds_expr(x, y):
    x_ds = ds_from_f64(x)
    y_ds = ds_from_f64(y)
    out_ds = ds_mul(ds_add(x_ds, y_ds), y_ds)
    return ds_to_f64(out_ds)


def time_fn(fn, *args, warmup=5, iters=50):
    # Warmup
    for _ in range(warmup):
        out = fn(*args)
        np.asarray(out)

    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
        np.asarray(out)
    end = time.perf_counter()

    total = end - start
    return total, total / iters


def benchmark_size(n, warmup=5, iters=50):
    rng = np.random.default_rng(0)

    x = jnp.array(rng.standard_normal(n), dtype=jnp.float64)
    y = jnp.array(rng.standard_normal(n), dtype=jnp.float64)

    t_f64_total, t_f64 = time_fn(f64_expr, x, y, warmup=warmup, iters=iters)
    t_f32_total, t_f32 = time_fn(f32_expr, x, y, warmup=warmup, iters=iters)
    t_rt_total, t_rt = time_fn(ds_roundtrip, x, warmup=warmup, iters=iters)
    t_ds_total, t_ds = time_fn(ds_expr, x, y, warmup=warmup, iters=iters)

    return {
        "N": n,
        "f64_sec": t_f64,
        "f32_sec": t_f32,
        "ds_roundtrip_sec": t_rt,
        "ds_expr_sec": t_ds,
        "ds_vs_f64": t_ds / t_f64 if t_f64 > 0 else np.nan,
        "ds_vs_f32": t_ds / t_f32 if t_f32 > 0 else np.nan,
        "roundtrip_vs_f64": t_rt / t_f64 if t_f64 > 0 else np.nan,
    }


def main():
    register_targets()

    print("devices:", jax.devices())
    print("default backend:", jax.default_backend())
    print()

    sizes = [1_000, 10_000, 100_000, 1_000_000]

    results = []
    for n in sizes:
        # fewer iterations for large sizes
        if n <= 10_000:
            iters = 200
        elif n <= 100_000:
            iters = 100
        else:
            iters = 30

        r = benchmark_size(n, warmup=10, iters=iters)
        results.append(r)

    header = (
        f"{'N':>10}  "
        f"{'f64 (ms)':>12}  "
        f"{'f32 (ms)':>12}  "
        f"{'ds rt (ms)':>12}  "
        f"{'ds expr (ms)':>14}  "
        f"{'ds/f64':>10}  "
        f"{'ds/f32':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['N']:10d}  "
            f"{1e3*r['f64_sec']:12.3f}  "
            f"{1e3*r['f32_sec']:12.3f}  "
            f"{1e3*r['ds_roundtrip_sec']:12.3f}  "
            f"{1e3*r['ds_expr_sec']:14.3f}  "
            f"{r['ds_vs_f64']:10.2f}  "
            f"{r['ds_vs_f32']:10.2f}"
        )


if __name__ == "__main__":
    main()
