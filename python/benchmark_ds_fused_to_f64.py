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
    ffi.register_ffi_target("ds_to_f64", ffi.pycapsule(lib.DsToF64), platform="cpu")
    ffi.register_ffi_target("ds_add", ffi.pycapsule(lib.DsAdd), platform="cpu")
    ffi.register_ffi_target("ds_mul", ffi.pycapsule(lib.DsMul), platform="cpu")
    ffi.register_ffi_target("ds_fused_add_mul", ffi.pycapsule(lib.DsFusedAddMul), platform="cpu")
    ffi.register_ffi_target("ds_fused_add_mul_to_f64", ffi.pycapsule(lib.DsFusedAddMulToF64), platform="cpu")


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


def ds_fused_add_mul_raw(x, y):
    out_type = jax.ShapeDtypeStruct(shape=x.shape + (2,), dtype=jnp.float32)
    return ffi.ffi_call("ds_fused_add_mul", out_type, vmap_method="broadcast_all")(x, y)


def ds_fused_add_mul_to_f64_raw(x, y):
    out_type = jax.ShapeDtypeStruct(shape=x.shape, dtype=jnp.float64)
    return ffi.ffi_call("ds_fused_add_mul_to_f64", out_type, vmap_method="broadcast_all")(x, y)


@jax.jit
def f64_expr(x, y):
    return (x + y) * y


@jax.jit
def ds_expr_unfused(x, y):
    x_ds = ds_from_f64(x)
    y_ds = ds_from_f64(y)
    out_ds = ds_mul(ds_add(x_ds, y_ds), y_ds)
    return ds_to_f64(out_ds)


@jax.jit
def ds_expr_fused_dsout(x, y):
    return ds_to_f64(ds_fused_add_mul_raw(x, y))


@jax.jit
def ds_expr_fused_f64out(x, y):
    return ds_fused_add_mul_to_f64_raw(x, y)


def time_fn(fn, *args, warmup=10, iters=50):
    for _ in range(warmup):
        out = fn(*args)
        np.asarray(out)

    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
        np.asarray(out)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def benchmark_one(n, iters):
    rng = np.random.default_rng(0)
    x = jnp.array(rng.standard_normal(n), dtype=jnp.float64)
    y = jnp.array(rng.standard_normal(n), dtype=jnp.float64)

    t_f64 = time_fn(f64_expr, x, y, iters=iters)
    t_unfused = time_fn(ds_expr_unfused, x, y, iters=iters)
    t_fused_dsout = time_fn(ds_expr_fused_dsout, x, y, iters=iters)
    t_fused_f64out = time_fn(ds_expr_fused_f64out, x, y, iters=iters)

    return {
        "N": n,
        "f64_ms": 1e3 * t_f64,
        "unfused_ms": 1e3 * t_unfused,
        "fused_dsout_ms": 1e3 * t_fused_dsout,
        "fused_f64out_ms": 1e3 * t_fused_f64out,
        "unfused_vs_f64": t_unfused / t_f64,
        "fused_dsout_vs_f64": t_fused_dsout / t_f64,
        "fused_f64out_vs_f64": t_fused_f64out / t_f64,
        "f64out_speedup_over_unfused": t_unfused / t_fused_f64out,
    }


def main():
    register_targets()

    print("devices:", jax.devices())
    print("default backend:", jax.default_backend())
    print()

    sizes = [1_000, 10_000, 100_000, 1_000_000]
    results = []

    for n in sizes:
        if n <= 10_000:
            iters = 200
        elif n <= 100_000:
            iters = 100
        else:
            iters = 30
        results.append(benchmark_one(n, iters))

    header = (
        f"{'N':>10}  "
        f"{'f64 (ms)':>10}  "
        f"{'unfused':>10}  "
        f"{'fusedDS':>10}  "
        f"{'fusedF64':>10}  "
        f"{'u/f64':>8}  "
        f"{'fds/f64':>8}  "
        f"{'ff64/f64':>9}  "
        f"{'u/ff64':>8}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['N']:10d}  "
            f"{r['f64_ms']:10.3f}  "
            f"{r['unfused_ms']:10.3f}  "
            f"{r['fused_dsout_ms']:10.3f}  "
            f"{r['fused_f64out_ms']:10.3f}  "
            f"{r['unfused_vs_f64']:8.2f}  "
            f"{r['fused_dsout_vs_f64']:8.2f}  "
            f"{r['fused_f64out_vs_f64']:9.2f}  "
            f"{r['f64out_speedup_over_unfused']:8.2f}"
        )


if __name__ == "__main__":
    main()
