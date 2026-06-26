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


def ds_mul(a, b):
    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    out_type = jax.ShapeDtypeStruct(shape=a.shape, dtype=jnp.float32)
    return ffi.ffi_call("ds_mul", out_type, vmap_method="broadcast_all")(a, b)


# Native baselines
@jax.jit
def native_from_f64_baseline(x):
    hi = x.astype(jnp.float32)
    lo = (x - hi.astype(jnp.float64)).astype(jnp.float32)
    return jnp.stack([hi, lo], axis=-1)


@jax.jit
def native_to_f64_baseline(x):
    return x[..., 0].astype(jnp.float64) + x[..., 1].astype(jnp.float64)


@jax.jit
def native_add_f32x2_baseline(a, b):
    return a + b


@jax.jit
def native_sub_f32x2_baseline(a, b):
    return a - b


@jax.jit
def native_mul_f32x2_baseline(a, b):
    return a * b


@jax.jit
def op_ds_from_f64(x):
    return ds_from_f64(x)


@jax.jit
def op_ds_to_f64(x_ds):
    return ds_to_f64(x_ds)


@jax.jit
def op_ds_add(a_ds, b_ds):
    return ds_add(a_ds, b_ds)


@jax.jit
def op_ds_sub(a_ds, b_ds):
    return ds_sub(a_ds, b_ds)


@jax.jit
def op_ds_mul(a_ds, b_ds):
    return ds_mul(a_ds, b_ds)


def time_fn(fn, *args, warmup=10, iters=50):
    for _ in range(warmup):
        out = fn(*args)
        np.asarray(out)

    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
        np.asarray(out)
    t1 = time.perf_counter()

    total = t1 - t0
    return total / iters


def benchmark_size(n, warmup=10, iters=50):
    rng = np.random.default_rng(0)

    x = jnp.array(rng.standard_normal(n), dtype=jnp.float64)
    y = jnp.array(rng.standard_normal(n), dtype=jnp.float64)

    # Precompute DS inputs once so add/sub/mul timing excludes conversion.
    x_ds = op_ds_from_f64(x)
    y_ds = op_ds_from_f64(y)

    # Also prepare native packed representation for rough baseline comparison.
    x_native_ds = native_from_f64_baseline(x)
    y_native_ds = native_from_f64_baseline(y)

    # Timings
    t_from = time_fn(op_ds_from_f64, x, warmup=warmup, iters=iters)
    t_to = time_fn(op_ds_to_f64, x_ds, warmup=warmup, iters=iters)
    t_add = time_fn(op_ds_add, x_ds, y_ds, warmup=warmup, iters=iters)
    t_sub = time_fn(op_ds_sub, x_ds, y_ds, warmup=warmup, iters=iters)
    t_mul = time_fn(op_ds_mul, x_ds, y_ds, warmup=warmup, iters=iters)

    # Native baselines
    t_from_native = time_fn(native_from_f64_baseline, x, warmup=warmup, iters=iters)
    t_to_native = time_fn(native_to_f64_baseline, x_native_ds, warmup=warmup, iters=iters)
    t_add_native = time_fn(native_add_f32x2_baseline, x_native_ds, y_native_ds, warmup=warmup, iters=iters)
    t_sub_native = time_fn(native_sub_f32x2_baseline, x_native_ds, y_native_ds, warmup=warmup, iters=iters)
    t_mul_native = time_fn(native_mul_f32x2_baseline, x_native_ds, y_native_ds, warmup=warmup, iters=iters)

    return {
        "N": n,

        "from_ms": 1e3 * t_from,
        "to_ms": 1e3 * t_to,
        "add_ms": 1e3 * t_add,
        "sub_ms": 1e3 * t_sub,
        "mul_ms": 1e3 * t_mul,

        "from_native_ms": 1e3 * t_from_native,
        "to_native_ms": 1e3 * t_to_native,
        "add_native_ms": 1e3 * t_add_native,
        "sub_native_ms": 1e3 * t_sub_native,
        "mul_native_ms": 1e3 * t_mul_native,

        "from_over_native": t_from / t_from_native if t_from_native > 0 else np.nan,
        "to_over_native": t_to / t_to_native if t_to_native > 0 else np.nan,
        "add_over_native": t_add / t_add_native if t_add_native > 0 else np.nan,
        "sub_over_native": t_sub / t_sub_native if t_sub_native > 0 else np.nan,
        "mul_over_native": t_mul / t_mul_native if t_mul_native > 0 else np.nan,
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
        results.append(benchmark_size(n, warmup=10, iters=iters))

    header1 = (
        f"{'N':>10}  "
        f"{'from':>10}  "
        f"{'to':>10}  "
        f"{'add':>10}  "
        f"{'sub':>10}  "
        f"{'mul':>10}"
    )
    print("Per-op DS times (ms)")
    print(header1)
    print("-" * len(header1))
    for r in results:
        print(
            f"{r['N']:10d}  "
            f"{r['from_ms']:10.3f}  "
            f"{r['to_ms']:10.3f}  "
            f"{r['add_ms']:10.3f}  "
            f"{r['sub_ms']:10.3f}  "
            f"{r['mul_ms']:10.3f}"
        )

    print()
    header2 = (
        f"{'N':>10}  "
        f"{'from/native':>12}  "
        f"{'to/native':>10}  "
        f"{'add/native':>11}  "
        f"{'sub/native':>11}  "
        f"{'mul/native':>11}"
    )
    print("Slowdown over rough native packed-array baselines")
    print(header2)
    print("-" * len(header2))
    for r in results:
        print(
            f"{r['N']:10d}  "
            f"{r['from_over_native']:12.2f}  "
            f"{r['to_over_native']:10.2f}  "
            f"{r['add_over_native']:11.2f}  "
            f"{r['sub_over_native']:11.2f}  "
            f"{r['mul_over_native']:11.2f}"
        )

    print()
    header3 = (
        f"{'N':>10}  "
        f"{'from_native':>12}  "
        f"{'to_native':>10}  "
        f"{'add_native':>11}  "
        f"{'sub_native':>11}  "
        f"{'mul_native':>11}"
    )
    print("Native baseline times (ms)")
    print(header3)
    print("-" * len(header3))
    for r in results:
        print(
            f"{r['N']:10d}  "
            f"{r['from_native_ms']:12.3f}  "
            f"{r['to_native_ms']:10.3f}  "
            f"{r['add_native_ms']:11.3f}  "
            f"{r['sub_native_ms']:11.3f}  "
            f"{r['mul_native_ms']:11.3f}"
        )


if __name__ == "__main__":
    main()
