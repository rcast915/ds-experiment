"""
Feasibility spike: cuBLAS FP64 emulation from inside a jitted JAX program,
through our own FFI call (cpp/emu_gemm/emu_dgemm_ffi.cc). No DS pass.

Checks, at each n:
  speed     emulated FFI DGEMM vs XLA's own f64 matmul and vs native FFI DGEMM;
            the standalone benchmark gave 5-8x over native DGEMM at 2048-4096
  accuracy  ||C - A B|| / ||A B|| against NumPy f64, for every path
  layout    the native FFI DGEMM must agree with XLA's matmul to f64 rounding
  scope     in one jitted program computing both XLA's matmul and the emulated
            call, XLA's output must be bitwise identical to XLA alone, i.e.
            emulation is confined to our handle
  memory    emulation with cuBLAS's own workspace and with 1 GiB of workspace
            from XLA's scratch allocator (run once with XLA preallocation on,
            once off; the host script does both)

    python3 tests/spike_emu_ffi.py <libemu_dgemm_ffi.so> [n ...]
"""

import ctypes
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

LIB = sys.argv[1]
SIZES = [int(a) for a in sys.argv[2:]] or [1024, 2048, 4096]
REPS = 10
WS_MB = 1024

lib = ctypes.cdll.LoadLibrary(LIB)
jax.ffi.register_ffi_target("emu_dgemm", jax.ffi.pycapsule(lib.EmuDgemm), platform="CUDA")


def ffi_gemm(a, b, emulate, ws_mb=0):
    out = jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), jnp.float64)
    return jax.ffi.ffi_call("emu_dgemm", out)(a, b, emulate=np.int64(emulate),
                                              ws_mb=np.int64(ws_mb))


def xla_mm(a, b):
    return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def bench(fn, *args):
    f = jax.jit(fn)
    out = f(*args)
    jax.block_until_ready(out)
    for _ in range(2):
        jax.block_until_ready(f(*args))
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*args))
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts)), jax.tree.map(np.asarray, out)


def main():
    with open("/proc/self/maps") as f:
        libs = sorted({l.split()[-1] for l in f if "libcublas" in l})
    print("JAX       :", jax.__version__, jax.devices())
    print("Prealloc  :", os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "(default: on)"))
    print("cuBLAS    :", *libs)
    print()

    ok_all = True
    for n in SIZES:
        rng = np.random.default_rng(n)
        a_np, b_np = rng.standard_normal((n, n)), rng.standard_normal((n, n))
        ref = a_np @ b_np
        nref = np.linalg.norm(ref)
        err = lambda c: float(np.linalg.norm(c - ref) / nref)
        a, b = jax.device_put(a_np), jax.device_put(b_np)

        rows = {}
        for name, fn in [
            ("XLA f64 matmul",          xla_mm),
            ("FFI DGEMM native",        lambda x, y: ffi_gemm(x, y, 0)),
            ("FFI DGEMM emulated",      lambda x, y: ffi_gemm(x, y, 1)),
            (f"  + {WS_MB} MiB XLA workspace", lambda x, y: ffi_gemm(x, y, 1, WS_MB)),
        ]:
            try:
                rows[name] = bench(fn, a, b)
            except Exception as e:                       # report and keep going
                print(f"  {name}: FAILED: {str(e).splitlines()[0][:200]}")
                ok_all = False
        print(f"n = {n}")
        base = rows.get("XLA f64 matmul", (None,))[0]
        nat = rows.get("FFI DGEMM native", (None,))[0]
        for name, (ms, c) in rows.items():
            extra = ""
            if base and name != "XLA f64 matmul":
                extra += f"   vs XLA {base / ms:5.2f}x"
            if nat and ("emulated" in name or "workspace" in name):
                extra += f"   vs native DGEMM {nat / ms:5.2f}x"
            print(f"  {name:28s} {ms:9.3f} ms   err {err(c):.2e}{extra}")

        # scope: XLA's matmul next to an emulated call in the same program
        try:
            _, (c_x, c_e) = bench(lambda x, y: (xla_mm(x, y), ffi_gemm(x, y, 1)), a, b)
            same = np.array_equal(c_x, rows["XLA f64 matmul"][1])
            print(f"  scope: XLA matmul beside emulated call is "
                  f"{'bitwise identical to XLA alone (OK)' if same else 'CHANGED (emulation leaked)'}")
            ok_all &= same
        except Exception as e:
            print(f"  scope check FAILED: {str(e).splitlines()[0][:200]}")
            ok_all = False

        if "FFI DGEMM native" in rows and "XLA f64 matmul" in rows:
            lay_ok = err(rows["FFI DGEMM native"][1]) < 1e-13
            print(f"  layout: native FFI DGEMM error {'at f64 level (OK)' if lay_ok else 'WRONG'}")
            ok_all &= lay_ok
        if "FFI DGEMM emulated" in rows and nat:
            sp = nat / rows["FFI DGEMM emulated"][0]
            print(f"  emulation {'TAKEN' if sp >= 1.5 else 'NOT taken'} ({sp:.2f}x over native DGEMM)")
        print()

    print("SPIKE:", "checks passed; see speedups above" if ok_all else "a check FAILED; see above")


if __name__ == "__main__":
    main()
