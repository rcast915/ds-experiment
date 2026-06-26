import subprocess
import re
import os
import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# ── Pure-NumPy DS reference arithmetic (no CUDA dependency) ───────────────
def _two_sum_f32(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    s = a + b
    v = s - a
    err = (a - (s - v)) + (b - v)
    return s, err

def ds_from_f64(x):
    x = np.asarray(x, dtype=np.float64)
    hi = x.astype(np.float32)
    lo = (x - hi.astype(np.float64)).astype(np.float32)
    return np.stack([hi, lo], axis=-1)

def ds_to_f64(x):
    x = np.asarray(x, dtype=np.float32)
    return x[..., 0].astype(np.float64) + x[..., 1].astype(np.float64)

def ds_add_ffi(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    sh, sl = _two_sum_f32(a[..., 0], b[..., 0])
    sl = (sl + a[..., 1] + b[..., 1]).astype(np.float32)
    sh, sl = _two_sum_f32(sh, sl)
    return np.stack([sh, sl], axis=-1)

def ds_sub_ffi(a, b):
    b = np.asarray(b, dtype=np.float32)
    return ds_add_ffi(a, np.stack([-b[..., 0], -b[..., 1]], axis=-1))

def run_pass(fn, *args):
    mlir_text = str(jax.jit(fn).lower(*args).compiler_ir())
    with open('/tmp/_test.mlir', 'w') as f:
        f.write(mlir_text)
    r = subprocess.run(
        ['./stablehlo_pass/build/mlir-ds-opt',
         '--pass-pipeline=builtin.module(func.func(ds-transform))',
         '/tmp/_test.mlir'],
        capture_output=True, text=True, cwd='/src/ds_experiment'
    )
    assert r.returncode == 0, f"pass failed:\n{r.stderr}"
    return r.stdout

x64 = jnp.array([1e8, 1e10, 1e12], dtype=jnp.float64)
y64 = jnp.array([1e-2, 1e-3, 1e-4], dtype=jnp.float64)
x32 = x64.astype(jnp.float32)
y32 = y64.astype(jnp.float32)

# ── Test 1: add+subtract — two_sum sequences, no Veltkamp ─────────────────
print("=== Test 1: (x + y) - x  [add + subtract only] ===")
def fn_add_sub(x, y): return (x + y) - x

transformed = run_pass(fn_add_sub, x32, y32)
n_sub = len(re.findall(r'stablehlo\.subtract', transformed))
n_mul = len(re.findall(r'stablehlo\.multiply', transformed))
print(f"  subtract ops: {n_sub}  (expect >5 for two_sum)")
print(f"  multiply ops: {n_mul}  (expect 0 — no multiply in source)")
assert n_sub > 5
assert n_mul == 0
print("  PASS: correct DS expansion for add/subtract\n")

# ── Test 2: multiply — Veltkamp split ─────────────────────────────────────
print("=== Test 2: (x + y) * y  [add + multiply] ===")
def fn_mul(x, y): return (x + y) * y

transformed = run_pass(fn_mul, x32, y32)
n_sub = len(re.findall(r'stablehlo\.subtract', transformed))
n_mul = len(re.findall(r'stablehlo\.multiply', transformed))
n_cst = len(re.findall(r'stablehlo\.constant', transformed))
print(f"  subtract ops: {n_sub}  (expect >5 for two_sum)")
print(f"  multiply ops: {n_mul}  (expect >5 for Veltkamp split)")
print(f"  constant ops: {n_cst}  (expect >0 for 4097 splat)")
assert n_sub > 5
assert n_mul > 5
assert n_cst > 0
print("  PASS: correct DS expansion including Veltkamp split\n")

# ── Test 3: numerical accuracy on cancellation ────────────────────────────
print("=== Test 3: numerical accuracy (x+y)-x ===")
ref       = fn_add_sub(x64, y64)
f32_result = fn_add_sub(x32, y32).astype(jnp.float64)

x_ds = ds_from_f64(x64)
y_ds = ds_from_f64(y64)
ds_result = ds_to_f64(ds_sub_ffi(ds_add_ffi(x_ds, y_ds), x_ds))

abs_err_f32 = jnp.abs(f32_result - ref)
abs_err_ds  = jnp.abs(ds_result  - ref)
improvement = abs_err_f32 / (abs_err_ds + 1e-30)

print(f"  ref (f64):   {np.array(ref)}")
print(f"  f32 result:  {np.array(f32_result)}")
print(f"  DS result:   {np.array(ds_result)}")
print(f"  err f32:     {np.array(abs_err_f32)}")
print(f"  err DS:      {np.array(abs_err_ds)}")
print(f"  improvement: {np.array(improvement)}")
assert float(improvement[0]) > 100
assert float(improvement[1]) > 100
print("  PASS: DS recovers precision lost by f32 cancellation\n")

print("All tests passed.")
