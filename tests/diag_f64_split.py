"""
Diagnostic: is an f64 argument's lo word reaching the GPU?

tests/diag_f64_ops.py showed every op on f64 inputs at f32-level accuracy,
including `x + 0.0`, which points at the f64 -> (hi, lo) split at function
entry (emitFromFloat) rather than at any one op. This script looks at that
split directly, three ways:

  1. Values. With DS_RETURN_PAIRS=1, `return x, x` hands back the raw
     (hi, lo) pair. Expected: hi == f32(x) and lo == x - f32(x).
  2. The pass's own output for that function (mlir-ds-opt, text form).
  3. XLA's HLO for it before and after optimization, to see what the
     backend did with the convert / optimization_barrier / subtract chain.

Each configuration runs in its own subprocess.

Usage (inside container, after ds_setup.sh):
  PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \\
      python3 tests/diag_f64_split.py
"""

import sys
import os
import glob
import shutil
import subprocess
import tempfile
from pathlib import Path

TESTS_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
PLUGIN_SO    = PROJECT_ROOT / "pjrt_plugin" / "build" / "libds_pjrt_plugin.so"
OPT_BINARY   = PROJECT_ROOT / "stablehlo_pass" / "build" / "mlir-ds-opt"

_VALUES_CODE = """
import os
import numpy as np
os.environ["JAX_ENABLE_X64"] = "1"
import jax
import jax.numpy as jnp

x = np.random.default_rng(7).uniform(0.5, 3.0, 4096)
want_hi = x.astype(np.float32).astype(np.float64)
want_lo = x - want_hi

def report(label, hi, lo):
    hi = np.asarray(hi, np.float64); lo = np.asarray(lo, np.float64)
    print(f"  {label}")
    print(f"    hi == f32(x)        : {int(np.sum(hi == want_hi))} / {x.size}")
    print(f"    lo == x - f32(x)    : {int(np.sum(lo == want_lo))} / {x.size}")
    print(f"    lo == 0             : {int(np.sum(lo == 0.0))} / {x.size}")
    print(f"    max |lo|            : {np.max(np.abs(lo)):.3e}   (expected {np.max(np.abs(want_lo)):.3e})")
    print(f"    max |hi + lo - x|/x : {np.max(np.abs(hi + lo - x) / x):.3e}")
    print(f"    first 3 (x, hi, lo, want_lo):")
    for i in range(3):
        print(f"      {x[i]!r}  {hi[i]!r}  {lo[i]!r}  {want_lo[i]!r}")

@jax.jit
def ds_split_probe(a):
    return a, a

@jax.jit
def ds_split_probe_add0(a):
    r = a + 0.0
    return r, r

report("return x, x", *ds_split_probe(jnp.array(x)))
report("r = x + 0.0; return r, r", *ds_split_probe_add0(jnp.array(x)))
"""

_LOWER_CODE = """
import os
os.environ["JAX_ENABLE_X64"] = "1"
import jax
import jax.numpy as jnp
print(jax.jit(lambda a: a + 0.0).lower(jax.ShapeDtypeStruct((4,), jnp.float64)).compiler_ir())
"""


def run(code: str, extra_env: dict) -> str:
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"
    env.pop("DS_BYPASS", None)
    for key, value in extra_env.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return f"  [subprocess failed]\n{r.stderr[-2000:]}"
    return r.stdout


def section(title: str):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


if __name__ == "__main__":
    if not os.environ.get("PJRT_NAMES_AND_LIBRARY_PATHS") or not PLUGIN_SO.exists():
        print("Needs the PJRT plugin (set PJRT_NAMES_AND_LIBRARY_PATHS).")
        sys.exit(1)

    # ── 1. Values ────────────────────────────────────────────────────────────
    section("1. Raw (hi, lo) of an f64 argument  [DS_RETURN_PAIRS=1]")
    print(run(_VALUES_CODE, {"DS_RETURN_PAIRS": "1"}))

    section("1b. Same, XLA algebraic simplifier disabled")
    print(run(_VALUES_CODE, {"DS_RETURN_PAIRS": "1",
                             "XLA_FLAGS": "--xla_disable_hlo_passes=algsimp"}))

    # ── 2. The pass's own output ─────────────────────────────────────────────
    section("2. mlir-ds-opt output for  lambda a: a + 0.0  (tensor<4xf64>)")
    mlir = run(_LOWER_CODE, {"JAX_PLATFORMS": "cpu", "PJRT_NAMES_AND_LIBRARY_PATHS": None})
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(mlir)
    r = subprocess.run(
        [str(OPT_BINARY),
         "--pass-pipeline=builtin.module(inline,func.func(ds-transform))", f.name],
        capture_output=True, text=True)
    os.unlink(f.name)
    print(r.stdout if r.returncode == 0 else f"[pass failed]\n{r.stderr[-2000:]}\n--- input ---\n{mlir}")

    # ── 3. XLA's HLO before / after optimization ─────────────────────────────
    section("3. XLA HLO for ds_split_probe_add0, before and after optimization")
    dump_dir = tempfile.mkdtemp(prefix="ds_xla_dump_")
    run(_VALUES_CODE, {"DS_RETURN_PAIRS": "1",
                       "XLA_FLAGS": f"--xla_dump_to={dump_dir} --xla_dump_hlo_as_text"})
    files = sorted(glob.glob(os.path.join(dump_dir, "*ds_split_probe_add0*")))
    shown = 0
    for stage in ("before_optimizations", "after_optimizations"):
        for path in files:
            name = os.path.basename(path)
            if stage in name and name.endswith(".txt") and "buffer" not in name \
                    and "memory" not in name and "thunk" not in name:
                print(f"\n--- {name} ---")
                with open(path) as fh:
                    text = fh.read()
                print(text if len(text) < 12000 else text[:12000] + "\n... [truncated]")
                shown += 1
    if not shown:
        print("  No matching dump files. Files written:")
        for path in sorted(glob.glob(os.path.join(dump_dir, "*")))[:60]:
            print("   ", os.path.basename(path))
    shutil.rmtree(dump_dir, ignore_errors=True)
