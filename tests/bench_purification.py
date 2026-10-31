"""
Density-matrix purification: a matmul-heavy electronic-structure kernel,
native f64 vs DS vs native f32.

Linear-scaling electronic-structure codes obtain the one-particle density
matrix P from the Hamiltonian H without diagonalization, by iterating a
matrix polynomial that drives the eigenvalues of P to 0 or 1. This uses the
canonical purification of Palser and Manolopoulos (trace-conserving): per
iteration two matrix products (P^2, P^3), three traces, and a coefficient c
that selects one of two update formulas. 25 iterations, written as a plain
Python loop so jax.jit unrolls it (the pass does not handle while loops).

Test system: 2D square-lattice tight-binding model, L x L sites (n = L^2),
nearest-neighbour hopping t = -1, checkerboard on-site energies +/-1 with
small random disorder, periodic boundaries, half filling (ne = n/2). The
checkerboard opens a gap of ~1.9 at the Fermi level, so 25 iterations
converge to f64 accuracy (checked against NumPy).

Runs, each in its own subprocess:
  f64        f64 inputs, DS_BYPASS=1                (native f64)
  DS         f64 inputs, plugin                     (DS elementwise + DS matmul kernel)
  DS native  f64 inputs, plugin, DS_GEMM_KERNEL=0   (DS elementwise, native f64 matmul)
  f32        f32 inputs, DS_BYPASS=1                (native f32)

Accuracy is measured against the exact density matrix from a NumPy f64
eigendecomposition: ||P - P_exact||/||P_exact||, idempotency ||P^2 - P||/||P||,
electron-count error |tr P - ne|, and band-energy relative error of tr(P H).

Usage (inside container, after ds_setup.sh):
  PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \\
      python3 tests/bench_purification.py [L ...]
"""

import sys
import os
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

TESTS_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
PLUGIN_SO    = PROJECT_ROOT / "pjrt_plugin" / "build" / "libds_pjrt_plugin.so"
OPT_BINARY   = PROJECT_ROOT / "stablehlo_pass" / "build" / "mlir-ds-opt"

ITERS       = 25
WARMUP_REPS = 2
BENCH_REPS  = 5
LATTICES    = [32, 46, 64]          # n = 1024, 2116, 4096


def hamiltonian(L, delta=1.0, t=-1.0, disorder=0.1, seed=0):
    rng = np.random.default_rng(seed)
    n = L * L
    H = np.zeros((n, n))
    idx = lambda i, j: (i % L) * L + (j % L)
    for i in range(L):
        for j in range(L):
            a = idx(i, j)
            H[a, a] = (delta if (i + j) % 2 == 0 else -delta) + disorder * rng.uniform(-1, 1)
            for b in (idx(i + 1, j), idx(i, j + 1)):
                H[a, b] = H[b, a] = t
    return H


def initial_guess(H, ne):
    """Palser-Manolopoulos starting point from Gershgorin bounds of H."""
    n = H.shape[0]
    off = np.abs(H).sum(1) - np.abs(np.diag(H))
    hmax = np.max(np.diag(H) + off)
    hmin = np.min(np.diag(H) - off)
    mu = np.trace(H) / n
    lam = min(ne / (hmax - mu), (n - ne) / (mu - hmin))
    return (lam / n) * (mu * np.eye(n) - H) + (ne / n) * np.eye(n)


# Injected via -c; argv: dtype, iters, warmup, reps, input .npz, output .npy
_CODE = """
import sys, os, time, json
import numpy as np
os.environ["JAX_ENABLE_X64"] = "1"
import jax
import jax.numpy as jnp

dtype, iters = sys.argv[1], int(sys.argv[2])
warmup, reps = int(sys.argv[3]), int(sys.argv[4])
in_path, out_path = sys.argv[5], sys.argv[6]
dt = jnp.float64 if dtype == "float64" else jnp.float32

data = np.load(in_path)
P0 = jnp.array(data["P0"], dtype=dt)
I = jnp.array(np.eye(P0.shape[0]), dtype=dt)   # an input, so the pass tracks it

def purify(P, I):
    for _ in range(iters):
        P2 = P @ P
        P3 = P2 @ P
        trP = jnp.sum(P * I)
        trP2 = jnp.sum(P2 * I)
        trP3 = jnp.sum(P3 * I)
        den = trP - trP2
        safe = den != 0.0
        c = jnp.where(safe, (trP2 - trP3) / jnp.where(safe, den, 1.0), 0.5)
        c = jnp.minimum(jnp.maximum(c, 0.0), 1.0)   # not jnp.clip: clamp isn't DS-transformed
        A = ((1.0 + c) * P2 - P3) / jnp.maximum(c, 0.5)
        B = ((1.0 - 2.0 * c) * P + (1.0 + c) * P2 - P3) / jnp.maximum(1.0 - c, 0.5)
        P = jnp.where(c >= 0.5, A, B)
    return P

fn = jax.jit(purify)
for _ in range(warmup):
    fn(P0, I).block_until_ready()
times = []
for _ in range(reps):
    t0 = time.perf_counter()
    out = fn(P0, I)
    out.block_until_ready()
    times.append(time.perf_counter() - t0)
np.save(out_path, np.asarray(out, dtype=np.float64))
print("RESULT " + json.dumps({"median": float(np.median(times))}))
"""

# Lower the same purify() for a small f64 problem on CPU and run the pass with
# DS_WARN_UNSUPPORTED=1: any op that consumes a DS value without being
# transformed is reported, and would silently run at native precision.
_LOWER_CODE = _CODE.split("data = np.load")[0] + """
def purify(P, I):
""" + _CODE.split("def purify(P, I):")[1].split("fn = jax.jit")[0] + """
s = jax.ShapeDtypeStruct((64, 64), jnp.float64)
print(jax.jit(purify).lower(s, s).compiler_ir())
"""


def check_coverage():
    env = {k: v for k, v in os.environ.items() if k != "PJRT_NAMES_AND_LIBRARY_PATHS"}
    env["JAX_PLATFORMS"] = "cpu"
    r = subprocess.run([sys.executable, "-c", _LOWER_CODE, "float64", "2", "0", "0", "-", "-"],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        return f"could not lower purify(): {r.stderr[-300:]}"
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(r.stdout)
    env2 = dict(env, DS_WARN_UNSUPPORTED="1", DS_GEMM_KERNEL_AVAILABLE="1")
    p = subprocess.run([str(OPT_BINARY),
                        "--pass-pipeline=builtin.module(inline,func.func(ds-transform))", f.name],
                       env=env2, capture_output=True, text=True)
    os.unlink(f.name)
    if p.returncode != 0:
        return f"pass failed: {p.stderr[-300:]}"
    warnings = [l for l in p.stderr.splitlines() if "unsupported op" in l]
    n_calls = p.stdout.count("ds_gemm")
    n_dots = p.stdout.count("stablehlo.dot_general")
    ok = (n_calls == 4 and n_dots == 0 and not warnings)
    lines = [f"2 iterations lowered: {n_calls} DS kernel calls, {n_dots} remaining dot_general, "
             f"{len(warnings)} untransformed ops -- "
             + ("OK" if ok else "PROBLEM: expected 4 kernel calls, 0 dot_general, 0 untransformed; "
                               "results below would not be fully DS")]
    lines += [f"    {w.strip()}" for w in warnings[:10]]
    return "\n  ".join(lines)


MODES = [
    # label,       dtype,     bypass, extra env
    ("f64",        "float64", True,   {}),
    ("DS",         "float64", False,  {}),
    ("DS native",  "float64", False,  {"DS_GEMM_KERNEL": "0"}),
    ("f32",        "float32", True,   {}),
]


def run_mode(dtype, bypass, extra_env, in_path, out_path):
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"
    for k in ("DS_BYPASS", "DS_GEMM_KERNEL", "DS_F32_MATMUL"):
        env.pop(k, None)
    if bypass:
        env["DS_BYPASS"] = "1"
    env.update(extra_env)
    r = subprocess.run(
        [sys.executable, "-c", _CODE, dtype, str(ITERS), str(WARMUP_REPS), str(BENCH_REPS),
         in_path, out_path],
        env=env, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    print(f"  [subprocess failed]\n{r.stderr[-1500:]}", file=sys.stderr)
    return None


def metrics(P, H, ne, P_exact, E_exact):
    nP = np.linalg.norm(P)
    return {
        "dP": np.linalg.norm(P - P_exact) / np.linalg.norm(P_exact),
        "idem": np.linalg.norm(P @ P - P) / nP,
        "tr": abs(np.trace(P) - ne),
        "E": abs((P * H).sum() - E_exact) / abs(E_exact),
    }


def main():
    lattices = [int(a) for a in sys.argv[1:]] or LATTICES
    if not os.environ.get("PJRT_NAMES_AND_LIBRARY_PATHS") or not PLUGIN_SO.exists():
        print("Needs the PJRT plugin (set PJRT_NAMES_AND_LIBRARY_PATHS).")
        return 1

    print("=" * 110)
    print(f"  Density-matrix purification (canonical, {ITERS} iterations = {2 * ITERS} matmuls), "
          f"2D tight-binding, half filling")
    print(f"  warmup={WARMUP_REPS}, reps={BENCH_REPS}, median wall time per full purification")
    print("=" * 110)

    print(f"\n  Pass coverage check: {check_coverage()}")

    with tempfile.TemporaryDirectory() as tmp:
        for L in lattices:
            H = hamiltonian(L)
            n, ne = L * L, (L * L) // 2
            w, V = np.linalg.eigh(H)
            P_exact = V[:, :ne] @ V[:, :ne].T
            E_exact = w[:ne].sum()
            in_path = os.path.join(tmp, f"in_{L}.npz")
            np.savez(in_path, P0=initial_guess(H, ne))

            print(f"\n  n = {n}  (L = {L}, ne = {ne}, gap {w[ne] - w[ne - 1]:.2f})")
            print(f"  {'run':<10} {'ms':>10} {'f64/run':>8}   {'||P-Pexact||':>12} "
                  f"{'||P^2-P||':>10} {'|tr P - ne|':>11} {'energy err':>10}")
            print("  " + "-" * 80)
            base = None
            for label, dtype, bypass, extra in MODES:
                out_path = os.path.join(tmp, f"out_{L}_{label.replace(' ', '_')}.npy")
                r = run_mode(dtype, bypass, extra, in_path, out_path)
                if r is None:
                    print(f"  {label:<10} {'ERR':>10}")
                    continue
                ms = r["median"] * 1000
                if label == "f64":
                    base = ms
                m = metrics(np.load(out_path), H, ne, P_exact, E_exact)
                ratio = f"{base / ms:7.2f}x" if base else f"{'-':>8}"
                print(f"  {label:<10} {ms:>10.2f} {ratio}   {m['dP']:>12.1e} {m['idem']:>10.1e} "
                      f"{m['tr']:>11.1e} {m['E']:>10.1e}")

    print("\nNotes:")
    print("  f64/run > 1 means that run is faster than native f64.")
    print("  Errors are against the exact density matrix from a NumPy f64 eigendecomposition.")
    print("  DS = DS elementwise ops + DS matmul kernel; DS native = native f64 matmuls (DS_GEMM_KERNEL=0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
