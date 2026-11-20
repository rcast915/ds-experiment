"""
Does loading the DS proxy, or the StableHLO version downgrade, change the
native f64 baseline?

The f64 baseline in every benchmark is the proxy with DS_BYPASS=1. Under
DS_BYPASS=1 the proxy forwards everything, including the plugin attributes,
so JAX serializes for the backend's own StableHLO version (1.25.3), exactly as
without the proxy. The downgrade to the pass's 1.16.3 happens only when the
pass is active. This script separates the two effects, timing the same native
f64 program (the density-matrix purification of bench_purification.py) three
ways, each in its own process:

  plain JAX     the real CUDA plugin (the .so the proxy wraps), no proxy
  DS_BYPASS=1   proxy loaded, forwarding everything        -> proxy overhead
  1.16.3 f64    proxy loaded, StableHLO reported as 1.16.3, pass run but its
                output discarded (DS_TEST_PASSTHROUGH=1), so XLA compiles the
                original program as serialized at 1.16.3  -> downgrade effect

The modes are interleaved over ROUNDS rounds, so a disturbance on a shared
machine shows up as spread within a mode, not as a difference between modes.
Outputs are compared against plain JAX (max |difference|; 0 = bitwise equal).

Usage (inside container, after ds_setup.sh):
  python3 tests/bench_f64_baseline.py [L ...]
"""

import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from bench_purification import _CODE, hamiltonian, initial_guess, ITERS, WARMUP_REPS, BENCH_REPS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROXY_SO = PROJECT_ROOT / "pjrt_plugin" / "build" / "libds_pjrt_plugin.so"
LATTICES = [32, 46, 64]          # n = 1024, 2116, 4096
ROUNDS = 3

DS_VARS = ("DS_BYPASS", "DS_TEST_PASSTHROUGH", "DS_GEMM_KERNEL", "DS_F32_MATMUL",
           "DS_STRICT", "DS_RETURN_PAIRS", "DS_PASS_MODE")


def real_plugin():
    """The JAX CUDA plugin .so (renamed *_disabled by ds_setup.sh)."""
    for p in sys.path:
        hits = sorted(glob.glob(os.path.join(p, "jax_plugins", "xla_cuda1*", "xla_cuda_plugin.so")))
        if hits:
            return hits[0]
    return None


def modes(real_so):
    return [
        # label,        plugin .so, extra env
        ("plain JAX",   real_so,    {}),
        ("DS_BYPASS=1", PROXY_SO,   {"DS_BYPASS": "1"}),
        ("1.16.3 f64",  PROXY_SO,   {"DS_TEST_PASSTHROUGH": "1"}),
    ]


def run(so, extra, in_path, out_path):
    env = {k: v for k, v in os.environ.items() if k not in DS_VARS}
    env.update(JAX_ENABLE_X64="1", PJRT_NAMES_AND_LIBRARY_PATHS=f"cuda:{so}", **extra)
    r = subprocess.run([sys.executable, "-c", _CODE, "float64", str(ITERS), str(WARMUP_REPS),
                        str(BENCH_REPS), in_path, out_path],
                       env=env, capture_output=True, text=True)
    producers = sorted({l.split("serialized by ")[1].strip()
                        for l in r.stderr.splitlines() if "serialized by" in l})
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])["median"] * 1000, producers
    print(f"  [subprocess failed]\n{r.stderr[-1500:]}", file=sys.stderr)
    return None, producers


def main():
    lattices = [int(a) for a in sys.argv[1:]] or LATTICES
    real_so = real_plugin()
    if not real_so or not PROXY_SO.exists():
        print(f"Needs the real CUDA plugin (found: {real_so}) and the proxy ({PROXY_SO}).")
        return 1
    ms = modes(real_so)

    print("=" * 100)
    print(f"  f64 baseline: plain JAX vs proxy (DS_BYPASS=1) vs StableHLO 1.16.3 serialization")
    print(f"  purification, {ITERS} iterations = {2 * ITERS} matmuls, native f64 throughout")
    print(f"  warmup={WARMUP_REPS}, reps={BENCH_REPS} per process (median), {ROUNDS} interleaved rounds")
    print(f"  real plugin: {real_so}")
    print("=" * 100)

    rc = 0
    with tempfile.TemporaryDirectory() as tmp:
        for L in lattices:
            H = hamiltonian(L)
            n, ne = L * L, (L * L) // 2
            in_path = os.path.join(tmp, f"in_{L}.npz")
            np.savez(in_path, P0=initial_guess(H, ne))

            times = {label: [] for label, _, _ in ms}
            prods = {label: set() for label, _, _ in ms}
            outs = {}
            for rnd in range(ROUNDS):
                for label, so, extra in ms:
                    out_path = os.path.join(tmp, f"out_{L}_{rnd}_{label[:5].replace(' ', '_')}.npy")
                    t, p = run(so, extra, in_path, out_path)
                    prods[label].update(p)
                    if t is None:
                        rc = 1
                        continue
                    times[label].append(t)
                    outs.setdefault(label, np.load(out_path))

            print(f"\n  n = {n}  (L = {L})")
            print(f"  {'run':<12} {'median ms':>10} {'min':>9} {'max':>9} {'vs plain':>9}  "
                  f"{'max |P - P_plain|':>17}  serialized by (proxy log)")
            print("  " + "-" * 96)
            base = np.median(times["plain JAX"]) if times["plain JAX"] else None
            for label, _, _ in ms:
                t = times[label]
                if not t:
                    print(f"  {label:<12} {'ERR':>10}")
                    continue
                med = np.median(t)
                rel = f"{med / base:8.3f}x" if base else f"{'-':>9}"
                diff = (np.max(np.abs(outs[label] - outs["plain JAX"]))
                        if "plain JAX" in outs else float("nan"))
                prod = ", ".join(sorted(prods[label])) or "(not logged)"
                print(f"  {label:<12} {med:>10.2f} {min(t):>9.2f} {max(t):>9.2f} {rel}  "
                      f"{diff:>17.1e}  {prod}")

    print("\nNotes:")
    print("  vs plain = median time / plain JAX median time (1.000x = no change).")
    print("  min/max over the rounds: a mode whose spread covers the others' medians shows no")
    print("  measurable difference. 0.0e+00 = output bitwise identical to plain JAX.")
    print("  DS_BYPASS=1 is the f64 baseline in every other benchmark; it is not downgraded.")
    print("  The 1.16.3 row is the f64 program as the pass receives it, compiled natively.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
