"""
Black-Scholes run for a profiler: warm up, then a few timed calls bracketed by
cudaProfilerStart/Stop, so Nsight records only the steady-state calls (no
compilation, no data transfer, no warm-up).

  python3 tests/profile_blackscholes.py <batch> <reps>

Native f64 or DS is chosen the usual way: DS_BYPASS=1 gives native f64, the
plugin without it gives DS. Inputs are always f64. Used by tests/run_nsight.sh.
"""

import os
import sys
import ctypes
from pathlib import Path

os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np
import jax
import jax.numpy as jnp

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))
from bench_blackscholes import _BENCH_CODE  # noqa: E402

# The kernel definitions from the benchmark payload, so both measure the same code.
_ns = {"np": np}
exec(_BENCH_CODE[_BENCH_CODE.index("INV_SQRT_2PI"):_BENCH_CODE.index("rng = np.random")], _ns)
black_scholes = _ns["black_scholes"]


def cudart():
    for name in ("libcudart.so", "libcudart.so.12", "/usr/local/cuda/lib64/libcudart.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16_777_216
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    rng = np.random.default_rng(42)
    inputs = [rng.uniform(lo, hi, n) for lo, hi in
              [(10.0, 200.0), (10.0, 200.0), (0.01, 0.10), (0.05, 0.65), (0.05, 2.0)]]
    otype = rng.integers(0, 2, n).astype(np.int32)
    args = [jnp.array(a) for a in inputs] + [jnp.array(otype)]

    fn = jax.jit(lambda *a: black_scholes(jnp, *a))
    for _ in range(5):
        fn(*args).block_until_ready()

    rt = cudart()
    if rt is None:
        print("[profile] libcudart not found: profiling the whole run, warm-up included",
              file=sys.stderr)
    else:
        rt.cudaProfilerStart()
    for _ in range(reps):
        fn(*args).block_until_ready()
    if rt is not None:
        rt.cudaProfilerStop()
    print(f"[profile] batch={n} reps={reps} mode={'f64' if os.environ.get('DS_BYPASS') == '1' else 'DS'}")


if __name__ == "__main__":
    main()
