"""
Go/no-go probe: can JAX/XLA f64 matmuls run on cuBLAS's FP64 emulation?

Plain JAX, no DS plugin. Needs jax[cuda13] whose cuBLAS is from CUDA 13.0
Update 2 or newer (see tests/run_xla_emulation_probe_host.sh).

XLA normally emits its own Triton GEMM for f64 dots, which cuBLAS emulation
cannot touch. --xla_gpu_enable_triton_gemm=false sends dots to cuBLAS instead
(custom call __cublas$gemm). cuBLAS emulation is then switched on from outside
with CUBLAS_EMULATE_DOUBLE_PRECISION=1, since XLA never sets the emulated math
mode on its handle. Reports exist that this has no effect under JAX, which is
what this probe checks.

Configurations, each in its own subprocess (env vars are read at cuBLAS init):
  triton      XLA default (Triton GEMM)
  cublas      Triton GEMM off -> native cuBLAS DGEMM
  cublas+emu  Triton GEMM off, CUBLAS_EMULATE_DOUBLE_PRECISION=1
  +eager      as above, CUBLAS_EMULATION_STRATEGY=eager

Emulation counts as taken when, against 'cublas' at the same n, the time drops
by at least 1.5x AND the output is not bitwise identical. Its accuracy is
reported as the relative Frobenius error against a NumPy f64 product.

    python3 tests/probe_xla_emulation.py [n ...]      (default 2048 4096)
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SIZES = [int(a) for a in sys.argv[1:]] or [2048, 4096]
REPS = 10

CONFIGS = [
    ("triton",     "",                                   {}),
    ("cublas",     "--xla_gpu_enable_triton_gemm=false", {}),
    ("cublas+emu", "--xla_gpu_enable_triton_gemm=false", {"CUBLAS_EMULATE_DOUBLE_PRECISION": "1"}),
    ("+eager",     "--xla_gpu_enable_triton_gemm=false", {"CUBLAS_EMULATE_DOUBLE_PRECISION": "1",
                                                          "CUBLAS_EMULATION_STRATEGY": "eager"}),
]

# argv: n, input .npz, output .npy
_CHILD = r"""
import sys, time, json, re
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
n, inp, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
d = np.load(inp)
a, b = jax.device_put(d["a"]), jax.device_put(d["b"])
f = jax.jit(lambda x, y: jnp.matmul(x, y, precision=jax.lax.Precision.HIGHEST))
hlo = f.lower(a, b).compile().as_text()
targets = sorted(set(re.findall(r'custom_call_target="([^"]+)"', hlo)))
kind = ("cublas" if any("cublas" in t for t in targets)
        else "triton" if ("triton" in hlo or "__triton" in hlo) else "other")
c = f(a, b).block_until_ready()
for _ in range(2):
    f(a, b).block_until_ready()
ts = []
for _ in range(%d):
    t0 = time.perf_counter(); f(a, b).block_until_ready(); ts.append(time.perf_counter() - t0)
np.save(out, np.asarray(c))
print("RESULT " + json.dumps({"ms": 1e3 * float(np.median(ts)), "kind": kind, "targets": targets}))
""" % REPS


def run(n, xla_flags, extra, inp, out):
    env = dict(os.environ)
    for k in ("CUBLAS_EMULATE_DOUBLE_PRECISION", "CUBLAS_EMULATION_STRATEGY", "XLA_FLAGS"):
        env.pop(k, None)
    if xla_flags:
        env["XLA_FLAGS"] = xla_flags
    env.update(extra)
    r = subprocess.run([sys.executable, "-c", _CHILD, str(n), inp, out],
                       env=env, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    print(f"    [subprocess failed]\n{r.stderr[-1500:]}")
    return None


def main():
    import importlib.metadata as md
    import jax
    vers = {}
    for pkg in ("jax", "jaxlib", "jax-cuda13-plugin", "nvidia-cublas", "nvidia-cublas-cu13"):
        try:
            vers[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    print("Versions :", ", ".join(f"{k} {v}" for k, v in vers.items()))
    print("Devices  :", jax.devices())
    print()

    verdicts = []
    with tempfile.TemporaryDirectory() as tmp:
        for n in SIZES:
            rng = np.random.default_rng(n)
            a, b = rng.standard_normal((n, n)), rng.standard_normal((n, n))
            inp = os.path.join(tmp, "in.npz")
            np.savez(inp, a=a, b=b)
            ref = a @ b
            nref = np.linalg.norm(ref)
            print(f"n = {n}")
            res, outs = {}, {}
            for name, flags, extra in CONFIGS:
                out = os.path.join(tmp, f"{name}.npy")
                r = run(n, flags, extra, inp, out)
                if r is None:
                    continue
                c = np.load(out)
                r["err"] = float(np.linalg.norm(c - ref) / nref)
                res[name], outs[name] = r, c
                print(f"  {name:11s} {r['ms']:9.3f} ms  err vs NumPy {r['err']:.2e}  "
                      f"GEMM path: {r['kind']} {r['targets']}")
            if "cublas" in res:
                base, cb = res["cublas"], outs["cublas"]
                if base["kind"] != "cublas":
                    print("  ! Triton GEMM off did not route the dot to cuBLAS; emulation cannot apply.")
                for name in ("cublas+emu", "+eager"):
                    if name not in res:
                        continue
                    r = res[name]
                    speed = base["ms"] / r["ms"]
                    same = np.array_equal(outs[name], cb)
                    taken = speed >= 1.5 and not same
                    verdicts.append((n, name, taken, r["err"]))
                    print(f"  {name:11s} vs native cuBLAS: {speed:.2f}x faster, output "
                          f"{'bitwise identical' if same else 'differs'} -> emulation "
                          f"{'TAKEN' if taken else 'NOT taken'}")
                if "triton" in res:
                    print(f"  native cuBLAS vs Triton: {res['triton']['ms'] / base['ms']:.2f}x "
                          f"(>1 means cuBLAS is faster)")
            print()

    taken = [v for v in verdicts if v[2]]
    f64_level = all(v[3] < 1e-13 for v in taken)
    print("VERDICT:", "GO -- JAX f64 matmuls reach cuBLAS emulation"
          + ("" if f64_level else " (but error above 1e-13: check accuracy)")
          if taken else "NO-GO -- emulation not taken from JAX in any configuration")


if __name__ == "__main__":
    main()
