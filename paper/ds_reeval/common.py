"""
Shared utilities for the ds_reeval suite: timing protocol, GPU/env
detection, JSON I/O, and small numeric helpers.

Deliberately does NOT import jax at module level. DS_BYPASS, DS_RETURN_PAIRS,
JAX_ENABLE_X64, XLA_FLAGS and PJRT_NAMES_AND_LIBRARY_PATHS are all read at
plugin-load / XLA-init time, so a worker script must finish setting
os.environ *before* it (or anything it imports) imports jax. Every function
here that needs jax/numpy imports it lazily, inside the function body, so
that `import common` itself is always safe regardless of env-var ordering.
"""
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

IQR_ESCALATE_THRESHOLD = 0.20
ESCALATE_REPS = 100

THIS_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = THIS_DIR / "results"

# Env vars whose value is worth recording alongside every result, since the
# whole point of this suite is that env state determines correctness.
_RELEVANT_ENV_KEYS = [
    "PJRT_NAMES_AND_LIBRARY_PATHS", "DS_BYPASS", "DS_TEST_PASSTHROUGH",
    "DS_PASS_MODE", "DS_RETURN_PAIRS", "JAX_ENABLE_X64", "XLA_FLAGS",
]

# nvidia-smi --query-gpu=compute_cap --format=csv,noheader -> sm_XXX name.
# Only the two target GPUs from the environment brief are named explicitly;
# anything else gets a best-effort fallback so the suite doesn't hard-fail
# on a third GPU, it just can't give it the "a"-suffix compute capability
# name (e.g. sm_90a vs sm_90) that NVIDIA uses for some architectures.
_ARCH_BY_COMPUTE_CAP = {
    "9.0": "sm_90a",  # H100 PCIe
    "8.9": "sm_89",   # L40S
}


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def snapshot_relevant_env():
    return {k: os.environ.get(k) for k in _RELEVANT_ENV_KEYS}


def detect_gpu(required=True):
    """Returns {'name', 'driver_version', 'compute_cap', 'arch'} via nvidia-smi.

    If nvidia-smi is unavailable and required=False, returns an
    'UNKNOWN-GPU' placeholder instead of raising, so `run_all.py --dry-run`
    can print a plan from a login node with no GPU attached. Real (non
    dry-run) invocations must call this with required=True: a result file
    tagged with an unknown GPU defeats the purpose of tagging at all.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        line = proc.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        name, driver, cc = parts[0], parts[1], parts[2]
        arch = _ARCH_BY_COMPUTE_CAP.get(cc, "sm_" + cc.replace(".", ""))
        return {"name": name, "driver_version": driver, "compute_cap": cc, "arch": arch}
    except Exception as e:
        if required:
            raise RuntimeError(
                "GPU detection via nvidia-smi failed: {}. Real runs must "
                "execute on a GPU node inside the container; use --dry-run "
                "to plan from elsewhere.".format(e)
            ) from e
        return {"name": "UNKNOWN-GPU", "driver_version": "unknown",
                "compute_cap": "unknown", "arch": "unknown"}


def sanitize_gpu_tag(name):
    """Filesystem-safe GPU tag, e.g. 'NVIDIA H100 PCIe' -> 'NVIDIA-H100-PCIe'."""
    return "".join(c if c.isalnum() else "-" for c in name).strip("-") or "UNKNOWN-GPU"


def default_result_path(gpu, experiment, config_name):
    """Fallback output path for standalone (non run_all.py) invocation."""
    tag = sanitize_gpu_tag(gpu["name"]) if isinstance(gpu, dict) else str(gpu)
    return RESULTS_ROOT / tag / "{}_{}.json".format(experiment, config_name)


def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def block_until_ready(x):
    """Blocks on a single device array or any pytree of them."""
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
        return x
    import jax
    for leaf in jax.tree_util.tree_leaves(x):
        leaf.block_until_ready()
    return x


def _percentile(sorted_values, p):
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    idx = p * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _iqr(values):
    s = sorted(values)
    return _percentile(s, 0.75) - _percentile(s, 0.25)


def timed_run(fn, warmup=5, reps=30, escalate_reps=ESCALATE_REPS,
              iqr_ratio_threshold=IQR_ESCALATE_THRESHOLD):
    """Measurement protocol shared by every experiment.

    fn: zero-arg callable returning a device array (or pytree of them).
    Runs `warmup` untimed calls, then `reps` timed calls, each immediately
    followed by block_until_ready so the timer captures device completion,
    not just dispatch. Reports the median wall time in ms. If the IQR
    exceeds `iqr_ratio_threshold` of the median (default 20%), re-times
    fresh at `escalate_reps` repetitions and reports that run instead,
    with escalated=True so the report can flag it.
    """
    for _ in range(warmup):
        block_until_ready(fn())

    def _sample(n):
        timings = []
        for _ in range(n):
            t0 = time.perf_counter()
            out = fn()
            block_until_ready(out)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)
        return timings

    timings = _sample(reps)
    median = statistics.median(timings)
    iqr = _iqr(timings)
    escalated = False
    if median > 0 and (iqr / median) > iqr_ratio_threshold:
        escalated = True
        timings = _sample(escalate_reps)
        median = statistics.median(timings)
        iqr = _iqr(timings)

    return {
        "timings_ms": timings,
        "median_ms": median,
        "iqr_ms": iqr,
        "reps_used": len(timings),
        "escalated": escalated,
    }


def gemm_tflops(n, seconds):
    """Effective TFLOPS for an NxN square GEMM, 2*N^3 FLOPs convention.

    Note this is the *standard* GEMM FLOP count, not the DS-adjusted count:
    a DS matmul issues 4 sub-GEMMs, so its true FLOP count is ~4x this. Both
    this suite and the paper report TFLOPS against the standard 2*N^3
    denominator for both f64 and DS so the two columns stay comparable; see
    the note report.py prints alongside every TFLOPS table.
    """
    return (2.0 * n ** 3) / seconds / 1e12


def half_ulp_f32(value):
    """Half the spacing between adjacent float32 values at |value|'s magnitude.

    This is the theoretical best-case error floor for any computation whose
    *final* output is quantized to a single f32 value, regardless of how
    much internal precision fed into it.
    """
    import numpy as np
    v32 = np.float32(value)
    return float(np.spacing(v32)) / 2.0


def assert_x64_enabled():
    """Fails loudly if float64 arrays are not actually being produced.

    Checking jax.config.jax_enable_x64 is not sufficient on its own: the
    environment brief for this suite explicitly warns that JAX can silently
    downcast f64 to f32 before the plugin ever sees the bytecode if
    JAX_ENABLE_X64 wasn't set *before* jax was imported. This checks
    observed behavior -- the dtype an undecorated array literal actually
    gets -- rather than trusting a config flag.
    """
    import jax.numpy as jnp
    dtype = jnp.zeros(1).dtype
    if dtype != jnp.float64:
        raise RuntimeError(
            "JAX_ENABLE_X64 is not actually active: jnp.zeros(1).dtype == "
            "{}, expected float64. This experiment requires f64 inputs. "
            "JAX_ENABLE_X64=1 must be set in the process environment "
            "*before* jax is imported (it cannot be turned on afterwards "
            "from within this script) -- refusing to silently continue in "
            "f32.".format(dtype)
        )


def x64_status():
    """Non-asserting variant of assert_x64_enabled().

    Experiment 3 is intentionally an f32-input test (matching the paper's
    own protocol for that specific claim), so it must not assert x64 -- it
    just records what was observed, for transparency in the JSON.
    """
    import jax.numpy as jnp
    dtype = jnp.zeros(1).dtype
    return {
        "jax_enable_x64_env": os.environ.get("JAX_ENABLE_X64"),
        "zeros_dtype_observed": str(dtype),
    }


def _guess_repo_root():
    # ds_reeval/ lives at <repo>/paper/ds_reeval/
    return THIS_DIR.parent.parent


def git_commit_info(repo_root=None):
    repo_root = repo_root or _guess_repo_root()
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty, "repo_root": str(repo_root)}
    except Exception as e:
        return {"commit": "unknown", "dirty": None, "repo_root": str(repo_root), "error": str(e)}


def cuda_toolkit_version():
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return "unknown (nvcc not on PATH)"
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True,
                              timeout=10, check=True).stdout
        for line in out.splitlines():
            if "release" in line.lower():
                return line.strip()
        return out.strip().splitlines()[-1]
    except Exception as e:
        return "unknown ({})".format(e)


def container_image_digest(cli_value=None):
    """Best-effort only: a process inside an already-running container has
    no reliable way to learn its own image digest unless told. Prefer the
    explicit --image-digest flag (see README.md for how to obtain it with
    `docker inspect` before/after `docker run`)."""
    if cli_value:
        return cli_value
    env_value = os.environ.get("DS_IMAGE_DIGEST")
    if env_value:
        return env_value
    return "unknown (pass --image-digest to run_all.py or export DS_IMAGE_DIGEST)"


def jax_version_or_unknown():
    try:
        import jax
        return jax.__version__
    except Exception as e:
        return "unknown ({})".format(e)


def build_manifest(args, gpu):
    """One manifest per GPU results directory (results/<gpu>/manifest.json),
    not a single shared file -- the suite is explicitly meant to run on two
    different clusters at two different times, and a single shared path
    would have the second run silently clobber the first run's provenance.
    """
    return {
        "generated_at": utc_now_iso(),
        "git": git_commit_info(),
        "container_image_digest": container_image_digest(getattr(args, "image_digest", None)),
        "jax_version": jax_version_or_unknown(),
        "cuda_toolkit_version": cuda_toolkit_version(),
        "stablehlo_version_documented": "1.16.3",
        "python_version": sys.version,
        "gpu": gpu,
        "run_all_args": vars(args) if hasattr(args, "__dict__") else str(args),
    }
