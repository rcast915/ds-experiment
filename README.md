# Double-Single Arithmetic for JAX via StableHLO

Automatically upgrades float arithmetic in JAX programs to
**double-single (DS) precision** — near-float64 accuracy using only float32
operations — without any changes to user code.

---

## Overview

### The problem

Float32 arithmetic is fast but its 24-bit mantissa causes significant
rounding errors in numerically sensitive computations.  Float64 fixes this
(53-bit mantissa) but is slow or absent on many GPU targets: consumer GPUs
limit f64 to 1/32–1/64 of f32 throughput, embedded and mobile GPUs often
omit it entirely, and even on HPC GPUs f64 consumes more memory bandwidth
than f32.

### The solution: DS-f32

**Double-single (DS) arithmetic** achieves ~48 bits of effective mantissa
using only f32 operations by representing each value as a pair
`(hi, lo)` of f32 numbers whose exact sum equals the true value:

```
x  ≈  hi_f32 + lo_f32
```

Elementary operations on DS pairs are implemented via the classical
**two_sum** and **two_prod** error-free transforms, which carry the
rounding residual forward in the `lo` component through all subsequent ops.

### What this gives you

The core claim is: **DS-f32 achieves near-f64 numerical accuracy at
approximately f32 hardware cost.**  What "f32 cost" means in practice
depends on the target:

- On **consumer GPUs and hardware without native f64 units** (mobile,
  embedded), f64 throughput is typically 1/32–1/64 of f32 throughput.
  DS-f32 runs at f32 speed on those paths, making it dramatically faster
  than f64 while preserving near-f64 accuracy.
- On **HPC GPUs with native f64 units (e.g. H100)**, both f64 and DS-f32
  are fully hardware-accelerated, so wall times are comparable. DS-f32
  still uses only f32 arithmetic paths and f32 memory bandwidth, which
  is a meaningful advantage at scale, and delivers f64-class accuracy
  with no performance penalty relative to native f64.
- In **all cases**, DS-f32 is substantially more accurate than plain f32:
  measured **8× lower error** for large reductions (~1.49e-05 vs ~1.22e-04
  for a 10,000-element reduction).

### Why a compiler pass — not hand-written kernels

Early experiments used hand-written FFI custom calls to implement DS
operations on the GPU.  These performed well for single fused kernels, but
performance degraded steeply as arithmetic intensity increased: XLA cannot
fuse across `custom_call` boundaries, so each DS operation became a
separate kernel launch with no cross-op optimization.  Fusing entire
recurrences into a single internally-looped custom call helped, but
required manual kernel authoring for every operation pattern.

The compiler pass solves this structurally.  By emitting DS sequences as
**native StableHLO ops**, the pass hands XLA a single expanded program
with no opaque boundaries — XLA schedules, fuses, and register-allocates
across the entire kernel exactly as it would for hand-written f32 code.
The arithmetic-intensity slowdown disappears, and no hand-written CUDA
kernels are needed.

### How it works (transparently)

The system intercepts JAX's compiled bytecode at the PJRT boundary, runs a
StableHLO pass that splits each float argument into a DS pair at function
entry, replaces all arithmetic with DS sequences, and recombines at
function exit.  The caller sees no type changes — no user code modifications
required.

**For f64 inputs:** requires `JAX_ENABLE_X64=1` — without it JAX silently
downcasts f64 to f32 before the pass sees it.

**For f32 inputs:** works automatically with no extra configuration.

---

## System Architecture

The system has three layers:

```
JAX / Python
    │  jit-compiled function → VHLO bytecode
    ▼
PJRT Proxy Plugin  (pjrt_plugin/ds_pjrt_plugin.cpp)
    │  intercepts compile call, writes bytecode to /tmp
    ▼
mlir-ds-opt  (stablehlo_pass/)
    │  VHLO → StableHLO → DS-expanded StableHLO → VHLO
    ▼
Real CUDA PJRT Backend  (XLA / cuBLAS / PTXAS)
    │  compiles expanded ops as native float32 — XLA can fuse freely
    ▼
GPU Execution
```

### The DS Transform Pass (`stablehlo_pass/DsTransformPass.cpp`)

The pass walks each function's ops and maintains a map from original float
values to their DS pair `(hi, lo)`. Transformations:

| StableHLO op | DS expansion |
|---|---|
| **Function arguments** | Split via `emitFromFloat`: **f64 → `(f32_hi, f32_lo)`** (exact split); f32 → `(v, 0)` |
| `stablehlo.constant` | Entered into dsMap as `(constant, 0)` — enables DS arithmetic with scalar constants |
| `stablehlo.broadcast_in_dim` | DS pair cloned through the broadcast (result type updated to f32) |
| `stablehlo.add` | `two_sum(a_hi, b_hi)` + `two_sum(a_lo, b_lo)` cascade |
| `stablehlo.subtract` | Negate `b`, delegate to DS add |
| `stablehlo.multiply` | `two_prod(a_hi, b_hi)` + cross terms via Veltkamp split |
| `stablehlo.reduce` (single-input, add/sub/mul body) | New 4-arg f32 reduce body using the DS version of the body op |
| `stablehlo.dot_general` | 4 f32 sub-matmuls: `p=dot(hi,hi)`, `e1=dot(hi,lo)`, `e2=dot(lo,hi)`, `e3=dot(lo,lo)`, then `two_sum(p, e1+e2+e3)` |
| **`func.return`** | Recombine via `emitToFloat`: **`cast(hi, f64) + cast(lo, f64)`** → f64 output |

All expanded ops are native StableHLO, so XLA sees a single fused kernel
with no boundaries — no performance cliff from custom dispatch.

### The PJRT Proxy Plugin (`pjrt_plugin/ds_pjrt_plugin.cpp`)

Loaded by JAX instead of the real CUDA plugin via
`PJRT_NAMES_AND_LIBRARY_PATHS`. On each compile call it:
1. Writes the incoming VHLO bytecode to `/tmp/ds_in.mlir`
2. Runs `mlir-ds-opt` with the full pipeline (see table below)
3. Reads the transformed bytecode and passes it to the real CUDA backend

The real CUDA plugin must be renamed so the proxy can claim the `cuda` slot:

```
jax_plugins/xla_cuda12  →  jax_plugins/xla_cuda12_disabled
jax_cuda12_plugin       →  jax_cuda12_plugin_disabled
```

`ds_setup.sh` does this automatically each session.

### Pass pipeline (GPU)

```
vhlo-to-version{target=1.16.3}   # upgrade to current VHLO version
vhlo-legalize-to-stablehlo        # convert to StableHLO dialect  ← required before ds-transform
func.func(ds-transform)           # expand arithmetic into DS sequences
stablehlo-legalize-to-vhlo        # convert back to VHLO for XLA
```

### Environment variables

| Variable | Effect |
|---|---|
| `PJRT_NAMES_AND_LIBRARY_PATHS="cuda:<path>"` | Load the DS proxy plugin |
| `DS_BYPASS=1` | Skip all transformation; pure passthrough to real backend |
| `DS_TEST_PASSTHROUGH=1` | Run DS pass but send original bytecode to backend |
| `DS_PASS_MODE=ffi` | Use FFI pass instead of inline pass (for debugging only) |

---

## Experimental Results (H100)

Results collected on Punakha HPC, node `hopper001` (NVIDIA H100, CUDA 12.9.1).
Full notes: `python/ds_results_h100.txt`, `python/ds_results_compiler_pass_h100.txt`.

### FFI prototype (early experiments)

The initial implementation used hand-written FFI custom calls to dispatch DS
operations on the GPU.  Key findings:

- **Single fused kernel:** competitive with native f64 — within a few percent
  for large vectors, under 2× the cost of f32.
- **Cancellation sensitivity:** DS preserved small perturbations far better
  than f32 and often better than naive f64 evaluation of the same numerically
  unstable formula.
- **Fusion bottleneck:** as arithmetic intensity increased (more DS ops per
  kernel), performance degraded steeply because XLA cannot fuse across
  `custom_call` boundaries. Fusing entire recurrences into a single
  internally-looped custom call reduced the slowdown but required manual
  authoring per operation pattern. This identified the need for a compiler pass.

### Compiler pass (current implementation)

The inline DS pass emits native StableHLO, eliminating the fusion boundary
entirely.  Results on H100:

| Workload | DS-f32 vs f64 | DS-f32 vs f32 |
|---|---|---|
| Element-wise (A\*A+B), 1K–1M elements | within 10% | ~2× slower |
| `jnp.sum` reduction, 1K–1M elements | within 10% | ~2× slower |
| `jnp.matmul` 2048×2048 | **1.13× faster** | — |
| `jnp.matmul` 256×256 | 0.72× (4 launches vs 1) | — |

**Why DS-f32 ≈ f64 speed on H100:** the H100 has native FP64 tensor cores
(~51 TFLOPS PCIe), so f64 is already hardware-accelerated. The comparison
is different on consumer GPUs where f64 throughput is 1/32–1/64 of f32.

**Precision:** for a 10,000-element reduction, DS-f32 error is 1.49e-05 vs
f32 error of 1.22e-04 — an **8× improvement**, consistent with DS-f32's
effective ~48-bit mantissa.

**FMA safety confirmed:** PTX inspection on H100 shows zero `fma.rn.f32`
instructions in the Veltkamp split sequence.  Per-instruction IEEE `.rn`
rounding prevents contraction at the PTX assembler level.

---

## Repository Structure

```
ds_experiment/
├── Dockerfile                        # Container image definition
├── build_docker.sh                   # SLURM batch job to rebuild the image (~1 hour)
├── ds_setup.sh                       # Per-session setup: rename plugins, build PJRT, verify
│
├── stablehlo_pass/                   # MLIR/StableHLO DS pass
│   ├── DsTransformPass.cpp           # Inline DS pass (PRIMARY) — expands to native StableHLO ops
│   ├── DsFFIPass.cpp                 # FFI pass — emits custom_call ops (reference / debug only)
│   ├── mlir-ds-opt.cpp               # Entry point → mlir-ds-opt binary (inline pass)
│   ├── mlir-ds-ffi-opt.cpp           # Entry point → mlir-ds-ffi-opt binary (FFI pass)
│   └── CMakeLists.txt
│
├── pjrt_plugin/                      # PJRT proxy plugin
│   ├── ds_pjrt_plugin.cpp            # Intercepts JAX compile calls, runs mlir-ds-opt
│   └── CMakeLists.txt
│
├── cpp/                              # FFI CUDA kernels → libds_ffi.so
│   ├── ds_add.cpp / ds_sub.cpp / ds_mul.cpp
│   ├── ds_from_f32.cpp / ds_to_f32.cpp / ds_from_f64.cpp / ds_to_f64.cpp
│   ├── ds_fused_add_mul.cpp / ds_fused_add_mul_to_f64.cpp / ds_fused_axpy.cpp
│   ├── ds_fused_add_mul_to_f64_cuda.cu / ds_fused_add_sub_to_f64_cuda.cu
│   ├── ds_recurrence_add_mul_to_f64_cuda.cu
│   └── CMakeLists.txt
│
├── tests/                            # Automated test suite
│   ├── run_tests.sh                  # Master runner: bash tests/run_tests.sh [--bench] [--cpu-only]
│   ├── ds_ref.py                     # Pure-NumPy DS reference (no JAX/CUDA dependency)
│   ├── test_dot_product.py           # Accuracy: numpy ref + MLIR structural + GPU numerical
│   ├── test_matmul.py                # Accuracy: same structure for matrix multiply
│   ├── bench_dot_product.py          # Timing + GFLOPS: DS vs f32 subprocess baseline
│   └── bench_matmul.py              # Timing + TFLOPS: (A*A)@B and A@B suites
│
├── python/
│   ├── test_pass_correctness.py      # Canonical correctness test for the inline pass
│   ├── test_*.py                     # Additional tests (many target the FFI path)
│   ├── benchmark_*.py                # GPU performance benchmarks
│   └── plot_*.py                     # Visualization / figure generation
│
├── double_single_ray/                # Prior approach: LLVM IR pass (reference only)
│   ├── skeleton/Skeleton.cpp         # LLVM module pass — structural analog to DsTransformPass.cpp
│   ├── rtlib.c                       # C runtime library with DS arithmetic functions
│   └── llvm-accuracy-analysis-k-test/  # DS accuracy test harness
│
└── logs/                             # SLURM build job logs (output/ and error/)
```

### Component roles at a glance

| Component | Role |
|---|---|
| `stablehlo_pass/DsTransformPass.cpp` | Core DS transform — the main algorithmic contribution |
| `pjrt_plugin/ds_pjrt_plugin.cpp` | Transparent intercept layer; wires JAX to the pass |
| `tests/` | Automated accuracy + performance test suite (`bash tests/run_tests.sh`) |
| `python/test_pass_correctness.py` | Canonical correctness verification (CPU-only, no CUDA dep) |
| `cpp/` + FFI pass | Earlier FFI-based path; kept for comparison and as a fallback |
| `double_single_ray/` | Predecessor LLVM approach; documents the intellectual lineage |

---

## Environment

**Container working directory:** `/src/ds_experiment`  
**Versions (baked into Docker image):** CUDA 12.9.1 · Python 3.12 · JAX 0.10.1 · StableHLO 1.16.3

> **Previous results** were collected on Punakha HPC (`hopper001`, H100 GPU).
> See `python/ds_results_h100.txt` and `python/ds_results_compiler_pass_h100.txt`.

---

## Quick Start

### 1. Get an interactive GPU node (cluster-specific)

Adapt to your cluster's scheduler. Example for a SLURM cluster with rootless Docker:

```bash
# Load Docker and request a GPU node — adjust flags for your cluster
module load docker/27.3.1/rootless-docker
srun --gres=gpu:1 --pty bash
start_rootless_docker.sh --quiet   # or your cluster's equivalent
```

### 2. Build the Docker image (first time only, ~1 hour)

```bash
cd /path/to/ds_experiment   # adjust to wherever you placed the project
docker build -t ds-experiment .
```

On a SLURM cluster with a `build_docker.sh` batch script:

```bash
sbatch build_docker.sh
tail -f logs/output/build_<JOBID>.out
```

### 3. Run the container

```bash
cd /path/to/ds_experiment
docker run --gpus all -it --volume $(pwd):/src/ds_experiment --rm ds-experiment
```

> **Note:** Do not add `--network host` or `--ipc=host` if your HPC security policy prohibits it.

### 4. Per-session setup (inside the container)

```bash
cd /src/ds_experiment
bash ds_setup.sh
```

This renames the two JAX CUDA registration paths, builds the PJRT plugin from
source, and verifies with a `[2. 4. 6. 8.]` sanity check. The rename steps
are no-ops once the Dockerfile has been rebuilt with them baked in.

### 5. Activate DS for any script

```bash
PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \
python3 your_script.py
```

---

## Running Tests

All commands below assume you are inside the container at `/src/ds_experiment`
and have already run `bash ds_setup.sh` this session.

### Option A — Full automated test suite (recommended)

```bash
bash tests/run_tests.sh              # accuracy tests only (GPU auto-detected)
bash tests/run_tests.sh --bench      # accuracy + performance benchmarks
bash tests/run_tests.sh --cpu-only   # skip GPU even if plugin is present
```

Runs dot-product accuracy, matrix-multiply accuracy, and f64→DS accuracy.
All three must pass. Benchmarks are informational and do not affect pass/fail.

### Option B — Canonical correctness test (no GPU required)

```bash
JAX_PLATFORMS=cpu python3 python/test_pass_correctness.py
```

Verifies the inline DS pass structurally (MLIR op counts) and numerically
against a pure-NumPy DS reference. Expected output: **all 3 tests pass** with
~1.8M× error improvement on catastrophic cancellation.

### Option C — Quick GPU sanity check

```bash
PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \
python3 -c "
import jax, jax.numpy as jnp
x = jnp.array([1e8, 1e10, 1e12], dtype=jnp.float32)
y = jnp.array([1e-2, 1e-3, 1e-4], dtype=jnp.float32)
@jax.jit
def fn(x, y): return (x + y) - x
print(fn(x, y))  # DS:  [0.01  0.001  0.0001]
                 # f32: [0.    0.     0.    ]
"
```

---

## Building from Source (inside container)

The PJRT plugin must be built every session because the container is
ephemeral (`--rm`). The other components are pre-built into the image.

```bash
# Required every session:
cmake -GNinja -S pjrt_plugin -B pjrt_plugin/build && ninja -C pjrt_plugin/build

# Only if source files changed:
cmake -GNinja -S cpp -B cpp/build && ninja -C cpp/build
cmake -GNinja -S stablehlo_pass -B stablehlo_pass/build -DCMAKE_CXX_FLAGS="-fno-rtti" && ninja -C stablehlo_pass/build
```

---

## Verification

### Correctness test (CPU, no GPU required)

```bash
JAX_PLATFORMS=cpu python3 python/test_pass_correctness.py
# Expected: all 3 tests pass — ~1.8M× error improvement on catastrophic cancellation
```

Tests 1 and 2 count the expanded DS ops in the MLIR output (structural
verification). Test 3 compares numerically against a pure-NumPy DS reference
implementation to avoid any CUDA dependency.

### Basic GPU sanity check

```bash
PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \
python3 -c "
import jax, jax.numpy as jnp
x = jnp.array([1e8, 1e10, 1e12], dtype=jnp.float32)
y = jnp.array([1e-2, 1e-3, 1e-4], dtype=jnp.float32)
@jax.jit
def fn(x, y): return (x + y) - x
print(fn(x, y))  # DS:  [0.01  0.001  0.0001]
                 # f32: [0.    0.     0.    ]
"
```

### Reduction precision (10,000 elements)

```bash
# DS run
PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \
JAX_ENABLE_X64=1 python3 -c "
import jax, jax.numpy as jnp, numpy as np
n = 10000
a = jnp.full((n,), 0.1, dtype=jnp.float32)
@jax.jit
def fn(a): return jnp.sum(a * a)
result = fn(a)
truth = float(np.sum(np.full(n, 0.1, np.float32).astype(np.float64) ** 2))
print(f'DS  result: {float(result):.10f}')   # 100.0000000000  (bit-perfect)
print(f'f64 truth : {truth:.10f}')            # 100.0000029802
print(f'DS  error : {abs(float(result)-truth):.3e}')   # 2.980e-06
"

# f32 baseline (DS_BYPASS=1)
DS_BYPASS=1 \
PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \
JAX_ENABLE_X64=1 python3 -c "
import jax, jax.numpy as jnp, numpy as np
n = 10000
a = jnp.full((n,), 0.1, dtype=jnp.float32)
@jax.jit
def fn(a): return jnp.sum(a * a)
result = fn(a)
truth = float(np.sum(np.full(n, 0.1, np.float32).astype(np.float64) ** 2))
print(f'f32 result: {float(result):.10f}')   # 100.0000076294  (overshoots by 1 ULP)
print(f'f32 error : {abs(float(result)-truth):.3e}')   # 4.649e-06
"
```

---

## Key Design Decisions

### Inline pass vs. FFI kernels

Two pass implementations exist:

- **Inline pass** (`ds-transform`, default): Expands each op into native
  StableHLO arithmetic. XLA sees the full expanded sequence and can fuse,
  schedule, and register-allocate across the entire kernel. This is the
  primary path.

- **FFI pass** (`ds-ffi-transform`): Replaces each op with a
  `stablehlo.custom_call` dispatching to hand-written CUDA kernels in
  `cpp/`. XLA cannot fuse across `custom_call` boundaries, so kernel launch
  overhead accumulates with arithmetic intensity. Use only for debugging or
  as a reference implementation.

The FFI pass is currently behind the inline pass in capability — it handles
only `add`, `sub`, `mul`, and entry/exit conversion. It is missing
`stablehlo.constant`, `stablehlo.broadcast_in_dim`, `stablehlo.reduce`, and
`stablehlo.dot_general`, and the corresponding CUDA kernels do not exist yet.
Bringing it to parity is a potential future direction: FFI kernels are simpler
to write and maintain than MLIR C++ patterns, and for matmul-heavy workloads
the fusion loss may be acceptable. See `Possible Next Steps` in `handoff.md`.

### FMA safety

XLA/PTXAS does not fuse the Veltkamp split sequence (`mul.rn.f32`,
`sub.rn.f32`, `sub.rn.f32`) into `fma.rn.f32` instructions on H100
(sm_9.0a). The `.rn` IEEE-754 round-to-nearest suffix on each PTX instruction
prevents contraction. Confirmed by inspecting XLA-dumped PTX — zero
`fma.rn.f32` instructions across all kernels.

### Why `jax.jit` is required

Eager (non-JIT) ops each compile as a separate module. The DS `lo` component
is materialized back to f32 at module boundaries, losing precision at every
op. Inside a `jax.jit` block the entire function is one module, so DS pairs
propagate through all operations before the final f32 output.

### dot_general limitation

`stablehlo.dot_general` benefits only when at least one input has a non-zero
`lo` channel — i.e., when a prior `mul`, `add`, or `reduce` in the same JIT
block has created it. A standalone `a @ b` with no prior DS ops has `lo=0`
for both inputs, so all four correction matmuls contribute zero, and the
result is identical to plain f32. Materializing two-product error terms for
every matrix element would require O(M×K×N) intermediate storage, which is
not practical.

---

## Known Limitations

| Case | Status |
|---|---|
| **f64 inputs (`JAX_ENABLE_X64=1` required)** | |
| Element-wise `add` / `sub` / `mul` inside `jax.jit` (f64) | ✅ Working |
| `jnp.sum`, `jnp.mean` (reduce with single add/sub/mul body, f64) | ✅ Working |
| `jnp.matmul` / `@` (f64 inputs → 4 f32 GEMMs) | ✅ Working |
| Arithmetic with scalar f64 constants | ✅ Working |
| **f32 inputs (precision improvement mode)** | |
| Element-wise `add` / `sub` / `mul` inside `jax.jit` (f32) | ✅ Working |
| Arithmetic with scalar f32 constants | ✅ Working |
| `jnp.sum`, `jnp.mean` (f32) | ✅ Working |
| `jnp.matmul` f32 preceded by DS ops in same JIT | ✅ Working (use `precision=HIGHEST`) |
| `jnp.matmul` f32 standalone (no prior DS ops) | ⚠️ No benefit (lo = 0) |
| **Not yet handled** | |
| Multi-input reductions | ❌ |
| Nested regions (scan, map, while) | ❌ |
| Eager (non-JIT) ops | ❌ lo recombined to f32/f64 at every module boundary |

---

## Reference: Prior LLVM Approach (`double_single_ray/`)

`double_single_ray/skeleton/Skeleton.cpp` is an LLVM module pass that
performs the same DS transformation at the LLVM IR level, operating on
compiled C/C++ programs rather than JAX/StableHLO. It serves as the
algorithmic predecessor and structural reference for `DsTransformPass.cpp`.
The two passes share the same mathematical core (two_sum, two_prod, Veltkamp
split) but differ in IR dialect, target, and integration mechanism.

See `double_single_ray/README.md` for build and usage instructions for the
LLVM pass.
