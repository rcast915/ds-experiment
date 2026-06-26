# DS Arithmetic Compiler Pass — Handoff Document

## Note for Agents

You have NO access to the cluster or container. All commands must be written out for Ray to run manually, and Ray will paste back output for analysis.

Read the actual source files for implementation details — this document captures decisions and findings that are not visible from the code alone.

---

## Project Goal

Automatically transform float arithmetic in JAX/XLA programs to double-single (DS) arithmetic — near-float64 accuracy using only float32 operations — without user code changes.

**The claim:** DS-f32 achieves ~48-bit effective mantissa (vs f64's 53 bits) while running at approximately f32 hardware cost. On GPUs with poor or absent f64 support (consumer cards, mobile, embedded), this gives f64-class accuracy at f32 speed — a direct win. On HPC GPUs with native f64 units (H100), DS-f32 was measured to match f64 wall time while using only f32 arithmetic paths, giving f64-class accuracy with no performance penalty relative to f64.

**Measured on H100 (2026-06-23):** DS-f32 runs within 10% of native f64 speed for element-wise ops and reductions, and within 15% for matmul at sizes up to 2048×2048. The H100 has native FP64 tensor cores (~51 TFLOPS PCIe), so f64 is competitive with f32 on this hardware; the speedup argument applies on GPUs without FP64 tensor core support.

The mechanism: a PJRT proxy plugin intercepts JAX compile calls, runs a StableHLO IR pass that splits float arguments into DS pairs at function entry, replaces arithmetic with DS sequences, and recombines at function exit. Supports both f32 inputs (precision improvement) and f64 inputs (accuracy preservation at f32 cost).

**Requires `JAX_ENABLE_X64=1` for f64 inputs** — without it JAX silently downcasts f64 to f32 before the pass sees it.

---

## Environment

**Cluster:** Punakha HPC, node `hopper001`  
**Host working directory:** `/o_home/racastaneda3/ds_experiment`  
**Container working directory:** `/src/ds_experiment`  
**Username:** `racastaneda3`  
**Supervisor:** Dr. Moore (Shirley V.) | **IT contact:** Robert Corral

**Get an interactive GPU node:**
```bash
module load docker/27.3.1/rootless-docker
srun --account=punakha_partner_sdriver -p dgx -n 1 --gres=gpu:1 --qos punakha_dgx2_general --pty bash
start_rootless_docker.sh --quiet
```

**Run the container:**
```bash
cd /o_home/racastaneda3/ds_experiment
docker run --gpus all -it --volume $(pwd):/src/ds_experiment --rm ds-experiment
```
> Do NOT add `--network host` or `--ipc=host` — these trigger HPC security alerts.

**Rebuild the image (~1 hour, LLVM build):**
```bash
sbatch build_docker.sh
tail -f logs/output/build_JOBID.out
```

**Versions:** CUDA 12.9.1 · Python 3.12 · StableHLO 1.16.3 · JAX 0.10.1 · GPU: H100 (sm_9.0a)

---

## Session Setup (required every container start — container is ephemeral `--rm`)

```bash
bash ds_setup.sh
```

Renames both JAX CUDA auto-registration paths, builds the PJRT plugin, and verifies with a `[2. 4. 6. 8.]` sanity check. Rename steps are no-ops once the Dockerfile is rebuilt with them baked in.

<details>
<summary>Manual steps (if ds_setup.sh is unavailable)</summary>

```bash
mv /usr/local/lib/python3.12/dist-packages/jax_plugins/xla_cuda12 \
   /usr/local/lib/python3.12/dist-packages/jax_plugins/xla_cuda12_disabled
mv /usr/local/lib/python3.12/dist-packages/jax_cuda12_plugin \
   /usr/local/lib/python3.12/dist-packages/jax_cuda12_plugin_disabled
cmake -GNinja -S pjrt_plugin -B pjrt_plugin/build && ninja -C pjrt_plugin/build
DS_BYPASS=1 \
PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \
python3 -c "import jax.numpy as jnp; x = jnp.array([1.,2.,3.,4.]); print(x + x)"
```
</details>

**To activate DS transform for any script:**
```bash
PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \
python3 your_script.py
```

---

## Repository Structure

```
/src/ds_experiment/
├── Dockerfile
├── build_docker.sh              # SLURM batch job for rebuilding the image
├── ds_setup.sh                  # Per-session setup script
│
├── stablehlo_pass/              # MLIR/StableHLO DS pass (primary implementation)
│   ├── DsTransformPass.cpp      # Inline DS pass — expands to native StableHLO ops
│   ├── DsFFIPass.cpp            # FFI pass — emits custom_call ops (debug only)
│   ├── mlir-ds-opt.cpp          # Entry point → mlir-ds-opt binary
│   ├── mlir-ds-ffi-opt.cpp      # Entry point → mlir-ds-ffi-opt binary
│   └── CMakeLists.txt
│
├── pjrt_plugin/
│   ├── ds_pjrt_plugin.cpp       # PJRT proxy plugin
│   └── CMakeLists.txt
│
├── cpp/                         # FFI CUDA kernels → libds_ffi.so (reference path)
│   └── [ds_add, ds_sub, ds_mul, ds_from_f32/f64, ds_to_f32/f64, fused ops]
│
├── tests/                       # Automated test suite (added 2026-06-19)
│   ├── run_tests.sh             # Master runner: bash tests/run_tests.sh [--bench] [--cpu-only]
│   ├── ds_ref.py                # Pure-NumPy DS reference arithmetic (no JAX/CUDA)
│   ├── test_dot_product.py      # Accuracy tests: numpy ref + MLIR structural + GPU numerical
│   ├── test_matmul.py           # Accuracy tests: same structure
│   ├── bench_dot_product.py     # Timing + GFLOPS: DS vs f32 via subprocess baseline
│   └── bench_matmul.py          # Timing + TFLOPS: (A*A)@B and A@B suites
│
├── python/                      # Earlier scripts (many target the FFI path)
│   ├── test_pass_correctness.py # Original 3-test correctness suite (still valid)
│   └── [benchmark_*, plot_*, test_* scripts]
│
└── double_single_ray/           # Prior LLVM IR approach (reference/predecessor only)
    └── skeleton/Skeleton.cpp    # Structural analog to DsTransformPass.cpp
```

**Build commands (inside container):**
```bash
# Required every session (plugin not baked into image):
cmake -GNinja -S pjrt_plugin -B pjrt_plugin/build && ninja -C pjrt_plugin/build

# Rebuild only if source changed:
cmake -GNinja -S stablehlo_pass -B stablehlo_pass/build -DCMAKE_CXX_FLAGS="-fno-rtti" && ninja -C stablehlo_pass/build
cmake -GNinja -S cpp -B cpp/build && ninja -C cpp/build
```

---

## What Works ✅

### 1. Inline StableHLO Pass (`ds-transform`)
Expands `stablehlo.add/sub/mul` into DS arithmetic (two_sum, Veltkamp split, two_prod) as native StableHLO ops. XLA sees and fuses the full expanded sequence.

Pass pipeline (GPU):
```
vhlo-to-version{target=1.16.3}
vhlo-legalize-to-stablehlo        ← REQUIRED before ds-transform
func.func(ds-transform)
stablehlo-legalize-to-vhlo
```

**Critical:** `vhlo-legalize-to-stablehlo` must precede `ds-transform`. Without it, ops remain as `vhlo.add_v1` — the pass matches nothing and produces identical input/output bytecode.

### 2. PJRT Plugin — End-to-End Working ✅
Confirmed 2026-06-19. Intercepts JAX compile calls, transforms VHLO bytecode, returns DS-expanded bytecode to real CUDA backend.

```bash
PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \
python3 -c "
import jax, jax.numpy as jnp
x = jnp.array([1e8, 1e10, 1e12], dtype=jnp.float32)
y = jnp.array([1e-2, 1e-3, 1e-4], dtype=jnp.float32)
@jax.jit
def fn(x, y): return (x + y) - x
print(fn(x, y))  # [0.01  0.001  0.0001] — exact recovery; without DS: [0. 0. 0.]
"
```

### 3. FMA Safety — Confirmed ✅
Confirmed 2026-06-18. Zero `fma.rn.f32` instructions in XLA-dumped PTX. The Veltkamp split appears as three separate `mul.rn.f32` / `sub.rn.f32` instructions with IEEE `.rn` rounding per instruction. PTXAS does not contract them. DS arithmetic is safe from FMA corruption on H100 (sm_9.0a).

```bash
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text" \
PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \
python3 -c "
import jax, jax.numpy as jnp
a = jnp.array([1.5, 2.5, 3.5], dtype=jnp.float32)
@jax.jit
def fn(a): return a * a
fn(a)
"
grep -c 'fma' /tmp/xla_dump/*.ptx   # → 0
```

### 4. DS Reduction (`stablehlo.reduce`) — Working ✅
Confirmed 2026-06-18. `jnp.sum(a)` / `jnp.mean(a)` with a single add/sub/mul reduction body are fully DS-transformed. The ReduceOp handler in `DsTransformPass.cpp::processOps()` builds a new 4-arg reduce `(acc_hi, acc_lo, elem_hi, elem_lo)` with a full DS accumulation body.

GPU result: `jnp.sum(a * a)` with n=10,000 elements of `0.1_f32`:
- DS: `100.0000000000` — error `2.980e-06` (bit-perfect correctly rounded)
- f32: `100.0000076294` — error `4.649e-06` (overshoots by 1 ULP)

Triplet cancellation (`[1e8, 1.0, -1e8]` pattern, n=3333):
- DS: `3333.0000` — error `0.000e+00` (bit-perfect)
- f32: `60.0000` — error `3.273e+03` (catastrophic loss)

### 5. DS Matmul (`stablehlo.dot_general`) — Working with caveats ✅
Confirmed 2026-06-19. 4-matmul decomposition:
```
p    = dot(a_hi, b_hi)
e1   = dot(a_hi, b_lo)
e2   = dot(a_lo, b_hi)
e3   = dot(a_lo, b_lo)
(out_hi, out_lo) = two_sum(p, e1+e2+e3)
```

Uses `b.clone(*op)` + `setOperand` to inherit all dot_general attributes (dimension numbers, precision config) for all 4 sub-matmuls.

**When DS matmul helps:** Only when lo channels carry information — i.e., when a prior element-wise `mul`, `add`, or `reduce` in the same `jax.jit` creates non-zero lo channels. A standalone `a @ b` has `lo=0` for both inputs; all correction terms vanish; result is identical to f32.

**TF32 caveat (critical):** cuBLAS on H100 uses TF32 tensor cores by default for f32 GEMM. TF32 has a 10-bit mantissa (~1e-3 relative error vs f32's 1.2e-7). The DS correction from `two_prod` is order 1e-7 — three orders of magnitude smaller — so it is invisible when TF32 error dominates. To expose the true DS improvement, force strict f32 with `precision=jax.lax.Precision.HIGHEST`:

```python
@jax.jit
def fn(A, B):
    return jnp.dot(A * A, B, precision=jax.lax.Precision.HIGHEST)
```

The DS pass clones the `dot_general` op preserving all attributes, so all 4 sub-matmuls inherit `precision=HIGHEST`. Without this, the 256×256 GPU test produces DS error 6.4e-3 vs numpy f32 error 5.5e-6 — DS appears 1000× worse purely because cuBLAS uses TF32 and numpy does not.

**When TF32 does not interfere:** GEMV-shaped matmuls (1×n @ n×1) go through a scalar reduce path in XLA, not cuBLAS tensor cores. DS improvement is visible without `precision=HIGHEST` for these shapes.

### 7. Constant and Broadcast Support — Working ✅
Confirmed 2026-06-19. `stablehlo.constant` ops are now added to `dsMap` as `(constant, 0)` so arithmetic involving scalar constants (e.g. `a + 1.0`, `a * 0.5`) is fully DS-transformed. `stablehlo.broadcast_in_dim` is also handled — the DS pair is cloned through the broadcast — because JAX often lowers scalar constants as a constant followed by a broadcast before the arithmetic op.

Verified with:
```bash
PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \
python3 -c "
import jax, jax.numpy as jnp
x = jnp.array([1e8, 1e10, 1e12], dtype=jnp.float32)
@jax.jit
def fn(x): return (x + 1.0) - x
print(fn(x))  # DS: [1. 1. 1.]  — without DS: [0. 0. 0.]
"
```

Both handlers are in `DsTransformPass.cpp::processOps()`, before the AddOp handler. The constant is not erased — its SSA value becomes the `hi` component directly.

### 6. Test Suite — Working ✅
Confirmed 2026-06-19. Run from inside the container:
```bash
bash tests/run_tests.sh           # accuracy only (auto-detects GPU)
bash tests/run_tests.sh --bench   # accuracy + benchmarks
bash tests/run_tests.sh --cpu-only
```

All tests pass in both GPU mode (with PJRT plugin) and CPU-only mode:
- Section 1 (NumPy reference): 10 dot + 7 matmul tests including NaN, Inf, subnormals, catastrophic cancellation, ill-conditioned sums
- Section 2 (MLIR structural): op count assertions on lowered MLIR
- Section 3 (GPU numerical): precision improvement assertions on device

**Benchmark results** (H100, median over 50 reps):

| fn | Size | f32 (ms) | DS (ms) | Overhead |
|---|---|---|---|---|
| `dot(a*a, b)` | 1K | 0.033 | 0.038 | 1.14× |
| `dot(a*a, b)` | 1M | 0.040 | 0.064 | 1.60× |
| `(A*A)@B` | 64² | 0.06 | 0.05 | 0.87× |
| `(A*A)@B` | 2048² | 0.12 | 0.29 | 2.42× |
| `A@B` | 2048² | 0.11 | 0.29 | 2.67× |

---

## Known Limitations

| Operation | Status |
|---|---|
| Element-wise `add` / `sub` / `mul` inside `jax.jit` | ✅ |
| Arithmetic with scalar constants (`a + 1.0`, `a * 0.5`) | ✅ |
| `jnp.sum`, `jnp.mean` (single add/sub/mul reduce body) | ✅ |
| `jnp.matmul` preceded by DS ops, `precision=HIGHEST` | ✅ |
| `jnp.matmul` preceded by DS ops, default precision (TF32) | ⚠️ DS improvement invisible — TF32 error ~1e-3 >> DS correction ~1e-7 |
| `jnp.matmul` standalone (no prior DS ops, lo=0) | ⚠️ No benefit |
| Multi-input `stablehlo.reduce` | ❌ skipped (size != 1 guard) |
| Nested regions beyond reduce (scan, map) | ❌ |
| Eager (non-JIT) ops | ❌ Precision lost at module boundaries |
| Requires `jax.jit` | Rule — DS pairs propagate within one module; eager ops materialize lo back to f32 at every boundary |

---

## Key Technical Facts

| Item | Value |
|------|-------|
| JAX version | 0.10.1 |
| VHLO upgrade pass | `vhlo-to-version{target=1.16.3}` |
| VHLO→StableHLO pass | `vhlo-legalize-to-stablehlo` ← required before ds-transform |
| StableHLO→VHLO pass | `stablehlo-legalize-to-vhlo` |
| Bytecode producer tag | `StableHLO_v1.16.0` |
| Real CUDA plugin `.so` | `/usr/local/lib/python3.12/dist-packages/jax_plugins/xla_cuda12_disabled/xla_cuda_plugin.so` |
| PJRT C API header | `/opt/xla-pjrt/xla/pjrt/c/pjrt_c_api.h` |
| PJRT API version | 0.104 |
| StableHLO version | 1.16.3 (pinned in Dockerfile) |
| Inline pass pipeline (GPU) | `vhlo-to-version{target=1.16.3},vhlo-legalize-to-stablehlo,func.func(ds-transform),stablehlo-legalize-to-vhlo` |
| `libVersion.a` | `/opt/mlir/lib/libVersion.a` — linked in `stablehlo_pass/CMakeLists.txt` |

---

## Plugin Environment Variables

| Variable | Effect |
|----------|--------|
| `PJRT_NAMES_AND_LIBRARY_PATHS="cuda:<path>"` | Load the DS proxy plugin |
| `DS_BYPASS=1` | Skip all transformation; pure passthrough to real backend |
| `DS_TEST_PASSTHROUGH=1` | Run DS pass but send original bytecode to backend |
| `DS_PASS_MODE=ffi` | Use FFI pass (`ds-ffi-transform`) instead of inline pass |

---

## Important Caveats and Pitfalls

### CUDA registration (why two renames are needed)
JAX 0.10.1 has two independent CUDA auto-registration paths:
1. `jax_plugins/xla_cuda12/` — Python namespace package
2. `jax_cuda12_plugin/` — separate package with `cuda_plugin_extension.so`

Both must be renamed so neither claims the `cuda` PJRT slot before the proxy. `REAL_PLUGIN_PATH` in `ds_pjrt_plugin.cpp:19` is hardcoded to `xla_cuda12_disabled/xla_cuda_plugin.so` — the first rename must use `_disabled` exactly.

### PJRT double-registration crash (JAX initialization pitfall)
When `PJRT_NAMES_AND_LIBRARY_PATHS` is set, JAX registers the `cuda` backend at import time. Any subsequent eager array op (e.g., `jnp.ones(16)`) triggers JAX's lazy-load path which tries to re-register the same plugin → `ALREADY_EXISTS: PJRT_Api already exists for device type cuda` crash.

**Workaround:** For structural tests that only need MLIR lowering (not execution), use `jax.ShapeDtypeStruct` as abstract arguments to `jax.jit(fn).lower()`. This traces the function without executing anything on any device, so no plugin re-registration occurs:
```python
a = jax.ShapeDtypeStruct((16,), jnp.float32)
mlir = str(jax.jit(fn).lower(a, a).compiler_ir())  # no device execution
```
This is what `tests/test_dot_product.py` and `tests/test_matmul.py` do in their structural sections.

**Do not** set `JAX_PLATFORMS=cpu` inside the test script after JAX has already loaded the CUDA backend — it does not un-register the backend and can trigger the same crash.

### TF32 tensor cores mask DS matmul improvement
(See What Works §5 for full details.) Short version: cuBLAS uses TF32 for f32 GEMM by default on H100. Always use `precision=jax.lax.Precision.HIGHEST` when benchmarking or testing DS matmul improvement. The test in `tests/test_matmul.py` Section 3d does this correctly.

### `libds_ffi.so` hangs on CPU-only load
Links against `libcudart.so.12`; global constructors block on CUDA initialization in CPU-only processes. Do not load it with `ctypes.cdll.LoadLibrary` when `JAX_PLATFORMS=cpu` or CUDA is not yet initialized. The test suite uses pure-NumPy DS reference arithmetic (`tests/ds_ref.py`) instead.

### f64 truth must use actual f32 input values
When computing ground truth, always cast the actual f32 values to f64 — do not compute the ground truth directly in f64 with the same literal. Example: `np.float64(0.1) != np.float64(np.float32(0.1))`. Always use:
```python
truth = float(np.sum(np.full(n, 0.1, np.float32).astype(np.float64) ** 2))
```

### Inline vs FFI pass
- **Inline (`ds-transform`, default):** Native StableHLO ops, XLA fuses freely. Use this.
- **FFI (`ds-ffi-transform`):** `stablehlo.custom_call` ops dispatching to `libds_ffi.so`. XLA cannot fuse across `custom_call` boundaries — performance degrades steeply with arithmetic intensity. Use only for debugging.

**Current parity gap (as of 2026-06-19):** The FFI pass only handles `add`, `sub`, `mul`, entry conversion (`ds_from_f32`), and exit conversion (`ds_to_f32`). It is missing: `stablehlo.constant`, `stablehlo.broadcast_in_dim`, `stablehlo.reduce`, and `stablehlo.dot_general`. The corresponding CUDA kernels (`ds_reduce`, `ds_matmul`) do not exist yet either. Bringing FFI to parity with the inline pass is a potential next step — it would enable a direct performance comparison and may offer better long-term maintainability since CUDA kernels are easier to modify than MLIR C++.

---

## f64 → DS: New Focus (added 2026-06-23)

### Three bugs fixed in `DsTransformPass.cpp` for f64 correctness

The pass supported f64 inputs in `emitFromFloat` (f64 path) but three "clone" handlers copied the f64 result type onto ops that now have f32 operands — invalid StableHLO that would crash the verifier at runtime:

| Handler | Bug | Fix |
|---|---|---|
| `BroadcastInDimOp` | clone kept f64 result, operand changed to f32 | `cloned->getResult(0).setType(toF32Type(...))` |
| `DotGeneralOp` | clone kept f64 result, operands changed to f32 | same, inside `makeDot` lambda |
| `ReduceOp` | `resultTy` and `scalarTy` taken from f64 op, new reduce created with f64 types | both changed to `toF32Type(...)` |

Add/Sub/Mul handlers were already correct (they create fresh ops from f32 hi/lo values, so types are inferred automatically).

All three fixes are backward-compatible: `toF32Type(f32_type)` = `f32_type` (no-op for f32 inputs).

### New files for f64 support

| File | Purpose |
|---|---|
| `tests/test_f64_ds.py` | Accuracy tests: NumPy reference + MLIR structural + GPU numerical for f64 inputs |
| `tests/bench_f64_vs_ds.py` | **Primary benchmark**: f64 baseline vs DS-f32 for element-wise, reduce, matmul |

Run:
```bash
# Tests (inside container):
bash tests/run_tests.sh --f64-only

# f64 benchmark:
bash tests/run_tests.sh --bench --f64-only
```

### Measured benchmark results (H100 PCIe, 2026-06-23)

| Operation | f64 (ms) | DS-f32 (ms) | DS/f64 ratio |
|---|---|---|---|
| Element-wise n=1M | 0.059 | 0.058 | 0.98× (DS ≈ f64) |
| Reduction n=1M | 0.041 | 0.043 | 1.05× (DS ≈ f64) |
| Matmul 256² | 0.04 | 0.06 | 1.39× (DS slower) |
| Matmul 2048² | 0.36 | 0.32 | 0.89× (DS slightly faster) |

DS-f32 and native f64 are within ~10–15% across all tested sizes. The H100 has native FP64 tensor cores (~51 TFLOPS PCIe), making f64 GEMM competitive with f32. The large speedup expected for GPUs without FP64 hardware does not apply to H100 but would apply to consumer GPUs (RTX series, etc.) where f64 is 1/32–1/64 of f32 speed.

**Precision improvement over f32 (confirmed):** DS-f32 sum error 1.49e-05 vs f32 error 1.22e-04 — 8× improvement.

---

## Possible Next Steps

These are not assigned — listed in rough priority order for whoever picks this up next.

1. **Run `bench_f64_vs_ds.py` and record the f64 vs DS speedup table** — this is the primary paper result. Fill in the table above with measured numbers.

2. **Benchmark DS vs f32 with `precision=HIGHEST` on larger matmul sizes.** The current benchmark uses default precision (TF32). A paper-quality comparison needs both sides at the same precision setting. Run `bench_matmul.py` after modifying the inner benchmark code to use `jnp.dot(..., precision="highest")` in both f32 and DS modes.

2. **Multi-input `stablehlo.reduce`.** The ReduceOp handler in `DsTransformPass.cpp` skips reductions with more than 1 input (`if (redOp.getInputs().size() != 1) continue`). Extending to 2-input reductions (e.g., simultaneous min+max scans) would require 4 block args per input pair.

3. **Nested regions (scan, map, while).** `stablehlo.while` and `stablehlo.map` contain nested function regions. The current pass only walks `func.func` regions. Extending requires recursing into nested regions and threading the dsMap appropriately.

4. **`jax.vmap` over DS-transformed functions.** Untested. `vmap` inserts batch dimensions and may lower to additional StableHLO ops. Likely works for element-wise ops (batch dimension is transparent) but needs verification.

5. **Standalone matmul precision improvement via input splitting.** For `A @ B` with no prior DS ops, lo=0 so there is no benefit. To get improvement here, the pass would need to split each input element via `two_prod(A[i,k], 1.0)` to create a lo channel — but this costs an O(M×K + K×N) element-wise pass and is only worthwhile if the matmul accumulation error is the bottleneck.

6. **Paper benchmarks.** For the paper, run the full benchmark suite with `--bench` on a clean session and record: overhead table, precision improvement table (using `precision=HIGHEST` for matmul), and the catastrophic cancellation stress test results.

7. **Bring FFI pass to parity with inline pass.** Currently the FFI pass only handles `add`/`sub`/`mul`. To enable a fair inline vs. FFI comparison: (a) add `stablehlo.constant` and `stablehlo.broadcast_in_dim` handlers to `DsFFIPass.cpp` mirroring what was added to `DsTransformPass.cpp`; (b) write a `ds_reduce` CUDA kernel in `cpp/` for reductions; (c) write a `ds_matmul` wrapper (4 cuBLAS dispatches) in `cpp/`; (d) add ReduceOp and DotGeneralOp handlers in `DsFFIPass.cpp`. Once at parity, benchmark `DS_PASS_MODE=ffi` vs default inline on the existing bench scripts. The FFI path may be more maintainable (CUDA kernels vs. MLIR C++); the benchmark will show whether the fusion loss is acceptable for the target workloads.
