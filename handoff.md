# DS Arithmetic Compiler Pass — Handoff Document

## Note for Agents

You have NO access to the cluster or container. All commands must be written out for Ray to run manually, and Ray will paste back output for analysis.

Read the actual source files for implementation details — this document captures decisions and findings that are not visible from the code alone.

---

## Project Goal

Automatically transform float arithmetic in JAX/XLA programs to double-single (DS) arithmetic — near-float64 accuracy using only float32 operations — without user code changes.

**The claim:** DS-f32 achieves ~48-bit effective mantissa (vs f64's 53 bits) while running at approximately f32 hardware cost. On GPUs with poor or absent f64 support (consumer cards, mobile, embedded), this gives f64-class accuracy at f32 speed — a direct win. On HPC GPUs with native f64 units (H100), DS-f32 was measured to match f64 wall time while using only f32 arithmetic paths, giving f64-class accuracy with no performance penalty relative to f64.

**Measured on H100 (2026-06-26, `h100_results.txt`):** DS-f32 runs within 10% of native f64 speed for element-wise ops and reductions at large sizes. For matmul, DS-f32 is within ~10% of f64 at 1024²–2048² and 1.13× faster than f64 at 2048×2048; at smaller sizes DS is slower (0.71× at 256²) due to 4 kernel launches vs 1 at latency-dominated sizes. The H100 has native FP64 tensor cores (~51 TFLOPS PCIe, confirmed: 48.6 TFLOPS measured at 2048²), so f64 is competitive with f32 on this hardware; the speedup argument applies on GPUs without FP64 tensor core support.

**Measured on L40S / Bridges-2 (2026-06-26, `l40_results.txt`):** The L40S has no native FP64 tensor cores (~1/64 f32 throughput for f64). DS-f32 matmul is **up to 12.87× faster than native f64** at 2048×2048 (DS: ~18 TFLOPS vs f64: ~1.4 TFLOPS), while delivering the same numerical accuracy. All 3 test suites pass on L40S. Element-wise and reduction workloads show little speedup (memory-bandwidth-bound — both saturate bandwidth equally); the speedup is concentrated in compute-bound matmul where the f64 CUDA-core bottleneck is most severe.

The mechanism: a PJRT proxy plugin intercepts JAX compile calls, runs a StableHLO IR pass that splits float arguments into DS pairs at function entry, replaces arithmetic with DS sequences, and recombines at function exit. Supports both f32 inputs (precision improvement) and f64 inputs (accuracy preservation at f32 cost).

**Requires `JAX_ENABLE_X64=1` for f64 inputs** — without it JAX silently downcasts f64 to f32 before the pass sees it.

---

## Environment

### Punakha HPC (primary development)
**Node:** `hopper001` · **GPU:** H100 (sm_9.0a)  
**Host working directory:** `/o_home/racastaneda3/ds_experiment`  
**Container working directory:** `/src/ds_experiment`  
**Username:** `racastaneda3`  
**Supervisor:** Dr. Moore (Shirley V.) | **IT contact:** Robert Corral

### Bridges-2 / PSC (L40S testing, 2026-06-26)
**GPU:** L40S (no native FP64 tensor cores)  
**Project allocation:** `cis260064p` (`/ocean/projects/cis260064p`)  
**Username:** `rcastane`  
**Container:** Singularity/Apptainer — use `submit_l40s.sh` to pull image and run tests  
**Image:** `docker://rcast915/ds-experiment:latest` (Docker Hub)

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
├── h100_results.txt             # Raw test suite output: H100 (2026-06-26) ← authoritative
├── l40_results.txt              # Raw test suite output: L40S (2026-06-26) ← authoritative
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
│   ├── test_f64_ds.py           # Accuracy tests: f64 inputs via DS plugin
│   ├── bench_dot_product.py     # Timing + GFLOPS: DS vs f32
│   ├── bench_matmul.py          # Timing + TFLOPS: (A*A)@B and A@B suites
│   └── bench_f64_vs_ds.py       # Timing: f64 baseline vs DS-f32 (primary research result)
│
├── python/                      # Earlier scripts and experiment narrative records
│   ├── test_pass_correctness.py # Original 3-test correctness suite (still valid)
│   ├── ds_results_h100.txt      # Narrative: FFI-era GPU experiments on H100
│   ├── ds_results_compiler_pass_h100.txt  # Narrative: compiler pass results on H100
│   ├── ds_results_l40s.txt      # Narrative: compiler pass results on L40S
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
Expands `stablehlo.add/sub/mul` into DS arithmetic (two_sum, Veltkamp split, two_prod) as native StableHLO ops. XLA sees and fuses the full expanded sequence. Also handled: `divide`, `sqrt`, `negate`, `abs`, `compare`, `select`, `maximum`, `minimum` (ported from `double_single_ray/.../double-single-lib`) and `exponential`, `log` (ported from `double-single-libm`, see section 8).

Pass pipeline (GPU):
```
vhlo-to-version{target=1.16.3}
vhlo-legalize-to-stablehlo        ← REQUIRED before ds-transform
inline                            ← removes func.call boundaries (JAX outlines jnp.where)
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

### 8. DS exp / log — Working ✅ (ported 2026-10-04, verified on H100 2026-10-08)

`stablehlo.exponential` → `emitDsExp`, `stablehlo.log` → `emitDsLog` in `DsTransformPass.cpp`, ported from `double-single-libm`'s `expds` (`exp.c`) and `logds` (`log.c`). The lookup tables are `stablehlo_pass/libmds/exp_table.h` and `log_table.h`, copied unmodified from that library and `#include`d by the pass.

**Verified on the host (no container needed):** `tests/ds_ref.py`'s `ds_exp` / `ds_log` are a NumPy transcription of the same sequence and match the C library bit-for-bit on 1.19M random double-single inputs (library compiled with gcc, compared through ctypes). The only intended difference is `log(0)`, below.

**Verified on H100 (2026-10-08):** `bash tests/run_tests.sh --bench` passes all 7 suites. `test_ds_exp_log.py`'s GPU results match the NumPy reference bitwise; `exp(a)*exp(-a) - 1` and `log(exp(a)) - a` come out at ~1e-14. Still not run: that file's `DS_RETURN_PAIRS=1` pair-accuracy section, and anything on L40S.

**Black-Scholes on H100 (2026-10-08, f64 inputs, max relative price error vs NumPy f64 / median ms):**

| Batch | f64 ms | DS ms | f32 ms | f64/DS | f64 relerr | DS relerr | f32 relerr |
|---|---|---|---|---|---|---|---|
| 1,024 | 0.043 | 0.056 | 0.041 | 0.76× | 2.3e-14 | 3.2e-11 | 1.2e-03 |
| 65,536 | 0.043 | 0.083 | 0.042 | 0.52× | 4.6e-13 | 1.9e-10 | 7.8e-03 |
| 1,048,576 | 0.078 | 0.266 | 0.062 | 0.29× | 1.1e-11 | 2.3e-10 | 7.9e-03 |
| 16,777,216 | 0.349 | 2.406 | 0.209 | 0.14× | 1.2e-11 | 3.6e-10 | 9.8e-03 |

DS is ~10^7× more accurate than native f32 and within ~30× of native f64, but slower than native f64 at every size on this GPU.

**Black-Scholes on L40S (Bridges-2, 2026-10-08, commit 82b3cf4, same columns):**

| Batch | f64 ms | DS ms | f32 ms | f64/DS | f64 relerr | DS relerr | f32 relerr |
|---|---|---|---|---|---|---|---|
| 1,024 | 0.400 | 0.358 | 0.400 | 1.12× | 2.3e-14 | 3.2e-11 | 1.2e-03 |
| 65,536 | 0.405 | 0.409 | 0.426 | 0.99× | 4.8e-13 | 1.9e-10 | 7.8e-03 |
| 1,048,576 | 0.750 | 0.702 | 0.376 | 1.07× | 1.1e-11 | 2.3e-10 | 7.9e-03 |
| 16,777,216 | 5.234 | 6.709 | 1.049 | 0.78× | 1.2e-11 | 3.6e-10 | 9.8e-03 |

DS errors are identical to H100's. No speedup over native f64 on L40S either: parity up to 1M elements, 0.78× at 16M. This run had a ~0.35-0.4 ms floor on every call (the 2026-06-26 L40S run had ~0.05 ms), so rows at or near that floor measure dispatch overhead, not arithmetic; only the 16M row is clearly above it. Same run, f64-vs-DS matmul: 1.43× / 1.62× / 4.46× / 9.55× at 256 / 512 / 1024 / 2048 (default precision; 12.87× in June, with DS at 1.33 ms now vs 0.95 ms then). All 7 suites, the three `DS_RETURN_PAIRS=1` pair-accuracy runs (exp 1.2e-14, log 5.2e-15) and `diag_f64_ops.py` (every op ~1e-14; cancellation cases ~1e-11) passed.

**f64 split bug found and fixed on the way (2026-10-08).** XLA folds `convert(convert(v, f32), f64)` back to `v`. The optimization barrier in `emitFromFloat` only covered the `lo` computation; `hi` was widened again, unbarriered, in `emitToFloat` at `func.return`, where the fold turned `f64(hi) + f64(lo)` into `v + f64(lo)` — every f64 result off by `lo` (f32-level). It also hit the small helper modules JAX compiles ahead of a jitted function, which the plugin transforms too, so inputs arrived already shifted by `lo`. Fix: `hi` is now the barrier's output, so every use goes through it; and `func.return` passes an untouched function argument straight through. Before the fix Black-Scholes DS relerr was ~1e-5 and the f64 `sum` test error 4.5e-5; after, ~3e-10 and 2.6e-12. `tests/diag_f64_ops.py` (per-op f64 accuracy) and `tests/diag_f64_split.py` (raw split values + HLO dumps) are the diagnostics that found it.

How the library's C maps to StableHLO:
- Branches (special cases, log's `zh >= sqrt(2)` step) → `stablehlo.select`; both arms are always computed.
- `memcpy` float↔int and shifts → `stablehlo.bitcast_convert`, `shift_left`, `shift_right_arithmetic`, `shift_right_logical`, `and` on `i32` tensors.
- Table reads → `stablehlo.gather` of size-1 slices from a rank-1 constant, with the index-vector dimension implied (`index_vector_dim == rank(indices)`). 2 gathers for exp, 3 for log.
- FMA-based `__libmds_mul_2_11` → `emitTwoProd` (Veltkamp), same substitution as divide.

Deliberate differences from the library:
- **`log(0)` returns `(-Inf, 0)`.** The library's comment says "Return -inf" but its code computes `1.0f / (0.0f * 0.0f)` = `+Inf` (confirmed by running it).
- `k = ssxh - SHIFTER` in exp is replaced by `convert(kint)` (same value, exact), and one `fast_two_sum(c2, q)` by `two_sum`. Reason: `(x + C) - C` is a shape XLA's algebraic simplifier may reassociate to `x + (C - C)`. `emitTwoSum` now also swaps its operands when the first is a constant, for the same reason — this applies to existing ops too (e.g. `1.0 - x`). **This is a precaution based on reading XLA's simplifier rules, not something observed in a dump**; worth confirming against optimized HLO.
- The log table index is clamped to 181, because special-case inputs still evaluate the main path.

Measured accuracy of the algorithms themselves (NumPy reference, relative error vs f64): exp ≤ 2^-42.9 for results in the normal range, degrading toward f32 level for results below ~1e-30 (the `lo` word goes subnormal); log ≤ 2^-47.5 for arguments ≥ 2, but up to 2^-38.5 for arguments within ~1% of 1.

**Black-Scholes:** `tests/bench_blackscholes.py` is a JAX port of the PARSEC kernel in `blackscholes/blackscholes_torch.py` (checked against `blackscholes/reference_scalar.py`: max abs diff 4e-14). It times native f64, DS-f32, and native f32, and reports price error against NumPy f64. Every float op in the kernel now has a DS handler. `tests/run_tests.sh --bench` runs it.

### 9. DS matmul is only f32-accurate — measured 2026-10-09 (H100)

`tests/diag_f64_matmul.py`, `A @ B` on random normal n×n **f64** inputs, relative Frobenius error vs NumPy f64 / median ms:

| n | f64 | DS (default) | DS (HIGHEST) | f32 (default) | f32 (HIGHEST) |
|---|---|---|---|---|---|
| 256 | 5.6e-16 / 0.044 | 2.9e-04 / 0.065 | 1.5e-07 / 0.073 | 2.9e-04 / 0.046 | 1.5e-07 / 0.047 |
| 1024 | 1.1e-15 / 0.101 | 2.9e-04 / 0.113 | 5.7e-07 / 0.275 | 2.9e-04 / 0.077 | 5.7e-07 / 0.118 |
| 2048 | 1.6e-15 / 0.353 | 2.9e-04 / 0.357 | 8.1e-07 / 1.536 | 2.9e-04 / 0.116 | 8.1e-07 / 0.410 |

DS error equals plain f32's to two digits at both precision settings. The four f32 GEMMs round every product and partial sum to f32 and nothing captures that rounding; the hi/lo cross terms only carry the inputs' lo words. At default precision the H100 runs those GEMMs on TF32 tensor cores, so an f64 matmul routed through DS comes back with ~3e-4 relative error.

Consequences: (1) every "DS-f32 vs f64" matmul speedup in this repo (12.87× / 9.55× on L40S, ~1× on H100) compares an f32-accurate result against an f64-accurate one — it is not evidence that DS is faster than f64 at comparable accuracy; README text saying "same numerical accuracy" is wrong for matmul. (2) With the plugin active, f64 user code containing a matmul silently loses ~8 to 12 digits. Elementwise ops and reductions are unaffected (~1e-14, `diag_f64_ops.py`). Not yet measured on L40S, where default precision uses a different dispatch path. Options not yet decided: leave f64 `dot_general` untransformed (correct, no speedup), or an error-free scheme such as Ozaki's (accurate, many more GEMMs).

**Decided and implemented 2026-10-31 (not yet built/run):** f64 `dot_general` now stays native f64 by default. Its DS operands are recombined to f64 (`emitToFloat`), the original op is kept, and its result is split back to DS for downstream ops; returned unchanged if it is a function result (`nativeValues`). `DS_F32_MATMUL=1` restores the 4-GEMM decomposition for f64 inputs. f32 matmuls are unchanged. Tests: `tests/test_f64_ds.py` 2c (structure, both modes) and 3c (GPU: f64 `(A*A)@B` must be < 1e-12); `tests/diag_f64_matmul.py` now shows default vs 4-GEMM columns; `tests/bench_f64_vs_ds.py` times the 4-GEMM mode explicitly and labels it f32-accurate. Remaining README matmul tables are historical and labelled as not accuracy-matched.

### 10. FMA upper bound — measured 2026-10-23 (L40S, job 49691105)

`tests/bench_fma_bound.py` with the pass's `DS_FAKE_FMA_TIMING=1` (error-free product at the cost of one FMA; results deliberately wrong, timing only). Black-Scholes, median ms:

| Batch | f64 | DS | DS+FMA* | f32 | f64/DS | f64/DS+FMA* | DS/DS+FMA* |
|---|---|---|---|---|---|---|---|
| 1,048,576 | 0.376 | 0.280 | 0.189 | 0.058 | 1.34× | 1.99× | 1.48× |
| 16,777,216 | 4.820 | 6.371 | 6.240 | 0.726 | 0.76× | 0.77× | 1.02× |
| 67,108,864 | 19.005 | 25.211 | 26.686 | 2.692 | 0.75× | 0.71× | 0.94× |

StableHLO ops in the kernel: native 135, DS 3110, DS+FMA* 2073.

An FMA cuts a third of the ops and helps 1.48× at 1M, where DS already beats f64 (1.34×; this node had no per-call floor, f32 at 0.058 ms), but does nothing at 16M and 64M. So at large batches DS is not limited by f32 arithmetic. Working hypothesis, not yet checked: XLA splits the ~3,000-op DS computation into several kernels and passes intermediates through GPU memory; at 1M those intermediates fit in the L40S's L2 cache, at 16M+ they don't. If so, the lever is fusion / memory traffic, not FMA, and adding an FMA to XLA is not worth it for speed on this kernel. Next check: count the kernels (fusions) in XLA's optimized HLO for DS Black-Scholes at 16M.

### 11. Nsight profile of DS Black-Scholes (H100, 2026-10-26) and the reduce_precision split

`bash tests/run_nsight.sh <label>` (Nsight Compute only; `nsys` is not in the image). 16M options, one call:

| | Native f64 | DS |
|---|---|---|
| Kernels | 1 (`loop_select_fusion`, 327 µs, 79% DRAM) | 29: 9 full-size + 20 single-thread |
| GPU time | 327 µs | 2,717 µs |

DS breakdown: two arithmetic-bound kernels 1,617 µs (60%); two memory-bound kernels 707 µs (26%); five `wrapped_convert` kernels 315 µs (12%); twenty single-thread kernels 79 µs (3%). Max 80 registers/thread, so no sign of register pressure. The split hypothesis is confirmed, but on the H100 arithmetic dominates; the convert kernels and single-thread kernels came from the optimization barrier added in the section 8 fix (it stopped input conversions fusing and constants folding).

Change (2026-10-26, verified on H100 the same day): `emitFromFloat`'s f64 path no longer uses the barrier. It computes `r = reduce_precision(v, 8 exp, 23 mant)`, `hi = f32(r)`, `lo = f32(v - r)`; XLA folding `convert(convert(r))` back to `r` is now exact, so no barrier is needed. Difference from a true conversion: values below ~1.2e-38 flush to hi = 0, lo = f32(v). Results: accuracy unchanged (f64 sum 1.7e-12; `diag_f64_ops.py` identical row for row; one structural test counted `reduce_precision` as a reduce, fixed in `tests/test_f64_ds.py`). Black-Scholes DS 2.406 → 2.123 ms at 16M (12% faster), 0.266 → 0.215 ms at 1M (19%); f64/DS 0.14× → 0.16× at 16M. Nsight, 16M, one call:

| DS kernel | µs | DRAM % | SM % | regs | limited by |
|---|---|---|---|---|---|
| `loop_add_minimum_fusion` | 131 | 88 | 53 | 32 | memory |
| `loop_add_convert_subtract_fusion` | 664 | 78 | 71 | 64 | memory |
| `loop_add_fusion` | 1,630 | 23 | 75 | 86 | arithmetic |

29 kernels → 3; 2,717 → 2,425 µs. XLA also merged the two former arithmetic kernels into one. On the H100, two-thirds of DS time is now one arithmetic-bound kernel that alone takes 5× native f64's whole run, so no fusion change can make DS win there; a third is two memory-bound kernels, i.e. two intermediate handoffs. Unverified guess: the cuts fall at the exp/log table gathers (the first kernel's name points at log's `minimum` index clamp).

### 12. L40S after the reduce_precision split — DS beats native f64 on Black-Scholes (2026-10-26, job 49853294)

`tests/run_full.sh` on Bridges-2 L40S; all suites, pair-accuracy runs and diagnostics pass. Black-Scholes, median ms:

| Batch | f64 | DS | f32 | f64/DS | f64 relerr | DS relerr | f32 relerr |
|---|---|---|---|---|---|---|---|
| 1,024 | 0.057 | 0.064 | 0.058 | 0.88× | 2.3e-14 | 3.2e-11 | 1.2e-03 |
| 65,536 | 0.088 | 0.082 | 0.056 | 1.07× | 4.8e-13 | 1.9e-10 | 7.8e-03 |
| 1,048,576 | 0.378 | 0.210 | 0.058 | 1.80× | 1.1e-11 | 2.3e-10 | 7.9e-03 |
| 16,777,216 | 4.817 | 3.802 | 0.727 | 1.27× | 1.2e-11 | 3.6e-10 | 9.8e-03 |

At 16M, DS went from 6.37 ms (2026-10-23, barrier split, 0.76× of f64) to 3.80 ms (1.27×), a 40% cut from removing the barrier alone; native f64 was 4.82 ms both times, so the runs are comparable. On the H100 the same change gave only 12%, consistent with the L40S being bound by memory traffic (lower bandwidth) where the H100 is bound by arithmetic. Accuracy is unchanged. This node had no per-call floor (f32 at 0.058 ms).

Matmul on L40S (`diag_f64_matmul.py`): DS is f32-accurate here too, 2.9e-4 at default precision (TF32) and 4.7e-7 at HIGHEST; speed vs f64 12.69× default / 6.96× HIGHEST at 2048². The matmul caveat in section 9 holds on both GPUs.

### 13. FMA upper bound after the barrier fix, with repeats (L40S, 2026-10-31, job 50045542)

`tests/bench_fma_bound.py`, 3 independent runs per cell (median of run medians; spread 0–1% everywhere). Black-Scholes, ms:

| Batch | f64 | DS | DS+FMA* | f32 | f64/DS | f64/DS+FMA* | DS/DS+FMA* |
|---|---|---|---|---|---|---|---|
| 1,048,576 | 0.377 | 0.199 | 0.167 | 0.057 | 1.89× | 2.26× | 1.19× |
| 16,777,216 | 4.817 | 3.805 | 3.880 | 0.726 | 1.27× | 1.24× | 0.98× |
| 67,108,864 | 19.005 | 17.610 | 18.412 | 2.691 | 1.08× | 1.03× | 0.96× |

Ops: native 135, DS 3,094, DS+FMA* 2,057. The section 12 result reproduces (16M: 3.805 vs 3.802 ms) and holds at 64M (1.08×). An FMA still helps only at 1M (+19%); at 16M and 64M it changes nothing, so large batches remain limited by something other than f32 arithmetic, most likely memory traffic through the two remaining kernel boundaries. Not measurable with Nsight on Bridges-2 (performance counters restricted); next step is listing the kernels and their array sizes from XLA's compiled program.

### 14. Standalone DS matmul kernel (option A) — H100 check (2026-10-31)

`cpp/ds_gemm/` (kernel + benchmark), run with `bash tests/run_ds_gemm.sh <label>`; needs only nvcc and cuBLAS. Per term: compensated multiply-add (Ogita–Rump–Oishi Dot2 extended to DS inputs) with real FMA, ~12 f32 instructions; low word renormalized every 8 terms. Tiling 64×64×8, 256 threads, 4×4 outputs per thread; 80 registers, no spills.

H100 (n×n, f64 inputs; error vs cuBLAS DGEMM, whose own error vs long double is 6e-16 to 3e-15):

| n | DGEMM ms | SGEMM ms (err) | DS ms + split ms | DS err | DGEMM/DS |
|---|---|---|---|---|---|
| 1024 | 0.051 | 0.065 (5.7e-7) | 0.590 + 0.015 | 3.9e-14 | 0.08× |
| 2048 | 0.300 | 0.346 (8.1e-7) | 4.445 + 0.056 | 5.4e-14 | 0.07× |
| 4096 | 2.302 | 2.851 (1.2e-6) | 34.858 + 0.195 | 7.7e-14 | 0.07× |

Accuracy is f64-class (5e-14 at 2048²), seven orders better than SGEMM. Speed on H100 is hopeless as expected (FP64 tensor cores, ~60 TFLOPS), but the kernel itself is efficient: ~12.4 instructions × n³ in 4.445 ms ≈ 24 T instr/s, roughly 70% of the H100's FP32 issue rate (estimate from the nominal 67 TFLOPS FP32 peak). Projection for L40S (nominal 91.6 TFLOPS FP32) at a similar efficiency: ~3–3.5 ms at 2048² against DGEMM's 12.2 ms, i.e. ~3×; an estimate, the L40S run decides.

**L40S result (2026-10-31, job 50066814) — GO.**

| n | DGEMM ms | SGEMM ms (err) | DS ms + split ms | DS err | DGEMM/DS |
|---|---|---|---|---|---|
| 256 | 0.072 | 0.016 (9.9e-8) | 0.076 + 0.010 | 1.9e-14 | 0.83× |
| 512 | 0.250 | 0.022 (2.1e-7) | 0.143 + 0.010 | 2.7e-14 | 1.63× |
| 1024 | 1.916 | 0.088 (2.1e-7) | 0.550 + 0.022 | 3.9e-14 | 3.35× |
| 2048 | 15.222 | 0.390 (4.1e-7) | 4.476 + 0.195 | 5.4e-14 | 3.26× |
| 4096 | 110.224 | 2.782 (6.6e-7) | 34.580 + 0.840 | 7.7e-14 | 3.11× |

Clears the go/no-go (≥1.5× at 2048², err < 1e-12) with 3.26× at 5.4e-14. The DS kernel runs at about the same absolute speed as on the H100 (~3.9 TFLOP/s-equivalent), roughly half the L40S's nominal FP32 issue rate, so tuning (larger per-thread tiles, vector loads, double buffering) has room. Note cuBLAS DGEMM here (15.2 ms at 2048²) is slower than JAX/XLA's f64 matmul measured earlier on L40S (12.2 ms); compare end-to-end JAX numbers against the JAX baseline once integrated.

### 15. DS matmul kernel integrated into the pass (2026-10-31; verified on H100 the same day)

- `cpp/ds_gemm/ds_gemm_kernel.cuh`: the kernel, shared by the benchmark and the handler; writes f64 or a (hi, lo) pair.
- `cpp/ds_gemm/ds_gemm_ffi.cu` → `cpp/ds_gemm/build/ffi/libds_gemm_ffi.so` (built by `ds_setup.sh` step 2b): typed XLA FFI handler `DsGemm` taking (a_hi, a_lo, b_hi, b_lo) f32 and returning (c_hi, c_lo) f32, on XLA's stream.
- `pjrt_plugin/ds_pjrt_plugin.cpp`: at init, finds the backend's PJRT FFI extension, dlopens the library, calls `register_handler` for target `ds_gemm` (platform names tried: CUDA, cuda, gpu), and on success `setenv("DS_GEMM_KERNEL_AVAILABLE", "1")` so mlir-ds-opt (spawned with `environ`) knows. `DS_GEMM_KERNEL=0` skips it. The FFI extension header is vendored in `pjrt_plugin/third_party/` (the image only has pjrt_c_api.h).
- Pass: f64 `dot_general` of the plain form A[M,K]@B[K,N] (rank 2, no batch dims, contracting 1×0, static shapes) → `stablehlo.custom_call @ds_gemm` (api_version 4 = typed FFI, row-major operand/result layouts), result kept as a DS pair. Otherwise native f64 (section 9). `DS_F32_MATMUL=1` still selects the old 4-GEMM path.
- Tests: `tests/test_f64_ds.py` 2c (kernel call emitted; batched falls back; `(A*A)@B` feeds the call) and 3c (GPU accuracy < 1e-12, labels which path ran); `tests/diag_f64_matmul.py` has `DS` (kernel) and `DS native` columns.
- **H100 result (`run_full.sh h100-kernel`):** compiled first time; plugin logs `DS matmul kernel registered (target ds_gemm, platform CUDA)`; all 7 suites pass; f64 `(A*A)@B` through JAX via the kernel: 1.96e-14. Matmul diagnostic, f64 inputs, through JAX:

  | n | native f64 | DS kernel | DS native fallback | DS 4-GEMM (default prec.) |
  |---|---|---|---|---|
  | 256 | 0.054 ms, 5.6e-16 | 0.166 ms, 1.9e-14 | 0.055 ms, 5.6e-16 | 0.072 ms, 2.9e-4 |
  | 1024 | 0.130 ms, 1.1e-15 | 0.691 ms, 3.8e-14 | 0.125 ms, 1.1e-15 | 0.121 ms, 2.9e-4 |
  | 2048 | 0.373 ms, 1.6e-15 | 4.579 ms, 5.4e-14 | 0.421 ms, 1.6e-15 | 0.343 ms, 2.9e-4 |

  Through-JAX kernel time at 2048² (4.58 ms) matches the standalone kernel plus split (4.50 ms): integration overhead is negligible. On the H100 the kernel is 12× slower than native f64, so registering it unconditionally is the wrong default there. **Decided 2026-10-31:** keep it always on when the plugin is active. The plugin is meant for GPUs with slow f64, where the elementwise rewrite is also only worthwhile; gating just the matmul would be inconsistent, and the H100 serves as the paper's counter-example with the kernel on. `DS_GEMM_KERNEL=0` turns it off. (A compute-capability gate was written and removed the same day.)
- **L40S result (2026-10-31, job 50067337) — the end-to-end matmul result.** Kernel registered; all 7 suites pass; f64 `A @ B` through JAX:

  | n | native f64 | DS kernel | speedup | DS err | DS native fallback |
  |---|---|---|---|---|---|
  | 256 | 0.287 ms | 0.375 ms | 0.77× | 1.9e-14 | 0.495 ms |
  | 1024 | 2.092 ms | 0.911 ms | 2.30× | 3.8e-14 | 1.928 ms |
  | 2048 | 12.614 ms | 4.424 ms | **2.85×** | 5.4e-14 | 12.687 ms |

  f64-class accuracy (5.4e-14 vs native 1.6e-15, f32 4e-7) at 2.85× native f64 through unmodified JAX. This node again had a ~0.3–0.45 ms per-call floor, so the 256² and 1024² rows are understated; 2048² is well above it. Black-Scholes in the same run: 1.23× at 16M.
- Main risks at first run (resolved on H100): whether the JAX 0.10.1 backend's FFI extension has `register_handler` and accepts the platform name; FFI header version skew between /opt/xla-ffi (XLA main at image build) and the backend.

### 16. Purification benchmark, and float converts cutting the DS chain (2026-10-31)

`tests/bench_purification.py` (`tests/run_purification.sh`): canonical density-matrix purification (Palser–Manolopoulos), 25 unrolled iterations = 50 matmuls, 2D tight-binding at half filling, n = 1024/2116/4096; accuracy vs NumPy eigh (density matrix, idempotency, electron count, band energy). NumPy check: f64 5e-15, f32 3e-6 with 4e-4 electrons lost.

First H100 run: DS results were *identical* to native f64 to every digit and only ~25% slower — i.e. not DS. The coverage check showed why: JAX emits float→float `stablehlo.convert`s (f64→f64, around `jnp.where` / weak-typed scalars) that the pass didn't handle, so the DS chain was cut after the first iteration; later matmuls ran as native f64 dot_generals. (Native f32 there: 7e-4 error, 1.2e-2 electrons lost — TF32 matmuls at default precision.)

Fixed (verified on H100 2026-10-31): ConvertOp between float types now carries the DS pair (f64→f32 rounds to f32 with lo = 0); the op snapshot is taken before `convertFuncArgs`, so the pass's own split ops are no longer walked (they had been reported as false "unsupported op" warnings). New tests in `test_f64_ds.py` (2d structural, 3c' GPU exactness across a widening convert). Black-Scholes may also have had such converts — rerun `run_full.sh` and compare with sections 12–13.

**H100 after the fix** (`run_full.sh h100-convert`: all 7 suites pass, Black-Scholes numbers unchanged — it was not affected; `run_purification.sh h100-v2`: coverage OK, 4 kernel calls / 0 dot_general / 0 untransformed in 2 iterations):

| n | run | ms | vs f64 | ‖P−Pexact‖ | ‖P²−P‖ | electrons lost | energy err |
|---|---|---|---|---|---|---|---|
| 4096 | f64 | 132.6 | 1.00× | 2.8e-14 | 3.6e-15 | 2.7e-12 | 3.9e-15 |
| 4096 | DS | 1732.9 | 0.08× | 3.9e-13 | 1.3e-13 | 9.3e-11 | 4.4e-16 |
| 4096 | DS native | 140.1 | 0.95× | 7.0e-14 | 6.3e-14 | 4.8e-11 | 1.1e-14 |
| 4096 | f32 | 23.8 | 5.58× | 7.1e-4 | 1.7e-4 | 3.2e-4 | 6.8e-5 |

DS is now genuinely DS (errors ~1e-13, distinct from f64) and f64-class; native f32 (TF32 matmuls) loses up to 1.2e-2 electrons across sizes. DS time at 4096 = 50 × ~35 ms kernel, as predicted; the H100 is the counter-example. L40S projection: f64 ~50 × 100 ms ≈ 5 s vs DS ≈ 1.8 s, i.e. ~2.5–3× — to be measured.

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

**Full benchmark results:** see `h100_results.txt` (H100, 2026-06-26) and `l40_results.txt` (L40S, 2026-06-26). Narrative analysis in `python/ds_results_compiler_pass_h100.txt` and `python/ds_results_l40s.txt`.

Key headline numbers (from the results files):
- DS vs f32 overhead at 2048²: **2.47×** on H100, **3.29×** on L40S (4 sub-matmuls vs 1)
- f64 vs DS-f32 speedup at 2048² matmul: **1.13×** on H100 (f64 tensor cores parity), **12.87×** on L40S (no f64 tensor cores)

---

## Known Limitations

| Operation | Status |
|---|---|
| Element-wise `add` / `sub` / `mul` / `divide` / `sqrt` inside `jax.jit` | ✅ |
| `negate`, `abs`, comparisons, `select` / `jnp.where`, `maximum` / `minimum` | ✅ |
| `exp` / `log` | ✅ (see section 8) |
| `sqrt(0)`, `x / 0` | ⚠️ NaN (not 0 / ±Inf) — property of the reference library's sequences |
| Any other float op (`tanh`, `sin`, `power`, `erf`, `reshape`, …) | ❌ Untransformed; DS precision lost from that op on. `DS_WARN_UNSUPPORTED=1` reports them |
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
| Inline pass pipeline (GPU) | `vhlo-to-version{target=1.16.3},vhlo-legalize-to-stablehlo,inline,func.func(ds-transform),stablehlo-legalize-to-vhlo` |
| Plugin temp files | Unique per compile call (`/tmp/ds_in_XXXXXX`, `ds_out_`, `ds_err_`, via `mkstemp`), removed when the call returns; the pass's stderr is forwarded to the plugin's stderr |
| `libVersion.a` | `/opt/mlir/lib/libVersion.a` — linked in `stablehlo_pass/CMakeLists.txt` |

---

## Plugin Environment Variables

| Variable | Effect |
|----------|--------|
| `PJRT_NAMES_AND_LIBRARY_PATHS="cuda:<path>"` | Load the DS proxy plugin |
| `DS_BYPASS=1` | Skip all transformation; pure passthrough to real backend |
| `DS_TEST_PASSTHROUGH=1` | Run DS pass but send original bytecode to backend |
| `DS_PASS_MODE=ffi` | Use FFI pass (`ds-ffi-transform`) instead of inline pass |
| `DS_RETURN_PAIRS=1` | A value returned exactly twice from a jitted function comes back as its raw `(hi, lo)` pair instead of the recombined result (test diagnostic) |
| `DS_WARN_UNSUPPORTED=1` | Report every op that consumes a DS-tracked value but is not transformed |

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

### Measured benchmark results (H100 PCIe, 2026-06-26)

See `h100_results.txt` for full tables. Summary from that run:

- Element-wise and reduction: DS-f32 within ~10% of native f64 at 1M elements (1.06× and 0.96× respectively)
- Matmul 2048²: DS-f32 **1.13× faster** than f64 (f64=0.35ms, DS=0.31ms; H100 f64 tensor cores confirmed at 48.6 TFLOPS)
- Matmul 256²: DS-f32 0.71× of f64 speed (4 launches vs 1 at latency-dominated size)

DS-f32 and native f64 are within ~10–15% at throughput-dominated sizes (≥1024²). The H100 has native FP64 tensor cores (~51 TFLOPS PCIe), making f64 GEMM competitive with f32. The large speedup expected for GPUs without FP64 hardware does not apply to H100 but does apply to the L40S (see `l40_results.txt`, up to 12.87× at 2048²).

**Precision improvement over f32 (confirmed):** DS-f32 sum error 1.49e-05 vs f32 error 1.22e-04 — 8× improvement (from `h100_results.txt`, f64 sum n=10000 test).

---

## Possible Next Steps

These are not assigned — listed in rough priority order for whoever picks this up next.

1. ~~**Run `bench_f64_vs_ds.py` and record the f64 vs DS speedup table**~~ ✅ **Done (2026-06-26).** H100: `h100_results.txt`. L40S: `l40_results.txt`. Up to 12.87× speedup on 2048² matmul on L40S.

2. **Benchmark DS vs f32 with `precision=HIGHEST` on larger matmul sizes.** The current benchmark uses default precision (TF32). A fair comparison needs both sides at the same precision setting. Run `bench_matmul.py` after modifying the inner benchmark code to use `jnp.dot(..., precision="highest")` in both f32 and DS modes.

2. **Multi-input `stablehlo.reduce`.** The ReduceOp handler in `DsTransformPass.cpp` skips reductions with more than 1 input (`if (redOp.getInputs().size() != 1) continue`). Extending to 2-input reductions (e.g., simultaneous min+max scans) would require 4 block args per input pair.

3. **Nested regions (scan, map, while).** `stablehlo.while` and `stablehlo.map` contain nested function regions. The current pass only walks `func.func` regions. Extending requires recursing into nested regions and threading the dsMap appropriately.

4. **`jax.vmap` over DS-transformed functions.** Untested. `vmap` inserts batch dimensions and may lower to additional StableHLO ops. Likely works for element-wise ops (batch dimension is transparent) but needs verification.

5. **Standalone matmul precision improvement via input splitting.** For `A @ B` with no prior DS ops, lo=0 so there is no benefit. To get improvement here, the pass would need to split each input element via `two_prod(A[i,k], 1.0)` to create a lo channel — but this costs an O(M×K + K×N) element-wise pass and is only worthwhile if the matmul accumulation error is the bottleneck.

6. **Re-run the L40S benchmarks on a quiet node** — the 2026-10-08 run had a ~0.4 ms per-call floor that hides everything but the largest sizes (section 8). Also still open: confirm against optimized HLO whether XLA really reassociates `(x + C) - C` (the reason for `emitTwoSum`'s constant swap).

7. **Bring FFI pass to parity with inline pass.** Currently the FFI pass only handles `add`/`sub`/`mul`. To enable a fair inline vs. FFI comparison: (a) add `stablehlo.constant` and `stablehlo.broadcast_in_dim` handlers to `DsFFIPass.cpp` mirroring what was added to `DsTransformPass.cpp`; (b) write a `ds_reduce` CUDA kernel in `cpp/` for reductions; (c) write a `ds_matmul` wrapper (4 cuBLAS dispatches) in `cpp/`; (d) add ReduceOp and DotGeneralOp handlers in `DsFFIPass.cpp`. Once at parity, benchmark `DS_PASS_MODE=ffi` vs default inline on the existing bench scripts. The FFI path may be more maintainable (CUDA kernels vs. MLIR C++); the benchmark will show whether the fusion loss is acceptable for the target workloads.
