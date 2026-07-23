# ds_reeval — paper claim re-evaluation suite

Re-measures three of the paper's (`paper/main.tex`) highest-priority claims
under corrected configurations, and produces machine-readable JSON plus a
human-readable `REPORT.md`.

**Status as of this commit: code only, nothing has been run.** This suite
was built by an agent that was explicitly instructed not to execute any
benchmark, accuracy measurement, or profiling run, and not to produce
anything under `results/`. There is no `results/` directory in this
checkout, and `REPORT.md` does not exist yet. Everything below is a runbook
for the human running this on the actual clusters — Claude did not run any
of it, and no numbers in `paper/main.tex` should be treated as re-verified
until you have run this suite and reviewed `REPORT.md`.

## What's here

| File | Purpose |
|---|---|
| `common.py` | Timing protocol, GPU detection, JSON I/O, x64 assertion. Not run directly. |
| `exp1_gemm_highest.py` | Worker: f64-vs-DS GEMM sweep at default and HIGHEST precision. |
| `exp2_tf32_dispatch.py` | Worker: dumps XLA HLO (+ optional `nsys` profile) to identify what kernel the DS sub-GEMMs actually dispatch to. |
| `exp3_pair_accuracy.py` | Worker: compares recombined-in-function vs. host-recombined (raw hi/lo) reduction error. |
| `exp4_divide_worst_case.py` | Worker: worst-case relative accuracy of DS divide post the `-t3` correction (random sweep + adversarial + power-of-two + beyond-safe-range probe). |
| `exp5_f64_reduction_bisection.py` | Worker: bisects the f64-input reduction ingestion path (ground truth/methodology, split fidelity, single op, error-vs-length, recombination) to root-cause the paper's f64 reduction residual. |
| `run_all.py` | Driver — launches one subprocess per (experiment × configuration) with the right environment. |
| `report.py` | Assembles all present `results/<gpu>/*.json` into `REPORT.md`. Degrades gracefully over partial results. |
| `test_return_pairs_structural.py` | CPU-only (no GPU) structural test for the new `DS_RETURN_PAIRS` pass flag. Run this first. |
| `.gitignore` | Keeps `results/`, `smoke_out/`, XLA/nsys dump artifacts, and generated reports out of git. |

Also part of this change, outside `ds_reeval/`:

- `../../stablehlo_pass/DsTransformPass.cpp` — added the `DS_RETURN_PAIRS=1`
  opt-in flag that Experiment 3 needs (see "DS_RETURN_PAIRS" section below).
  **This means `stablehlo_pass` must be rebuilt before running anything
  here** — see Step 1.

## Step 0 — read this if you only read one section

Experiments 1–5 need a GPU node inside the project's Docker/Singularity
container, on **both** target clusters (Punakha H100 and Bridges-2 L40S —
the paper's claims are cross-GPU, so both need a run). Budget for two
separate cluster sessions. Nothing here needs both GPUs simultaneously.
Experiments 4 and 5 additionally require the divide/sqrt commits already
in `stablehlo_pass/DsTransformPass.cpp` (the `-t3` correction and
`emitDsDivByScalar`) — Step 1's rebuild covers this, since it's the same
binary Experiments 1–3 already require rebuilding.

## Step 1 — rebuild `stablehlo_pass` (required, one-time per cluster/container)

`DsTransformPass.cpp` changed (added `DS_RETURN_PAIRS`). `mlir-ds-opt` is
a compiled binary, not interpreted, so this **will not take effect** until
rebuilt. Do this inside the container, every session, before anything else
in this suite (Experiments 1 and 2 don't touch the new flag, but there's no
reason to run them against a stale binary either):

```bash
cd /src/ds_experiment
cmake -GNinja -S stablehlo_pass -B stablehlo_pass/build -DCMAKE_CXX_FLAGS="-fno-rtti"
ninja -C stablehlo_pass/build
```

(On Bridges-2, `submit_l40s.sh` already does this exact rebuild
automatically after the bind-mount, because the sandboxed image's
pre-built binary gets shadowed by the mounted source tree — see its
`[3/4]` step. On Punakha there is no automatic rebuild step for
`stablehlo_pass` in `ds_setup.sh` (it only rebuilds `pjrt_plugin`), so you
must run the two lines above by hand after `bash ds_setup.sh`.)

## Step 2 — structural sanity check (CPU-only, no GPU, ~seconds)

Before spending any GPU allocation on Experiment 3, confirm the new pass
flag actually does what it's supposed to:

```bash
cd /src/ds_experiment
python3 paper/ds_reeval/test_return_pairs_structural.py
```

Expected output: `PASS: DS_RETURN_PAIRS structural test (3 checks x 2 modes, all as expected)`.
If this fails, something is wrong with the rebuild or the flag — fix that
before running Experiment 3 on a GPU. This test never touches `results/`;
it prints PASS/FAIL to stdout only.

## Step 3 — dry run (any machine, no GPU needed, instant)

Sanity-check the plan before committing GPU time:

```bash
cd /src/ds_experiment/paper/ds_reeval
python3 run_all.py --dry-run
```

Prints every subprocess command and environment `run_all.py` would launch.
Executes nothing, writes nothing (not even `manifest.json`).

## Step 4 — smoke test (GPU required, ~2–5 min)

Reduced sizes/reps, fast correctness pass, writes to `smoke_out/` (never
`results/`) so it can't be mistaken for a real measurement:

```bash
python3 run_all.py --smoke
python3 report.py --results-dir smoke_out --out SMOKE_REPORT.md
```

Confirm `SMOKE_REPORT.md` looks sane (numbers present, no error strings,
Experiment 3's "pairs" error is much smaller than "standard") before
committing to the full run.

## Step 5 — full run, per cluster

### Punakha (H100 PCIe, sm_90a, node `hopper001`)

```bash
module load docker/27.3.1/rootless-docker
srun --account=punakha_partner_sdriver -p dgx -n 1 --gres=gpu:1 \
     --qos punakha_dgx2_general --pty bash
start_rootless_docker.sh --quiet

cd /o_home/racastaneda3/ds_experiment
docker run --gpus all -it --volume $(pwd):/src/ds_experiment --rm ds-experiment
```

Inside the container:

```bash
cd /src/ds_experiment
bash ds_setup.sh                                   # every session (ephemeral --rm container)
cmake -GNinja -S stablehlo_pass -B stablehlo_pass/build -DCMAKE_CXX_FLAGS="-fno-rtti" \
  && ninja -C stablehlo_pass/build                 # Step 1 above — required this session too

# image digest for manifest.json -- run on the HOST, before/after `docker run`,
# not inside the container:
#   docker inspect --format='{{.Id}}' ds-experiment
# (RepoDigests is usually empty for a purely local `docker build`; use the
# image Id instead unless you've pushed/pulled this tag from a registry.)

cd paper/ds_reeval
python3 test_return_pairs_structural.py             # Step 2
python3 run_all.py --dry-run                         # Step 3
python3 run_all.py --smoke && python3 report.py --results-dir smoke_out --out SMOKE_REPORT.md   # Step 4
python3 run_all.py --image-digest <id-from-host>      # Step 5 -- the real run
python3 report.py
```

### Bridges-2 / PSC (L40S, sm_89, project `cis260064p`)

The project already has a working SLURM batch script (`submit_l40s.sh`)
that pulls the Singularity image, builds a writable sandbox, binds the repo
in at `/src/ds_experiment`, and rebuilds `stablehlo_pass` against the
bind-mounted (i.e. your locally-edited) source. It currently ends by
running `tests/run_tests.sh --bench`; **do not edit that script as part of
this change** (out of scope here) — instead, either:

**Option A — interactive session, run ds_reeval commands by hand:**

```bash
srun --partition=GPU-shared --gres=gpu:l40s-48:1 --account=cis260064p \
     --time=3:00:00 --pty bash

export APPTAINER_CACHEDIR=$LOCAL/.apptainer
export APPTAINER_TMPDIR=$LOCAL/.apptainer/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
singularity pull "$LOCAL/ds-experiment.sif" docker://rcast915/ds-experiment:latest
singularity build --sandbox "$LOCAL/ds-sandbox" "$LOCAL/ds-experiment.sif"
mkdir -p "$LOCAL/ds-sandbox/jet" "$LOCAL/ds-sandbox/ocean"

singularity exec --nv --writable \
  --bind "$HOME/ds-experiment":/src/ds_experiment \
  "$LOCAL/ds-sandbox" bash
```

Then inside the container shell:

```bash
cd /src/ds_experiment
bash ds_setup.sh
cmake -GNinja -S stablehlo_pass -B stablehlo_pass/build -DCMAKE_CXX_FLAGS="-fno-rtti" \
  && ninja -C stablehlo_pass/build

cd paper/ds_reeval
python3 test_return_pairs_structural.py
python3 run_all.py --dry-run
python3 run_all.py --smoke && python3 report.py --results-dir smoke_out --out SMOKE_REPORT.md
python3 run_all.py   # image digest: see note below
python3 report.py
```

Image digest for `manifest.json` on this path: there's no local `docker`
daemon inside Apptainer. Best options, in order of preference: (1) if you
have `docker`/`skopeo`/`crane` available on the login node,
`skopeo inspect docker://rcast915/ds-experiment:latest | grep Digest`; (2)
otherwise pass something identifying at least the tag and pull date, e.g.
`--image-digest "docker.io/rcast915/ds-experiment:latest@$(date -I)"`, and
note in your own records that this is a tag reference, not a true content
digest.

**Option B — extend `submit_l40s.sh` yourself** by adding the `ds_reeval`
commands (Step 1's rebuild is already done for you by the existing script)
after its `[3/4]` step and before `[4/4] Done` — this file does not do that
for you, since modifying an existing SLURM script for a specific cluster
account (`cis260064p`) wasn't part of what was asked here.

## What a full run produces

```
ds_reeval/results/
├── <gpu-tag>/                                   # e.g. NVIDIA-H100-PCIe or NVIDIA-L40S
│   ├── manifest.json                            # commit, container digest, JAX/CUDA versions, timestamp
│   ├── exp1_gemm_highest_f64_default.json
│   ├── exp1_gemm_highest_f64_highest.json
│   ├── exp1_gemm_highest_ds_default.json
│   ├── exp1_gemm_highest_ds_highest.json
│   ├── exp2_tf32_dispatch_default.json
│   ├── exp2_tf32_dispatch_highest.json
│   ├── xla_dumps/{default,highest}/             # raw XLA HLO text + nsys reports (large; gitignored)
│   ├── exp3_pair_accuracy_standard.json
│   ├── exp3_pair_accuracy_pairs.json
│   ├── exp4_divide_worst_case_internal.json
│   ├── exp4_divide_worst_case_observable.json
│   ├── exp5_f64_reduction_bisection_ground_truth.json
│   ├── exp5_f64_reduction_bisection_split_fidelity.json
│   ├── exp5_f64_reduction_bisection_single_op.json
│   ├── exp5_f64_reduction_bisection_length_scan.json
│   └── exp5_f64_reduction_bisection_recombination.json
└── <other-gpu-tag>/...                          # after the second cluster's run
ds_reeval/REPORT.md                                # written by report.py, reads both gpu dirs if present
```

One manifest per GPU directory, not a single shared one — the two clusters
are visited at different times with potentially different commits/container
builds, and a single shared path would have the second run silently
clobber the first run's provenance.

## Expected runtime

- Structural test: seconds.
- `--dry-run`: instant.
- `--smoke`: 2–5 minutes (small sizes, few reps, still real GPU execution).
- Full run: Experiment 1 dominates at large sizes (2048², optionally 4096²
  with `--include-4096`); expect single-digit minutes per GPU on H100, a
  bit longer on L40S where native f64 GEMM is the slow baseline being
  measured (per the paper, 12.19 ms at 2048² alone — the point of the
  whole exercise). Experiment 2 adds time only if `nsys` is available
  (profiling has overhead; HLO-dump-only is fast). Experiment 3 is small
  (a single 10,000-element reduction) and fast regardless. Experiment 4's
  random sweep is the biggest new cost (default 3 seeds × 100,000 samples
  per mode, `--divide-samples-per-seed` to adjust) but each seed is one
  batched GPU divide call, so still well under a minute per mode in
  practice. Experiment 5's five stages are all small (single elements or
  length ≤10,000), dominated by process startup, not compute.

## Reading `REPORT.md`

Per GPU: the manifest summary, then per-experiment tables, ending in a
suite-wide summary table (`| # | Claim | Paper says | Measured | Verdict |`).
Experiment 1 gets PASS/CHANGED verdicts (>20% relative deviation from the
paper's published number flags CHANGED); Experiment 3 gets
CONFIRMED/REFUTED/INCONCLUSIVE per the hypothesis test described in its
docstring. Experiment 2's verdict is a best-effort YES/NO/UNKNOWN per
precision mode based on whatever kernel evidence (`nsys` and/or HLO dump)
was actually available — read the "evidence quality" column before trusting
a YES/NO at face value; `nsys_kernel_names` is stronger evidence than
`hlo_dump_text_match`, which in turn is stronger than
`hlo_dump_present_but_inconclusive`.

Experiment 4 has no PASS/FAIL verdict against a prior published number —
`paper/main.tex` doesn't have a divide worst-case figure yet, only a
pre-fix baseline (1.16e-7) to compare the post-fix number against;
IMPROVED/REGRESSED in the summary table reflects that comparison, not a
reproduction check. Experiment 5 similarly has no PASS/FAIL — its
"verdict" is either a stated root cause (most likely: a methodology
mismatch between the existing f64 test and the f32-input claim it's
compared against, see Stage 1) or the narrowest bracket the stages that
ran actually establish.

`report.py` is safe to re-run any time (e.g. after only one cluster's run
has finished) — it only reads whatever JSON currently exists under
`results/` and rewrites `REPORT.md` from scratch; it never launches
anything.

## The `DS_RETURN_PAIRS` flag

Added to `stablehlo_pass/DsTransformPass.cpp` specifically to make
Experiment 3 possible. Background: the pass always recombines a DS
`(hi, lo)` pair back to a single f32/f64 value at `func.return` — there was
previously no way to observe the raw pair from outside the compiled
function, so there was no way to test whether the ~4.65e-6 reduction error
the paper reports is dominated by that final recombination (output
quantization) rather than by accumulated DS arithmetic error.

**Mechanism:** off by default (zero behavior change unless set).
`DS_RETURN_PAIRS=1`, read directly from the process environment inside
`DsTransformPass::runOnOperation()` (mirroring how `DS_BYPASS` /
`DS_TEST_PASSTHROUGH` / `DS_PASS_MODE` are read in `ds_pjrt_plugin.cpp` —
`mlir-ds-opt` is always spawned as a child process that inherits the
caller's full environment via `posix_spawn(..., environ)`, so no plumbing
through the pass-pipeline string was needed). When set, if the *same*
DS-tracked SSA value is returned twice in one `func.return` (e.g.
`return s, s` for a DS-tracked `s` — this is exactly what
`exp3_pair_accuracy.py --mode pairs` does), the first occurrence is
replaced with the raw `hi` component and the second with the raw `lo`
component, instead of both being independently recombined to the same
redundant f32/f64 value.

**Why this design and not a 1-output-to-2-output change:** JAX fixes a
`jax.jit`-ed function's output arity/shapes *before* the plugin ever sees
the bytecode (from tracing the Python function, well before
`PJRT_Client_Compile` is intercepted). If the pass changed a function from
1 declared output to 2, or changed an output's shape, the compiled
executable's actual output count/shape would no longer match what JAX's
Python layer is expecting to unflatten — a real risk of a hard crash or
worse at the PJRT boundary, and not something to introduce without being
able to test it end-to-end (which this task explicitly could not do). The
doubled-return trick sidesteps this entirely: the Python-traced function
already has 2 outputs of the same type/shape whether the flag is on or
off, so `DS_RETURN_PAIRS` only changes *which value* lands in each slot,
never the function's signature.

**Scope:** applies to operands whose declared return type is either f32
(the common case — `hi`/`lo` are already that type, so they're substituted
directly) or f64. For an f64-typed doubled return, substituting a raw f32
`hi`/`lo` directly would be ill-typed, so each is widened first
(`convert(hi, f64)`, `convert(lo, f64)` — both exact, since widening f32 to
f64 never loses precision) and *those* are substituted instead of the
(lossy-at-output) `hi+lo` sum `emitToFloat` would otherwise produce. Either
way the `FuncOp`'s result-type signature never changes — the substituted
value always matches the type the operand already had. The f64 case was
added specifically because `exp5_f64_reduction_bisection.py` needed to
observe a raw f64-sourced DS pair, which the original f32-only version of
this flag could not do (see its stage docstrings and
`DsTransformPass.cpp`'s func.return comment for the exact mechanism). A
value returned only once, or 3+ times, is unaffected (falls back to normal
recombination) — the flag only ever special-cases the exact doubled-return
pattern, for either type.

**Bug found and fixed (f64 case) — see `exp5_f64_reduction_bisection.py`'s
module docstring and `DsTransformPass.cpp`'s func.return comment for the
full account.** Initial testing found several `exp5` stages showing
symptoms not explained by the MLIR the pass emits (which was correct).
Root cause: substitution was decided per-operand from "1st or 2nd
occurrence seen so far" with no check on the *total* occurrence count, so
a value returned exactly once also hit "1st occurrence" and silently lost
`lo`. Predates this session's f64 extension, but was invisible for f32
(dropped `lo` and full `hi+lo` recombination are typically bit-identical
after f32 output quantization) and untested for a single return under the
flag (Experiment 3 always used the genuine doubled-return pattern). Fixed
by requiring a value's total occurrence count to be exactly 2 before
substituting at all; confirmed via `null_test_return_pairs_noop.py` (f64
flag on/off now bit-identical for a single return) and a new structural
regression case in `test_return_pairs_structural.py`. Experiment 3's own
figure and `exp4`'s divide worst-case measurement were never affected —
both use the doubled-return pattern on f32 arrays throughout, and both
were independently reconfirmed unaffected before the fix was even found.

**Verifying the "default behavior is untouched" guarantee:** the
`DS_RETURN_PAIRS == false` code path in `DsTransformPass.cpp` is,
line-for-line, the pre-existing `func.return` handling logic, just moved
under an `if` — so it is unchanged by construction, not merely by testing.
`test_return_pairs_structural.py` (Step 2) is the executable check that
this claim holds: with the flag off, it asserts the output still contains
exactly the same op pattern (2 independent recombinations: 4 converts + 2
adds) the pass has always produced for a doubled return.
