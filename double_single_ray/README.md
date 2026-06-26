# llvm-pass-skeleton

An LLVM module pass that replaces all `double` (64-bit float) arithmetic in a
compiled program with **double-single (DS) arithmetic** — a software-emulated
higher-precision format built from two `float32` values.  The transformation is
performed entirely at the IR level via an LLVM plugin, so the target source code
is unchanged.

---

## Building

Requirements: LLVM 17+, CMake 3.6+, a C++17 compiler.

```bash
cd llvm-pass-skeleton
mkdir build && cd build
cmake ..
make
cd ..
```

The plugin shared library will be at `build/skeleton/SkeletonPass.so` (Linux) or
`build/skeleton/SkeletonPass.dylib` (macOS).

---

## Running the Pass

### 1. Build the runtime library

The pass replaces arithmetic with calls into a small C runtime (`rtlib.c`).
Compile it separately — once without `main` (for linking against transformed
programs), and once with `main` (for the standalone demo):

```bash
# Library object (no main) — used when linking your own program
clang -DRTLIB_NO_MAIN -O2 -c rtlib.c \
      -Illvm-accuracy-analysis-k-test/double-single-lib \
      -o rtlib_nomain.o

# Standalone object (with main demo)
clang -O2 -c rtlib.c \
      -Illvm-accuracy-analysis-k-test/double-single-lib \
      -o rtlib.o
```

Also compile the underlying DS arithmetic library:

```bash
clang -O2 -c llvm-accuracy-analysis-k-test/double-single-lib/double-binary32.c \
      -o double-binary32.o
```

### 2. Emit LLVM IR for your source file

```bash
clang -O1 -S -emit-llvm your_program.c -o your_program.ll
```

> Use `-O1` or higher so that LLVM emits clean arithmetic IR. `-O0` tends to
> produce memory-heavy IR that limits what the pass can transform.

### 3. Run the pass and capture the transformed IR

```bash
opt -load-pass-plugin build/skeleton/SkeletonPass.so \
    -passes="skeleton" \
    -disable-output \
    your_program.ll 2> your_program_transformed.ll
```

The pass prints the transformed module to `stderr`, so redirect accordingly.

### 4. Assemble and link

```bash
# Assemble transformed IR to object
clang -c your_program_transformed.ll -o your_program_transformed.o

# Link with the runtime and DS library
clang your_program_transformed.o rtlib_nomain.o double-binary32.o -lm -o your_program_ds

./your_program_ds
```

### Quick end-to-end example (`a.c`)

```bash
clang -O1 -S -emit-llvm a.c -o a.ll
opt -load-pass-plugin build/skeleton/SkeletonPass.so \
    -passes="skeleton" -disable-output a.ll 2> a_transformed.ll
clang -c a_transformed.ll -o a_transformed.o
clang a_transformed.o rtlib_nomain.o double-binary32.o -lm -o test_ds
./test_ds
```

---

## Algorithm: How the Pass Transforms `double` to DS

The transformation is a whole-module IR rewrite pass (`SkeletonPass`).  Every
instruction that touches a `double` value is replaced with an equivalent
sequence that operates on a `DoubleSingle` struct (`{float hi, float lo}`).

### The DoubleSingle representation

A `double` value `x` is represented as two `float32` values:

```
x ≈ hi + lo      where  hi = (float)x,  lo = (float)(x - (double)hi)
```

`hi` holds the high-order bits and `lo` captures the rounding error.  Together
they provide roughly twice the mantissa bits of a single `float32`, approaching
`double` precision using only `float32` hardware.

### Pass structure

The pass runs once over every non-empty function in the module.  For each
function it:

1. **Snapshots** all original instructions before iterating, so newly emitted
   runtime calls are never re-processed.
2. **Processes** each original instruction through `processInstruction`.
3. **Fixes up PHI nodes** after all instructions are processed (PHI incoming
   values may not yet exist when the PHI is first seen).
4. **Deletes** original instructions in reverse order, skipping any that still
   have live uses.

### Instruction-by-instruction rules

| Original instruction | Transformation |
|---|---|
| `alloca double` | Replace with `alloca DoubleSingle`; record the original type in `origTypeMap`. |
| `alloca [N x double]` | Replace with `alloca [N x DoubleSingle]` recursively via `transformType`. |
| `load double, ptr` | If the pointer is in `valueMap`, emit `load DoubleSingle, newPtr`. |
| `store double val, ptr` | If ptr is mapped: store the DS value, or call `convert_double_to_ds` for a double constant/argument. |
| `getelementptr` | Remap the base pointer to its DS counterpart; re-emit with the same indices. |
| `fadd / fsub / fmul / fdiv` | Store each operand into a scratch `alloca`, call the corresponding runtime function (`external_double_add`, etc.), load the result. |
| `fneg` | Call `external_double_neg`. |
| `llvm.sqrt` / `llvm.fabs` | Call `external_double_sqrt` / `external_double_fabs`. |
| `llvm.fmuladd(a, b, c)` | Call `external_double_fmadd(result, a, b, c)` (implemented as `mul` then `add`). |
| `llvm.memcpy` on a double array | Walk every element with `emitConvertElements`: load each `double`, call `convert_double_to_ds` into the DS array slot. |
| `llvm.memset` (zero) on a double array | Walk every element with `emitZeroElements`: store `{0.0f, 0.0f}` into each DS slot. |
| `phi double` | Create a `phi DoubleSingle`; incoming values are fixed up after all instructions are processed. |
| `select double` | Emit `select DoubleSingle` using the same condition. |
| `fcmp` | Convert both DS operands back to `double` via `ds_to_double`, then re-emit `fcmp`. |
| `sitofp / uitofp → double` | Keep the original cast; call `convert_double_to_ds` on the result to produce a DS value. |
| `fptosi / fptoui double → int` | Call `ds_to_double` on the DS operand, then cast the resulting `double` to the integer type. |
| `fptrunc double → float` | Extract the `hi` field of the DS struct (dropping `lo`). |
| `fpext float → double` | Build a DS struct with the float as `hi` and `0.0f` as `lo`. |
| `ret double` | Convert the DS result back to `double` via `ds_to_double` before returning. |
| External call with `double` args/return | Convert each tracked DS arg back to `double`; if the return type is `double`, wrap the returned value in DS. |
| `bitcast ptr` (typed-pointer IR) | Mark for deletion since its alloca is being replaced. |

### Scratch allocas

Four `DoubleSingle` stack slots are allocated at function entry:

- `ds.op1`, `ds.op2` — operand staging for binary and unary ops.
- `ds.op3` — third operand slot used exclusively for `fmuladd` to avoid
  aliasing with `ds.result`.
- `ds.result` — receives the output of every runtime arithmetic call.

These are reused across all operations within a function, keeping stack
overhead constant regardless of the number of arithmetic operations.

### Runtime library (`rtlib.c`)

| Function | Signature | Purpose |
|---|---|---|
| `convert_double_to_ds` | `(DS*, double) → void` | Split a `double` into `{hi, lo}` |
| `ds_to_double` | `(DS*) → double` | Reconstruct `double` from `hi + lo` |
| `external_double_add/sub/mul/div` | `(DS*, DS*, DS*) → void` | DS binary arithmetic |
| `external_double_neg/sqrt/fabs` | `(DS*, DS*) → void` | DS unary operations |
| `external_double_fmadd` | `(DS*, DS*, DS*, DS*) → void` | Fused multiply-add via `mul` then `add` |
| `print_ds_value` | `(DS) → void` | Debug printer |

The actual arithmetic algorithms live in
`llvm-accuracy-analysis-k-test/double-single-lib/double-binary32.{h,c}`.
