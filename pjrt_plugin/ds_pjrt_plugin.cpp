#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <unistd.h>
#include <fcntl.h>
#include <string>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <spawn.h>
#include <sys/wait.h>
#include <cuda_runtime.h>

extern char** environ;
#include "xla/pjrt/c/pjrt_c_api.h"

static const char* REAL_PLUGIN_PATH =
    "/usr/local/lib/python3.12/dist-packages/jax_plugins/xla_cuda12_disabled/xla_cuda_plugin.so";
static const char* DS_OPT_BINARY =
    "/src/ds_experiment/stablehlo_pass/build/mlir-ds-opt";
static const char* DS_FFI_OPT_BINARY =
    "/src/ds_experiment/stablehlo_pass/build/mlir-ds-ffi-opt";

static bool use_ffi_pass() { const char* m = getenv("DS_PASS_MODE"); return m && std::string(m)=="ffi"; }
static bool bypass_pass()  { const char* b = getenv("DS_BYPASS");    return b && std::string(b)=="1"; }
static bool test_passthrough() { const char* t = getenv("DS_TEST_PASSTHROUGH"); return t && std::string(t)=="1"; }

static PJRT_Api ds_api;
static PJRT_Client_Compile* real_compile_fn = nullptr;
static bool initialized = false;

static int run_command(const char* binary, const char* pipeline,
                       const char* inp, const char* out, const char* err) {
    std::string pa = std::string("--pass-pipeline=") + pipeline;
    std::string pr = "--emit-bytecode-producer=StableHLO_v1.16.0";
    char* argv[] = {
        const_cast<char*>(binary),
        const_cast<char*>(pa.c_str()),
        const_cast<char*>("--emit-bytecode"),
        const_cast<char*>(pr.c_str()),
        const_cast<char*>(inp),
        const_cast<char*>("-o"),
        const_cast<char*>(out),
        nullptr
    };
    pid_t pid;
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, STDERR_FILENO, err,
                                     O_WRONLY|O_CREAT|O_TRUNC, 0644);
    int rc = posix_spawn(&pid, binary, &fa, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) return -1;

    int status;
    if (waitpid(pid, &status, 0) < 0) return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

// A temp file with a name unique to one compile call, removed when the
// call returns. The old code used fixed paths (/tmp/ds_in.mlir,
// /tmp/ds_out.mlir, /tmp/ds_pjrt_err.txt) shared by EVERY compilation on
// the machine. A single jax.jit call triggers several separate XLA
// compilations (array-construction/conversion helpers compiled ahead of
// the user's own function), and separate processes on the same node share
// /tmp, so overlapping compile calls could race on those paths and hand
// one call's transformed program to a different call's executable slot.
// mkstemp's O_CREAT|O_EXCL semantics make each call's paths unique and
// race-free by construction, eliminating the class of bug rather than
// narrowing the window.
class TempFile {
public:
    explicit TempFile(const char* prefix) {
        std::string tmpl = std::string(prefix) + "XXXXXX";
        std::vector<char> buf(tmpl.begin(), tmpl.end());
        buf.push_back('\0');
        int fd = mkstemp(buf.data());
        if (fd < 0) return;
        close(fd);
        path_.assign(buf.data());
    }
    ~TempFile() { if (ok()) unlink(path_.c_str()); }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    bool ok() const { return !path_.empty(); }
    const char* path() const { return path_.c_str(); }

private:
    std::string path_;
};

// Copy the pass's captured stderr (errors, DS_WARN_UNSUPPORTED reports) to
// this process's stderr, so it is visible without a shared log file.
static void forward_pass_stderr(const char* err_path) {
    std::ifstream err_file(err_path);
    std::string line;
    while (std::getline(err_file, line))
        fprintf(stderr, "[ds_pjrt] mlir-ds-opt: %s\n", line.c_str());
}

static PJRT_Error* ds_compile_wrapper(PJRT_Client_Compile_Args* args) {
    if (bypass_pass()) {
        return real_compile_fn(args);
    }

    TempFile in_tmp("/tmp/ds_in_"), out_tmp("/tmp/ds_out_"), err_tmp("/tmp/ds_err_");
    if (!in_tmp.ok() || !out_tmp.ok() || !err_tmp.ok()) {
        fprintf(stderr, "[ds_pjrt] Failed to create unique temp files. Falling back.\n");
        return real_compile_fn(args);
    }
    const char* in_path = in_tmp.path();
    const char* out_path = out_tmp.path();

    // 1. Write original bytecode to disk
    std::ofstream in_file(in_path, std::ios::binary);
    in_file.write(args->program->code, args->program->code_size);
    in_file.close();
    if (!in_file) {
        fprintf(stderr, "[ds_pjrt] Failed to write input file. Falling back.\n");
        return real_compile_fn(args);
    }

    // 2. Determine pipeline
    //
    // `inline` runs before ds-transform (inline pass only -- FFI pass
    // pipeline intentionally left untouched, out of scope here): JAX
    // outlines some primitives (confirmed: jnp.where) into a separate
    // private helper function called via func.call, rather than emitting
    // an inline stablehlo.select. DsTransformPass runs per-func.func and
    // has no func.call handler, so a DS-tracked value crossing such a call
    // boundary loses its (hi, lo) pairing -- the callee re-splits its own
    // arguments as fresh (v, 0) entry points, silently discarding the
    // caller's correction. Since mlir-ds-opt.cpp already calls
    // registerAllPasses(), MLIR's generic inliner is already available;
    // running it first eliminates these call boundaries before
    // ds-transform ever sees them, rather than requiring ds-transform to
    // understand call/return itself.
    const char* binary = use_ffi_pass() ? DS_FFI_OPT_BINARY : DS_OPT_BINARY;
    const char* pipeline = use_ffi_pass() ?
        "builtin.module(vhlo-to-version{target=1.16.3},vhlo-legalize-to-stablehlo,func.func(ds-ffi-transform),stablehlo-legalize-to-vhlo)" :
        "builtin.module(vhlo-to-version{target=1.16.3},vhlo-legalize-to-stablehlo,inline,func.func(ds-transform),stablehlo-legalize-to-vhlo)";

    // 3. Run transformation
    int rc = run_command(binary, pipeline, in_path, out_path, err_tmp.path());
    forward_pass_stderr(err_tmp.path());

    if (rc != 0) {
        fprintf(stderr, "[ds_pjrt] Pass failed with code %d. Falling back to original.\n", rc);
        return real_compile_fn(args);
    }

    if (test_passthrough()) {
        fprintf(stderr, "[ds_pjrt] Test passthrough active. Sending original args.\n");
        return real_compile_fn(args);
    }

    // 4. Read transformed bytecode size
    std::ifstream out_file(out_path, std::ios::binary | std::ios::ate);
    if (!out_file) {
        fprintf(stderr, "[ds_pjrt] Failed to open output file. Falling back.\n");
        return real_compile_fn(args);
    }
    size_t transformed_size = out_file.tellg();
    out_file.seekg(0, std::ios::beg);

    fprintf(stderr, "[ds_pjrt] transformed %zu -> %zu bytes\n", args->program->code_size, transformed_size);

    // 5. ALLOCATE PINNED MEMORY
    char* pinned_buf = nullptr;
    cudaError_t err = cudaMallocHost((void**)&pinned_buf, transformed_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ds_pjrt] cudaMallocHost failed: %s. Falling back.\n", cudaGetErrorString(err));
        return real_compile_fn(args);
    }

    out_file.read(pinned_buf, transformed_size);
    if (!out_file) {
        fprintf(stderr, "[ds_pjrt] Failed to read output file. Falling back.\n");
        cudaFreeHost(pinned_buf);
        return real_compile_fn(args);
    }
    out_file.close();

    // 6. Duplicate structs with correct sizes
    PJRT_Program modified_prog = *(args->program);
    modified_prog.code = pinned_buf;
    modified_prog.code_size = transformed_size;

    PJRT_Client_Compile_Args modified_args = *args;
    modified_args.program = &modified_prog;

    // 7. Dispatch to real backend
    fprintf(stderr, "[ds_pjrt] Dispatching with pinned memory...\n");
    PJRT_Error* res = real_compile_fn(&modified_args);
    args->executable = modified_args.executable; // CRITICAL FIX: Pass the compiled executable back to JAX

    // 8. Cleanup
    // // cudaFreeHost(pinned_buf); // Intentional leak to fix use-after-free // Intentional leak to fix use-after-free
    return res;
}

// Boilerplate PJRT Proxy Initialization
extern "C" const PJRT_Api* GetPjrtApi() {
    if (!initialized) {
        void* handle = dlopen(REAL_PLUGIN_PATH, RTLD_LAZY | RTLD_LOCAL);
        if (!handle) {
            fprintf(stderr, "[ds_pjrt] Failed to load real plugin: %s\n", dlerror());
            return nullptr;
        }

        typedef const PJRT_Api* (*GetPjrtApiFn)();
        GetPjrtApiFn real_get_api = (GetPjrtApiFn)dlsym(handle, "GetPjrtApi");
        if (!real_get_api) {
            fprintf(stderr, "[ds_pjrt] Failed to find GetPjrtApi\n");
            return nullptr;
        }

        const PJRT_Api* real_api = real_get_api();
        memcpy(&ds_api, real_api, sizeof(PJRT_Api));

        real_compile_fn = ds_api.PJRT_Client_Compile;
        ds_api.PJRT_Client_Compile = ds_compile_wrapper;

        initialized = true;
        fprintf(stderr, "[ds_pjrt] Proxy initialized\n");
    }
    return &ds_api;
}
