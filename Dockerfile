# Two stages. Stage 1 builds LLVM/MLIR + StableHLO (CUDA-independent, ~1 h);
# stage 2 is the CUDA 13 runtime image and copies /opt/mlir from stage 1, so a
# CUDA or JAX change does not rebuild LLVM.
#
# Versions: CUDA 13.0.2 · Python 3.12 · JAX 0.12.1 (jax[cuda13]) · StableHLO
# 1.16.3 for the pass. JAX 0.12.1 itself is at StableHLO 1.25.3; the PJRT
# proxy reports 1.16.3 to JAX so programs reach the pass in a version it can
# parse (see pjrt_plugin/ds_pjrt_plugin.cpp, kPassStablehlo).

# ══ Stage 1: LLVM/MLIR + StableHLO ═══════════════════════════════════════════
FROM ubuntu:24.04 AS mlir

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    cmake ninja-build git build-essential python3 \
    libzstd-dev zlib1g-dev libffi-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# ── StableHLO (pinned to v1.16.3) ─────────────────────────────────────────────
# Clone first so we can read its required LLVM commit before building LLVM.
RUN git clone --depth=1 --branch v1.16.3 \
    https://github.com/openxla/stablehlo.git /tmp/stablehlo-src

# ── LLVM + MLIR (at the commit StableHLO v1.16.3 requires) ───────────────────
# -j4 cap prevents OOM on machines with limited RAM.
# -fno-rtti matches LLVM's own build flags — required to link against MLIR static libs.
RUN LLVM_COMMIT=$(cat /tmp/stablehlo-src/build_tools/llvm_version.txt | tr -d '[:space:]') && \
    git clone --filter=blob:none --no-checkout \
        https://github.com/llvm/llvm-project.git /tmp/llvm-src && \
    git -C /tmp/llvm-src fetch --depth=1 origin "$LLVM_COMMIT" && \
    git -C /tmp/llvm-src checkout "$LLVM_COMMIT"

RUN cmake -GNinja \
      -S /tmp/llvm-src/llvm \
      -B /tmp/llvm-build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/mlir \
      -DCMAKE_CXX_FLAGS="-fno-rtti" \
      -DLLVM_ENABLE_PROJECTS="mlir" \
      -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
      -DLLVM_INCLUDE_TESTS=OFF \
      -DLLVM_INCLUDE_BENCHMARKS=OFF \
      -DMLIR_INCLUDE_TESTS=OFF \
      -DLLVM_INSTALL_UTILS=ON \
      -DLLVM_ENABLE_PLUGINS=ON \
    && ninja -j4 -C /tmp/llvm-build install \
    && rm -rf /tmp/llvm-src /tmp/llvm-build

RUN cmake -GNinja \
      -S /tmp/stablehlo-src \
      -B /tmp/stablehlo-build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/mlir \
      -DCMAKE_CXX_FLAGS="-fno-rtti" \
      -DMLIR_DIR=/opt/mlir/lib/cmake/mlir \
      -DLLVM_DIR=/opt/mlir/lib/cmake/llvm \
      -DSTABLEHLO_BUILD_EMBEDDED=OFF \
      -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \
    && ninja -j4 -C /tmp/stablehlo-build install \
    && cp -r /tmp/stablehlo-src/stablehlo /opt/mlir/include/stablehlo \
    && find /tmp/stablehlo-build/stablehlo -name "*.h.inc" | \
         while IFS= read -r f; do \
           rel="${f#/tmp/stablehlo-build/}"; \
           mkdir -p "/opt/mlir/include/$(dirname "$rel")"; \
           cp "$f" "/opt/mlir/include/$rel"; \
         done \
    && rm -rf /tmp/stablehlo-src /tmp/stablehlo-build


# ══ Stage 2: CUDA 13 runtime image ═══════════════════════════════════════════
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# ── System packages ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3-pip \
    cmake ninja-build git wget curl vim \
    build-essential lsb-release \
    libzstd-dev zlib1g-dev libffi-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1

COPY --from=mlir /opt/mlir /opt/mlir

# ── Python / JAX (pinned) ──────────────────────────────────────────────────────
RUN pip install "jax[cuda13]==0.12.1" numpy

# Disable JAX CUDA auto-registration so our PJRT proxy can claim the cuda slot.
# ds_pjrt_plugin.cpp loads xla_cuda13_disabled as the real plugin.
RUN mv /usr/local/lib/python3.12/dist-packages/jax_plugins/xla_cuda13 \
       /usr/local/lib/python3.12/dist-packages/jax_plugins/xla_cuda13_disabled && \
    mv /usr/local/lib/python3.12/dist-packages/jax_cuda13_plugin \
       /usr/local/lib/python3.12/dist-packages/jax_cuda13_plugin_disabled

# ── XLA FFI headers: the ones shipped with jaxlib 0.12.1 ──────────────────────
RUN ln -s "$(python3 -c 'import jax.ffi; print(jax.ffi.include_dir())')" /opt/xla-ffi && \
    test -f /opt/xla-ffi/xla/ffi/api/ffi.h

# ── PJRT C API headers at the XLA commit jaxlib 0.12.1 was built from ────────
# (jax-v0.12.1 MODULE.bazel pins openxla/xla 97b8433b; PJRT API 0.118)
ARG XLA_COMMIT=97b8433b3ed82b78b742f751d2c7f510e5d78d25
RUN mkdir -p /opt/xla-pjrt/xla/pjrt/c && \
    for h in pjrt_c_api.h pjrt_c_api_ffi_extension.h; do \
      wget -q -O /opt/xla-pjrt/xla/pjrt/c/$h \
        https://raw.githubusercontent.com/openxla/xla/${XLA_COMMIT}/xla/pjrt/c/$h || exit 1; \
    done

# ── PATH ───────────────────────────────────────────────────────────────────────
ENV PATH="/opt/mlir/bin:${PATH}"

# ── Build FFI kernels + DS pass ────────────────────────────────────────────────
WORKDIR /src/ds_experiment
COPY . /src/ds_experiment

RUN cmake -GNinja -S cpp -B cpp/build && ninja -C cpp/build

RUN cmake -GNinja \
      -S stablehlo_pass \
      -B stablehlo_pass/build \
      -DCMAKE_CXX_FLAGS="-fno-rtti" \
    && ninja -C stablehlo_pass/build

# ── Default shell ──────────────────────────────────────────────────────────────
CMD ["/bin/bash"]
