FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

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

# ── Python / JAX ───────────────────────────────────────────────────────────────
RUN pip install "jax[cuda12]" numpy

# Disable JAX CUDA auto-registration so our PJRT proxy can claim the cuda slot.
# ds_pjrt_plugin.cpp hardcodes xla_cuda12_disabled as the real plugin path.
RUN mv /usr/local/lib/python3.12/dist-packages/jax_plugins/xla_cuda12 \
       /usr/local/lib/python3.12/dist-packages/jax_plugins/xla_cuda12_disabled && \
    mv /usr/local/lib/python3.12/dist-packages/jax_cuda12_plugin \
       /usr/local/lib/python3.12/dist-packages/jax_cuda12_plugin_disabled

# ── XLA FFI headers ────────────────────────────────────────────────────────────
RUN mkdir -p /opt/xla-ffi/xla/ffi/api
RUN wget -q -O /opt/xla-ffi/xla/ffi/api/c_api.h \
      https://raw.githubusercontent.com/openxla/xla/main/xla/ffi/api/c_api.h && \
    wget -q -O /opt/xla-ffi/xla/ffi/api/api.h \
      https://raw.githubusercontent.com/openxla/xla/main/xla/ffi/api/api.h && \
    wget -q -O /opt/xla-ffi/xla/ffi/api/ffi.h \
      https://raw.githubusercontent.com/openxla/xla/main/xla/ffi/api/ffi.h

# ── PJRT C API headers ────────────────────────────────────────────────────────
RUN mkdir -p /opt/xla-pjrt/xla/pjrt/c && \
    wget -q -O /opt/xla-pjrt/xla/pjrt/c/pjrt_c_api.h \
      https://raw.githubusercontent.com/openxla/xla/main/xla/pjrt/c/pjrt_c_api.h

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
