#!/usr/bin/env bash
# Run once per container session from /src/ds_experiment.
# Disables JAX CUDA auto-registration, builds the PJRT plugin, and verifies.
# Note: once the Dockerfile is rebuilt, the rename steps become no-ops.
set -e

XPLUG=/usr/local/lib/python3.12/dist-packages/jax_plugins/xla_cuda12
JPKG=/usr/local/lib/python3.12/dist-packages/jax_cuda12_plugin

echo "[ds_setup] Step 1: disable JAX CUDA auto-registration"
if [ -d "$XPLUG" ]; then
    mv "$XPLUG" "${XPLUG}_disabled"
    echo "  renamed xla_cuda12 → xla_cuda12_disabled"
else
    echo "  xla_cuda12 already disabled"
fi
if [ -d "$JPKG" ]; then
    mv "$JPKG" "${JPKG}_disabled"
    echo "  renamed jax_cuda12_plugin → jax_cuda12_plugin_disabled"
else
    echo "  jax_cuda12_plugin already disabled"
fi

echo "[ds_setup] Step 2: build PJRT plugin"
cmake -GNinja -S pjrt_plugin -B pjrt_plugin/build > /dev/null
ninja -C pjrt_plugin/build

echo "[ds_setup] Step 3: verify"
out=$(DS_BYPASS=1 \
  PJRT_NAMES_AND_LIBRARY_PATHS="cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so" \
  python3 -c "import jax.numpy as jnp; x = jnp.array([1.,2.,3.,4.]); print(x + x)" 2>/dev/null)
if echo "$out" | grep -q "\[2. 4. 6. 8.\]"; then
    echo "  OK: $out"
else
    echo "  FAIL: $out"
    exit 1
fi

echo ""
echo "Ready. Activate DS with:"
echo "  PJRT_NAMES_AND_LIBRARY_PATHS=\"cuda:/src/ds_experiment/pjrt_plugin/build/libds_pjrt_plugin.so\""
