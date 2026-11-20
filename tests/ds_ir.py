"""
Make JAX-lowered StableHLO text readable by mlir-ds-opt.

mlir-ds-opt is built against StableHLO 1.16.3; JAX 0.12.1 lowers at 1.25.3,
and textual MLIR is not version-stable (newer attributes or syntax may not
parse). Versioned bytecode is: serialize JAX's module as a portable artifact
at the pass's version (XLA's serializer, as the PJRT client does), then let
mlir-ds-opt itself convert it back to StableHLO text. Tests then hand that text to the pass exactly as before.
The PJRT proxy does the same negotiation for real compiles (it reports
1.16.3 to JAX; see pjrt_plugin/ds_pjrt_plugin.cpp).

    from ds_ir import pass_ready
    text = pass_ready(str(jax.jit(fn).lower(*args).compiler_ir()))
"""

import os
import subprocess
import tempfile
from pathlib import Path

PASS_STABLEHLO = "1.16.3"
OPT_BINARY = Path(__file__).resolve().parent.parent / "stablehlo_pass" / "build" / "mlir-ds-opt"


def pass_ready(text, opt=OPT_BINARY):
    """StableHLO text from JAX -> equivalent StableHLO text at PASS_STABLEHLO.

    Uses XLA's serializer, the same one JAX's PJRT client uses for real
    compiles (it also handles CHLO ops, which arrive as stablehlo.composite;
    these are turned into calls to their decompositions, which the tests'
    `inline` step then flattens, as in the PJRT proxy's pipeline).
    """
    from jax._src.lib import _jax
    data = _jax.mlir.serialize_portable_artifact(text, PASS_STABLEHLO)
    with tempfile.NamedTemporaryFile(suffix=".mlirbc", delete=False) as f:
        f.write(data)
    try:
        r = subprocess.run(
            [str(opt),
             f"--pass-pipeline=builtin.module(vhlo-to-version{{target={PASS_STABLEHLO}}},"
             "vhlo-legalize-to-stablehlo,func.func(stablehlo-legalize-composite-to-call))",
             f.name],
            capture_output=True, text=True)
    finally:
        os.unlink(f.name)
    if r.returncode != 0:
        raise RuntimeError(f"mlir-ds-opt could not read StableHLO {PASS_STABLEHLO} bytecode:\n"
                           + r.stderr[-1000:])
    return r.stdout
