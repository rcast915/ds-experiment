import os
import jax
import jax.numpy as jnp
from jax import export

def f(x, y):
    return (x + y) * y

x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
y = jnp.array([4.0, 5.0, 6.0], dtype=jnp.float64)

exp = export.export(jax.jit(f))(x, y)
mlir_text = exp.mlir_module()

os.makedirs("artifacts", exist_ok=True)
with open("artifacts/baseline.mlir", "w") as fobj:
    fobj.write(mlir_text)

print("Wrote artifacts/baseline.mlir")
