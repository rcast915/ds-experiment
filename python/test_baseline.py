import jax
import jax.numpy as jnp

def f(x, y):
    return (x + y) * y

x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
y = jnp.array([4.0, 5.0, 6.0], dtype=jnp.float64)

out = jax.jit(f)(x, y)
print(out)
