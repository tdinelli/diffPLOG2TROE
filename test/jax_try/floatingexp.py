import jax.numpy as jnp
import numpy as np

vals = jnp.array([0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001,
                  1E-10, 1E-11, 1E-12, 1E-13, 1E-14, 1E-15, 1E-16, 1E-17, 1E-18, 1E-19], dtype=jnp.float64)

for i in vals:
    print(f"val: {i} exp {jnp.exp(i)}")
