import jax
import jax.numpy as jnp
from jaxls import Shaped, symbolic_shape, typed

Batch, Length = symbolic_shape("Batch, Length")


@typed
def foo(
    x: tuple[
        Shaped[jax.Array, (Batch, Length), jnp.float32],
        Shaped[jax.Array, (Batch, Length), jnp.float32],
    ],
) -> tuple[
    Shaped[jax.Array, (Batch, Length), jnp.float32],
    Shaped[jax.Array, (Batch, Length), jnp.float32],
]:
    a = x[0] + 3
    b = (x[0][0] + 2, x[1] + a + 2)
    return b
