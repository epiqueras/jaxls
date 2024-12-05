import jax
import jax.numpy as jnp
from jaxls import Shaped, symbolic_shape, typed

Batch, Length = symbolic_shape("Batch, Length")


@typed
def foo(
    x: Shaped[jax.Array, (Batch, Length), jnp.float32],
) -> Shaped[jax.Array, (Batch, Length), jnp.float32]:
    b = x + 2
    return b
