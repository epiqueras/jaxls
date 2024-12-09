import jax
import jax.numpy as jnp
from flax import nnx
from jaxls import Shaped, symbolic_shape, typed, typed_nnx

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
    return (a, a)


Batch, DIn, DOut = symbolic_shape("Batch, DIn, DOut")


@typed_nnx
class Linear(nnx.Module):
    def __init__(self, din: Shaped[int, DIn], dout: Shaped[int, DOut]):
        self.w = nnx.Param(jnp.ones((din, dout)))
        self.b = nnx.Param(jnp.zeros((dout,)))

    def __call__(self, x: Shaped[jax.Array, (Batch, DIn), jnp.float32]):
        y = x @ self.w
        out = y + self.b
        return out
