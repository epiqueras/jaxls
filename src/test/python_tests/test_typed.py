"""
Tests for jaxls typed module features.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxls import Shaped, symbolic_shape, typed


def test_basic_example():
    """Tests basic example."""

    @dataclass
    class Class:
        value: int

    Batch, Length = symbolic_shape("Batch, Length")

    @typed(y=Class(value=1))
    def foo(
        x: Shaped[jax.Array, (Batch, Length), jnp.float32], *, y: Class
    ) -> Shaped[jax.Array, (Batch, Length), jnp.float32]:
        b = x + 1 + y.value
        return b
