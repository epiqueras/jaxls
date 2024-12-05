"""
Tests for jaxls typed module features.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from hamcrest import assert_that, is_
from jaxls import Shaped, symbolic_shape, type_registry, typed


def _get_last_eqn_types():
    return list(list(type_registry.frames.values())[-1].values())[-1]


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

    foo(jnp.ones((1, 2)), y=Class(value=1))

    eqn_types = _get_last_eqn_types()
    [out_shape] = eqn_types.out_shapes
    assert_that(
        str(out_shape), is_("Shape(shape=('Batch', 'Length'), dtype='float32')")
    )

    json_dump = type_registry.model_dump_json()
    type_registry.model_validate_json(json_dump)


def test_basic_error():
    """Tests basic example with an error."""

    Batch, Length = symbolic_shape("Batch, Length")

    @typed
    def foo(
        x: Shaped[jax.Array, (Batch, Length), jnp.float32],
    ) -> Shaped[jax.Array, (Batch, Length), jnp.float32]:
        b = x + 1
        return b @ b

    eqn_types = _get_last_eqn_types()
    assert_that(
        str(eqn_types.message),
        is_(
            "dot_general requires contracting dimensions to have the same shape, got (Length,) and (Batch,)."
        ),
    )

    json_dump = type_registry.model_dump_json()
    type_registry.model_validate_json(json_dump)
