"""
Tests for jaxls typed module features.
"""

from dataclasses import dataclass
from functools import partial

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

    @partial(jax.tree_util.register_dataclass, data_fields=["value"], meta_fields=[])
    @dataclass
    class ClassPytree:
        value: Shaped[jax.Array, (Batch, Length), jnp.float32]

    @typed(c=Class(value=1))
    def foo(
        x: Shaped[jax.Array, (Batch, Length), jnp.float32],
        p: ClassPytree,
        t: tuple[
            Shaped[jax.Array, (Batch, Length), jnp.float32],
            Shaped[jax.Array, (Batch, Length), jnp.float32],
        ],
        *,
        c: Class,
    ) -> Shaped[jax.Array, (Batch, Length), jnp.float32]:
        b = x + 1 + c.value + t[0] + t[1] + p.value
        return b

    val = jnp.ones((1, 2))
    foo(val, ClassPytree(value=val), (val, val), c=Class(value=1))

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
