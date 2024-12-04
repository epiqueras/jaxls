from dataclasses import dataclass
from functools import partial
from inspect import Parameter, signature
from types import SimpleNamespace
from typing import TYPE_CHECKING, Annotated, Any, Callable, Type, TypeVar

import jax
import jax.numpy as jnp
from jax import core, export
from jax._src import source_info_util

symbolic_shape = export.symbolic_shape
Frame = source_info_util.Frame


if TYPE_CHECKING:
    Shaped = Annotated
else:

    class Shaped:
        def __class_getitem__(
            cls, indices: tuple[Type[jax.Array], tuple[int, ...], jnp.dtype]
        ) -> SimpleNamespace:
            _, shape, dtype = indices
            return SimpleNamespace(shape=shape, dtype=dtype)


def _parse_annotation(annotation: Any) -> jax.ShapeDtypeStruct | None:
    if annotation == Parameter.empty:
        return None
    if (
        not isinstance(annotation, SimpleNamespace)
        or not hasattr(annotation, "shape")
        or not hasattr(annotation, "dtype")
    ):
        raise ValueError(f"Unsupported annotation: {annotation}.")
    return jax.ShapeDtypeStruct(annotation.shape, annotation.dtype)


@dataclass(kw_only=True, frozen=True, slots=True)
class EqnTypes:
    out_shapes: list[jax.ShapeDtypeStruct]
    frame: Frame


def _process_eqn(
    eqn: core.JaxprEqn,
) -> None | EqnTypes:
    out_shapes = []
    for outvar in eqn.outvars:
        out_aval = outvar.aval
        if not isinstance(out_aval, core.ShapedArray):
            return None
        out_shapes.append(jax.ShapeDtypeStruct(out_aval.shape, out_aval.dtype))
    user_frame = source_info_util.user_frame(eqn.source_info)
    if user_frame is None:
        return None
    return EqnTypes(out_shapes=out_shapes, frame=user_frame)


@dataclass(kw_only=True, frozen=True, slots=True)
class Types:
    eqns: list[EqnTypes]


def _process_jaxpr(jaxpr: core.ClosedJaxpr) -> Types:
    eqns = []

    for eqn in jaxpr.eqns:
        eqn_types = _process_eqn(eqn)
        if eqn_types is None:
            continue
        eqns.append(eqn_types)

    return Types(eqns=eqns)


_T = TypeVar("_T", bound=Callable[..., Any])


def typed(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        func, *args = args
    else:
        func = None

    def decorator(func: _T) -> _T:
        sig = signature(func)
        shape_args = []
        shape_kwargs = {}
        for i, (name, param) in enumerate(sig.parameters.items()):
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                raise ValueError("typed decorator does not support varargs.")

            # Look for decorator overrides.
            if param.kind == param.POSITIONAL_ONLY:
                if i < len(args):
                    shape_args.append(args[i])
                    continue
            if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                if name in kwargs:
                    shape_kwargs[name] = kwargs[name]
                    continue
            if param.kind == param.POSITIONAL_OR_KEYWORD:
                # If we got here it could be because it was not passed as a keyword.
                if i < len(args):
                    shape_args.append(args[i])
                    continue

            # Parse the annotation, if any.
            annotation = _parse_annotation(param.annotation)
            if annotation is None:
                raise ValueError(f"Unsupported parameter kind: {param.kind}.")

            # If we got here, we have an annotation, and no overrides.
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                shape_args.append(annotation)
            else:
                shape_kwargs[name] = annotation
        static_kwargs = {k: v for k, v in kwargs.items() if k not in shape_kwargs}
        shape_return = _parse_annotation(sig.return_annotation)

        jaxpr, out_shape = jax.make_jaxpr(
            partial(func, **shape_kwargs), **static_kwargs, return_shape=True
        )(*shape_args)
        if shape_return is not None and shape_return != out_shape:
            raise ValueError(
                f"Return type {shape_return} does not match inferred type {out_shape}."
            )

        types = _process_jaxpr(jaxpr)
        # TODO: Put types somewhere for consumption.
        print(types)

        return func

    # Handle both @typed and @typed() usage.
    if func is not None:
        return decorator(func)
    return decorator
