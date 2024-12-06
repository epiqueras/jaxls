import ast
import inspect
import runpy
import traceback
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from inspect import Parameter, signature
from pathlib import Path
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Type,
    TypeVar,
    cast,
    get_args,
    get_origin,
)

import jax
import jax.numpy as jnp
import pydantic
import typer
from jax import core, export
from jax._src import source_info_util, traceback_util

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


ShapeDtypeStructTree = Any


def pp_shapes(tree: ShapeDtypeStructTree) -> str:
    """Returns a pretty printed string representation of a ShapeDtypeStructTree."""

    def dtype2str(dtype: jnp.dtype | str) -> str:
        if isinstance(dtype, str):
            dtype = jnp.dtype(dtype)
        kind = dtype.kind
        itemsize = dtype.itemsize
        return f"{kind}{itemsize * 8}"  # e.g., 'f32' for np.float32

    def shape2str(x: jax.ShapeDtypeStruct) -> str:
        shape_str = ", ".join(str(s) for s in x.shape)
        return f"{dtype2str(x.dtype)}[{shape_str}]"

    return str(jax.tree_util.tree_map(shape2str, tree))


def _parse_annotation(annotation: Any) -> ShapeDtypeStructTree | None:
    if annotation == Parameter.empty:
        return None
    if (
        isinstance(annotation, SimpleNamespace)
        or hasattr(annotation, "shape")
        or hasattr(annotation, "dtype")
    ):
        # Bottomed out on a shaped type.
        return jax.ShapeDtypeStruct(annotation.shape, annotation.dtype)

    annotations = getattr(annotation, "__annotations__", None)
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotations is None and origin is None:
        # Bottomed out on a non complex type.
        raise ValueError(f"Unsupported annotation: {annotation}.")
    if annotations is not None:
        return annotation(**{k: _parse_annotation(v) for k, v in annotations.items()})
    if isinstance(origin, (type(tuple), type(list), type(set), type(frozenset))):
        return origin([_parse_annotation(a) for a in args])
    raise ValueError(f"Unsupported annotation: {annotation=} {origin=} {args=}.")


@dataclass(kw_only=True, frozen=True, slots=True)
class Shape:
    shape: tuple[str, ...]
    dtype: str


@dataclass(kw_only=True, frozen=True, slots=True)
class EqnTypes:
    frame: Frame
    out_shapes: list[Shape]
    message: str | None = None
    tooltip: str | None = None


def _process_eqn(
    eqn: core.JaxprEqn,
) -> None | EqnTypes:
    out_shapes: list[Shape] = []
    for outvar in eqn.outvars:
        out_aval = outvar.aval
        if not isinstance(out_aval, core.ShapedArray) or not isinstance(
            out_aval.shape, tuple
        ):
            return None
        shape = cast(tuple, out_aval.shape)
        out_shapes.append(
            Shape(shape=tuple(str(s) for s in shape), dtype=str(out_aval.dtype))
        )
    user_frame = source_info_util.user_frame(eqn.source_info)
    if user_frame is None:
        return None
    return EqnTypes(frame=user_frame, out_shapes=out_shapes)


@dataclass(kw_only=True, frozen=True, slots=True)
class Types:
    eqns: list[EqnTypes]


def _process_jaxpr(jaxpr: core.ClosedJaxpr) -> Types:
    eqns: list[EqnTypes] = []
    for eqn in jaxpr.eqns:
        eqn_types = _process_eqn(eqn)
        if eqn_types is None:
            continue
        eqns.append(eqn_types)
    return Types(eqns=eqns)


def _process_error(frame: traceback.FrameSummary, message: str) -> Types:
    eqn = EqnTypes(
        frame=Frame(
            file_name=frame.filename,
            function_name="",
            start_line=(frame.lineno or 1),
            start_column=(frame.colno or 0),
            end_line=(frame.end_lineno or 1),
            end_column=(frame.end_colno or 0),
        ),
        out_shapes=[],
        message=message,
    )
    return Types(eqns=[eqn])


def _get_return_info(func: Callable[..., Any]) -> tuple[int | None, int | None]:
    source_lines, starting_line = inspect.getsourcelines(func)
    source = "".join(source_lines)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
            for body_item in node.body:
                if isinstance(body_item, ast.Return) and body_item.value:
                    return (
                        starting_line + body_item.lineno - 1,
                        (body_item.value.end_col_offset or 0),
                    )
    return None, None


_dont_error = False


def _process_out_shape_mismatch(
    frame: Frame, shape_return: ShapeDtypeStructTree, out_shape: ShapeDtypeStructTree
) -> EqnTypes:
    return_shape_str = pp_shapes(shape_return)
    inferred_shape_str = pp_shapes(out_shape)
    message = f"Return type {return_shape_str} does not match inferred type {inferred_shape_str}."
    if not _dont_error:
        raise ValueError(message)
    eqn = EqnTypes(
        frame=frame,
        out_shapes=[],
        message=f"{return_shape_str} != {inferred_shape_str}",
        tooltip=message,
    )
    return eqn


class TypeRegistry(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, strict=True)
    frames: dict[str, dict[int, EqnTypes]]


type_registry: TypeRegistry = TypeRegistry(frames={})


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

        try:
            jaxpr, out_shape = jax.make_jaxpr(
                partial(func, **shape_kwargs), **static_kwargs, return_shape=True
            )(*shape_args)
            types = _process_jaxpr(jaxpr)
            if (
                shape_return is not None
                and out_shape is not None
                and shape_return != out_shape
            ):
                return_line, return_col = _get_return_info(func)
                return_frame = types.eqns[-1].frame._replace(
                    start_line=return_line, end_column=return_col
                )
                out_shape_mismatch_eqn = _process_out_shape_mismatch(
                    return_frame,
                    shape_return,
                    out_shape,
                )
                types.eqns.append(out_shape_mismatch_eqn)
        except Exception as e:
            if e.__traceback__ is None:
                return func
            tb = traceback_util.filter_traceback(e.__traceback__)
            tb = traceback.extract_tb(tb)
            frame = tb[1]
            types = _process_error(frame, str(e))

        frames = defaultdict(dict)
        for eqn_types in types.eqns:
            file_name = eqn_types.frame.file_name
            frames[file_name][
                hash(
                    (
                        eqn_types.frame.file_name,
                        eqn_types.frame.start_line,
                        eqn_types.frame.end_column,
                    )
                )
            ] = eqn_types
        type_registry.frames.update(frames)

        return func

    # Handle both @typed and @typed() usage.
    if func is not None:
        return decorator(func)
    return decorator


def run(path: Path):
    global _dont_error
    _dont_error = True
    runpy.run_path(str(path))
    _dont_error = False
    json_dump = type_registry.model_dump_json()
    typer.echo(json_dump)
