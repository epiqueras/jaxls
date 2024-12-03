from typing import Any, Callable, TypeVar

import jax

T = TypeVar("T", bound=Callable[..., Any])


def typed(func: T) -> T:
    return func


print(jax)
