from enum import Enum

from .typed import run as typed_run


class Method(str, Enum):
    typed = "typed"


methods = {
    Method.typed: typed_run,
}
