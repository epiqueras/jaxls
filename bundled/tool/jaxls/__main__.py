import sys
from pathlib import Path

import typer

from .methods import Method, methods


def app(method: Method, path: Path, *, use_stdin: bool = False):
    if use_stdin:
        code = sys.stdin.read()
    else:
        code = None
    methods[method](path, code)


def main():
    typer.run(app)


if __name__ == "__main__":
    main()
