from pathlib import Path

import typer

from .methods import Method, methods


def app(method: Method, path: Path):
    methods[method](path)


def main():
    typer.run(app)


if __name__ == "__main__":
    main()
