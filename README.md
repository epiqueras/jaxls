## Getting Started

1. Install `nox` in the activated environment: `pip install nox`.
2. Run `nox --session install` to install Python dependencies.
3. Install test dependencies `python -m pip install -r src/test/python_tests/requirements.txt && python -m pip install -e .`. You will have to install these to run tests from the Test Explorer.
4. Run: `Debug Extension and Python`.
5. `nox --session build_package` to build the extension.
