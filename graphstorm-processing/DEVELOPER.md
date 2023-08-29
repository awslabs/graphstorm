## Developer Guide

The project is set up using `poetry` to make easier for developers to jump into the project.

The steps we recommend are:

### Install pyenv

`pyenv` is a tool to manage multiple Python version installations. It can be installed
through the installer below on a Linux machine:

```
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
```

or use brew on a Mac:

```
brew update
brew install pyenv
```

For more info on `pyenv` see https://github.com/pyenv/pyenv

### Create a Python 3.9 env and activate it.

We use Python 3.9 in our images so this most closely resembles the execution environment
on SageMaker.

```
pyenv install 3.9
pyenv global 3.9
```

> Note: We recommend not mixing up conda and pyenv. When developing for this project,
simply `conda deactivate` until there's no `conda` env active (even `base`) and
just rely on pyenv+poetry to handle dependencies.

### Install poetry

`poetry` is a dependency management system for Python. To install it
use:

```
curl -sSL https://install.python-poetry.org | python3 -
```

### Install dependencies through poetry

Now we are ready to install our dependencies through `poetry`.

We have split the project dependencies into the "main" dependencies
that `poetry` installs by default, and the `dev` dependency group
that installs that dependencies that are only needed to develop
the library.

**On a Linux system** (tested on Ubuntu 16.04) run:

```
# Install all dependencies into local .venv
poetry install --with dev
```

Once all dependencies are installed you should be able to run the unit tests
for the project and continue with development using:

```
poetry run pytest ./tests
```

You can also activate and use the virtual environment using:

```
poetry shell
# We're now using the graphstorm-processing-py3.9 env so we can just run
pytest ./tests
```

To learn more about poetry see: https://python-poetry.org/docs/basic-usage/

### Use `black` to format code

We use [black](https://black.readthedocs.io/en/stable/index.html)
to format code in this project. `black` is an
opinionated formatter that helps speed up development
and code reviews. It is included in our `dev` dependencies
so it will be installed along with the other dev dependencies.

To use `black` in the project you can run (from the project's root,
same level as `pyproject.toml`)

```
# From the project's root directory, graphstorm-processing run:
black .
```

To get a preview of the changes `black` would make you can use:

```
black . --diff --color
```

You can auto-formatting with `black` to VSCode using the
[Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)

#### Use a pre-commit hook to ensure black runs before commits

We include a pre-commit config file with `black` to make it
easier for developers to use before committing.

We include `pre-commit` in the project's `dev` dependencies, so once
you have activated the project's venv (`poetry shell`) you can just run:

```
pre-commit install
```

which will install the `black` hook into your local repository and ensure
it runs before every commit.

### Use mypy to lint code

We include the `mypy` linter as a dependency under
the `dev` group of dependencies. `mypy` makes Python
development faster and safer through type annotations.

We recommend using VSCode and enabling the mypy linter
to get in-editor annotations:

https://code.visualstudio.com/docs/python/linting#_general-settings

You can also lint the project code through:

```
poetry run mypy ./graphstorm_processing
```

To learn more about `mypy` and how it can help development
see: https://mypy.readthedocs.io/en/stable/

Our goal is to minimize `mypy` errors as much as possible for
the project. New code should be linted and not introduce
additional mypy errors. When necessary it's OK to use
`type: ignore` to silence `mypy` errors inline, but this
should be used sparingly.
