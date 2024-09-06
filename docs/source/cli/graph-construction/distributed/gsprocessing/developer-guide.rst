.. _gsprocessing_developer_guide:

Developer Guide
---------------

The document helps developers set up their environment for development.
It includes steps for setting up the local development environment with poetry,
followed by guidelines on using linters and type checkers to ensure code quality.


The steps we recommend are:

Install JDK 8, 11
~~~~~~~~~~~~~~~~~

PySpark requires a compatible Java installation to run, so
you will need to ensure your active JDK is using either
Java 8 or 11.

On MacOS you can do this using ``brew``:

.. code-block:: bash

    brew install openjdk@11

On Linux it will depend on your distribution's package
manager. For Ubuntu you can use:

.. code-block:: bash

    sudo apt install openjdk-11-jdk

On Amazon Linux 2 you can use:

.. code-block:: bash

    sudo yum install java-11-amazon-corretto-headless
    sudo yum install java-11-amazon-corretto-devel

Install ``pyenv``
~~~~~~~~~~~~~~~~~

``pyenv`` is a tool to manage multiple Python version installations. It
can be installed through the installer below on a Linux machine:

.. code-block:: bash

   curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

or use ``brew`` on a Mac:

.. code-block:: bash

   brew update
   brew install pyenv

For more info on ``pyenv`` see `its documentation. <https://github.com/pyenv/pyenv>`_

Create a Python 3.9 env and activate it.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use Python 3.9 in our images so this most closely resembles the
execution environment on our Docker images that will be used for distributed
training.

.. code-block:: bash

   pyenv install 3.9
   pyenv global 3.9

..

   Note: We recommend not mixing up ``conda`` and ``pyenv``. When developing for
   this project, simply ``conda deactivate`` until there's no ``conda``
   env active (even ``base``) and just rely on ``pyenv`` and ``poetry`` to handle
   dependencies.

Install ``poetry``
~~~~~~~~~~~~~~~~~~

``poetry`` is a dependency and build management system for Python. To install it
use:

.. code-block:: bash

   curl -sSL https://install.python-poetry.org | python3 -

Install dependencies through ``poetry``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we are ready to install our dependencies through ``poetry``.

We have split the project dependencies into the “main” dependencies that
``poetry`` installs by default, and the ``dev`` dependency group that
installs that dependencies that are only needed to develop the library.

**On a POSIX system** (tested on Ubuntu, CentOS, MacOS) run:

.. code-block:: bash

   # Install all dependencies into local .venv
   poetry install --with dev

Once all dependencies are installed you should be able to run the unit
tests for the project and continue with development using:

.. code-block:: bash

   poetry run pytest ./graphstorm-processing/tests

You can also activate and use the virtual environment using:

.. code-block:: bash

   poetry shell
   # We're now using the graphstorm-processing-py3.9 env so we can just run
   pytest ./graphstorm-processing/tests

To learn more about ``poetry`` see its `documentation <https://python-poetry.org/docs/basic-usage/>`_

Use ``black`` to format code [optional]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use `black <https://black.readthedocs.io/en/stable/index.html>`_ to
format code in this project. ``black`` is an opinionated formatter that
helps speed up development and code reviews. It is included in our
``dev`` dependencies so it will be installed along with the other dev
dependencies.

To use ``black`` in the project you can run (from the project's root,
same level as ``pyproject.toml``)

.. code-block:: bash

   # From the project's root directory, graphstorm-processing run:
   black .

To get a preview of the changes ``black`` would make you can use:

.. code-block:: bash

   black . --diff --color

You can auto-formatting with ``black`` to VSCode using the `Black
Formatter <https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter>`__


Use mypy and pylint to lint code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We include the ``mypy`` and ``pylint`` linters as a dependency under the ``dev`` group
of dependencies. These linters perform static checks on your code and
can be used in a complimentary manner.

We recommend `using VSCode and enabling the mypy linter <https://code.visualstudio.com/docs/python/linting#_general-settings>`_
to get in-editor annotations.

You can also lint the project code through:

.. code-block:: bash

   poetry run mypy ./graphstorm_processing

To learn more about ``mypy`` and how it can help development
`see its documentation <https://mypy.readthedocs.io/en/stable/>`_.


Our goal is to minimize ``mypy`` errors as much as possible for the
project. New code should be linted and not introduce additional mypy
errors. When necessary it's OK to use ``type: ignore`` to silence
``mypy`` errors inline, but this should be used sparingly.

As a project, GraphStorm requires a 10/10 pylint score, so
ensure your code conforms to the expectation by running

.. code-block:: bash

    pylint --rcfile=/path/to/graphstorm/tests/lint/pylintrc

on your code before commits. To make this easier we include
a pre-commit hook below.

Use a pre-commit hook to ensure ``black`` and ``pylint`` run before commits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To make code formatting and ``pylint`` checks easier for graphstorm-processing
developers, we recommend using a pre-commit hook.

We include ``pre-commit`` in the project's ``dev`` dependencies, so once
you have activated the project's venv (``poetry shell``) you can just
create a file named ``.pre-commit-config.yaml`` with the following contents:

.. code-block:: yaml

    # .pre-commit-config.yaml
    repos:
        - repo: https://github.com/psf/black
            rev: 23.7.0
            hooks:
            - id: black
                language_version: python3.9
                files: 'graphstorm_processing\/.*\.pyi?$|tests\/.*\.pyi?$|scripts\/.*\.pyi?$'
                exclude: 'python\/.*\.pyi'
        - repo: local
            hooks:
            - id: pylint
                name: pylint
                entry: pylint
                language: system
                types: [python]
                args:
                [
                    "--rcfile=./tests/lint/pylintrc"
                ]


And then run:

.. code-block:: bash

   pre-commit install

which will install the ``black`` and ``pylint`` hooks into your local repository and
ensure it runs before every commit.

.. note::

    The pre-commit hook will also apply to all commits you make to the root
    GraphStorm repository. Since Graphstorm doesn't use ``black``, you might
    want to remove the ``black`` hook. You can do so from the root repo
    using ``rm -rf .git/hooks``.

    Both projects use ``pylint`` to check Python files so we'd still recommend using
    that hook even if you're doing development for both GSProcessing and GraphStorm.
