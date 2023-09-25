#!/usr/bin/env bash

set -Eeuox pipefail

PYTHON_VERSION="3.9.18"

# Python won’t try to write .pyc or .pyo files on the import of source modules
# Force stdin, stdout and stderr to be totally unbuffered. Good for logging
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
PYTHONIOENCODING=UTF-8
LANG=C.UTF-8
LC_ALL=C.UTF-8
# LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"
# LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/conda/lib"
# PATH=/opt/conda/bin:$PATH




# Set up pyenv
sudo yum erase -y openssl-devel
sudo yum install -y \
    bzip2-devel\
    gcc \
    git \
    libffi-devel \
    ncurses-devel \
    openssl11-devel \
    readline-devel \
    sqlite-devel \
    sudo \
    xz-devel

LIBDIR="/home/hadoop/"
PYENV_ROOT="${LIBDIR}/.pyenv"

git clone https://github.com/pyenv/pyenv.git ${PYENV_ROOT}

export PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

eval "$(pyenv init -)"

pyenv install ${PYTHON_VERSION}
pyenv global ${PYTHON_VERSION}



which python

python -V

pip install poetry==1.6.1

POETRY_NO_INTERACTION=1
POETRY_VIRTUALENVS_IN_PROJECT=1
POETRY_VIRTUALENVS_CREATE=1
POETRY_CACHE_DIR=/tmp/poetry_cache

git clone https://github.com/thvasilo/graphstorm.git

cd graphstorm/graphstorm-processing

git checkout origin/gs-processing --detach

pwd
python -m pip install .
