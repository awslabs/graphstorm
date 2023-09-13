.. _apigraphstorm:

.. currentmodule:: graphstorm

graphstorm
============

    The ``graphstorm`` package contains a set of functions for environment setup.
    Users can directly use the following code to use these functions.

    >>> import graphstorm as gs
    >>> gs.initialize()
    >>> gs.get_rank()

.. autosummary::
    :toctree: ../generated/

    gsf.initialize
    gsf.get_feat_size
    utils.get_rank
    utils.get_world_size
