.. _apiconfig:

graphstorm.config
===================

GraphStorm model training and inference CLIs have a set of :ref:`built-in configurations <configurations-run>`.
These configurations are defined and managed in ``graphstorm.config`` module whose ``get_argument_parser()``
method can help users to load these built-in configurations either from a yaml file or from CLI arguments.
This ``get_argument_parser()`` is useful when users want to convert customized models to use GraphStorm CLIs.

.. currentmodule:: graphstorm.config

Configuration Argument Parser
------------------------------

.. autosummary::
    :toctree: ../generated/
    :nosignatures:

    get_argument_parser