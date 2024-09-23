.. _apiconfig:

graphstorm.config
===================

GraphStorm model training and inference CLIs have a set of :ref:`built-in configurations <configurations-run>`.
These configurations are defined and managed in ``graphstorm.config`` module whose ``get_argument_parser()``
method can help users to load these built-in configurations either from a yaml file or from CLI arguments.
This ``get_argument_parser()`` is useful when users want to convert customized models to use GraphStorm CLIs.
GraphStorm creates a ``GSConfig`` object to store the configurations loaded from a yaml file or CLI arguments,
and then pass this ``GSConfig`` object to other modules to extract related configurations.

.. currentmodule:: graphstorm.config

Configuration
--------------

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: configtemplate.rst

    get_argument_parser
    GSConfig